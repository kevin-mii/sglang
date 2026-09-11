"""The ``aiter_sparse`` ROCm attention backend (aiter's gluon ``pa_decode_sparse``) and its Triton
split-KV combine must match the torch reference, the kernels they replace and aiter's own reduce on
the served DeepSeek-V4 packed fp8 KV layout, bitwise repeatable and batch-invariant."""

import math
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


NOPE, ROPE, D = 448, 64, 512


PAGE = 256


BYTES = 584


SCALE = D**-0.5


# Relative tolerance for the short-list case: about 3x the measured aiter error
# (~3e-3) and 40x under what attending one stray key on a 5-key list costs (~0.4).
TOL_SHORT = 1e-2


def _pack_cache(num_blocks, device, gen, *, fp8_view=True):
    """Random bf16 keys in the packed fp8 layout: cache [num_blocks, PAGE, 1, BYTES] and the dequantized keys [slots, D] fp32."""
    slots = num_blocks * PAGE
    k = torch.randn(slots, D, generator=gen) * 0.5
    nope = k[:, :NOPE].reshape(slots, NOPE // 64, 64)
    amax = nope.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(min=-127, max=127)
    scale = torch.pow(2.0, exp)
    nope_fp8 = (nope / scale).to(torch.float8_e4m3fn)
    nope_deq = nope_fp8.float() * scale
    rope = k[:, NOPE:].to(torch.bfloat16)
    raw = torch.zeros(num_blocks, PAGE * BYTES, dtype=torch.uint8)
    data = raw[:, : PAGE * 576].view(num_blocks, PAGE, 576)
    data[:, :, :NOPE] = nope_fp8.view(torch.uint8).reshape(num_blocks, PAGE, NOPE)
    data[:, :, NOPE:] = rope.view(torch.uint8).reshape(num_blocks, PAGE, 2 * ROPE)
    scales = raw[:, PAGE * 576 :].view(num_blocks, PAGE, 8)
    scales[:, :, :7] = (exp.reshape(num_blocks, PAGE, 7) + 127).to(torch.uint8)
    cache = raw.view(num_blocks, PAGE, 1, BYTES)
    if fp8_view:
        cache = cache.view(torch.float8_e4m3fn)
    deq = torch.cat([nope_deq.reshape(slots, NOPE), rope.float()], dim=1).to(device)
    return cache.to(device), deq


def _masked(indices, lengths):
    """One length-folded list; ``_fold_lengths_into_index_lists`` returns the (main, extra) pair."""
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        _fold_lengths_into_index_lists,
    )

    return _fold_lengths_into_index_lists(indices, lengths)[0]


def _reference(q, sink, sets):
    """q [b, 1, h, D] bf16; sets = [(deq_keys [slots, D], indices [b, 1, w], lengths [b])].
    Softmax over the valid gathered keys plus the sink logit; V is the full key."""
    b, _, h, _ = q.shape
    out = torch.zeros(b, 1, h, D, device=q.device)
    for i in range(b):
        keys = []
        for deq, idx, length in sets:
            sel = idx[i, 0]
            pos = torch.arange(sel.numel(), device=sel.device)
            valid = (sel >= 0) & (pos < length[i])
            keys.append(deq[sel[valid].long()])
        k = torch.cat(keys, dim=0)  # [n, D]
        s = q[i, 0].float() @ k.T * SCALE  # [h, n]
        logits = torch.cat([s, sink[:, None].float()], dim=1)
        p = torch.softmax(logits, dim=1)[:, :-1]
        out[i, 0] = p @ k
    return out


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseBackend(CustomTestCase):
    def _assert_matches_reference_and_tilelang(
        self, batch, heads, swa_lengths, topk_lengths, seed=0, tol=3e-2
    ):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        swa_cache, swa_deq = _pack_cache(2, dev, gen)
        topk_cache, topk_deq = _pack_cache(5, dev, gen)
        q = (
            (torch.randn(batch, 1, heads, D, generator=gen) * 0.5)
            .to(torch.bfloat16)
            .to(dev)
        )
        sink = (torch.randn(heads, generator=gen) * 0.5).to(dev)
        swa_idx = (
            torch.stack(
                [torch.randperm(2 * PAGE, generator=gen)[:128] for _ in range(batch)]
            )
            .to(torch.int32)
            .unsqueeze(1)
            .to(dev)
        )
        topk_idx = (
            torch.stack(
                [torch.randperm(5 * PAGE, generator=gen)[:512] for _ in range(batch)]
            )
            .to(torch.int32)
            .unsqueeze(1)
            .to(dev)
        )
        swa_len = torch.tensor(swa_lengths, dtype=torch.int32, device=dev)
        topk_len = torch.tensor(topk_lengths, dtype=torch.int32, device=dev)
        # Some -1 padding inside the length too: must be skipped by both kernels.
        topk_idx[:, 0, 3] = -1
        ref = _reference(
            q, sink, [(swa_deq, swa_idx, swa_len), (topk_deq, topk_idx, topk_len)]
        )
        kwargs = dict(
            q=q,
            k_cache=swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            attn_sink=sink,
            extra_k_cache=topk_cache,
        )
        # the tilelang kernel only builds for 64 padded heads, so compare on the real heads
        pad = 64 - heads if heads < 64 else 0
        q_pad = torch.nn.functional.pad(q, (0, 0, 0, pad))
        sink_pad = torch.nn.functional.pad(sink, (0, pad))
        tilelang = flash_mla_with_kvcache_entrypoint(
            backend="tilelang",
            indices=swa_idx,
            topk_length=swa_len,
            extra_indices_in_kvcache=topk_idx,
            extra_topk_length=topk_len,
            **dict(kwargs, q=q_pad, attn_sink=sink_pad),
        )[0][:, :, :heads]
        # The backend folds the lengths into the index lists before this call.
        got = flash_mla_with_kvcache_entrypoint(
            backend="aiter_sparse",
            indices=_masked(swa_idx, swa_len),
            topk_length=swa_len,
            extra_indices_in_kvcache=_masked(topk_idx, topk_len),
            extra_topk_length=topk_len,
            **kwargs,
        )[0]
        self.assertEqual(got.shape, q.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        scale = ref.abs().max().item()
        err = (got.float() - ref).abs().max().item() / scale
        err_tl = (tilelang.float() - ref).abs().max().item() / scale
        self.assertLess(
            err, tol, f"aiter vs reference {err:.4f} (tilelang {err_tl:.4f})"
        )
        self.assertLess(
            (got.float() - tilelang.float()).abs().max().item() / scale, tol
        )
        for _ in range(5):
            again = flash_mla_with_kvcache_entrypoint(
                backend="aiter_sparse",
                indices=_masked(swa_idx, swa_len),
                topk_length=swa_len,
                extra_indices_in_kvcache=_masked(topk_idx, topk_len),
                extra_topk_length=topk_len,
                **kwargs,
            )[0]
            self.assertTrue(torch.equal(again, got))

    def test_full_lists_16_heads(self):
        self._assert_matches_reference_and_tilelang(1, 16, [128], [512])

    def test_short_context_lengths(self):
        # a context shorter than the window and the top-k width: the length masks live slots left in the list
        self._assert_matches_reference_and_tilelang(
            3, 16, [101, 128, 5], [100, 512, 1], seed=1
        )

    def test_short_lists_skip_the_padding_key(self):
        """The -1 inside the length (index 3 of the top-k list) must not be attended: on
        lists this short every key carries a visible share of the softmax mass, so a stray
        key -- wrapped to the last slot or otherwise -- fails at this tolerance where the
        640-key cases above would absorb it."""
        self._assert_matches_reference_and_tilelang(
            2, 16, [2, 3], [4, 5], seed=6, tol=TOL_SHORT
        )

    def test_padded_heads(self):
        # The model pads the per-rank heads to 64 (zero q, zero sink).
        self._assert_matches_reference_and_tilelang(
            2, 64, [128, 64], [512, 300], seed=2
        )

    def _real_vs_padded_heads(self, batch, seed):
        """The 16-head call must return bitwise what the padded 64-head call returned on the real heads."""
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        heads = 16
        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        swa_cache, _ = _pack_cache(2, dev, gen)
        topk_cache, _ = _pack_cache(5, dev, gen)
        q = (torch.randn(batch, 1, heads, D, generator=gen) * 0.5).to(torch.bfloat16)
        q = q.to(dev)
        sink = (torch.randn(heads, generator=gen) * 0.5).to(dev)
        swa_idx = torch.stack(
            [torch.randperm(2 * PAGE, generator=gen)[:128] for _ in range(batch)]
        )
        topk_idx = torch.stack(
            [torch.randperm(5 * PAGE, generator=gen)[:512] for _ in range(batch)]
        )
        swa_idx = swa_idx.to(torch.int32).unsqueeze(1).to(dev)
        topk_idx = topk_idx.to(torch.int32).unsqueeze(1).to(dev)
        swa_len = torch.randint(1, 129, (batch,), generator=gen).to(torch.int32)
        topk_len = torch.randint(1, 513, (batch,), generator=gen).to(torch.int32)
        kwargs = dict(
            backend="aiter_sparse",
            k_cache=swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            extra_k_cache=topk_cache,
            indices=_masked(swa_idx, swa_len.to(dev)),
            extra_indices_in_kvcache=_masked(topk_idx, topk_len.to(dev)),
        )
        q_pad = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
        sink_pad = torch.nn.functional.pad(sink, (0, 64 - heads))
        padded = flash_mla_with_kvcache_entrypoint(
            q=q_pad, attn_sink=sink_pad, **kwargs
        )[0]
        real = flash_mla_with_kvcache_entrypoint(q=q, attn_sink=sink, **kwargs)[0]
        self.assertEqual(real.shape, q.shape)
        self.assertTrue(torch.equal(real, padded[:, :, :heads]))

    def test_real_heads_match_padded_heads_bs1(self):
        self._real_vs_padded_heads(1, seed=3)

    def test_real_heads_match_padded_heads_bs3(self):
        self._real_vs_padded_heads(3, seed=4)

    def test_real_heads_match_padded_heads_bs8(self):
        self._real_vs_padded_heads(8, seed=5)

    def test_head_pad_predicate_follows_kernel_choice(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.hip_flash_mla import (
            hip_attention_needs_head_pad,
        )

        # gfx950 "auto" is the aiter kernel, which takes the real head count.
        with envs.SGLANG_HACK_FLASHMLA_BACKEND.override("auto"):
            self.assertFalse(hip_attention_needs_head_pad())
        with envs.SGLANG_HACK_FLASHMLA_BACKEND.override("triton"):
            self.assertFalse(hip_attention_needs_head_pad())
        # the tilelang kernel is built for the 64-padded widths
        with envs.SGLANG_HACK_FLASHMLA_BACKEND.override("tilelang"):
            self.assertTrue(hip_attention_needs_head_pad())

    def test_fold_lengths_into_index_lists(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            _fold_lengths_into_index_lists,
        )

        idx = torch.arange(2 * 1 * 6, dtype=torch.int32, device="cuda").view(2, 1, 6)
        out, extra = _fold_lengths_into_index_lists(
            idx, torch.tensor([2, 6], dtype=torch.int32, device="cuda")
        )
        self.assertEqual(out[0, 0].tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(out[1, 0].tolist(), [6, 7, 8, 9, 10, 11])
        self.assertIsNone(extra)
        # no lengths: nothing to fold, the caller keeps its list
        self.assertEqual(_fold_lengths_into_index_lists(idx, None), (None, None))

    def test_fold_cache_follows_the_index_source(self):
        """Layers between two index sources fold the first source's list once; the layers
        after the second source must fold its list, not reuse the first one's."""
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4AttnMetadata,
            _fold_lengths_for_aiter_sparse,
        )

        dev = "cuda"
        i32 = dict(dtype=torch.int32, device=dev)
        swa = torch.arange(2 * 1 * 64, **i32).view(2, 1, 64)
        swa_len = torch.tensor([64, 64], **i32)
        core = DSV4AttnMetadata(
            page_size=64,
            page_table=torch.zeros(1, **i32),
            raw_out_loc=torch.zeros(1, **i32),
            cuda_int32_kwargs={},
            seq_lens_casual=torch.zeros(2, **i32),
            positions_casual=torch.zeros(2, **i32),
            swa_page_indices=swa,
            swa_topk_lengths=swa_len,
            index_topk=64,
        )
        core.c2_sparse_page_indices = torch.arange(2 * 1 * 64, **i32).view(2, 1, 64)
        core.c2_sparse_topk_lengths = torch.tensor([3, 64], **i32)
        extra = core.sparse_page_indices(2)

        _, first = _fold_lengths_for_aiter_sparse(
            core, 2, swa, swa_len, extra, core.c2_sparse_topk_lengths
        )
        self.assertEqual(first[0, 0, :4].tolist(), [0, 1, 2, -1])
        # the next index source writes new picks into the same buffer
        core.drop_folded_sparse_indices(2)
        extra.add_(1000)
        core.c2_sparse_topk_lengths.fill_(2)
        _, second = _fold_lengths_for_aiter_sparse(
            core, 2, swa, swa_len, extra, core.c2_sparse_topk_lengths
        )
        self.assertEqual(second[0, 0, :4].tolist(), [1000, 1001, -1, -1])
        # ratio 1's folds survive a ratio-2 rewrite
        core._aiter_sparse_masked_indices[(1, 0, (), None, None)] = "kept"
        core.drop_folded_sparse_indices(2)
        self.assertEqual(
            list(core._aiter_sparse_masked_indices), [(1, 0, (), None, None)]
        )


SWA, TOPK = 128, 512


def _prefill_lists(num_tokens, device, gen):
    """Causal prefill lists as the indexer emits them: SWA holds the 128 most recent
    slots, top-k 512 random earlier slots, both -1 padded."""
    pos = torch.arange(num_tokens)
    swa = pos[:, None] - torch.arange(SWA)[None, :]
    swa = torch.where(swa >= 0, swa, torch.full_like(swa, -1))
    topk = torch.argsort(torch.rand(num_tokens, num_tokens, generator=gen), dim=1)
    topk = topk[:, :TOPK]
    topk = torch.where(topk <= pos[:, None], topk, torch.full_like(topk, -1))
    topk = topk.sort(dim=1, descending=True).values
    return (
        swa.to(torch.int32).view(num_tokens, 1, SWA).to(device),
        topk.to(torch.int32).view(num_tokens, 1, TOPK).to(device),
    )


def _reference_prefill(q, sink, sets, chunk=256):
    """q [t, 1, h, D] bf16; sets = [(deq_keys [slots, D], indices [t, 1, w])].
    fp32 softmax over the valid (!= -1) gathered keys plus the sink logit."""
    t, _, h, _ = q.shape
    out = torch.empty(t, 1, h, D, device=q.device, dtype=torch.float32)
    for t0 in range(0, t, chunk):
        t1 = min(t, t0 + chunk)
        idx = torch.cat([i[t0:t1, 0] for _, i in sets], dim=1)
        keys = torch.cat(
            [deq[i[t0:t1, 0].clamp(min=0).long()] for deq, i in sets], dim=1
        )
        s = torch.einsum("chd,cnd->chn", q[t0:t1, 0].float(), keys) * SCALE
        s = torch.where((idx >= 0)[:, None, :], s, torch.full_like(s, float("-inf")))
        logits = torch.cat([s, sink[None, :, None].expand(t1 - t0, h, 1)], dim=-1)
        p = torch.softmax(logits, dim=-1)[..., :-1]
        out[t0:t1, 0] = torch.einsum("chn,cnd->chd", p, keys)
    return out


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparsePrefill(CustomTestCase):
    NUM_TOKENS = 2048
    HEADS = 16

    @classmethod
    def setUpClass(cls):
        gen = torch.Generator(device="cpu").manual_seed(0)
        dev = torch.device("cuda")
        blocks = cls.NUM_TOKENS // PAGE + 1
        cls.swa_cache, swa_deq = _pack_cache(blocks, dev, gen, fp8_view=False)
        cls.topk_cache, topk_deq = _pack_cache(blocks, dev, gen, fp8_view=False)
        cls.q = (
            (torch.randn(cls.NUM_TOKENS, 1, cls.HEADS, D, generator=gen) * 0.5)
            .to(torch.bfloat16)
            .to(dev)
        )
        cls.sink = (torch.randn(cls.HEADS, generator=gen) * 0.5).to(dev)
        cls.swa_idx, cls.topk_idx = _prefill_lists(cls.NUM_TOKENS, dev, gen)
        cls.ref = _reference_prefill(
            cls.q, cls.sink, [(swa_deq, cls.swa_idx), (topk_deq, cls.topk_idx)]
        )

    def _run(self, backend, rows=slice(None)):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        return flash_mla_with_kvcache_entrypoint(
            backend=backend,
            q=self.q[rows],
            k_cache=self.swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            indices=self.swa_idx[rows],
            topk_length=None,
            attn_sink=self.sink,
            extra_k_cache=self.topk_cache,
            extra_indices_in_kvcache=self.topk_idx[rows],
            extra_topk_length=None,
        )[0]

    def test_matches_reference_and_triton(self):
        got = self._run("aiter_sparse")
        self.assertEqual(got.shape, self.q.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        scale = self.ref.abs().max().item()
        err = (got.float() - self.ref).abs().max().item() / scale
        triton_out = self._run("triton")
        err_triton = (triton_out.float() - self.ref).abs().max().item() / scale
        # Measured 2.5e-3 for both kernels (bf16 q, bf16 probabilities).
        self.assertLess(
            err, 1e-2, f"aiter vs reference {err:.2e} (triton {err_triton:.2e})"
        )
        self.assertLess(
            (got.float() - triton_out.float()).abs().max().item() / scale, 1e-2
        )
        for _ in range(5):
            self.assertTrue(torch.equal(self._run("aiter_sparse"), got))

    def test_batch_invariant(self):
        from sglang.srt.layers.attention import hip_flash_mla

        # both batches are at or above the unsplit threshold, so each row runs one program in the same order
        half = self.NUM_TOKENS // 2
        self.assertGreaterEqual(
            half, hip_flash_mla._AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS
        )
        full = self._run("aiter_sparse")
        part = self._run("aiter_sparse", rows=slice(0, half))
        self.assertTrue(torch.equal(full[:half], part))


def _freqs(device, max_pos=8192, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.rand(max_pos, ROPE // 2, generator=gen) * 2 * math.pi
    freqs_cis = torch.polar(torch.ones_like(angles), angles).to(device)
    return freqs_cis, torch.view_as_real(freqs_cis).flatten(-2).contiguous()


def _model_inverse_rope(x, freqs_real, positions):
    """The model's standalone inverse RoPE of the attention output: `fused_rope_inplace(...,
    inverse=True)` with the batched flat kernel (`set_batched_rope(True)`)."""
    from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

    set_batched_rope(True)
    freqs_cis = torch.view_as_complex(freqs_real.view(freqs_real.shape[0], -1, 2))
    fused_rope_inplace(x, None, freqs_cis, positions, inverse=True)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseDecodeReduce(CustomTestCase):
    def _inputs(self, batch, heads, seed, swa_len=128, topk_len=512):

        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        swa_cache, _ = _pack_cache(2, dev, gen)
        topk_cache, _ = _pack_cache(5, dev, gen)
        q = (torch.randn(batch, heads, D, generator=gen) * 0.5).to(torch.bfloat16)
        sink = (torch.randn(heads, generator=gen) * 0.5).to(dev)
        swa_idx = torch.stack(
            [torch.randperm(2 * PAGE, generator=gen)[:128] for _ in range(batch)]
        )
        topk_idx = torch.stack(
            [torch.randperm(5 * PAGE, generator=gen)[:512] for _ in range(batch)]
        )
        swa_idx = swa_idx.to(torch.int32).unsqueeze(1).to(dev)
        topk_idx = topk_idx.to(torch.int32).unsqueeze(1).to(dev)

        def lengths(n):
            return torch.full((batch,), n, dtype=torch.int32, device=dev)

        return dict(
            q=q.to(dev),
            sink=sink,
            swa_cache=swa_cache.view(torch.uint8).squeeze(2),
            topk_cache=topk_cache.view(torch.uint8).squeeze(2),
            swa_idx=_masked(swa_idx, lengths(swa_len)).reshape(-1),
            topk_idx=_masked(topk_idx, lengths(topk_len)).reshape(-1),
        )

    @staticmethod
    def _indptr(n, width):
        return torch.arange(0, (n + 1) * width, width, dtype=torch.int32, device="cuda")

    def _aiter(self, inputs, kv_splits, skip_reduce):
        from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

        n = inputs["q"].shape[0]
        return pa_decode_sparse(
            inputs["q"],
            inputs["swa_cache"],
            inputs["swa_idx"],
            self._indptr(n, 128),
            inputs["sink"],
            SCALE,
            extra_cache=inputs["topk_cache"],
            extra_indices=inputs["topk_idx"],
            extra_indptr=self._indptr(n, 512),
            kv_splits=kv_splits,
            skip_reduce=skip_reduce,
        )

    def test_bitwise_against_aiter_reduce(self):
        from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
            aiter_sparse_split_reduce,
        )

        for batch, heads, splits, seed in [
            (1, 16, 4, 0),
            (1, 16, 2, 1),
            (1, 16, 8, 2),
            (3, 16, 4, 3),
            (8, 16, 8, 4),
            (1, 64, 4, 5),
            (16, 16, 4, 6),
        ]:
            with self.subTest(batch=batch, heads=heads, splits=splits):
                # Partial lists on the larger batches: some splits come out empty.
                lens = (128, 512) if batch == 1 else (77, 301)
                inputs = self._inputs(batch, heads, seed, *lens)
                ref = self._aiter(inputs, splits, skip_reduce=False)
                acc, m, lsum = self._aiter(inputs, splits, skip_reduce=True)
                self.assertEqual(tuple(acc.shape), (batch, splits, heads, D))
                got = aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"])
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(got, ref))
                self.assertTrue(
                    torch.equal(
                        aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"]), got
                    )
                )

    def test_no_sink(self):
        from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

        from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
            aiter_sparse_split_reduce,
        )

        inputs = self._inputs(4, 16, 7, 77, 301)
        args = (
            inputs["q"],
            inputs["swa_cache"],
            inputs["swa_idx"],
            self._indptr(4, 128),
            None,
            SCALE,
        )
        ref = pa_decode_sparse(*args, kv_splits=4)
        acc, m, lsum = pa_decode_sparse(*args, kv_splits=4, skip_reduce=True)
        self.assertTrue(torch.equal(aiter_sparse_split_reduce(acc, m, lsum, None), ref))

    def test_inverse_rope_matches_flat_kernel(self):
        from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
            aiter_sparse_split_reduce,
        )

        dev = torch.device("cuda")
        _, fr = _freqs(dev)
        for batch, heads, splits, pos_dtype in [
            (1, 16, 4, torch.int64),
            (5, 16, 4, torch.int32),
            (2, 64, 2, torch.int64),
        ]:
            with self.subTest(batch=batch, heads=heads, splits=splits):
                inputs = self._inputs(batch, heads, 20 + batch)
                acc, m, lsum = self._aiter(inputs, splits, skip_reduce=True)
                pos = torch.randint(0, 8192, (batch,), device=dev, dtype=pos_dtype)
                plain = aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"])
                ref = plain.clone()
                _model_inverse_rope(ref[..., -ROPE:], fr, pos)
                got = aiter_sparse_split_reduce(
                    acc, m, lsum, inputs["sink"], inv_rope=(fr, pos)
                )
                self.assertTrue(torch.equal(got, ref))
                self.assertTrue(torch.equal(got[..., :-ROPE], plain[..., :-ROPE]))
        # Random partials: a large sample of the rope arithmetic alone.
        gen = torch.Generator(device="cpu").manual_seed(11)
        T, S, H = 256, 4, 16
        acc = (torch.randn(T, S, H, D, generator=gen) * 20).to(dev)
        m = torch.randn(T, S, H, generator=gen).to(dev)
        lsum = (torch.rand(T, S, H, generator=gen) * 50 + 1).to(dev)
        sink = torch.randn(H, generator=gen).to(dev)
        pos = torch.randint(0, 8192, (T,), generator=gen).to(dev)
        ref = aiter_sparse_split_reduce(acc, m, lsum, sink)
        _model_inverse_rope(ref[..., -ROPE:], fr, pos)
        got = aiter_sparse_split_reduce(acc, m, lsum, sink, inv_rope=(fr, pos))
        self.assertTrue(torch.equal(got, ref))

    def test_entrypoint_folds_inverse_rope(self):
        """With `inv_rope` the entrypoint must equal the kernel plus the flat inverse rope, for every HIP kernel."""
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        dev = torch.device("cuda")
        _, fr = _freqs(dev)
        for batch, heads, seed in [(1, 16, 30), (6, 16, 31), (2, 64, 32)]:
            with self.subTest(batch=batch, heads=heads):
                inputs = self._inputs(batch, heads, seed, 77 if batch > 1 else 128, 512)
                pos = torch.randint(0, 8192, (batch,), device=dev)
                kwargs = dict(
                    q=inputs["q"].unsqueeze(1),
                    k_cache=inputs["swa_cache"].unsqueeze(2).view(torch.float8_e4m3fn),
                    head_dim_v=D,
                    block_table=None,
                    cache_seqlens=None,
                    tile_scheduler_metadata=None,
                    softmax_scale=SCALE,
                    is_fp8_kvcache=True,
                    attn_sink=inputs["sink"],
                    indices=inputs["swa_idx"].view(batch, 1, 128),
                    extra_k_cache=inputs["topk_cache"]
                    .unsqueeze(2)
                    .view(torch.float8_e4m3fn),
                    extra_indices_in_kvcache=inputs["topk_idx"].view(batch, 1, 512),
                )
                for backend in (
                    ("aiter_sparse", "tilelang") if heads == 64 else ("aiter_sparse",)
                ):
                    ref = flash_mla_with_kvcache_entrypoint(backend=backend, **kwargs)[
                        0
                    ]
                    ref = ref.clone()
                    _model_inverse_rope(ref.view(batch, heads, D)[..., -ROPE:], fr, pos)
                    got = flash_mla_with_kvcache_entrypoint(
                        backend=backend, inv_rope=(fr, pos), **kwargs
                    )[0]
                    self.assertEqual(got.shape, ref.shape, backend)
                    self.assertTrue(torch.equal(got, ref), backend)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseDecodeSplitPin(CustomTestCase):
    """``SGLANG_OPT_HIP_ATTN_KV_SPLITS`` pins the split-KV count of the decode rows: with it
    a row's output is bitwise the same at every batch size; without it aiter's cost model
    changes the count (hence the fp32 combine order) with the batch size past 64 rows."""

    ROWS = 96  # aiter picks 2 splits here, 4 at 1..64 rows

    def _run(self, batch, seed=40):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        dev = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        swa_cache, _ = _pack_cache(2, dev, gen)
        topk_cache, _ = _pack_cache(5, dev, gen)
        heads = 16
        # row i of every batch gets the same query and lists
        rows = []
        for i in range(self.ROWS):
            g = torch.Generator(device="cpu").manual_seed(seed * 1000 + i)
            rows.append(
                (
                    (torch.randn(heads, D, generator=g) * 0.5).to(torch.bfloat16),
                    torch.randperm(2 * PAGE, generator=g)[:128],
                    torch.randperm(5 * PAGE, generator=g)[:512],
                )
            )
        q = torch.stack([r[0] for r in rows])[:batch].unsqueeze(1).to(dev)
        swa_idx = torch.stack([r[1] for r in rows])[:batch].to(torch.int32)
        topk_idx = torch.stack([r[2] for r in rows])[:batch].to(torch.int32)
        sink = (torch.randn(heads, generator=gen) * 0.5).to(dev)

        def lens(n):
            return torch.full((batch,), n, dtype=torch.int32, device=dev)

        return flash_mla_with_kvcache_entrypoint(
            backend="aiter_sparse",
            q=q,
            k_cache=swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            attn_sink=sink,
            extra_k_cache=topk_cache,
            indices=_masked(swa_idx.unsqueeze(1).to(dev), lens(128)),
            extra_indices_in_kvcache=_masked(topk_idx.unsqueeze(1).to(dev), lens(512)),
        )[0]

    def test_pinned_splits_are_batch_invariant(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_HIP_ATTN_KV_SPLITS.override(4):
            one, eight, many = self._run(1), self._run(8), self._run(self.ROWS)
        self.assertTrue(torch.equal(one, eight[:1]))
        self.assertTrue(torch.equal(eight, many[:8]))


if __name__ == "__main__":
    unittest.main()
