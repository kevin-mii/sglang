"""Gluon sparse prefill on an fp8 KV pool must match the Triton kernel it replaces."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

# One TP4 rank of MiniMax-M3: 16 query heads, 1 KV head, 1 index head.
Q_HEADS, DIM, BLOCK, TOPK = 16, 128, 128, 16
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1


def _inputs(prefixes, exts, kv_dtype, k_scale, v_scale):
    torch.manual_seed(20261007)
    device = "cuda"
    seqs = [p + e for p, e in zip(prefixes, exts)]
    nslots = sum(seqs) + 1024
    # Every request reads a shuffled set of pool slots.
    perm = torch.randperm(nslots, device=device, dtype=torch.int32)
    req_to_token = torch.zeros((len(seqs), max(seqs)), dtype=torch.int32, device=device)
    start = 0
    for b, seq in enumerate(seqs):
        req_to_token[b, :seq] = perm[start : start + seq]
        start += seq

    def pool(scale):
        return (torch.randn(nslots, 1, DIM, device=device) / scale).to(kv_dtype)

    k_cache, v_cache, idx_k_cache = pool(k_scale), pool(v_scale), pool(1.0)
    total_q = sum(exts)
    q = torch.randn(total_q, Q_HEADS, DIM, device=device, dtype=torch.bfloat16)
    idx_q = torch.randn(total_q, 1, DIM, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0] + exts, dtype=torch.int32, device=device).cumsum(0)
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_q=idx_q,
        idx_k_cache=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=torch.arange(len(seqs), dtype=torch.int32, device=device),
        cu_seqlens=cu.to(torch.int32),
        seq_lens=torch.tensor(seqs, dtype=torch.int32, device=device),
        prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=device),
        max_seqlen_q=max(exts),
        max_seqlen_k=max(seqs),
    )


@unittest.skipUnless(torch.version.hip, "AITER Gluon prefill is ROCm-only")
class TestMiniMaxGluonPrefillFp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            raise unittest.SkipTest("Gluon sparse prefill requires gfx950")
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.layers.attention.minimax_sparse_ops import gluon_prefill

        if gluon_prefill.pa_decode_gluon is None:
            raise unittest.SkipTest("AITER Gluon paged attention is unavailable")
        cls.fp8_dtype = fp8_dtype
        cls.gluon = gluon_prefill

    def check(self, prefixes, exts, k_scale=1.0, v_scale=1.0):
        from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
            flash_prefill_with_topk_index,
        )
        from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
            flash_prefill_with_gqa_share_sparse,
        )

        x = _inputs(prefixes, exts, self.fp8_dtype, k_scale, v_scale)
        seq_lens_cpu = x["seq_lens"].cpu()
        self.assertTrue(
            self.gluon.can_use_gluon_prefill(
                x["q"],
                x["k_cache"],
                x["v_cache"],
                None,
                BLOCK,
                seq_lens_cpu,
                None,
                k_scale,
                v_scale,
            ),
            "an fp8 KV pool must take the Gluon path",
        )
        common = dict(
            req_to_token=x["req_to_token"],
            slot_ids=x["slot_ids"],
            cu_seqlens=x["cu_seqlens"],
            seq_lens=x["seq_lens"],
            prefix_lens=x["prefix_lens"],
        )
        # The production indexer picks the blocks, with M3's init/local rules.
        _, topk_idx = flash_prefill_with_topk_index(
            q=x["idx_q"],
            k_cache=x["idx_k_cache"],
            v_cache=None,
            sink=None,
            max_seqlen_q=x["max_seqlen_q"],
            max_seqlen_k=x["max_seqlen_k"],
            block_size_q=1,
            block_size_k=BLOCK,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            disable_index_value=True,
            **common,
        )
        triton_out = flash_prefill_with_gqa_share_sparse(
            q=x["q"],
            k_cache=x["k_cache"],
            v_cache=x["v_cache"],
            sink=None,
            topk_idx=topk_idx,
            block_size_q=1,
            block_size_k=BLOCK,
            max_seqlen_q=x["max_seqlen_q"],
            k_scale=k_scale,
            v_scale=v_scale,
            **common,
        )
        gluon_out = self.gluon.gluon_sparse_prefill(
            q=x["q"],
            k_cache=x["k_cache"],
            v_cache=x["v_cache"],
            topk_idx=topk_idx,
            req_to_token=x["req_to_token"],
            req_pool_indices=x["slot_ids"],
            cu_seqlens=x["cu_seqlens"],
            seq_lens=x["seq_lens"],
            prefix_lens=x["prefix_lens"],
            seq_lens_cpu=seq_lens_cpu,
            block_size_k=BLOCK,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        err = (gluon_out.float() - triton_out.float()).abs().max().item()
        self.assertLess(err, 1e-2, f"gluon vs triton max abs error {err}")

    def test_chunk_over_long_prefix(self):
        for prefix in (32768, 190000):
            with self.subTest(prefix=prefix):
                self.check([prefix], [8192])

    def test_batched_extends(self):
        # A fresh request, a chunk boundary mid-block, and a short agent-turn extend.
        self.check([0, 65000, 120000], [3000, 4900, 292])

    def test_non_unit_scales(self):
        self.check([32768], [2048], k_scale=0.25, v_scale=1.5)


if __name__ == "__main__":
    unittest.main()
