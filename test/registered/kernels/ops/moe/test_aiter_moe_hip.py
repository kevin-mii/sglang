"""The one-launch MoE sorting, the ROCm decode router gate and the fused gate + sort must reproduce aiter's `moe_sorting` and `topk_gating` bit for bit."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


try:
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting
except ImportError:  # aiter is absent off ROCm
    aiter_moe_sorting = None

try:
    from aiter import topk_gating as aiter_topk_gating
except ImportError:  # aiter is absent off ROCm
    aiter_topk_gating = None


NUM_EXPERTS = 384
MODEL_DIM = 5120
HIDDEN = 5120
TOPK = 6
ROUTED_SCALING = 1.5


def _routing(num_tokens, topk, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device=device, generator=g)[:topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(num_tokens, topk, device=device, generator=g)
    return ids, weights


def _assert_same_sort(test, ref, out, block_size, num_valid_rows, num_tokens):
    """Compare the region aiter's GEMMs read (up to the padded total)."""
    test.assertTrue(torch.equal(ref[3], out[3]), "num_valid_ids")
    total = int(ref[3][0])
    ref_ids, out_ids = ref[0][:total], out[0][:total]
    tokens = ref_ids & 0xFFFFFF
    test.assertTrue(torch.equal(tokens, out_ids & 0xFFFFFF), "sorted tokens")
    racy_slot = (tokens >= num_valid_rows) & (tokens < num_tokens)
    test.assertTrue(
        torch.equal(
            torch.where(racy_slot, 0, ref_ids), torch.where(racy_slot, 0, out_ids)
        ),
        "sorted slots",
    )
    test.assertTrue(torch.equal(ref[1][:total], out[1][:total]), "sorted weights")
    test.assertTrue(
        torch.equal(ref[2][: total // block_size], out[2][: total // block_size]),
        "sorted expert ids",
    )
    test.assertEqual(ref[4].shape, out[4].shape)
    test.assertTrue(torch.equal(ref[4], out[4]), "moe_buf")


@unittest.skipUnless(is_hip() and aiter_moe_sorting is not None, "ROCm + aiter only")
class TestFusedAiterMoeSorting(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.aiter_moe_sorting_fused import (
            fused_aiter_moe_sorting,
            local_expert_ids_from_mask,
        )

        self.fused = fused_aiter_moe_sorting
        self.local_ids = local_expert_ids_from_mask
        self.device = torch.device("cuda")

    def _mask(self, num_local, rank):
        mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=self.device)
        mask[rank * num_local : (rank + 1) * num_local] = 1
        return mask

    def test_matches_aiter_expert_parallel(self):
        for num_local, topk in ((96, 6), (48, 8), (384, 6)):
            for rank in (0, NUM_EXPERTS // num_local - 1):
                mask = self._mask(num_local, rank)
                local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
                for num_tokens in (1, 31, 64):
                    for block_size in (16, 64):
                        ids, weights = _routing(
                            num_tokens, topk, num_tokens, self.device
                        )
                        ref = aiter_moe_sorting(
                            ids,
                            weights,
                            NUM_EXPERTS,
                            MODEL_DIM,
                            torch.bfloat16,
                            block_size,
                            mask,
                            None,
                            0,
                            accumulate=True,
                        )
                        out = self.fused(
                            ids,
                            weights,
                            local_ids,
                            num_local,
                            NUM_EXPERTS,
                            MODEL_DIM,
                            torch.bfloat16,
                            block_size,
                            zero_moe_buf=True,
                        )
                        _assert_same_sort(
                            self, ref, out, block_size, num_tokens, num_tokens
                        )
                        self.assertEqual(int(out[4].abs().sum()), 0)

    def test_padded_rows_are_masked_in_place(self):
        mask = self._mask(96, 1)
        local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
        for num_tokens in (2, 17, 64):
            for num_valid in (1, num_tokens // 2, num_tokens):
                ids, weights = _routing(
                    num_tokens, 6, 7 * num_tokens + num_valid, self.device
                )
                count = torch.tensor([num_valid], dtype=torch.int32, device=self.device)
                masked_ids, masked_weights = ids.clone(), weights.clone()
                masked_ids[num_valid:] = 0
                masked_weights[num_valid:] = 0.0
                ref = aiter_moe_sorting(
                    masked_ids,
                    masked_weights,
                    NUM_EXPERTS,
                    MODEL_DIM,
                    torch.bfloat16,
                    32,
                    mask,
                    None,
                    0,
                    accumulate=True,
                )
                out = self.fused(
                    ids,
                    weights,
                    local_ids,
                    96,
                    NUM_EXPERTS,
                    MODEL_DIM,
                    torch.bfloat16,
                    32,
                    zero_moe_buf=True,
                    num_token_non_padded=count,
                )
                _assert_same_sort(self, ref, out, 32, num_valid, num_tokens)
                # The fills happened in the same launch.
                self.assertTrue(torch.equal(ids, masked_ids))
                self.assertTrue(torch.equal(weights, masked_weights))

    def test_without_expert_mask(self):
        num_experts = 64
        local_ids = self.local_ids(None, num_experts, self.device)
        for num_tokens in (1, 8, 32):
            g = torch.Generator(device=self.device).manual_seed(num_tokens)
            ids = torch.stack(
                [
                    torch.randperm(num_experts, device=self.device, generator=g)[:6]
                    for _ in range(num_tokens)
                ]
            ).to(torch.int32)
            weights = torch.rand(num_tokens, 6, device=self.device, generator=g)
            for accumulate in (True, False):
                ref = aiter_moe_sorting(
                    ids,
                    weights,
                    num_experts,
                    1024,
                    torch.bfloat16,
                    32,
                    None,
                    None,
                    0,
                    accumulate=accumulate,
                )
                out = self.fused(
                    ids,
                    weights,
                    local_ids,
                    num_experts,
                    num_experts,
                    1024,
                    torch.bfloat16,
                    32,
                    zero_moe_buf=accumulate,
                )
                _assert_same_sort(self, ref, out, 32, num_tokens, num_tokens)

    def test_runner_override_scopes_to_requests(self):
        """The wrapper must answer only inside a scoped request and otherwise fall back, masking the deferred rows."""
        import aiter.fused_moe as aiter_fused_moe

        from sglang.kernels.ops.moe.aiter_moe_sorting_fused import (
            AITER_FUSED_SORT_MAX_TOKENS,
        )
        from sglang.srt.layers.moe.moe_runner import aiter as runner

        self.assertTrue(runner._install_fused_sorting_override())
        wrapped = aiter_fused_moe.moe_sorting
        self.assertIsNot(wrapped, aiter_moe_sorting)
        mask = self._mask(96, 0)
        ids, weights = _routing(8, 6, 3, self.device)
        # Outside a request: aiter's function, untouched inputs.
        ref = aiter_moe_sorting(
            ids, weights, NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 32, mask, None, 0
        )
        out = wrapped(
            ids, weights, NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 32, mask, None, 0
        )
        _assert_same_sort(self, ref, out, 32, 8, 8)
        # Inside a request with a pad count: fused kernel, rows masked in place.
        count = torch.tensor([3], dtype=torch.int32, device=self.device)
        masked_ids, masked_weights = ids.clone(), weights.clone()
        masked_ids[3:] = 0
        masked_weights[3:] = 0.0
        ref = aiter_moe_sorting(
            masked_ids,
            masked_weights,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            32,
            mask,
            None,
            0,
        )
        with runner._fused_sorting_scope(count) as request:
            out = wrapped(
                ids, weights, NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 32, mask, None, 0
            )
        self.assertTrue(request.fired)
        _assert_same_sort(self, ref, out, 32, 3, 8)
        self.assertTrue(torch.equal(ids, masked_ids))
        # Too many rows: aiter's kernel, but the deferred masks still happen.
        big_ids, big_weights = _routing(
            AITER_FUSED_SORT_MAX_TOKENS + 1, 6, 5, self.device
        )
        count = torch.tensor([10], dtype=torch.int32, device=self.device)
        with runner._fused_sorting_scope(count) as request:
            wrapped(
                big_ids,
                big_weights,
                NUM_EXPERTS,
                MODEL_DIM,
                torch.bfloat16,
                32,
                mask,
                None,
                0,
            )
        self.assertTrue(request.fired)
        self.assertEqual(int(big_ids[10:].abs().sum()), 0)
        self.assertEqual(float(big_weights[10:].abs().sum()), 0.0)
        self.assertNotEqual(int(big_ids[:10].abs().sum()), 0)


def _aiter_gate(logits, bias, topk, renorm, rsf):
    weights = torch.empty(
        logits.shape[0], topk, dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=logits.device)
    aiter_topk_gating(
        weights, ids, logits, bias, renorm, rsf, score_func="sqrtsoftplus"
    )
    return weights, ids


@unittest.skipUnless(
    is_hip() and is_gfx95_supported() and aiter_topk_gating is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGate(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.moe.rocm_router_gate import (
            ROCM_ROUTER_MAX_TOKENS,
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_max_tokens,
            rocm_router_reduce_partials,
        )

        self.max_tokens = ROCM_ROUTER_MAX_TOKENS
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.max_tokens_for = rocm_router_max_tokens
        self.reduce = rocm_router_reduce_partials
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias_bf16 = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)

    def _randn(self, *shape, scale=1.0):
        return torch.randn(*shape, device=self.device, generator=self.gen) * scale

    def _assert_same_gate(self, logits, bias, renorm=True, rsf=ROUTED_SCALING, msg=""):
        for topk in (TOPK, 8):
            ref_w, ref_i = _aiter_gate(logits, bias, topk, renorm, rsf)
            out_w, out_i = self.gate(logits, bias, topk, renorm, rsf)
            self.assertTrue(torch.equal(ref_i, out_i), f"ids {msg} topk {topk}")
            self.assertTrue(torch.equal(ref_w, out_w), f"weights {msg} topk {topk}")

    def test_gate_matches_aiter_random_logits(self):
        for num_tokens in (1, 7, 64, 1024):
            for scale in (1.0, 40.0):
                logits = self._randn(num_tokens, NUM_EXPERTS, scale=scale)
                self._assert_same_gate(
                    logits, self.bias_bf16, msg=f"fp32 {num_tokens} {scale}"
                )
                self._assert_same_gate(
                    logits, self.bias_bf16.float(), msg=f"fp32 bias {num_tokens}"
                )
                self._assert_same_gate(
                    logits.to(torch.bfloat16), self.bias_bf16, msg=f"bf16 {num_tokens}"
                )
                self._assert_same_gate(logits, None, msg=f"no bias {num_tokens}")
                self._assert_same_gate(
                    logits,
                    self.bias_bf16,
                    renorm=False,
                    rsf=1.0,
                    msg=f"no renorm {num_tokens}",
                )

    def test_gate_matches_aiter_on_ties(self):
        zero_bias = torch.zeros(NUM_EXPERTS, device=self.device, dtype=torch.bfloat16)
        for num_tokens in (16, 512):
            levels = torch.randint(
                0, 3, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels, zero_bias, msg="3 levels")
            levels = torch.randint(
                0, 8, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            self._assert_same_gate(levels - 3, self.bias_bf16, msg="8 levels")
            self._assert_same_gate(
                torch.zeros(num_tokens, NUM_EXPERTS, device=self.device),
                zero_bias,
                msg="all equal",
            )
            few = torch.full(
                (num_tokens, NUM_EXPERTS), float("-inf"), device=self.device
            )
            few[:, :3] = 1.0
            self._assert_same_gate(few, zero_bias, msg="3 finite experts")

    def test_gate_matches_aiter_non_finite(self):
        for num_tokens in (16, 512):
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[logits < 0] = float("-inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="-inf")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, ::7] = float("nan")
            self._assert_same_gate(logits, self.bias_bf16, msg="nan")
            logits = self._randn(num_tokens, NUM_EXPERTS, scale=3.0)
            logits[:, 5] = float("inf")
            self._assert_same_gate(logits, self.bias_bf16, msg="+inf")

    def test_gate_score_every_bf16_value(self):
        # force the first six experts with a huge bias, no renorm, scale 1: weights are the raw scores
        values = (
            torch.arange(0, 65536, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .to(self.device)
            .float()
        )
        rows = 65536 // TOPK + 1
        padded = torch.zeros(rows * TOPK, device=self.device)
        padded[:65536] = values
        logits = torch.full((rows, NUM_EXPERTS), -1.0, device=self.device)
        logits[:, :TOPK] = padded.view(rows, TOPK)
        bias = torch.zeros(NUM_EXPERTS, device=self.device)
        bias[:TOPK] = 1e30
        ref_w, ref_i = _aiter_gate(logits, bias, TOPK, False, 1.0)
        out_w, out_i = self.gate(logits, bias, TOPK, False, 1.0)
        self.assertTrue(torch.equal(ref_i, out_i))
        both_nan = torch.isnan(ref_w) & torch.isnan(out_w)
        self.assertTrue(torch.equal(ref_w[~both_nan], out_w[~both_nan]))

    def test_max_tokens_gate(self):
        self.assertEqual(
            self.max_tokens_for(
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                topk=TOPK,
                weight_dtype=torch.bfloat16,
            ),
            self.max_tokens,
        )
        for kwargs in (
            dict(
                num_experts=256,
                hidden_size=HIDDEN,
                topk=TOPK,
                weight_dtype=torch.bfloat16,
            ),
            dict(
                num_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                topk=TOPK,
                weight_dtype=torch.float16,
            ),
            dict(
                num_experts=NUM_EXPERTS,
                hidden_size=5000,
                topk=TOPK,
                weight_dtype=torch.bfloat16,
            ),
        ):
            self.assertEqual(self.max_tokens_for(**kwargs), -1)

    def test_gemv_accuracy_batch_invariance_and_repeatability(self):
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        x = self._randn(self.max_tokens, HIDDEN).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(self.max_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(self.gemv(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        for num_tokens in (1, 2, 16, 17, 32, 33, 64):
            rows = x[:num_tokens]
            out = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(self.gemv(rows, weight), out)
            self.assertTrue(
                torch.equal(out, full[:num_tokens]), f"batch of {num_tokens}"
            )
            # The same row moved to another position of the batch.
            shifted = torch.roll(x, shifts=num_tokens, dims=0)
            out_shifted = torch.empty_like(full)
            self.reduce(self.gemv(shifted, weight), out_shifted)
            self.assertTrue(torch.equal(out_shifted, torch.roll(full, num_tokens, 0)))
        for _ in range(20):
            again = torch.empty_like(full)
            self.reduce(self.gemv(x, weight), again)
            self.assertTrue(torch.equal(again, full))

    def test_select_experts_consumes_partials(self):
        # select_experts must gate the partials exactly like the reduced logits and fill the logits buffer
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        config = TopKConfig(
            top_k=TOPK,
            renormalize=True,
            correction_bias=self.bias_bf16,
            routed_scaling_factor=ROUTED_SCALING,
            scoring_func="sqrtsoftplus",
            allow_routed_experts_capture=False,
        )
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        for num_tokens in (1, 30, 64):
            x = self._randn(num_tokens, HIDDEN).to(torch.bfloat16)
            partials = self.gemv(x, weight)
            logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(partials, logits)
            num_valid = torch.tensor(
                [max(1, num_tokens // 2)], dtype=torch.int32, device=self.device
            )
            for pad in (None, num_valid):
                ref = select_experts(
                    x, logits.clone(), config, num_token_non_padded=pad
                )
                buffer = torch.empty_like(logits)
                out = select_experts(
                    x,
                    buffer,
                    config,
                    num_token_non_padded=pad,
                    router_logits_partials=partials,
                )
                self.assertTrue(torch.equal(ref.topk_ids, out.topk_ids))
                self.assertTrue(torch.equal(ref.topk_weights, out.topk_weights))
                self.assertEqual(out.router_logits.data_ptr(), buffer.data_ptr())
                self.assertTrue(torch.equal(out.router_logits, logits))
            config.torch_native = True
            try:
                out = select_experts(
                    x, torch.empty_like(logits), config, router_logits_partials=partials
                )
            finally:
                config.torch_native = False
            self.assertTrue(torch.equal(out.router_logits, logits))

    def test_fused_gate_on_partials(self):
        weight = (self._randn(NUM_EXPERTS, HIDDEN) * 0.02).to(torch.bfloat16)
        for num_tokens in (1, 64):
            x = self._randn(num_tokens, HIDDEN).to(torch.bfloat16)
            partials = self.gemv(x, weight)
            logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
            self.reduce(partials, logits)
            ref_w, ref_i = _aiter_gate(
                logits, self.bias_bf16, TOPK, True, ROUTED_SCALING
            )
            fused_logits = torch.empty_like(logits)
            out_w, out_i = self.gate(
                fused_logits,
                self.bias_bf16,
                TOPK,
                True,
                ROUTED_SCALING,
                partials=partials,
            )
            self.assertTrue(torch.equal(fused_logits, logits))
            self.assertTrue(torch.equal(ref_i, out_i))
            self.assertTrue(torch.equal(ref_w, out_w))
            for _ in range(20):
                again_w, again_i = self.gate(
                    torch.empty_like(logits),
                    self.bias_bf16,
                    TOPK,
                    True,
                    ROUTED_SCALING,
                    partials=self.gemv(x, weight),
                )
                self.assertTrue(
                    torch.equal(again_i, out_i) and torch.equal(again_w, out_w)
                )


@unittest.skipUnless(
    is_hip()
    and is_gfx95_supported()
    and aiter_topk_gating is not None
    and aiter_moe_sorting is not None,
    "ROCm gfx95 + aiter only",
)
class TestRocmRouterGateSort(CustomTestCase):
    """One launch = gate launch + sorting launch, bit for bit, and aiter for both."""

    def setUp(self):
        from sglang.kernels.ops.moe.aiter_moe_sorting_fused import (
            fused_aiter_moe_sorting,
            local_expert_ids_from_mask,
        )
        from sglang.kernels.ops.moe.rocm_router_gate import (
            rocm_router_gate,
            rocm_router_gemv_split_k,
            rocm_router_reduce_partials,
        )
        from sglang.kernels.ops.moe.rocm_router_gate_sort import (
            ROCM_GATE_SORT_MAX_TOKENS,
            _handoff_buffer,
            rocm_router_gate_sort,
        )

        self.max_tokens = ROCM_GATE_SORT_MAX_TOKENS
        self.fused = rocm_router_gate_sort
        self.handoff = _handoff_buffer
        self.gate = rocm_router_gate
        self.gemv = rocm_router_gemv_split_k
        self.reduce = rocm_router_reduce_partials
        self.sort = fused_aiter_moe_sorting
        self.local_ids = local_expert_ids_from_mask
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.bias = (
            torch.randn(NUM_EXPERTS, device=self.device, generator=self.gen) * 0.5
        ).to(torch.bfloat16)
        self.weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device=self.device, generator=self.gen)
            * 0.02
        ).to(torch.bfloat16)

    def _mask(self, num_local, rank):
        if num_local == NUM_EXPERTS:
            return None
        mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=self.device)
        mask[rank * num_local : (rank + 1) * num_local] = 1
        return mask

    def _inputs(self, num_tokens, ties):
        if ties:
            logits = torch.randint(
                0, 3, (num_tokens, NUM_EXPERTS), device=self.device, generator=self.gen
            ).float()
            return logits, None
        x = torch.randn(num_tokens, HIDDEN, device=self.device, generator=self.gen).to(
            torch.bfloat16
        )
        partials = self.gemv(x, self.weight)
        logits = torch.empty(num_tokens, NUM_EXPERTS, device=self.device)
        self.reduce(partials, logits)
        return logits, partials

    def _assert_gate_sort_matches_two_launches(
        self, num_tokens, topk, block_size, num_local, rank, pad, ties
    ):
        msg = f"M={num_tokens} topk={topk} block={block_size} local={num_local}/{rank} pad={pad} ties={ties}"
        mask = self._mask(num_local, rank)
        local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
        logits, partials = self._inputs(num_tokens, ties)
        count = (
            torch.tensor(
                [max(1, num_tokens - 1)], dtype=torch.int32, device=self.device
            )
            if pad
            else None
        )
        n_valid = int(count) if pad else num_tokens
        # reference: the two launches, and aiter itself
        ref_w, ref_i = self.gate(
            logits.clone(), self.bias, topk, True, ROUTED_SCALING, partials=partials
        )
        aiter_w, aiter_i = _aiter_gate(logits, self.bias, topk, True, ROUTED_SCALING)
        self.assertTrue(
            torch.equal(aiter_w, ref_w) and torch.equal(aiter_i, ref_i), msg
        )
        ref_sort = self.sort(
            ref_i,
            ref_w,
            local_ids,
            num_local,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask is not None,
            num_token_non_padded=count,
        )
        masked_i, masked_w = aiter_i.clone(), aiter_w.clone()
        masked_i[n_valid:] = 0
        masked_w[n_valid:] = 0.0
        aiter_sort = aiter_moe_sorting(
            masked_i,
            masked_w,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask,
            None,
            0,
            accumulate=mask is not None,
        )
        fused_logits = torch.empty_like(logits) if partials is not None else logits
        out = self.fused(
            fused_logits,
            self.bias,
            topk,
            True,
            ROUTED_SCALING,
            partials,
            local_ids,
            NUM_EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            block_size,
            mask is not None,
            num_token_non_padded=count,
        )
        self.assertTrue(torch.equal(out[1], masked_i), f"ids {msg}")
        self.assertTrue(torch.equal(out[0], masked_w), f"weights {msg}")
        if partials is not None:
            self.assertTrue(torch.equal(fused_logits, logits), f"logits {msg}")
        _assert_same_sort(self, ref_sort, out[2:], block_size, n_valid, num_tokens)
        _assert_same_sort(self, aiter_sort, out[2:], block_size, n_valid, num_tokens)
        self.assertEqual(int(self.handoff(self.device).abs().sum()), 0, msg)

    def test_matches_two_launches_and_aiter(self):
        for num_tokens in range(1, self.max_tokens + 1):
            for topk in (6, 8):
                for block_size in (16, 32, 64):
                    for num_local, rank in (
                        (96, 0),
                        (96, 3),
                        (48, 7),
                        (NUM_EXPERTS, 0),
                    ):
                        for pad in (False, True):
                            for ties in (False, True):
                                self._assert_gate_sort_matches_two_launches(
                                    num_tokens,
                                    topk,
                                    block_size,
                                    num_local,
                                    rank,
                                    pad,
                                    ties,
                                )

    def test_repeated_and_graph_replayed_launches(self):
        """The hand-off buffer is left clean, so back-to-back launches and graph replays agree."""
        mask = self._mask(96, 1)
        local_ids = self.local_ids(mask, NUM_EXPERTS, self.device)
        for num_tokens in range(1, self.max_tokens + 1):
            logits, partials = self._inputs(num_tokens, False)
            ref_w, ref_i = self.gate(
                logits.clone(), self.bias, TOPK, True, ROUTED_SCALING, partials=partials
            )
            ref_sort = self.sort(
                ref_i,
                ref_w,
                local_ids,
                96,
                NUM_EXPERTS,
                MODEL_DIM,
                torch.bfloat16,
                32,
                True,
            )
            args = (
                self.bias,
                TOPK,
                True,
                ROUTED_SCALING,
                partials,
                local_ids,
                NUM_EXPERTS,
                MODEL_DIM,
                torch.bfloat16,
                32,
                True,
            )
            for _ in range(20):
                out = self.fused(torch.empty_like(logits), *args)
                self.assertTrue(
                    torch.equal(out[1], ref_i) and torch.equal(out[0], ref_w)
                )
                _assert_same_sort(self, ref_sort, out[2:], 32, num_tokens, num_tokens)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outs = [
                        self.fused(torch.empty_like(logits), *args) for _ in range(4)
                    ]
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(5):
                graph.replay()
                torch.cuda.synchronize()
                for out in outs:
                    self.assertTrue(
                        torch.equal(out[1], ref_i) and torch.equal(out[0], ref_w)
                    )
                    _assert_same_sort(
                        self, ref_sort, out[2:], 32, num_tokens, num_tokens
                    )
            self.assertEqual(int(self.handoff(self.device).abs().sum()), 0)

    def test_runtime_glue_hands_the_sort_to_the_runner(self):
        """select_experts learns the sorting arguments from the runner's first call, then the
        override returns the fused launch's outputs; a changed argument falls back."""
        import aiter.fused_moe as aiter_fused_moe

        from sglang.srt.layers.moe import rocm_fused_front as glue
        from sglang.srt.layers.moe.moe_runner import aiter as runner
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        self.assertTrue(runner._install_fused_sorting_override())
        wrapped = aiter_fused_moe.moe_sorting
        glue.reset_for_tests()
        config = TopKConfig(
            top_k=TOPK,
            renormalize=True,
            correction_bias=self.bias,
            routed_scaling_factor=ROUTED_SCALING,
            scoring_func="sqrtsoftplus",
            allow_routed_experts_capture=False,
        )
        mask = self._mask(96, 2)
        sort_args = (NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 32, mask, None, 0)
        for num_tokens in (1, 2):
            logits, partials = self._inputs(num_tokens, False)
            x = torch.randn(num_tokens, HIDDEN, device=self.device).to(torch.bfloat16)
            ref = select_experts(x, logits.clone(), config)
            ref_sort = aiter_moe_sorting(ref.topk_ids, ref.topk_weights, *sort_args)
            # first call: gate alone, the override sorts and records the arguments
            first = select_experts(
                x, torch.empty_like(logits), config, router_logits_partials=partials
            )
            self.assertTrue(torch.equal(first.topk_ids, ref.topk_ids))
            with runner._fused_sorting_scope(None):
                out = wrapped(first.topk_ids, first.topk_weights, *sort_args)
            _assert_same_sort(self, ref_sort, out, 32, num_tokens, num_tokens)
            # second call: one launch; the override hands back its outputs
            second = select_experts(
                x, torch.empty_like(logits), config, router_logits_partials=partials
            )
            self.assertTrue(torch.equal(second.topk_ids, ref.topk_ids))
            self.assertTrue(torch.equal(second.topk_weights, ref.topk_weights))
            pending = glue._pending_sorts[second.topk_ids.data_ptr()]
            self.assertIsNotNone(pending.outputs)
            with runner._fused_sorting_scope(None):
                out = wrapped(second.topk_ids, second.topk_weights, *sort_args)
            self.assertIs(out[0], pending.outputs[0])
            _assert_same_sort(self, ref_sort, out, 32, num_tokens, num_tokens)
            self.assertNotIn(second.topk_ids.data_ptr(), glue._pending_sorts)
            # a different block size: the pending outputs are refused, the ids sorted again
            third = select_experts(
                x, torch.empty_like(logits), config, router_logits_partials=partials
            )
            other_args = (NUM_EXPERTS, MODEL_DIM, torch.bfloat16, 64, mask, None, 0)
            ref_sort_64 = aiter_moe_sorting(ref.topk_ids, ref.topk_weights, *other_args)
            with runner._fused_sorting_scope(None):
                out = wrapped(third.topk_ids, third.topk_weights, *other_args)
            _assert_same_sort(self, ref_sort_64, out, 64, num_tokens, num_tokens)
            # padded rows: the fused launch masks them for the sort and in place
            count = torch.tensor([1], dtype=torch.int32, device=self.device)
            fourth = select_experts(
                x,
                torch.empty_like(logits),
                config,
                num_token_non_padded=count,
                router_logits_partials=partials,
            )
            masked_i, masked_w = ref.topk_ids.clone(), ref.topk_weights.clone()
            masked_i[1:] = 0
            masked_w[1:] = 0.0
            ref_sort_pad = aiter_moe_sorting(masked_i, masked_w, *other_args)
            with runner._fused_sorting_scope(count):
                out = wrapped(fourth.topk_ids, fourth.topk_weights, *other_args)
            self.assertTrue(torch.equal(fourth.topk_ids, masked_i))
            self.assertTrue(torch.equal(fourth.topk_weights, masked_w))
            _assert_same_sort(self, ref_sort_pad, out, 64, 1, num_tokens)
        glue.reset_for_tests()


if __name__ == "__main__":
    unittest.main()
