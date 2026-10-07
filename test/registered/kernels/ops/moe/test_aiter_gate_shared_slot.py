"""On aiter, the sigmoid gate's own shared-expert column must equal the appended one.

MiniMax-M3 routing (sigmoid + correction bias, top-4 of 128 routed experts, one
fused shared expert, routed_scaling_factor 2.0) lets moe_fused_gate write the
shared column instead of running fused_append_shared_experts afterwards. The
row must stay bit-identical to the append path, shared id and weight included.
"""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe import fused_moe_triton_kernels
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

NUM_ROUTED = 128
TOPK_ROUTED = 4
ROUTED_SCALING_FACTOR = 2.0
# Decode batches up to the cuda-graph cap, plus EAGLE verify rows (bs x draft tokens).
NUM_TOKENS = (1, 2, 3, 4, 7, 8, 16, 24, 32, 48, 63, 64, 96, 127, 128)


@unittest.skipUnless(is_hip(), "aiter path is ROCm only")
class TestAiterGateSharedSlot(CustomTestCase):
    def _cfg(self, bias):
        return TopKConfig(
            top_k=TOPK_ROUTED + 1,
            renormalize=True,
            scoring_func="sigmoid",
            correction_bias=bias,
            num_fused_shared_experts=1,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=True,
            allow_routed_experts_capture=False,
        )

    def _select(self, cfg, logits, *, gate_writes_shared):
        hidden = torch.zeros(logits.shape[0], 16, device="cuda", dtype=torch.bfloat16)
        append = fused_moe_triton_kernels.fused_append_shared_experts
        calls = []
        # Forcing the predicate off reproduces the gate + append path it replaces.
        gate_patch = (
            nullcontext()
            if gate_writes_shared
            else patch.object(
                topk_module,
                "_aiter_sigmoid_gate_writes_shared_slot",
                return_value=False,
            )
        )

        def counting_append(*args, **kwargs):
            calls.append(1)
            return append(*args, **kwargs)

        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", True),
            patch.object(
                fused_moe_triton_kernels,
                "fused_append_shared_experts",
                counting_append,
            ),
            gate_patch,
        ):
            out = select_experts(
                hidden_states=hidden, router_logits=logits, topk_config=cfg, layer_id=0
            )
        return out.topk_ids, out.topk_weights, len(calls)

    def test_gate_shared_column_matches_append(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        bias = torch.randn(NUM_ROUTED, device="cuda", generator=gen) * 0.1
        cfg = self._cfg(bias)
        for dtype in (torch.bfloat16, torch.float32):
            for m in NUM_TOKENS:
                with self.subTest(dtype=dtype, num_tokens=m):
                    logits = (
                        torch.randn(m, NUM_ROUTED, device="cuda", generator=gen) * 3
                    ).to(dtype)
                    ref_ids, ref_w, ref_calls = self._select(
                        cfg, logits, gate_writes_shared=False
                    )
                    ids, w, calls = self._select(cfg, logits, gate_writes_shared=True)
                    self.assertEqual(ref_calls, 1)
                    self.assertEqual(calls, 0, "append kernel still launched")
                    self.assertEqual(ids.shape, (m, TOPK_ROUTED + 1))
                    self.assertEqual(ids.dtype, ref_ids.dtype)
                    self.assertEqual(w.dtype, ref_w.dtype)
                    self.assertTrue(torch.equal(ids, ref_ids))
                    self.assertTrue(torch.equal(w, ref_w))
                    self.assertTrue(bool((ids[:, -1] == NUM_ROUTED).all()))
                    self.assertTrue(bool((w[:, -1] == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
