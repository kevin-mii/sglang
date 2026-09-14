"""On the aiter path a MiniMax-M3 row must be the HF top-4 (renormalized x2) plus one shared 1.0."""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.test_utils import CustomTestCase


class TestAiterFusedSharedTopK(CustomTestCase):
    NUM_EXPERTS = 128
    TOPK_ROUTED = 4
    ROUTED_SCALING_FACTOR = 2.0

    def _reference(self, logits, bias):
        # HF: sigmoid scores, top-k on score+bias, renormalize winners, then routed_scaling_factor
        scores = logits.sigmoid()
        ref_ids = torch.topk(scores + bias, self.TOPK_ROUTED, dim=-1).indices
        ref_w = scores.gather(1, ref_ids)
        ref_w = ref_w / ref_w.sum(-1, keepdim=True) * self.ROUTED_SCALING_FACTOR
        return ref_ids, ref_w

    def _run_aiter(self, cfg, logits, hidden):
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", True),
        ):
            return select_experts(
                hidden_states=hidden, router_logits=logits, topk_config=cfg, layer_id=0
            )

    def _check_row_layout(self, out, ref_ids, ref_w):
        ids, weights = out.topk_ids, out.topk_weights
        self.assertEqual(tuple(ids.shape), (ids.shape[0], self.TOPK_ROUTED + 1))
        routed_ids, shared_ids = ids[:, : self.TOPK_ROUTED], ids[:, self.TOPK_ROUTED]
        # Exactly one shared column, at id num_experts, weight 1.0.
        self.assertTrue(bool((shared_ids == self.NUM_EXPERTS).all()))
        self.assertTrue(bool((routed_ids < self.NUM_EXPERTS).all()))
        torch.testing.assert_close(
            weights[:, self.TOPK_ROUTED], torch.ones_like(weights[:, self.TOPK_ROUTED])
        )
        # All K routed experts survive, matching the HF top-k set and weights.
        self.assertTrue(
            torch.equal(
                torch.sort(routed_ids, dim=1).values,
                torch.sort(ref_ids.to(routed_ids.dtype), dim=1).values,
            )
        )
        order = torch.argsort(routed_ids, dim=1)
        ref_order = torch.argsort(ref_ids, dim=1)
        torch.testing.assert_close(
            weights[:, : self.TOPK_ROUTED].gather(1, order),
            ref_w.gather(1, ref_order),
            rtol=1e-4,
            atol=1e-5,
        )

    def _inputs(self):
        torch.manual_seed(0)
        num_tokens = 64
        logits = torch.randn(num_tokens, self.NUM_EXPERTS, device="cuda").float()
        bias = (torch.randn(self.NUM_EXPERTS, device="cuda") * 0.1).float()
        hidden = torch.zeros(num_tokens, 16, device="cuda", dtype=torch.bfloat16)
        return logits, bias, hidden

    def _cfg(self, **overrides):
        kwargs = dict(
            top_k=self.TOPK_ROUTED + 1,
            renormalize=True,
            scoring_func="sigmoid",
            correction_bias=None,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            apply_routed_scaling_factor_on_output=True,
            allow_routed_experts_capture=False,
        )
        kwargs.update(overrides)
        return TopKConfig(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required (Triton gate)")
    def test_row_is_hf_top4_plus_one_shared_slot(self):
        """A double marker drops a routed expert and counts the shared one twice."""
        logits, bias, hidden = self._inputs()
        ref_ids, ref_w = self._reference(logits, bias)
        out = self._run_aiter(self._cfg(correction_bias=bias), logits, hidden)
        self._check_row_layout(out, ref_ids, ref_w)


if __name__ == "__main__":
    unittest.main()
