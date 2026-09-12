"""Regression test: on the aiter (HIP) path the fused shared expert must be
appended exactly once, and the gate must still return all K routed experts.

MiniMax-M3 (sigmoid + correction bias, top-4 routed + 1 fused shared expert,
routed_scaling_factor 2.0) routes through ``biased_topk_jit_kernel_impl`` ->
``moe_fused_gate``. That gate treats ``topk`` as the total width including the
shared slot, while ``_post_process_topk_ids`` appends the shared expert itself
whenever ``_use_aiter`` is set. Passing ``num_fused_shared_experts`` to the gate
as well produced rows of [3 routed, shared, shared]: one routed expert lost and
the shared expert counted twice (GSM8K-500 0.81 fused vs 0.88 unfused on
MI350X). The gate must be asked for K_routed plain routed experts on every aiter
path, exactly as PR #36515 already did for the per-rank shared-slot path.
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

# On ROCm import topk with aiter enabled so the aiter grouped kernels are
# bound; on CUDA the JIT-gate cases run by patching the platform flags.
if is_hip() and not envs.SGLANG_USE_AITER.is_set():
    envs.SGLANG_USE_AITER.set(True)

import sglang.kernels.ops.moe.fused_moe_triton_kernels as fused_kernels
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.test_utils import CustomTestCase


class TestAiterFusedSharedTopK(CustomTestCase):
    NUM_EXPERTS = 128
    TOPK_ROUTED = 4
    ROUTED_SCALING_FACTOR = 2.0

    def _reference(self, logits, bias):
        # HF MiniMax-M3 semantics: sigmoid scores, top-k on score + bias,
        # renormalize the raw scores of the winners, then routed_scaling_factor.
        scores = logits.sigmoid()
        ref_ids = torch.topk(scores + bias, self.TOPK_ROUTED, dim=-1).indices
        ref_w = scores.gather(1, ref_ids)
        ref_w = ref_w / ref_w.sum(-1, keepdim=True) * self.ROUTED_SCALING_FACTOR
        return ref_ids, ref_w

    def _run_aiter(self, cfg, logits, hidden):
        """select_experts on the aiter path; returns (out, append_kernel_calls)."""
        calls = []
        real_append = fused_kernels.fused_append_shared_experts

        def spy(*args, **kwargs):
            calls.append(1)
            return real_append(*args, **kwargs)

        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", True),
            patch.object(fused_kernels, "fused_append_shared_experts", spy),
        ):
            out = select_experts(
                hidden_states=hidden, router_logits=logits, topk_config=cfg, layer_id=0
            )
        return out, len(calls)

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
    def test_aiter_jit_gate_folds_shared_slot(self):
        """MiniMax-M3 shape: the JIT gate writes the shared slot itself, the
        append kernel is skipped, and the row is the HF top-4 + shared 1.0."""
        logits, bias, hidden = self._inputs()
        ref_ids, ref_w = self._reference(logits, bias)
        out, appends = self._run_aiter(self._cfg(correction_bias=bias), logits, hidden)
        self._check_row_layout(out, ref_ids, ref_w)
        self.assertEqual(appends, 0, "fold engaged: no separate append launch")

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip(), "aiter grouped top-k kernel (ROCm)"
    )
    def test_aiter_grouped_path_must_not_fold(self):
        """Grouped top-k is not served by the JIT gate: the fold must stay off
        and the append kernel must still produce exactly one shared column."""
        logits, bias, hidden = self._inputs()
        ref_ids, ref_w = self._reference(logits, bias)
        # One group covering all experts keeps the routed set identical to the
        # ungrouped reference. The aiter grouped kernel always folds
        # routed_scaling_factor into its weights (it rejects
        # apply_routed_scaling_factor_on_output=True), so the reference is
        # the same scaled row as the ungrouped case.
        cfg = self._cfg(
            correction_bias=bias,
            num_expert_group=1,
            topk_group=1,
            use_grouped_topk=True,
            apply_routed_scaling_factor_on_output=False,
        )
        out, appends = self._run_aiter(cfg, logits, hidden)
        self._check_row_layout(out, ref_ids, ref_w)
        self.assertEqual(appends, 1, "grouped path keeps the append launch")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required (Triton gate)")
    def test_aiter_non_unit_shared_scale_must_not_fold(self):
        """A shared-expert scaling factor other than 1.0 cannot come from the
        gate; the append kernel must apply it."""
        logits, bias, hidden = self._inputs()
        cfg = self._cfg(correction_bias=bias, fused_shared_experts_scaling_factor=0.5)
        out, appends = self._run_aiter(cfg, logits, hidden)
        self.assertEqual(appends, 1)
        torch.testing.assert_close(
            out.topk_weights[:, self.TOPK_ROUTED],
            torch.full_like(out.topk_weights[:, self.TOPK_ROUTED], 0.5),
        )
        self.assertTrue(
            bool((out.topk_ids[:, self.TOPK_ROUTED] == self.NUM_EXPERTS).all())
        )


if __name__ == "__main__":
    unittest.main()
