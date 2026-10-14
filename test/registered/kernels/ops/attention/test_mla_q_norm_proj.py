import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: q_lora 1536, kv_lora 512 (+64 rope), 12 heads x 192 q_b outputs; the
# merged a-projection output rows are [q_a 1536 | kv_a 576 | gate 1536].
QA, KV, N, ROW = 1536, 512, 2304, 3648


@unittest.skipIf(torch.version.hip is None, "ROCm MLA query prep")
class TestMlaQNormProj(CustomTestCase):
    @torch.inference_mode()
    def test_matches_norms_then_projection(self):
        """One launch equals RMSNorm of q_a and kv_a (fp32, one bf16 rounding, as
        aiter's fused_qk_rmsnorm) followed by the q_b projection with fp32
        accumulation, within one bf16 rounding; strided inputs."""
        from sglang.kernels.ops.attention.mla_q_norm_proj import (
            covered,
            mla_q_norm_proj,
        )

        g = torch.Generator(device="cuda").manual_seed(0)
        weight = (torch.randn(N, QA, generator=g, device="cuda") * 0.03).bfloat16()
        q_w = (torch.rand(QA, generator=g, device="cuda") + 0.5).bfloat16()
        kv_w = (torch.rand(KV, generator=g, device="cuda") + 0.5).bfloat16()
        eps = 1e-6

        def rmsnorm(x, w):
            x = x.float()
            return (
                x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()
            ).bfloat16()

        for rows in (1, 4, 16):
            a = (torch.randn(rows, ROW, generator=g, device="cuda") * 2).bfloat16()
            q_a, kv_a = a[:, :QA], a[:, QA : QA + KV]
            self.assertTrue(covered(q_a, kv_a, weight))
            q, kv = mla_q_norm_proj(q_a, q_w, eps, kv_a, kv_w, eps, weight)
            want_kv = rmsnorm(kv_a, kv_w)
            want_q = (rmsnorm(q_a, q_w).float() @ weight.float().t()).bfloat16()
            with self.subTest(rows=rows):
                for got, want in ((kv, want_kv), (q, want_q)):
                    want = want.float()
                    beyond_ulp = (got.float() - want).abs() > want.abs() * 2**-7 + 1e-3
                    self.assertEqual(beyond_ulp.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
