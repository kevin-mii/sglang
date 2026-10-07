"""The gfx950 router GEMV matches fp32 `torch.mm` on every M bucket."""

import unittest

import torch

from sglang.kernels.ops.gemm import router_gemv as rg
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

# MiniMax-M3 router: 128 experts, hidden 6144.
NUM_EXPERTS, HIDDEN = 128, 6144


@unittest.skipUnless(rg._is_gfx95_supported, "ROCm gfx95 only")
class TestRouterGemv(CustomTestCase):
    def test_all_row_buckets_match_fp32_reference(self):
        torch.manual_seed(0)
        w = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
        # 65..128 rows are EAGLE target-verify shapes (bs x draft tokens).
        for m in (1, 4, 8, 9, 16, 17, 32, 33, 64, 65, 96, 127, 128):
            x = torch.randn(m, HIDDEN, device="cuda").to(torch.bfloat16)
            self.assertTrue(rg.router_gemv_supported(x, w), msg=f"m={m}")
            ref = torch.mm(x.float(), w.float().t())
            out = rg.router_gemv(x, w)
            self.assertEqual(out.dtype, torch.float32)
            # split-K sums in a different order than torch.mm
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3, msg=f"m={m}")
            # the second call reuses the self-cleaning split-K counter
            torch.testing.assert_close(rg.router_gemv(x, w), ref, rtol=1e-3, atol=1e-3)
        x = torch.randn(rg._MAX_M + 1, HIDDEN, device="cuda").to(torch.bfloat16)
        self.assertFalse(rg.router_gemv_supported(x, w))


if __name__ == "__main__":
    unittest.main()
