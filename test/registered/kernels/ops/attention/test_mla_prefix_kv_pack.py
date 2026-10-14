import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.mla_prefix_kv_pack import (
    gather_prefix_latent,
    pack_prefix_kv_fp8,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: 12 heads padded to the FP8 prefill kernel's 16.
H, HP, DN, DV, R, P = 12, 16, 128, 128, 512, 64


def _pad_heads(x, padded):
    reps = -(-padded // x.shape[1])
    return x.repeat(1, reps, 1)[:, :padded, :].contiguous()


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMLAPrefixKVPack(unittest.TestCase):
    @torch.inference_mode()
    def test_matches_unfused_expansion_bitwise(self):
        """The two-launch expansion must produce the exact fp8 K/V bytes the
        aiter backend's index_select / cast / cat / pad_heads / cast chain does:
        padding heads tiled as h % H, the rope part carried as cached bytes."""
        fp8 = torch.float8_e4m3fn
        torch.manual_seed(0)
        slots = 4096
        cache = (torch.randn(slots, 1, R + P, device="cuda") * 2).to(fp8)
        w = (torch.randn(H * (DN + DV), R, device="cuda") * 0.05).to(torch.bfloat16)
        for num_rows in (1, 37, 3000):
            idx = torch.randperm(slots, device="cuda")[:num_rows].to(torch.int32)
            kb = torch.index_select(cache, 0, idx)
            kvc_ref, k_pe = torch.split(kb, [R, P], dim=-1)
            kvc_ref = kvc_ref.to(torch.bfloat16).squeeze(1)
            kv_ref = F.linear(kvc_ref, w).view(-1, H, DN + DV)
            k_ref, v_ref = torch.split(kv_ref, [DN, DV], dim=-1)
            k_ref = torch.cat(
                [k_ref, torch.broadcast_to(k_pe.to(torch.bfloat16), (num_rows, H, P))],
                dim=-1,
            )
            k_ref = _pad_heads(k_ref, HP).to(fp8)
            v_ref = _pad_heads(v_ref, HP).to(fp8)

            kvc, pe = gather_prefix_latent(cache, idx, R)
            k, v = pack_prefix_kv_fp8(
                F.linear(kvc, w),
                pe,
                num_heads=H,
                num_heads_padded=HP,
                nope_dim=DN,
                v_dim=DV,
            )
            with self.subTest(num_rows=num_rows):
                self.assertTrue(torch.equal(kvc, kvc_ref))
                self.assertTrue(
                    torch.equal(k.view(torch.uint8), k_ref.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(v.view(torch.uint8), v_ref.view(torch.uint8))
                )


if __name__ == "__main__":
    unittest.main()
