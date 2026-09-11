"""Pin the e2m1 tie rounding of every HIP FP4 indexer quantizer.

The low-ratio Triton paths round ties to even like CUDA; both AITER ratio-4 kernels quantize
through gfx950 `v_cvt_scalef32_pk_fp4_f32`, also ties-to-even, where CUDA's ratio-4 kernels
send the odd ties 0.75 / 1.75 / 3.5 toward zero. Exact ties (power-of-two block scales) are
fed to each path and the convention asserted.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
# What each convention returns for TIES (and their negatives, by symmetry).
RNE = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
CUDA_TOWARD_ZERO = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SCALES = (2.0**-3, 1.0, 2.0**4)


def _tie_row(scale: float) -> torch.Tensor:
    """[128]: four 32-blocks of [6, +ties, -ties, 0...] times a power-of-two scale."""
    block = [6.0] + TIES + [-t for t in TIES]
    block += [0.0] * (32 - len(block))
    return (torch.tensor(block) * scale).repeat(4)


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 nibbles [..., 64] (low nibble first) -> values [..., 128]."""
    p = packed.view(torch.uint8).to(torch.int64).cpu()
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).flatten(-2)
    mag = E2M1[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def _expected(scale: float, convention) -> torch.Tensor:
    block = [6.0] + convention + [-v for v in convention]
    block += [0.0] * (32 - len(block))
    return (torch.tensor(block) * scale).repeat(4)


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "ROCm only")
class TestDsv41Fp4TieRoundingHip(CustomTestCase):
    def _check(self, name, scale, got, convention):
        self.assertTrue(
            torch.equal(got.float().cpu(), _expected(scale, convention)),
            f"{name} at scale {scale}: {got[1:8].tolist()} vs {convention}",
        )

    def _e8m0(self, scale: float) -> int:
        return 127 + int(torch.log2(torch.tensor(scale)))

    def test_low_ratio_triton_paths_round_half_to_even_like_cuda(self):
        """Same Triton quantizer as CUDA (`rne=True`): ties to even on both."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            pack_fp4_query_flydsl,
            read_fp4_index_k_split,
            store_fp4_index_k_cache_split,
        )
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )
        from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
            rope_fake_quant_pack_indexer,
        )

        for scale in SCALES:
            row = _tie_row(scale).cuda().to(torch.bfloat16)
            self.assertTrue(torch.equal(row.float().cpu(), _tie_row(scale)))

            fp4, sf = quantize_fp4_indexer_tensor(row.view(1, 128), rne=True)
            self.assertEqual((sf.cpu() & 0xFF).item(), self._e8m0(scale))
            self._check(
                "quantize_fp4_indexer_tensor(rne=True)",
                scale,
                _unpack(fp4)[0] * scale,
                RNE,
            )
            # The threshold variant is CUDA's ratio-4 convention.
            fp4, _ = quantize_fp4_indexer_tensor(row.view(1, 128), rne=False)
            self._check(
                "quantize_fp4_indexer_tensor(rne=False)",
                scale,
                _unpack(fp4)[0] * scale,
                CUDA_TOWARD_ZERO,
            )

            q_fp4, q_scale = pack_fp4_query_flydsl(
                row.view(1, 1, 128).expand(1, 16, 128).contiguous()
            )
            self.assertEqual(q_scale.unique().tolist(), [0, self._e8m0(scale)])
            self._check(
                "pack_fp4_query_flydsl", scale, _unpack(q_fp4)[0, 0] * scale, RNE
            )

            payload = torch.zeros((1, 1, 4, 64, 16), dtype=torch.uint8, device="cuda")
            k_scale = torch.zeros((1, 1, 4, 64), dtype=torch.uint8, device="cuda")
            loc = torch.tensor([5], dtype=torch.int64, device="cuda")
            store_fp4_index_k_cache_split(
                row.view(1, 128), payload, k_scale, loc, page_size=64, rne=True
            )
            k_fp4, k_sf = read_fp4_index_k_split(payload, k_scale, loc, page_size=64)
            self.assertEqual((k_sf.cpu() & 0xFF).item(), self._e8m0(scale))
            self._check(
                "store_fp4_index_k_cache_split", scale, _unpack(k_fp4)[0] * scale, RNE
            )

            freqs = torch.ones(1, 32, dtype=torch.complex64, device="cuda")
            packed, _ = rope_fake_quant_pack_indexer(row.view(1, 1, 128), freqs, 64)
            self._check(
                "rope_fake_quant_pack_indexer", scale, _unpack(packed)[0] * scale, RNE
            )
            fq = rope_tail_fake_quant_fp4(row.view(1, 128), freqs, 64)
            self._check("rope_tail_fake_quant_fp4", scale, fq[0], RNE)

    def test_aiter_ratio4_kernels_round_half_to_even(self):
        """The AITER ratio-4 quantizers must round ties to even, unlike the CUDA ratio-4
        kernels' odd ties toward zero."""
        import aiter

        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            read_fp4_index_k_split,
        )

        cos = torch.ones(1, 32, dtype=torch.bfloat16, device="cuda")
        sin = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
        pos = torch.zeros(1, dtype=torch.int64, device="cuda")
        for scale in SCALES:
            row = _tie_row(scale).cuda().to(torch.bfloat16)

            # Query kernel with the Hadamard rotate off so the ties reach the cvt.
            q = row.view(1, 1, 128).expand(1, 64, 128).contiguous()
            q_fp4 = torch.empty((1, 64, 64), dtype=aiter.dtypes.fp4x2, device="cuda")
            q_scale = torch.empty((1, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
            aiter.rope_rotate_activation(
                q_fp4,
                q,
                cos,
                sin,
                pos,
                rope_dim=64,
                out_scale=q_scale,
                group_size=32,
                shuffle_scale=True,
                do_rotate_act=False,
            )
            self.assertEqual(q_scale.unique().tolist(), [self._e8m0(scale)])
            self._check(
                "aiter.rope_rotate_activation", scale, _unpack(q_fp4)[0, 0] * scale, RNE
            )

            # K kernel: x = +-1 makes the RMSNorm exact; the weight carries the ties.
            sign = torch.where(torch.arange(128, device="cuda") % 2 == 1, -1.0, 1.0)
            x = sign.to(torch.bfloat16).view(1, 1, 128)
            w = (row.float() * sign).to(torch.bfloat16)
            kv = torch.zeros((1, 1, 4, 64, 16), dtype=torch.uint8, device="cuda").view(
                torch.float4_e2m1fn_x2
            )
            k_scale = torch.zeros((1, 1, 4, 64), dtype=torch.uint8, device="cuda")
            slots = torch.tensor([9], dtype=torch.int64, device="cuda")
            aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
                kv,
                k_scale,
                x,
                w,
                cos,
                sin,
                pos,
                slots,
                0.0,
                rope_dim=64,
                kv_block_size=64,
                group_size=32,
                shuffle_scale=True,
                do_rotate_act=False,
            )
            k_fp4, k_sf = read_fp4_index_k_split(kv, k_scale, slots, page_size=64)
            self.assertEqual((k_sf.cpu() & 0xFF).item(), self._e8m0(scale))
            self._check(
                "aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache",
                scale,
                _unpack(k_fp4)[0] * scale,
                RNE,
            )

    def test_aiter_query_path_keeps_fp32_until_the_cvt(self):
        """The served ratio-4 query path must have no bf16 intermediate: an fp32 ties-
        to-even emulation reproduces it bit for bit."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_q_indexer_fp4,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import round_fp4

        h = torch.tensor([[1.0]])
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = h.cuda()
        rsqrt128 = torch.tensor(0.08838834764831845, dtype=torch.float32, device="cuda")

        n = 512
        x = torch.randint(
            -6,
            7,
            (n, 64, 128),
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(0),
        ).to(torch.bfloat16)
        cos = torch.ones(1, 32, dtype=torch.bfloat16, device="cuda")
        sin = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
        pos = torch.zeros(n, dtype=torch.int64, device="cuda")
        q_fp4, q_scale = aiter_q_indexer_fp4(x, cos, sin, pos)
        exps = q_scale[:, 0].permute(0, 3, 2, 1).reshape(n, 64, 4).to(torch.int32) - 127
        got = _unpack(q_fp4).cuda() * exps.float().exp2().repeat_interleave(32, -1)

        af = (x.float() @ h) * rsqrt128
        blocks = af.unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
        bits = (amax / 6.0).view(torch.int32)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = (e << 23).view(torch.float32)
        expected = (round_fp4((blocks / scale).clamp(-6.0, 6.0)) * scale).flatten(-2)
        self.assertTrue(torch.equal(exps, (e - 127).squeeze(-1)))
        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
