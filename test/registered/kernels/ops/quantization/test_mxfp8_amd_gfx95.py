"""The gfx950 native MXFP8 GEMV and dense route against fp64 and the bf16-dequant route: within one bf16 ulp, repeatable, batch-invariant."""

import collections
import json
import os
import re
import unittest

import torch

from sglang.kernels.ops.quantization import mxfp8_native_amd_gfx95
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    bf16_dequant_blockscaled_linear,
    dequant_block_fp8_weight_to_bf16,
    fake_quant_fp8_activation,
    fp8_grid_quantize,
)
from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
    mxfp8_gemv,
    mxfp8_native_blockscaled_linear,
    prepare_mxfp8_native_weight,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")


# (N, K): a TP4 projection, the TP4 shared-expert down projection (K = 576, whose tail short of
# a 128-wide step is zero-padded in the weight and masked in the activation), a small odd one.
SHAPES = [
    (1792, 5120),
    (5120, 576),
    (96, 384),
]


def _quant_weight_block32(w: torch.Tensor):
    """fp8 e4m3 weight with one ue8m0 (power of two, fp32) scale per 32x32 block, ceil rule;
    a last row block short of 32 rows gets its own scale, as in a checkpoint."""
    n, k = w.shape
    row_blocks = -(-n // 32)
    padded = torch.nn.functional.pad(w.float(), (0, 0, 0, row_blocks * 32 - n))
    blocks = padded.view(row_blocks, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(row_blocks * 32, k)[:n], torch.exp2(e).view(row_blocks, k // 32)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 scaled-MFMA kernel")
class TestMxfp8GemvGfx95(CustomTestCase):
    def _make(self, n, k, m, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        w = w * torch.exp2(torch.randint(-4, 3, (n, 1), device="cuda").float()).to(
            w.dtype
        )
        wq, ws = _quant_weight_block32(w)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x = x * torch.exp2(torch.randint(-3, 4, (m, 1), device="cuda").float()).to(
            x.dtype
        )
        return wq, ws, x

    def test_matches_fp64_reference_and_both_encodings_agree(self):
        for n, k in SHAPES:
            for m in (1, 17):
                wq, ws, x = self._make(n, k, m)
                w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
                xq, xs = fp8_grid_quantize(x)
                x_fq = fake_quant_fp8_activation(x)
                w_deq = dequant_block_fp8_weight_to_bf16(wq, ws, [32, 32])
                ref = x_fq.double() @ w_deq.double().t()

                out_fp8 = mxfp8_gemv(xq, w_sh, ws8, xs)
                out_bf16 = mxfp8_gemv(x, w_sh, ws8)
                out_grid = mxfp8_gemv(x_fq, w_sh, ws8)
                self.assertTrue(torch.equal(out_fp8, out_bf16), (n, k, m))
                self.assertTrue(torch.equal(out_fp8, out_grid), (n, k, m))
                # fp32 accumulation vs fp64: within bf16 output rounding of the reference.
                err = (out_fp8.double() - ref).abs()
                tol = 2.0**-7 * ref.abs() + 2.0**-7 * ref.abs().max()
                self.assertTrue(
                    bool((err <= tol).all()), (n, k, m, (err / tol).max().item())
                )
                # Most outputs round to the same bf16 as the fp64 reference.
                frac = (out_fp8 != ref.to(torch.bfloat16)).float().mean().item()
                self.assertLess(frac, 0.02, (n, k, m, frac))

    def test_non_finite_activations_encode_alike(self):
        """A NaN or +-inf in a bf16 activation quantizes in the GEMV as fp8_grid_quantize does
        (codes clamped to +-448, NaN to -448), so the input encoding cannot decide whether
        the row turns into NaN."""
        n, k = 96, 384
        wq, ws, _ = self._make(n, k, 1)
        w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
        for bad in (float("nan"), float("inf"), float("-inf")):
            x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
            x[0, 5] = bad
            xq, xs = fp8_grid_quantize(x)
            out_bf16 = mxfp8_gemv(x, w_sh, ws8)
            out_fp8 = mxfp8_gemv(xq, w_sh, ws8, xs)
            self.assertTrue(
                torch.equal(out_bf16.view(torch.int16), out_fp8.view(torch.int16)), bad
            )


def _wide_range(*shape: int) -> torch.Tensor:
    """Values spanning 2^-12 .. 2^8, so fp32 partial sums round and their order shows."""
    t = torch.randn(*shape, device="cuda")
    return t * torch.exp2(torch.randint(-12, 8, shape, device="cuda").float())


ROUTE_SHAPES = [(1792, 5120), (5120, 576)]
# one M per kernel: the gemv, the dot_scaled tile of the 64-row bucket, that of the 4096-row bucket
MS = (1, 33, 1025)


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 native MXFP8 route")
class TestMxfp8NativeRouteGfx95(CustomTestCase):
    def _weights(self, n, k, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        wq, ws = _quant_weight_block32(w)
        w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
        w_bf16 = dequant_block_fp8_weight_to_bf16(wq, ws, [32, 32])
        return w_sh, ws8, w_bf16

    def test_within_one_bf16_ulp_of_the_bf16_route(self):
        for n, k in ROUTE_SHAPES:
            w_sh, ws8, w_bf16 = self._weights(n, k)
            for m in MS:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                # groups whose amax is below the 1e-10 floor, where a quantizer off the CUDA rule diverges
                x[0] *= 1e-13
                ref = bf16_dequant_blockscaled_linear(x, w_bf16)
                out = mxfp8_native_blockscaled_linear(x, w_sh, ws8)
                self.assertEqual(out.shape, ref.shape)
                # same products, different fp32 summation order: within one bf16 ulp of the row's largest output
                row_max = ref.float().abs().amax(dim=1, keepdim=True).clamp(min=1.0)
                ulp_of_row_max = torch.exp2(torch.floor(torch.log2(row_max)) - 7)
                diff = (out.float() - ref.float()).abs()
                self.assertTrue(
                    bool((diff <= ulp_of_row_max).all()),
                    (n, k, m, (diff / ulp_of_row_max).max().item()),
                )
                self.assertLess(
                    _bf16_ulp_diff(out, ref).gt(1).float().mean().item(),
                    5e-3,
                    (n, k, m),
                )
                # the fp8-grid input and the fp8 + scales input must give the same result as the plain bf16 input
                out_grid = mxfp8_native_blockscaled_linear(
                    fake_quant_fp8_activation(x), w_sh, ws8
                )
                self.assertTrue(torch.equal(out_grid, out), (n, k, m))
                xq, xs = fp8_grid_quantize(x)
                out_q = mxfp8_native_blockscaled_linear(xq, w_sh, ws8, input_scale=xs)
                self.assertTrue(torch.equal(out_q, out), (n, k, m))

    def test_repeatable_and_batch_invariant_inside_each_kernel(self):
        """Rows that share a kernel (the gemv up to 32 tokens; the dot_scaled GEMM in
        the 4096-row bucket) sum in the same order at every batch size, so a prefix of
        a batch is bitwise the batch's prefix. Wide-range data, so a changed order shows."""
        torch.manual_seed(1)
        for n, k in ((1792, 5120), (5120, 2048)):
            wq, ws = _quant_weight_block32(_wide_range(n, k))
            w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
            # prefixes that stay in the batch's kernel: the gemv buckets 1 .. 32, and the
            # dot_scaled 4096-row bucket
            for m_hi, prefixes in ((32, (1, 4, 16)), (1100, (1025, 1062))):
                x = _wide_range(m_hi, k).to(torch.bfloat16)
                full = mxfp8_native_blockscaled_linear(x, w_sh, ws8)
                self.assertTrue(
                    torch.equal(mxfp8_native_blockscaled_linear(x, w_sh, ws8), full)
                )
                for m in prefixes:
                    part = mxfp8_native_blockscaled_linear(
                        x[:m].contiguous(), w_sh, ws8
                    )
                    self.assertTrue(torch.equal(part, full[:m]), (n, k, m_hi, m))

    def test_gemv_rows_of_a_shape_share_the_wave_count(self):
        """The waves split K, so their count fixes a row's fp32 sum order: a table that
        changes it between M buckets makes a row's bits depend on the batch size."""
        path = os.path.join(
            os.path.dirname(mxfp8_native_amd_gfx95.__file__),
            "mxfp8_gemv_gfx95_configs.json",
        )
        with open(path) as f:
            configs = json.load(f)["configs"]
        waves = collections.defaultdict(set)
        for key, config in configs.items():
            shape = key.rsplit(":", 1)[0]
            waves[shape].add(int(re.fullmatch(r"w(\d+)s\d+r\d+t\d+k", config)[1]))
        mixed = {shape: sorted(w) for shape, w in waves.items() if len(w) > 1}
        self.assertEqual(mixed, {})


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 native MXFP8 route")
class TestFp8LinearGfx95Routes(CustomTestCase):
    """Fp8LinearMethod on a 32-block ue8m0 checkpoint: the plain bf16 input, the fused
    producers' fp8-grid wrapper and their MXFP8 wrapper must give the same rows, on a
    shape the native kernels tile and on one that keeps the bf16-dequant route."""

    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _linear(self, n, k):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
        layer = make_tp1_column_parallel_linear(
            quant_config, n, k, skip_block_quant_check=True
        )
        torch.manual_seed(n + k)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 10
        wq, ws = _quant_weight_block32(w)
        load_linear_weights(layer, weight=wq, weight_scale_inv=ws)
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_wrapped_inputs_match_the_plain_input(self):
        # K = 5152 takes the native route through its K tail; N = 1872 ends in half a
        # 32-row scale block, which only the bf16-dequant route serves
        for n, k, native in (
            (1856, 5120, True),
            (1856, 5152, True),
            (1872, 5120, False),
        ):
            layer = self._linear(n, k)
            self.assertEqual(layer.mxfp8_native_ready, native, (n, k))
            for m in (1, 33, 1025):
                with self.subTest(n=n, m=m):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    x_grid = fake_quant_fp8_activation(x)
                    plain = layer(x)[0]
                    on_grid = layer(Fp8GridActivation(x_grid))[0]
                    quantized = layer(Mxfp8Activation(*fp8_grid_quantize(x)))[0]
                    self.assertTrue(torch.equal(on_grid, plain))
                    self.assertTrue(torch.equal(quantized, plain))


if __name__ == "__main__":
    unittest.main()
