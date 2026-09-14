"""Small constant-length extends over a long prefix must match `extend_attention_fwd`."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="jit-kernel-unit-test-amd")

BF16_ATOL, BF16_RTOL = 2e-2, 1e-2
FP8_ATOL, FP8_RTOL = 8e-2, 2e-2

# MiniMax-M3 dense layer at TP4: 16 local query heads on one KV head.
H_Q, H_KV, HEAD_DIM = 16, 1, 128


def _build_inputs(prefix_lens, l_ext, cache_dtype):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(0)
    bs = len(prefix_lens)
    total_prefix = sum(prefix_lens)
    tokens = bs * l_ext

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=gen)

    q = randn(tokens, H_Q, HEAD_DIM)
    k = randn(tokens, H_KV, HEAD_DIM)
    v = randn(tokens, H_KV, HEAD_DIM)
    k_buffer = randn(total_prefix, H_KV, HEAD_DIM).to(cache_dtype)
    v_buffer = randn(total_prefix, H_KV, HEAD_DIM).to(cache_dtype)
    qo_indptr = torch.arange(0, tokens + 1, l_ext, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(prefix_lens, dtype=torch.int32, device=device), dim=0
    )
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestSmallExtendVerifyParity(CustomTestCase):
    """A wrong prefix range or row mapping at extend lengths 1..8 shows as a mismatch."""

    def _check(self, kernel, prefix_lens, l_ext, cache_dtype, atol, rtol, k_scale=1.0):
        q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = _build_inputs(
            prefix_lens, l_ext, cache_dtype
        )
        scale = HEAD_DIM**-0.5
        ref = torch.empty_like(q)
        out = torch.empty_like(q)
        common = (
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            l_ext,
            k_scale,
            1.0,
        )
        extend_attention_fwd(q, k, v, ref, k_buffer, v_buffer, *common, sm_scale=scale)
        ran = kernel(
            q,
            k,
            v,
            out,
            k_buffer,
            v_buffer,
            *common,
            sm_scale=scale,
            max_bs=len(prefix_lens),
        )
        self.assertTrue(ran)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_shared_kv_bf16_lengths_1_to_8(self):
        for l_ext in (1, 3, 8):
            with self.subTest(l_ext=l_ext):
                self._check(
                    verify_shared_kv_fwd,
                    [1, 4096, 20001],
                    l_ext,
                    torch.bfloat16,
                    BF16_ATOL,
                    BF16_RTOL,
                )

    def test_shared_kv_fp8_long_prefix(self):
        self._check(
            verify_shared_kv_fwd,
            [100_000, 777],
            8,
            fp8_dtype,
            FP8_ATOL,
            FP8_RTOL,
            k_scale=2.0,
        )

    def test_splitkv_bf16_and_fp8(self):
        self._check(
            verify_splitkv_fwd, [2048, 65_537], 2, torch.bfloat16, BF16_ATOL, BF16_RTOL
        )
        self._check(
            verify_splitkv_fwd,
            [8, 30_000],
            5,
            fp8_dtype,
            FP8_ATOL,
            FP8_RTOL,
        )


if __name__ == "__main__":
    unittest.main()
