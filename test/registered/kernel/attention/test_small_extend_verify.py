"""Small constant-length EXTENDs over a long cached prefix on the verify kernels.

A new agent turn (or a request restarted from the prefix cache) is an EXTEND of
a few tokens over a very long cached prefix. TritonAttnBackend routes such
batches (constant per-request extend length <= SMALL_EXTEND_MAX_TOKENS, all
requests with a cached prefix) to the split-KV / grouped-head verify kernels,
whose shape they share. This checks (a) the routing predicate and (b) kernel
parity against extend_attention_fwd for extend lengths 1..8 over long prefixes.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.verify_mla import verify_shared_kv_fwd
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="jit-kernel-unit-test-amd")

BF16_ATOL, BF16_RTOL = 2e-2, 1e-2
FP8_ATOL, FP8_RTOL = 8e-2, 2e-2

# MiniMax-M3 dense layer at TP4: 16 local query heads on one KV head.
H_Q, H_KV, HEAD_DIM = 16, 1, 128


def _forward_batch(mode, extend_seq_lens_cpu, extend_prefix_lens_cpu="auto"):
    if extend_prefix_lens_cpu == "auto":
        extend_prefix_lens_cpu = (
            [4096] * len(extend_seq_lens_cpu) if extend_seq_lens_cpu else []
        )
    return SimpleNamespace(
        forward_mode=mode,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
    )


_FAKE_BACKEND = SimpleNamespace(
    SMALL_EXTEND_MAX_TOKENS=TritonAttnBackend.SMALL_EXTEND_MAX_TOKENS
)


class TestSmallExtendPredicate(CustomTestCase):
    """_is_small_constant_extend is host-side only; SMALL_EXTEND_MAX_TOKENS is
    all it reads from the backend, so a namespace stands in for self."""

    def _pred(self, mode, ext, kv_numel=1000, prefix="auto"):
        kv = torch.empty(kv_numel, dtype=torch.int64) if kv_numel else None
        return TritonAttnBackend._is_small_constant_extend(
            _FAKE_BACKEND, _forward_batch(mode, ext, prefix), kv
        )

    def test_constant_small_extend_routes(self):
        for n in (1, 3, TritonAttnBackend.SMALL_EXTEND_MAX_TOKENS):
            self.assertTrue(self._pred(ForwardMode.EXTEND, [n, n, n]))

    def test_ragged_or_large_extend_does_not_route(self):
        self.assertFalse(self._pred(ForwardMode.EXTEND, [2, 3]))
        big = TritonAttnBackend.SMALL_EXTEND_MAX_TOKENS + 1
        self.assertFalse(self._pred(ForwardMode.EXTEND, [big, big]))

    def test_other_modes_and_empty_prefix_do_not_route(self):
        self.assertFalse(self._pred(ForwardMode.DECODE, [1, 1]))
        self.assertFalse(self._pred(ForwardMode.TARGET_VERIFY, [4, 4]))
        self.assertFalse(self._pred(ForwardMode.EXTEND, [4, 4], kv_numel=0))
        self.assertFalse(self._pred(ForwardMode.EXTEND, []))
        self.assertFalse(self._pred(ForwardMode.EXTEND, None))

    def test_zero_prefix_request_does_not_route(self):
        # Every request needs a cached prefix: a fresh request mixed into the
        # batch, an all-fresh batch, or unknown prefix lengths fall through.
        self.assertTrue(self._pred(ForwardMode.EXTEND, [4, 4], prefix=[1, 200_000]))
        self.assertFalse(self._pred(ForwardMode.EXTEND, [4, 4], prefix=[0, 4096]))
        self.assertFalse(self._pred(ForwardMode.EXTEND, [4, 4], prefix=[0, 0]))
        self.assertFalse(self._pred(ForwardMode.EXTEND, [4, 4], prefix=None))
        self.assertFalse(self._pred(ForwardMode.EXTEND, [4, 4], prefix=[4096]))


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
    def _check(self, kernel, prefix_lens, l_ext, cache_dtype, atol, rtol, k_scale=1.0):
        q, k, v, kb, vb, qo_indptr, kv_indptr, kv_indices = _build_inputs(
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
        extend_attention_fwd(q, k, v, ref, kb, vb, *common, sm_scale=scale)
        ran = kernel(
            q, k, v, out, kb, vb, *common, sm_scale=scale, max_bs=len(prefix_lens)
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
