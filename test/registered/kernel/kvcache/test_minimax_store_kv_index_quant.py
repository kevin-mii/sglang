"""`store_kv_index_quant` must write exactly what the unfused stores write."""

import pytest
import torch

from sglang.kernels.ops.kvcache.minimax_store_kv_index_quant import (
    can_store_kv_index_quant,
    store_kv_index_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

dev = "cuda"
SLOTS = 4096
UNIT_SCALES = (None, None, None, None)
SCALES = (0.5, 2.0, 0.25, 4.0)


def _reference_store(x, cache, loc, scale):
    # as in set_kv_buffer, the scale applies only where the store casts
    y = x
    if x.dtype != cache.dtype and scale is not None:
        y = x / scale
    cache[loc] = y.to(cache.dtype)


def _assert_close(fused, ref, cache_dtype, scaled, name):
    diff = (fused.float() - ref.float()).abs()
    if cache_dtype.itemsize == 1 and scaled:
        # fp32 vs bf16 division before the fp8 cast lands a few elements one ulp apart
        mismatched = (diff > 1e-6).sum().item()
        assert mismatched <= int(0.02 * diff.numel()), (name, mismatched)
    else:
        assert diff.max().item() == 0, (name, diff.max().item())


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize(
    "T,H,D,Di",
    [(1, 1, 128, 128), (32, 1, 128, 128), (513, 1, 128, 128), (7, 2, 128, 64)],
)
@pytest.mark.parametrize("has_v", [False, True])
@pytest.mark.parametrize("scales", [UNIT_SCALES, SCALES])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_matches_unfused_stores(cache_dtype, T, H, D, Di, has_v, scales, idx_dtype):
    torch.manual_seed(T * 31 + H * 7 + Di)
    # views of one wide row buffer, like the qkv/index projection splits in the model
    row = torch.randn(T, 3 * H * D + 2 * Di, dtype=torch.bfloat16, device=dev) * 20
    k = row[:, : H * D].view(T, H, D)
    v = row[:, H * D : 2 * H * D].view(T, H, D)
    idx_k = row[:, 3 * H * D : 3 * H * D + Di].view(T, 1, Di)
    idx_v = row[:, 3 * H * D + Di :].view(T, 1, Di) if has_v else None

    loc = torch.randperm(SLOTS, device=dev)[:T].to(idx_dtype)
    k_cache = torch.zeros(SLOTS, H, D, dtype=cache_dtype, device=dev)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(SLOTS, 1, Di, dtype=cache_dtype, device=dev)
    idx_v_cache = torch.zeros_like(idx_k_cache) if has_v else None
    unfused = [c.clone() for c in (k_cache, v_cache, idx_k_cache)]
    unfused_idx_v = idx_v_cache.clone() if has_v else None

    assert can_store_kv_index_quant(
        k, v, k_cache, v_cache, idx_k, idx_k_cache, idx_v, idx_v_cache
    )
    store_kv_index_quant(
        k, v, k_cache, v_cache, idx_k, idx_k_cache, idx_v, idx_v_cache, loc, *scales
    )
    _reference_store(k, unfused[0], loc, scales[0])
    _reference_store(v, unfused[1], loc, scales[1])
    _reference_store(idx_k, unfused[2], loc, scales[2])
    if has_v:
        _reference_store(idx_v, unfused_idx_v, loc, scales[3])
    torch.cuda.synchronize()

    scaled = scales[0] is not None
    _assert_close(k_cache, unfused[0], cache_dtype, scaled, "k")
    _assert_close(v_cache, unfused[1], cache_dtype, scaled, "v")
    _assert_close(idx_k_cache, unfused[2], cache_dtype, scaled, "idx_k")
    if has_v:
        _assert_close(idx_v_cache, unfused_idx_v, cache_dtype, scaled, "idx_v")
    # unaddressed rows stay untouched
    mask = torch.ones(SLOTS, dtype=torch.bool, device=dev)
    mask[loc.long()] = False
    assert k_cache[mask].float().abs().max().item() == 0


def test_mixed_cache_dtypes_scale_only_the_cast_side():
    """bf16 main cache with an fp8 index cache: the K/V scale must not be
    applied (no cast), the index scale must."""
    T, H, D, Di = 16, 1, 128, 128
    torch.manual_seed(0)
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    idx_k = torch.randn(T, 1, Di, dtype=torch.bfloat16, device=dev)
    loc = torch.arange(T, device=dev, dtype=torch.int64)
    k_cache = torch.zeros(SLOTS, H, D, dtype=torch.bfloat16, device=dev)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(SLOTS, 1, Di, dtype=torch.float8_e4m3fn, device=dev)

    store_kv_index_quant(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        None,
        None,
        loc,
        0.5,
        0.5,
        0.25,
        None,
    )
    torch.cuda.synchronize()
    assert torch.equal(k_cache[loc], k)
    assert torch.equal(v_cache[loc], v)
    ref = torch.zeros_like(idx_k_cache)
    ref[loc] = (idx_k / 0.25).to(torch.float8_e4m3fn)
    diff = (idx_k_cache.float() - ref.float()).abs()
    assert (diff > 1e-6).sum().item() <= int(0.02 * diff.numel())


def test_rejects_unsupported_layouts():
    T, H, D = 4, 1, 128
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    idx_k = torch.randn(T, 1, D, dtype=torch.bfloat16, device=dev)
    fp8 = torch.float8_e4m3fn
    ok_cache = torch.zeros(SLOTS, H, D, dtype=fp8, device=dev)

    def ok(
        k_=k, v_=v, kc=ok_cache, vc=ok_cache, ik=idx_k, ikc=ok_cache, iv=None, ivc=None
    ):
        return can_store_kv_index_quant(k_, v_, kc, vc, ik, ikc, iv, ivc)

    assert ok()
    # Non-power-of-2 head dim, on K or on V.
    odd_cache = torch.zeros(SLOTS, H, 96, dtype=fp8, device=dev)
    odd = torch.randn(T, H, 96, dtype=torch.bfloat16, device=dev)
    assert not ok(k_=odd, kc=odd_cache)
    assert not ok(v_=odd, vc=odd_cache)
    # fp32 input.
    assert not ok(k_=k.float())
    assert not ok(v_=v.float())
    # Head-dim-strided input (dim stride != 1): two heads laid out dim-major.
    two_head_cache = torch.zeros(SLOTS, 2, D, dtype=fp8, device=dev)
    kt = torch.randn(T, D, 2, dtype=torch.bfloat16, device=dev).transpose(1, 2)
    assert kt.stride(2) != 1
    assert not ok(k_=kt, kc=two_head_cache)
    assert not ok(v_=kt, vc=two_head_cache)
    # index-V without its cache (or the reverse) is rejected.
    assert not ok(iv=idx_k)
    assert not ok(ivc=ok_cache)
    assert ok(iv=idx_k, ivc=ok_cache)
