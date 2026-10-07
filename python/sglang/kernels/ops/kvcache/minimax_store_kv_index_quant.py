"""Fused MiniMax-M3 sparse-cache store with the fp8 cast folded in.

One Triton launch per layer scales, casts and scatters the main K/V heads, the
index-K head and the optional index-V head into their token-major caches. It
serves the casting (fp8 pool) case that the raw-byte store
(`minimax_store_kv_index`) cannot, on CUDA and ROCm alike, and writes the same
bytes as the unfused `set_kv_buffer` / `set_index_k_buffer` stores.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_cast(x, inv_scale, cache_ptr, SCALED: tl.constexpr):
    # Same bytes as the unfused store: torch divides a bf16/fp16 tensor by a
    # Python scalar as x * (1/scale) in fp32, rounds to the input dtype, then
    # casts to the cache dtype.
    if SCALED:
        x = (x.to(tl.float32) * inv_scale).to(x.dtype)
    return x.to(cache_ptr.dtype.element_ty)


@triton.jit
def _store_kv_index_quant_kernel(
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    ik_ptr,
    ikc_ptr,
    iv_ptr,
    ivc_ptr,
    loc_ptr,
    k_inv_scale,
    v_inv_scale,
    ik_inv_scale,
    iv_inv_scale,
    sk_t,
    sk_h,
    sv_t,
    sv_h,
    sik_t,
    siv_t,
    skc_t,
    skc_h,
    svc_t,
    svc_h,
    sikc_t,
    sivc_t,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    HAS_IDX_V: tl.constexpr,
    K_SCALED: tl.constexpr,
    V_SCALED: tl.constexpr,
    IK_SCALED: tl.constexpr,
    IV_SCALED: tl.constexpr,
):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dv = tl.arange(0, V_HEAD_DIM)
    # Match the unfused stores on the reserved cuda-graph padding slot 0 (padding
    # rows may hold NaN): store_cache skips it, the K-only index scatter does not.
    if loc != 0:
        for h in tl.static_range(NUM_KV_HEADS):
            k = tl.load(k_ptr + t * sk_t + h * sk_h + offs_d)
            tl.store(
                kc_ptr + loc * skc_t + h * skc_h + offs_d,
                _scale_cast(k, k_inv_scale, kc_ptr, K_SCALED),
            )
            v = tl.load(v_ptr + t * sv_t + h * sv_h + offs_dv)
            tl.store(
                vc_ptr + loc * svc_t + h * svc_h + offs_dv,
                _scale_cast(v, v_inv_scale, vc_ptr, V_SCALED),
            )
    offs_i = tl.arange(0, IDX_DIM)
    idx_mask = offs_i < IDX_DIM
    if HAS_IDX_V:
        # the K+V index pool stores through store_cache
        idx_mask = idx_mask & (loc != 0)
    ik = tl.load(ik_ptr + t * sik_t + offs_i)
    tl.store(
        ikc_ptr + loc * sikc_t + offs_i,
        _scale_cast(ik, ik_inv_scale, ikc_ptr, IK_SCALED),
        mask=idx_mask,
    )
    if HAS_IDX_V:
        iv = tl.load(iv_ptr + t * siv_t + offs_i)
        tl.store(
            ivc_ptr + loc * sivc_t + offs_i,
            _scale_cast(iv, iv_inv_scale, ivc_ptr, IV_SCALED),
            mask=idx_mask,
        )


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


_SUPPORTED_CACHE_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


def can_store_kv_index_quant(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
) -> bool:
    """Whether every (input, cache) pair has a layout the kernel can address."""

    def _pair_storable(x: torch.Tensor, cache: torch.Tensor) -> bool:
        return (
            x.dtype in (torch.bfloat16, torch.float16)
            and cache.dtype in _SUPPORTED_CACHE_DTYPES
            and x.dim() == 3
            and cache.dim() == 3
            and x.shape[1] == cache.shape[1]
            and x.shape[2] == cache.shape[2]
            and _is_pow2(cache.shape[2])
            and x.stride(2) == 1
            and cache.stride(2) == 1
        )

    if not (_pair_storable(k, k_cache) and _pair_storable(v, v_cache)):
        return False
    if v.shape[1] != k.shape[1]:
        return False
    if not _pair_storable(idx_k, idx_k_cache) or idx_k.shape[1] != 1:
        return False
    if (idx_v is None) != (idx_v_cache is None):
        return False
    if idx_v is not None and not (
        _pair_storable(idx_v, idx_v_cache) and idx_v.shape[2] == idx_k.shape[2]
    ):
        return False
    return True


def _inv_scale(
    scale: Optional[float], x: torch.Tensor, cache: torch.Tensor
) -> Optional[float]:
    # As in MHATokenToKVPool.set_kv_buffer: a scale applies only where the store
    # casts, and None means unit (no division at all).
    if scale is None or x.dtype == cache.dtype:
        return None
    # torch's `x / scale` multiplies by the fp32 reciprocal of the fp32 scale.
    return float(np.float32(1.0) / np.float32(scale))


def store_kv_index_quant(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    loc: torch.Tensor,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
) -> None:
    """`cache[loc] = (x / scale).to(cache.dtype)` for K, V, index-K and index-V in one launch.

    Inputs are `[tokens, heads, dim]` (index tensors have one head), caches
    `[slots, heads, dim]`, `loc` one int32/int64 slot per token. Unlike
    `MHATokenToKVPool.set_kv_buffer`, the inputs are not modified in place.
    """
    T, H, D = k.shape
    if T == 0:
        return
    has_idx_v = idx_v is not None
    if not has_idx_v:
        idx_v, idx_v_cache = idx_k, idx_k_cache

    inv_scales = (
        _inv_scale(k_scale, k, k_cache),
        _inv_scale(v_scale, v, v_cache),
        _inv_scale(idx_k_scale, idx_k, idx_k_cache),
        _inv_scale(idx_v_scale if has_idx_v else None, idx_v, idx_v_cache),
    )
    _store_kv_index_quant_kernel[(T,)](
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        loc,
        *(1.0 if s is None else s for s in inv_scales),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        idx_k.stride(0),
        idx_v.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        idx_k_cache.stride(0),
        idx_v_cache.stride(0),
        NUM_KV_HEADS=H,
        HEAD_DIM=D,
        V_HEAD_DIM=v.shape[2],
        IDX_DIM=idx_k.shape[2],
        HAS_IDX_V=has_idx_v,
        K_SCALED=inv_scales[0] is not None,
        V_SCALED=inv_scales[1] is not None,
        IK_SCALED=inv_scales[2] is not None,
        IV_SCALED=inv_scales[3] is not None,
        # one program per token with a static head loop: sized for the few KV heads per rank
        num_warps=1,
    )
