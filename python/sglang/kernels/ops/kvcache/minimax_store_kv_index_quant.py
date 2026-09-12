"""Fused MiniMax-M3 sparse-cache store with on-the-fly (fp8) quantization.

One Triton launch per layer writes the main K/V heads, the index-K head and the
optional index-V head into their token-major caches, applying the per-tensor
KV scales and the cache dtype cast in registers. It replaces the unfused
sequence used when the pools are fp8 (``x.div_(scale)`` + ``.to(fp8)`` for K
and V, the index ``k / scale`` + ``.to(fp8)`` and the index scatter, plus the
main store kernel): 7-8 small launches per layer -> 1.

Triton, so it runs on CUDA and ROCm alike; the CUDA raw-byte JIT kernel
(``minimax_store_kv_index``) stays the fast path when no cast is needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


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
    k_scale,
    v_scale,
    ik_scale,
    iv_scale,
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
):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dv = tl.arange(0, V_HEAD_DIM)
    for h in tl.static_range(NUM_KV_HEADS):
        k = tl.load(k_ptr + t * sk_t + h * sk_h + offs_d).to(tl.float32) / k_scale
        tl.store(
            kc_ptr + loc * skc_t + h * skc_h + offs_d, k.to(kc_ptr.dtype.element_ty)
        )
        v = tl.load(v_ptr + t * sv_t + h * sv_h + offs_dv).to(tl.float32) / v_scale
        tl.store(
            vc_ptr + loc * svc_t + h * svc_h + offs_dv, v.to(vc_ptr.dtype.element_ty)
        )
    offs_i = tl.arange(0, IDX_DIM)
    ik = tl.load(ik_ptr + t * sik_t + offs_i).to(tl.float32) / ik_scale
    tl.store(ikc_ptr + loc * sikc_t + offs_i, ik.to(ikc_ptr.dtype.element_ty))
    if HAS_IDX_V:
        iv = tl.load(iv_ptr + t * siv_t + offs_i).to(tl.float32) / iv_scale
        tl.store(ivc_ptr + loc * sivc_t + offs_i, iv.to(ivc_ptr.dtype.element_ty))


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
    k_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
) -> bool:
    """Shapes/dtypes this kernel serves: token-major ``[slots, heads, dim]``
    caches in a bf16/fp16/fp8 dtype, bf16/fp16 inputs, power-of-2 head dims."""
    if k.dtype not in (torch.bfloat16, torch.float16):
        return False
    if k_cache.dtype not in _SUPPORTED_CACHE_DTYPES:
        return False
    if idx_k_cache.dtype not in _SUPPORTED_CACHE_DTYPES:
        return False
    if idx_v_cache is not None and idx_v_cache.dtype not in _SUPPORTED_CACHE_DTYPES:
        return False
    if k_cache.dim() != 3 or idx_k_cache.dim() != 3:
        return False
    if k.dim() != 3 or idx_k.dim() != 3:
        return False
    if not (_is_pow2(k_cache.shape[2]) and _is_pow2(idx_k_cache.shape[2])):
        return False
    # Contiguous within a head row (dim stride 1) on both sides.
    if k.stride(2) != 1 or idx_k.stride(2) != 1:
        return False
    if k_cache.stride(2) != 1 or idx_k_cache.stride(2) != 1:
        return False
    return True


def store_kv_index_quant(
    k: torch.Tensor,  # [T, H, D] bf16/fp16
    v: torch.Tensor,  # [T, H, Dv]
    k_cache: torch.Tensor,  # [slots, H, D] cache dtype
    v_cache: torch.Tensor,  # [slots, H, Dv]
    idx_k: torch.Tensor,  # [T, 1, Di]
    idx_k_cache: torch.Tensor,  # [slots, 1, Di]
    idx_v: Optional[torch.Tensor],  # [T, 1, Di] or None
    idx_v_cache: Optional[torch.Tensor],
    loc: torch.Tensor,  # [T] int32/int64 slot ids
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
) -> None:
    """``cache[loc] = (x / scale).to(cache.dtype)`` for K, V, index-K (and
    index-V), one launch. ``None`` scale means unit scale."""
    T, H, D = k.shape
    Dv = v.shape[2]
    Di = idx_k.shape[2]
    if T == 0:
        return
    has_iv = idx_v is not None
    if not has_iv:
        idx_v, idx_v_cache = idx_k, idx_k_cache
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
        float(1.0 if k_scale is None else k_scale),
        float(1.0 if v_scale is None else v_scale),
        float(1.0 if idx_k_scale is None else idx_k_scale),
        float(1.0 if idx_v_scale is None else idx_v_scale),
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
        V_HEAD_DIM=Dv,
        IDX_DIM=Di,
        HAS_IDX_V=has_iv,
        num_warps=1,
    )
