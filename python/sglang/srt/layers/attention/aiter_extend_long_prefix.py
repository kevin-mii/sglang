"""EXTEND attention over a long cached prefix served by aiter's CK paged
batch-prefill kernel (``mha_batch_prefill``) instead of the Triton extend
kernel.

The chunk's K/V are already in the KV cache when the backend calls attention
(``save_kv_cache`` runs first), so the whole causal attention of the chunk over
prefix + chunk is one paged-prefill call: page indices = prefix indices
followed by the chunk's cache locations, causal mask bottom-right aligned. For
an fp8 cache q is cast to the cache dtype (as the Triton kernel does) and the
per-tensor K/V scales are passed as descales.

Measured on MI350X (16 q heads / 1 kv head per rank, head_dim 128, fp8 KV,
198K prefix): 8192 rows 25.8 -> 16.1 ms, 3222 rows 10.7 -> 8.0 ms; at 1231
rows and below the CK kernel is slower than the split-prefix Triton path
(fixed cost of a serial sweep per query tile), hence the row threshold.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.extend_attention import _copy_unified_indices_kernel

_mha_batch_prefill_func = None
_descale_cache: dict = {}


def aiter_batch_prefill_available() -> bool:
    global _mha_batch_prefill_func
    if _mha_batch_prefill_func is not None:
        return True
    try:
        from aiter.ops.mha import mha_batch_prefill_func
    except Exception:
        return False
    _mha_batch_prefill_func = mha_batch_prefill_func
    return True


def _descale(value: float, device: torch.device) -> torch.Tensor:
    key = (float(value), str(device))
    t = _descale_cache.get(key)
    if t is None:
        t = torch.full((1,), float(value), dtype=torch.float32, device=device)
        _descale_cache[key] = t
    return t


def build_paged_kv_indices(
    prefix_kv_indptr: torch.Tensor,
    prefix_kv_indices: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 (kv_indptr, page indices) over prefix + chunk per request."""
    device = prefix_kv_indptr.device
    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
    lens = prefix_lens + extend_seq_lens[:bs]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    total = int(prefix_kv_indices.numel()) + int(out_cache_loc.numel())
    pages = torch.empty(total, dtype=torch.int32, device=device)
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        out_cache_loc,
        kv_indptr,
        pages,
        bs,
    )
    return kv_indptr, pages


def extend_attention_fwd_aiter_paged(
    q: torch.Tensor,
    o: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    max_extend_len: int,
    max_seq_len: int,
    sm_scale: float,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """q [T, Hq, D] bf16; k/v_buffer [pages, Hkv, D] (page size 1); o [T, Hq, Dv]
    bf16 written in place. Causal over prefix + chunk."""
    assert aiter_batch_prefill_available()
    kv_dtype = k_buffer.dtype
    is_fp8 = kv_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if is_fp8:
        qk = q.to(kv_dtype)
        dev = q.device
        q_descale = _descale(1.0, dev)
        k_descale = _descale(1.0 if k_scale is None else k_scale, dev)
        v_descale = _descale(1.0 if v_scale is None else v_scale, dev)
    else:
        qk = q if q.dtype == kv_dtype else q.to(kv_dtype)
        q_descale = k_descale = v_descale = None
    if qo_indptr.dtype != torch.int32:
        qo_indptr = qo_indptr.to(torch.int32)
    _mha_batch_prefill_func(
        qk,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        page_indices,
        int(max_extend_len),
        int(max_seq_len),
        softmax_scale=sm_scale,
        causal=True,
        out=o,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return o
