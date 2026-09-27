"""Split-K tl.dot_scaled route for the DSpark stage-0 main projection."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


_MAINPROJ_N = 5120
_MAINPROJ_K = 15360
_LOGGED_SHAPES: set[tuple[int, int]] = set()


@triton.jit
def _mainproj_dot_scaled_splitk_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    partial_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    K_PER_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + k_offset + tl.arange(0, BLOCK_K)
        offs_sk = (
            pid_k * K_PER_SPLIT + k_offset
        ) // 32 + tl.arange(0, BLOCK_K // 32)
        x = tl.load(
            x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        grouped = tl.reshape(x, (BLOCK_M, BLOCK_K // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(grouped), axis=2), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + (
            (bits & 0x7FFFFF) != 0
        ).to(tl.int32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        recip_bits = (254 - exponent) << 23
        recip = recip_bits.to(tl.float32, bitcast=True)
        quantized = tl.minimum(
            tl.maximum(grouped * recip[:, :, None], -448.0), 448.0
        ).to(tl.float8e4nv)
        x_q = tl.reshape(quantized, (BLOCK_M, BLOCK_K))
        x_scale = exponent.to(tl.uint8)

        w = tl.load(
            w_ptr
            + offs_n[:, None].to(tl.int64) * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None],
            other=0.0,
        )
        w_scale = tl.load(
            ws_ptr
            + (offs_n // 32)[:, None].to(tl.int64) * stride_wsn
            + offs_sk[None, :] * stride_wsk,
            mask=n_mask[:, None],
            other=0,
        )
        acc = tl.dot_scaled(
            x_q, x_scale, "e4m3", w.T, w_scale, "e4m3", acc
        )

    partial_ptrs = (
        partial_ptr
        + pid_k.to(tl.int64) * M * N
        + offs_m[:, None] * N
        + offs_n[None, :]
    )
    tl.store(
        partial_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _mainproj_splitk_reduce_kernel(
    partial_ptr,
    out_ptr,
    M,
    N,
    SPLIT_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in range(SPLIT_K):
        partial = tl.load(
            partial_ptr
            + split.to(tl.int64) * M * N
            + offs_m[:, None] * N
            + offs_n[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += partial
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def prepare_mainproj_dot_scaled_cache(
    weight_shuffled: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Undo the native MFMA lane order for the canonical dot_scaled operand."""
    if (
        weight_shuffled.dtype != torch.float8_e4m3fn
        or weight_shuffled.shape
        != (_MAINPROJ_N // 16, _MAINPROJ_K // 128, 2048)
        or weight_scale_ue8m0.dtype != torch.uint8
        or weight_scale_ue8m0.shape
        != (_MAINPROJ_N // 32, _MAINPROJ_K // 32)
    ):
        raise NotImplementedError

    shuffled = weight_shuffled.view(torch.uint8).view(
        _MAINPROJ_N // 16,
        _MAINPROJ_K // 128,
        2,
        2,
        16,
        2,
        16,
    )
    canonical = (
        shuffled.permute(0, 4, 1, 5, 2, 3, 6)
        .contiguous()
        .view(_MAINPROJ_N, _MAINPROJ_K)
        .view(torch.float8_e4m3fn)
    )
    return canonical, weight_scale_ue8m0.contiguous()


def mainproj_dot_scaled_splitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the fixed-shape DSpark main projection with a split-K reduce."""
    if (
        x.dim() != 2
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not 8 <= x.shape[0] <= 32
        or x.shape[1] != _MAINPROJ_K
        or weight.dtype != torch.float8_e4m3fn
        or weight.shape != (_MAINPROJ_N, _MAINPROJ_K)
        or not weight.is_contiguous()
        or weight_scale.dtype != torch.uint8
        or weight_scale.shape
        != (_MAINPROJ_N // 32, _MAINPROJ_K // 32)
        or not weight_scale.is_contiguous()
    ):
        raise NotImplementedError

    m, k = x.shape
    if m <= 24:
        block_n, block_k, split_k = 64, 128, 8
    else:
        block_n, block_k, split_k = 64, 256, 4
    if k % block_k != 0 or (k // block_k) % split_k != 0:
        raise NotImplementedError

    partial = torch.empty(
        (split_k, m, _MAINPROJ_N),
        dtype=torch.float32,
        device=x.device,
    )
    _mainproj_dot_scaled_splitk_kernel[
        (triton.cdiv(m, 16), _MAINPROJ_N // block_n, split_k)
    ](
        x,
        weight,
        weight_scale,
        partial,
        m,
        _MAINPROJ_N,
        k,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        k // split_k,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=2,
        num_stages=2,
    )

    out = torch.empty(
        (m, _MAINPROJ_N), dtype=torch.bfloat16, device=x.device
    )
    _mainproj_splitk_reduce_kernel[(triton.cdiv(m, 16), _MAINPROJ_N // 32)](
        partial,
        out,
        m,
        _MAINPROJ_N,
        split_k,
        BLOCK_M=16,
        BLOCK_N=32,
        num_warps=1,
    )

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    log_key = (rank, m)
    if log_key not in _LOGGED_SHAPES:
        _LOGGED_SHAPES.add(log_key)
        print(f"MAINPROJ_DOTSCALED rank={rank} M={m}", flush=True)
    return out
