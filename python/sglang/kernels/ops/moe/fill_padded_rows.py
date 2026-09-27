"""Fused padded-row fill for MoE top-k outputs.

Migrated from ``sglang.srt.layers.moe.topk`` (RFC #29630, Phase 2.5), where two
near-identical copies had accumulated; this keeps the later, runtime-winning
copy (explicit raises instead of asserts).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_padded_rows_kernel(
    out_ptr,
    num_token_non_padded_ptr,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    n_valid = tl.load(num_token_non_padded_ptr)
    if row >= n_valid:
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < n_cols
        ptrs = out_ptr + row * stride_row + cols
        fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
        tl.store(ptrs, fill, mask=mask)


@triton.jit
def _fill_padded_rows_pair_kernel(
    ids_ptr,
    weights_ptr,
    num_token_non_padded_ptr,
    n_cols,
    ids_fill_value,
    ids_stride_row,
    weights_stride_row,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    n_valid = tl.load(num_token_non_padded_ptr)
    if row >= n_valid:
        cols = tl.arange(0, BLOCK_COLS)
        mask = cols < n_cols
        ids = ids_ptr + row * ids_stride_row + cols
        weights = weights_ptr + row * weights_stride_row + cols
        ids_fill = tl.full(
            (BLOCK_COLS,), ids_fill_value, dtype=ids_ptr.dtype.element_ty
        )
        weights_fill = tl.zeros((BLOCK_COLS,), dtype=weights_ptr.dtype.element_ty)
        tl.store(ids, ids_fill, mask=mask)
        tl.store(weights, weights_fill, mask=mask)


def _can_fuse_padded_region(x: torch.Tensor) -> bool:
    # The fused kernel uses one program per row and assumes a row-major 2D
    # tensor (columns contiguous); fall back to eager for anything else.
    return x.dim() == 2 and x.stride(1) == 1


def _fill_padded_rows(
    x: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    fill_value,
) -> None:
    """Set ``x[row, :] = fill_value`` for every padded row (row index
    ``>= num_token_non_padded``) using a single Triton launch.

    Replaces the eager ``arange + (>=) + boolean index_put_`` sequence, which
    issues several launch-latency-bound kernels per call. The grid is static
    (one program per row) and the pad count is read from device memory inside
    the kernel, so this is safe to capture inside a CUDA/HIP graph.
    """
    # Metadata-only checks (no device sync): the kernel reads a single scalar
    # routing count from device memory, so it must be a 1-element integer tensor
    # on the same device as ``x``. Use explicit raises (not asserts) so the
    # checks survive ``python -O`` and invalid inputs fail loudly instead of
    # turning into opaque Triton/memory errors.
    if not isinstance(num_token_non_padded, torch.Tensor):
        raise TypeError("num_token_non_padded must be a torch.Tensor")
    if num_token_non_padded.numel() != 1:
        raise ValueError(
            "num_token_non_padded must be a single-element tensor, got shape "
            f"{tuple(num_token_non_padded.shape)}"
        )
    if num_token_non_padded.dtype.is_floating_point:
        raise TypeError(
            "num_token_non_padded must be an integer tensor, got "
            f"{num_token_non_padded.dtype}"
        )
    if num_token_non_padded.device != x.device:
        raise ValueError("num_token_non_padded and x must be on the same device")
    n_rows, n_cols = x.shape
    _fill_padded_rows_kernel[(n_rows,)](
        x,
        num_token_non_padded,
        n_cols,
        fill_value,
        x.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
    )


def _fill_padded_rows_pair(
    ids: torch.Tensor,
    weights: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    *,
    ids_fill_value: int,
) -> bool:
    """Fill padded top-k IDs and zero their weights in one launch."""
    if (
        not _can_fuse_padded_region(ids)
        or not _can_fuse_padded_region(weights)
        or ids.shape != weights.shape
        or ids.device != weights.device
        or not isinstance(num_token_non_padded, torch.Tensor)
        or num_token_non_padded.numel() != 1
        or num_token_non_padded.dtype.is_floating_point
        or num_token_non_padded.device != ids.device
    ):
        return False
    n_rows, n_cols = ids.shape
    _fill_padded_rows_pair_kernel[(n_rows,)](
        ids,
        weights,
        num_token_non_padded,
        n_cols,
        ids_fill_value,
        ids.stride(0),
        weights.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
    )
    return True
