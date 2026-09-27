"""Full-vocabulary argmax from vocab-parallel logits shards."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.row_argmax import _argmax_pair

_LOGGED_SHAPES: set[tuple[int, int, int]] = set()
_PACKED_BUFFERS: dict[
    tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]
] = {}


@triton.jit
def _shard_argmax_partial_kernel(
    X,
    PV,
    PI,
    n_cols,
    row_stride,
    valid_cols,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    per_split = tl.cdiv(n_cols, SPLITS)
    start = split * per_split
    end = tl.minimum(start + per_split, n_cols)
    best_value = float("-inf")
    best_index = n_cols
    for offset in tl.range(start, end, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        valid = cols < tl.minimum(end, valid_cols)
        values = tl.load(
            X + row * row_stride + cols, mask=valid, other=float("-inf")
        ).to(tl.float32)
        indices = tl.where(cols < valid_cols, cols, n_cols)
        value, index = tl.reduce((values, indices), 0, _argmax_pair)
        best_value, best_index = _argmax_pair(best_value, best_index, value, index)
    tl.store(PV + row * SPLITS + split, best_value)
    tl.store(PI + row * SPLITS + split, best_index)


@triton.jit
def _shard_argmax_final_kernel(
    PV,
    PI,
    LOCAL_PACKED,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    tp_rank,
    n_cols,
    rows,
):
    row = tl.program_id(0)
    splits = tl.arange(0, BLOCK)
    valid = splits < SPLITS
    values = tl.load(PV + row * SPLITS + splits, mask=valid, other=float("-inf"))
    indices = tl.load(PI + row * SPLITS + splits, mask=valid, other=n_cols)
    best_value, index = tl.reduce((values, indices), 0, _argmax_pair)
    value_bits = best_value.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    tl.store(LOCAL_PACKED + row, value_bits)
    global_index = index.to(tl.int64) + tp_rank * n_cols
    global_index = tl.where(
        index == n_cols, 0x7FFFFFFFFFFFFFFF, global_index
    )
    tl.store(LOCAL_PACKED + rows + row, global_index)


@triton.jit
def _shard_argmax_combine_kernel(
    PACKED,
    OUT,
    rows,
    TP_SIZE: tl.constexpr,
    GLOBAL_VOCAB: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, TP_SIZE)
    base = ranks * (2 * rows)
    value_bits = tl.load(PACKED + base + row)
    values = value_bits.to(tl.int32).to(tl.float32, bitcast=True)
    ids = tl.load(PACKED + base + rows + row)
    _, index = tl.reduce((values, ids), 0, _argmax_pair)
    index = tl.where(index == 0x7FFFFFFFFFFFFFFF, 0, index)
    tl.store(OUT + row, index)


def shard_argmax(
    logits: torch.Tensor,
    *,
    tp_rank: int,
    tp_group,
    valid_cols: int | None = None,
) -> torch.Tensor:
    """Return ``torch.argmax(gathered_logits, dim=-1)`` without gathering logits."""
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise NotImplementedError
    rows, n_cols = logits.shape
    valid_cols = n_cols if valid_cols is None else int(valid_cols)
    tp_size = int(tp_group.world_size)
    if tp_size <= 1 or not 0 <= valid_cols <= n_cols:
        raise NotImplementedError

    block = 4096
    splits = triton.cdiv(n_cols, block)
    partial_values = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    partial_indices = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    _shard_argmax_partial_kernel[(rows, splits)](
        logits,
        partial_values,
        partial_indices,
        n_cols,
        logits.stride(0),
        valid_cols,
        SPLITS=splits,
        BLOCK=block,
        num_warps=4,
    )
    buffer_key = (rows, tp_size, logits.device)
    packed_buffers = _PACKED_BUFFERS.get(buffer_key)
    if packed_buffers is None:
        local_packed = torch.empty(
            (2 * rows,), dtype=torch.int64, device=logits.device
        )
        gathered_packed = torch.empty(
            (tp_size * 2 * rows,), dtype=torch.int64, device=logits.device
        )
        _PACKED_BUFFERS[buffer_key] = (local_packed, gathered_packed)
    else:
        local_packed, gathered_packed = packed_buffers
    _shard_argmax_final_kernel[(rows,)](
        partial_values,
        partial_indices,
        local_packed,
        SPLITS=splits,
        BLOCK=triton.next_power_of_2(splits),
        tp_rank=tp_rank,
        n_cols=n_cols,
        rows=rows,
        num_warps=1,
    )

    tp_group.all_gather_into_tensor(gathered_packed, local_packed)

    global_vocab = tp_size * n_cols
    output = torch.empty((rows,), dtype=torch.int64, device=logits.device)
    _shard_argmax_combine_kernel[(rows,)](
        gathered_packed,
        output,
        rows,
        TP_SIZE=triton.next_power_of_2(tp_size),
        GLOBAL_VOCAB=global_vocab,
        num_warps=1,
    )

    log_key = (int(tp_rank), rows, n_cols)
    if log_key not in _LOGGED_SHAPES:
        _LOGGED_SHAPES.add(log_key)
        print(
            f"SHARD_ARGMAX_PACKED rank={tp_rank} N={rows} packed=1",
            flush=True,
        )
    return output
