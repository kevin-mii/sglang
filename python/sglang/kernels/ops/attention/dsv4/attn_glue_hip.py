"""Single-launch replacements for the torch glue around the DeepSeek-V4.1 sparse
attention metadata on HIP.

bs-1 decode on MI350X is launch-bound (~4.4 us per launch inside a HIP graph
replay), so a chain of small aten kernels costs its launch count, not its work.
Every kernel here reproduces the torch expression it replaces bit for bit; the
docstrings quote that expression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_indices_by_length_kernel(
    idx_ptr,
    len_ptr,
    out_ptr,
    idx2_ptr,
    len2_ptr,
    out2_ptr,
    rows_per_len,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SECOND: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)
    length_row = row // rows_per_len
    offs = tl.arange(0, BLOCK)
    if which == 0:
        length = tl.load(len_ptr + length_row)
        for w0 in tl.static_range(0, W1, BLOCK):
            w = w0 + offs
            m = w < W1
            v = tl.load(idx_ptr + row * W1 + w, mask=m, other=0)
            tl.store(out_ptr + row * W1 + w, tl.where(w < length, v, -1), mask=m)
    elif HAS_SECOND:
        length = tl.load(len2_ptr + length_row)
        for w0 in tl.static_range(0, W2, BLOCK):
            w = w0 + offs
            m = w < W2
            v = tl.load(idx2_ptr + row * W2 + w, mask=m, other=0)
            tl.store(out2_ptr + row * W2 + w, tl.where(w < length, v, -1), mask=m)


def _mask_shape(indices: torch.Tensor, lengths: torch.Tensor) -> Tuple[int, int, int]:
    assert indices.is_contiguous(), indices.stride()
    assert lengths.dim() == 1 and lengths.is_contiguous()
    w = indices.shape[-1]
    rows = indices.numel() // w if w else 0
    b = lengths.shape[0]
    assert b > 0 and rows % b == 0, (indices.shape, lengths.shape)
    return rows, rows // b, w


def mask_indices_by_length(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    indices2: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``torch.where(arange(w) < lengths.view(-1, 1, 1), indices, -1)`` for one or
    two ``[b, s, w]`` index tensors in one launch (the sparse decode kernel takes
    no per-row length and skips -1). ``lengths`` is ``[b]``; every ``s`` row of a
    batch entry shares its length. Returns new tensors of the input dtypes."""
    rows, rows_per_len, w1 = _mask_shape(indices, lengths)
    out = torch.empty_like(indices)
    has_second = indices2 is not None
    if has_second:
        rows2, rows_per_len2, w2 = _mask_shape(indices2, lengths2)
        assert rows2 == rows and rows_per_len2 == rows_per_len, (
            indices.shape,
            indices2.shape,
        )
        out2 = torch.empty_like(indices2)
    else:
        w2 = w1
        indices2, lengths2, out2 = indices, lengths, out
    if rows == 0:
        return out, (out2 if has_second else None)
    block = min(1024, triton.next_power_of_2(max(w1, w2)))
    _mask_indices_by_length_kernel[(rows, 2 if has_second else 1)](
        indices,
        lengths,
        out,
        indices2,
        lengths2,
        out2,
        rows_per_len,
        W1=w1,
        W2=w2,
        BLOCK=block,
        HAS_SECOND=has_second,
        num_warps=4,
    )
    return out, (out2 if has_second else None)


@triton.jit
def _expand_index_page_table_kernel(
    pt_ptr,
    out_ptr,
    n_pages,
    pt_stride,
    BPP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < n_pages
    page = tl.load(pt_ptr + row * pt_stride + p, mask=m, other=0)
    base = page.to(tl.int64) * BPP
    k = tl.arange(0, BPP)
    expanded = base[:, None] + k[None, :]
    tl.store(
        out_ptr + row * (n_pages * BPP) + p[:, None] * BPP + k[None, :],
        expanded.to(tl.int32),
        mask=m[:, None],
    )


def expand_index_page_table(
    page_table: torch.Tensor, blocks_per_page: int
) -> torch.Tensor:
    """``(page_table.to(int64) * bpp).unsqueeze(-1) + arange(bpp)`` reshaped to
    ``[bs, n * bpp]`` int32: the block table of a low-ratio indexer-K pool that
    pages at a fraction of a FULL page (see ``_expand_index_page_table``)."""
    if blocks_per_page == 1:
        return page_table
    assert blocks_per_page & (blocks_per_page - 1) == 0, blocks_per_page
    assert page_table.dim() == 2 and page_table.stride(1) == 1
    bs, n = page_table.shape
    out = torch.empty(
        (bs, n * blocks_per_page), dtype=torch.int32, device=page_table.device
    )
    if bs == 0 or n == 0:
        return out
    block = 256
    _expand_index_page_table_kernel[(bs, triton.cdiv(n, block))](
        page_table,
        out,
        n,
        page_table.stride(0),
        BPP=blocks_per_page,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _low_ratio_compression_metadata_kernel(
    seq_lens_ptr,
    raw_out_loc_ptr,
    c1_out_loc_ptr,
    c1_clamp1_ptr,
    c2_out_loc_ptr,
    c2_clamp1_ptr,
    rows,
    num_write_tokens,
    HAS_C1: tl.constexpr,
    HAS_C2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < rows
    mw = r < num_write_tokens
    seq_len = tl.load(seq_lens_ptr + r, mask=m, other=0)
    raw_out_loc = tl.load(raw_out_loc_ptr + r, mask=mw, other=0).to(tl.int64)
    if HAS_C1:
        # ratio 1: every token completes a group, so out_loc is raw_out_loc itself
        tl.store(c1_out_loc_ptr + r, raw_out_loc, mask=mw)
        tl.store(c1_clamp1_ptr + r, tl.maximum(seq_len, 1), mask=m)
    if HAS_C2:
        completes = (seq_len % 2) == 0
        tl.store(c2_out_loc_ptr + r, tl.where(completes, raw_out_loc // 2, -1), mask=mw)
        tl.store(c2_clamp1_ptr + r, tl.maximum(seq_len // 2, 1), mask=m)


def low_ratio_compression_metadata(
    seq_lens_casual: torch.Tensor,
    raw_out_loc: torch.Tensor,
    low_ratios: Tuple[int, ...],
) -> dict:
    """``_low_ratio_compression_metadata`` for ratios 1 and 2 in one launch::

        cR_out_loc             = where(seq_lens[:nw] % R == 0, raw_out_loc.to(int64) // R, -1)
        cR_topk_lengths_clamp1 = (seq_lens // R).clamp_min(1).to(int32)

    ``seq_lens_casual`` int32 ``[rows]``, ``raw_out_loc`` ``[nw <= rows]``. Returns
    ``{"c1_out_loc": ..., "c1_topk_lengths_clamp1": ..., "c2_...": ...}`` for the
    ratios present."""
    assert seq_lens_casual.dtype is torch.int32 and seq_lens_casual.is_contiguous()
    assert raw_out_loc.dim() == 1 and raw_out_loc.is_contiguous()
    assert set(low_ratios) <= {1, 2}, low_ratios
    rows = seq_lens_casual.shape[0]
    nw = raw_out_loc.shape[0]
    assert nw <= rows, (nw, rows)
    device = seq_lens_casual.device
    out = {}
    for ratio in (1, 2):
        if ratio in low_ratios:
            out[f"c{ratio}_out_loc"] = torch.empty(nw, dtype=torch.int64, device=device)
            out[f"c{ratio}_topk_lengths_clamp1"] = torch.empty(
                rows, dtype=torch.int32, device=device
            )
    if not out or rows == 0:
        return out
    dummy = seq_lens_casual
    _low_ratio_compression_metadata_kernel[(triton.cdiv(rows, 1024),)](
        seq_lens_casual,
        raw_out_loc,
        out.get("c1_out_loc", dummy),
        out.get("c1_topk_lengths_clamp1", dummy),
        out.get("c2_out_loc", dummy),
        out.get("c2_topk_lengths_clamp1", dummy),
        rows,
        nw,
        HAS_C1=1 in low_ratios,
        HAS_C2=2 in low_ratios,
        BLOCK=1024,
        num_warps=4,
    )
    return out


@triton.jit
def _sparse_buffers_kernel(
    c4_clamp1_ptr,
    c4_raw_ptr,
    c4_sparse_ptr,
    c4_sparse_raw_ptr,
    c4_page_ptr,
    c1_clamp1_ptr,
    c1_sparse_ptr,
    c1_page_ptr,
    c2_clamp1_ptr,
    c2_sparse_ptr,
    c2_page_ptr,
    rows,
    topk,
    width,
    HAS_C1: tl.constexpr,
    HAS_C2: tl.constexpr,
    BLOCK: tl.constexpr,
    FILL_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        # one program clamps the per-row lengths: a few ints per row
        for r0 in range(0, rows, BLOCK):
            r = r0 + tl.arange(0, BLOCK)
            m = r < rows
            c4_clamp1 = tl.load(c4_clamp1_ptr + r, mask=m, other=0)
            c4_raw = tl.load(c4_raw_ptr + r, mask=m, other=0)
            tl.store(c4_sparse_ptr + r, tl.minimum(c4_clamp1, topk), mask=m)
            tl.store(c4_sparse_raw_ptr + r, tl.minimum(c4_raw, topk), mask=m)
            if HAS_C1:
                c1_clamp1 = tl.load(c1_clamp1_ptr + r, mask=m, other=0)
                tl.store(c1_sparse_ptr + r, tl.minimum(c1_clamp1, topk), mask=m)
            if HAS_C2:
                c2_clamp1 = tl.load(c2_clamp1_ptr + r, mask=m, other=0)
                tl.store(c2_sparse_ptr + r, tl.minimum(c2_clamp1, topk), mask=m)
    else:
        # the rest fill the -1 page-index buffers, one buffer per range of programs
        fill = pid - 1
        n = rows * width
        per_buffer = tl.cdiv(n, FILL_BLOCK)
        which = fill // per_buffer
        i = (fill - which * per_buffer) * FILL_BLOCK + tl.arange(0, FILL_BLOCK)
        m = i < n
        if which == 0:
            tl.store(c4_page_ptr + i, -1, mask=m)
        elif which == 1:
            if HAS_C1:
                tl.store(c1_page_ptr + i, -1, mask=m)
            elif HAS_C2:
                tl.store(c2_page_ptr + i, -1, mask=m)
        else:
            if HAS_C1 and HAS_C2:
                tl.store(c2_page_ptr + i, -1, mask=m)


def _ceil_align(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def sparse_buffers(
    *,
    c4_topk_lengths_clamp1: torch.Tensor,
    c4_topk_lengths_raw: torch.Tensor,
    c1_topk_lengths_clamp1: Optional[torch.Tensor],
    c2_topk_lengths_clamp1: Optional[torch.Tensor],
    index_topk: int,
    page_index_align: int,
) -> dict:
    """``init_flashmla_related``'s tensors in one launch::

        c4_sparse_topk_lengths     = clamp(c4_topk_lengths_clamp1, max=topk)
        c4_sparse_topk_lengths_raw = clamp(c4_topk_lengths_raw, max=topk)
        cR_sparse_topk_lengths     = clamp(cR_topk_lengths_clamp1, max=topk)
        c{4,R}_sparse_page_indices = pad(full((rows, topk), -1), align)   # int32

    for the ratios R whose clamp-1 lengths are given. Returns a dict keyed by
    attribute name."""
    for t in (
        c4_topk_lengths_clamp1,
        c4_topk_lengths_raw,
        c1_topk_lengths_clamp1,
        c2_topk_lengths_clamp1,
    ):
        assert t is None or (
            t.dtype is torch.int32 and t.dim() == 1 and t.is_contiguous()
        ), t
    rows = c4_topk_lengths_clamp1.shape[0]
    assert c4_topk_lengths_raw.shape[0] == rows
    device = c4_topk_lengths_clamp1.device
    width = _ceil_align(index_topk, page_index_align)
    i32 = dict(dtype=torch.int32, device=device)
    out = {
        "c4_sparse_topk_lengths": torch.empty(rows, **i32),
        "c4_sparse_topk_lengths_raw": torch.empty(rows, **i32),
        "c4_sparse_page_indices": torch.empty((rows, width), **i32),
    }
    has_c1 = c1_topk_lengths_clamp1 is not None
    has_c2 = c2_topk_lengths_clamp1 is not None
    for ratio, has in ((1, has_c1), (2, has_c2)):
        if has:
            out[f"c{ratio}_sparse_topk_lengths"] = torch.empty(rows, **i32)
            out[f"c{ratio}_sparse_page_indices"] = torch.empty((rows, width), **i32)
    if rows == 0:
        return out
    dummy = out["c4_sparse_topk_lengths"]
    fill_block = 1024
    per_buffer = triton.cdiv(rows * width, fill_block)
    n_buffers = 1 + int(has_c1) + int(has_c2)
    _sparse_buffers_kernel[(1 + per_buffer * n_buffers,)](
        c4_topk_lengths_clamp1,
        c4_topk_lengths_raw,
        out["c4_sparse_topk_lengths"],
        out["c4_sparse_topk_lengths_raw"],
        out["c4_sparse_page_indices"],
        c1_topk_lengths_clamp1 if has_c1 else dummy,
        out.get("c1_sparse_topk_lengths", dummy),
        out.get("c1_sparse_page_indices", dummy),
        c2_topk_lengths_clamp1 if has_c2 else dummy,
        out.get("c2_sparse_topk_lengths", dummy),
        out.get("c2_sparse_page_indices", dummy),
        rows,
        index_topk,
        width,
        HAS_C1=has_c1,
        HAS_C2=has_c2,
        BLOCK=1024,
        FILL_BLOCK=fill_block,
        num_warps=4,
    )
    return out
