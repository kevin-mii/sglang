"""Single-launch replacements for the DeepSeek-V4.1 decode glue on HIP must be bitwise the torch chains they replace."""

from __future__ import annotations

import random
import sys

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel
import torch

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_index_page_table,
    low_ratio_compression_metadata,
    mask_indices_by_length,
    sparse_buffers,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    pack_fp4_query_flydsl,
    sort_selection_rows,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend import (
    _expand_index_page_table,
    _low_ratio_compression_metadata,
    _low_ratio_sparse_buffers,
    _pad_last_dim,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    CandidateBlocks,
    _aot_topk_sorts_output,
    topk_transform_paged_sorted,
    topk_within_candidate_blocks_hip,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="HIP decode glue for the AITER / FlyDSL DeepSeek-V4.1 path (gfx95x).",
)

DEVICE = "cuda"


def _seed(seed: int) -> random.Random:
    torch.manual_seed(seed)
    return random.Random(seed)


def _topk_inputs(rng, bs, width, topk, page_size, lens):
    # a row is at most as long as its logits: the kernel reads scores[0, len) and
    # page_table[0, len // page_size], so a longer row reads memory past the tensor
    # end, and what it selects there (a stale word as the page) differs by launch
    assert max(lens, default=0) <= width, (lens, width)
    # distinct scores per row: the radix top-k breaks a tie at the threshold in
    # atomic-counter order, so two launches over tied scores can select different
    # sets, and this compares two launches
    scores = (
        torch.stack([torch.randperm(width, device=DEVICE).float() for _ in range(bs)])
        * 0.37
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    n_pages = (width + page_size - 1) // page_size
    page_table = torch.randint(
        0, 1 << 20, (bs, n_pages), dtype=torch.int32, device=DEVICE
    )
    return scores, seq_lens, page_table


@pytest.mark.skipif(
    not _aot_topk_sorts_output(), reason="sgl_kernel predates sort_output"
)
@pytest.mark.parametrize("topk", [512, 1024, 64, 100])
@pytest.mark.parametrize("with_raw", [True, False])
def test_sorted_topk_epilogue_matches_transform_then_sort(topk: int, with_raw: bool):
    rng = _seed(topk)
    for width in (1024, 70000):
        for bs in (1, 33):
            page_size = rng.choice([16, 64])
            # the topk + 1 edge (a radix row of exactly topk picks) only where the
            # logits are that wide; at width == topk it would run past the row
            edges = [
                n
                for n in (0, 1, topk - 1, topk, topk + 1, width // 2, width)
                if n <= width
            ]
            lens = [rng.choice(edges) for _ in range(bs)]
            scores, seq_lens, page_table = _topk_inputs(
                rng, bs, width, topk, page_size, lens
            )
            # the unsorted transform, then the order sort_selection_rows gives it: by
            # position with raw indices, by slot without (the served decode shape);
            # the Triton sort for a power-of-two k, a torch stable sort otherwise
            ref = torch.empty(bs, topk, dtype=torch.int32, device=DEVICE)
            ref_raw = torch.empty_like(ref) if with_raw else None
            torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
                scores, seq_lens, page_table, ref, page_size, ref_raw
            )
            if topk & (topk - 1) == 0:
                sort_selection_rows(ref, ref_raw)
            else:
                by = ref_raw if with_raw else ref
                key = torch.where(by < 0, torch.iinfo(torch.int32).max, by)
                order = torch.sort(key, dim=1, stable=True).indices
                ref = torch.gather(ref, 1, order)
                if with_raw:
                    ref_raw = torch.gather(ref_raw, 1, order)
            out = torch.empty_like(ref)
            out_raw = torch.empty_like(ref) if with_raw else None
            topk_transform_paged_sorted(
                scores, seq_lens, page_table, out, page_size, out_raw
            )
            assert torch.equal(out, ref), (bs, width, page_size, lens)
            if with_raw:
                assert torch.equal(out_raw, ref_raw)
            # padding last, keys ascending
            for row in (ref_raw if with_raw else ref).tolist():
                n = sum(x >= 0 for x in row)
                assert all(x < 0 for x in row[n:])
                assert row[:n] == sorted(row[:n])


def _candidates(rng, rows, num_blocks, topk_blocks, block_size, seq_lens):
    ids = torch.full((rows, topk_blocks), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        reach = min(num_blocks, -(-int(seq_lens[r]) // block_size))
        picks = rng.sample(range(reach), min(reach, topk_blocks))
        rng.shuffle(picks)
        if picks:
            ids[r, : len(picks)] = torch.tensor(picks, dtype=torch.int32, device=DEVICE)
    block_lens = (seq_lens + block_size - 1) // block_size
    compact_lens = (torch.clamp(block_lens, max=topk_blocks) * block_size).to(
        torch.int32
    )
    width = topk_blocks * block_size
    return CandidateBlocks(
        ids=ids,
        compact_lens=compact_lens,
        compact_page_table=torch.zeros((rows, 1), dtype=torch.int32, device=DEVICE),
        compact_page_size=1 << (width - 1).bit_length(),
        block_size=block_size,
    )


@pytest.mark.parametrize("topk", [512, 64])
@pytest.mark.parametrize("with_raw", [True, False])
def test_sorted_candidate_mapping_matches_pack_then_sort(topk: int, with_raw: bool):
    rng = _seed(11 + topk)
    block_size, topk_blocks, page_size = 64, 16, 64
    for rows in (1, 7):
        width = 8192
        lens = [
            rng.choice([0, 1, 300, topk, topk + 5, 3000, width]) for _ in range(rows)
        ]
        scores, seq_lens, page_table = _topk_inputs(
            rng, rows, width, topk, page_size, lens
        )
        cands = _candidates(
            rng, rows, width // block_size, topk_blocks, block_size, seq_lens
        )
        outs = []
        for sort in (False, True):
            page = torch.empty(rows, topk, dtype=torch.int32, device=DEVICE)
            raw = torch.empty_like(page) if with_raw else None
            topk_within_candidate_blocks_hip(
                scores,
                seq_lens,
                cands,
                page_table=page_table,
                page_size=page_size,
                page_indices=page,
                raw_indices=raw,
                sort_output=sort,
            )
            if not sort:
                sort_selection_rows(page, raw)
            outs.append((page, raw))
        assert torch.equal(outs[0][0], outs[1][0])
        if with_raw:
            assert torch.equal(outs[0][1], outs[1][1])


def _ref_mask(indices, lengths):
    w = indices.shape[-1]
    pos = torch.arange(w, device=indices.device, dtype=lengths.dtype)
    return torch.where(pos < lengths.view(-1, 1, 1), indices, indices.new_full((), -1))


@pytest.mark.parametrize("s", [1, 4])
@pytest.mark.parametrize("len_dtype", [torch.int32, torch.int64])
def test_mask_indices_by_length_single_and_pair(s: int, len_dtype):
    rng = _seed(3)
    for b in (1, 6):
        w1, w2 = 128, 512
        idx1 = torch.randint(-1, 1 << 20, (b, s, w1), dtype=torch.int32, device=DEVICE)
        idx2 = torch.randint(-1, 1 << 20, (b, s, w2), dtype=torch.int32, device=DEVICE)
        len1 = torch.tensor(
            [rng.choice([0, 1, 5, w1, w1 + 3]) for _ in range(b)],
            dtype=len_dtype,
            device=DEVICE,
        )
        len2 = torch.tensor(
            [rng.choice([0, 1, 17, w2 - 1, w2, 999]) for _ in range(b)],
            dtype=len_dtype,
            device=DEVICE,
        )
        out1, out2 = mask_indices_by_length(idx1, len1, idx2, len2)
        assert torch.equal(out1, _ref_mask(idx1, len1))
        assert torch.equal(out2, _ref_mask(idx2, len2))
        only, none = mask_indices_by_length(idx2, len2)
        assert none is None and torch.equal(only, _ref_mask(idx2, len2))
        assert out1.dtype is idx1.dtype and out1.shape == idx1.shape


@pytest.mark.parametrize("bpp", [1, 2, 4])
def test_expand_index_page_table(bpp: int):
    _seed(5)
    for bs, n in ((1, 4608), (3, 17), (0, 10)):
        page_table = torch.randint(
            0, 1 << 20, (bs, n), dtype=torch.int32, device=DEVICE
        )
        ref = _expand_index_page_table(
            page_table, full_page_size=64 * bpp, compress_ratio=1, index_page_size=64
        )
        out = expand_index_page_table(page_table, bpp)
        assert out.dtype is torch.int32 and out.shape == ref.shape
        assert torch.equal(out, ref)
    # a strided (row-sliced) table is read through its strides
    page_table = torch.randint(0, 1 << 20, (8, 33), dtype=torch.int32, device=DEVICE)[
        ::2
    ]
    assert torch.equal(
        expand_index_page_table(page_table, 4),
        _expand_index_page_table(
            page_table, full_page_size=256, compress_ratio=1, index_page_size=64
        ),
    )


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratios", [(1,), (2,), (1, 2)])
def test_low_ratio_compression_metadata(loc_dtype, ratios):
    rng = _seed(7)
    for rows, nw in ((1, 1), (9, 9), (12, 5)):
        seq_lens = torch.tensor(
            [rng.choice([0, 1, 2, 3, 1000, 1001, 65535]) for _ in range(rows)],
            dtype=torch.int32,
            device=DEVICE,
        )
        raw_out_loc = torch.randint(0, 1 << 24, (nw,), dtype=loc_dtype, device=DEVICE)
        out = low_ratio_compression_metadata(seq_lens, raw_out_loc, ratios)
        assert set(out) == {
            f"c{r}_{k}" for r in ratios for k in ("out_loc", "topk_lengths_clamp1")
        }
        for r in ratios:
            ref_loc, ref_clamp1 = _low_ratio_compression_metadata(
                r, seq_lens, raw_out_loc
            )
            assert out[f"c{r}_out_loc"].dtype is ref_loc.dtype
            assert torch.equal(out[f"c{r}_out_loc"], ref_loc)
            assert out[f"c{r}_topk_lengths_clamp1"].dtype is ref_clamp1.dtype
            assert torch.equal(out[f"c{r}_topk_lengths_clamp1"], ref_clamp1)


@pytest.mark.parametrize("topk", [512, 1000])
@pytest.mark.parametrize("ratios", [(), (1,), (2,), (1, 2)])
def test_sparse_buffers(topk: int, ratios):
    rng = _seed(9)
    for rows in (1, 5, 70):

        def random_lengths():
            return torch.tensor(
                [
                    rng.choice([1, 2, topk - 1, topk, topk + 1, 1 << 16])
                    for _ in range(rows)
                ],
                dtype=torch.int32,
                device=DEVICE,
            )

        c4_clamp1, c4_raw = random_lengths(), random_lengths()
        low = {r: random_lengths() for r in ratios}
        out = sparse_buffers(
            c4_topk_lengths_clamp1=c4_clamp1,
            c4_topk_lengths_raw=c4_raw,
            c1_topk_lengths_clamp1=low.get(1),
            c2_topk_lengths_clamp1=low.get(2),
            index_topk=topk,
            page_index_align=64,
        )
        ref_page = _pad_last_dim(
            torch.full((rows, topk), -1, dtype=torch.int32, device=DEVICE)
        )
        assert torch.equal(
            out["c4_sparse_topk_lengths"], torch.clamp(c4_clamp1, max=topk)
        )
        assert torch.equal(
            out["c4_sparse_topk_lengths_raw"], torch.clamp(c4_raw, max=topk)
        )
        assert torch.equal(out["c4_sparse_page_indices"], ref_page)
        for r in (1, 2):
            if r in ratios:
                ref_len, ref_pi, _ = _low_ratio_sparse_buffers(low[r], topk, False)
                assert torch.equal(out[f"c{r}_sparse_topk_lengths"], ref_len)
                assert torch.equal(out[f"c{r}_sparse_page_indices"], ref_pi)
            else:
                assert f"c{r}_sparse_page_indices" not in out


def _ref_init_compression_metadata(
    seq_lens, positions, raw_out_loc, page_table, page_size
):
    nw = raw_out_loc.shape[0]
    out = []
    for ratio in (4, 128):
        should = seq_lens[:nw] % ratio == 0
        out_loc = torch.where(
            should, raw_out_loc[:nw] // ratio, torch.zeros_like(raw_out_loc[:nw])
        )
        out += [
            out_loc.to(torch.int64),
            (positions & ~(ratio - 1)).to(torch.int32),
            (seq_lens // ratio).to(torch.int32),
            torch.clamp(seq_lens // ratio, min=1).to(torch.int32),
        ]
    max_pages = page_table.shape[1]
    c128_page_size = page_size // 128
    width = c128_page_size * max_pages
    offs = torch.arange(width, device=seq_lens.device)
    page_idx = offs // c128_page_size
    vals = (
        page_table.to(torch.int64)[:, page_idx.clamp(max=max_pages - 1)]
        * c128_page_size
        + offs % c128_page_size
    )
    vals = torch.where(page_idx[None, :] < max_pages, vals, torch.zeros_like(vals))
    valid = offs[None, :] < (seq_lens // 128)[:, None]
    out.append(torch.where(valid, vals, torch.full_like(vals, -1)).to(torch.int32))
    return out


@pytest.mark.parametrize("max_pages", [1, 37, 4608])
@pytest.mark.parametrize("page_size", [128, 256])
def test_init_compression_metadata_grid(max_pages: int, page_size: int):
    rng = _seed(13)
    for bs, nw in ((1, 1), (6, 6), (6, 4)):
        seq_lens = torch.tensor(
            [
                rng.choice(
                    [0, 1, 4, 127, 128, 129, 5000, 128 * max_pages * page_size // 128]
                )
                for _ in range(bs)
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
        positions = torch.clamp(seq_lens - 1, min=0)
        raw_out_loc = torch.randint(0, 1 << 24, (nw,), dtype=torch.int64, device=DEVICE)
        page_table = torch.randint(
            0, 1 << 20, (bs, max_pages), dtype=torch.int32, device=DEVICE
        )
        outs = init_compression_metadata(
            seq_lens, positions, raw_out_loc, page_table, page_size, True
        )
        refs = _ref_init_compression_metadata(
            seq_lens, positions, raw_out_loc, page_table, page_size
        )
        assert len(outs) == len(refs) == 9
        for got, ref in zip(outs, refs):
            assert got.dtype is ref.dtype, (got.dtype, ref.dtype)
            assert torch.equal(got, ref)
        # scalars only
        outs = init_compression_metadata(
            seq_lens, positions, raw_out_loc, None, 0, False
        )
        assert outs[-1] is None
        for got, ref in zip(outs[:-1], refs[:-1]):
            assert torch.equal(got, ref)


def pack_fp4_query_flydsl_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The three-launch form of ``pack_fp4_query_flydsl``: the shared quantizer, then zeros
    and a permuted copy into the scale layout."""
    num_tokens, heads = q.shape[0], q.shape[1]
    assert heads % 16 == 0 and heads <= 64, heads
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(num_tokens, heads, 64)
    sf_bytes = q_sf.view(torch.uint8).view(num_tokens, heads // 16, 16, 4)
    q_scale = torch.zeros((num_tokens, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    q_scale[:, 0, :, :, : heads // 16] = sf_bytes.permute(0, 3, 2, 1)
    return q_fp4, q_scale


@pytest.mark.parametrize("heads", [64, 32, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pack_fp4_query_flydsl_single_launch(heads: int, dtype):
    _seed(17)
    for tokens in (1, 3, 40):
        q = torch.randn(tokens, heads, 128, device=DEVICE, dtype=dtype) * 4
        # exact fp4 grid points and tie values, zeros and a huge group
        q[0, 0, :32] = torch.tensor(
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4, device=DEVICE, dtype=dtype
        )
        q[0, 0, 32:64] = 0.0
        q[0, 0, 64:96] = 3.0e4
        ref_fp4, ref_scale = pack_fp4_query_flydsl_torch(q)
        fp4, scale = pack_fp4_query_flydsl(q)
        assert fp4.dtype is ref_fp4.dtype and scale.dtype is ref_scale.dtype
        assert fp4.shape == ref_fp4.shape and scale.shape == ref_scale.shape
        assert torch.equal(fp4, ref_fp4)
        assert torch.equal(scale, ref_scale)
    empty = torch.empty(0, heads, 128, device=DEVICE, dtype=dtype)
    fp4, scale = pack_fp4_query_flydsl(empty)
    assert fp4.shape == (0, heads, 64) and scale.shape == (0, 1, 4, 16, 4)


@pytest.mark.parametrize("compressed_kv", [False, True])
def test_rope_fake_quant_gathers_freqs_by_position(compressed_kv: bool):
    from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
        rope_tail_fake_quant_fp4,
    )

    _seed(19)
    table = torch.polar(
        torch.ones(4096, 32, device=DEVICE),
        torch.rand(4096, 32, device=DEVICE) * 6.283,
    )
    for tokens, heads in ((1, 64), (5, 64), (17, 1)):
        x = torch.randn(tokens, heads, 128, device=DEVICE, dtype=torch.bfloat16) * 3
        for pos_dtype in (torch.int64, torch.int32):
            pos = torch.randint(0, 4096, (tokens,), device=DEVICE, dtype=pos_dtype)
            ref = rope_tail_fake_quant_fp4(
                x, table[pos], 64, compressed_kv=compressed_kv
            )
            out = rope_tail_fake_quant_fp4(
                x, table, 64, compressed_kv=compressed_kv, positions=pos
            )
            assert torch.equal(out, ref)


def test_page_table_from_req_to_token_matches_torch():
    from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
        page_table_from_req_to_token,
    )

    _seed(23)
    req_to_token = torch.randint(
        0, 2**20, (300, 8192), device=DEVICE, dtype=torch.int32
    )
    req_to_token[
        7, :64
    ] = -3  # torch floor-divides; the slot values are never negative in serving
    for bs, max_seq_len, page in (
        (1, 1000, 64),
        (8, 4097, 64),
        (64, 8192, 64),
        (5, 63, 64),
        (3, 128, 128),
        (2, 8191, 32),
    ):
        req = torch.randint(0, 300, (bs,), device=DEVICE, dtype=torch.int32)
        req[0] = 7
        ref = (req_to_token[req, :max_seq_len:page] // page).to(torch.int32)
        got = page_table_from_req_to_token(req_to_token, req, max_seq_len, page)
        assert got.shape == ref.shape and got.dtype == torch.int32
        assert torch.equal(got, ref)
    empty = page_table_from_req_to_token(
        req_to_token, torch.empty(0, device=DEVICE, dtype=torch.int32), 1000, 64
    )
    assert empty.shape == (0, 16)


def test_widen_pair_i64_matches_casts():
    from sglang.kernels.ops.attention.dsv4.attn_glue_hip import widen_pair_i64

    _seed(29)
    for bs in (0, 1, 8, 64, 257, 5000):
        a = torch.randint(0, 300, (bs,), device=DEVICE, dtype=torch.int32)
        b = torch.randint(0, 2**31 - 1, (bs,), device=DEVICE, dtype=torch.int32)
        oa, ob = widen_pair_i64(a, b)
        assert oa.dtype == ob.dtype == torch.int64
        assert torch.equal(oa, a.to(torch.int64)) and torch.equal(ob, b.to(torch.int64))
    oa, ob = widen_pair_i64(
        torch.tensor([3, -1], device=DEVICE, dtype=torch.int64),
        torch.tensor([9, 0], device=DEVICE, dtype=torch.int32),
    )
    assert oa.tolist() == [3, -1] and ob.tolist() == [9, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
