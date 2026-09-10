"""The fused FlyDSL split-layout index-K writer must match the ROCm pool's unfused writer byte for byte."""

import os
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fp4_rope import (  # noqa: E402
    BATCHES,
    EPS,
    INDEX_PAGE_SIZE,
    RATIOS,
    ROPE_DIM,
    _inputs,
    _norm,
    _torch_rmsnorm,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="the split index-K layout is the ROCm pool's"
)


def _run_split(n, ratio, seed, *, exact_sum=True, pad=0):
    """Run the split-layout entry against the pool's unfused writer into separate payload / scale buffers."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        store_fp4_index_k_cache_split,
    )
    from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
        index_k_norm_rope_pack_store_split,
    )
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4

    x, positions, loc, freqs = _inputs(n, ratio, seed, exact_sum=exact_sum, pad=pad)
    norm = _norm(seed + 1)
    freqs_cis = torch.view_as_real(freqs).flatten(-2).contiguous().float()

    npages = ((int(loc.max()) if n else 0) // INDEX_PAGE_SIZE) + 2
    got_payload = torch.zeros(
        npages, 1, 4, INDEX_PAGE_SIZE, 16, dtype=torch.uint8, device="cuda"
    )
    got_scale = torch.zeros(
        npages, 1, 4, INDEX_PAGE_SIZE, dtype=torch.uint8, device="cuda"
    )
    ref_payload, ref_scale = torch.zeros_like(got_payload), torch.zeros_like(got_scale)

    index_k_norm_rope_pack_store_split(
        x,
        norm.weight.data,
        EPS,
        freqs_cis,
        positions,
        loc,
        got_payload.view(torch.float4_e2m1fn_x2),
        got_scale,
        ratio=ratio,
    )
    live = loc > 0
    if int(live.sum()):
        store_fp4_index_k_cache_split(
            _rope_fq4(
                _torch_rmsnorm(x[live], norm.weight.data, EPS),
                freqs[(positions & ~(ratio - 1))[live]],
                ROPE_DIM,
            ),
            ref_payload.view(torch.float4_e2m1fn_x2),
            ref_scale,
            loc[live],
            page_size=INDEX_PAGE_SIZE,
            rne=True,
        )
    return (got_payload, got_scale), (ref_payload, ref_scale), loc


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("n", BATCHES)
@pytest.mark.parametrize("pad", (0, 3))
def test_split_layout_matches_the_hip_writer_exactly(n, ratio, pad):
    """Every split-layout byte must equal the pool's unfused writer, and padded rows must publish nothing."""
    if pad > n:
        pytest.skip("more padded rows than rows")
    got, ref, loc = _run_split(n, ratio, seed=3000 + n * 7 + ratio, pad=pad)
    for what, g, r in zip(("payload", "scale"), got, ref):
        assert torch.equal(g, r), (
            f"{n=} {ratio=} {pad=}: {int((g != r).sum())} of {g.numel()} {what} "
            f"bytes differ from the split-layout writer"
        )
