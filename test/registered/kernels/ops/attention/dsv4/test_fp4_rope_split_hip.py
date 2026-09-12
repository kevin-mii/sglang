"""The fused FlyDSL split-layout index-K writer must match the ROCm pool's unfused writer byte for byte."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_rope import INDEX_PAGE_SIZE
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    torch.version.hip is None, reason="the split index-K layout is the ROCm pool's"
)

# `index_head_dim` and `qk_rope_head_dim` as served.
HEAD_DIM = 128
ROPE_DIM = 64
EPS = 1e-6
BATCHES = (1, 31, 64, 65)
# Both low ratios reach this kernel; the group position is masked out of the
# token position in-kernel, so the ratio changes what it reads.
RATIOS = (1, 2)


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit torch RMSNorm: `RMSNorm.forward` may switch kernels with the batch size."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def _norm(seed: int) -> RMSNorm:
    """`DeepseekV41Indexer.k_norm` as the model holds it: bf16 weight, because
    the parameter is created inside `set_default_torch_dtype(model dtype)`."""
    with set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(HEAD_DIM, EPS).cuda()
    assert norm.weight.dtype == torch.bfloat16
    # Not `ones`: a constant weight cannot catch a wrong per-element index.
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HEAD_DIM, generator=g, device="cuda"))
    return norm


def _inputs(n, ratio, seed, *, pad=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # magnitudes from {0.5, 1, 2}: the squares sum exactly in fp32 in any order, so
    # both norms agree bitwise
    mag = torch.tensor([0.5, 1.0, 2.0], device="cuda")[
        torch.randint(3, (n, HEAD_DIM), generator=g, device="cuda")
    ]
    sign = torch.where(
        torch.rand(n, HEAD_DIM, generator=g, device="cuda") < 0.5, -1.0, 1.0
    )
    x = (mag * sign).bfloat16()
    # Every token is the last of its group, so every row would publish.
    positions = torch.arange(n, device="cuda", dtype=torch.int64) * ratio + (ratio - 1)
    loc = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)
    if pad:
        # The padded suffix of a CUDA-graph batch: the reserved dummy slot.
        loc[-pad:] = 0
    ang = torch.randn(
        (int(positions.max()) + 2) if n else 2,
        ROPE_DIM // 2,
        generator=g,
        device="cuda",
    )
    freqs = torch.polar(torch.ones_like(ang), ang)
    return x, positions, loc, freqs


def _run_split(n, ratio, seed, *, pad=0):
    """Run the split-layout entry against the pool's unfused writer into separate payload / scale buffers."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        store_fp4_index_k_cache_split,
    )
    from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
        index_k_norm_rope_pack_store_split,
    )
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4

    x, positions, loc, freqs = _inputs(n, ratio, seed, pad=pad)
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
@pytest.mark.parametrize("pad", (0, 1))
def test_split_layout_matches_the_hip_writer_exactly(n, ratio, pad):
    """Every split-layout byte must equal the pool's unfused writer, and padded rows must publish nothing."""
    got, ref, loc = _run_split(n, ratio, seed=3000 + n * 7 + ratio, pad=pad)
    for what, g, r in zip(("payload", "scale"), got, ref):
        assert torch.equal(g, r), (
            f"{n=} {ratio=} {pad=}: {int((g != r).sum())} of {g.numel()} {what} "
            f"bytes differ from the split-layout writer"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
