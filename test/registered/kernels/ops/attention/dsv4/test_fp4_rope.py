"""Check the fused index-key writes and index-query packing bytewise against the Triton path."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.fp4_rope import (
    INDEX_PAGE_SIZE,
    SLOT_BYTES,
    index_k_norm_rope_pack_store,
    index_q_rope_pack,
    index_q_rope_pack_weights,
)
from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
    rope_fake_quant_pack_indexer,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a GPU",
)

# `index_head_dim` and `qk_rope_head_dim` as served. The one-warp tiling is
# specific to this pair and the kernel asserts it.
HEAD_DIM = 128
ROPE_DIM = 64
EPS = 1e-6
BATCHES = (1, 31, 64, 65)
# Both low ratios reach this kernel; the group position is masked out of the
# token position in-kernel, so the ratio changes what it reads.
RATIOS = (1, 2)


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Use explicit torch RMSNorm so the oracle stays independent of fused kernels;
    RMSNorm.forward can change implementation with batch size.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def _norm(seed: int, weight_scale: float = 1.0) -> RMSNorm:
    """`DeepseekV41Indexer.k_norm` as the model holds it: bf16 weight, because
    the parameter is created inside `set_default_torch_dtype(model dtype)`."""
    with set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(HEAD_DIM, EPS).cuda()
    assert norm.weight.dtype == torch.bfloat16
    # Not `ones`: a constant weight cannot catch a wrong per-element index.
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(
            torch.randn(HEAD_DIM, generator=g, device="cuda") * weight_scale
        )
    return norm


def _inputs(n, ratio, seed, *, pad=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Magnitudes from {0.5, 1, 2}: the squares are exact multiples of 0.25
    # and 128 of them sum to at most 512, so every partial sum is exact in
    # fp32 whatever the order and both norms agree bitwise.
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


def _run(n, ratio, seed, *, pad=0, weight_scale=1.0):
    """One kernel call and the Triton reference, into separate buffers."""
    x, positions, loc, freqs = _inputs(n, ratio, seed, pad=pad)
    norm = _norm(seed + 1, weight_scale)
    freqs_cis = torch.view_as_real(freqs).flatten(-2).contiguous().float()

    npages = ((int(loc.max()) if n else 0) // INDEX_PAGE_SIZE) + 2
    got = torch.zeros(
        npages, INDEX_PAGE_SIZE * SLOT_BYTES, dtype=torch.uint8, device="cuda"
    )
    ref = torch.zeros_like(got)

    index_k_norm_rope_pack_store(
        x, norm.weight.data, EPS, freqs_cis, positions, loc, got, ratio=ratio
    )
    # The reference writes every row it is given, so hand it only the rows the
    # kernel publishes; `test_padded_rows_publish_nothing` covers the rest.
    live = loc > 0
    if int(live.sum()):
        rope_fake_quant_pack_indexer(
            _torch_rmsnorm(x[live], norm.weight.data, EPS),
            freqs[(positions & ~(ratio - 1))[live]],
            ROPE_DIM,
            cache=ref,
            loc=loc[live],
        )
    return got, ref, loc


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("n", BATCHES)
def test_matches_the_triton_writer_exactly(n, ratio):
    """No tolerance: every byte of the buffer, on input whose norm statistic is
    order-independent. This is the gate on the write half -- both quantization
    stages, the nibble order, the block-exponent bytes and the slot address."""
    got, ref, _ = _run(n, ratio, seed=1000 + n * 7 + ratio)
    assert torch.equal(got, ref), (
        f"{n=} {ratio=}: {int((got != ref).sum())} of {got.numel()} cache bytes "
        f"differ from the Triton writer"
    )


# RMSNorm divides the overall input scale out, so the magnitude the quantizers
# see is set by `k_norm.weight`. These three place the block absmax on either
# side of both amax floors: 1.0 clears them, 1e-5 puts `amax / 6` under the
# packer's 1e-4 (so its floor decides the exponent while the fake-quant's does
# not), and 1e-38 reaches the fake-quant's own `6 * 2**-126`.
WEIGHT_SCALES = (1.0, 1e-5, 1e-38)


@pytest.mark.parametrize("weight_scale", WEIGHT_SCALES)
@pytest.mark.parametrize("ratio", RATIOS)
def test_matches_across_the_amax_floors(ratio, weight_scale):
    """The two quantization stages take their amax floors on opposite sides of
    the divide by 6 -- `max(amax, 6 * 2**-126) * (1 / 6)` against
    `max(amax / 6, 1e-4)` -- so they are genuinely different scales and neither
    stage can be dropped. Nothing separates them until a block's absmax lands
    below `6e-4`, which is what the small weights here arrange."""
    got, ref, _ = _run(64, ratio, seed=8000 + ratio, weight_scale=weight_scale)
    assert got.any(), "the input degenerated to an all-zero payload"
    assert torch.equal(got, ref), (
        f"{ratio=} {weight_scale=:g}: {int((got != ref).sum())} of {got.numel()} "
        f"cache bytes differ from the Triton writer"
    )


@pytest.mark.parametrize("ratio", RATIOS)
def test_padded_rows_publish_nothing(ratio):
    """A padded graph row carries `loc == 0`, the reserved dummy slot, and must
    leave the cache alone -- slot 0 of page 0 included, which is exactly where
    it would land if the check were missing."""
    n, pad = 16, 5
    got, ref, loc = _run(ratio=ratio, n=n, seed=3000 + ratio, pad=pad)
    assert int((loc == 0).sum()) == pad
    assert torch.equal(got, ref), "a padded row reached the cache"
    slot0 = got[0, : SLOT_BYTES // 2]
    assert not slot0.any(), f"{int(slot0.count_nonzero())} bytes written to slot 0"


@pytest.mark.parametrize("ratio", RATIOS)
def test_slot_is_loc_across_a_page_boundary(ratio):
    """The slot address is `loc` split by the index page size, which is 64 and
    not the FULL page size. Scattered over a boundary, only those slots move."""
    want = [
        1,
        INDEX_PAGE_SIZE - 1,
        INDEX_PAGE_SIZE,
        INDEX_PAGE_SIZE + 1,
        2 * INDEX_PAGE_SIZE,
    ]
    n = len(want)
    x, positions, _, freqs = _inputs(n, ratio, seed=5000 + ratio)
    loc = torch.tensor(want, device="cuda", dtype=torch.int64)
    norm = _norm(5001 + ratio)
    npages = want[-1] // INDEX_PAGE_SIZE + 2
    cache = torch.zeros(
        npages, INDEX_PAGE_SIZE * SLOT_BYTES, dtype=torch.uint8, device="cuda"
    )
    index_k_norm_rope_pack_store(
        x,
        norm.weight.data,
        EPS,
        torch.view_as_real(freqs).flatten(-2).contiguous().float(),
        positions,
        loc,
        cache,
        ratio=ratio,
    )
    payload = INDEX_PAGE_SIZE * (HEAD_DIM // 2)
    written = set()
    for p in range(npages):
        for s in range(INDEX_PAGE_SIZE):
            value = cache[p, s * (HEAD_DIM // 2) : (s + 1) * (HEAD_DIM // 2)]
            scale = cache[p, payload + s * 4 : payload + s * 4 + 4]
            if value.any() or scale.any():
                written.add(p * INDEX_PAGE_SIZE + s)
    assert written == set(want), f"wrote slots {sorted(written)}, wanted {want}"


def test_positions_select_the_group_freqs():
    """The group position is masked out of the token position in-kernel. At
    ratio 2 a token at an odd position must rotate by the *even* one, so the
    same row fed as ratio 1 at that even position must produce the same bytes."""
    n = 8
    x, _, loc, freqs = _inputs(n, 2, seed=6000)
    norm = _norm(6001)
    freqs_cis = torch.view_as_real(freqs).flatten(-2).contiguous().float()
    odd = torch.arange(n, device="cuda", dtype=torch.int64) * 2 + 1
    even = odd - 1

    out = []
    for positions, ratio in ((odd, 2), (even, 1)):
        cache = torch.zeros(
            2, INDEX_PAGE_SIZE * SLOT_BYTES, dtype=torch.uint8, device="cuda"
        )
        index_k_norm_rope_pack_store(
            x, norm.weight.data, EPS, freqs_cis, positions, loc, cache, ratio=ratio
        )
        out.append(cache)
    assert out[0].any()
    assert torch.equal(out[0], out[1]), "ratio 2 did not mask the position down"


def test_empty_batch():
    """An idle decode step launches nothing and must not fault."""
    got, ref, _ = _run(0, 2, seed=7000)
    assert torch.equal(got, ref)


# `config.index_n_heads` is 64, the served count. 1 is the degenerate row split,
# where a row *is* a token and a wrong `row / heads` cannot show.
Q_HEADS = (1, 2, 8, 64)
# The Q path has no norm weight to scale, so the input carries the magnitude.
# Below 6e-4 a block absmax puts `amax / 6` under the packer's 1e-4 floor, so
# that floor -- and not the fake-quant's `6 * 2**-126` -- sets the exponent;
# 1e-38 reaches the fake-quant's own floor as well.
Q_INPUT_SCALES = (1.0, 1e-3, 1e-4, 1e-5, 1e-38)
# `_ceil_ue8m0_exp(1e-4)`, the exponent the packer's floor pins a block to.
PACKER_FLOOR_EXPONENT = 114
# Positions are drawn from this many rows of the freqs table.
Q_MAX_POS = 512


def _q_inputs(n, heads, seed, *, input_scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn(n, heads, HEAD_DIM, generator=g, device="cuda") * input_scale
    ).bfloat16()
    # Unsorted and repeating, so a row/token mix-up cannot cancel out.
    positions = torch.randint(
        Q_MAX_POS, (n,), generator=g, device="cuda", dtype=torch.int64
    )
    ang = torch.randn(Q_MAX_POS, ROPE_DIM // 2, generator=g, device="cuda")
    return x, positions, torch.polar(torch.ones_like(ang), ang)


def _run_q(n, heads, seed, *, input_scale=1.0, pos_dtype=torch.int64):
    """One kernel call and the Triton reference in its plain (no-norm, no-cache)
    mode -- the mode `_low_ratio_index_topk_decode` calls."""
    x, positions, freqs = _q_inputs(n, heads, seed, input_scale=input_scale)
    positions = positions.to(pos_dtype)
    freqs_cis = torch.view_as_real(freqs).flatten(-2).contiguous().float()
    got = index_q_rope_pack(x, freqs_cis, positions)
    ref = rope_fake_quant_pack_indexer(x, freqs, ROPE_DIM, positions=positions)
    return got, ref


def _assert_q_equal(got, ref, what):
    payload, scale = got
    ref_payload, ref_scale = ref
    assert torch.equal(payload, ref_payload), (
        f"{what}: {int((payload != ref_payload).sum())} of {payload.numel()} "
        f"payload bytes differ from the Triton packer"
    )
    assert torch.equal(scale, ref_scale), (
        f"{what}: {int((scale != ref_scale).sum())} of {scale.numel()} scale "
        f"words differ from the Triton packer"
    )


def _block_exponents(scale):
    """[rows] packed scale words -> [rows, 4] ue8m0 exponents."""
    shifts = torch.arange(4, device=scale.device) * 8
    return (scale.to(torch.int64)[:, None] >> shifts) & 0xFF


@pytest.mark.parametrize("heads", Q_HEADS)
@pytest.mark.parametrize("n", BATCHES)
def test_q_matches_the_triton_packer_exactly(n, heads):
    """No tolerance, on arbitrary random input: the RoPE tail, both quantization
    stages, the nibble order and the packed scale word."""
    got, ref = _run_q(n, heads, seed=10000 + n * 13 + heads)
    assert got[0].any(), "the input degenerated to an all-zero payload"
    _assert_q_equal(got, ref, f"{n=} {heads=}")


@pytest.mark.parametrize("heads", (1, Q_HEADS[-1]))
@pytest.mark.parametrize("input_scale", Q_INPUT_SCALES)
def test_q_matches_across_the_amax_floors(input_scale, heads):
    """The two quantization stages take their amax floors on opposite sides of
    the divide by 6 -- `max(amax, 6 * 2**-126) * (1 / 6)` against
    `max(amax / 6, 1e-4)` -- so they are genuinely different scales and neither
    stage can be dropped. Nothing separates them until a block's absmax lands
    below 6e-4, which is what the small input scales here arrange."""
    got, ref = _run_q(64, heads, seed=11000 + heads, input_scale=input_scale)
    _assert_q_equal(got, ref, f"{input_scale=:g} {heads=}")
    # Pin the premise, so a case cannot silently stop exercising the floor.
    exponents = _block_exponents(got[1])
    if input_scale <= 1e-4:
        assert bool((exponents == PACKER_FLOOR_EXPONENT).all()), (
            f"{input_scale=:g} was meant to put every block absmax under 6e-4, "
            f"but the exponents span {int(exponents.min())}..{int(exponents.max())}"
        )
    else:
        assert bool((exponents > PACKER_FLOOR_EXPONENT).all()), (
            f"{input_scale=:g} was meant to clear the packer's floor, but some "
            f"exponent reached {int(exponents.min())}"
        )


# The served indexer: 64 heads of 128, softmax scale 128^-0.5, times heads^-0.5.
Q_WEIGHT_SCALE = 128**-0.5 * 64**-0.5


@pytest.mark.parametrize("heads", Q_HEADS)
@pytest.mark.parametrize("n", BATCHES)
def test_q_weights_epilogue_is_bitwise_head_weights(n, heads):
    """`index_q_rope_pack_weights` must return the packed query of
    `index_q_rope_pack` unchanged and, for the weights, exactly what the torch
    chain `(w * scale).float()` produces on the raw `weights_proj` output: an
    fp32 multiply by the fp32-rounded scalar, round-to-nearest-even to bf16,
    widened to fp32. Random bf16 weights over several magnitudes so the bf16
    rounding is exercised at every exponent."""
    x, positions, freqs = _q_inputs(n, heads, seed=20000 + n * 7 + heads)
    freqs_cis = torch.view_as_real(freqs).flatten(-2).contiguous().float()
    g = torch.Generator(device="cuda").manual_seed(20001 + n * 7 + heads)
    raw = (torch.randn(n, heads, generator=g, device="cuda") * 8.0).bfloat16()
    payload, scale, weights = index_q_rope_pack_weights(
        x, freqs_cis, positions, raw, Q_WEIGHT_SCALE
    )
    _assert_q_equal(
        (payload, scale), index_q_rope_pack(x, freqs_cis, positions), f"{n=} {heads=}"
    )
    expected = (raw * Q_WEIGHT_SCALE).float()
    assert weights.dtype == torch.float32 and weights.shape == (n, heads)
    assert torch.equal(weights, expected), (
        f"{n=} {heads=}: {int((weights != expected).sum())} of {weights.numel()} "
        f"head weights differ from (w * scale).float()"
    )


def test_q_accepts_int32_positions():
    """The kernel is instantiated for both position dtypes."""
    _assert_q_equal(*_run_q(8, 64, seed=12000, pos_dtype=torch.int32), "int32 pos")


def test_q_empty_batch():
    """An idle decode step launches nothing and must not fault."""
    _assert_q_equal(*_run_q(0, 64, seed=14000), "empty batch")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
