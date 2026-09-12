"""Check ratio-2 pooling within two bf16 ulps and the pair state and cache stores bitwise."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.c2 import (
    c2_decode_norm,
    c2_decode_norm_rope_store,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    RMSNorm,
    rope_tail,
)
from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_compressed_kv
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or (is_hip() and not is_gfx95_supported()),
    reason="requires a GPU; on ROCm the byte-for-byte e4m3fn cache oracle needs gfx950"
    " (gfx942's hardware fp8 pack encodes E4M3FNUZ)",
)

EPS = 1e-6
# The 584-byte FlashMLA layout fixes head_dim at 512:
# 448 fp8 nope values plus 64 bf16 RoPE values.
HEAD_DIM = 512
BATCHES = (1, 8, 64)
# `CompressStatePool.ring_size`, positions per request slot in the pair state.
# Two values, because the addressing is modular and a wrong wrap only shows on
# one of them: the served size is `next_power_of_2(draft + 1)`.
RING_SIZES = (2, 8)

ROPE_DIM = 64
RATIO = 2
# Slots per page of the *compressed* pool, i.e. the FULL page size over the
# ratio. 128 // 2 as served.
PAGE_SIZE = 64
SLOT_BYTES = 584
PAGE_BYTES = -(-SLOT_BYTES * PAGE_SIZE // 576) * 576

# Two bf16 ulp: the intermediate bf16 cast in `finish` makes one ulp reachable and two
# the ceiling. The floor only engages on elements the pair pooling cancelled to near zero.
POOL_RTOL = 2**-6
POOL_ATOL = 2**-20


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Use explicit torch RMSNorm so the oracle stays independent of fused kernels;
    RMSNorm.forward can change implementation with batch size.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def _norm(dim: int, seed: int) -> RMSNorm:
    """`DeepseekV41Compressor.norm` as the model holds it: bf16 weight, because
    the parameter is created inside `set_default_torch_dtype(model dtype)`."""
    with set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(dim, EPS).cuda()
    assert norm.weight.dtype == torch.bfloat16
    # Not `ones`: a constant weight cannot catch a wrong per-element index.
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, generator=g, device="cuda"))
    return norm


def _inputs(
    n,
    dim,
    seed,
    *,
    positions=None,
    req=None,
    raw_out_loc=None,
    num_state_rows=None,
    ring_size=RING_SIZES[-1],
):
    """Inputs exercise distinct partner rows and alternating parity without padding;
    a separate case covers the scheduler's int64 output locations.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_state_rows = num_state_rows if num_state_rows is not None else n + 2
    kv_input = torch.randn(n, 2 * dim, generator=g, device="cuda", dtype=torch.float32)
    kv_state = torch.randn(
        num_state_rows * ring_size,
        2 * dim,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    if positions is None:
        positions = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    if req is None:
        req = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)
    if raw_out_loc is None:
        # Odd and never 0, so no row is taken for padding and `>> 1` gives each
        # row its own compressed slot.
        raw_out_loc = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 3
    return kv_input, kv_state, positions, req, raw_out_loc


def _state_rows(req, positions, ring_size):
    """`CompressStatePool.translate_from_req_position_to_state_loc`, for the
    slot a row reads (`pos - 1`) and the one it writes (`pos`)."""
    base = req.to(torch.int64) * ring_size
    pos = positions.to(torch.int64)
    return base + (pos - 1) % ring_size, base + pos % ring_size


def _torch_reference(kv_input, kv_state, norm, positions, req, raw_out_loc, ring_size):
    """The ratio-2 decode branch and `finish`, transcribed. Updates `kv_state`
    in place like the kernel does; returns `(latent, completes)`."""
    dim = kv_input.shape[1] // 2
    kv, score = kv_input[:, :dim], kv_input[:, dim:]
    state_kv, state_score = kv_state[:, :dim], kv_state[:, dim:]
    odd = (positions.to(torch.int64) % 2) == 1
    read, write = _state_rows(req, positions, ring_size)
    # The ring separates the two: a completing row reads what `pos - 1` left, a
    # pending one writes its own slot, so there is no read-modify-write of one
    # row and no ordering to respect between them.
    partner_kv, partner_score = state_kv[read], state_score[read]
    # Only an even *live* row writes. Scattering the others as no-ops would not
    # be equivalent: padded rows share a `req_pool_idx` with a live request, and
    # a duplicated index makes the scatter pick a winner rather than skip.
    parks = (~odd) & (raw_out_loc != 0)
    w = write[parks]
    assert w.unique().numel() == w.numel(), "a decode step parks once per request"
    state_kv[w] = kv[parks]
    state_score[w] = score[parks]
    pooled = DeepseekV41Compressor.pool_pairs(
        torch.stack([partner_kv, kv], dim=1),
        torch.stack([partner_score, score], dim=1),
    )
    return _torch_rmsnorm(pooled.to(torch.bfloat16), norm.weight.data, EPS), odd


def _compare(got, expected, rows, ctx):
    got, expected = got.float()[rows], expected.float()[rows]
    if got.numel() == 0:
        return
    diff = (got - expected).abs()
    worst = (diff / (POOL_ATOL + POOL_RTOL * expected.abs())).max().item()
    assert worst <= 1.0, (
        f"{ctx}: out of tolerance at {worst:.3f}x the bound "
        f"(rtol={POOL_RTOL}, atol={POOL_ATOL}); "
        f"max|diff| {diff.max().item():.3e} on |ref| up to "
        f"{expected.abs().max().item():.3e}"
    )


def _run(n, dim, seed, *, ring_size=RING_SIZES[-1], **kw):
    """One `c2_decode_norm` call against the reference, returning the two pair states
    so callers can add their own assertions."""
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(
        n, dim, seed, ring_size=ring_size, **kw
    )
    norm = _norm(dim, seed + 1)
    ref_state, got_state = kv_state.clone(), kv_state.clone()

    expected, odd = _torch_reference(
        kv_input, ref_state, norm, positions, req, raw_out_loc, ring_size
    )
    got = c2_decode_norm(
        kv_input,
        got_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        ring_size=ring_size,
    )
    # A padded row still computes and still publishes its latent -- the caller
    # discards that row -- so the tolerance gate covers the live ones.
    live = odd & (raw_out_loc != 0)
    _compare(got, expected, live, f"{n=} {dim=} {seed=}")
    return got, expected, odd, got_state, ref_state


# ---------------------------------------------------------------- pool + norm


@pytest.mark.parametrize("ring_size", RING_SIZES)
@pytest.mark.parametrize("n", BATCHES)
def test_mixed_parity(n, ring_size):
    """Both parities in one batch: the odd rows complete a group against the
    state, the even rows park themselves in it. Both ring sizes, because the
    slot arithmetic is modular and a wrong wrap shows on only one of them."""
    *_, got_state, ref_state = _run(n, HEAD_DIM, seed=1000 + n, ring_size=ring_size)
    # Pure copy on the even rows, untouched on the odd ones -- no arithmetic, so
    # nothing here is allowed to differ by even one bit.
    assert torch.equal(got_state, ref_state), "pair state diverged"


@pytest.mark.parametrize("n", BATCHES)
def test_raw_out_loc_int64(n):
    """`raw_out_loc` arrives as the scheduler's int64 `out_cache_loc` in the
    served model (the unit tests above hand int32). The kernel indexes either
    width and must produce the same bytes: same latent, same pair state."""
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(
        n, HEAD_DIM, seed=7000 + n
    )
    norm = _norm(HEAD_DIM, 7001 + n)
    state32, state64 = kv_state.clone(), kv_state.clone()
    args = (norm.weight.data, positions, req)
    kw = {"ring_size": RING_SIZES[-1]}
    got32 = c2_decode_norm(kv_input, state32, *args, raw_out_loc, EPS, **kw)
    got64 = c2_decode_norm(
        kv_input, state64, *args, raw_out_loc.to(torch.int64), EPS, **kw
    )
    odd = (positions.to(torch.int64) % 2) == 1
    assert torch.equal(got32[odd], got64[odd]), "latent differs by loc dtype"
    assert torch.equal(state32, state64), "pair state differs by loc dtype"


def test_pair_state_carried_across_two_steps():
    """The load-bearing property: a group spans two decode steps. Step one is
    all-even, so every row only parks; step two is all-odd and must pool
    against exactly what step one left behind."""
    n = 8
    g = torch.Generator(device="cuda").manual_seed(3000)
    first = torch.randn(
        n, 2 * HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
    )
    second = torch.randn(
        n, 2 * HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
    )
    kv_state = torch.randn(
        n * RING_SIZES[-1],
        2 * HEAD_DIM,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    # Rotated by one so the carry cannot be confused with `req == row`.
    req = ((torch.arange(n, device="cuda") + 1) % n).to(torch.int64)
    raw_out_loc = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 3
    even = torch.full((n,), 4, device="cuda", dtype=torch.int32)
    odd = even + 1
    norm = _norm(HEAD_DIM, 3001)

    ref_state, got_state = kv_state.clone(), kv_state.clone()
    ring = RING_SIZES[-1]
    _torch_reference(first, ref_state, norm, even, req, raw_out_loc, ring)
    expected, mask = _torch_reference(
        second, ref_state, norm, odd, req, raw_out_loc, ring
    )

    args = (norm.weight.data,)
    kw = {"ring_size": ring}
    c2_decode_norm(first, got_state, *args, even, req, raw_out_loc, EPS, **kw)
    got = c2_decode_norm(second, got_state, *args, odd, req, raw_out_loc, EPS, **kw)

    assert mask.all(), "step two must be all-odd"
    _compare(got, expected, mask, "two-step")
    assert torch.equal(got_state, ref_state), "pair state diverged across steps"
    # A step-two row must actually depend on step one: pooling against the
    # original state instead would give a different answer.
    stale = c2_decode_norm(
        second, kv_state.clone(), *args, odd, req, raw_out_loc, EPS, **kw
    )
    assert not torch.equal(got, stale), "step two ignored what step one parked"


def test_padded_rows_publish_nothing():
    """Graph-padding rows with raw_out_loc == 0 must not write cache or state,
    even when their req_pool_idx aliases a live request.
    """
    n = 8
    num_state_rows = 5
    req = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], device="cuda", dtype=torch.int64)
    # Both parities among the padded rows: an even one would park in the state,
    # an odd one would consume it, and neither may happen.
    positions = torch.tensor([2, 3, 4, 5, 6, 0, 1, 0], device="cuda", dtype=torch.int32)
    raw_out_loc = torch.tensor(
        [7, 9, 11, 13, 15, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    got, expected, odd, got_state, ref_state = _run(
        n,
        HEAD_DIM,
        seed=5000,
        positions=positions,
        req=req,
        raw_out_loc=raw_out_loc,
        num_state_rows=num_state_rows,
    )
    assert torch.equal(got_state, ref_state), "a padded row wrote the pair state"
    live = odd & (raw_out_loc != 0)
    assert live.any(), "test needs a live completing row"
    _compare(got, expected, live, "padded")


def test_saturated_and_tied_scores():
    """The closed form is `exp(-|s0 - s1|)`: a tie must give exactly 0.5/0.5,
    and a large gap must underflow to a clean one-sided pick, not a NaN."""
    n = 6
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(n, HEAD_DIM, seed=6000)
    positions = torch.arange(1, 2 * n + 1, 2, device="cuda", dtype=torch.int32)
    ring = RING_SIZES[-1]
    # Row 0 ties with its partner; the odd rows sit 200 above theirs and rows
    # 2 and 4 sit 200 below, so `exp(-|delta|)` underflows in fp32 from either
    # side. The partner is the state row the kernel reads, not row `req`.
    read, _ = _state_rows(req, positions, ring)
    kv_input[:, HEAD_DIM:] = 0.0
    kv_state[:, HEAD_DIM:] = 0.0
    kv_input[1::2, HEAD_DIM:] = 200.0
    kv_state[read[2::2], HEAD_DIM:] = 200.0
    norm = _norm(HEAD_DIM, 6001)
    ref_state, got_state = kv_state.clone(), kv_state.clone()

    expected, odd = _torch_reference(
        kv_input, ref_state, norm, positions, req, raw_out_loc, ring
    )
    got = c2_decode_norm(
        kv_input,
        got_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        ring_size=ring,
    )
    assert odd.all() and torch.isfinite(got.float()).all()
    _compare(got, expected, odd, "saturated")
    # The saturated rows are one-sided picks: the odd rows keep their own kv,
    # rows 2 and 4 take their partner's.
    own = _torch_rmsnorm(
        kv_input[:, :HEAD_DIM].to(torch.bfloat16), norm.weight.data, EPS
    )
    partner = _torch_rmsnorm(
        kv_state[read, :HEAD_DIM].to(torch.bfloat16), norm.weight.data, EPS
    )
    rows = torch.zeros(n, dtype=torch.bool, device="cuda")
    rows[1::2] = True
    _compare(got, own, rows, "saturated own")
    rows = torch.zeros(n, dtype=torch.bool, device="cuda")
    rows[2::2] = True
    _compare(got, partner, rows, "saturated partner")


def test_empty_batch():
    """An idle decode step launches nothing and must not fault."""
    got, *_ = _run(0, HEAD_DIM, seed=7000)
    assert got.shape == (0, 512)


# ------------------------------------------------- + RoPE, fp4 quant and store


def _freqs(max_pos, seed):
    """`layer.freqs_cis` and the fp32 real/imag-interleaved view the kernel
    indexes itself, at `positions - 1`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ang = torch.randn(max_pos, ROPE_DIM // 2, generator=g, device="cuda")
    freqs = torch.polar(torch.ones_like(ang), ang)
    return freqs, torch.view_as_real(freqs).flatten(-2).contiguous().float()


def _cache(max_slot):
    """A compressed-pool buffer wide enough for `max_slot`, zeroed so an
    untouched slot is recognizable."""
    return torch.zeros(
        max_slot // PAGE_SIZE + 2, PAGE_BYTES, dtype=torch.uint8, device="cuda"
    )


def _torch_store(latent, freqs, cache, slots):
    """Use torch arithmetic as an independent RoPE/fake-quant oracle;
    use the production writer for the shared 584-byte FlashMLA layout.
    """
    fq4 = fake_quant_compressed_kv(rope_tail(latent, freqs, ROPE_DIM))
    fused_store_cache(
        input=fq4, cache=cache, indices=slots, page_size=PAGE_SIZE, type="flashmla"
    )


def _run_fusion(n, dim, seed, *, ring_size=RING_SIZES[-1], **kw):
    """One `c2_decode_norm_rope_store` call, plus everything the store gates need."""
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(
        n, dim, seed, ring_size=ring_size, **kw
    )
    norm = _norm(dim, seed + 1)
    freqs, freqs_cis = _freqs(int(positions.max().item()) + 2 if n else 2, seed + 2)
    slots = (raw_out_loc // RATIO).to(torch.int64)
    cache = _cache(int(slots.max().item()) if n else 0)

    odd = (positions.to(torch.int64) % 2) == 1
    got = c2_decode_norm_rope_store(
        kv_input,
        kv_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        freqs_cis,
        cache,
        page_size=PAGE_SIZE,
        ring_size=ring_size,
    )
    live = odd & (raw_out_loc != 0)
    return dict(
        got=got,
        live=live,
        cache=cache,
        slots=slots,
        # Per row, the way `_low_ratio_write_group` gathers `freqs_cis[group_pos]`.
        # The kernel indexes `positions - 1` itself; only completing rows are
        # compared, and those are all at an odd position, so the clamp is dead.
        freqs=freqs[(positions.to(torch.int64) - 1).clamp_min(0)],
    )


@pytest.mark.parametrize("n", BATCHES)
def test_store_accepts_the_pool_fp8_view(n):
    """`get_extra_key_buffer` hands the compressed pool out viewed as
    `float8_e4m3fn`, not as the uint8 it was allocated as. The kernel must take
    that view and write the same bytes it writes through the uint8 buffer."""
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(
        n, HEAD_DIM, seed=8000 + n
    )
    norm = _norm(HEAD_DIM, 8001 + n)
    _, freqs_cis = _freqs(int(positions.max().item()) + 2 if n else 2, 8002 + n)
    slots_max = int((raw_out_loc // RATIO).max().item()) if n else 0
    cache_u8, cache_fp8 = _cache(slots_max), _cache(slots_max)
    args = (norm.weight.data, positions, req, raw_out_loc, EPS, freqs_cis)
    got_u8 = c2_decode_norm_rope_store(
        kv_input,
        kv_state.clone(),
        *args,
        cache_u8,
        page_size=PAGE_SIZE,
        ring_size=RING_SIZES[-1],
    )
    got_fp8 = c2_decode_norm_rope_store(
        kv_input,
        kv_state.clone(),
        *args,
        cache_fp8.view(torch.float8_e4m3fn),
        page_size=PAGE_SIZE,
        ring_size=RING_SIZES[-1],
    )
    odd = (positions.to(torch.int64) % 2) == 1
    assert torch.equal(got_u8[odd], got_fp8[odd]), "latent differs by cache dtype"
    assert torch.equal(cache_u8, cache_fp8), "cache bytes differ by cache dtype"


@pytest.mark.parametrize("n", BATCHES)
def test_store_is_bitwise_the_production_writer(n):
    """The hard gate on the write half, with the pooling residual factored out:
    the reference is driven by the kernel's *own* latent, so the only thing
    under test is RoPE, the fp4 fake-quant and the 584-byte layout. No
    tolerance -- every byte of every slot, and every byte outside them."""
    r = _run_fusion(n, HEAD_DIM, seed=10000 + n)
    live = r["live"]
    ref_cache = torch.zeros_like(r["cache"])
    if live.any():
        _torch_store(r["got"][live], r["freqs"][live], ref_cache, r["slots"][live])
    assert torch.equal(r["cache"], ref_cache), (
        f"{n=}: {int((r['cache'] != ref_cache).sum())} of "
        f"{r['cache'].numel()} cache bytes differ from the production writer"
    )


def test_store_skips_even_and_padded_rows():
    """Nothing but a live completing row may reach the cache. Every row here is
    either at an even position or padded, so the buffer must come back exactly
    as it went in -- including compressed slot 0, which is what `raw_out_loc == 0`
    would land on if the pad check were missing."""
    n = 8
    positions = torch.tensor([0, 2, 4, 6, 8, 1, 3, 5], device="cuda", dtype=torch.int32)
    # The live rows are all even; the padded ones cover both parities.
    raw_out_loc = torch.tensor(
        [3, 5, 7, 9, 11, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    r = _run_fusion(
        n,
        HEAD_DIM,
        seed=12000,
        positions=positions,
        raw_out_loc=raw_out_loc,
        req=torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], device="cuda", dtype=torch.int64),
        num_state_rows=5,
    )
    assert not r["live"].any(), "test needs no live completing row"
    assert not r["cache"].any(), (
        f"{int((r['cache'] != 0).sum())} cache bytes written with nothing to store"
    )


def test_store_slot_is_raw_out_loc_over_ratio():
    """The kernel derives its slot in-kernel as `raw_out_loc >> 1` rather than
    reading `c2_out_loc`. Scattered across a page boundary and beyond, the
    written slots must be exactly that set and nothing else."""
    n = 6
    want = [
        1,
        PAGE_SIZE - 1,
        PAGE_SIZE,
        PAGE_SIZE + 1,
        2 * PAGE_SIZE - 1,
        2 * PAGE_SIZE,
    ]
    # Odd, so `>> 1` also proves the shift is a floor and not a round.
    raw_out_loc = torch.tensor(
        [s * RATIO + 1 for s in want], device="cuda", dtype=torch.int32
    )
    positions = torch.arange(1, 2 * n + 1, 2, device="cuda", dtype=torch.int32)
    r = _run_fusion(
        n, HEAD_DIM, seed=13000, positions=positions, raw_out_loc=raw_out_loc
    )
    assert r["live"].all()
    written = set()
    cache = r["cache"]
    for p in range(cache.shape[0]):
        for s in range(PAGE_SIZE):
            value = cache[p, s * 576 : (s + 1) * 576]
            base = 576 * PAGE_SIZE + s * 8
            if value.any() or cache[p, base : base + 8].any():
                written.add(p * PAGE_SIZE + s)
    assert written == set(want), f"wrote slots {sorted(written)}, wanted {want}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
