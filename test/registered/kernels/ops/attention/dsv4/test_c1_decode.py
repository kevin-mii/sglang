"""Check ratio-1 latents against the float64 RMSNorm within the bf16 boundary band and the cache writes byte-for-byte."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm, rope_tail
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
BATCHES = (1, 8, 128)

ROPE_DIM = 64
# Slots per page of the *compressed* pool, i.e. the FULL page size over the
# ratio -- which at ratio 1 is the FULL page size itself.
PAGE_SIZE = 128
SLOT_BYTES = 584
PAGE_BYTES = -(-SLOT_BYTES * PAGE_SIZE // 576) * 576

# within this many bf16 ulps of a rounding boundary (eight fp32 ulps) fp32 arithmetic
# may land on either side
MIDPOINT_SLACK = 8 * 2**-16


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit torch RMSNorm: `RMSNorm.forward` may switch kernels with the batch size."""
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


def _inputs(n, dim, seed, *, positions=None, out_loc=None):
    """int64 positions, int32 out_loc; `test_out_loc_int64` covers the int64 out_loc."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_input = torch.randn(n, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    if positions is None:
        positions = torch.arange(3, n + 3, device="cuda", dtype=torch.int64)
    if out_loc is None:
        # Never 0, so no row is taken for padding and each gets its own slot.
        out_loc = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    return kv_input, positions, out_loc


def _freqs(max_pos, seed):
    """`layer.freqs_cis` and the fp32 real/imag-interleaved view the kernel
    indexes itself, at `positions` -- no `- 1` at ratio 1."""
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
    """Torch RoPE and fake-quant, then the production writer for the 584-byte layout."""
    fq4 = fake_quant_compressed_kv(rope_tail(latent, freqs, ROPE_DIM))
    fused_store_cache(
        input=fq4, cache=cache, indices=slots, page_size=PAGE_SIZE, type="flashmla"
    )


def _run(n, dim, seed, *, sentinel=None, **kw):
    """One `c1_decode_norm_rope_store` call, plus everything a gate needs."""
    kv_input, positions, out_loc = _inputs(n, dim, seed, **kw)
    norm = _norm(dim, seed + 1)
    freqs, freqs_cis = _freqs(int(positions.max()) + 2 if n else 2, seed + 2)
    slots = out_loc.to(torch.int64)
    cache = _cache(int(slots.max()) if n else 0)

    out = None
    if sentinel is not None:
        out = torch.full((n, dim), sentinel, device="cuda", dtype=torch.bfloat16)
    got = c1_decode_norm_rope_store(
        kv_input,
        norm.weight.data,
        positions,
        out_loc,
        EPS,
        freqs_cis,
        cache,
        page_size=PAGE_SIZE,
        out=out,
    )
    return dict(
        got=got,
        kv_input=kv_input,
        norm=norm,
        cache=cache,
        slots=slots,
        out_loc=out_loc,
        # `out_loc == 0` is the padded-row marker, and the only one: at ratio 1
        # the compressed slot is the FULL slot, so there is no -1 sentinel.
        live=out_loc > 0,
        # Per row, the way `_low_ratio_write_group` gathers `freqs_cis[group_pos]`.
        freqs=freqs[positions],
    )


# ------------------------------------------------------------------- the store


@pytest.mark.parametrize("n", BATCHES)
def test_store_is_bitwise_the_production_writer(n):
    """Byte for byte, driven by the kernel's own latent so only the RoPE tail, the fp4
    fake-quant and the 584-byte layout are under test."""
    r = _run(n, HEAD_DIM, seed=1000 + n)
    live = r["live"]
    ref_cache = torch.zeros_like(r["cache"])
    _torch_store(r["got"][live], r["freqs"][live], ref_cache, r["slots"][live])
    assert torch.equal(r["cache"], ref_cache), (
        f"{n=}: {int((r['cache'] != ref_cache).sum())} of "
        f"{r['cache'].numel()} cache bytes differ from the production writer"
    )


@pytest.mark.parametrize("n", BATCHES)
def test_store_accepts_the_pool_fp8_view(n):
    """`get_extra_key_buffer` hands the pool out as `float8_e4m3fn`; the kernel must
    write the same bytes through that view as through the uint8 buffer."""
    seed = 5000 + n
    kv_input, positions, out_loc = _inputs(n, HEAD_DIM, seed)
    norm = _norm(HEAD_DIM, seed + 1)
    _, freqs_cis = _freqs(int(positions.max()) + 2 if n else 2, seed + 2)
    slots_max = int(out_loc.max()) if n else 0
    cache_u8, cache_fp8 = _cache(slots_max), _cache(slots_max)
    args = (kv_input, norm.weight.data, positions, out_loc, EPS, freqs_cis)
    got_u8 = c1_decode_norm_rope_store(*args, cache_u8, page_size=PAGE_SIZE)
    got_fp8 = c1_decode_norm_rope_store(
        *args, cache_fp8.view(torch.float8_e4m3fn), page_size=PAGE_SIZE
    )
    assert torch.equal(got_u8, got_fp8), "latent differs by cache dtype"
    assert torch.equal(cache_u8, cache_fp8), "cache bytes differ by cache dtype"


@pytest.mark.parametrize("n", BATCHES)
def test_out_loc_int64(n):
    """The fused decode path passes the scheduler's int64 `out_cache_loc`; either
    width must give the same latent and cache bytes."""
    seed = 6000 + n
    kv_input, positions, out_loc = _inputs(n, HEAD_DIM, seed)
    norm = _norm(HEAD_DIM, seed + 1)
    _, freqs_cis = _freqs(int(positions.max()) + 2 if n else 2, seed + 2)
    slots_max = int(out_loc.max()) if n else 0
    cache32, cache64 = _cache(slots_max), _cache(slots_max)
    got32 = c1_decode_norm_rope_store(
        kv_input,
        norm.weight.data,
        positions,
        out_loc,
        EPS,
        freqs_cis,
        cache32,
        page_size=PAGE_SIZE,
    )
    got64 = c1_decode_norm_rope_store(
        kv_input,
        norm.weight.data,
        positions,
        out_loc.to(torch.int64),
        EPS,
        freqs_cis,
        cache64,
        page_size=PAGE_SIZE,
    )
    assert torch.equal(got32, got64), "latent differs by loc dtype"
    assert torch.equal(cache32, cache64), "cache bytes differ by loc dtype"


def test_store_slot_is_out_loc():
    """At ratio 1 the compressed slot is `out_loc` itself; scattered across a page
    boundary, exactly those slots and no others are written."""
    want = [
        1,
        PAGE_SIZE - 1,
        PAGE_SIZE,
        PAGE_SIZE + 1,
        2 * PAGE_SIZE - 1,
        2 * PAGE_SIZE,
    ]
    out_loc = torch.tensor(want, device="cuda", dtype=torch.int32)
    r = _run(len(want), HEAD_DIM, seed=3000, out_loc=out_loc)
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


def test_padded_rows_publish_nothing():
    """Padded `out_loc == 0` rows write no slot, slot 0 included; the caller discards
    their latents."""
    n = 8
    positions = torch.tensor([2, 3, 4, 5, 6, 0, 0, 0], device="cuda", dtype=torch.int64)
    out_loc = torch.tensor(
        [7, 9, 11, 13, 15, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    r = _run(
        n, HEAD_DIM, seed=4000, positions=positions, out_loc=out_loc, sentinel=-7.5
    )
    live = r["live"]
    assert live.any() and not live.all(), "test needs both live and padded rows"

    ref_cache = torch.zeros_like(r["cache"])
    _torch_store(r["got"][live], r["freqs"][live], ref_cache, r["slots"][live])
    assert torch.equal(r["cache"], ref_cache), (
        f"{int((r['cache'] != ref_cache).sum())} cache bytes differ -- a padded "
        "row reached the cache, or a live row did not"
    )
    assert not r["cache"][0, : 576 + 8].any(), "a padded row wrote compressed slot 0"
    assert (r["got"][~live] != -7.5).any(), (
        "a padded row skipped its latent; it is expected to publish one"
    )


def test_empty_batch():
    """An idle decode step launches nothing and must not fault."""
    r = _run(0, HEAD_DIM, seed=6000)
    assert r["got"].shape == (0, 512)
    assert not r["cache"].any()


# ------------------------------------------------------------------ the latent


def _exact_norm(kv_input, weight):
    """`finish` in float64, where the bf16 result no longer depends on the sum order."""
    x = kv_input.double()
    scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return weight.double() * x * scale


def _bf16_boundary(exact):
    """`(ulp, margin)`: the bf16 ulp at each exact value, and how far that value
    sits from the nearest bf16 rounding boundary."""
    rounded = exact.to(torch.bfloat16).double()
    _, exponent = torch.frexp(rounded)
    # frexp puts the mantissa in [0.5, 1), so the exponent is one above the
    # IEEE one; bf16 keeps 7 mantissa bits, hence `- 8` rather than `- 7`.
    ulp = torch.ldexp(torch.ones_like(rounded), exponent - 8)
    return ulp, ulp / 2 - (exact - rounded).abs()


def _latent_gate(r, ctx):
    got, kv_input, norm = r["got"], r["kv_input"], r["norm"]
    if got.numel() == 0:
        return
    expected = _torch_rmsnorm(kv_input, norm.weight.data, EPS)
    exact = _exact_norm(kv_input, norm.weight.data)
    ulp, margin = _bf16_boundary(exact)
    resolvable = margin > MIDPOINT_SLACK * ulp

    # Bitwise against torch on everything fp32 can decide.
    differs = got.view(torch.int16) != expected.view(torch.int16)
    bad = differs & resolvable
    assert not bad.any(), (
        f"{ctx}: {int(bad.sum())} latent elements differ from the torch "
        f"RMSNorm away from a rounding boundary; worst "
        f"{(got.double() - expected.double()).abs().max().item():.3e}"
    )
    # And correctly rounded off the float64 value everywhere, boundary included.
    error = (got.double() - exact).abs()
    bound = ulp * (0.5 + MIDPOINT_SLACK)
    assert (error <= bound).all(), (
        f"{ctx}: latent is not the correctly-rounded bf16 of the exact norm at "
        f"{int((error > bound).sum())} elements"
    )


@pytest.mark.parametrize("n", BATCHES)
def test_latent_is_the_torch_norm(n):
    """The pre-RoPE latent is `finish`, and the index-K branch's `wk` projection
    reads it, so it is a published output and not an intermediate."""
    _latent_gate(_run(n, HEAD_DIM, seed=7000 + n), f"{n=}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
