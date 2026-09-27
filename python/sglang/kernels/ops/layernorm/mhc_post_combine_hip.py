"""Fused mHC post and combine for the DSpark draft attention boundary."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_combine_hip_kernel(
    X,
    R,
    P,
    C,
    A,
    RO,
    Y,
    H: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    mask = h < H
    x = tl.load(X + row * H + h, mask, 0).to(tl.float32)
    r0 = tl.load(R + (row * 4 + 0) * H + h, mask, 0).to(tl.float32)
    r1 = tl.load(R + (row * 4 + 1) * H + h, mask, 0).to(tl.float32)
    r2 = tl.load(R + (row * 4 + 2) * H + h, mask, 0).to(tl.float32)
    r3 = tl.load(R + (row * 4 + 3) * H + h, mask, 0).to(tl.float32)

    collapsed = tl.full((B,), 0, tl.float32)
    for j in tl.static_range(4):
        post = tl.load(P + row * 4 + j)
        c0 = tl.load(C + row * 16 + 0 * 4 + j)
        c1 = tl.load(C + row * 16 + 1 * 4 + j)
        c2 = tl.load(C + row * 16 + 2 * 4 + j)
        c3 = tl.load(C + row * 16 + 3 * 4 + j)
        pre = tl.load(A + row * 4 + j)
        mixed = tl.fma(post, x, c0 * r0)
        mixed = tl.fma(c1, r1, mixed)
        mixed = tl.fma(c2, r2, mixed)
        mixed = tl.fma(c3, r3, mixed)
        rounded = mixed.to(tl.bfloat16)
        tl.store(RO + (row * 4 + j) * H + h, rounded, mask)
        collapsed = collapsed + pre * rounded.to(tl.float32)
    tl.store(Y + row * H + h, collapsed, mask)


def mhc_post_combine_hip(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
):
    """Return ``(updated residual, combined input)`` in one launch."""
    m = x.shape[0]
    assert x.shape == (m, 5120)
    assert residual.shape == (m, 4, 5120)
    assert post.shape == pre.shape == (m, 4)
    assert comb.shape == (m, 4, 4)
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == pre.dtype == torch.float32
    assert all(t.is_contiguous() for t in (x, residual, post, comb, pre))

    updated = torch.empty_like(residual)
    combined = torch.empty_like(x)
    if m == 0:
        return updated, combined

    block = 1024
    _mhc_post_combine_hip_kernel[(m, triton.cdiv(5120, block))](
        x,
        residual,
        post,
        comb,
        pre,
        updated,
        combined,
        H=5120,
        B=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return updated, combined
