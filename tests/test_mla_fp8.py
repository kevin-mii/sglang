"""mla_fp8 against an fp32 reference over the same fp8 K/V (dequantized) and bf16 Q.

Errors are max |out - ref| / max |ref| per call. The fp8 kernel casts Q and P to fp8 too, so it sits near
aiter's fp8 prefill (~6e-2 on random data), not near bf16 kernels (~3e-3).
"""

import math

import pytest
import torch

import mla_fp8

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def make(seqs, H=12, HKv=12, k_scale=1.0, v_scale=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    tq, tk = sum(a for a, _ in seqs), sum(b for _, b in seqs)
    q = torch.randn(tq, H, 192, device=DEV, generator=g).to(torch.bfloat16)
    k = (torch.randn(tk, HKv, 192, device=DEV, generator=g) / k_scale).to(FP8)
    v = (torch.randn(tk, HKv, 128, device=DEV, generator=g) / v_scale).to(FP8)
    return q, k, v


def cu(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def reference(q, k, v, seqs, scale, causal, k_scale, v_scale):
    H, HKv = q.shape[1], k.shape[1]
    outs, lses, qo, ko = [], [], 0, 0
    for nq, nk in seqs:
        qs = q[qo : qo + nq].float().transpose(0, 1)
        ks = (k[ko : ko + nk].float() * k_scale).repeat_interleave(H // HKv, 1).transpose(0, 1)
        vs = (v[ko : ko + nk].float() * v_scale).repeat_interleave(H // HKv, 1).transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * scale
        if causal:
            mask = torch.arange(nk, device=DEV)[None, :] > torch.arange(nq, device=DEV)[:, None] + nk - nq
            s.masked_fill_(mask, float("-inf"))
        lse = torch.logsumexp(s, -1)
        p = torch.softmax(s, -1).nan_to_num(0.0)
        outs.append((p @ vs).transpose(0, 1))
        lses.append(lse)
        qo, ko = qo + nq, ko + nk
    return torch.cat(outs), torch.cat(lses, dim=1)


def run(seqs, causal=True, H=12, HKv=12, k_scale=1.0, v_scale=1.0, lse=False, host=True, seed=0):
    q, k, v = make(seqs, H, HKv, k_scale, v_scale, seed)
    scale = 192**-0.5
    cq, ck = cu([a for a, _ in seqs]), cu([b for _, b in seqs])
    res = mla_fp8.fwd_varlen(
        q, k, v, cq.to(DEV), ck.to(DEV), max(a for a, _ in seqs), max(b for _, b in seqs), causal, scale,
        k_scale, v_scale, host_cu_q=cq if host else None, host_cu_k=ck if host else None, return_lse=lse)
    out, l = (res if lse else (res, None))
    ref, ref_lse = reference(q, k, v, seqs, scale, causal, k_scale, v_scale)
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    lse_err = None
    if lse:
        finite = ref_lse.isfinite()
        lse_err = (l[finite] - ref_lse[finite]).abs().max().item()
    return err, lse_err


SHAPES = [
    [(64, 64)], [(1, 1)], [(17, 300)], [(300, 300)], [(256, 1000)], [(1000, 1000)], [(128, 50128)],
    [(1900, 51900)], [(5, 3), (40, 700), (333, 333), (1, 4097)], [(512, 300)],
]


@pytest.mark.parametrize("seqs", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_accuracy(seqs, causal):
    err, _ = run(seqs, causal)
    assert err < 0.1, err


def test_scales_lse_dev():
    err, lse_err = run([(300, 2000), (64, 64)], True, k_scale=0.05, v_scale=4.0, lse=True)
    assert err < 0.1 and lse_err < 0.1, (err, lse_err)
    err, _ = run([(300, 2000), (64, 64)], True, host=False)
    assert err < 0.1, err


if __name__ == "__main__":
    for causal in (True, False):
        for seqs in SHAPES:
            err, _ = run(seqs, causal)
            print(f"causal={causal} {seqs}: rel err {err:.2e}")
    print("scales + lse:", run([(300, 2000), (64, 64)], True, k_scale=0.05, v_scale=4.0, lse=True))
    print("device cu_seqlens:", run([(300, 2000), (64, 64)], True, host=False))
