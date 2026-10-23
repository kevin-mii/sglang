#!/usr/bin/env python3
"""Kimi-K3 MLA prefill attention per layer at TP8 (12 heads, Q/K 192, V 128), three kernels:
  aiter    the server's path today: Q tiled to 16 heads + fp8 cast, mla_prefill_ps_asm_fwd, mla_reduce_v1
  mm-bf16  moonmath-ai/amd-kernels, bf16 Q/K/V
  mla-fp8  this repo: bf16 Q, fp8 K/V (12 heads, no padding)
Plans built once per shape (once per forward in the server). CUDA events, median of interleaved passes.
Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=<amd-kernels checkout>:<this repo>:<this repo>/bench python bench/bench_mla_prefill_3way.py
"""
import statistics, sys
import torch
import moonmath_amd as ma
import mla_fp8
import bench_moonmath_mla_prefill as b

SHAPES = b.SHAPES + [("no prefix, 4K", [(4096, 4096)]), ("no prefix, 16K", [(16384, 16384)]),
                     ("1.9K new on 16K", [(1900, 17900)])]

def main():
    torch.manual_seed(0)
    print("| Shape | aiter fp8 (ms) | moonmath bf16 (ms) | mla-fp8 (ms) | mla-fp8 vs aiter | rel err aiter / mm / mla-fp8 |")
    print("|---|---|---|---|---|---|")
    for label, reqs in SHAPES:
        tq, tk = sum(x for x, _ in reqs), sum(y for _, y in reqs)
        q = torch.randn(tq, b.H, b.DQK, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(tk, b.H, b.DQK, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(tk, b.H, b.DV, dtype=torch.bfloat16, device="cuda")
        k8, v8 = k.to(b.FP8), v.to(b.FP8)
        k16 = k8.repeat(1, 2, 1)[:, :b.HP].contiguous(); v16 = v8.repeat(1, 2, 1)[:, :b.HP].contiguous()
        cq, ck = b.cu([x for x, _ in reqs]), b.cu([y for _, y in reqs])
        mq, mk = max(x for x, _ in reqs), max(y for _, y in reqs)
        ap = b.AiterPrefill(reqs)
        mp = ma.VarlenPlan().plan(cq.cuda(), ck.cuda(), mq, mk, tq, tk, heads=b.H, kv_heads=b.H, head_dims=(192, 128),
                                  causal=True, host_cu_seqlens_q=cq, host_cu_seqlens_k=ck)
        fp = mla_fp8.VarlenPlan()
        fp.plan(cq.cuda(), ck.cuda(), mq, mk, tq, tk, b.H, b.H, True, cq, ck)
        ref, rows = b.reference(q, k8.to(torch.bfloat16), v8.to(torch.bfloat16), reqs)
        def err(o):
            return ((o[rows].float() - ref).abs().max() / ref.abs().max()).item()
        e_a = err(ap.run(q, k16, v16)); e_m = err(mp.run(q, k, v, softmax_scale=b.SCALE)); e_f = err(fp.run(q, k8, v8, b.SCALE))
        iters = 20 if tq * tk < 2e8 else 5
        runs = {"a": [], "m": [], "f": []}
        for _ in range(5):
            runs["a"] += b.time_ms(lambda: ap.run(q, k16, v16), iters, 1)
            runs["m"] += b.time_ms(lambda: mp.run(q, k, v, softmax_scale=b.SCALE), iters, 1)
            runs["f"] += b.time_ms(lambda: fp.run(q, k8, v8, b.SCALE), iters, 1)
        ta, tm, tf = (statistics.median(runs[x]) for x in "amf")
        print(f"| {label} | {ta:.3f} | {tm:.3f} | {tf:.3f} | {ta / tf:.2f}x | {e_a:.1e} / {e_m:.1e} / {e_f:.1e} |", flush=True)
        mp.close(); fp.close()
        del q, k, v, k8, v8, k16, v16, ap, mp, fp, ref
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
