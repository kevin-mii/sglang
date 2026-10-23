#!/usr/bin/env python3
"""mla-fp8 alone at the K3 shapes that matter (aiter fp8 in ms for reference, from bench_mla_prefill_3way)."""
import statistics, sys
import torch
import mla_fp8
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import bench_moonmath_mla_prefill as b
AITER = {"first 8K chunk": 0.329, "8K chunk at 32K": 1.472, "last 8K chunk of 50K": 2.330, "agentic 1.9K on 50K": 0.677,
         "4 agentic turns": 2.819, "extend 512 on 50K": 0.228, "extend 128 on 50K": 0.125, "no prefix 2K": 0.092,
         "1.9K on 16K": 0.270}
SHAPES = [("first 8K chunk", [(8192, 8192)]), ("8K chunk at 32K", [(8192, 32768)]), ("last 8K chunk of 50K", [(8192, 50000)]),
          ("agentic 1.9K on 50K", [(1900, 51900)]), ("4 agentic turns", [(1900, 51900)] * 4),
          ("extend 512 on 50K", [(512, 50512)]), ("extend 128 on 50K", [(128, 50128)]), ("no prefix 2K", [(2048, 2048)]),
          ("1.9K on 16K", [(1900, 17900)])]
torch.manual_seed(0)
tot_a = tot_f = 0
for label, reqs in SHAPES:
    tq, tk = sum(x for x, _ in reqs), sum(y for _, y in reqs)
    q = torch.randn(tq, 12, 192, dtype=torch.bfloat16, device="cuda")
    k8 = torch.randn(tk, 12, 192, device="cuda").to(torch.float8_e4m3fn); v8 = torch.randn(tk, 12, 128, device="cuda").to(torch.float8_e4m3fn)
    cq, ck = b.cu([x for x, _ in reqs]), b.cu([y for _, y in reqs])
    fp = mla_fp8.VarlenPlan(); fp.plan(cq.cuda(), ck.cuda(), max(x for x, _ in reqs), max(y for _, y in reqs), tq, tk, 12, 12, True, cq, ck)
    iters = 20 if tq * tk < 2e8 else 5
    t = statistics.median(sum((b.time_ms(lambda: fp.run(q, k8, v8, b.SCALE), iters, 1) for _ in range(5)), []))
    pairs = sum(nq * (nk - nq) + nq * (nq + 1) // 2 for nq, nk in reqs)
    print(f"{label:24s} {t:7.3f} ms  vs aiter {AITER[label]:.3f}  ({AITER[label] / t:.2f}x)  {pairs * 640 * 12 / t / 1e9:6.0f} TFLOP/s", flush=True)
    fp.close()
