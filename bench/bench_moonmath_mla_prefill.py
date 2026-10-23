#!/usr/bin/env python3
"""moonmath-ai/amd-kernels' gfx950 MLA prefill vs the aiter path Kimi-K3 runs today.

The two kernels:
- moonmath: bf16 attention with Q/K head dim 192 and V head dim 128.
- aiter, as Kimi-K3's aiter backend runs it on gfx950:
  - the 12 heads per TP8 rank are tiled to 16;
  - Q is cast to fp8 (K/V arrive as fp8 from the prefix pack);
  - then mla_prefill_ps_asm_fwd and mla_reduce_v1.

Both attend causally, bottom-right aligned: the keys are the whole sequence and the new tokens sit at its tail, as in
the server's extend. Each shape is checked against an fp32 reference (the last 256 query rows of each
request) before it is timed.

Timing:
- CUDA events, median over interleaved passes.
- The aiter column covers the server's whole per-layer cost: Q pad, cast, kernel, reduce and head slice.
- Plans (PS metadata, VarlenPlan) are built once per shape, as the server does once per forward.

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=<amd-kernels checkout> python bench/bench_moonmath_mla_prefill.py
"""

import statistics
import sys

import moonmath_amd as ma
import torch
from aiter import (
    get_ps_metadata_info_v1,
    get_ps_metadata_v1,
    mla_prefill_ps_asm_fwd,
    mla_reduce_v1,
)

H, HP, DQK, DV = 12, 16, 192, 128  # K3 heads per TP8 rank, aiter's padded head count, head dims
TILE_Q = 256
FP8 = torch.float8_e4m3fn
SCALE = DQK**-0.5
DEV = "cuda"

# (label, [(new tokens, whole sequence length) per request])
SHAPES = [
    ("prefill, first 8K chunk", [(8192, 8192)]),
    ("prefill, 8K chunk at 32K", [(8192, 32768)]),
    ("prefill, last 8K chunk of 50K", [(8192, 50000)]),
    ("agentic turn: 1.9K new on 50K", [(1900, 51900)]),
    ("4 agentic turns batched", [(1900, 51900)] * 4),
    ("extend 512 on 50K", [(512, 50512)]),
    ("extend 128 on 50K", [(128, 50128)]),
    ("no prefix, 2K", [(2048, 2048)]),
]


def cu(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


class AiterPrefill:
    def __init__(self, reqs):
        q_lens = [a for a, _ in reqs]
        kv_lens = [b for _, b in reqs]
        self.qo_cpu, self.kv_cpu = cu(q_lens), cu(kv_lens)
        self.max_q = max(q_lens)
        (
            (wm_n, wm_t),
            (wi_n, wi_t),
            (winfo_n, winfo_t),
            (ri_n, ri_t),
            (rf_n, rf_t),
            (rp_n, rp_t),
        ) = get_ps_metadata_info_v1(
            batch_size=len(reqs),
            num_head_k=HP,
            max_qlen=self.max_q,
            qlen_granularity=TILE_Q,
        )
        self.work_meta = torch.empty(wm_n, dtype=wm_t, device=DEV)
        self.work_indptr = torch.empty(wi_n, dtype=wi_t, device=DEV)
        self.work_info = torch.empty(winfo_n, dtype=winfo_t, device=DEV)
        self.reduce_indptr = torch.empty(ri_n, dtype=ri_t, device=DEV)
        self.reduce_final = torch.empty(rf_n, dtype=rf_t, device=DEV)
        self.reduce_partial = torch.empty(rp_n, dtype=rp_t, device=DEV)
        get_ps_metadata_v1(
            self.qo_cpu,
            self.kv_cpu,
            torch.tensor(kv_lens, dtype=torch.int32),
            1,
            HP,
            self.work_meta,
            self.work_indptr,
            self.work_info,
            self.reduce_indptr,
            self.reduce_final,
            self.reduce_partial,
            qhead_granularity=1,
            qlen_granularity=TILE_Q,
            kvlen_granularity=128,
            block_size=1,
            is_causal=True,
            need_lse=False,
        )
        self.num_partial = self.reduce_partial.size(0)  # the server's static bound
        self.qo = self.qo_cpu.to(DEV)
        self.kv = self.kv_cpu.to(DEV)
        self.kv_indices = torch.arange(int(self.kv_cpu[-1]), dtype=torch.int32, device=DEV)
        self.one = torch.ones((), dtype=torch.float32, device=DEV)

    def run(self, q, k8, v8):
        """q: bf16 [Tq, H, 192]; k8 / v8: fp8 [Tk, HP, *], already head-tiled (as the prefix pack emits them)."""
        reps = -(-HP // H)
        qp = q.repeat(1, reps, 1)[:, :HP, :].contiguous().to(FP8)
        rows = max(self.num_partial, 1) * TILE_Q
        logits = torch.empty((rows, HP, DV), dtype=torch.float32, device=DEV)
        lse = torch.empty((rows, HP), dtype=torch.float32, device=DEV)
        out = torch.empty((q.shape[0], HP, DV), dtype=torch.bfloat16, device=DEV)
        mla_prefill_ps_asm_fwd(
            qp, k8, v8, self.qo, self.kv, self.kv_indices, self.work_indptr, self.work_info,
            self.max_q, SCALE, True, logits, lse, out, self.one, self.one, self.one,
        )
        mla_reduce_v1(
            logits, lse, self.reduce_indptr, self.reduce_final, self.reduce_partial, TILE_Q, 0, out, None
        )
        return out[:, :H]


def reference(q, k, v, reqs, rows=256):
    """fp32 causal attention, bottom-right aligned per request, for the last `rows` query rows of each request
    (they see the most keys). Returns (reference, the global row indices it covers)."""
    outs, sel, qo, ko = [], [], 0, 0
    for nq, nk in reqs:
        r0 = max(nq - rows, 0)
        qs = q[qo + r0 : qo + nq].float().transpose(0, 1)
        ks = k[ko : ko + nk].float().transpose(0, 1)
        vs = v[ko : ko + nk].float().transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * SCALE
        pos = torch.arange(r0, nq, device=DEV)[:, None] + nk - nq
        s.masked_fill_(torch.arange(nk, device=DEV)[None, :] > pos, float("-inf"))
        outs.append((s.softmax(-1) @ vs).transpose(0, 1))
        sel.append(torch.arange(qo + r0, qo + nq, device=DEV))
        qo, ko = qo + nq, ko + nk
    return torch.cat(outs), torch.cat(sel)


def time_ms(fn, iters, passes):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    per_pass = []
    for _ in range(passes):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        per_pass.append(a.elapsed_time(b) / iters)
    return per_pass


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}  heads/rank {H} (aiter tiles to {HP}), Q/K {DQK}, V {DV}")
    print("| Shape | aiter fp8 (ms) | moonmath bf16 (ms) | moonmath / aiter | x24 MLA layers, aiter -> moonmath (ms) "
          "| rel err aiter / moonmath |")
    print("|---|---|---|---|---|---|")
    for label, reqs in SHAPES:
        tq, tk = sum(a for a, _ in reqs), sum(b for _, b in reqs)
        q = torch.randn(tq, H, DQK, dtype=torch.bfloat16, device=DEV)
        k = torch.randn(tk, H, DQK, dtype=torch.bfloat16, device=DEV)
        v = torch.randn(tk, H, DV, dtype=torch.bfloat16, device=DEV)
        reps = -(-HP // H)
        k8 = k.repeat(1, reps, 1)[:, :HP, :].contiguous().to(FP8)
        v8 = v.repeat(1, reps, 1)[:, :HP, :].contiguous().to(FP8)

        ap = AiterPrefill(reqs)
        mp = ma.VarlenPlan().plan(
            cu([a for a, _ in reqs]).to(DEV), cu([b for _, b in reqs]).to(DEV),
            max(a for a, _ in reqs), max(b for _, b in reqs), tq, tk, heads=H, kv_heads=H,
            head_dims=(DQK, DV), causal=True,
            host_cu_seqlens_q=cu([a for a, _ in reqs]), host_cu_seqlens_k=cu([b for _, b in reqs]),
        )

        ref, rows = reference(q, k, v, reqs)
        o_a = ap.run(q, k8, v8)[rows].float()
        o_m = mp.run(q, k, v, softmax_scale=SCALE)[rows].float()
        scale = ref.abs().max()
        err_a = ((o_a - ref).abs().max() / scale).item()
        err_m = ((o_m - ref).abs().max() / scale).item()
        if not (err_a < 0.1 and err_m < 0.05):
            print(f"{label}: accuracy check failed (aiter {err_a:.2e}, moonmath {err_m:.2e})", file=sys.stderr)
        del ref, o_a, o_m

        iters = 20 if tq * tk < 2e8 else 5
        a_runs, m_runs = [], []
        for _ in range(5):  # interleave the two kernels
            a_runs += time_ms(lambda: ap.run(q, k8, v8), iters, 1)
            m_runs += time_ms(lambda: mp.run(q, k, v, softmax_scale=SCALE), iters, 1)
        ta, tm = statistics.median(a_runs), statistics.median(m_runs)
        print(f"| {label} | {ta:.3f} | {tm:.3f} | {tm / ta:.2f}x | {24 * ta:.1f} -> {24 * tm:.1f} "
              f"| {err_a:.1e} / {err_m:.1e} |", flush=True)
        mp.close()
        del q, k, v, k8, v8, ap, mp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
