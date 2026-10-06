"""kernel_breakdown.py TRACE_DIR [TP_RANK]: GPU time per kernel category in a sglang torch-profiler trace.

Prints each category's share of summed kernel time and the top kernels, so optimization ceilings
(indexer, sparse attention, MoE, ...) can be read off one steady-state profile.
"""
import collections, gzip, json, os, re, sys

CATS = [  # first match wins
    ("indexer", r"index|topk|top_k|radix|_score|histogram|select"),
    ("sparse_attn", r"flash_decode|sparse|_fwd_kernel|paged_attention|pa_decode|decode_attention|verify|extend_attention|mha|fmha|attn"),
    ("moe", r"moe|fused_moe|expert|mixed_moe|sorting|mxfp4|swiglu|silu"),
    ("allreduce", r"allreduce|all_reduce|cross_device|reduce_scatter|all_gather|nccl|quick_reduce|qr_"),
    ("norm_rope", r"rmsnorm|rms_norm|layernorm|rope|rotary|qknorm"),
    ("gemm", r"gemm|Cijk|matmul|gemv|bpreshuffle|wvSplit|mm_|splitk"),
    ("kv_store", r"kvcache|kv_cache|store|set_kv|copy_all_layer"),
    ("sampling", r"sampl|argmax|softmax|logit"),
]

d = sys.argv[1]; rank = sys.argv[2] if len(sys.argv) > 2 else "0"
f = [x for x in os.listdir(d) if f"TP-{rank}." in x or f"TP-{rank}-" in x][0]
t = json.load(gzip.open(os.path.join(d, f)))
k = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_op")]
tot = collections.Counter(); per = collections.Counter(); cat_of = {}
for e in k:
    n = e["name"]; per[n] += e["dur"]
    c = cat_of.get(n)
    if c is None:
        c = next((c for c, p in CATS if re.search(p, n, re.I)), "other"); cat_of[n] = c
    tot[c] += e["dur"]
s = sum(tot.values())
span = (max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k))
print(f"{f}: {len(k)} kernels, busy {s/1e3:.1f} ms over a {span/1e3:.1f} ms span ({100*s/span:.0f}% busy)")
for c, v in tot.most_common():
    print(f"  {c:12s} {v/1e3:9.2f} ms  {100*v/s:5.1f}%")
print("top kernels:")
for n, v in per.most_common(25):
    print(f"  {v/1e3:8.2f} ms {100*v/s:5.1f}%  [{cat_of[n]}] {n[:110]}")
