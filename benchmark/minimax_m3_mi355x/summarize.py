import json, sys, glob, re, os
D = sys.argv[1]
f = os.path.join(D, "profile_export_aiperf.json")
if not os.path.exists(f):
    print(f"{D}: no summary json"); sys.exit(0)
j = json.load(open(f))
def g(tag, k="avg"):
    r = j.get(tag); return None if r is None else r.get(k)
tt = g("total_token_throughput"); ot = g("output_token_throughput"); it = g("input_token_throughput")
line = (f"{os.path.basename(D)}: dur={g('benchmark_duration'):.0f}s reqs={g('request_count'):.0f} "
        f"total_tok/s={tt:.0f} ({tt/4:.0f}/GPU) out_tok/s={ot:.1f} in_tok/s={it:.0f} "
        f"TTFT ms p50={g('time_to_first_token','p50'):.0f} p90={g('time_to_first_token','p90'):.0f} p99={g('time_to_first_token','p99'):.0f} | "
        f"ITL ms p50={g('inter_token_latency','p50'):.1f} p90={g('inter_token_latency','p90'):.1f} p99={g('inter_token_latency','p99'):.1f} | "
        f"E2E s p50={g('request_latency','p50')/1000:.1f} p90={g('request_latency','p90')/1000:.1f} p99={g('request_latency','p99')/1000:.1f} | "
        f"ISL avg={g('input_sequence_length'):.0f} p50={g('input_sequence_length','p50'):.0f} p90={g('input_sequence_length','p90'):.0f} max={g('input_sequence_length','max'):.0f} | "
        f"OSL avg={g('output_sequence_length'):.0f} p50={g('output_sequence_length','p50'):.0f} p90={g('output_sequence_length','p90'):.0f} | "
        f"per-user out tok/s avg={g('output_token_throughput_per_user'):.1f}")
# cache accounting from usage metrics if present
for t in ("usage_prompt_tokens", "usage_completion_tokens", "theoretical_prefix_cache_hit"):
    if j.get(t): line += f" | {t} avg={j[t].get('avg')}"
# server-side deltas
def load(p):
    d = {}
    for ln in open(p):
        m = re.match(r'^sglang:(prompt_tokens_total|generation_tokens_total|cached_tokens_total|num_requests_total|evicted_tokens_total|num_retracted_requests_total)\{[^}]*\}\s+([0-9.e+]+)', ln)
        if m: d[m.group(1)] = d.get(m.group(1), 0) + float(m.group(2))
    return d
try:
    a = load(os.path.join(D, "server_metrics_before.prom")); b = load(os.path.join(D, "server_metrics_after.prom"))
    dp = b.get("prompt_tokens_total",0)-a.get("prompt_tokens_total",0); dc = b.get("cached_tokens_total",0)-a.get("cached_tokens_total",0); dg = b.get("generation_tokens_total",0)-a.get("generation_tokens_total",0)
    line += f" | server(incl warmup): prompt={dp:.0f} cached={dc:.0f} ({(dc/dp*100) if dp else 0:.1f}%) fresh_prefill={dp-dc:.0f} gen={dg:.0f} evicted={b.get('evicted_tokens_total',0)-a.get('evicted_tokens_total',0):.0f} retracted_reqs={b.get('num_retracted_requests_total',0)-a.get('num_retracted_requests_total',0):.0f}"
except Exception as e:
    line += f" | server metrics n/a ({e})"
print(line)
