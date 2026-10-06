"""agentx_summary.py OUTDIR: one-line summary of an AIPerf AgentX run (reproduce.sh's in-window rate, per GPU at TP4)."""
import json, sys
d = sys.argv[1]; s = json.load(open(f"{d}/profile_export_aiperf.json"))
recs = [json.loads(l) for l in open(f"{d}/profile_export.jsonl")]
ok = [r for r in recs if r.get("error") is None and r["metadata"]["benchmark_phase"] == "profiling"]
err = sum(1 for r in recs if r.get("error") is not None and r["metadata"]["benchmark_phase"] == "profiling")
v = lambda r, k: r["metrics"][k]["value"] if isinstance(r["metrics"][k], dict) else r["metrics"][k]
tin = sum(v(r, "input_sequence_length") for r in ok); tout = sum(v(r, "output_sequence_length") for r in ok)
span = (max(r["metadata"]["request_end_ns"] for r in ok) - min(r["metadata"]["request_start_ns"] for r in ok)) / 1e9
p = lambda k, st: s[k][st]
row = {"dir": d, "requests": len(ok), "errors": err, "span_s": span, "total_tok_s_gpu": (tin + tout) / span / 4,
       "out_tok_s_gpu": tout / span / 4, "reported_total_tok_s_gpu": p("total_token_throughput", "avg") / 4,
       "ttft_p50_ms": p("time_to_first_token", "p50"), "ttft_p90_ms": p("time_to_first_token", "p90"),
       "itl_p50_ms": p("inter_token_latency", "p50"), "itl_p90_ms": p("inter_token_latency", "p90"),
       "interactivity_p90": 1000 / p("inter_token_latency", "p90")}
json.dump(row, open(f"{d}/summary.json", "w"), indent=1)
print(f"{d.split('/')[-1]}: total {row['total_tok_s_gpu']:,.0f} tok/s/GPU | out {row['out_tok_s_gpu']:,.0f} | TTFT p50/p90 "
      f"{row['ttft_p50_ms']:.0f}/{row['ttft_p90_ms']:.0f} ms | ITL p50/p90 {row['itl_p50_ms']:.1f}/{row['itl_p90_ms']:.1f} ms | "
      f"interactivity p90 {row['interactivity_p90']:.0f} tok/s/user | {len(ok)} req, {err} err, {span:.0f} s")
