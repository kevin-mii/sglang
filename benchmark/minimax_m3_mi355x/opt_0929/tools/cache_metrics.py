"""cache_metrics.py RUN_DIR...: prefix-cache / KV-pressure totals over AIPerf's profiling window."""
import json, sys
def agg(m, name):
    s = m.get(name)
    if not s: return None
    ser = [x for x in s["series"] if x["labels"].get("tp_rank", "0") == "0"]
    if s["type"] == "counter":
        return sum(x["stats"].get("total", 0.0) for x in ser)
    st = [x["stats"] for x in ser]
    return {k: max(x.get(k, 0) for x in st) for k in ("avg", "max") if any(k in x for x in st)}
def by_label(m, name, label):
    s = m.get(name) or {"series": []}
    out = {}
    for x in s["series"]:
        if x["labels"].get("tp_rank", "0") == "0":
            k = x["labels"].get(label, "?"); out[k] = out.get(k, 0.0) + x["stats"].get("total", 0.0)
    return out
for d in sys.argv[1:]:
    m = json.load(open(f"{d}/server_metrics_export.json"))["metrics"]
    prompt = agg(m, "sglang:prompt_tokens"); cached = by_label(m, "sglang:cached_tokens", "cache_source")
    tot_cached = sum(cached.values()); evicted = agg(m, "sglang:evicted_tokens")
    used = agg(m, "sglang:kv_used_tokens"); cap = agg(m, "sglang:max_total_num_tokens")
    extra = {k: agg(m, k) for k in m if "hicache" in k and m[k]["type"] == "counter"}
    print(f"{d.split('/')[-1]}: prompt {prompt/1e6:.2f}M, cached {tot_cached/1e6:.2f}M {({k: round(v/1e6, 2) for k, v in cached.items()})}, "
          f"recomputed {(prompt - tot_cached)/1e6:.2f}M ({100*(prompt-tot_cached)/max(prompt,1):.1f}%), evicted {evicted/1e6:.2f}M, "
          f"KV used avg/max {used['avg']/1e6:.2f}/{used['max']/1e6:.2f}M of {cap['max']/1e6:.2f}M"
          + (f", hicache {({k.split(':')[1]: round(v/1e6, 2) for k, v in extra.items()})}" if extra else ""))
