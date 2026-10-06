"""graph_tune.py SHAPES_JSON CAPTURE_BS [NDT] [MAX_ADD]: padding report and greedy capture-size additions.

SHAPES_JSON: a SGLANG_GRAPH_SHAPE_STATS histogram. CAPTURE_BS: comma list of captured batch sizes (requests).
Rows per request: 1 for DECODE / DRAFT, NDT (draft tokens, default 4) for TARGET_VERIFY / DRAFT_EXTEND.
Prints padding = (captured - live) / captured over all replays per (runner, mode), then greedily adds the batch
sizes that remove the most replay-weighted padded rows.
"""
import bisect, collections, json, sys

rows = json.load(open(sys.argv[1]))
base = sorted(int(x) for x in sys.argv[2].split(","))
ndt = int(sys.argv[3]) if len(sys.argv) > 3 else 4
max_add = int(sys.argv[4]) if len(sys.argv) > 4 else 8
width = {"DECODE": 1, "DRAFT": 1, "TARGET_VERIFY": ndt, "DRAFT_EXTEND": ndt}


def cost(hist, caps, w):
    live = sum(n * r for r, n in hist.items())
    cap = sum(n * caps[bisect.bisect_left(caps, -(-r // w))] * w for r, n in hist.items())
    return live, cap


groups = collections.defaultdict(collections.Counter)
for r in rows:
    groups[(r["runner"], r["mode"])][r["live_rows"]] += r["replays"]

for (runner, mode), hist in sorted(groups.items()):
    w = width.get(mode, 1)
    total = sum(hist.values())
    live, cap = cost(hist, base, w)
    print(f"{runner}/{mode}: {total} replays, {w} rows/request, padding {100 * (cap - live) / cap:.2f}% "
          f"({(cap - live) / total:.2f} padded rows per replay)")
    reqs = collections.Counter()
    for r, n in hist.items():
        reqs[-(-r // w)] += n
    print("   most frequent live batch sizes:", ", ".join(f"{b}:{n}" for b, n in reqs.most_common(12)))
    caps, added = list(base), []
    for _ in range(max_add):
        best = None
        for b in sorted(reqs):
            if b in caps:
                continue
            _, c2 = cost(hist, sorted(caps + [b]), w)
            if best is None or c2 < best[1]:
                best = (b, c2)
        if best is None or best[1] >= cap:
            break
        b, new_cap = best
        print(f"   + capture bs {b:3d}: padding -> {100 * (new_cap - live) / new_cap:.2f}% "
              f"(-{(cap - new_cap) / total:.2f} rows/replay)")
        cap, caps = new_cap, sorted(caps + [b]); added.append(b)
    print(f"   suggested additions: {sorted(added)}")
