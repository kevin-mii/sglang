# Agent prompt: MiniMax-M3 MXFP4 on 8x MI355X (paste into a fresh session on the new node)

You are continuing performance work on MiniMax-M3 (MXFP4) serving with SGLang on a node with 8x AMD MI355X.

**Read first:** `benchmark/minimax_m3_mi355x/rebase_0923/HANDOFF.md` on branch `M3-perf-rebase` of https://github.com/kevin-mii/sglang.
It has the full context, setup commands, first results, measured TTFT analysis, and the operational rules. Also skim
`benchmark/minimax_m3_mi355x/OPTIMIZATIONS.md` (every knob tried on M3-perf) and `docs/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx`.

## Workload (fixed; do not change it)
- Model `amd/MiniMax-M3-MXFP4` with the EAGLE3 draft `Inferact/MiniMax-M3-EAGLE3-GQA` (MXFP4 + spec decode are required).
- ISL 74,176 tokens with a 90% shared cached prefix (the prefix is warmed first), OSL 650 (ignore_eos), GSM8K-text prompts.
  Closed loop at concurrency c = 64, 80, 128 with 5 x c requests each. All 8 GPUs; per-GPU numbers divide by 8.
- Client: `rebase_0923/m3_prefix_bench.py` (prints the report table).

## Targets
- **TTFT p50 < 3 s** at every concurrency (currently 4.9 / 6.6 / 8.5 s at c = 64 / 80 / 128).
- **>= 2.5K output tok/s total (8 GPUs) at c = 128.** Already met: 3.5K (439/GPU). Keep it while fixing TTFT.
- Quality must hold: GSM8K-500 5-shot in 0.85-0.89 on any config you report.

## Steps
1. **Node health first.** Follow HANDOFF "Reproduce on a clean node" (docker image by digest, branch checkout at `/sgl-workspace/sglang`,
   the aiter patch plus tuned CSV, weights). Then run the eviction check from HANDOFF "Operational rules" while one replica starts.
   If `evicted_ms` grows by seconds per 10 s, or `/health` takes more than about 10 min, stop and report: the node is bad.
2. **Reproduce the baseline table** (2 x TP4 + cache-aware router, default `serve_m3.sh`). It should land near
   c64 4.9 s / 246, c80 6.6 s / 353, c128 8.5 s / 439 (TTFT p50 / out tok/s/GPU). Run GSM8K-500.
3. **Cut TTFT.** It is dominated by serialized prefill of synchronized arrival waves (213 ms per 7.4K-token prefill, one per 8K chunk).
   Work through the lever list in HANDOFF "Where TTFT goes": larger chunk plus max-prefill-tokens, MAXRUN 64, fairness reserve,
   then prefill/decode disaggregation. Use `burst_ttft.py` for fast A/B before running the full table. Change one knob at a time.
4. For each config you keep, report the full table (same format) plus GSM8K-500, the exact `up.sh` knobs, and the git commit.
   Commit scripts and results under `rebase_0923/results/` on `M3-perf-rebase`, and update HANDOFF.md.

## Rules that matter
- Start replicas **one at a time** (`up.sh` blocks until healthy). Concurrent 243 GB loads collapse H2D bandwidth.
- **Never SIGKILL a live GPU server.** Use `stop_port.sh PORT`, then wait until `/sys/class/kfd/kfd/proc/` is empty and no sglang process
  is in `D` state before the next start. Don't `pkill -f` a pattern contained in your own command line.
- Keep `TORCHINDUCTOR_COMPILE_THREADS=1` (set in `serve_m3.sh`). If prefill-graph capture is too slow on a healthy node, try the default
  and watch for `/dev/kfd` inheritance by the inductor workers.
- `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4` is the benchmark setting. It corrupts long-context tool-call output (see `endpoint/ENDPOINT.md`),
  so say so next to any number you report with it.
- Don't push to other branches or open PRs without asking the user (kevin.mi@radixark.ai).
