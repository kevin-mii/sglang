# MiniMax-M3 optimization campaign: planning doc

Branch `M3-opt-0929` (from `M3-perf-rebase` `276976ba11`). Box: 8x MI350X VF (see `/scratch/m3/ENV_HANDOFF.md` for the
proot environment and the hostcall workarounds every server here needs). This file is the plan of record; the status tracker at the
bottom is updated as work lands.

---

## Part A: the request (verbatim)

# MiniMax-M3: SGLang Optimization Handoff

Hi team — could you review and integrate the two MiniMax-M3 optimization PRs below, and take ownership of developing the two additional serving optimizations described here?

## 1. Review and integrate the existing PRs

| PR | Optimization | Review focus |
|---|---|---|
| [#41488 — Indexer-only decode CP](https://github.com/sgl-project/sglang/pull/41488) | Partitions indexer context-block reads across TP ranks and exchanges compact top-k candidates. | Correctness, communication overhead, and the batch/context crossover where CP becomes beneficial. |
| [#41397 — FlyDSL paged attention + GPU work planner](https://github.com/sgl-project/sglang/pull/41397) | Integrates AITER FlyDSL paged attention and GPU planning for dense, uneven-context graph decode. Sparse calls retain static partitioning. | Numerical qualification, graph integration, and end-to-end serving performance. |

Confirm each PR's current EAGLE3 support and implement any missing compatibility before qualifying the combined **TP4 + EAGLE3** serving configuration.

Below are the two optimizatison we suggest:

## 2. Develop graph capture tuning for actual execution batches

Profile the running **decode and EAGLE3 verification batches**, then tune the capture-size list around frequently observed shapes.

- Record live request count, verification tokens per request, selected graph size, and replay frequency.
- Add capture sizes where frequent padding causes measurable overhead.
- Measure padding using the same row definition for live execution and graph capture:

  ```text
  padding = (captured rows - live rows) / captured rows
  ```

- Track capture time, graph memory usage, and remaining GPU KV-cache capacity. Additional graphs must justify any reduction in cache capacity.
- Preserve the existing fallback for unsupported shapes.

**Goal:** reduce decode and verification latency without reducing cache capacity enough to offset the gain.

## 3. Develop CPU-backed prefix caching for M3

Evaluate and extend **HiCache or an EAGLE3-compatible LMCache integration** when the reusable prefix working set exceeds GPU capacity.

- **Complete state:** offload and restore both **main K/V and index-K**, preserving consistent token/page mappings.
- **Cross-rank consistency:** reuse only the **contiguous prefix available across every required TP rank and cache component**.
- **Retention:** compare **LRU and SLRU** under the same CPU-memory budget, explicitly identifying which policy governs each tier.
- **Buffer lifetime:** protect source and destination buffers during transfers. Release references after completion and handle misses, cancellation, and errors correctly.
- **Transfer efficiency:** measure CPU-to-GPU restore time and NUMA placement against the prefill computation avoided.

**Goal:** reduce prefix recomputation and sustain higher concurrency while maintaining interactivity.

## 4. Validate each change independently, then combine

Run matched A/B tests at concurrency **24, 32, 40, and 48**, using identical agentic traces, hardware, model settings, and **real EAGLE3 acceptance**.

| Experiment | Primary measurements |
|---|---|
| Indexer CP off vs on | Indexer-chain latency including communication, selection correctness, batch/context crossover, serving throughput and interactivity |
| Existing attention vs FlyDSL + planner | Attention latency, numerical correctness, graph-replay behavior, serving throughput and interactivity |
| Default vs tuned graph sizes | Padding, decode/verify latency, capture time, graph memory, available GPU KV capacity |
| CPU tier off vs on | Recomputed prompt tokens, restored tokens, transfer time, cache hit rates |
| LRU vs SLRU | Prefix retention, recomputation, eviction behavior at equal CPU-memory budgets |
| Combined configuration | Output throughput, p90 interactivity, TTFT, correctness |

Keep concurrency fixed when measuring the benefit of CPU offload. Increasing concurrency while enabling offload is a separate operating-point comparison and does not isolate the offload benefit.

**Requested outcome:** integration of the two existing PRs, development of the two additional optimizations by the RadixArk / SGLang team, and a validated serving configuration with reproducible benchmark results.

---

## Part B: execution plan

### Fixed setup (all experiments)

- Model `amd/MiniMax-M3-MXFP4` + EAGLE3 `Inferact/MiniMax-M3-EAGLE3-GQA` (3 steps / 4 draft tokens, **real acceptance**), TP4, the
  `reproduce.sh real` server config (`rebase_0923/serve_m3.sh`) plus the node's hostcall knobs.
- Workload: the AgentX agentic traces (`inferencex-agentx-mvp`, SemiAnalysis AIPerf fork at InferenceX's pinned commit, seed 42), as in
  `OPTIMIZATIONS.md`. Concurrency 24 / 32 / 40 / 48. Same traces, same seed, same GPUs class for every A/B.
- Matched A/B: the box has 8 GPUs, so A and B run as two TP4 servers side by side (GPUs 0-3 and 4-7), then swap sides for a second
  pass to cancel any per-GPU difference. Where a change needs the whole box, run A then B on the same GPUs.
- `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ`: report every number with the value used. The AgentX configs use 4 (corrupts long-context tool
  calls, `endpoint/ENDPOINT.md`); the production endpoint uses 1. Primary A/Bs at 1; note if an optimization interacts with 4.
- Quality gate for any kept config: GSM8K-500 5-shot in 0.85-0.89, plus the long-context tool-call check if attention/indexer changes.

### Work items, in order

1. **Benchmark harness.** Set up the AIPerf AgentX client (from `reproduce.sh bench`) under `/scratch/m3`, confirm a short c=24 run
   reproduces the order of magnitude in `OPTIMIZATIONS.md`, and add a metrics sampler for per-step shapes where needed.
2. **PR #41397 (FlyDSL paged attention + planner).** Fetch, check its base against this branch, merge, and check EAGLE3: does the
   verify/draft-extend path (multi-token per request) go through the new dense decode kernel, or fall back? Implement what is missing.
   Qualify numerics (kernel-level vs the existing Triton path, then GSM8K) and graph replay, then A/B at c=24-48.
3. **PR #41488 (indexer-only decode CP).** Same flow. Measure the indexer chain with communication and the batch/context crossover
   (context length x batch sweep at TP4), then serving A/B.
4. **Graph capture tuning.** Instrument the decode/verify graph runner to log live requests, tokens per request, selected graph size and
   replay count; compute padding as (captured rows - live rows) / captured rows. Derive a capture list from the observed histogram,
   measure capture time, graph memory, KV capacity (`max_total_num_tokens`) and decode/verify latency, A/B against the default list.
5. **CPU-backed prefix cache.** Survey HiCache support for M3 (main K/V + index-K, fp8 pools, TP consistency) and LMCache/EAGLE3.
   Extend HiCache where state is missing, measure restore time and NUMA placement, then CPU tier off vs on at fixed concurrency and
   LRU vs SLRU at equal CPU budget.
6. **Combine** the kept changes, run the full table at c=24-48, quality gate, and record the final config and commands.

### Measurement definitions

- Throughput: AIPerf total token throughput per GPU (prompt incl. cache hits + completion) and output tok/s per GPU; interactivity: p90
  per-user output tok/s; TTFT p50/p90.
- Padding: (captured rows - live rows) / captured rows, rows = tokens entering the graph (decode: requests; verify: requests x draft tokens).
- KV capacity: `max_total_num_tokens` from the server log.

---

## Status tracker

| Item | Status | Result / notes |
|---|---|---|
| Planning doc | done | this file |
| Benchmark harness | done | AIPerf SA fork 754356e9 at `/scratch/m3/aiperf-sa-venv`; `/scratch/m3/bin/agentx.sh PORT CONC 900 OUT` (the scenario enforces >= 900 s). A/A at c=24, identical servers on GPUs 0-3 vs 4-7: 26,108 vs 26,284 total tok/s/GPU (0.7%), ITL p90 20.7 / 20.7 ms, TTFT p50 833 / 733 ms. Side-by-side A/B resolves ~1-2% throughput; TTFT medians need repeats. Baseline c=24 (TOPK_FREQ=1): 25.4-26.3K total, 208-213 out tok/s/GPU. |
| #41397 FlyDSL PA + planner | measured, integration deferred | Branch `opt/flydsl` (PR diff applies cleanly). Blockers for our config: `--attention-backend aiter`, page size 16/64/128 (ours 1), no speculation, SHUFFLE-5D main KV for all layers (breaks this branch's NHD readers and HiCache's per-token host copies). EAGLE3 is reachable: AITER `pa_decode` supports `query_length` > 1 with dense causal masking (MTP). Microbench on M3 dense-layer verify shapes (16 q heads, 1 KV head, fp8, 4 draft tokens, uneven ~170K contexts; `test/manual/minimax_m3/flydsl/bench_dense_verify.py`): FlyDSL **planned** 98 / 243 / 324 / 378 / 571 us at 8/16/24/32/48 requests vs our `_verify_mla_prefix_stage1` 149 / 416 / 531 / 676 / 882 us (**1.5-1.8x**), max diff vs fp32 ~2e-4; FlyDSL static partitions are 2x slower than ours, so the planner is the win. Dense verify is 6.5-7.6% of the steady c=32 step, so ~3% of step time. Tried the planner idea inside our Triton kernel (length-proportional splits): <= 10% on the same shapes, reverted. Next step if pursued: SHUFFLE layout for the 3 dense layers only, FlyDSL for their decode/verify. |
| #41488 indexer CP | **kept** | `opt/indexer-cp`. Fixes on top of the PR: (1) gate rejected all speculation, now chain EAGLE verify; (2) read KV heads from `ModelConfig` (VL config); (3) **packed CP scorer** for verify rows (4 draft rows x 4 heads per 16-row tile, one K read per request): 0.31-0.40x native packed time at >=24 req x >=128K, exact selected-ID parity on 30 shapes; (4) the PR only passed `indexer_cp` from ordinary decode, so **verify never used CP** until wired (serving profiles confirm `_score_shard_packed` replaces `_decode_score_kernel`). Steady c=32 baseline profile: indexer 30.8% of GPU time (`_decode_score_kernel` 26%). Serving A/B vs baseline (same box halves, TOPK_FREQ=1): c=24 25,441 -> **28,182** (+10.8%), ITL p50 13.1 -> 11.5 ms, interactivity p90 46 -> 56; c=32 32,393 -> **33,952** (+4.8%), ITL p50 17.7 -> 12.8 ms; c=40 31,571 -> **33,229** (+5.3%), ITL p50 21.1 -> 15.2 ms. TTFT p50 rises (980 -> 1,769 ms at c=32). GSM8K-500 0.862-0.872. |
| Graph capture tuning | **kept (small)** | `SGLANG_GRAPH_SHAPE_STATS` histogram, rows = requests x draft tokens. Live verify batches are mostly 3-15 requests even at c=32 (the rest wait on prefill). Padding: default list 3.8%, tuned list (default + 9/11/13/15/17/33/35/37) 0.9%, every size 1..48 0.0%. Cost (TP0 log): verify capture 15.5 s -> 34.8 s (tuned) -> 42.4 s (all); verify graph memory 2.18 -> 2.25 -> 2.38 GB, taken from the post-KV reserve, so `max_total_num_tokens` is unchanged (8.249M). Serving (combined branch, total tok/s/GPU): default 27,956 / 34,556 / 33,250 / 15,995 at c=24/32/40/48, tuned 28,844 / 34,633 / 33,197 / 16,086, all-sizes 28,547 / 34,348 / 33,235 / -. Side-swap control (default list on the tuned run's GPUs 4-7): 27,787 / 35,162 at c=24/32, so the tuned list is +3.8% at c=24 and noise elsewhere. Removing the last 0.9% of padding gains nothing: padding is not the bottleneck. GSM8K tuned 0.850, all-sizes 0.840. |
| Serving fixes (`opt/kv-indices-parallel`) | kept | (1) `create_flashinfer_kv_indices_triton` ran one program per request over ~170K tokens every step (639 us x2, 3.1% of GPU time): token-block-parallel launch. (2) **c=48 crash**: `PrefillBudget.available_chunk_tokens` returned the whole chunk when decode headroom exceeded the pool; the allocator had 104 tokens for a 4096-token continuation and the scheduler died (baseline and CP both crashed at c=48). Continuation now claims only existing tokens or waits; regression test in `test_prefill_memory_budget.py`. (3) HiCache index-K host pool lacked `can_use_write_back_jit` (M3 could not start with HiCache). |
| CPU prefix cache | **kept** | HiCache M3 stack covers main KV + index-K + EAGLE3 draft KV host pools; reuse clamped to the shortest prefix across components. Host budget: sglang counts charged page cache as used; dropping our own checkpoint pages (posix_fadvise, `drop_weight_cache.py` during startup) allows 200 GB main + 95 GB index per rank = 13.0M host tokens (1.58x the 8.25M GPU pool). Without it, recompute is 5.5 / 8.8 / 10.9 / 22% at c=24/32/40/48 and c=48 thrashes (36.7M tokens recomputed per 920 s). **Matrix on the combined branch, 200 GB, total tok/s/GPU at c=32 / c=48:** off 34,556 / 15,995; LRU write-through 35,287 / **38,496** (c=24 27,215, c=40 36,206; c=48 host hits 43.5M tokens, recompute 8.6%, 0 dropped; GSM8K 0.872); SLRU 34,638 / 38,970 (TTFT p90 6.3 vs 7.9 s at c=48 but host hits 13.9M, recompute 14.1%; GSM8K 0.868): a tie, LRU kept for lower recompute; write-back 34,082 / 34,297 (**-11% at c=48**, evictions stall on the host copy; GSM8K 0.892): rejected; `--numa-node 0 0 0 0` + `SGLANG_SET_CPU_AFFINITY=0` 33,718 / 38,665 (GSM8K 0.852): no change. NUMA: 4 ranks x ~306 GB = 1.22 TB exceed node 0 (1.06 TB), so ranks 2/3 spill to node 1 (302 / 103 GB there) whatever the binding; restore bandwidth is the same either way (35.6 vs 36.2 GB/s per rank), so binding only matters at <= ~150 GB/rank. Restore cost: 36 GB/s per rank, 32.7 s total in a c=48 run for ~1,650 s of prefill avoided; backup 500 GB in 224 s. Correctness: host-restored greedy output within the device-vs-device nondeterminism spread. Abort robustness (`/scratch/m3/hicache_abort_test.py`): 80K prefix evicted to host (13.02M host tokens used), 30 reuse requests cancelled at 0.02-1.5 s: health 200, 0 running, 0 dropped, host usage unchanged, follow-up request reuses all 79,999 tokens. 300 GB host tier (19.5M host tokens, 2.4x the GPU pool; needs the whole box's page cache dropped first, not just the checkpoint's) at c=48: **40,351** (+4.8% vs 200 GB), TTFT p90 5.8 vs 7.9 s, GSM8K 0.862. Two HiCache servers on one box do not fit (2 x 1.18 TB > 1.9 TB): enable it on one TP4 server per box, or halve `--hicache-size`. |
| Combined config | **done** | `opt/combined` (kevin-mii, 2cefbdad72) = indexer CP + serving fixes + shape stats, plus flags `--enable-hierarchical-cache --hicache-size 200 --radix-eviction-policy lru` and the tuned `--cuda-graph-bs-decode` list. Final table below. |


## Final result

Same AgentX traces (seed 42), 900 s per point, TP4 + EAGLE3 (real acceptance), `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1`, total tok/s/GPU:

| Config | c=24 | c=32 | c=40 | c=48 | GSM8K-500 |
|---|---|---|---|---|---|
| Baseline (`M3-perf-rebase`) | 25,441 | 32,393 | 31,571 | crashes (OOM in `alloc_token_slots`) | 0.862-0.872 |
| #41488 as submitted | = baseline (gate disables CP under speculation; verify never calls CP) | | | | |
| Combined (CP + our fixes) | 27,956 | 34,556 | 33,250 | 15,995 (thrash) | 0.860 |
| Combined + tuned graphs | **28,844** | 34,633 | 33,197 | 16,086 | 0.850 |
| Combined + HiCache LRU 200 GB | 27,215 | **35,287** | 36,206 | 38,496 | 0.872 |
| **Final: combined + HiCache LRU 200 GB + tuned graphs** | 27,241 | 34,657 | **36,771** | 39,780 | 0.868 |
| Combined + HiCache LRU 300 GB | | | | **40,351** | 0.862 |

Final vs baseline: +7% (c=24), +7% (c=32), +16.5% (c=40), c=48 runs at 39.8K instead of crashing. Final ITL p50/p90 11.2/18.7,
14.7/27.3, 15.9/28.7, 16.9/35.1 ms; TTFT p50 0.87 / 1.09 / 1.52 / 2.07 s. HiCache costs ~5% at c=24 (host backup traffic with almost no
host hits: 0.45M tokens), so for a c<=24 deployment drop `--enable-hierarchical-cache` and keep the rest.

Where the gain comes from, all on top of the two PRs (neither helps our config as submitted):

1. Indexer CP made to serve EAGLE3: chain-verify gate, KV heads from `ModelConfig`, packed scorer (4 draft rows x 4 heads per 16-row
   tile), wiring into verify / small-extend. +5-11% throughput and -12 to -28% ITL p50 at c=24-40.
2. Prefill-budget fix: c=48 no longer crashes.
3. HiCache made to start and fit for M3 (index-K host pool attribute, page-cache drop so 200 GB/rank fits): +9-11% at c=40, 2.4x at c=48.
4. Token-block-parallel kv-indices copy (3.1% of GPU time): ITL, throughput flat.
5. Tuned capture list: +3.8% at c=24 without HiCache.

Not kept: FlyDSL PA (#41397; 1.5-1.8x on dense verify, ~3% of step, needs page >= 16 + SHUFFLE layout that breaks HiCache), balanced
verify splits (<= 10% kernel, reverted), SLRU (tie), write-back (-11% at c=48), NUMA binding (no effect; cannot fit one node at 200 GB),
every-size capture list (no gain over tuned).

Final server: `opt/combined`, `rebase_0923/serve_m3.sh` flags + node hostcall knobs, `SGLANG_MINIMAX_M3_INDEXER_CP=1`,
`--enable-hierarchical-cache --hicache-size 200 --radix-eviction-policy lru --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
17 18 20 22 24 26 28 30 32 33 35 37 40 44 48`; drop the checkpoint from the page cache during startup (`drop_weight_cache.py`), or the
host budget check fails. With one HiCache server per box and all page cache dropped, `--hicache-size 300` adds another
~5% at c=48. Results: `/scratch/m3/results/opt0929/`.

## Post-merge A/Bs and lever sweep (2026-10-03 .. 10-05)

All side by side on the two TP4 halves, same day, 900 s per point, TOPK_FREQ=1, real acceptance.
A/A noise is ~1% within a round; identical baselines drift ~3% between rounds, so each lever is read
only against its own round's baseline.

| Experiment | c | Baseline | Variant | Verdict |
|---|---:|---:|---:|---|
| Packed CP scorer (unpacked = merged main) | 24 | 26,576 unpacked | **28,380 packed (+6.8%)** | port it |
| | 32 | 32,294 unpacked | **34,282 packed (+6.2%)** | ITL p50 17.1 -> 12.7 ms |
| PR B: token-block-parallel kv-indices | 24 | 28,858 off | 28,185 on (-2.3%) | dropped, no gain |
| `--max-running-requests 64` (hicache 150 GB) | 48 | 32,100 | 30,105 (-6.2%) | rejected: more live KV, more evictions |
| `--mem-fraction-static 0.9` (hicache 150 GB) | 40 | 34,852 | 34,177 (-1.9%) | rejected: TTFT better, ITL worse |
| `--chunked-prefill-size 16384` (hicache 150 GB) | 40 | 36,024 | 35,725 (-0.8%) | rejected: noise |
| Prefill-budget fix removed (hicache 300 GB) | 48 | 40,943 with | 39,926 without, 0 errors | fix not needed with the CPU tier |

Merged main (#41488, #42166) runs CP *unpacked*: our published +10.8% at c=24 was packed CP; merged
main delivers ~+4.5% until the packed scorer lands (`feat/m3-indexer-cp-packed-verify`, 2 commits on main).

Draft depth was not run: acceptance is 2.65 tokens/verify, and 5 draft rows x 4 heads overflows the
16-row CP tile, which turns packing off. Every production lever in `M3_MI350X_STATUS.md` is on and
verified not to fall back (custom AR, quick-reduce INT4, Gluon prefill, extend-long-prefix, fairness
0.5, kv-splits 64, breakable prefill graphs); its lossy levers (TOPK_FREQ 4, forced acceptance, fp4
MoE activations) stay off, and its structural ones (EP, TP2xDP2, Gluon no-copy, aiter glue fusion)
were already ruled out with measurements.

**Best config (unchanged by the sweep):** packed CP + `rebase_0923/serve_m3.sh` defaults (mem 0.85,
chunk 8192, maxrun 48, EAGLE3 3/4) + TOPK_FREQ=1; c>=40 add `--enable-hierarchical-cache --hicache-size
300 --radix-eviction-policy lru` (one per box, page cache dropped first); c<=32 tier off and the tuned
`--cuda-graph-bs-decode` list. c=48 40,943 / c=40 36,771 / c=32 34,282-35,392 / c=24 28,380-28,844.
