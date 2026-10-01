# MiniMax-M3 MXFP4 optimization campaign: handoff

State as of 2026-09-30. Everything measured here is on one 8x MI350X (VF) box, TP4, EAGLE3 with real
acceptance, `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1`, AgentX agentic traces (SemiAnalysis AIPerf fork,
seed 42, 900 s per point). Full detail and the rejected-variant matrix: `PLAN.md` in this folder.

## 1. Branches

All on `https://github.com/kevin-mii/sglang`.

| Branch | Head | What it is |
|---|---|---|
| `M3-perf-rebase` | 276976ba11 | baseline this campaign started from |
| `M3-opt-0929` | 4318f48e97 | docs only: `PLAN.md`, this file |
| **`opt/combined`** | **2cefbdad72** | **the config to run**: indexer CP + serving fixes + shape stats |
| `opt/indexer-cp` | 721ae7f110 | #41488 port + our four fixes |
| `opt/kv-indices-parallel` | a4c9ef4066 | kv-indices copy, prefill-budget crash fix, HiCache index-K fix |
| `opt/graph-shapes` | dd3732cd9a | `SGLANG_GRAPH_SHAPE_STATS` replay histogram |
| `opt/flydsl` | 9eb4959b0e | #41397 staged + dense-verify microbench; **not merged**, see section 5 |

Worktrees on the box: `/sgl-workspace/opt-combined`, `opt-misc`, `opt-graph`, `opt-flydsl`.

Note: commit 4318f48e97 was pushed to `sgl-project/sglang` by mistake (that checkout's `origin` is
upstream, not the fork), creating a stray branch `M3-opt-0929` there. No PR was opened. It is also on
kevin-mii, where it belongs. **Open question for Kevin: delete the upstream branch?**
`git push origin --delete M3-opt-0929` from `/scratch/m3/image/rootfs/sgl-workspace/sglang`.

## 2. The recommended config

```bash
# opt/combined, rebase_0923/serve_m3.sh flags + the node's hostcall knobs, plus:
SGLANG_MINIMAX_M3_INDEXER_CP=1 \
  --enable-hierarchical-cache --hicache-size 200 --radix-eviction-policy lru \
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 22 24 26 28 30 32 33 35 37 40 44 48
```

Total tok/s/GPU:

| Config | c=24 | c=32 | c=40 | c=48 | GSM8K-500 |
|---|---|---|---|---|---|
| Baseline | 25,441 | 32,393 | 31,571 | crashes | 0.862-0.872 |
| Both PRs as submitted | = baseline (see section 5) | | | | |
| CP + tuned graphs | **28,844** | 34,633 | 33,197 | 16,086 | 0.850 |
| CP + CPU tier 200 GB + tuned graphs | 27,241 | 34,657 | **36,771** | 39,780 | 0.868 |
| CP + CPU tier 300 GB (c=48 only) | | | | **40,351** | 0.862 |

vs baseline: +13% c=24, +7-9% c=32, +16.5% c=40; c=48 runs at ~40K instead of crashing.

Operational constraints, all learned the hard way:

- **Drop the page cache before startup.** sglang counts charged page cache as used host memory. ~450 GB
  of file cache killed even the 300 GB attempt. `drop_weight_cache.py` only covers the checkpoint; the
  one-shot sweep over `/scratch /tmp /root` took it 451 GB -> 4 GB.
- **One HiCache server per box.** Two TP4 servers each want ~1.18 TB; the box has 1.9 TB.
- **At c<=24, leave the CPU tier off** (costs ~5% there: backup traffic, almost no host hits).
- Never SIGKILL a live server: `/scratch/m3/bin/stop_graceful.sh PORT`. Start replicas one at a time.

## 3. What we changed on top of the PRs

1. **Indexer CP made to serve EAGLE3** (`opt/indexer-cp`): the PR's gate rejected all speculation (now
   rejects only topk>1); KV heads read from `ModelConfig` (M3's VL config); **packed scorer** (4 draft
   rows x 4 heads per 16-row tile, one K read per request, 0.31-0.40x native at >=24 req x >=128K,
   exact parity on 30 shapes); and the PR only passed `indexer_cp` from ordinary decode, so **verify
   never used CP** until wired. +5-11% throughput, -12 to -28% ITL p50 at c=24-40.
2. **Prefill-budget fix**: `available_chunk_tokens` handed out the whole chunk when decode headroom
   exceeded the pool; the allocator had 104 tokens for a 4096-token continuation and the scheduler died.
   Baseline and CP both crashed at c=48. Regression test in `test_prefill_memory_budget.py`.
3. **HiCache made to work for M3**: index-K host pool was missing `can_use_write_back_jit` (server would
   not start); plus the page-cache accounting above. +9-11% at c=40, 2.4x at c=48.
4. **Token-block-parallel kv-indices copy** (was 3.1% of GPU time): ITL improves, throughput flat.
5. **Tuned capture list**: +3.8% at c=24, confirmed against a side-swap control on the same GPUs.

## 4. Robustness evidence

- Abort test (`/scratch/m3/hicache_abort_test.py`): 80K prefix evicted to host, 30 reuse requests
  cancelled at 0.02-1.5 s -> health 200, 0 running, 0 dropped, host usage unchanged, follow-up reuses
  all 79,999 tokens.
- Host-restore correctness: device-vs-device greedy is already nondeterministic; the host hit's logprob
  sits inside that spread. GSM8K 0.868 with restores active.
- Restore cost: 36 GB/s per rank; 32.7 s total in a c=48 run for ~1,650 s of prefill avoided.

## 5. Why the two PRs gave ~0% as submitted

- **#41488 (indexer CP)**: gate turns CP off under speculation, so EAGLE3 never reaches it; and even
  un-gated, only plain decode calls it, which EAGLE3 verify does not use. Fixed, kept.
- **#41397 (FlyDSL PA + planner)**: needs `--attention-backend aiter`, page >= 16 (ours is 1), no
  speculation, and a SHUFFLE-5D main KV layout that breaks this branch's NHD readers and HiCache's
  per-token host copies. Microbench on M3 dense verify shapes: planned FlyDSL 1.5-1.8x our kernel, max
  diff ~2e-4 -- but dense verify is ~3% of step time. Measured, not integrated.

## 6. Where to go next

Ranked by expected value:

1. **Raise the CPU tier further.** 300 GB gave +4.8% over 200 GB at c=48 and was never pushed higher;
   19.5M host tokens is 2.4x the GPU pool. Try 400-500 GB with the page cache dropped, and re-check the
   c=24/32 cost. Cheapest remaining win.
2. **TTFT at high concurrency.** c=48 TTFT p90 is 5.8-7.1 s. Prefill is the bottleneck everywhere
   (verify batches stay at 3-15 requests even at c=32 because the rest wait on prefill), so chunked
   prefill sizing / scheduler admission is the next real lever, not decode kernels.
3. **FlyDSL for the 3 dense layers only**, keeping sparse layers and HiCache on the current layout.
   ~3% of step at 1.5-1.8x, so ~1-2% end to end. Only worth it if (2) stalls.
4. **Indexer CP beyond TP4** and interaction with `TOPK_FREQ=4` (untested; 4 corrupts long-context tool
   calls, which is why every number here uses 1).

Dead ends, do not re-run: SLRU (tie with LRU, more recompute), write-back (-11% at c=48), NUMA binding
(no effect; 4 ranks x ~306 GB cannot fit node 0's 1.06 TB), every-size capture list (no gain -- padding
is not the bottleneck), balanced verify splits (<=10% kernel, reverted).

## 7. Harness

- Env: proot rootfs at `/scratch/m3/image/rootfs`, entered via `/scratch/m3/bin/inimg`. Hostcall
  workarounds (MI350X VF has no PCIe atomics) in `/scratch/m3/bin/knobs.env`. Full env notes for a
  different model: `/scratch/m3/ENV_HANDOFF.md`.
- Bench: `/scratch/m3/bin/agentx.sh PORT CONC 900 OUTDIR` (the scenario enforces >= 900 s), summarize
  with `agentx_summary.py OUTDIR`, cache counters with `cache_metrics.py OUTDIR` (**pass absolute
  paths** -- relative paths do not survive the chroot).
- One variant end to end: `variant_sweep.sh PORT GPUS TAG "24 32 40 48" EXTRA="..."` (restarts the
  server, GSM8K gate, then AgentX at each concurrency).
- A/B method: two TP4 servers side by side (GPUs 0-3 vs 4-7), then swap sides. A/A noise ~1% on
  throughput, ~10% on TTFT. Anything under ~2% needs the swap before you believe it.
- Raw results: `/scratch/m3/results/opt0929/`.
- Known counter bug: `cache_metrics.py` reports negative "recomputed" when a run follows a warm server
  (prompt tokens counted after the cache was already populated). Throughput numbers are unaffected.
