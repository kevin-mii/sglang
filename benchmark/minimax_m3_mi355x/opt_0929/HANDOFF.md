# MiniMax-M3 MXFP4 on MI350X/MI355X: handoff (2026-10-06)

Everything needed to continue on another machine. The measurement record, including every rejected
lever, is in `PLAN.md` next to this file. The harness is in `tools/`.

## TL;DR

- **Upstream:** #41488 (indexer CP, with our fixes) and #42166 (HiCache K-only host pool) are
  **merged**. #42614 (packed CP scorer, +6-7%) is **open, awaiting review**. That is the only open item
  of ours.
- **Best config** is unchanged by the final lever sweep: packed CP + `rebase_0923/serve_m3.sh`
  defaults + `TOPK_FREQ=1`; add the CPU tier at c>=40. c=48 **40,943**, c=40 **36,771**,
  c=32 **34,282-35,392**, c=24 **28,380-28,844** total tok/s/GPU (900 s AgentX, real acceptance).
- **Merged main today runs CP unpacked**, so it delivers about +4.5% at c=24 rather than the +10.8%
  we measured on a tree with packing. Do not quote +10.8% for main until #42614 lands.
- **Next:** shepherd #42614, then the cookbook, then a 3600 s confirmation ladder. Full list in section 8.

## 1. Upstream status

| PR | What | State |
|---|---|---|
| [#41488](https://github.com/sgl-project/sglang/pull/41488) | ThomasNing's TP4 indexer context partitioning, plus our commits: `kv_heads` from `ModelConfig` (AttributeError on M3's VL config), chain-verify gate allowlist (was rejecting all speculation), registered AMD parity test, CPU gate test; dropped a committed results JSON and the manual harness | **merged 2026-10-02** |
| [#42166](https://github.com/sgl-project/sglang/pull/42166) | `MHATokenToKOnlyPoolHost` sets `can_use_write_back_jit` (M3 could not start with `--enable-hierarchical-cache`) | **merged 2026-10-03** |
| [#42614](https://github.com/sgl-project/sglang/pull/42614) | Packed CP scoring for EAGLE verify rows: +6.8% c=24, +6.2% c=32 | **open** |
| [#36575](https://github.com/sgl-project/sglang/pull/36575) | zcnrex's fused add-RMSNorm + per-token fp8 quant (we pushed a cleanup and a status comment earlier) | open, not ours to drive |

**Investigated and deliberately not filed:**

- **Prefill-budget chunk-continuation fix** (`fix/prefill-budget-chunk-continuation`). It is a real bug:
  the base `PrefillBudget.available_chunk_tokens` returns `chunk_limit` when the pool is exhausted, and
  the scheduler dies in `alloc_token_slots`. Both SWA subclasses already return `None`, so a one-line
  version matching them exists. **It is not needed for our recipe.** With the CPU tier on, c=48 ran
  clean without it (39,926, 0 errors). It still bites high-concurrency deployments without HiCache.
- **PR B, token-block-parallel kv-indices** (`perf/kv-indices-token-block-parallel`): no throughput gain
  (c=24 28,185 on vs 28,858 off). Dropped.
- **FlyDSL #41397**: about 1-2% end to end, and its SHUFFLE KV layout is incompatible with the CPU tier.
  Not pursued (the upstream PR itself is still open).

**Two main-vs-node incompatibilities found while A/B-ing #42614, both unfiled:**

1. `kernels/jit/csrc/elementwise/kvcache.cuh` keeps a device `assert` in `store_kernel`. On nodes
   without device hostcall support it makes hipGraph capture fail with *"operation cannot be performed
   in the present state"*. `opt/combined` carries the 7-line fix (`#ifndef USE_ROCM` around the assert,
   bounds-checked write instead). This is a small upstream PR worth filing.
2. The image's prebuilt `sgl_kernel` 0.4.7 `rotary_embedding` rejects the fp32 cos/sin cache that
   main's EAGLE3 draft now passes. This is image skew; a matching `sgl_kernel` build fixes it.

The PR backlog from the earlier M3-perf work (16 `pr/*` branches, reviewed, never opened) is indexed in
`../PR_INDEX.md`. Most of its upstream prerequisites have since merged; check before reviving any.

## 2. Branches (github.com/kevin-mii/sglang)

| Branch | Use it for |
|---|---|
| `M3-opt-0929` | Docs: this file, `PLAN.md`, `tools/` |
| **`opt/combined`** | **The runnable best-config tree** (M3-perf-rebase + CP + packed scorer + fixes + graph-shape stats). Benchmark from this. |
| `feat/m3-indexer-cp-packed-verify` | Head of #42614, on current main |
| `opt/indexer-cp` | CP work in two commits (port + ours), the source of #41488's fixes |
| `opt/graph-shapes` | `SGLANG_GRAPH_SHAPE_STATS` replay-shape histogram (observability; not upstreamed) |
| `opt/kv-indices-parallel`, `opt/flydsl` | Dropped experiments, kept for reference |
| `fix/prefill-budget-chunk-continuation` | The unfiled fix above, with its regression test |
| `M3-perf-rebase` | Base of `opt/combined` (rebased M3-perf, 2026-09-23) |

`opt/combined` is **not** main. It carries the M3-perf work that never went upstream, and it is what
every number in this file was measured on, unless a row says otherwise.

## 3. Best config

```bash
# from an opt/combined checkout; serve_m3.sh defaults: mem 0.85, chunk 8192, maxrun 48, EAGLE3 3 steps / 4 draft
SGLANG_MINIMAX_M3_INDEXER_CP=1 TOPK_FREQ=1 \
  bash benchmark/minimax_m3_mi355x/rebase_0923/serve_m3.sh
```

| Concurrency | Add | Total tok/s/GPU |
|---|---|---:|
| c>=40 | `EXTRA="--enable-hierarchical-cache --hicache-size 300 --radix-eviction-policy lru"` | c=48 40,943; c=40 36,771 |
| c<=32 | `EXTRA="--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 22 24 26 28 30 32 33 35 37 40 44 48"` | c=32 34,282-35,392; c=24 28,380-28,844 |

GSM8K-500 is 0.85-0.90 on every kept variant.

Rules for the CPU tier:

- **One tiered server per box.** A 200 GB tier needs about 1.18 TB of host RAM, 300 GB about 1.77 TB.
  Two 150 GB tiers fit 1.9 TB, which is what the lever A/Bs used.
- **Drop the page cache first** (`tools/drop_page_cache.py /scratch /tmp /root`). sglang deliberately
  counts charged page cache as used host memory, and we have seen 450 GB of stale cache refuse a
  300 GB tier.
- **Off at c<=24**: there it costs about 5% (backup traffic, almost no host hits).
- **LRU, write-through.** SLRU ties; write-back is -11% at c=48.

`serve_m3.sh` already sets every production lever from `M3_MI350X_STATUS.md`, and none fall back
silently: custom AR, quick-reduce INT4, Gluon sparse prefill, Triton and aiter extend-long-prefix,
prefill fairness 0.5, kv-splits 64, breakable prefill graphs, fp8 KV.

**`SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ` must be 1.** At 4, the model drops `="` from tool-call tags past
~60K context (32 of 42 raw invokes malformed at 4, 0 of 54 at 1). The old published numbers, and all
of ATOM's, use 4.

## 4. Measurement record

All 900 s AgentX runs, TP4, EAGLE3 with real acceptance, TOPK_FREQ=1, total tok/s/GPU, unless stated.

**Campaign, start to finish:**

| Stage | c=24 | c=32 | c=40 | c=48 |
|---|---:|---:|---:|---:|
| Baseline (M3-perf-rebase) | 25,441 | 32,393 | 31,571 | crash |
| + indexer CP (packed) | 28,182 | 33,952 | 33,229 | 15,995 (thrash) |
| + CPU tier 200 GB + tuned graphs | 27,241 | 34,657 | 36,771 | 39,780 |
| + CPU tier 300 GB | | | | 40,943 |

**Post-merge A/Bs and the lever sweep:** each pair ran side by side on the two GPU halves, same day.

| Experiment | c | Baseline | Variant | Verdict |
|---|---:|---:|---:|---|
| Packed vs unpacked CP scorer | 24 | 26,576 | **28,380 (+6.8%)** | #42614 |
| | 32 | 32,294 | **34,282 (+6.2%)** | ITL p50 17.1 -> 12.7 ms |
| PR B kv-indices | 24 | 28,858 off | 28,185 on | dropped |
| `--max-running-requests 64` | 48 | 32,100 | 30,105 (-6.2%) | rejected |
| `--mem-fraction-static 0.9` | 40 | 34,852 | 34,177 (-1.9%) | rejected |
| `--chunked-prefill-size 16384` | 40 | 36,024 | 35,725 (-0.8%) | rejected |
| Prefill-budget fix removed (300 GB tier) | 48 | 40,943 | 39,926, 0 errors | not needed |

Lever rows used 150 GB tiers on both arms, so their absolute numbers are lower than the best config.

**Settled without a run:** draft depth. Acceptance is 2.65 tokens/verify, and 5 draft rows x 4 heads
overflows the packed scorer's 16-row tile, which turns packing off.

**Already ruled out in `M3_MI350X_STATUS.md`:** expert parallelism, TP2xDP2, Gluon no-copy page layout,
aiter MoE glue fusion (needs upstream aiter work), PTPC-FP8 dense (a quality lever, perf-neutral).

**Comparing against ATOM or the old published numbers needs care.** Those are 3600 s runs with forced
acceptance (outputs are not the model's) and TOPK_FREQ=4 on MI355X. Ours are 900 s, real acceptance,
TOPK_FREQ=1 on MI350X. Our 900 s c=24 baseline (25,441) also sits well below the old 3600 s figure
(34,785): TOPK_FREQ accounts for about 10%, and duration plausibly for the rest. That has never been
isolated (section 8).

**Noise:** about 1% within a same-day side-by-side pair; identical baselines drift about 3% between
rounds. Judge a variant only against its own pair's baseline.

## 5. Setting up a new machine

**Get the code and models:**

```bash
git clone https://github.com/kevin-mii/sglang && cd sglang
git worktree add ../opt-combined opt/combined
```

```bash
python -c "from huggingface_hub import snapshot_download as d; d('amd/MiniMax-M3-MXFP4', local_dir='/scratch/m3/models/MiniMax-M3-MXFP4'); d('Inferact/MiniMax-M3-EAGLE3-GQA', local_dir='/scratch/m3/models/MiniMax-M3-EAGLE3-GQA')"
```

**Container.** Use `lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260923`, or a newer gfx950 ROCm 7.2.4
image whose `sgl_kernel` matches the tree you run.

- **If the node has docker:** run the image directly, bind-mount `/scratch`, and skip `inimg`/proot.
- **If it is a locked-down pod like the original** (no docker, no mount/chroot): unpack the image to
  `/scratch/m3/image/rootfs` and use `tools/inimg`, which wraps proot.
  `rebase_0923/HANDOFF.md`, section "Node attempt 2", covers that setup from scratch.

**Hostcall check, first thing.** Start any server with `AMD_LOG_LEVEL=1`. If the log says
`Pcie atomics not enabled, hostcall not supported`, the node is like the original (MI350X VFs), and you
need the workarounds in `tools/knobs.env` plus the aiter config below. If it doesn't, drop
`SGLANG_ROCM_NO_HOSTCALL`, `AITER_NO_HOSTCALL` and the buffered-printf flag from `knobs.env`. On a
normal node they only cost a little speed.

- `AITER_CONFIG_GEMM_BF16` points to a bf16 tuned-GEMM table without FlyDSL rows (FlyDSL needs hostcall).
  Regenerate it with `rebase_0923/nodeenv/aiter_bf16_nohostcall_csv.py`.
- If a kernel still fails with `hipErrorIllegalState` or `AQL dispatch failed`, find the offender with
  `grep -a -l -r hidden_hostcall_buffer ~/.cache/sglang <aiter>/aiter/jit` and delete it so it rebuilds.

**Benchmark client.** Create the venv inside the container, with the container's Python:

```bash
git clone https://github.com/SemiAnalysisAI/aiperf /scratch/m3/aiperf-sa && git -C /scratch/m3/aiperf-sa checkout 754356e9
python -m venv /scratch/m3/aiperf-sa-venv && /scratch/m3/aiperf-sa-venv/bin/pip install -e /scratch/m3/aiperf-sa
```

**Harness.** Every script assumes `/scratch/m3` and calls its siblings through `/scratch/m3/bin`:

```bash
mkdir -p /scratch/m3/bin /scratch/m3/logs /scratch/m3/results/opt0929 && cp benchmark/minimax_m3_mi355x/opt_0929/tools/* /scratch/m3/bin/
```

On a docker node, `inimg` is unnecessary: make it a one-line `exec "$@"` shim so the other scripts keep
working.

## 6. Harness

| Script | Does |
|---|---|
| `variant_sweep.sh PORT GPUS TAG "CONCS" [VAR=val ...]` | Restart a server from `TREE` (default `/sgl-workspace/opt-combined`) with knobs, run the GSM8K-500 gate, then AgentX at each concurrency. `MAXRUN=`, `MEMFRAC=`, `CHUNK=`, `STEPS=`, `DRAFT=`, `EXTRA="..."` override `serve_m3.sh` |
| `ab_round.sh TAG VAR=val` | One lever side by side: baseline on GPUs 0-3, lever on 4-7 (7 min stagger), 150 GB tier both; `CONC=` picks the concurrency |
| `agentx.sh PORT CONC 900 OUTDIR` | One AgentX point. **The scenario enforces >= 900 s**; shorter needs `--unsafe-override` and isn't comparable |
| `agentx_summary.py OUTDIR` / `cache_metrics.py OUTDIR` | Throughput/latency line; device/host cache hits, recompute, evictions |
| `stop_graceful.sh PORT` | SIGTERM the process group, wait up to 180 s, never SIGKILL |
| `drop_page_cache.py DIR...` | Make room for the CPU tier |
| `graph_tune.py SHAPES.json CAPTURE_BS` | Graph padding report from a `SGLANG_GRAPH_SHAPE_STATS` dump |
| `kernel_breakdown.py` | Per-kernel share from a profiler trace (`POST /start_profile`) |
| `hicache_abort_test.py URL` | Cancel requests mid host-restore, then check health and host gauges |

Parity tests for the CP kernels run on one GPU:

```bash
python -m pytest test/registered/amd/test_minimax_indexer_cp.py test/registered/unit/layers/attention/test_minimax_indexer_cp_gate.py
```

## 7. Gotchas that cost hours

- **`pkill -f` / `pgrep -f` patterns match your own shell** whenever the pattern appears in your command
  line, and the shell kills itself (exit 144). Anchor the pattern (`pgrep -f "^/bin/bash /scratch/m3/bin/x.sh"`)
  or kill by PID.
- **Never edit a bash script while it runs.** Bash reads it by byte offset and resumes at the wrong place.
  Kill it, edit, relaunch.
- **Stale server logs.** A log keeps the previous run's content until the next server starts. Check the
  `ARGS:` line before trusting an error or result, and in monitors only count files newer than a launch
  marker. Two "crashes" this campaign were stale.
- **Wait for `/sys/class/kfd/kfd/proc/` to empty** after stopping a server before starting the next.
  Start large replicas one at a time.
- **Verify an injected-bug test can actually fail.** Our first one (masking by group length) could not,
  because the affected block is always force-selected by the local window. A guard that can never fail
  looks exactly like proof.
- **`git rm -r <dir>` removes pre-existing files too.** Check `git diff --name-status` for unexpected `D`s.
- **`pre-commit run --all-files` reformats unrelated files.** Run it on the changed files only.
- **Pushes:** in `/scratch/m3/image/rootfs/sgl-workspace/sglang`, `origin` is upstream sgl-project. Push
  to the `kevin` remote. One stray upstream branch has already had to be deleted.
- **TOPK_FREQ** must be 1 for anything that serves real traffic (section 3).

## 8. Next steps, in order

1. **Shepherd #42614.** It registers `test/registered/amd/test_minimax_indexer_cp.py` on
   `stage-b-test-1-gpu-small-amd`. If a reviewer asks for an A/B on main, it needs a node with hostcall
   support and an `sgl_kernel` matching main (section 1).
2. **Cookbook** (`docs/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx`, config-driven; read
   `.claude/skills/cookbook-add-model`, and `cookbook-migrate-model` if the page needs it). Add the MXFP4
   + EAGLE3 TP4 recipe on MI350X/MI355X: CP env, the CPU-tier flags and their one-per-box /
   page-cache / off-at-c<=24 rules, TOPK_FREQ=1 with the tool-call reason, and the tuned graph list.
   Quote packed numbers only after #42614 merges; until then CP on main is worth about +4.5%.
3. **3600 s confirmation ladder** (c=24/32/40/48) per the `agentx-benchmarking` skill, on the best config.
   This also settles the 900 s vs 3600 s question in section 4. Any comparison against ATOM needs a
   forced-acceptance arm on the same protocol.
4. **File the `kvcache.cuh` ROCm assert fix** upstream. It is small, and it is what lets main capture
   graphs on hostcall-less nodes.
5. **Prefill-budget fix**, only if high-concurrency deployments without HiCache matter (section 1).
