# DSV4-Flash MI350X optimization plan — ranked by bs=1 e2e impact

Date: 2026-08-31. Baseline: warm bs=1, 24k ctx, 250 out on 8x MI350X TP8
(sglang main `52e1c247`, aiter `c16d44b9`, cookbook mi355x fp4 low-latency recipe).

## Baseline model of one request

- Prefill (cold 24.5k tokens): 438 ms
- Decode: 250 tokens x 7.69 ms/step = 1923 ms  ← **82% of e2e**
- E2E ≈ 2.36 s

Decode step time is nearly flat in batch (`logs/bs_sweep_24k.log`, il=24000):

| bs | ms/step | vs bs=1 |
|---:|---:|---:|
| 1 | 7.69 | — |
| 2 | 7.87 | +2% |
| 4 | 8.12 | +6% |
| 8 | 8.44 | +10% |

This is the key lever for estimating DSpark: a TARGET_VERIFY step at M=5
(block-4 draft + bonus) costs about what a bs=5 decode step costs, i.e.
**≈8.2 ms ceiling** — and true verify is cheaper still (reads one sequence's
KV once, vs 5 sequences at bs=5). The target model is weight-bandwidth-bound
at small M, so extra verify tokens are nearly free.

Decode step composition (from TP-3 trace, scaled to the warm 7.7 ms step):

| component | ms/step | share |
|---|---:|---:|
| Dense CK GEMM (M=1 blockscale bpreshuffle, 150 launches) | ~1.23 | 16% |
| MoE (moe1+moe2+sorting+router gemm) | ~1.25 | 16% |
| MHC pre/post + rmsnorm | ~0.77 | 10% |
| Allreduce (87 latency-bound calls) | ~0.76 | 10% |
| Attention decode (wv_splitk + paged_decode_split) | ~0.59 | 8% |
| Quant | ~0.36 | 5% |
| Long tail (indexer, topk, rope, copies, sampling) | ~2.7 | 35% |

## Ranked plan

### 1. DSpark speculative decoding on ROCm — e2e 1.8–2.2x
**PROBE RESULT (2026-08-31): WORKS.** Three attempts
(`launch_server_dspark_probe.sh`, logs `server_dspark_probe*.log`):
1. Recipe + DSPARK flags → rank abort at draft-graph capture:
   `HIP error: operation not permitted on an event last recorded in a
   capturing stream` — the draft sampler's `spec_tp_sync` `dist.broadcast`
   is captured into the HIP graph and the RCCL watchdog's event query kills
   it (torch's capture-time watchdog guard appears CUDA-only).
2. `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` + `TORCH_NCCL_ENABLE_MONITORING=0`
   → same crash.
3. **`SGLANG_DSPARK_FOLDED_PROPOSAL=0`** (documented precision fallback:
   proposal head runs eager, all graphs kept) → **server up.**
   - accept len **3.08**, accept rate 0.52 (block 4)
   - bs=1 il=24k: **237 tok/s vs 130 target-only = 1.83x decode**
   - GSM8K-20: **0.950 — identical to target-only baseline**
   - verify step ≈13.0 ms (eager Markov block costs ~4 ms/step)

**PORT COMPLETE (2026-08-31): folded path works on ROCm.**
Root cause: `GroupCoordinator.broadcast` always used
`torch.distributed.broadcast`, violating sglang's own graph-mode collective
contract (see the table in `graph_capture()`: torch.distributed is
graph-disabled; PyNccl is graph-enabled). `all_reduce` had the dispatch;
`broadcast` did not — DSpark's in-graph draft TP sync is the first captured
broadcast. Patch (`sglang-dsv4flash-mi350x/patches/dspark_rocm_graph_broadcast.patch`):
dispatch `broadcast` to `pynccl_comm` when it is active (i.e. inside
`graph_capture()`); eager path unchanged. Applied to
`/sgl-workspace/sglang` in container `dsv4-mi350x-profile`.

Measured, folded (CUDA-default) config, `launch_server_dspark.sh`:
- log line `DSpark draft proposal (greedy + sampling) folded into the draft
  cuda graph` confirms in-graph proposal; server captures cleanly with NO
  fallback envs.
- bs=1 il=24k: **256 tok/s** (eager fallback 237, target-only 130) = **1.97x**
- accept len 3.08 (block 4); GSM8K-200: 0.965 / 0.935 (greedy,
  batching-nondeterminism spread); matches target-only baseline.
- verify step ≈12.0 ms: target M=5 ~8.2 + draft block ~3 + sync ~1.
- B200-parity trace captured: `profile/1788217030.843674/` shows
  `TARGET_VERIFY bs=1` steps + identical EXTEND 16384/8159 cold prefill,
  707.1 ms window (B200: 704.5 ms).

**Allreduce fusion A/B (item 2): OFF — no upside.** With
`--enable-aiter-allreduce-fusion` on top: throughput 255-261 (≤ +2%, within
noise). GSM8K-200 gave 0.900/0.885 vs 0.900-0.965 without; after anchoring
the harness spread (target-only baseline scored 0.945/0.910 on the same
parallel-32 greedy harness, i.e. ±3 pts run-to-run), the fusion scores sit at
the low end but are not conclusively a regression. Since it buys no
throughput at bs=1, keep it off; re-evaluate at higher batch.
Accuracy parity summary (GSM8K-200, parallel 32, greedy): target-only
0.945/0.910 (mean 0.928); DSpark folded 0.965/0.935/0.905/0.900/0.900
(mean 0.921) — **parity**.

**Stability watch:** one GPU memory access fault (gfxhub0 no-retry page
fault, AID3.XCD0) during a parallel-32 GSM8K run on the clean config at
~34 running reqs, right after aiter "no tuned config M:372 N:4096 K:12288 →
default config" GEMM fallbacks. The fusion-config server survived 4 identical
evals, so sporadic; stress rerun in `logs/server_dspark_stress.log`. The
untuned mixed-batch GEMM default config is both the perf item (#3) and the
lead stability suspect.

Remaining headroom to the original 2.5x estimate: ~3 ms/step of draft-block
cost (draft forward + Markov head) — tune draft-side kernels / aiter csv rows
for the draft shapes, item #3 below.
- Estimate: verify step ≈ 8.0 ms (bs-sweep) + ~1 ms draft block ≈ 9 ms/step.
  At accept 2.5 / 3.0 / 3.5 tokens/step → 3.6 / 3.0 / 2.6 ms per token vs 7.69
  baseline → decode 2.1x / 2.6x / 3.0x → **e2e 2.36 s → 1.34 / 1.19 / 1.08 s**.
- Nothing else on the list is within an order of magnitude of this.
- Blocker status: `arg_groups/speculative_hook.py:351` only checks
  `device.startswith(("cuda","npu"))` — ROCm torch reports "cuda", so the gate
  likely passes. Real risk is CUDA-only sgl-kernel ops in
  `dspark_components/` (draft sampler, verify epilogue, KV inject).
- Step 1 (cheap probe, ~15 min): restart server with
  `--speculative-algorithm DSPARK --speculative-dspark-block-size 4`.
  Outcomes: (a) import/launch failure → port list is exactly the failing ops;
  (b) starts but `accept len: 1.00, accept rate: 0.00` → draft head broken on
  ROCm (same trap as EAGLE on 0813); (c) works → measure accept len, done.
- Also restores trace parity with the B200 methodology (TARGET_VERIFY steps).

### 2. Enable aiter allreduce fusion — decode −4–5%, near-zero effort
- 87 allreduce calls/step, 0.76 ms, latency-bound at 6144 hidden bs=1.
- The recipe doesn't set `--enable-aiter-allreduce-fusion` (the M3 launch did).
  Fusing norm+AR / one-shot path should cut 0.3–0.4 ms/step.
- Try the flag first; verify kernel count drops in a re-trace.

### 3. aiter GEMM tuning for DSV4 decode shapes (M≤8) — decode −4–6%
**DONE (2026-08-31), round 1 on `dsv4-perf` (`dd6153d9e3`).** Flash TP8 dense
shapes (N=512/1536 K=4096, N=4096 K=12288, N=8192 K=1024) tuned across
M grid incl. exact M=5 (verify). Result: **256 → 273 tok/s (+6.6%)**,
GSM8K-200 0.945, zero GEMM fallback logs. Artifacts + repro in
`benchmark/deepseek_v4_amd_tuning/` on the branch.
Root-cause note for the MoE side (round 2): `--enforce-shared-experts-fusion`
shifts the fmoe key from the shipped tuned rows (4096/256/**256**/topk **6**)
to (**257**/topk **7**) — no tuned rows existed for the fused shape at any
token count. `--mxfp4-flydsl` tuner mode is the wrong family (a4w4); default
mode tunes the runtime's afp8_wfp4 candidates.
- 1.23 ms/step in CK blockscale-bpreshuffle GEMM at M=1; DSV4 dense shapes are
  likely untuned in aiter's csvs (image ships `dsv4_bf16_tuned_gemm.csv` but
  the hot path is fp8 `gemm_a8w8_blockscale_bpreshuffle_ck`).
- M3 playbook applies verbatim (bpreshuffle rows for the fused qkv / o_proj /
  shared-expert shapes gave 25–40% per-GEMM there). Save 0.3–0.5 ms/step.
- After #1 lands, add M=5/6 rows for verify steps.

### 4. MoE small-batch sorting fusion — decode −4% — DEFERRED (aiter-side)
2026-09-01 status: aiter `c16d44b9` already ships
`fused_dynamic_mx_quant_moe_sort_hip`, and sorting cost is tied to the tuned
`block_m` — which the fmoe tuning round (reverted, see README) would have set.
Blocked behind the same aiter fmoe-tuner layout investigation.
- `opus_moe_sorting_entry` 0.38 ms/step at bs=1 plus separate quant launches.
- Same shape as M3's `SGLANG_OPT_AITER_SMALL_MOE_SORT` (sorting + stage-1
  quant in one launch). Save ~0.3 ms/step.

### 5. MHC kernel fusion (3 kernels → 2, match B200 structure) — decode −3%, prefill −15 ms — DEFERRED (aiter kernel work)
2026-09-01 status: requires new HIP kernels in aiter (fuse
`mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse_rmsnorm` + `mhc_post` into 2
launches like B200's tilelang pair). Multi-day kernel engineering; target
~0.25 ms/step decode + 15 ms prefill at HBM parity. Not attempted on this
branch.
- AMD runs `mhc_pre_gemm_sqrsum` + `mhc_pre_big_fuse_rmsnorm` + `mhc_post`
  (71 ms prefill vs B200's 56 ms with 2 tilelang kernels; memory-bound at HBM
  parity so the 1.28x is real inefficiency). ~0.25 ms/step decode.
- Highest kernel-engineering effort of the small items.

### 6. Kill copies in unified-KV paged prefill — TTFT −2% — DEFERRED
2026-09-01 status: the copy is the OPUS prefill kernel's hard requirement for
fully-contiguous q (`paged_prefill.py` forces `q.contiguous()`); q's layout is
produced deep in `_forward_prepare` across CP/spec/decode paths. Clean fix is
strided-q support in the OPUS kernel (aiter) or a careful audit of the q
producer; ≤8 ms of a 438 ms prefill (<1% e2e for 250-token generations), so
deferred rather than landing a risky blind change. The int32/index
`.contiguous()` calls in the same file are already no-ops at runtime.
- `aten::copy_` inside `sparse_attn_v4_paged_prefill` + `forward_aiter`
  layout conversions: elementwise/copy is 16.4 ms vs 9.1 ms on B200 (1.80x at
  HBM parity). Saves ~8 ms of the 438 ms prefill. Decode impact negligible.
- Only matters for TTFT-sensitive / multi-turn agentic serving.

### Non-issue (checked, dropped)
- Serving-path CPU overhead: serving-bench mean TPOT 12.57 ms was inflated by
  the profiled+JIT first request; median ITL 7.85 ms matches the clean bench
  (7.69 ms). No standing gap.

## Stack-up

Items 2–5 together: −1.1 to −1.5 ms/step → decode −15–19% → **e2e −12–16%**
without DSpark. With DSpark at accept 3.0 they compound (verify step has the
same composition): step 9 → ~7.8 ms, per-token ~2.6 ms →
**combined e2e ≈ 2.2–2.4x** over today's baseline.

Suggested order: #2 (flag flip) → #1 probe (same restart) → #3 (csv tuning,
no code) → #4 → #5/#6. Re-trace after each with `run_profile.sh` and compare
via `compare_prefill.py` / `analyze_ranks.py`; read kernels from the straggler
rank only.
