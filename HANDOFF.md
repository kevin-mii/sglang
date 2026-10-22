# DeepSeek-V4.1 mono on SGLang: handoff (updated 2026-10-21 evening)

## 0. Latest state (supersedes sections 1-2 where they differ)

Everything is rebased onto SGLang main 5e79333c95 and plugin main b27d604, validated, and pushed.

**Plugin (sgl-project/sgl-amd-mono)**
- [#21](https://github.com/sgl-project/sgl-amd-mono/pull/21): open, CI green.
- [#14](https://github.com/sgl-project/sgl-amd-mono/pull/14): now MERGEABLE at d668364. Main was merged in, plus the `check_k1` import fix, as new commits on top.
- PR A / B / C are rebased onto it and pushed as `wip/dsv41-pr-a-rebased` (83cd0a6), `wip/dsv41-pr-b-rebased` (b2a76d5) and `wip/dsv41-pr-c-rebased` (76ff1d7).
  - PR C needed `torch_quant` -> `quantization/fp4_fake_quant`.
  - C still carries the fixup commit, to be squashed when its PR opens.
  - Open them after #14 merges.

**SGLang**
- [sgl-project/sglang#45658](https://github.com/sgl-project/sglang/pull/45658): meta launch cuts, from `kevin-mii:amd/dsv4-hip-meta-launches`.
  - It needed a `unified_kv_gate` import fix after #45112; only the e2e caught it.
- [sgl-project/sglang#45663](https://github.com/sgl-project/sglang/pull/45663): DSpark draft and accept launch cuts, from `kevin-mii:amd/dspark-draft-launch-cuts`.
- Glue cuts on `kevin-mii:amd/hip-logits-lastdim-gather`: aiter last-dim logits gather into the logits buffer, and the accept's 3 TP syncs in one NCCL group. **Held by the user, no PR.**
- [#45045](https://github.com/sgl-project/sglang/pull/45045) is superseded by upstream [#45546](https://github.com/sgl-project/sglang/pull/45546), which pins plugin main b27d604. Reuse it for the next bump after #14.
- Combined stacks:
  - `wip/dsv41-best-stack` = DSpark PR + meta PR.
  - `wip/dsv41-best-stack2` = that + the glue cuts. It's the local worktree `sglang-stack2`, not pushed; rebuild it by cherry-picking.

**Validation, all on the new base, MI350X TP4**
- All SGLang PR unit tests pass.
- `check_k1` passes at every width, token parallel included.
- Plugin kernels 4, K2i 14, unit 120.
- e2e 9/9 for #14, meta, DSpark, the full stack and the glue cuts. GSM8K 0.895-0.915.

**Perf** (decode_ab, ABAB means vs SGLang main + plugin main + #21):

| | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| everything | +12.2% | +10.1% | +12.2% | +14.1% |
| plugin stack (#14+A+B+C) | +7.2% | +6.0% | +7.3% | +7.6% |
| #14 alone | +2.7% | +4.2% | +3.7% | +3.0% |
| meta PR | +3.2% | +3.1% | +1.7% | +1.8% |
| DSpark PR | +2.2% | +1.9% | +1.5% | +3.7% |
| DSpark PR on meta + plugin stack | +2.3% | +1.0% | +1.3% | +3.4% |
| glue cuts on the full stack | +0.2% | +1.0% | +1.4% | +1.1% |

Absolute numbers for the full stack: 738.6 / 595.7 / 490.7 / 370.6 tok/s/user.

**Profiles of the full stack**
- c1: 4924 us/step, 246 kernels. Mono ~3850 us.
- c8: 9402 us/step, 257 kernels. Mono ~8130 us.

**Suggested merge order**
1. #21.
2. #14, which unlocks A -> B -> C.
3. Then a pin bump to the merged plugin.
4. #45658 and #45663 are independent and can be reviewed in parallel.

**Later on 2026-10-21**

- **ATOM's `write_step_meta` is ported** (SGLang side) on `kevin-mii:amd/dsv41-verify-step-meta` (f93193a651), which stacks on #45658.
  - `verify_step_meta` fuses the 10 target-verify metadata launches into 1. It takes 4.5 us at c1, where the 10 launches took ~45 us.
  - 31 glue tests pass, including a field-by-field fused-vs-chain backend check. e2e is 9/9 on the full stack and alone.
  - Full stack: +1.2% at c1, neutral at c2-c8. Kernels per step go from 243 to 234.
  - **Open it after #45658 merges** (user decision). Body: `pr-bodies/body_stepmeta.md`.
- **Plugin per-PR numbers on the new base** (ABAB, each against its parent). A and B also pass e2e alone.

  | | c1 | c2 | c4 | c8 |
  |---|---|---|---|---|
  | A | +2.2% | +0.4% | +2.5% | +4.6% |
  | B | +0.7% | +0.3% | +0.9% | +1.1% |
  | C | +1.6% | +0.7% | -0.6% | -0.2% |

  - Bodies: `pr-bodies/prA_body.md`, `prB_body.md`, `prC_body.md`.
  - C's fixup is squashed as local branch `dsv41-pr-c-final` (75626ee, same tree as 76ff1d7). Push it under a new name when C opens.
  - Open A, then B, then C once #14 merges (user decision).
- **Glue cuts** (`amd/hip-logits-lastdim-gather`): still held (user decision). Body: `pr-bodies/body_glue2.md`.
- **Best stack**: SGLang meta + DSpark + glue cuts + step-meta, on the plugin stack. It runs ~750 / 603 / 490 / 372 tok/s/user at c1/c2/c4/c8, and the c1 step is 4807 us with 234 kernels.
- **Long context (64k, c1)** adds only ~85 us/step over short context. SGLang's topk_v2 split-select and main kernel are the growth (~+45 us).

**2026-10-22 (overnight)**

- Draft-block metadata is also fused (`draft_step_meta`, second commit on `amd/dsv41-verify-step-meta`).
  - 6 launches become 1 (4.5 us). Kernels/step at c1: 243 -> 234 -> 229.
  - 41 glue tests pass; e2e 9/9.
- **AgentX, official settings** (temperature 1.0; no simulated-acceptance shortcuts, which upstream sglang#45433 made opt-in). Full best stack (SGLang stack4 + plugin #14+A+B+C) vs current main (5e79333c95 + plugin main):

  | | TTT/GPU | P90 | P50 |
  |---|---|---|---|
  | c1 | 6883 vs 6541 (+5.2%) | 664.5 vs 546.8 (+21.5%) | 701.6 vs 633.1 (+10.8%) |
  | c4 | 7673 vs 7801 (-1.6%, within the run-to-run swing) | 578.1 vs 505.1 (+14.5%) | 692.6 vs 627.4 (+10.4%) |

- **Shortcuts back on** gives +0.9/-0.2/-0.6/-0.3% (decode_ab, temperature 1.0). Not worth it; InferenceX's dsv4 multi-node recipe also sets `SGLANG_SIMULATE_ACC_GREEDY=0`.
- **64k context, old best vs new** (greedy): +1.8% c1, +1.9% c4. No long-context regression.
- **Next ideas** (each ~1%):
  - a vocab-sharded rejection sampler (no corrected-logits gather or full-vocab softmax; real sampling serving);
  - folding the sampling accept into the verify graph (today the graph's epilogue runs masked and the eager accept/commit runs again).

- **Draft-site TP sync skip** (user-approved): `kevin-mii:amd/dspark-draft-sync-skip`, on the DSpark PR branch.
  - On a vocab-sharded Markov head every rank picks each draft token from identical all-gathered candidates, so the per-position rank-0 broadcast (`DSPARK_DRAFT`) is skipped there. An unsharded head and the one-row greedy kernel keep it.
  - e2e 9/9. ABAB: greedy +0.9/+0.1/+1.1/+1.2%; T=1.0 +0.7/+0.4/+0.2/+0.8%.
  - Open as a PR only with the user's OK.
- #14: the user approved the squash merge, but Claude's auto mode denies `gh pr merge`. Merge it in the UI.

**Remaining glue ideas (each <1%)**
- Deferred greedy verify argmax: saves the 73 us bf16 logits gather at c8. It needs a second set of verify graphs.
- aiter all-gather is capped at 80 CTAs.
- Markov TP syncs, 5 x 11 us: the user keeps the default `SGLANG_SPEC_TP_SYNC`.
- The compressor, ~85 us/step at c1: ATOM also runs it outside its kernels, and fusing it changes numerics.
- Token-parallel `_gather_rows` pack/unpack copies.


Goal: every ROCm/ATOM DSv4.1 mono-kernel optimization (ATOM 5e8d913, ROCm/ATOM#2545), plus more, for the best
DSv4.1-Flash decode on SGLang (MI350X/MI355X, TP4, DSpark, `--monokernel-backend amd` through sgl-amd-mono).

Everything is pushed. Nothing is left only on the old node, except the worktrees (listed at the end for reference).
This branch (`kevin-mii/sglang` `wip/dsv41-handoff`) holds this file, the benchmark scripts (`scripts/`) and every
PR body draft (`pr-bodies/`).

## 1. Where things are

### sgl-project/sgl-amd-mono (plugin)

| Branch | Head | What | State |
|---|---|---|---|
| `fix-check-k1-regroup` | 607f5c5 | `check_k1` imports moved by sglang#45112 (#17 did spec/draft) | **PR [sgl-amd-mono#21](https://github.com/sgl-project/sgl-amd-mono/pull/21)** open, CI 2/2 green, mergeable |
| `deepseek-v41-atom-parity` | 0e94e05 | ATOM parity, first batch (3e3702f + merge of #12 + stamps fix) | **PR [sgl-amd-mono#14](https://github.com/sgl-project/sgl-amd-mono/pull/14)** open, **CONFLICTING** with main b27d604 |
| `wip/dsv41-pr-a-followup` | 7c6e527 | PR A: token parallel (>=12 rows), MoE down pipeline, K2 (task, tile) units | stacked on #14, body ready |
| `wip/dsv41-pr-b-index-query` | c90e940 | PR B: index query inside K1 (bit-exact), weights before the q wait, CTA per head tile | on A, body ready |
| `wip/dsv41-pr-c-glue-engram` | 6727e41 | PR C: K1 writes DSpark captures, int32 SWA slots, tok-par gathers head rows, Engram staging (one AR) | on B; commits gl ae5d77c, tc f047195, fixup 4700131, en 6727e41; body needs the Engram bullet |
| `wip/dsv41-compat-regroup-shim` | 3db8fb1 | B + a try-new/fall-back-old import shim for post-#45112 SGLang | dev only, never a PR |

A/B/C still use the **pre-#45112 SGLang names** (`unified_kv_kernels.env_gate`, `mhc.hc_combine`). When rebasing
onto plugin main, take #17's imports (`srt/layers/attention/dsv4/unified_kv_gate.py`, `mhc_combine`) and #21's
`check_k1` imports. Check with `test/unit/test_contract.py` against the SGLang tree first.

### kevin-mii/sglang (fork)

| Branch | Head | Base | What |
|---|---|---|---|
| `amd/dsv4-meta-launches-pr` | 3aa3cbddc0 | main 617c8b67b1 (old) | HIP DSv4 metadata launch cuts (~15 launches/step). Body: `pr-bodies/sglang_dsv4_meta.md`. Not opened. |
| `amd/dspark-draft-launches` | 4fa9231023 | main f8fd41c36e | DSpark draft launch cuts: fused Markov bias GEMV, shard-kept draft logits, draft params in one launch, strided HIP KV commit, accept writes static buffers. Body: `pr-bodies/sglang_dspark.md`. Not opened. |
| `wip/dsv41-best-stack-dspark` | 328cc643b2 | 617c8b67b1 | The measured best SGLang tree (meta + DSpark stack). Pair with plugin `wip/dsv41-pr-c-glue-engram`. Never a PR. |
| `wip/dsv41-handoff` | - | orphan | this file + scripts + PR bodies |

SGLang main is now 5e79333c95; plugin main is b27d604. SGLang main's plugin pin (`python/pyproject_other.toml`,
e07665c) is **broken** for `--monokernel-backend amd` since sglang#45112. [sglang#45045](https://github.com/sgl-project/sglang/pull/45045) (pin bump) is open.

## 2. Next steps (all already approved by the user)

1. **sgl-amd-mono#21**: wait for review/merge (CI green).
2. **Meta PR**: rebase `amd/dsv4-meta-launches-pr` onto current main (it cherry-picked cleanly onto f8fd41c36e).
   Rerun `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py` (17 expected) and the plugin
   e2e (9), push, open the PR with `pr-bodies/sglang_dsv4_meta.md`. Update its "Accuracy Tests" plugin commit and
   base. The branch is pushed but not a PR. Pushing a rebase over it needs a force-push, so push the rebased branch
   under a new name instead (e.g. `amd/dsv4-meta-launches`), or ask first.
3. **DSpark PR**: rebase `amd/dspark-draft-launches` onto current main if needed. Rerun:
   - `test/registered/kernels/ops/speculative/test_dspark_markov_sample.py` (10)
   - `test/registered/unit/spec/test_dspark_draft_block.py` (6)
   - `test/registered/spec/dspark/test_dspark_draft_graph.py` (4)
   - `test/registered/spec/dspark/test_dspark_kernel_parity.py` (20 subtests)
   - the plugin e2e with **plugin main + #21** (instead of the shim).

   Then push and open the PR with `pr-bodies/sglang_dspark.md`, updating the plugin line to the commit used.
4. **Pin bump** (sglang#45045 or a new one): a plugin commit containing #17 + #21.
5. **Plugin A/B/C**: blocked on #14. #14 first needs main merged in (conflicts; ask the user before pushing to it).
   Then rebase A, B and C in turn onto the new #14 head, with the regroup names above. Rerun on each:
   - `check_k1` / `check_k2` (TP2 + TP4, `--widths 6,12,24,48`)
   - `test/kernels/test_deepseek_v41_kernels.py` (4)
   - `test/kernels/test_deepseek_v41_k2i.py` (14)
   - `test/e2e/test_deepseek_v41_server.py` (9)
   - `test/unit` (120)

   Then ABAB each against its parent, and open them with the bodies in `pr-bodies/`.
   - PR C's body still needs an Engram staging bullet: one int32 all-reduce up front for every Engram layer's rows,
     as ATOM stages them. Its enab ABAB vs tc was -0.1/+0.2/+1.8/0.0% at c1/c2/c4/c8, so it is neutral and kept
     for parity.

Rules that applied on this work:
- No force-push on PR branches.
- Ask before opening SGLang PRs, posting comments or editing bodies. Edit PR bodies by PATCHing them via `gh api`.
- No `/tag-and-rerun-ci`.
- Ping only on PR Test (Base) failures.

## 3. Decisions already taken (do not reopen)

- **Deployment knobs stay default.** No `SGLANG_SPEC_TP_SYNC=init`, no `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`.
  - knobab2 on the best build: greedy draft alone is now noise (+0.8/+0.4/+1.0/-0.9%).
  - greedy + TP_SYNC=init gives +1.6/+3.2/+1.4/+0.8%.
- **No numerics-changing speedups** (FP8 lm_head, K1 index-key fusion).
- **ATOM's in-kernel index top-k is excluded.** Its `lds_topk` tree is ~2x slower than SGLang `topk_v2` on gfx950
  (table in the PR B body).
  - Tuning tried: compacting emit, LDS-only barriers, predicated atomics, 32k segments, bitmap placement.
  - Don't retry without a different algorithm (SGLang-style 12-bit coarse histogram + candidate refine).
- **Parked (no e2e gain, judged by ABAB, not timelines):**
  - MoE stream work: cross-task ug prefetch, UG_AHEAD 4, staged down. It gave -0.1/-0.4/-1.6/+0.2%.
  - Splitting the 64 extra down tiles.
  - Local route in every CTA.
  - `HSA_ENABLE_SDMA=0`.

## 4. Results

Per-user decode tok/s:
- Script: `decode_ab.py`, run as ABAB means.
- Setup: MI350X, TP4, DSpark, forced acceptance 3.51, 5 rounds x 3000 tokens.

| Build | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| SGLang main + plugin main, morning of 10-20 | 691 | 558 | 445 | 327 |
| best stack (plugin C 6727e41 on sglang 328cc643b2) | ~720 | ~584 | ~483 | ~366 |

AgentX (aiperf, 900 s, TP4, AL 3.51; X = 1000/P90 ITL):

| Build | c1 P90 / P50 / TTT/GPU | c4 P90 / P50 / TTT/GPU |
|---|---|---|
| main, morning of 10-19 | 594.8 / 635.1 / 6484 | 506.4 / 629.8 / 7493 |
| best stack | 656.5 / 702.2 / 6731 | same-session ABAB vs the 10-20 best: P50 +2.0%, P90 +2.6% |

Per-PR ABAB numbers are in each body:
- meta: +1.7/+2.3/+3.9/+1.3%
- DSpark: +0.9/+2.1/+4.5/+9.3%
- A: +1.6/-1.0/0.0/+5.7%
- B: +1.2/-1.7/+0.4/+1.7%
- C: within noise

Noise:
- decode_ab ~1% between sessions; c2 swings ~2%.
- AgentX TTT ±3.5% run to run.

## 5. Remaining "beyond ATOM" ideas (ranked)

1. **Plugin step-meta kernel**, like ATOM's `write_step_meta`: about 10 launches in the post-draft verify prep,
   ~45 µs/step at c1. Glue launches are worth more than their kernel time at c1: ~15 launches cut gave +2.9%.
2. **c1 inter-forward glue** is ~1100 µs/step (187 kernels). By call site:

   | Call site | µs/step |
   |---|---|
   | graph-input copies | ~97 |
   | MXFP8 linears (N 25600, K 6144) | 83 |
   | eager extend path | 74 |
   | SPEC_TP_SYNC broadcasts | 63 |
   | fp32 logits all-gathers | 59 |
   | lm_head GEMMs (at roofline) | 109 |

3. **DSpark sharded verify argmax.** V4.1 logits are bf16 without softcap, so it needs `LogitsProcessor`
   generalized. Expected gain is small, ~0.3-0.4%.
4. **Per-layer glue at c1:**
   - Index-source layers: 37-65 µs.
   - Engram layers: ~74 µs each before staging. Their RCCL int32 AR misses aiter's custom AR, because bf16 is
     viewed as int32 for bit-exactness.

## 6. Environment and how to reproduce

Hardware: MI350X gfx950; the old node used GPUs 2-5 for TP4.

Software stack on the old node:
- ROCm 10.0.0 pip-SDK venv (`scripts/env-full.sh`, `/opt/rocm` deliberately off the paths).
- torch 2.11.0+rocm10.0.0.
- aiter main e7d2453f.
- FlyDSL 0.3.4.1 overlay.
- sgl_kernel 0.4.7.

Settings that every run needs:
- `AITER_BF16_FP8_MOE_BOUND=0`, otherwise graph capture crashes.
- `DEEPSEEK_V41_FLASH_MODEL_PATH=<model>` plus `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
  otherwise the server downloads the model.
- Tests: `HF_ENDPOINT=http://127.0.0.1:9`, and run only the target test file (the launcher retries online).
- To use the plugin without a pip install, put a generated dist-info on `PYTHONPATH`
  (`scripts/oot_site_plugin/`, entry point `sglang.srt.plugins: sgl_amd_mono = sgl_amd_mono:register_specs`).
  The plugin e2e needs `OOT_SITE=` pointing at it. `scripts/bin/sglang` is the console script for a checkout.

Scripts:
- `scripts/oot_env.sh`: env for the SGLang tree `$W` + plugin `$OOT`.
- `scripts/run_perf.sh` + `scripts/decode_ab.py`: decode ABAB.
  - `ARMS="label:plugin_tree@sglang_tree:on[:VAR=V,...] ..."`, `AB_ROUNDS`, `AB_TOKENS`, `CONCURRENCY`.
  - `AB_CONTEXT=N` for a long shared prefix.
  - Example: `scripts/chain_dsp3.sh` (e2e then ABAB).
- `scripts/run_agx2.sh` + `scripts/serve_agx.sh` + `scripts/client_agentx.sh`: AgentX.
  - Uses the published MI355X agentic recipe, forced acceptance (`SGLANG_SIMULATE_ACC_LEN=3.51`,
    `..._METHOD=match-expected`, `..._TOKEN_MODE=real-draft-token`).
  - Example: `scripts/chain_agxfinal.sh`.

The scripts hard-code the old node's paths (`/scratch/agentx/...`, the scratchpad path): edit `S=`, `W`, `OOT`
and the model path.

Kernel timelines: `SGLANG_AMD_MONO_TIMELINE=1` with `check_ffn` / `check_k1` / `check_k2` and
`--timeline --widths 6,12,24,48`. Judge by the last CTA's end. Cross-rank skew dominates the harness marks.

## 7. Gotchas learned

**FlyDSL**
- A name assigned in a dynamic `if` that also exists outside becomes carried state. So does the context dict when
  you call `c.get(...)` inside the branch ("Cannot extract IR values from bound method"). Read such values before
  the branch and use distinct names.
- A plain `for ... in range()` in a jit function becomes a device loop; use `range_constexpr` to unroll.
- `gpu.barrier()` waits vmcnt. Use a workgroup fence + `s_barrier` for LDS-only hand-offs.
- vmcnt is in order on gfx950: any wait drains every earlier load. "Prefetch, then poll" never overlaps, so put
  polls before prefetches and pin the order with `rocdl.sched_barrier(0)`.

**Collectives**
- aiter's custom all-gather needs a 16-byte-multiple input, else it silently falls back to RCCL. That is why the
  DSpark candidate parts are even.

**Test harness**
- `check_k2` with token parallel: unowned rows are `torch.empty` and may hold NaN bits. Seed a finite sentinel.
- The plugin e2e rewrites the tracked `tmp_output_gsm8k.txt`. Run `git checkout -- tmp_output_gsm8k.txt` and
  never `git add -A` after an e2e.
- Its log always shows 4 memory faults + 12 hipErrorIllegalAddress, from the intentional
  `SGLANG_AMD_MONO_FAULT_INJECT` test.
- e2e helpers reuse any live server. Make sure no orphan server is up before an A/B.

**Workflow**
- Run `ruff --select F821,F401` on changed files before GPU chains.
- Never whole-file checkout across worktrees with different bases: it reverts upstream changes. Use
  `git diff A B -- files | git apply -3`.
- Pre-commit runs offline with `PRE_COMMIT_HOME=<cache>`.

## 8. Old node worktrees (reference only)

| Path | Contents |
|---|---|
| `/scratch/agentx/sglang-pr` | meta PR branch |
| `/scratch/agentx/sglang-dspark-pr` | DSpark PR branch |
| `/scratch/agentx/sglang-markov4` | best stack |
| `/scratch/agentx/sglang-main2` | plain main f8fd41c36e |
| `/scratch/agentx/sgl-amd-mono-pra`, `-iw`, `-gl`, `-tc`, `-en` | plugin A / B / C chain |
| `/scratch/agentx/sgl-amd-mono-iwm` | shim |
| `/scratch/agentx/sgl-amd-mono-rg` | #21 |
| `/scratch/agentx/sgl-amd-mono-tk`, `-mn`, `-ml` | parked experiments |

### Parked plugin experiments

The parked experiments' committed branches are in `dsv41-plugin-parked.bundle` (72 KB, 18 branches). Their bases
are on sgl-project/sgl-amd-mono. To restore them:

```
git fetch origin
git fetch dsv41-plugin-parked.bundle 'refs/heads/*:refs/heads/parked/*'
```

| Branch | What it holds |
|---|---|
| `perf-index-topk`, `perf-index-k2`, `perf-index-topk-tune`, `perf-indexer-clean` | in-kernel top-k (slower) |
| `perf-moe-*`, `perf-down-*`, `perf-ug-place`, `perf-woa-xo`, `perf-local-route` | MoE stream (no e2e gain) |
| `perf-followup-clean` | the early "fu" arm |
| `perf-token-parallel`, `perf-down-pipe`, `perf-k2-tiles` | pre-squash histories of PR A |

Uncommitted sweep edits in a few of these worktrees were not kept.
