# DeepSeek-V4-Flash MI350X/MI355X (gfx950) aiter tuning artifacts

Tuned kernel configs for DeepSeek-V4-Flash-0731 fp4 at TP8 on gfx950, produced
with aiter's stock tuners (aiter commit `c16d44b9`). These are staging
artifacts for the `dsv4-perf` branch; their long-term home is
`aiter/configs/model_configs/`.

## Dense GEMM (a8w8 blockscale bpreshuffle)

The Flash TP8 dense shapes were untuned in aiter (only 7168-hidden DSV4-Pro
rows existed), so every decode/verify-step GEMM used the default config:

- N=512  K=4096, N=1536 K=4096, N=4096 K=12288, N=8192 K=1024
- M grid: exact {1,2,4,5,8} (5 = DSpark block-4 verify at bs=1),
  16-multiples to 512, pow2 to 16384 (aiter pads lookups M -> mult-of-16 ->
  pow2)

Reproduce:

```bash
cd /sgl-workspace/aiter
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --preshuffle \
  -i dsv4_flash_tp8_untuned_gemm_shapes.csv -o <tuned>.csv
```

Install: append the tuned rows to
`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`
(or point `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` at the csv) and
remove the merged cache `/tmp/aiter_configs/`.

Measured (bs=1, il=24k, DSpark block 4, 8x MI350X VF): 256 -> 273 tok/s
(+6.6%); GSM8K-200 0.945 (within the 0.90-0.965 harness band).

## FlyDSL fmoe (fp8 x mxfp4)

`--enforce-shared-experts-fusion` shifts the MoE shape key from the tuned
(model 4096, inter 256, expert 256, topk 6) rows to (expert **257**, topk
**7**), which had no tuned rows at any token count — every MoE call used the
heuristic FlyDSL fallback. Tuned token grid: pow2 1..16384 (runtime token
lookups are pow2-bucketed).

**Status: tuned, but REVERTED — do not install.**

- `--mxfp4-flydsl` is the wrong tuner family (a4w4; all candidates fail on a
  missing codegen'd `aux_sortzi_NE257_TOPK7` sorting instance).
- Default mode (`gemm_moe_tune.py -i ... -o ... --mp 8`) tunes the runtime's
  afp8_wfp4 FlyDSL candidates and produced rows for all 15 token sizes with
  reported err 0.0-5.5% — but installing them **breaks the model end to end**:
  DSpark accept length drops 3.08 → 1.00 and GSM8K-20 drops to 0.05 (garbage
  output). The tuner's isolated validation does not cover the runtime's
  fused-shared-expert layout for this shape. Evidence kept as
  `dsv4_flash_tp8_tuned_fmoe.csv.BROKEN-DO-NOT-USE`.
- Measured upside was small anyway: the heuristic FlyDSL fallback picks
  (t32x128x256 pm1) are within ~10% of the tuner's best at decode token sizes
  (~0.06 ms of a 12 ms verify step, ≈0.5% e2e). Needs aiter-side
  investigation (tuner layout coverage for expert=N+1 fused-shared shapes)
  before retrying; not worth blocking this branch on.

**Retry with the M3 methodology (2026-09-01)** — retuned with
`-o2 <profile_all> --errRatio 0.005`, then hand-selected per-token rows
restricted to the runtime-proven plain `flydsl_moe2_afp8_wfp4_bf16_*` family
(the M3 playbook; deploy would be via `AITER_CONFIG_FMOE=<single csv>`, never
model_configs appends). Outcome (`dsv4_flash_tp8_fmoe_profile_all.csv`):
- decode tokens (1-8): best plain-family candidates run 24-37 us vs ~17 us
  for the current heuristic picks — installing them would REGRESS decode.
  Every tuner candidate faster than the heuristic is in the broken
  layout/opus family.
- prefill tokens (>=4096): only cktile-stage1 rows beat the heuristic and
  only by ~8% of MoE-kernel time (~3-4 ms of a 438 ms prefill) — not worth
  the correctness risk of another unproven family.
Conclusion: with aiter `c16d44b9` the heuristic already sits at the working
family's optimum; the real fix is aiter-side (make the v2 layout family
correct for expert=N+1 fused-shared shapes — it is ~40% faster at decode).
No csv is installed.

**Root cause (2026-09-01, isolated in aiter's own harness):** the layout
family is NOT numerically broken — `op_tests/test_moe_2stage.py` validates the
exact broken csv rows at 4096/256/257/7 to ~1e-4 logits diff, including with
runtime-style forced shared-expert routing. The divergence is the **weight
shuffle layout**: for a8w4 (fp8 act x fp4x2 weight) sglang's Fp8MoEMethod
shuffles expert weights **non-interleaved**
(`fp8.py: shuffle_gu_intv = gu_intv and not _use_aiter_a8w4`), while aiter's
tuner/test harness preps a8w4 weights **gate-up-interleaved**
(`shuffle_weight_a16w4(..., gate_up=True)`). Tuner winners are therefore
selected under a layout the runtime doesn't have; the runtime heuristic
happens to only pick kernels compatible with the non-interleaved layout, and
the tuner's layout/opus-family winners mis-read it -> garbage. Fix belongs in
aiter's fmoe tuner (prep weights to match the runtime a8w4 layout, or emit
the layout as a csv column the runtime dispatch checks).

**Layout-flip experiment (2026-09-01, reverted):** flipping sglang's a8w4
expert-weight shuffle to interleaved (verified byte-identical to the tuner's
`shuffle_weight_a16w4` prep; `shuffle_weight(is_guinterleave=True)` == it)
plus deploying the tuned csv via `AITER_CONFIG_FMOE` crashes the server with
`HSA_STATUS_ERROR_EXCEPTION` (hardware page fault) in the first prefill —
including with all opus rows swapped for harness-validated plain/flydsl
alternatives, i.e. the flydsl `moe2_layout` family itself faults under real
serving at small token counts while passing the isolated harness with the
same shape and weight layout. **Full root-cause session (2026-09-01, second pass with gh research).**
Upstream already has the two fixable pieces in flight:
- [ROCm/aiter#4998](https://github.com/ROCm/aiter/pull/4998) (open): the
  FlyDSL v2 stage2 wrapper (`_flydsl_v2_stage2_wrapper`) passes dynamic
  per-call tensors through an `@lru_cache`d uint8-view helper
  (`_mxfp4_scale_u8`); tensor hashing is id-based, so multi-layer serving
  returns stale views of freed GPU memory. Found independently by a Kimi-K3 +
  SGLang TP8 deployment (their symptom: OOM after aiter#4642 moved K3 to
  `flydsl_moe2_layout_*`). **This was our HSA hardware fault** — applying the
  PR's 13-line fix (patch: `aiter_opus/fused_moe_pr4998_stale_view_fix.patch`,
  applied in the container) eliminates the crash entirely.
- [ROCm/aiter#4300](https://github.com/ROCm/aiter/pull/4300) (open, verified
  on dsv4_fp8fp4): aiter's own run_config had drifted from test_moe_2stage
  the same way; its fix list confirms the canonical a8w4 harness contract
  (gate_mode=INTERLEAVE + `shuffle_weight_a16w4` weights +
  `AITER_BF16_FP8_MOE_BOUND=0`).

Result matrix after applying #4998 (server no longer crashes; GSM8K-20 as the
garbage detector; "flip" = `shuffle_weight(is_guinterleave=True)`, verified
byte-identical to the tuner's `shuffle_weight_a16w4` prep):
- stock layout, no csv (production heuristic): 0.95 (correct)
- stock layout + tuned csv: 0.05
- flipped weights + csv: 0.05
- flipped weights + csv + gate_mode=INTERLEAVE: 0.10
- harness, synthetic weights, same csv rows, same shape, forced shared
  routing: ~1e-4 logits diff (correct)

So the layout family mis-reads sglang's DSV4 expert weights under EVERY
layout combination tried, while the heuristic asm family reads them
correctly, and the harness validates the same kernels on synthetic weights.
The remaining divergence is the EFFECTIVE weight/scale layout produced by
`Fp8MoEMethod`'s fp4-expert branch vs the harness prep — note
`Mxfp4MoEMethod` runs a checkpoint gate/up de-interleave permute
(`if gate_up_interleaved:` before its shuffles) that `Fp8MoEMethod`'s
fp4-expert branch does not, so the DSV4 checkpoint's native gate/up row
order likely differs from the harness's synthetic GGUU. Pickup: dump one
expert's pre-shuffle w13 rows + scales from the loader and diff against
harness prep of the same tensors; then either add the missing permute for
the interleave path or teach the tuner the runtime layout. Payoff once
solved: ~5-8% of the bs=32 decode step (MoE is 22.5%); ~0 at bs=1.

## DSV4-Pro-0813 (2026-09-01)

Same methodology, run against `deepseek-ai/DeepSeek-V4-Pro-0813` (fp8
block-quant checkpoint; experts requantized to mxfp4 on gfx95 -> same
a8w4 runtime family, so the fmoe csv-tuning block above applies to Pro
unchanged).

- `dsv4_pro_tp8_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`: all 27 M
  for the one untuned Pro TP8 dense shape family N=7168 K=21504
  (errRatio 0). Install by appending to aiter's
  `dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv` (idempotent check
  first) and removing `/tmp/aiter_configs/`.
- Upstream gap found while validating the dp8+TBO lane: DSpark verify
  cannot be split by two-batch overlap —
  `batch_overlap/two_batch_overlap.py:split_spec_info` reads
  `spec_info.retrieve_index`, which `DFlashVerifyInput` does not have
  (EAGLE-only assumption). Until fixed, TBO lanes are target-only for
  DSpark models; at 80-way concurrency DSpark (chat ~3.7k tok/s) beats
  the TBO lane without it (2.6k), so the standing Pro config keeps spec
  and skips TBO.
