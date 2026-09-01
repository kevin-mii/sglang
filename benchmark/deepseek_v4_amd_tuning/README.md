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
same shape and weight layout. There is therefore a SECOND divergence layer
between the tuner harness and sglang's runtime `fused_moe` invocation
(suspects: scale tensor dtype/2D-vs-flat layout, hidden/intermediate pad
args, moe_sorting buffer sizing). Notes for whoever picks this up: the
runtime layout is a hybrid (weights non-interleaved via
`shuffle_gu_intv = gu_intv and not _use_aiter_a8w4`, scales interleaved),
the tuner has no layout axis at all, and the fast family is interleave-only.
Expected payoff once fixed: ~5-8% of the bs=32 decode step (MoE is 22.5%);
~0 at bs=1 (heuristic already optimal in the compatible family).
