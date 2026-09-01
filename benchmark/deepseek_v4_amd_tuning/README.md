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
