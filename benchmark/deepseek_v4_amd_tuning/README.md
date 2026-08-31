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

Reproduce:

```bash
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl \
  -i dsv4_flash_tp8_untuned_fmoe_shapes.csv -o <tuned>.csv --mp 8
```

Install as `aiter/configs/model_configs/dsv4_flash_fp8fp4_tp8_tuned_fmoe.csv`.
