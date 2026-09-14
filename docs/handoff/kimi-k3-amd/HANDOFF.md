# Kimi-K3 on SGLang / 8x MI350X — handoff (2026-09-14)

## Goal
Make SGLang the fastest Kimi-K3 server on AMD MI350X/MI355X on the InferenceX AgentX trace (and the fixed 1k/8k cells), beat the published MI355X vLLM/ATOM curve, then upstream the changes. Interactivity = 1 / p90 of per-request mean ITL (tok/s per user); throughput = (input+output tokens)/duration/8 per GPU.

## Fastest branch and configs (state of the art on this node)
- Branch: `kevin/k3-dcp-rocm-pr` = upstream/main 2f5cc8e33e + 4 commits (023740cdfb deferred f_b, a50c4b09f0 BCG opt-in, b1c8f3adad free-segment coalescing + unit test, b827ca2e89 cookbook cell). Patches: `pr/patches/0001..0004`, PR texts `pr/PR_A..D.md`. Worktree `/root/sglang-k3-pr`; the container's live tree `/root/sglang-k3-sota` is on branch `pr-eval` (same head). Fork remote `kevinmii` (github.com/kevin-mii/sglang), `upstream` = sgl-project. NOT pushed yet (ask before pushing / opening PRs).
- Image `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260910`, container `k3-sota` on the docker2 daemon (`docker -H unix:///var/run/docker2.sock`), mounts: /root/sglang-k3-sota -> /sgl-workspace/sglang-sota, /mnt/doscratch/models -> /models (Kimi-K3, Kimi-K3-DSpark), /root/kevinmi/AMD_K3_Perf -> /results. vLLM v0.29.0 container `k3-vllm`.
- Base launch (launch_sglang.sh adds `--tp-size 8 --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --mem-fraction-static $MEM_FRAC --cuda-graph-max-bs-decode 256` and env `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1`; runner env adds `SGLANG_AITER_MLA_GLUON=1 SGLANG_K3_RADIX4_TOPK=1 SGLANG_K3_KDA_FUSED_BACKEND=aiter`; image env `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`):
  `--attention-backend aiter --dcp-size 8 --dcp-comm-backend a2a --context-length 1048576 --chunked-prefill-size 8192 --enable-aiter-allreduce-fusion --max-running-requests 2xconc --watchdog-timeout 3600`
- Per point (AgentX protocol pins DSPARK accept length: `SGLANG_SIMULATE_ACC_LEN=<AL> SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_RAGGED_VERIFY_MODE=static`):
  | conc | spec | extra flags | mem | result (FAST 1200 s window) |
  | --- | --- | --- | --- | --- |
  | 1 | DSPARK block 7, AL 3.78 (non-DCP run) | | 0.85 | 2181 tok/s/GPU @ P90 111 |
  | 4 | DSPARK 7, AL 3.78 (non-DCP run) | | 0.85 | 2130 @ 46.8 |
  | 8 | DSPARK 3, AL 3.0 | `--prefill-decode-interval 6 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 400` | 0.9 | 3842 @ 43.2 |
  | 16 | DSPARK 3, AL 3.0 | `--prefill-decode-interval 6 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 600` | 0.9 | 4490 @ 19.1 (no spec: 4154 @ 18.1 with interval 16) |
  | 32 | none | `--enable-hierarchical-cache --hicache-ratio 3 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first_direct --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 600` (no interleave) | 0.85 | 5866 @ 11.2 (measured on the older asm-decode branch; PR branch with interval 16 gave 4714 @ 8.1, so keep the interval off at 32) |
  Published: B200 5510 @ 40 / 3246 @ 70 / 2123 @ 101; MI355X best-of(ATOM, vLLM) 7196 @ 40 / 3385 @ 70 / 2031 @ 101.
- Accuracy under DCP8: GSM8K-200 @ parallel 32 0.995 (no spec) / 0.985 (DSPARK 3) — `benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 32`. Correctness probe: `python3 tools/dcp_probe.py` (expects SCORE 8/8).

## Harness
- InferenceX at /mnt/doscratch/k3_perf/InferenceX (900f198, aiperf submodule 754356e); aiperf venv /mnt/doscratch/k3_perf/aiperf_runtime/venv; HF corpus semianalysisai/cc-traces-weka-062126 cached in /mnt/doscratch/hf_home.
- SGLang point: `MEM_FRAC=.. EXTRA_ENV=".." MIXED=" " ATTN_OVERRIDE="--attention-backend aiter" HICACHE_RATIO=3 ./agentx_run2.sh <tag> <conc> <spec_tokens> <acc_len|0> <hicache on|off> 1 <extra server flags>` -> agentx/<tag>/ + one summary line (per_gpu total, intvty p50/p90, ttft, hit rates). FAST=1 -> 1200 s + 1 warmup/lane; use 3600 s + 10/lane for publishable points.
- vLLM point (InferenceX MI355X recipe, DCP8 + DSpark synthetic AL): `./agentx_run_vllm.sh <tag> <conc> auto 1` (JSON args go through a script file; needs free GPUs at gpu-mem 0.9).
- Validation scripts: `tools/test_dcp_aiter_prefill.sh <tag> [flags]` (no spec) and `BLOCK=3 XFLAGS=".." KDA_ENV=".." tools/test_dcp_dspark.sh <tag>` (spec): probe, GSM8K, 200k prefill, 8x100k ITL. Stall test `tools/stall_load_test.py` (run inside the container). Decode profile `tools/profile_dcp_decode.py` (CUDA graphs hide kernels: launch with `--disable-cuda-graph` for a breakdown).
- All results: agentx/RESULTS.md; ledger runs/20260911_kimi_k3_sota_humanize/humanize/attempt-ledger.md (rows 1-21); plans in runs/.../analysis/; report SOTA_LOOP_FINAL_REPORT.md.

## Why we are behind the published curve (evidence)
1. Verify/decode step cost at batch 4-8: p50 per-request mean ITL 13 ms at c4 with AL 3.78 => ~49 ms per verify step; the published 70 tok/s tail implies ~30 ms. Suspects: bf16 SSM state drops KDA verify to the Triton fallback under spec (cookbook note); the DSpark draft runs on the triton backend on ROCm ("DSPARK draft worker only supports ... falling back to triton"); MoE at 32-64 tokens/step; 48 DCP a2a exchanges/step.
2. Short answers (130-170 tokens) absorb one or two prefill-chunk stalls (0.6-1 s each): that is the p50->p90 gap at c4/c8. Default scheduler runs chunks back-to-back (39 s stalls); `--prefill-decode-interval N` bounds it (N counts steps: 16 without spec, 6 with; off at 32); vLLM keeps decodes in every step (mixed chunk, 4k budget).
3. Cold-prefill volume is 2x the trace's ideal without a host tier (c8 hit 96.4% vs 98%); KDA state pool (300 fp32 slots, 5/req) was the eviction cause at c16 (bf16 + 600 slots fixed 88 -> 95%).
4. c32 gap is kernels: ATOM's A4W4 SiTU MoE + FlyDSL stage-2 fp8 + INT4 quick-reduce + LMCache 128-192 GB/rank.

## Parity with the vLLM / ATOM recipes (analysis/agentx-plan.md has the table)
Have: A8W4 SiTU, fp8 KV, asm head padding, DCP8 a2a on aiter, FlyDSL, DSPARK+ReplaySSM, synthetic AL, max-num-seqs 2xconc, chunk 8192. Queued (gated on GSM8K >= 0.97): `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `--speculative-draft-kv-cache-dtype fp8_e4m3`; ATOM-only: `AITER_SITUV2_A4W4=1` (replaces A8W4; SGLang's mxfp4 path honors the layout), `AITER_FLYDSL_STAGE2_FP8=1`. Not honored for K3: `--language-model-only`. No NUMA binding.

## Remaining work, in order (each with the acceptance signal)
1. Run the armed queues once GPUs are free: `agentx_queue14.sh` (parity validation + gate, bs-8 verify profile with graphs off, ladder c1-sp6/c4-sp3/c8-sp3(+mixed 4k)/c16-sp3(+mixed)/c32-hic, vLLM c8/c16/c4) then `agentx_queue15.sh` (A4W4, stage-2 fp8, gated; c4/c8/c32 with accepted envs). Watcher `tools/gpu_free_wait.sh` starts the containers when all 8 GPUs are < 8 GB used. Record each point in agentx/RESULTS.md.
2. Profile the verify step (from queue14 logs/profile_verify_bs8.txt): split KDA / draft / MoE / attention / a2a. Then: fp32 SSM + `--mamba-radix-cache-strategy extra_buffer_lazy` under spec vs bf16 (step time and cache hit); `--linear-attn-verify-backend` options; draft on aiter.
3. Decide interleave vs mixed chunk from the c8/c16 variants; if mixed wins, PR a fix/cookbook note; if the interval wins, PR a token-counting (spec-aware) interval.
4. Host tier at c8-c16 (HiCache ratio 3, io kernel, page_first_direct) to bring cold volume to the trace's ideal; note the "Unsupported element_size = 576 for JIT HiCache kernel" fallback -> kernel PR candidate.
5. Full ladder at 3600 s / 10 warmups per lane for publishable numbers (c1, c4, c8, c16, c32), then the vLLM (and if the image is pulled, ATOM `rocm/atom-dev:...kimi_k3_agentic_0907`) comparison on the same node.
6. PRs (see pr/): open A-D; then draft KV pool widened 8x under DCP (11.5 GB/rank wasted), `--language-model-only` for kimi_k3.py, fused KDA verify with bf16 state, DSpark draft on aiter, HiCache 576-byte kernel, LMCache under DCP, cookbook envs after gates.

## Gotchas
- Other users share this node: on 2026-09-14 09:09 a MiniMax-H3 diffusion job (container minimax-h3-hwopt-u8, default docker daemon) took all GPUs and my containers were SIGKILLed. Never kill others' processes. `docker -H unix:///var/run/docker2.sock start k3-sota k3-vllm` restores mine (mounts intact).
- `pkill -f` matches your own shell: use `[s]glang` patterns and skip $$/$PPID. After `pkill -9` remove `/tmp/aiter_configs/*.lock` in the container. The harness kills long background commands when host MemFree is low: poll in <= 600 s foreground loops.
- `SGLANG_K3_KDA_FUSED_BACKEND=aiter` crashes upstream at decode graph capture without commit 023740cdfb. DSPARK under DCP needs `SGLANG_RAGGED_VERIFY_MODE=static`. HiCache + DCP: L1/L2 only (L3 storage rejected). `--enable-mixed-chunk` under DCP untested on upstream.
- KDA state pool: 300 fp32 slots default (~25 MB/slot/rank); 900 fp32 does not fit; bf16 halves it. DSPARK adds ReplaySSM ring buffers (~11 GB/rank at 16 running) and a widened draft pool; use mem 0.9 under spec.
- Commits: user.email mikevin920@yahoo.com, no session trailer, no attribution footer in commits or PR text. Never print HF tokens. Do not post externally without asking.
