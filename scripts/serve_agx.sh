# The PUBLISHED MI355X agentic.yaml (InferenceX commit 47adb59a, run 37423021942,
# dashboard 2026-10-07), reproduced on the latest-main source stack -- sglang
# 2f1acef3d8 (Oct 9) + aiter e7d2453f with the image's patches -- which is newer
# than that run's image v0.5.21-rocm720-mi35x-20261006. No Docker on this box.
# Base args/env are copied verbatim; only the per-point knobs are variables:
#   TP (2|4)  MEMFRAC  CHUNK  PDI  TAILS (swa-prefix-tails)  MAXRUN  GRAPHBS
#   SCHED     schedule-policy (unset = engine default, as published)
#   DEP=1     B200's DEP4 arm: TP4/EP4 + attn-dp 4 + dp-lm-head +
#             local-control-broadcast + total_requests (not in the MI355X yaml)
# Acceptance is NOT set here; the caller passes the AgentX triple, because a
# direct launch bypasses the srt driver that would otherwise inject it.
W=${SGLTREE:-/scratch/agentx/${SGL:-sgl-main}}
TP=${TP:-4}
source /scratch/dsv41/env-full.sh >/dev/null 2>&1
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export PYTHONPATH=/scratch/agentx/main-overlay:/scratch/dsv41/sgk-k3rev:$W/python:${OOT:+$OOT:}${OOT_SITE:+$OOT_SITE:}/scratch/agentx/aiter-${AITER:-main}
export AITER_JIT_DIR=/scratch/agentx/cache-${AITER:-main}/aiter-jit
export TRITON_CACHE_DIR=/scratch/agentx/cache-${AITER:-main}/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/agentx/cache-${AITER:-main}/inductor
export SGLANG_JIT_CACHE_DIR=/scratch/agentx/cache-${AITER:-main}/sglang-jit
# STACK=old: the published image's code line (v0.5.21-era wt-bench + aiter acf8fdf9 with
# the image patches, flydsl from the venv), for anchoring regressions on main.
if [ "${STACK:-}" = "old" ]; then
  W=/scratch/dsv41/wt-bench
  export PYTHONPATH=/scratch/dsv41/sgk-k3rev:$W/python:/scratch/dsv41/aiter
  export AITER_JIT_DIR=/scratch/agentx/cache-old/aiter-jit TRITON_CACHE_DIR=/scratch/agentx/cache-old/triton
  export TORCHINDUCTOR_CACHE_DIR=/scratch/agentx/cache-old/inductor SGLANG_JIT_CACHE_DIR=/scratch/agentx/cache-old/sglang-jit
fi
# --- published env, verbatim ---
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING=1 SGLANG_DSV41_REASONING_EFFORT=high
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=0 TRITON_HIP_USE_ASYNC_COPY=0
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE AITER_BF16_FP8_MOE_BOUND=0
export TORCH_BLAS_PREFER_HIPBLASLT=1 HSA_NO_SCRATCH_RECLAIM=0 GPU_MAX_HW_QUEUES=2
if [ "$TP" = "2" ]; then GPUS=2,3; else GPUS=2,3,4,5; fi
PAR=(--tensor-parallel-size $TP)
# Bisect knobs (defaults reproduce the published recipe):
#   EP=4      TP attention + EP MoE (TEP4), no DP attention
#   DEPLITE=1 DEP4 without --enable-dp-lm-head / local-control-broadcast
#   FUSE=0    drop --enforce-shared-experts-fusion (shared expert unfused, as EP forces)
#   SPEC=0    no DSPARK speculative decoding
[ -n "${EP:-}" ] && PAR+=(--expert-parallel-size $EP)
if [ "${DEP:-0}" = "1" ] && [ "${DEPLITE:-0}" = "1" ]; then
  # DSpark + DP attention requires --enable-dp-lm-head; drop only local-control-broadcast
  PAR=(--tensor-parallel-size 4 --expert-parallel-size 4 --attn-dp-size 4 --enable-dp-lm-head
       --load-balance-method total_requests --tokenizer-worker-num 8)
  GPUS=2,3,4,5
elif [ "${DEP:-0}" = "1" ]; then
  PAR=(--tensor-parallel-size 4 --expert-parallel-size 4 --attn-dp-size 4
       --enable-dp-lm-head --enable-dp-attention-local-control-broadcast
       --load-balance-method total_requests --tokenizer-worker-num 8)
  GPUS=2,3,4,5
fi
# REPLAY=0 drops --enable-decoder-swa-bounded-replay. Required for DEP: main
# rejects it with DP attention ("cannot be combined with DP attention yet").
# sgl-project/sglang#43077 lifts that only for DP attention + TP MoE with
# attention-TP 1, not for DEP4's EP MoE.
if [ "${REPLAY:-1}" = "1" ]; then REPLAY_FLAG=--enable-decoder-swa-bounded-replay; else REPLAY_FLAG=; fi
OPT=()
SPECF=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
[ "${SPEC:-1}" = "0" ] && SPECF=()
FUSEF=(--enforce-shared-experts-fusion)
[ "${FUSE:-1}" = "0" ] && FUSEF=()
[ -n "${SCHED:-}" ] && OPT+=(--schedule-policy "$SCHED")
# EXTRA: free-form extra server args for bisects (word-split)
[ -n "${EXTRA:-}" ] && OPT+=($EXTRA)
cd /tmp
HIP_VISIBLE_DEVICES=$GPUS exec python -m sglang.launch_server \
  --model-path /scratch/models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  "${PAR[@]}" \
  "${SPECF[@]}" \
  "${FUSEF[@]}" --fp8-gemm-backend aiter \
  --cuda-graph-max-bs-decode ${GRAPHBS:-64} \
  ${REPLAY_FLAG} --cuda-graph-backend-prefill disabled \
  --mem-fraction-static ${MEMFRAC:-0.7} --chunked-prefill-size ${CHUNK:-16384} \
  --prefill-decode-interval ${PDI:-16} --swa-prefix-tails ${TAILS:-1024} \
  --max-running-requests ${MAXRUN:-32} \
  "${OPT[@]}" \
  --reasoning-parser auto --tool-call-parser auto \
  --watchdog-timeout 3600 --enable-metrics --host 127.0.0.1 --port ${SRVPORT:-30000}
