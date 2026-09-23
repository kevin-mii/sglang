#!/bin/bash
# Runs INSIDE the image (via bin/inimg). reproduce.sh `serve real` config, parameterized.
# Env: GPUS PORT MEMFRAC MAXRUN CHUNK STEPS DRAFT TOPK_FREQ EXTRA (extra CLI flags) LOG
: "${GPUS:=0,1,2,3}" "${PORT:=30001}" "${MEMFRAC:=0.85}" "${MAXRUN:=48}" "${CHUNK:=8192}" "${STEPS:=3}" "${DRAFT:=4}" "${TOPK_FREQ:=4}" "${EXTRA:=}" "${TP:=4}"
: "${LOG:=${M3_WORK:-/scratch}/logs/server_${PORT}.log}"
export HIP_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1
export SGLANG_USE_AITER=1 NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1
# inductor compile workers inherit /dev/kfd and fork; that drives amdgpu into slow SRCU teardown for every GPU client
export TORCHINDUCTOR_COMPILE_THREADS=1
export SGLANG_M3_ALLOW_CUSTOM_AR=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SGLANG_CUSTOM_AR_ONE_STAGE_MAX_BYTES=262144
export SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL=1 SGLANG_MINIMAX_OPT_USE_GLUON_PREFILL=1 SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=$TOPK_FREQ
export SGLANG_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1
export SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=${FAIRNESS:-0.5} SGLANG_TIMEOUT_KEEP_ALIVE=3600
{ echo "GIT: $(git -C $(cd "$(dirname "$0")/../../.." && pwd) rev-parse HEAD) $(date -u)"; env | grep -E "^(SGLANG|AITER|ROCM|HIP|NCCL|GPTOSS)_" | sort;
  echo "ARGS: tp=$TP mem=$MEMFRAC maxrun=$MAXRUN chunk=$CHUNK steps=$STEPS draft=$DRAFT extra=$EXTRA"; } > "$LOG"
SPEC="--speculative-algorithm EAGLE3 --speculative-draft-model-path ${M3_WORK:-/scratch}/models/MiniMax-M3-EAGLE3-GQA --speculative-num-steps $STEPS --speculative-eagle-topk 1 --speculative-num-draft-tokens $DRAFT --speculative-attention-mode decode"
[ "${NOSPEC:-}" = 1 ] && SPEC=""
exec python -m sglang.launch_server --model-path ${M3_WORK:-/scratch}/models/MiniMax-M3-MXFP4 --served-model-name MiniMax-M3 --trust-remote-code \
  --tp $TP --host 0.0.0.0 --port "$PORT" --kv-cache-dtype fp8_e4m3 --chunked-prefill-size $CHUNK --mem-fraction-static $MEMFRAC \
  $SPEC --triton-attention-num-kv-splits 64 \
  --cuda-graph-backend-prefill breakable --reasoning-parser auto --tool-call-parser auto --enable-metrics --enable-cache-report \
  --watchdog-timeout 3600 --max-running-requests $MAXRUN $EXTRA >> "$LOG" 2>&1
