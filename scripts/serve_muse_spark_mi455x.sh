#!/usr/bin/env bash
# Muse Spark Open (Quark MXFP4) on one MI455X node (gfx1250 A0, 4 GPUs): TP4/EP4.
# Portable defaults: Triton attention and Triton MoE with aiter off. aiter's
# attention kernels are not validated on gfx1250 (see 3rdparty/amd/muse_spark).
# Override any of the variables below to try the MI355X fast paths.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/models/Muse-Spark-Open-816B-A42B-Quark-MXFP4}"
PORT="${PORT:-30000}"
ATTN="${ATTN:-triton}"
CTX="${CTX:-32768}"
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-4}"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
export MUSE_MOE_BACKEND="${MUSE_MOE_BACKEND:-triton}"
# A0 runtime rewrites B0 instructions; never disable hotswap on gfx1250 A0.
unset HSA_HOTSWAP_DISABLE NCCL_MIN_NCHANNELS

# Avoid resolving /sgl-workspace/aiter as a namespace package.
cd /tmp
exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size "${TP_SIZE}" --ep-size "${EP_SIZE}" \
  --chat-template "${MODEL}/chat_template.jinja" \
  --context-length "${CTX}" \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --attention-backend "${ATTN}" \
  --host 0.0.0.0 --port "${PORT}" \
  "$@"
