# env for the out-of-tree split: sglang from $W (default the mono-oot worktree), sgl-amd-mono
# from $OOT, its entry points from a generated dist-info (no pip install)
S=/scratch/tmp/claude-0/-sgl-workspace/4bb9f594-e169-4b38-90ad-4a5cb2483512/scratchpad
source /scratch/dsv41/env-full.sh >/dev/null 2>&1
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
W=${W:-/scratch/agentx/mono-oot}; OOT=${OOT:-/scratch/agentx/sgl-amd-mono}
export PYTHONPATH=/scratch/agentx/main-overlay:/scratch/dsv41/sgk-k3rev:$W/python:$OOT/python:$OOT:${OOT_SITE:-$S/oot_site}:/scratch/agentx/aiter-main
export AITER_JIT_DIR=/scratch/agentx/cache-main/aiter-jit TRITON_CACHE_DIR=/scratch/agentx/cache-main/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/agentx/cache-main/inductor SGLANG_JIT_CACHE_DIR=/scratch/agentx/cache-main/sglang-jit
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=0 AITER_BF16_FP8_MOE_BOUND=0
export DEEPSEEK_V41_FLASH_MODEL_PATH=/scratch/models/DeepSeek-V4.1-Flash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
