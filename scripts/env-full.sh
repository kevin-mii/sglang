# source /scratch/dsv41/env-full.sh
# SGLang ROCm 10.0.0 (pip SDK) env for gfx950, mirroring docker/rocm.Dockerfile
# GPU_ARCH=gfx950-rocm1000. /opt/rocm on this host is ROCm 7.0 and is deliberately
# kept OFF every search path; everything resolves to the pip SDK's _rocm_sdk_devel.

export DSV41=/scratch/dsv41
export VIRTUAL_ENV=$DSV41/venv
_SP=$VIRTUAL_ENV/lib/python3.12/site-packages

# ROCm SDK (rocm1000-base stage)
export ROCM_HOME=$_SP/_rocm_sdk_devel
export ROCM_PATH=$ROCM_HOME
export HIP_PATH=$ROCM_HOME
export CPATH=$ROCM_HOME/include
export LIBRARY_PATH=$ROCM_HOME/lib
# Replace (not prepend) the inherited LD_LIBRARY_PATH/PATH: the container's
# defaults point at /opt/venv, /opt/conda, /opt/pytorch (ROCm 7.0 stack).
export LD_LIBRARY_PATH=$ROCM_HOME/lib
export PATH=$VIRTUAL_ENV/bin:$ROCM_HOME/llvm/bin:$ROCM_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME
hash -r 2>/dev/null

# Arch
export GPU_ARCH=gfx950-rocm1000
export GPU_ARCH_LIST=gfx950
export GPU_ARCHS=gfx950          # AITER JIT target (skips rocminfo-based detection for builds)
export PYTORCH_ROCM_ARCH=gfx950
export AMDGPU_TARGET=gfx950

# Keep pip from replacing the ROCm torch stack (Dockerfile PIP_CONSTRAINT)
export PIP_CONSTRAINT=$DSV41/constraints/torch-rocm.txt
export PIP_CACHE_DIR=$DSV41/pip-cache
export UV_CACHE_DIR=$DSV41/uv-cache
export UV_PYTHON_INSTALL_DIR=$DSV41/uv-python

# AITER
export PYTHONPATH=$DSV41/aiter
export AITER_USE_SYSTEM_TRITON=1

# Caches
export HF_HOME=/scratch/hf_home
export XDG_CACHE_HOME=$DSV41/cache/xdg
export TRITON_CACHE_DIR=$DSV41/cache/triton
export AITER_JIT_DIR=$DSV41/cache/aiter-jit
export SGLANG_JIT_CACHE_DIR=$DSV41/cache/sglang-jit
export TORCHINDUCTOR_CACHE_DIR=$DSV41/cache/inductor
export TILELANG_CACHE_DIR=$DSV41/cache/tilelang
export TVM_FFI_CACHE_DIR=$DSV41/cache/tvm-ffi
export CARGO_TARGET_DIR=$DSV41/cache/cargo-target
mkdir -p "$TRITON_CACHE_DIR" "$AITER_JIT_DIR" "$SGLANG_JIT_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TILELANG_CACHE_DIR" "$TVM_FFI_CACHE_DIR" "$XDG_CACHE_HOME" 2>/dev/null

# Locale (container sets LC_ALL=en_US.UTF-8 which is not generated here)
export LC_ALL=C.UTF-8 LANG=C.UTF-8

# Runtime env from the Dockerfile's final stage
export SGLANG_DISABLE_CUDNN_CHECK=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_INT4_WEIGHT=0
export SGLANG_MOE_PADDING=1
export SGLANG_ROCM_DISABLE_LINEARQUANT=0
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export NCCL_MIN_NCHANNELS=112
export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
export LIBGL_ALWAYS_INDIRECT=1

unset _SP
