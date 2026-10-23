"""Build mla_fp8._C: the gfx950 fp8 MLA prefill kernel (csrc/mla_fp8.hip) and its torch bindings."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
HIP_FLAGS = ["-O3", "-std=c++17", "-ffast-math", "-fno-math-errno", "-mllvm", "-amdgpu-early-inline-all=true",
             "-fno-slp-vectorize", "--offload-arch=gfx950"] + os.environ.get("MLA8_FLAGS", "").split()

setup(
    name="mla-fp8",
    version="0.1.0",
    packages=["mla_fp8"],
    ext_modules=[CUDAExtension(name="mla_fp8._C", sources=["csrc/mla_fp8_api.cpp", os.environ.get("MLA8_KERNEL", "csrc/mla_fp8.hip")],
                               include_dirs=[str(ROOT / "csrc")],
                               extra_compile_args={"cxx": ["-O3"], "nvcc": HIP_FLAGS})],
    cmdclass={"build_ext": BuildExtension},
)
