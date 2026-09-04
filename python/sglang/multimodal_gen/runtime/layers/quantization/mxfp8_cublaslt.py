# SPDX-License-Identifier: Apache-2.0
"""``--quantization mxfp8_cublaslt``: online MXFP8 (block-32 e8m0 scales) on cuBLASLt.

Weights quantize once at load (per-32 groups along K, scales in the
``SWIZZLE_32_4_4`` layout); activations quantize in one Triton pass
(``mxfp8_quantize_swizzled``) unless the producer already hands over a
prequantized ``(fp8, scales)`` tuple; the GEMM is
``torch.nn.functional.scaled_mm(..., ScalingType.BlockWise1x32)``, i.e. cuBLASLt
on SM100, which runs MXFP8 at the per-tensor fp8 rate. Layers with K % 32 != 0
or N % 16 != 0 keep the ``Fp8LinearMethod`` path."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from sglang.kernels.ops.diffusion import mxfp8_quantize_swizzled
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)

_E8M0 = torch.float8_e8m0fnu


class Mxfp8CublasltConfig(Fp8Config):
    @classmethod
    def get_name(cls) -> str:
        return "mxfp8_cublaslt"

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, Fp8LinearMethod):
            return Mxfp8CublasltLinearMethod(self)
        return method


class Mxfp8CublasltLinearMethod(Fp8LinearMethod):
    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight.data
        layer.mxfp8_cublaslt = (
            not self.quant_config.is_checkpoint_fp8_serialized
            and not self.block_quant
            and not self.use_marlin
            and weight.is_cuda
            and weight.dtype in (torch.bfloat16, torch.float16)
            and weight.shape[1] % 32 == 0
            and weight.shape[0] % 16 == 0
        )
        if not layer.mxfp8_cublaslt:
            super().process_weights_after_loading(layer)
            return
        qweight, scales = mxfp8_quantize_swizzled(
            weight.contiguous().to(torch.bfloat16)
        )
        # [N, K] fp8 row-major; scaled_mm takes weight.t() as the [K, N] operand
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.weight_scale = Parameter(scales, requires_grad=False)
        layer.input_scale = None

    def accepts_mxfp8_input(self, layer: Module) -> bool:
        return bool(getattr(layer, "mxfp8_cublaslt", False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not getattr(layer, "mxfp8_cublaslt", False):
            return super().apply(layer, x, bias)
        if isinstance(x, tuple):
            qinput, x_scale = x
            lead_shape = None
        else:
            lead_shape = x.shape[:-1]
            x2d = x.reshape(-1, x.shape[-1])
            if x2d.dtype != torch.bfloat16:
                x2d = x2d.to(torch.bfloat16)
            qinput, x_scale = mxfp8_quantize_swizzled(x2d.contiguous())
        out = F.scaled_mm(
            qinput,
            layer.weight.t(),
            x_scale.view(_E8M0),
            F.ScalingType.BlockWise1x32,
            layer.weight_scale.view(_E8M0),
            F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias,
            output_dtype=torch.bfloat16,
        )
        if lead_shape is not None and len(lead_shape) != 1:
            out = out.view(*lead_shape, out.shape[-1])
        return out


__all__ = ["Mxfp8CublasltConfig", "Mxfp8CublasltLinearMethod"]
