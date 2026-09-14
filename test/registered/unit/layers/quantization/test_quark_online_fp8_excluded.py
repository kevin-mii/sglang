"""The online-FP8 skip list must match whole module names."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bare_config() -> QuarkConfig:
    # no quantized groups, nothing excluded by name
    return QuarkConfig(quant_config={"packed_modules_mapping": {}, "exclude": []})


class TestQuarkOnlineFp8Excluded(CustomTestCase):
    def test_disabled_by_default(self):
        cfg = _bare_config()
        with envs.SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED.override(False):
            self.assertFalse(
                cfg._online_fp8_for_excluded("model.layers.3.self_attn.qkv_proj")
            )

    def test_skip_modules_stay_bf16(self):
        cfg = _bare_config()
        with envs.SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED.override(True):
            self.assertTrue(
                cfg._online_fp8_for_excluded("model.layers.3.self_attn.qkv_proj")
            )
            self.assertTrue(
                cfg._online_fp8_for_excluded("model.layers.3.self_attn.o_proj")
            )
            # dense MLP of the first layers: quantized ("gate_proj" != "gate")
            self.assertTrue(
                cfg._online_fp8_for_excluded("model.layers.0.mlp.gate_proj")
            )
            # the indexer projections are quantized too
            self.assertTrue(
                cfg._online_fp8_for_excluded("model.layers.3.self_attn.index_qkv_proj")
            )
            # router gate and lm_head stay bf16
            self.assertFalse(
                cfg._online_fp8_for_excluded("model.layers.3.block_sparse_moe.gate")
            )
            self.assertFalse(cfg._online_fp8_for_excluded("lm_head"))

    def test_custom_skip_list(self):
        cfg = _bare_config()
        with envs.SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED.override(True):
            with envs.SGLANG_QUARK_ONLINE_FP8_SKIP_MODULES.override("o_proj"):
                self.assertFalse(
                    cfg._online_fp8_for_excluded("model.layers.3.self_attn.o_proj")
                )
                self.assertTrue(
                    cfg._online_fp8_for_excluded("model.layers.3.block_sparse_moe.gate")
                )

    def test_online_config_is_dynamic_unserialized(self):
        cfg = _bare_config()
        fp8 = cfg._online_fp8_config
        self.assertIsInstance(fp8, Fp8Config)
        self.assertFalse(fp8.is_checkpoint_fp8_serialized)
        self.assertEqual(fp8.activation_scheme, "dynamic")
        self.assertIsNone(fp8.weight_block_size)
        self.assertIs(cfg._online_fp8_config, fp8)


if __name__ == "__main__":
    unittest.main()
