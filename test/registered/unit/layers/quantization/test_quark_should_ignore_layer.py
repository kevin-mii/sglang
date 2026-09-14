"""Unit tests for quark's should_ignore_layer on packed (fused) layers.

Quark checkpoints list excluded layers per unfused projection name. For a
fused layer (qkv_proj -> [q_proj, k_proj, v_proj]) every shard must agree on
the scheme, but a shard whose projection name never appears in the exclude
list is one the checkpoint does not carry at all (MiniMax-M3's
``index_v_proj`` on value-disabled indexer layers). Such a shard must not
veto its excluded siblings.
"""

import unittest
from types import MappingProxyType

from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FUSED = MappingProxyType(
    {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "index_qkv_proj": ["index_q_proj", "index_k_proj", "index_v_proj"],
    }
)


class TestQuarkShouldIgnoreLayer(CustomTestCase):
    def test_unfused_layer_matches_exclude_list(self):
        ignore = ["model.layers.0.self_attn.o_proj"]
        self.assertTrue(
            should_ignore_layer("model.layers.0.self_attn.o_proj", ignore, FUSED)
        )
        self.assertFalse(
            should_ignore_layer("model.layers.1.self_attn.o_proj", ignore, FUSED)
        )

    def test_fused_layer_all_shards_excluded(self):
        ignore = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        self.assertTrue(
            should_ignore_layer("model.layers.0.self_attn.qkv_proj", ignore, FUSED)
        )

    def test_fused_layer_mixed_schemes_raise(self):
        # k_proj appears in the exclude list for another layer, so the shard
        # projection is known to the checkpoint: a real scheme mismatch.
        ignore = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.5.self_attn.k_proj",
        ]
        with self.assertRaises(ValueError):
            should_ignore_layer("model.layers.0.self_attn.qkv_proj", ignore, FUSED)

    def test_absent_shard_does_not_veto_excluded_siblings(self):
        # index_v_proj never appears in the exclude list: the checkpoint has
        # no such weight (value-disabled indexer layers). The fused layer
        # follows its excluded siblings instead of raising.
        ignore = [
            "model.layers.3.self_attn.indexer.index_q_proj",
            "model.layers.3.self_attn.indexer.index_k_proj",
        ]
        self.assertTrue(
            should_ignore_layer(
                "model.layers.3.self_attn.indexer.index_qkv_proj", ignore, FUSED
            )
        )

    def test_absent_shard_first_still_follows_siblings(self):
        # Same case with the absent projection listed first in the mapping.
        fused = MappingProxyType({"index_qkv_proj": ["index_v_proj", "index_q_proj"]})
        ignore = ["model.layers.3.self_attn.indexer.index_q_proj"]
        self.assertTrue(
            should_ignore_layer(
                "model.layers.3.self_attn.indexer.index_qkv_proj", ignore, fused
            )
        )


if __name__ == "__main__":
    unittest.main()
