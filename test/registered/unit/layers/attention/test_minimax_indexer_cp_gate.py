import ast
import inspect
import unittest

from sglang.srt.layers.attention import minimax_sparse_backend
from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
    draft_is_chain_layout,
    unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The shape MiniMax-M3 CP was validated on; each case below perturbs one field.
SUPPORTED = dict(
    gfx950=True,
    tp_size=4,
    attn_tp_size=4,
    attn_cp_size=1,
    attn_dp_size=1,
    index_heads=4,
    kv_heads=4,
    head_dim=128,
    block_size=128,
    topk=16,
    score_type="max",
    max_context_len=1 << 20,
    radix_topk=True,
    draft_is_chain=True,
    tbo=False,
    hisparse=False,
    fp8_query=False,
    dense_sparse_decode=False,
)


class TestMiniMaxIndexerCPGate(CustomTestCase):
    def test_only_validated_draft_layouts_enable_cp(self):
        """The gate keyed on EAGLE's top-k, so every other algorithm read as a chain
        and enabled CP on a verify layout it cannot score (silently wrong top-k)."""
        enabled = {
            (None, None): True,
            ("EAGLE", 1): True,
            ("EAGLE3", 1): True,
            ("EAGLE3", 4): False,  # tree draft
            ("DSPARK", None): False,  # ragged verify lengths
            ("NGRAM", None): False,  # tree lives in the verify mask
            ("DFLASH", None): False,
            ("STANDALONE", 1): False,
        }
        for (algorithm, eagle_topk), expected in enabled.items():
            with self.subTest(algorithm=algorithm, eagle_topk=eagle_topk):
                reason = unsupported_reason(
                    **{
                        **SUPPORTED,
                        "draft_is_chain": draft_is_chain_layout(algorithm, eagle_topk),
                    }
                )
                self.assertEqual(reason is None, expected, reason)

    def test_each_unsupported_runtime_feature_disables_cp(self):
        for field, value in (
            ("gfx950", False),
            ("tp_size", 8),
            ("attn_cp_size", 2),
            ("kv_heads", 8),
            ("topk", 32),
            ("score_type", "sum"),
            ("radix_topk", False),
            ("tbo", True),
            ("hisparse", True),
            ("fp8_query", True),
            ("dense_sparse_decode", True),
        ):
            with self.subTest(field=field):
                self.assertIsNotNone(unsupported_reason(**{**SUPPORTED, field: value}))


class TestEveryDecodeCallSiteForwardsCP(CustomTestCase):
    def test_all_minimax_sparse_decode_calls_pass_indexer_cp(self):
        """CP reached only ordinary decode, so EAGLE verify -- the path it was written
        for -- silently kept scoring on one rank and the optimization did nothing."""
        tree = ast.parse(inspect.getsource(minimax_sparse_backend))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "minimax_sparse_decode"
        ]
        self.assertTrue(calls, "no minimax_sparse_decode calls found; test is stale")
        missing = [
            node.lineno
            for node in calls
            if not any(kw.arg == "indexer_cp" for kw in node.keywords)
        ]
        self.assertEqual(
            missing, [], f"call sites not forwarding indexer_cp: {missing}"
        )


if __name__ == "__main__":
    unittest.main()
