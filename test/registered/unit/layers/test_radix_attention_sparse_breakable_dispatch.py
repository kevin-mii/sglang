"""RadixAttention picks the eager-break sparse wrapper under a breakable graph.

Under ``--cuda-graph-backend-prefill breakable`` the MiniMax-M3 sparse
attention (``idx_q`` kwarg) must run as an ``eager_on_graph`` break, like the
dense path, instead of being captured into the graph segments: its per-batch
metadata (cu_seqlens, page layout, top-k block tables) would otherwise be
replayed from the capture batch for every later batch.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import radix_attention as ra
from sglang.test.test_utils import CustomTestCase


def _make_layer() -> ra.RadixAttention:
    layer = ra.RadixAttention.__new__(ra.RadixAttention)
    torch.nn.Module.__init__(layer)
    layer.tp_q_head_num = 2
    layer.tp_k_head_num = 1
    layer.tp_v_head_num = 1
    layer.qk_head_dim = 8
    layer.v_head_dim = 8
    layer.layer_id = 3
    return layer


class TestRadixAttentionSparseBreakableDispatch(CustomTestCase):
    def _dispatch(self, in_breakable_graph: bool):
        layer = _make_layer()
        forward_batch = MagicMock()
        forward_batch.forward_mode.is_extend.return_value = True
        q = torch.zeros(4, 2 * 8)
        k = torch.zeros(4, 1 * 8)
        v = torch.zeros(4, 1 * 8)
        idx_q = torch.zeros(4, 1, 8)
        idx_k = torch.zeros(4, 1 * 8)
        breakable = MagicMock(name="breakable_unified_sparse_attention_with_output")
        captured = MagicMock(name="unified_sparse_attention_with_output")
        with (
            patch.object(ra, "get_tc_piecewise_forward_context", return_value=object()),
            patch.object(
                ra, "is_in_breakable_cuda_graph", return_value=in_breakable_graph
            ),
            patch.object(
                ra, "breakable_unified_sparse_attention_with_output", breakable
            ),
            patch.object(ra, "unified_sparse_attention_with_output", captured),
        ):
            idx_out, attn_out = layer.forward(
                q, k, v, forward_batch, save_kv_cache=True, idx_q=idx_q, idx_k=idx_k
            )
        self.assertEqual(tuple(attn_out.shape), (4, 2 * 8))
        self.assertEqual(tuple(idx_out.shape), (4, 1 * 8))
        return breakable, captured

    def test_breakable_graph_uses_eager_break_wrapper(self):
        breakable, captured = self._dispatch(in_breakable_graph=True)
        breakable.assert_called_once()
        captured.assert_not_called()
        self.assertEqual(breakable.call_args.args[8], 3)  # layer_id threaded through

    def test_outside_breakable_graph_uses_custom_op(self):
        breakable, captured = self._dispatch(in_breakable_graph=False)
        captured.assert_called_once()
        breakable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
