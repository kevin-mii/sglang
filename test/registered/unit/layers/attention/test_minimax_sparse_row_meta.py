"""The flattened-row metadata must be request-major with causal lengths."""

import unittest

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.row_meta import (
    chain_verify_row_meta,
    flattened_extend_row_meta,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChainVerifyRowMeta(CustomTestCase):
    def test_rows_are_request_major_and_causal(self):
        """A draft-major order or an off-by-one length would attend the wrong KV."""
        prefix = torch.tensor([10, 3], dtype=torch.int64)
        req_pool_indices = torch.tensor([7, 2], dtype=torch.int32)
        per_query_req, per_query_seq_lens = chain_verify_row_meta(
            prefix, req_pool_indices, num_draft_tokens=3
        )
        self.assertEqual(per_query_req.tolist(), [7, 7, 7, 2, 2, 2])
        self.assertEqual(per_query_seq_lens.tolist(), [11, 12, 13, 4, 5, 6])
        self.assertEqual(per_query_seq_lens.dtype, torch.int32)


class TestFlattenedExtendRowMeta(CustomTestCase):
    def test_ragged_extends_do_not_pack(self):
        """Unequal extends must fall back to per-row scoring and still get causal lengths."""
        meta = flattened_extend_row_meta(
            torch.tensor([1, 0], dtype=torch.int32),
            prefix_lens=[8, 8],
            extend_lens=[3, 1],
            seq_lens_dtype=torch.int64,
        )
        self.assertEqual(meta.per_query_req.tolist(), [1, 1, 1, 0])
        self.assertEqual(meta.per_query_seq_lens.tolist(), [9, 10, 11, 9])
        self.assertEqual(meta.per_query_seq_lens.dtype, torch.int64)
        self.assertEqual(meta.max_seqlen, 11)
        self.assertEqual(meta.packed, 1)
        self.assertEqual(meta.rows, 4)


if __name__ == "__main__":
    unittest.main()
