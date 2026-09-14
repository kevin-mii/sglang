"""Unit tests for the flattened-row metadata helpers of the MiniMax-M3 sparse
attention backend (EAGLE chain verify on CUDA/ROCm)."""

import unittest

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.row_meta import (
    chain_verify_row_meta,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChainVerifyRowMeta(CustomTestCase):
    def test_rows_are_request_major_and_causal(self):
        # GPU verify batches keep seq_lens at the prefix length; the ndt draft
        # slots follow it, so row j of a request attends prefix + j + 1 tokens.
        prefix = torch.tensor([10, 3], dtype=torch.int64)
        req_pool_indices = torch.tensor([7, 2], dtype=torch.int32)
        per_query_req, per_query_seq_lens = chain_verify_row_meta(
            prefix, req_pool_indices, num_draft_tokens=3
        )
        self.assertEqual(per_query_req.tolist(), [7, 7, 7, 2, 2, 2])
        self.assertEqual(per_query_seq_lens.tolist(), [11, 12, 13, 4, 5, 6])
        self.assertEqual(per_query_seq_lens.dtype, torch.int32)

    def test_single_draft_token(self):
        prefix = torch.tensor([5], dtype=torch.int32)
        per_query_req, per_query_seq_lens = chain_verify_row_meta(
            prefix, torch.tensor([0], dtype=torch.int32), num_draft_tokens=1
        )
        self.assertEqual(per_query_req.tolist(), [0])
        self.assertEqual(per_query_seq_lens.tolist(), [6])


if __name__ == "__main__":
    unittest.main()
