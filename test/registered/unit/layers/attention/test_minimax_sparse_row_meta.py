"""Unit tests for the flattened-row metadata helpers of the MiniMax-M3 sparse
attention backend (EAGLE chain verify and small extends on CUDA/ROCm)."""

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


class TestFlattenedExtendRowMeta(CustomTestCase):
    def test_equal_extends_pack(self):
        meta = flattened_extend_row_meta(
            torch.tensor([4, 9], dtype=torch.int32),
            prefix_lens=[100, 20],
            extend_lens=[2, 2],
            seq_lens_dtype=torch.int32,
        )
        self.assertEqual(meta.per_query_req.tolist(), [4, 4, 9, 9])
        self.assertEqual(meta.per_query_seq_lens.tolist(), [101, 102, 21, 22])
        self.assertEqual(meta.max_seqlen, 102)
        self.assertEqual(meta.packed, 2)
        self.assertEqual(meta.rows, 4)

    def test_ragged_extends_do_not_pack(self):
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
