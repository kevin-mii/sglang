"""Token-block-parallel page-table copies for speculative decoding.

``create_flashinfer_kv_indices_triton`` (EAGLE verify / draft-extend
``kv_indices``) and ``generate_draft_decode_kv_indices`` (Triton multi-step
draft page tables) both have a mode that spreads one request's copy over
several programs. The Triton draft backend and the EAGLE metadata builders
now use it above ``KV_INDICES_TOKEN_BLOCKS_MIN_WIDTH``; the outputs must be
bit-identical to the single-program copy, and the launch helper must keep
short tables on the historical path.
"""

import unittest

import torch

from sglang.kernels.ops.kvcache.kv_indices import (
    KV_INDICES_TOKEN_BLOCKS_MIN_WIDTH,
    create_flashinfer_kv_indices_triton,
    kv_indices_num_token_blocks,
    kv_indices_token_blocks_for_copy,
)
from sglang.kernels.ops.speculative.cache_locs import generate_draft_decode_kv_indices
from sglang.srt.utils import get_device, next_power_of_2
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=40, stage="jit-kernel-unit", runner_config="amd")

MAX_BS = 16


class TestKvIndicesTokenBlockParallel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        torch.manual_seed(0)

    def test_launch_helper_thresholds(self):
        self.assertEqual(kv_indices_token_blocks_for_copy(4096, 8), 1)
        self.assertEqual(
            kv_indices_token_blocks_for_copy(KV_INDICES_TOKEN_BLOCKS_MIN_WIDTH - 1, 1),
            1,
        )
        wide = kv_indices_token_blocks_for_copy(KV_INDICES_TOKEN_BLOCKS_MIN_WIDTH, 1)
        self.assertEqual(
            wide, kv_indices_num_token_blocks(KV_INDICES_TOKEN_BLOCKS_MIN_WIDTH, 1)
        )
        self.assertGreater(wide, 1)

    def _tables(self, bs: int, max_len: int, table_width: int):
        dev = self.device
        req_to_token = torch.randint(
            1, 1 << 22, (MAX_BS, table_width), dtype=torch.int32, device=dev
        )
        req_pool_indices = torch.randperm(MAX_BS, device=dev)[:bs].to(torch.int32)
        seq_lens = torch.randint(1, max_len, (bs,), dtype=torch.int32, device=dev)
        seq_lens[0] = max_len
        return req_to_token, req_pool_indices, seq_lens

    def _check_kv_indices(self, bs: int, max_len: int, table_width: int):
        dev = self.device
        req_to_token, req_pool_indices, seq_lens = self._tables(
            bs, max_len, table_width
        )
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
        kv_indptr[1:] = torch.cumsum(seq_lens, 0)
        total = int(kv_indptr[-1].item())
        serial = torch.empty(total, dtype=torch.int32, device=dev)
        parallel = torch.empty_like(serial)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            serial,
            table_width,
        )
        blocks = kv_indices_num_token_blocks(table_width, bs)
        self.assertGreater(blocks, 1)
        create_flashinfer_kv_indices_triton[(bs, blocks)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            parallel,
            table_width,
            TOKEN_BLOCK_PARALLEL=True,
        )
        torch.testing.assert_close(parallel, serial, rtol=0, atol=0)

    def _check_draft_page_tables(self, bs: int, max_len: int, table_width: int):
        dev = self.device
        steps, topk, page_size = 3, 1, 1
        req_to_token, req_pool_indices, seq_lens = self._tables(
            bs, max_len, table_width
        )
        width = bs * topk * (max_len + steps) + 64
        kv_serial = torch.zeros((steps, width), dtype=torch.int64, device=dev)
        kv_parallel = torch.zeros_like(kv_serial)
        indptr_serial = torch.zeros(
            (steps, bs * topk + 1), dtype=torch.int32, device=dev
        )
        indptr_parallel = torch.zeros_like(indptr_serial)
        positions = torch.zeros(bs * topk, dtype=torch.int64, device=dev)

        def args(kv, indptr):
            return (
                req_pool_indices,
                req_to_token,
                seq_lens,
                kv,
                indptr,
                positions,
                table_width,
                width,
                bs * topk + 1,
                next_power_of_2(bs),
                next_power_of_2(steps),
                next_power_of_2(bs * topk),
                page_size,
            )

        generate_draft_decode_kv_indices[(steps, bs, topk)](
            *args(kv_serial, indptr_serial)
        )
        blocks = kv_indices_num_token_blocks(table_width, steps * bs * topk)
        self.assertGreater(blocks, 1)
        generate_draft_decode_kv_indices[(steps * blocks, bs, topk)](
            *args(kv_parallel, indptr_parallel), NUM_STEPS=steps
        )
        torch.testing.assert_close(indptr_parallel, indptr_serial, rtol=0, atol=0)
        torch.testing.assert_close(kv_parallel, kv_serial, rtol=0, atol=0)

    def test_kv_indices_long_context(self):
        self._check_kv_indices(bs=3, max_len=60_000, table_width=65_536)
        self._check_kv_indices(bs=1, max_len=200_000, table_width=262_144)

    def test_kv_indices_ragged_batch(self):
        self._check_kv_indices(bs=7, max_len=40_000, table_width=65_536)

    def test_draft_page_tables_long_context(self):
        self._check_draft_page_tables(bs=3, max_len=60_000, table_width=65_536)
        self._check_draft_page_tables(bs=1, max_len=200_000, table_width=262_144)

    def test_draft_page_tables_ragged_batch(self):
        self._check_draft_page_tables(bs=5, max_len=40_000, table_width=65_536)


if __name__ == "__main__":
    unittest.main()
