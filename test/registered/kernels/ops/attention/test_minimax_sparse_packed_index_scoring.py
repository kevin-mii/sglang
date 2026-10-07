"""Packed per-request index scoring must select the same blocks as per-row scoring."""

import unittest

import torch

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

# MiniMax-M3 sparse layers: 128-token blocks, top-16, no init block, one local block.
BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
NUM_DRAFT_TOKENS = 4


class TestPackedIndexScoring(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()

    def _run(
        self,
        prefixes,
        *,
        req_slots,
        num_q_heads=1,
        pack=NUM_DRAFT_TOKENS,
        k_dtype=torch.bfloat16,
        score_type="max",
        local_blocks=1,
        radix_topk=True,
        padding_requests=0,
    ):
        dev = self.device
        gen = torch.Generator(device=dev).manual_seed(len(prefixes) * 1000 + pack)
        max_len = max(prefixes) + pack
        num_table_rows = max(req_slots) + 1
        # Permuted token slots: every request reads a scattered slice of the pool.
        num_slots = num_table_rows * max_len + 1
        req_to_token = (
            torch.randperm(num_slots - 1, device=dev, generator=gen)[
                : num_table_rows * max_len
            ]
            .add_(1)
            .to(torch.int32)
            .reshape(num_table_rows, max_len)
        )
        k_cache = torch.randn(
            num_slots, 1, HEAD_DIM, device=dev, generator=gen, dtype=torch.float32
        )
        rows = (len(prefixes) + padding_requests) * pack
        q = torch.randn(
            rows, num_q_heads, HEAD_DIM, device=dev, generator=gen, dtype=torch.float32
        )
        # Draft keys point away from the request's queries: a shorter row's own
        # local block then scores low unless it is forced, and the shared tail
        # block cannot win on keys the row must not see.
        for r, (prefix, slot) in enumerate(zip(prefixes, req_slots)):
            away = -q[r * pack : (r + 1) * pack].reshape(-1, HEAD_DIM).sum(0) * 10.0
            for j in range(1, pack):
                k_cache[req_to_token[slot, prefix + j].long(), 0] = away
        k_cache = k_cache.to(k_dtype)
        q = q.to(torch.bfloat16)
        seq_lens = [p + j + 1 for p in prefixes for j in range(pack)]
        slot_ids = [s for s in req_slots for _ in range(pack)]
        # CUDA-graph padding requests: seq_lens fill value 1 plus the draft offsets.
        seq_lens += [1 + j + 1 for _ in range(padding_requests) for j in range(pack)]
        slot_ids += [0] * (padding_requests * pack)
        kwargs = dict(
            sink=None,
            k_cache=k_cache,
            v_cache=None,
            req_to_token=req_to_token,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=dev),
            max_seqlen=max_len,
            slot_ids=torch.tensor(slot_ids, dtype=torch.int64, device=dev),
            block_size=BLOCK_SIZE,
            topk=TOPK,
            init_blocks=0,
            local_blocks=local_blocks,
            score_type=score_type,
            disable_index_value=True,
            page_size=1,
        )
        with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(radix_topk):
            _, ref_idx, _ = flash_decode_with_topk_idx(q, **kwargs, packed_queries=1)
            _, packed_idx, _ = flash_decode_with_topk_idx(
                q, **kwargs, packed_queries=pack
            )
        self.assertEqual(ref_idx.shape, (num_q_heads, rows, TOPK))
        self.assertEqual(ref_idx.shape, packed_idx.shape)
        ref_sets = ref_idx.sort(dim=-1).values
        packed_sets = packed_idx.sort(dim=-1).values
        mismatched = (ref_sets != packed_sets).any(-1).nonzero().tolist()
        self.assertEqual(mismatched, [], "(head, row) pairs with different top-k")

    def test_m3_verify_rows_straddling_block_boundaries(self):
        """A shorter row must keep its own local block when the draft tail crosses into the next block."""
        prefixes = [BLOCK_SIZE * 40 - 2, BLOCK_SIZE * 23 - 1, 7777, BLOCK_SIZE * 60]
        for radix_topk in (True, False):
            with self.subTest(radix_topk=radix_topk):
                self._run(prefixes, req_slots=[5, 2, 7, 1], radix_topk=radix_topk)

    def test_fp8_index_k(self):
        """An fp8 index K cache (widened to the bf16 query on load) packs the same way."""
        prefixes = [BLOCK_SIZE * 31 - 3, 4321, BLOCK_SIZE * 17 - 2]
        for k_dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            with self.subTest(k_dtype=k_dtype):
                self._run(prefixes, req_slots=[3, 6, 0], k_dtype=k_dtype)

    def test_graph_padding_rows(self):
        """Padding requests (short rows on slot 0) next to long ones must not disturb either."""
        self._run([BLOCK_SIZE * 50 - 2, 3000], req_slots=[4, 9], padding_requests=2)

    def test_multi_head_and_lse(self):
        """Several index heads per row (TP < 4) and the lse score share the un-pack."""
        self._run([BLOCK_SIZE * 25 - 1, 5000], req_slots=[1, 3], num_q_heads=4)
        self._run(
            [BLOCK_SIZE * 25 - 1, 5000], req_slots=[1, 3], score_type="lse", pack=3
        )

    def test_no_local_blocks_falls_back_to_per_row(self):
        """Without local blocks packing cannot be exact, so it must be skipped."""
        self._run([BLOCK_SIZE * 25 - 1, 5000], req_slots=[1, 3], local_blocks=0)


class TestSharedTopkPublish(CustomTestCase):
    def test_source_layer_publishes_packed_topk(self):
        """Reuse layers must read exactly the source layer's selection from the shared buffer."""
        dev = get_device()
        gen = torch.Generator(device=dev).manual_seed(0)
        prefixes, req_slots = [BLOCK_SIZE * 30 - 2, 6000], [2, 1]
        max_len = max(prefixes) + NUM_DRAFT_TOKENS
        num_slots = len(prefixes) * max_len + 1
        req_to_token = torch.zeros(3, max_len, dtype=torch.int32, device=dev)
        perm = torch.randperm(num_slots - 1, device=dev, generator=gen) + 1
        req_to_token[req_slots] = perm.to(torch.int32).view(len(prefixes), max_len)
        rows = len(prefixes) * NUM_DRAFT_TOKENS

        def randn(*shape):
            return torch.randn(*shape, device=dev, generator=gen).to(torch.bfloat16)

        q, idx_q = randn(rows, 16, HEAD_DIM), randn(rows, 1, HEAD_DIM)
        k_cache, v_cache, idx_k_cache = (randn(num_slots, 1, HEAD_DIM) for _ in "kvi")
        args = (q, None, k_cache, v_cache, idx_q, None, idx_k_cache, None)
        kwargs = dict(
            req_to_token=req_to_token,
            slot_ids=torch.tensor(req_slots, device=dev).repeat_interleave(
                NUM_DRAFT_TOKENS
            ),
            seq_lens=torch.tensor(
                [p + j + 1 for p in prefixes for j in range(NUM_DRAFT_TOKENS)],
                device=dev,
            ),
            max_seqlen=max_len,
            block_size_q=1,
            block_size_k=BLOCK_SIZE,
            topk=TOPK,
            init_blocks=0,
            local_blocks=1,
            disable_index_value=True,
            packed_queries=NUM_DRAFT_TOKENS,
        )
        _, ref_o = minimax_sparse_decode(*args, **kwargs)
        shared = torch.full((1, rows, TOPK), -7, dtype=torch.int32, device=dev)
        _, source_o = minimax_sparse_decode(*args, **kwargs, topk_out=shared)
        _, reuse_o = minimax_sparse_decode(*args, **kwargs, cached_topk_idx=shared)
        torch.testing.assert_close(source_o, ref_o, rtol=0, atol=0)
        torch.testing.assert_close(reuse_o, ref_o, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
