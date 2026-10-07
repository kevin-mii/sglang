"""Splitting the score-only prefill indexer over KV blocks must not change the top-k."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

# One TP4 rank of MiniMax-M3: one index head, top-16 blocks of 128, local block only.
DIM, BLOCK, TOPK = 128, 128, 16
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1
NSLOTS = 400_000


def _req_to_token(lens, page_size, device):
    """Shuffled pages; tokens are contiguous inside a page, as the allocator guarantees."""
    padded = [(n + page_size - 1) // page_size for n in lens]
    pages = torch.randperm(NSLOTS // page_size, device=device, dtype=torch.int32)
    table = torch.zeros((len(lens), max(lens)), dtype=torch.int32, device=device)
    start = 0
    for b, n in enumerate(lens):
        req_pages = pages[start : start + padded[b]]
        start += padded[b]
        slots = req_pages[:, None] * page_size + torch.arange(page_size, device=device)
        table[b, :n] = slots.flatten()[:n].to(torch.int32)
    return table


@unittest.skipUnless(torch.version.hip, "the score-only indexer is gfx950-only")
class TestMiniMaxScoreOnlyIndexSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            raise unittest.SkipTest("the score-only indexer requires gfx950")
        from sglang.kernels.ops.attention.minimax_sparse.prefill import (
            flash_with_topk_idx,
        )
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        cls.mod = flash_with_topk_idx
        torch.manual_seed(20261007)
        cls.k_cache = (torch.randn(NSLOTS, 1, DIM, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        cls.fp8_dtype = fp8_dtype

    def _topk(self, k_cache, table, lens, exts, page_size, target_programs):
        device = "cuda"
        prefixes = [n - e for n, e in zip(lens, exts)]
        cu = torch.tensor([0] + exts, dtype=torch.int32, device=device).cumsum(0)
        torch.manual_seed(1)
        q = (torch.randn(sum(exts), 1, DIM, device=device) * 0.3).to(torch.bfloat16)
        with patch.object(self.mod, "_SCORE_ONLY_TARGET_PROGRAMS", target_programs):
            _, topk_idx = self.mod.flash_prefill_with_topk_index(
                q=q,
                k_cache=k_cache,
                v_cache=None,
                sink=None,
                req_to_token=table,
                slot_ids=torch.arange(len(lens), dtype=torch.int32, device=device),
                cu_seqlens=cu.to(torch.int32),
                seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
                prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=device),
                max_seqlen_q=max(exts),
                max_seqlen_k=max(lens),
                block_size_q=1,
                block_size_k=BLOCK,
                topk=TOPK,
                init_blocks=INIT_BLOCKS,
                local_blocks=LOCAL_BLOCKS,
                disable_index_value=True,
                page_size=page_size,
            )
        return topk_idx

    def check(self, k_cache, cases, page_size):
        for lens, exts in cases:
            with self.subTest(lens=lens, exts=exts, page_size=page_size):
                table = _req_to_token(lens, page_size, "cuda")
                # A budget of 1 program degenerates to the unsplit grid.
                ref = self._topk(k_cache, table, lens, exts, page_size, 1)
                out = self._topk(k_cache, table, lens, exts, page_size, 2048)
                self.assertTrue(
                    torch.equal(ref, out), f"{(ref != out).sum()} top-k IDs differ"
                )

    def test_bf16_index_cache(self):
        cases = [
            ([190000], [10]),
            ([190000, 70000], [1536, 300]),
            ([40960], [8192]),
            ([6000, 5000, 4000], [64, 5000, 1]),
        ]
        self.check(self.k_cache, cases, page_size=1)

    def test_fp8_index_cache_per_page_slots(self):
        cases = [([190000], [10]), ([120000, 190000], [2048, 4096])]
        self.check(self.k_cache.to(self.fp8_dtype), cases, page_size=64)


if __name__ == "__main__":
    unittest.main()
