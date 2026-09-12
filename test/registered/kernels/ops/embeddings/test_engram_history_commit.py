"""In-place history updates must match the concatenation oracle across replays."""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_hash import (
    engram_commit_decode_history,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


class TestEngramHistoryCommit(CustomTestCase):
    def test_decode_commit_matches_torch_chain(self):
        """The one-launch decode commit is ``history[where(out_loc == 0, pad, slots)] =
        tokens[:, :n-1].flip(-1)``; padded rows land on the spare row, live rows bitwise."""
        torch.manual_seed(3)
        for bs, n, slots in (
            (1, 4, 10),
            (8, 4, 300),
            (64, 6, 300),
            (16, 2, 40),
            (3, 9, 5),
        ):
            with self.subTest(bs=bs, n=n):
                pad_row = slots
                base = torch.randint(
                    0, 1000, (slots + 1, n - 1), device="cuda", dtype=torch.int32
                )
                tokens = torch.randint(
                    0, 1000, (bs, n), device="cuda", dtype=torch.int32
                )
                req = torch.randperm(slots, device="cuda")[:bs]
                out_loc = torch.randint(1, 5000, (bs,), device="cuda")
                out_loc[bs // 2 :] = 0  # graph padding rows
                ref = base.clone()
                rows = torch.where(out_loc == 0, torch.full_like(req, pad_row), req)
                ref[rows] = tokens[:, : n - 1].flip(-1).to(ref.dtype)
                got = base.clone()
                engram_commit_decode_history(got, tokens, req, out_loc, pad_row)
                self.assertTrue(torch.equal(got[:slots], ref[:slots]))
                # without out_cache_loc every row is live
                ref = base.clone()
                ref[req] = tokens[:, : n - 1].flip(-1)
                got = base.clone()
                engram_commit_decode_history(got, tokens, req, None, pad_row)
                self.assertTrue(torch.equal(got, ref))


if __name__ == "__main__":
    unittest.main()
