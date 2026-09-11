"""In-place history updates must match the concatenation oracle across replays."""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_hash import (
    engram_commit_decode_history,
    engram_commit_history,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


class TestEngramHistoryCommit(CustomTestCase):
    def test_graph_replay_matches_oracle(self):
        torch.manual_seed(1)
        for width in (1, 2, 3, 7, 33, 65):
            with self.subTest(history_width=width):
                history = torch.randint(
                    0, 1000, (12, width + 3), device="cuda", dtype=torch.int32
                )[:, :width]
                tokens = torch.randint(0, 1000, (7, 9), device="cuda")[:, :6]
                slots = torch.tensor([8, 1, 5, 2, 10, 0, 7], device="cuda")
                commit = torch.arange(7, device="cuda", dtype=torch.int32)

                def update():
                    engram_commit_history(history, tokens, slots, commit)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    update()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    update()
                for _ in range(3):
                    slots.copy_(slots.roll(1))
                    commit.copy_(commit.roll(2))
                    tokens.add_(1)
                    expected = history.clone()
                    window = torch.cat([expected[slots], tokens.int()], dim=1)
                    cols = commit.long()[:, None] + torch.arange(width, device="cuda")
                    expected[slots] = window.gather(1, cols)
                    graph.replay()
                    torch.testing.assert_close(history, expected, rtol=0, atol=0)

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

    def test_decode_commit_graph_replay(self):
        torch.manual_seed(4)
        history = torch.zeros(9, 3, device="cuda", dtype=torch.int32)
        tokens = torch.randint(0, 1000, (4, 4), device="cuda", dtype=torch.int32)
        slots = torch.tensor([2, 5, 7, 1], device="cuda")
        out_loc = torch.tensor([3, 4, 0, 0], device="cuda")

        def update():
            engram_commit_decode_history(history, tokens, slots, out_loc, 8)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            update()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update()
        for _ in range(3):
            tokens.copy_(
                torch.randint(0, 1000, (4, 4), device="cuda", dtype=torch.int32)
            )
            expected = history.clone()
            expected[slots[:2]] = tokens[:2, :3].flip(-1)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(history[:8], expected[:8]))


if __name__ == "__main__":
    unittest.main()
