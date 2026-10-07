"""Causal verify metadata and routing with no ordinary prefill metadata;
small extends served as flattened decode rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm verify integration")
class TestMiniMaxROCmVerify(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        self.backend.is_hip = True
        self.backend.speculative_num_draft_tokens = 4
        self.backend._linear_verify_meta = None
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([127, 2047], dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.tensor([3, 1], dtype=torch.int64, device="cuda"),
            out_cache_loc=torch.arange(8, dtype=torch.int64, device="cuda"),
            extend_seq_lens=None,
            extend_prefix_lens=None,
        )
        self.q = torch.zeros(8, 1, 128, dtype=torch.bfloat16, device="cuda")
        self.seen = []

        def decode(q, k, v, layer, batch, save_kv_cache, **kwargs):
            self.seen.append(batch)
            # Expose the metadata received by the unchanged sparse decoder.
            return None, torch.stack((batch.req_pool_indices, batch.seq_lens), -1)

        self.backend.forward_decode = decode

    def run_verify(self):
        self.backend._init_rocm_linear_verify_metadata(self.batch)
        return self.backend.forward_extend(
            self.q,
            self.q,
            self.q,
            None,
            self.batch,
            idx_q=self.q,
            idx_k=self.q,
            idx_v=None,
        )[1]

    def assert_causal_rows(self, output, expected):
        torch.testing.assert_close(
            output,
            torch.tensor(expected, dtype=torch.int64, device="cuda"),
            rtol=0,
            atol=0,
        )
        self.assertIsNot(self.seen[-1], self.batch)
        self.assertIs(self.seen[-1].out_cache_loc, self.batch.out_cache_loc)
        self.assertIsNone(self.batch.extend_seq_lens)
        self.assertEqual(self.batch.seq_lens.numel(), 2)

    def test_verify_keeps_original_batch_and_cache_locations(self):
        output = self.run_verify()
        self.assert_causal_rows(
            output,
            [
                [3, 128],
                [3, 129],
                [3, 130],
                [3, 131],
                [1, 2048],
                [1, 2049],
                [1, 2050],
                [1, 2051],
            ],
        )
        self.assertEqual(self.batch.seq_lens.tolist(), [127, 2047])

    def test_graph_replay_reads_new_prefixes_and_requests(self):
        self.run_verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.run_verify()
        self.batch.seq_lens.copy_(
            torch.tensor([4095, 255], dtype=torch.int32, device="cuda")
        )
        self.batch.req_pool_indices.copy_(
            torch.tensor([2, 0], dtype=torch.int64, device="cuda")
        )
        graph.replay()
        self.assert_causal_rows(
            output,
            [
                [2, 4096],
                [2, 4097],
                [2, 4098],
                [2, 4099],
                [0, 256],
                [0, 257],
                [0, 258],
                [0, 259],
            ],
        )
        self.assertEqual(self.batch.seq_lens.tolist(), [4095, 255])

    def test_missing_metadata_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing MiniMax-M3 linear verify"):
            self.backend.forward_extend(
                self.q,
                self.q,
                self.q,
                None,
                self.batch,
                idx_q=self.q,
                idx_k=self.q,
                idx_v=None,
            )


@unittest.skipUnless(torch.version.hip, "ROCm verify integration")
class TestMiniMaxSmallExtendRows(unittest.TestCase):
    def test_decode_rows_match_sparse_prefill(self):
        """Each flattened row must attend exactly its causal prefix, as the prefill path does."""
        from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
            get_cu_seqblocks,
        )
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
            minimax_sparse_decode,
            minimax_sparse_prefill,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
        backend.is_npu = False
        backend._use_msa_decode = False
        backend.use_dense_sparse_decode = False
        backend.hisparse_coordinator = None
        backend._small_extend_rows = None
        # The first request's new tokens cross into a new 128-token block.
        prefixes, extends, slots = [128 * 20 - 3, 5000], [8, 3], [2, 1]
        seq_lens = [p + e for p, e in zip(prefixes, extends)]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=extends,
            extend_prefix_lens_cpu=prefixes,
            seq_lens=torch.tensor(seq_lens, device="cuda"),
            req_pool_indices=torch.tensor(slots, device="cuda"),
        )
        self.assertTrue(backend._is_small_extend(batch))
        rows = backend._small_extend_row_batch(batch)
        self.assertIs(backend._small_extend_row_batch(batch), rows)
        self.assertEqual(batch.seq_lens.tolist(), seq_lens)

        gen = torch.Generator(device="cuda").manual_seed(0)
        max_len, num_rows = max(seq_lens), sum(extends)
        num_slots = len(slots) * max_len + 1
        req_to_token = torch.zeros(3, max_len, dtype=torch.int32, device="cuda")
        perm = torch.randperm(num_slots - 1, device="cuda", generator=gen) + 1
        req_to_token[slots] = perm.to(torch.int32).view(len(slots), max_len)

        def randn(*shape):
            return torch.randn(*shape, device="cuda", generator=gen).bfloat16()

        q, idx_q = randn(num_rows, 16, 128), randn(num_rows, 1, 128)
        k_cache, v_cache, idx_k_cache = (randn(num_slots, 1, 128) for _ in "kvi")
        cu_seqlens = torch.tensor([0, 8, 11], dtype=torch.int32, device="cuda")
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max(extends), 1, 128, extends
        )
        ref = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            None,
            None,
            req_to_token,
            batch.req_pool_indices,
            cu_seqlens,
            batch.seq_lens.int(),
            torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
            max(extends),
            max_len,
            1,
            128,
            16,
            0,
            1,
            disable_index_value=True,
            seqlens_cpu=extends,
            seq_lens_cpu=batch.seq_lens.cpu(),
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
        )[1]
        out = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            None,
            req_to_token,
            rows.req_pool_indices,
            rows.seq_lens,
            max_len,
            1,
            128,
            16,
            0,
            1,
            disable_index_value=True,
        )[1]
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
