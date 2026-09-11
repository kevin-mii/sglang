"""Decoder SWA bounded replay on the HIP radix backend: the late-layer tail metadata
(each request's last SWA_WINDOW extend rows, windows floored at the tail start) and
the switch onto it, checked against the flag-off build of the same batch."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

SWA_WINDOW = 128
PAGE_SIZE = 256
NUM_REQ_SLOTS = 6
MAX_CONTEXT = 1024
NUM_FULL_SLOTS = NUM_REQ_SLOTS * MAX_CONTEXT + 1
TOPK_BLOCKS, BLOCK_SIZE = 16, 8

# (extend_len, cached_prefix_len, request slot): a request longer than the window,
# one shorter than the window behind a cached prefix (its floor still cuts the
# prefix off), one exactly the window, and a one-token extend.
REQUESTS = [(300, 0, 5), (37, 40, 2), (128, 0, 0), (1, 500, 4)]


def _make_backend(device, *, bounded_replay=True):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.device = device
    backend.cuda_int32_kwargs = {"device": device, "dtype": torch.int32}
    backend.swa_page_size = SWA_WINDOW
    backend.page_size = PAGE_SIZE
    g = torch.Generator().manual_seed(0)
    # Distinct full slots per (request slot, position); slot 0 stays the dummy.
    req_to_token = (
        torch.randperm(NUM_REQ_SLOTS * MAX_CONTEXT, generator=g).view(
            NUM_REQ_SLOTS, MAX_CONTEXT
        )
        + 1
    )
    backend.req_to_token = req_to_token.to(device=device, dtype=torch.int32)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
    backend.MAX_SEQ_LEN_FOR_CAPTURE = MAX_CONTEXT
    # full -> swa: injective and not the identity, so a wrong source shows.
    full_to_swa = (torch.arange(NUM_FULL_SLOTS) * 3 + 7).to(
        device=device, dtype=torch.int64
    )
    backend.token_to_kv_pool = SimpleNamespace(
        full_to_swa_index_mapping=full_to_swa,
        translate_loc_from_full_to_swa=lambda idx: full_to_swa[idx.to(torch.int64)],
        unified_swa_pages=0,
        get_index_k_page_size=lambda ratio: 64,
    )
    backend.index_topk = 512
    backend.present_ratios = (1,)
    backend.low_ratios = (1,)
    backend.has_c4 = False
    backend.has_c128 = False
    backend.candidate_masks = None
    backend.low_ratio_identity_skip = False
    backend.low_ratio_candidate_span = None
    backend.enable_deepseek_v4_fp4_indexer = False
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.topk = 0
    backend.mtp_enabled = False
    backend.speculative_num_steps = 0
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = None
    backend.is_draft_worker = False
    backend.is_dspark_draft = False
    backend.enable_decoder_swa_bounded_replay = bounded_replay
    backend.forward_metadata = None
    backend.tail_forward_metadata = None
    return backend


def _make_batch(backend, requests=REQUESTS):
    device = backend.device
    extend_lens = [n for n, _, _ in requests]
    seq_lens = [n + p for n, p, _ in requests]
    slots = [s for _, _, s in requests]
    positions, req_rows = [], []
    for (n, p, _), b in zip(requests, range(len(requests))):
        positions += list(range(p, p + n))
        req_rows += [b] * n
    positions = torch.tensor(positions, dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    req_rows = torch.tensor(req_rows, dtype=torch.int64, device=device)
    # The allocator wrote these slots into req_to_token before the forward.
    out_cache_loc = backend.req_to_token[req_pool_indices[req_rows], positions].to(
        torch.int64
    )
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(requests),
        extend_seq_lens_cpu=extend_lens,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        req_pool_indices=req_pool_indices,
        out_cache_loc=out_cache_loc,
        positions=positions,
        tbo_children=None,
        tbo_parent_token_range=None,
    )


def _expected_tail(requests, tail_len=SWA_WINDOW):
    """Per request (first extend row, tail length, absolute floor)."""
    rows, start = [], 0
    for n, p, _ in requests:
        t = min(tail_len, n)
        rows.append((start + n - t, t, p + n - t))
        start += n
    return rows


_NO_C4_PLAN = mock.patch(
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"
    ".create_paged_compressor_data",
    lambda *a, **k: None,
)


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestLateLayerTailMetadataHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.backend = _make_backend(self.device)
        self.batch = _make_batch(self.backend)
        with _NO_C4_PLAN:
            self.full = self.backend._prefill_metadata_for_batch(self.batch)
            self.tail_metadata = self.backend._build_late_layer_tail_metadata(
                self.batch
            )
        self.tail = self.tail_metadata.late_layer_tail

    def test_tail_is_each_request_suffix(self):
        tail, batch = self.tail, self.batch
        expected = _expected_tail(REQUESTS)
        self.assertEqual(tail.extend_seq_lens_cpu, [t for _, t, _ in expected])
        self.assertEqual(tail.extend_seq_lens.tolist(), tail.extend_seq_lens_cpu)
        want_rows = [r for s, t, _ in expected for r in range(s, s + t)]
        self.assertEqual(tail.token_indices.tolist(), want_rows)
        self.assertIsNone(tail.contiguous_start)
        self.assertTrue(torch.equal(tail.positions, batch.positions[want_rows]))
        full_to_swa = self.backend.token_to_kv_pool.full_to_swa_index_mapping
        self.assertTrue(
            torch.equal(
                tail.swa_out_cache_loc,
                full_to_swa[batch.out_cache_loc[want_rows]].to(torch.int32),
            )
        )
        core = self.tail_metadata.core_attn_metadata
        self.assertIs(core.swa_out_cache_loc, tail.swa_out_cache_loc)
        self.assertEqual(
            core.raw_out_loc.tolist(), batch.out_cache_loc[want_rows].tolist()
        )
        self.assertEqual(
            self.tail_metadata.low_ratio_req_indices.tolist(),
            [
                int(batch.req_pool_indices[b])
                for b, (_, t, _) in enumerate(expected)
                for _ in range(t)
            ],
        )
        self.assertTrue(
            torch.equal(self.tail_metadata.low_ratio_pos_i64, tail.positions)
        )
        self.assertIsNone(self.full.late_layer_tail)

    def test_single_request_tail_is_a_view(self):
        backend = _make_backend(self.device)
        batch = _make_batch(backend, [REQUESTS[0]])
        with _NO_C4_PLAN:
            tail = backend._build_late_layer_tail_metadata(batch).late_layer_tail
        n = REQUESTS[0][0]
        self.assertEqual(tail.contiguous_start, n - SWA_WINDOW)
        rows = tail.rows(batch.positions)
        self.assertEqual(rows.data_ptr(), batch.positions[n - SWA_WINDOW :].data_ptr())
        self.assertEqual(tail.extend_seq_lens_cpu, [SWA_WINDOW])

    def test_windows_are_floored_at_the_tail_start(self):
        core, full_core = (
            self.tail_metadata.core_attn_metadata,
            self.full.core_attn_metadata,
        )
        full_to_swa = self.backend.token_to_kv_pool.full_to_swa_index_mapping
        width = core.swa_page_indices.shape[1]
        self.assertEqual(width % 64, 0)
        self.assertEqual(width, full_core.swa_page_indices.shape[1])
        row = 0
        for (n, p, slot), (first, t, floor) in zip(REQUESTS, _expected_tail(REQUESTS)):
            for i in range(t):
                pos = floor + i
                visible = pos - floor + 1
                want = torch.full((width,), -1, dtype=torch.int32, device=self.device)
                want[:visible] = full_to_swa[
                    self.backend.req_to_token[slot, pos - visible + 1 : pos + 1]
                    .flip(0)
                    .to(torch.int64)
                ].to(torch.int32)
                self.assertTrue(
                    torch.equal(core.swa_page_indices[row], want), (slot, pos)
                )
                self.assertEqual(int(core.swa_topk_lengths[row]), visible, (slot, pos))
                self.assertEqual(int(core.positions_casual[row]), pos)
                self.assertEqual(int(core.seq_lens_casual[row]), pos + 1)
                full_row = first + i
                if pos + 1 == visible or i == t - 1 and n >= SWA_WINDOW:
                    # Nothing to floor: the row reads what the flag-off build's
                    # does (the flag-off build leaves its unused slots unmasked).
                    self.assertTrue(
                        torch.equal(
                            core.swa_page_indices[row, :visible],
                            full_core.swa_page_indices[full_row, :visible],
                        ),
                        (slot, pos),
                    )
                    self.assertEqual(
                        int(core.swa_topk_lengths[row]),
                        int(full_core.swa_topk_lengths[full_row]),
                    )
                else:
                    self.assertLess(
                        int(core.swa_topk_lengths[row]),
                        int(full_core.swa_topk_lengths[full_row]),
                    )
                row += 1
        self.assertEqual(row, core.swa_page_indices.shape[0])
        # Behind a cached prefix the first tail row sees only itself, not the prefix.
        _, t1, _ = _expected_tail(REQUESTS)[1]
        first_of_second = _expected_tail(REQUESTS)[0][1]
        self.assertEqual(int(core.swa_topk_lengths[first_of_second]), 1)
        self.assertEqual(int(self.full.core_attn_metadata.swa_topk_lengths[300]), 41)

    def test_low_ratio_metadata_follows_the_tail_rows(self):
        core, full_core = (
            self.tail_metadata.core_attn_metadata,
            self.full.core_attn_metadata,
        )
        rows = self.tail.token_indices
        self.assertEqual(core.low_ratios, (1,))
        for name in ("c1_out_loc", "c1_topk_lengths_clamp1", "c1_sparse_topk_lengths"):
            self.assertTrue(
                torch.equal(getattr(core, name), getattr(full_core, name)[rows]), name
            )
        self.assertEqual(
            core.sparse_page_indices(1).shape,
            (rows.numel(), full_core.sparse_page_indices(1).shape[1]),
        )
        c1 = self.tail_metadata.c1_indexer_metadata
        self.assertTrue(
            torch.equal(c1.page_table, self.full.c1_indexer_metadata.page_table[rows])
        )
        self.assertTrue(
            torch.equal(c1.c4_seq_lens, self.full.c1_indexer_metadata.c4_seq_lens[rows])
        )
        self.assertIn(1, self.tail_metadata.fp4_low_ratio_prefill_workspaces)
        self.assertIsNot(
            self.tail_metadata.fp4_low_ratio_prefill_workspaces[1],
            self.full.fp4_low_ratio_prefill_workspaces.get(1),
        )


@unittest.skipUnless(is_hip(), "HIP DeepSeek-V4 backend")
class TestLateLayerTailSwitchHip(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.backend = _make_backend(self.device)
        self.batch = _make_batch(self.backend)
        with _NO_C4_PLAN:
            self.full = self.backend._prefill_metadata_for_batch(self.batch)
            self.tail_metadata = self.backend._build_late_layer_tail_metadata(
                self.batch
            )
        self.backend.forward_metadata = self.full
        self.backend.tail_forward_metadata = self.tail_metadata

    def _candidates(self, rows):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            CandidateBlocks,
        )

        return CandidateBlocks(
            ids=torch.arange(rows * TOPK_BLOCKS, device=self.device)
            .view(rows, TOPK_BLOCKS)
            .to(torch.int32),
            compact_lens=torch.arange(rows, device=self.device).to(torch.int32),
            compact_page_table=torch.zeros(
                rows, 1, dtype=torch.int32, device=self.device
            ),
            compact_page_size=64,
            block_size=BLOCK_SIZE,
        )

    def test_enter_carries_topk_rows_cuts_masks_and_exit_restores(self):
        backend, tail = self.backend, self.tail_metadata.late_layer_tail
        full_core = self.full.core_attn_metadata
        num_tokens = full_core.seq_lens_casual.shape[0]
        # The index source before the switch wrote distinct top-k rows.
        full_core.c1_sparse_page_indices.copy_(
            torch.arange(full_core.c1_sparse_page_indices.numel(), device=self.device)
            .view_as(full_core.c1_sparse_page_indices)
            .to(torch.int32)
        )
        full_core.c1_sparse_raw_indices.copy_(full_core.c1_sparse_page_indices + 1)
        extend_lens = self.batch.extend_seq_lens_cpu
        masks = [
            self._candidates(extend_lens[0]),
            None,
            torch.rand(extend_lens[2], 50, device=self.device) > 0.5,
            self._candidates(extend_lens[3]),
        ]
        backend.candidate_masks = list(masks)

        saved = backend.enter_late_layer_tail(self.batch)

        self.assertIs(backend.forward_metadata, self.tail_metadata)
        tail_core = self.tail_metadata.core_attn_metadata
        rows = tail.token_indices
        for name in (
            "c1_sparse_page_indices",
            "c1_sparse_raw_indices",
            "c1_sparse_topk_lengths",
        ):
            self.assertTrue(
                torch.equal(getattr(tail_core, name), getattr(full_core, name)[rows]),
                name,
            )
        cut = backend.candidate_masks
        self.assertEqual(len(cut), len(masks))
        for b, (mask, t) in enumerate(zip(masks, tail.extend_seq_lens_cpu)):
            n = extend_lens[b]
            if mask is None:
                self.assertIsNone(cut[b])
            elif isinstance(mask, torch.Tensor):
                self.assertTrue(torch.equal(cut[b], mask[n - t :]))
            else:
                self.assertTrue(torch.equal(cut[b].ids, mask.ids[n - t :]))
                self.assertTrue(
                    torch.equal(cut[b].compact_lens, mask.compact_lens[n - t :])
                )
                self.assertEqual(cut[b].compact_page_table.shape[0], t)
                self.assertEqual(cut[b].block_size, mask.block_size)
        # The store target is the tail's while the batch still carries the full one.
        self.assertEqual(self.batch.out_cache_loc.shape[0], num_tokens)
        self.assertIs(backend.get_swa_out_cache_loc(self.batch), tail.swa_out_cache_loc)

        backend.exit_late_layer_tail(saved, self.batch)
        self.assertIs(backend.forward_metadata, self.full)
        self.assertEqual(len(backend.candidate_masks), len(masks))
        for got, want in zip(backend.candidate_masks, masks):
            self.assertIs(got, want)
        full_to_swa = backend.token_to_kv_pool.full_to_swa_index_mapping
        self.assertTrue(
            torch.equal(
                backend.get_swa_out_cache_loc(self.batch),
                full_to_swa[self.batch.out_cache_loc].to(torch.int32),
            )
        )

    def test_init_forward_metadata_builds_the_tail_only_under_the_flag(self):
        for flag in (True, False):
            backend = _make_backend(self.device, bounded_replay=flag)
            batch = _make_batch(backend)
            with _NO_C4_PLAN:
                backend.init_forward_metadata(batch)
            self.assertIsNone(backend.forward_metadata.late_layer_tail)
            if flag:
                tail = backend.tail_forward_metadata.late_layer_tail
                self.assertEqual(
                    tail.extend_seq_lens_cpu,
                    [t for _, t, _ in _expected_tail(REQUESTS)],
                )
            else:
                self.assertIsNone(backend.tail_forward_metadata)

    def test_prefill_indexer_scores_the_tail_rows(self):
        """An all-identity batch: the extend indexer fills one row per tail token,
        sliced by the tail's lengths, not the full extend's."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_index_topk_hip_extend,
        )

        backend, tail = self.backend, self.tail_metadata.late_layer_tail
        backend.low_ratio_identity_skip = True
        backend.enter_late_layer_tail(self.batch)
        indexer = SimpleNamespace(
            index_topk=512,
            is_candidate_source=False,
            uses_candidates=False,
            candidate_topk_blocks=TOPK_BLOCKS,
            candidate_block_size=BLOCK_SIZE,
        )
        layer = SimpleNamespace(layer_id=21, compress_ratio=1, indexer=indexer)
        pos = tail.positions.to(torch.int64)
        low_ratio_index_topk_hip_extend(backend, layer, None, None, pos, self.batch)
        page_indices = self.tail_metadata.core_attn_metadata.sparse_page_indices(1)
        self.assertEqual(page_indices.shape[0], pos.numel())
        # Row of position pos in request slot s selects compressed positions 0..pos.
        row = 0
        for (n, p, slot), (_, t, floor) in zip(REQUESTS, _expected_tail(REQUESTS)):
            for i in range(t):
                position = floor + i
                want = backend.req_to_token[slot, : position + 1].to(torch.int32)
                self.assertTrue(
                    torch.equal(page_indices[row, : position + 1], want),
                    (slot, position),
                )
                self.assertTrue(bool((page_indices[row, position + 1 :] == -1).all()))
                row += 1


if __name__ == "__main__":
    unittest.main()
