"""Breakable prefill CUDA graphs on ROCm for the DeepSeek-V4.1 radix backend: the
captured metadata object stays active across a replay, the SWA store target keeps
its storage and takes the padded static batch's values, and every break-time field
is the eager build for the live batch.
"""

import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

INT32 = dict(dtype=torch.int32)


def _core_metadata(base: int, low_ratios=(1, 2), num_tokens: int = 2):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DSV4AttnMetadata,
    )

    rows = torch.arange(num_tokens, **INT32)
    md = DSV4AttnMetadata(
        page_size=256,
        page_table=(base + 1 + rows)[:, None].repeat(1, 2),
        raw_out_loc=base + 5 + rows,
        cuda_int32_kwargs={"dtype": torch.int32},
        seq_lens_casual=base + 7 + rows,
        positions_casual=base + 9 + rows,
        swa_page_indices=(base + 11 + rows)[:, None].repeat(1, 2),
        swa_topk_lengths=base + 15 + rows,
        index_topk=512,
        low_ratios=low_ratios,
    )
    md.swa_out_cache_loc = base + 17 + rows
    for name in (
        "c4_out_loc",
        "c128_out_loc",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c4_sparse_topk_lengths",
        "c4_sparse_topk_lengths_raw",
        "c128_topk_lengths_clamp1",
        "c128_topk_lengths_raw",
    ):
        setattr(md, name, base + 20 + rows)
    for name in (
        "c4_sparse_page_indices",
        "c4_sparse_raw_indices",
        "c128_page_indices",
    ):
        setattr(md, name, (base + 30 + rows)[:, None].repeat(1, 2))
    for ratio in low_ratios:
        setattr(md, f"c{ratio}_out_loc", base + 40 + ratio + rows)
        setattr(md, f"c{ratio}_topk_lengths_clamp1", base + 50 + rows)
        setattr(md, f"c{ratio}_sparse_topk_lengths", base + 60 + rows)
        setattr(
            md,
            f"c{ratio}_sparse_page_indices",
            (base + 70 + rows)[:, None].repeat(1, 2),
        )
        setattr(
            md, f"c{ratio}_sparse_raw_indices", (base + 80 + rows)[:, None].repeat(1, 2)
        )
        setattr(md, f"c{ratio}_flashmla_metadata", object())
    for name in (
        "c0_flashmla_metadata",
        "c4_flashmla_metadata",
        "c128_flashmla_metadata",
    ):
        setattr(md, name, object())
    return md


@unittest.skipUnless(is_hip(), "the HIP radix backend is ROCm only")
class TestHipBreakableGraphMetadataContract(CustomTestCase):
    def test_refresh_pins_the_store_target_and_rebinds_the_rest(self):
        capture, live = _core_metadata(0), _core_metadata(1000)
        pinned = capture.swa_out_cache_loc
        capture._aiter_sparse_masked_indices = {"stale": None}
        live._aiter_sparse_masked_indices = None

        capture.refresh_for_breakable_cuda_graph_replay_(live)

        # the store target is read inside the captured segments: same storage, live contents
        self.assertIs(capture.swa_out_cache_loc, pinned)
        self.assertEqual(pinned.tolist(), [1017, 1018])
        # Everything else is read at the eager breaks and follows the live build.
        for f in dataclasses.fields(capture):
            if f.name == "swa_out_cache_loc":
                continue
            with self.subTest(field=f.name):
                self.assertIs(getattr(capture, f.name), getattr(live, f.name))
        # The aiter_sparse length-fold cache is per forward; the live build brings None.
        self.assertIsNone(capture._aiter_sparse_masked_indices)

    def test_refresh_rejects_a_store_target_of_another_bucket(self):
        capture, live = (
            _core_metadata(0, num_tokens=2),
            _core_metadata(1000, num_tokens=3),
        )
        with self.assertRaises(AssertionError):
            capture.refresh_for_breakable_cuda_graph_replay_(live)

    def test_top_level_refresh_keeps_graph_pool_scratch(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4Metadata,
        )

        capture = DSV4Metadata(_core_metadata(0), indexer_metadata=None)
        live = DSV4Metadata(_core_metadata(1000), indexer_metadata=None)
        scratch = torch.zeros(2, 64, 512)
        capture.q_pad_buffer = scratch
        live.c1_indexer_metadata = object()
        live.fp4_low_ratio_prefill_workspaces = {1: object()}
        pinned = capture.core_attn_metadata.swa_out_cache_loc

        capture.refresh_for_breakable_cuda_graph_replay_(live)

        self.assertIs(capture.core_attn_metadata.swa_out_cache_loc, pinned)
        self.assertIs(capture.q_pad_buffer, scratch)
        self.assertIs(capture.c1_indexer_metadata, live.c1_indexer_metadata)
        self.assertIs(
            capture.fp4_low_ratio_prefill_workspaces,
            live.fp4_low_ratio_prefill_workspaces,
        )

    def _backend(self, **attrs):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )

        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.has_c4 = backend.has_c128 = False
        backend.forward_metadata = None
        for name, value in attrs.items():
            setattr(backend, name, value)
        return backend

    def test_replay_entry_keeps_the_captured_metadata_active(self):
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4Metadata,
        )

        capture = DSV4Metadata(_core_metadata(0), indexer_metadata=None)
        live = DSV4Metadata(_core_metadata(1000), indexer_metadata=None)
        live.core_attn_metadata.swa_out_cache_loc = None
        backend = self._backend()
        calls = []
        forward_batch = SimpleNamespace(name="live")
        static_forward_batch = SimpleNamespace(name="static")

        def build(batch):
            calls.append(("build", batch))
            return live

        def in_graph(batch):
            calls.append(("in_graph", batch))
            # The store target comes from the padded static batch.
            self.assertIs(backend.forward_metadata, live)
            live.core_attn_metadata.swa_out_cache_loc = torch.tensor([7, 8], **INT32)

        backend._prefill_metadata_for_batch = build
        backend.init_forward_metadata_in_graph = in_graph
        backend._refresh_fp4_prefill_workspace = lambda batch: calls.append(
            ("workspace", batch)
        )
        pinned = capture.core_attn_metadata.swa_out_cache_loc

        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture, forward_batch, static_forward_batch=static_forward_batch
        )

        self.assertEqual(
            calls,
            [
                ("build", forward_batch),
                ("in_graph", static_forward_batch),
                ("workspace", forward_batch),
            ],
        )
        self.assertIs(backend.forward_metadata, capture)
        self.assertIs(capture.core_attn_metadata.swa_out_cache_loc, pinned)
        self.assertEqual(pinned.tolist(), [7, 8])
        self.assertIs(
            capture.core_attn_metadata.seq_lens_casual,
            live.core_attn_metadata.seq_lens_casual,
        )

    def test_capture_entry_rejects_layouts_it_does_not_pin(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with mock.patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=False,
        ):
            backend = self._backend(has_c4=True)
            with self.assertRaisesRegex(NotImplementedError, "c4 / c128"):
                backend.init_forward_metadata_for_breakable_cuda_graph_capture(batch)
        with mock.patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=True,
        ):
            backend = self._backend()
            with self.assertRaisesRegex(NotImplementedError, "unified_kv"):
                backend.init_forward_metadata_for_breakable_cuda_graph_capture(batch)


@unittest.skipUnless(
    is_hip() and torch.cuda.is_available(), "the fp8-grid wrapper is a gfx950 path"
)
class TestBreakInputsKeepTheirWrapper(CustomTestCase):
    """The row slice and the BCG weak-ref pass must hand `Fp8GridActivation` on, or the
    fp8 linear reads a bare tuple."""

    def test_live_rows_slice_inside_the_wrapper(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Fp8GridActivation
        from sglang.srt.models.deepseek_common.amd import deepseek_v4_gfx95_dense

        rows = torch.arange(6.0, device="cuda").view(3, 2)
        with (
            mock.patch.object(deepseek_v4_gfx95_dense, "_is_hip", True),
            mock.patch.object(deepseek_v4_gfx95_dense, "_is_gfx95_supported", True),
            mock.patch.object(
                deepseek_v4_gfx95_dense, "Fp8GridActivation", Fp8GridActivation
            ),
        ):
            sliced = deepseek_v4_gfx95_dense.live_rows(Fp8GridActivation(rows), 2)
            self.assertIsInstance(sliced, Fp8GridActivation)
            self.assertTrue(torch.equal(sliced.x, rows[:2]))
            self.assertTrue(
                torch.equal(deepseek_v4_gfx95_dense.live_rows(rows, 2), rows[:2])
            )

    def test_weak_ref_pass_keeps_namedtuple_types(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Fp8GridActivation
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            _weak_ref_if_tensor,
        )

        x = torch.ones(4, device="cuda")
        weak = _weak_ref_if_tensor(Fp8GridActivation(x))
        self.assertIsInstance(weak, Fp8GridActivation)
        self.assertEqual(weak.x.data_ptr(), x.data_ptr())
        plain = _weak_ref_if_tensor((x, 3))
        self.assertIs(type(plain), tuple)
        self.assertEqual(plain[1], 3)


class TestPrefillRunnerUsesCapturedMetadataContract(CustomTestCase):
    """The runner side of the contract the HIP backend opts into."""

    def _runner(self, attn_backend):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.use_captured_attn_metadata = True
        runner.attn_metadata_buffers = {}
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        return runner

    def test_capture_stashes_and_replay_refreshes_per_bucket(self):
        attn_backend = mock.Mock()
        stashed = object()
        attn_backend.init_forward_metadata_for_breakable_cuda_graph_capture.return_value = stashed
        runner = self._runner(attn_backend)
        capture_batch = SimpleNamespace(name="capture")
        live_batch = SimpleNamespace(name="live")
        static_batch = SimpleNamespace(name="static")

        runner._init_forward_metadata_for_capture(capture_batch, 96)
        runner._prepare_forward_metadata_for_replay(live_batch, static_batch, 96)

        self.assertIs(runner.attn_metadata_buffers[96], stashed)
        attn_backend.init_forward_metadata.assert_not_called()
        attn_backend.prepare_forward_metadata_for_breakable_cuda_graph_replay.assert_called_once_with(
            stashed, live_batch, static_forward_batch=static_batch
        )


def _extend_batch(
    *,
    seq_lens,
    extend_lens,
    req_pool_indices,
    out_cache_loc,
    device,
) -> ForwardBatch:
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    extend_t = torch.tensor(extend_lens, dtype=torch.int32, device=device)
    positions = torch.cat(
        [
            torch.arange(s - e, s, dtype=torch.int64, device=device)
            for s, e in zip(seq_lens, extend_lens)
        ]
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=torch.zeros(sum(extend_lens), dtype=torch.int64, device=device),
        req_pool_indices=torch.tensor(
            req_pool_indices, dtype=torch.int32, device=device
        ),
        seq_lens=seq_lens_t,
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=sum(seq_lens),
        positions=positions,
    )
    batch.extend_prefix_lens = seq_lens_t - extend_t
    batch.extend_prefix_lens_cpu = [s - e for s, e in zip(seq_lens, extend_lens)]
    batch.extend_seq_lens = extend_t
    batch.extend_seq_lens_cpu = list(extend_lens)
    batch.extend_start_loc = torch.cumsum(extend_t, dim=0) - extend_t
    batch.extend_num_tokens = sum(extend_lens)
    batch.num_token_non_padded_cpu = sum(extend_lens)
    return batch


def _tensor_fields(obj):
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name, None)
        if isinstance(value, torch.Tensor):
            yield f.name, value


@unittest.skipUnless(
    is_hip() and torch.cuda.is_available(), "the HIP radix backend is ROCm only"
)
class TestHipBreakableGraphCaptureReplay(CustomTestCase):
    """A captured store must read the refreshed target, and break-time metadata and
    attention must equal the eager build for the same batch."""

    BUCKET = 96

    def test_capture_replay_matches_eager(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
            DSV4_PAGE_SIZE,
            DSV4AttentionCase,
            _populate_swa_kv_cache,
            build_dsv4_attention_fixture,
        )

        device = "cuda"
        case = DSV4AttentionCase(
            name="hip_bcg_extend",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(0, 0),
            extend_lens=(40, 24),
        )
        fixture = build_dsv4_attention_fixture(
            self,
            case,
            device=device,
            max_context_len=2 * DSV4_PAGE_SIZE,
            compression_ratios=[0, 2, 1],
        )
        backend = fixture.backend
        pool = fixture.runner.token_to_kv_pool
        self.assertEqual(backend.low_ratios, (1, 2))
        _populate_swa_kv_cache(
            fixture, max_context_len=2 * DSV4_PAGE_SIZE, device=device
        )
        q_input, _ = fixture.actual_module.project(fixture.input_hidden)
        live = fixture.forward_batch
        live.num_token_non_padded_cpu = case.num_input_tokens
        num_tokens = case.num_input_tokens

        def attention(forward_batch):
            return backend.forward(
                q=q_input,
                k=q_input,
                v=q_input,
                layer=fixture.actual_module.attn,
                forward_batch=forward_batch,
                compress_ratio=0,
                save_kv_cache=False,
                attn_sink=fixture.actual_module.attn_sink,
            )

        # capture batch of a bucket: one request of BUCKET tokens on the static (all-zero) out_cache_loc slot
        capture_batch = _extend_batch(
            seq_lens=[self.BUCKET],
            extend_lens=[self.BUCKET],
            req_pool_indices=[0],
            out_cache_loc=torch.zeros(self.BUCKET, dtype=torch.int64, device=device),
            device=device,
        )
        # static view of the live batch: out_cache_loc padded to the bucket with the dummy slot
        static = copy.copy(live)
        static.out_cache_loc = torch.nn.functional.pad(
            live.out_cache_loc, (0, self.BUCKET - num_tokens), value=0
        )

        with torch.no_grad(), forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata(live)
            eager = backend.forward_metadata
            eager_out = attention(live)

            captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                capture_batch
            )
            self.assertIs(backend.forward_metadata, captured)
            pinned = captured.core_attn_metadata.swa_out_cache_loc
            self.assertEqual(tuple(pinned.shape), (self.BUCKET,))
            # the capture batch is one request of BUCKET tokens, so its page table is bucket-wide
            self.assertEqual(
                captured.core_attn_metadata.page_table.shape[1],
                (self.BUCKET + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE,
            )

            # A segment's store reads the target by address.
            sink = torch.empty_like(pinned)
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                sink.copy_(backend.get_swa_out_cache_loc(capture_batch))
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(graph):
                sink.copy_(backend.get_swa_out_cache_loc(capture_batch))

            backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                captured, live, static_forward_batch=static
            )
            self.assertIs(backend.forward_metadata, captured)
            self.assertIs(captured.core_attn_metadata.swa_out_cache_loc, pinned)
            graph.replay()
            torch.cuda.synchronize()
            expected_target = pool.translate_loc_from_full_to_swa(
                static.out_cache_loc
            ).to(torch.int32)
            self.assertTrue(torch.equal(sink, expected_target))
            self.assertTrue(
                torch.equal(
                    pinned[:num_tokens], eager.core_attn_metadata.swa_out_cache_loc
                )
            )

            # Every break-time field is the eager build for the live batch.
            for name, value in _tensor_fields(eager.core_attn_metadata):
                if name == "swa_out_cache_loc":
                    continue
                with self.subTest(field=name):
                    self.assertTrue(
                        torch.equal(getattr(captured.core_attn_metadata, name), value)
                    )
            for ratio in (1, 2):
                for name in ("page_table", "c4_seq_lens"):
                    with self.subTest(ratio=ratio, field=name):
                        self.assertTrue(
                            torch.equal(
                                getattr(
                                    captured.low_ratio_indexer_metadata(ratio), name
                                ),
                                getattr(eager.low_ratio_indexer_metadata(ratio), name),
                            )
                        )
            self.assertEqual(set(captured.fp4_low_ratio_prefill_workspaces), {1, 2})

            # The attention break on the refreshed metadata is the eager attention.
            replay_out = attention(live)
            self.assertTrue(torch.equal(replay_out, eager_out))

            # A second replay with a different batch shape rebinds again.
            live2 = _extend_batch(
                seq_lens=[24, 40],
                extend_lens=[24, 40],
                req_pool_indices=[1, 0],
                out_cache_loc=torch.flip(live.out_cache_loc, dims=[0]),
                device=device,
            )
            static2 = copy.copy(live2)
            static2.out_cache_loc = torch.nn.functional.pad(
                live2.out_cache_loc, (0, self.BUCKET - num_tokens), value=0
            )
            backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                captured, live2, static_forward_batch=static2
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(
                    sink,
                    pool.translate_loc_from_full_to_swa(static2.out_cache_loc).to(
                        torch.int32
                    ),
                )
            )
            backend.init_forward_metadata(live2)
            eager2 = backend.forward_metadata
            self.assertTrue(
                torch.equal(
                    captured.core_attn_metadata.seq_lens_casual,
                    eager2.core_attn_metadata.seq_lens_casual,
                )
            )


if __name__ == "__main__":
    unittest.main()
