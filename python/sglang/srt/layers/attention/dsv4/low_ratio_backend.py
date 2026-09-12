"""DeepSeek V4.1 low compress ratio (1 / 2) sources shared by the DSV4 attention backends:
compression, index-K publishing, the torch reference indexer and their metadata helpers.
Each backend supplies `_low_ratio_index_topk` over its own indexer kernels."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.candidate_torch import (
    CandidateMasks,
    mask_topk_scores,
    published_masks,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    _rope_fq4,
    token_req_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks
from sglang.srt.layers.cp.interleave import interleave_rows_per_request
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

PAGE_INDEX_ALIGNED_SIZE = 64


T = TypeVar("T", bound=Optional[torch.Tensor])


@functools.lru_cache(maxsize=None)
def _is_sm100_or_newer() -> bool:
    """The DeepGEMM fp8_fp4 mqa-logits kernels need SM100/SM120; Hopper takes the torch indexer."""
    return torch.cuda.get_device_capability()[0] >= 10


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    if target_size == curr_size:
        # F.pad would clone even at the aligned size
        return x
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _expand_index_page_table(
    page_table: torch.Tensor,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
) -> torch.Tensor:
    """Expand the FULL page table into the block table of a low-ratio indexer-K
    pool, which pages at `index_page_size` slots rather than a FULL page.

    The kernel resolves compressed slot j through
    page_table[b, j // index_page_size] * index_page_size + j % index_page_size,
    which with this expansion is the c1/c2 KV pool slot of the same position.
    [bs, n_full_pages] -> [bs, n_full_pages * blocks_per_page], int32.
    """
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, (
        f"{full_page_size = } / {compress_ratio = } must be a multiple of "
        f"{index_page_size = }"
    )
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    bs, n = page_table.shape
    base = page_table.to(torch.int64) * blocks_per_page
    offsets = torch.arange(blocks_per_page, device=page_table.device, dtype=torch.int64)
    expanded = base.unsqueeze(-1) + offsets  # [bs, n, blocks_per_page]
    return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30


@functools.cache
def _has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


def _low_ratio_source_projections(layer, x, q_lora, positions, bufs):
    """Projections of a ratio-1/2 source layer, run as an eager break on the
    live rows into static buffers: the compressor's kv / score and the indexer's
    query and head weights. These GEMMs pick their algorithm by M, so at the
    bucket size their rows differ from eager; everything downstream is
    row-independent. Padded rows are zeroed."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    real = get_tc_piecewise_forward_context().forward_batch.num_token_non_padded_cpu
    if real is None:
        real = x.shape[0]

    def put(name, value):
        buf = bufs[name]
        buf[:real].copy_(value)
        buf[real:].zero_()

    if real == 0:
        # An idle DP-attention rank replays on fabricated rows with no live
        # token; a zero-row GEMM is a launch error, so only zero the buffers.
        for buf in bufs.values():
            buf.zero_()
        return

    if layer.compressor is not None:
        kv, score = layer.compressor.project(x[:real])
        put("kv", kv)
        if score is not None:
            put("score", score)
    if layer.indexer is not None:
        indexer = layer.indexer
        put("q", indexer.queries(q_lora[:real], layer.freqs_cis[positions[:real]]))
        put("w", indexer.head_weights(x[:real]))


@functools.cache
def _bcg_low_ratio_source_projections_fn():
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        eager_on_graph,
    )

    return eager_on_graph(True)(_low_ratio_source_projections)


def _bcg_low_ratio_source_projections(*args):
    return _bcg_low_ratio_source_projections_fn()(*args)


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        values = values.tolist()
    return [int(v) for v in values]


def _low_ratio_compression_metadata(
    compress_ratio: int, seq_lens_casual: torch.Tensor, raw_out_loc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ratio 1/2 counterpart of the triton c4/c128 metadata; a completed group's
    latent lives at raw_out_loc // ratio, -1 for a token that completes none."""
    num_write_tokens = raw_out_loc.shape[0]
    completes_group = seq_lens_casual[:num_write_tokens] % compress_ratio == 0
    out_loc = torch.where(
        completes_group, raw_out_loc.to(torch.int64) // compress_ratio, -1
    )
    topk_lengths_clamp1 = (seq_lens_casual // compress_ratio).clamp_min(1)
    return out_loc, topk_lengths_clamp1.to(torch.int32)


def _low_ratio_sparse_buffers(
    topk_lengths_clamp1: torch.Tensor, topk: int, is_prefill: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fixed-shape top-k buffers the index_source layers fill in place. The
    extra_topk_length the kernel reads comes from positions, not from the indexer."""
    sparse_topk_lengths = torch.clamp(topk_lengths_clamp1, max=topk)
    page_indices = _pad_last_dim(
        torch.full(
            (topk_lengths_clamp1.size(0), topk),
            -1,
            dtype=torch.int32,
            device=topk_lengths_clamp1.device,
        )
    )
    raw_indices = torch.empty_like(page_indices) if is_prefill else None
    return sparse_topk_lengths, page_indices, raw_indices


class LowRatioBackendMixin:
    @property
    def low_ratio_prefill_graph(self) -> bool:
        """The ratio-1/2 sources can run inside the prefill CUDA graph (the
        DeepGEMM paged indexer, so Blackwell CUDA only)."""
        return (
            bool(self.low_ratios) and _has_dense_fp4_indexer() and _is_sm100_or_newer()
        )

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch: ForwardBatch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Runs on every ratio 1/2 layer before its attention; the metadata and
        latents it writes are what the layers after it attend through."""
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.encoder_swa_replay:
            run_compressor = False
        if dsa_use_prefill_cp(forward_batch) and forward_batch.forward_mode.is_extend():
            self._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                run_compressor=run_compressor,
                run_indexer=run_indexer,
            )
            return
        meta = self.forward_metadata
        # only the HIP metadata hoists the per-token inputs
        hoisted_req = getattr(meta, "low_ratio_req_indices", None)
        hoisted_pos = getattr(meta, "low_ratio_pos_i64", None)
        if (
            hoisted_req is not None
            and hoisted_pos is not None
            and hoisted_pos.shape[0] == positions.shape[0]
        ):
            # Bucket-sized under the prefill graph; an eager break sees the
            # live rows only and falls through.
            req, pos = hoisted_req, hoisted_pos
        else:
            req = token_req_indices(forward_batch, num_tokens=positions.shape[0])
            pos = positions.to(torch.int64)
        if (
            forward_batch.forward_mode.is_extend()
            and self._low_ratio_in_prefill_graph()
        ):
            bufs = self._source_projection_buffers(x.shape[0], layer.compress_ratio)
            _bcg_low_ratio_source_projections(layer, x, q_lora, pos, bufs)
            if run_compressor and layer.compressor is not None:
                self._low_ratio_compress_torch(
                    layer, x, req, pos, projected=(bufs["kv"], bufs.get("score"))
                )
            if run_indexer and layer.indexer is not None:
                self._low_ratio_index_topk_prefill_graph(
                    layer, pos, bufs["q"], bufs["w"]
                )
            return
        if run_compressor and layer.compressor is not None:
            self._low_ratio_compress(layer, x, req, pos, forward_batch)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk(layer, x, q_lora, req, pos, forward_batch)

    def _forward_low_ratio_sources_cp(
        self, *, layer, x, q_lora, positions, forward_batch, run_compressor, run_indexer
    ) -> None:
        """Every rank writes the whole prompt's compressed state and scores its own rows."""
        cp_meta = forward_batch.attn_cp_metadata
        total = int(cp_meta.total_seq_lens)
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens_cpu = tail.local_lens_cpu
            req_global, pos_global = tail.req_global, tail.pos_global
        else:
            q_lens_cpu = interleave_rows_per_request(
                _as_int_list(forward_batch.extend_seq_lens_cpu),
                get_parallel().attn_cp_rank,
                get_parallel().attn_cp_size,
            )
            req_global = token_req_indices(forward_batch, num_tokens=total)
            pos_global = forward_batch.positions[:total].to(torch.int64)
        num_local = sum(q_lens_cpu)
        if run_compressor and layer.compressor is not None:
            x_global = cp_materialize_global_token_order(
                x.contiguous(), forward_batch, torch.cuda.current_stream()
            )[:total]
            self._low_ratio_compress_torch(layer, x_global, req_global, pos_global)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk_dense(
                layer,
                x[:num_local],
                q_lora[:num_local],
                positions[:num_local].to(torch.int64),
                forward_batch,
                torch.tensor(q_lens_cpu, dtype=torch.int32, device=x.device),
                q_lens_cpu,
            )

    def _low_ratio_compress(self, layer, x, req, pos, forward_batch) -> None:
        if forward_batch.forward_mode.is_decode():
            self._low_ratio_compress_decode(layer, x, req, pos)
        elif (
            forward_batch.forward_mode.is_target_verify()
            and not self.is_dspark_draft
            and layer.compress_ratio == 2
            and layer.compressor.use_fused_compress
            and read_ragged_verify_mode() is not RaggedVerifyMode.COMPACT
            and self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 1
            and x.shape[0]
            == forward_batch.batch_size * self.speculative_num_draft_tokens
        ):
            # Static verify is request-major with consecutive positions. Compact
            # verify has variable block lengths and must retain the general path.
            self._low_ratio_compress_fused(
                layer, x, req, pos, draft_len=self.speculative_num_draft_tokens
            )
        else:
            self._low_ratio_compress_torch(
                layer,
                x,
                req,
                pos,
                fuse_index_store=(
                    forward_batch.forward_mode.is_target_verify()
                    and layer.compressor.use_fused_compress
                ),
            )

    def _low_ratio_in_prefill_graph(self) -> bool:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        return self.low_ratio_prefill_graph and is_in_breakable_cuda_graph()

    def _low_ratio_compress_decode(self, layer, x, req, pos) -> None:
        # decided at load time with the projection layout, see `fused_low_ratio_compress_supported`
        if layer.compressor.use_fused_compress:
            self._low_ratio_compress_fused(layer, x, req, pos)
            return
        if layer.compress_ratio == 1:
            core = self.forward_metadata.core_metadata
            kv, _ = layer.compressor.project(x)
            slots = torch.where(
                core.c1_out_loc >= 0, core.c1_out_loc, torch.zeros_like(core.c1_out_loc)
            )
            self._low_ratio_write_group(
                layer,
                kv,
                slots,
                pos,
                fuse_index_store=(
                    x.is_cuda
                    and torch.version.cuda is not None
                    and _is_sm100_or_newer()
                ),
            )
            return
        if not (x.is_cuda and torch.version.cuda):
            self._low_ratio_compress_torch(layer, x, req, pos)
            return

        from sglang.kernels.ops.attention.dsv4.pair_pool_decode import pair_pool_decode

        core = self.forward_metadata.core_metadata
        state = self.token_to_kv_pool.get_attention_compress_states(layer.layer_id)
        kv, score = layer.compressor.project(x)
        pooled, group_pos, slots = pair_pool_decode(
            kv,
            score,
            pos,
            core.raw_out_loc,
            core.c2_out_loc,
            req,
            state.kv_score_buffer.kv,
            state.kv_score_buffer.score,
            state.kv_score_buffer.shape[0] - 1,
            ring_size=state.ring_size,
        )
        self._low_ratio_write_group(
            layer,
            pooled,
            slots,
            group_pos,
            fuse_index_store=_is_sm100_or_newer(),
        )

    def _low_ratio_compress_fused(self, layer, x, req, pos, *, draft_len=1) -> None:
        """Fused compressor write, index-key projection, then fused index-key write.
        Both write kernels consume metadata dtypes directly and suppress padded stores.
        """
        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.c2 import (
            c2_decode_norm_rope_store,
            c2_verify_norm_rope_store,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_rope import (
            index_k_norm_rope_pack_store,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
            index_k_norm_rope_pack_store_split,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        compressor = layer.compressor
        layer_id = layer.layer_id
        # Contiguous complex64 freqs_cis gives a real/imag-interleaved view without copying.
        freqs_cis = torch.view_as_real(layer.freqs_cis).flatten(-2)
        kv_cache = pool.get_extra_key_buffer(layer_id)
        page_size = pool.get_extra_key_page_size(layer_id)
        assert kv_cache is not None

        if layer.compress_ratio == 1:
            # At ratio 1, c1_out_loc equals the int64 raw_out_loc supplied by the scheduler.
            latent = c1_decode_norm_rope_store(
                compressor.wkv(x),
                compressor.norm.weight.data,
                pos,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
            )
            out_loc = core.c1_out_loc
        else:
            # pending-pair ring: | kv | score | at req * ring_size + pos % ring_size
            state = pool.get_attention_compress_states(layer_id)
            c2_compress = (
                c2_verify_norm_rope_store
                if draft_len > 1
                else c2_decode_norm_rope_store
            )
            verify_args = {"draft_len": draft_len} if draft_len > 1 else {}
            latent = c2_compress(
                compressor.project_fused(x),
                state.kv_score_buffer.kv_score,
                compressor.norm.weight.data,
                pos,
                req,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
                ring_size=state.ring_size,
                **verify_args,
            )
            out_loc = core.c2_out_loc

        indexer = layer.indexer
        if indexer is not None and indexer.owns_k:
            # out_loc is -1 for an incomplete group and 0 for padding; the kernel stores neither
            assert out_loc is not None
            k = indexer.forward_wk(latent)
            if pool.low_ratio_index_k_is_split(layer_id):
                # FlyDSL split payload / scale layout; same bytes as store_fp4_index_k_cache_split
                index_k_norm_rope_pack_store_split(
                    k,
                    indexer.k_norm.weight.data,
                    indexer.k_norm.eps,
                    freqs_cis,
                    pos,
                    out_loc,
                    pool.get_index_k_fp4_payload_buffer(layer_id),
                    pool.get_index_k_fp4_scale_buffer(layer_id),
                    ratio=layer.compress_ratio,
                )
            else:
                index_k_norm_rope_pack_store(
                    k,
                    indexer.k_norm.weight.data,
                    indexer.k_norm.eps,
                    freqs_cis,
                    pos,
                    out_loc,
                    pool.get_index_k_with_scale_buffer(layer_id),
                    ratio=layer.compress_ratio,
                )

    def _low_ratio_compress_torch(
        self, layer, x, req, pos, projected=None, *, fuse_index_store=False
    ) -> None:
        core = self.forward_metadata.core_metadata
        num_tokens = pos.shape[0]
        kv, score = projected if projected is not None else layer.compressor.project(x)
        if not num_tokens:
            return
        if layer.compress_ratio == 1:
            self._low_ratio_write_group(
                layer,
                kv,
                core.c1_out_loc[:num_tokens],
                pos,
                fuse_index_store=fuse_index_store,
            )
            return

        partner_kv, partner_score = self._low_ratio_pair_partners(
            layer_id=layer.layer_id,
            kv=kv,
            score=score,
            req=req,
            pos=pos,
            pad=core.raw_out_loc[:num_tokens] == 0,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        group_pos = torch.where(pos % 2 == 1, pos - 1, pos)
        out_loc = core.c2_out_loc[:num_tokens]
        slots = torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc))
        self._low_ratio_write_group(
            layer, pooled, slots, group_pos, fuse_index_store=fuse_index_store
        )

    def _low_ratio_pair_partners(
        self, *, layer_id, kv, score, req, pos, pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the preceding token, then publish this batch's trailing ring window.

        Each request occupies consecutive rows and positions. Keeping only its
        last ring_size rows gives every write a distinct live slot, including
        when a prefill chunk is longer than the ring. Reads precede all writes
        so wrapping cannot overwrite a cross-chunk partner.
        """
        state = self.token_to_kv_pool.get_attention_compress_states(layer_id)
        ring = state.ring_size
        num_tokens = pos.shape[0]
        if not num_tokens:
            return kv, score

        read_pos = (pos - 1).masked_fill(pad, -1)
        carried = state.get_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, read_pos)
        )
        in_batch = torch.zeros_like(pad)
        in_batch[1:] = (
            (req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1) & ~pad[1:] & ~pad[:-1]
        )
        partner_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried.kv)
        partner_score = torch.where(
            in_batch[:, None], torch.roll(score, 1, 0), carried.score
        )

        keep = ~pad
        if num_tokens > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        state.set_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, write_pos),
            KVAndScore.from_kv_score(kv=kv, score=score),
        )
        return partner_kv, partner_score

    def _low_ratio_write_group(
        self, layer, pooled, slots, group_pos, *, fuse_index_store=False
    ) -> None:
        pool = self.token_to_kv_pool
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        # Index keys come from the pre-RoPE latent, so publish them first. Stored
        # as fp4 (per-32 ue8m0, no hadamard), matching the reference indexer.
        if layer.indexer is not None and layer.indexer.owns_k:
            if (
                fuse_index_store
                and latent.is_cuda
                and torch.version.cuda is not None
                and latent.dtype == torch.bfloat16
                and layer.indexer.index_head_dim == 128
            ):
                from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
                    rope_fake_quant_pack_indexer,
                )

                indexer = layer.indexer
                k = indexer.k_norm(indexer.forward_wk(latent))
                rope_fake_quant_pack_indexer(
                    k,
                    freqs,
                    indexer.rope_head_dim,
                    cache=pool.get_index_k_with_scale_buffer(layer.layer_id),
                    loc=slots,
                )
            else:
                pool.set_index_k_fp4(
                    layer_id=layer.layer_id,
                    loc=slots,
                    cache_k=layer.indexer.index_keys(latent, freqs),
                )
        # The FlashMLA cache requantizes the FP4/E4M3 latent into its FP8 layout.
        latent = _rope_fq4(latent, freqs, layer.rope_head_dim, compressed_kv=True)
        pool.set_extra_key_buffer_fused(
            layer_id=layer.layer_id, loc=slots, cache_k=latent
        )

    def _low_ratio_extend_k_slots(self, *, ratio, lc_per_req, req_pool_indices, device):
        """Per request, the c1/c2 pool slots of its visible compressed positions, and
        each request's start offset in their concatenation."""
        slot_chunks, starts, start = [], [], 0
        for r, lc in enumerate(lc_per_req):
            starts.append(start)
            if lc == 0:
                continue
            j = torch.arange(lc, device=device)
            slot_chunks.append(
                self.req_to_token[req_pool_indices[r], j * ratio].to(torch.int64)
                // ratio
            )
            start += lc
        return slot_chunks, starts

    # TODO(candidate): torch prefill still publishes / consumes masks inline; same
    # move as above.
    def _low_ratio_index_topk_torch(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        # Attention scans sparse_topk_lengths slots and skips -1 entries.
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        # A compressed position is visible once the query has passed its last token.
        compress_lens = (pos + 1) // ratio
        topk = indexer.index_topk
        publish = [] if indexer.is_candidate_source else None
        consume = (
            published_masks(self.forward_metadata.candidate_metadata).request_masks
            if indexer.uses_candidates
            else None
        )
        for b, r in enumerate(torch.unique_consecutive(req).tolist()):
            tok = (req == r).nonzero().squeeze(1)
            lens = compress_lens[tok]
            lc = int(lens.max().item())
            if lc == 0:
                # Consumers address masks by request position, including empty requests.
                if publish is not None:
                    publish.append(
                        torch.zeros(0, 0, dtype=torch.bool, device=pos.device)
                    )
                continue
            j = torch.arange(lc, device=pos.device)
            slots_j = self.req_to_token[r, j * ratio].to(torch.int64) // ratio
            # Dequantize only this request's visible K rows; the full table is
            # pool-sized.
            index_k = pool.get_low_ratio_index_k_dequant(layer.layer_id, slots_j)
            k = min(topk, lc)
            # Every step below is per query row; chunk rows so the [rows, heads, lc]
            # bf16 scores stay under the budget (16 GiB at once for a 16k-token prompt).
            rows_per_chunk = max(
                1,
                _TORCH_INDEXER_SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2),
            )
            masks = [] if publish is not None else None
            for start in range(0, tok.numel(), rows_per_chunk):
                rows = slice(start, start + rows_per_chunk)
                tok_c, lens_c = tok[rows], lens[rows]
                s = indexer.scores(q[tok_c], index_k, weights[tok_c])
                s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
                if masks is not None:
                    masks.append(
                        select_candidate_blocks(
                            s,
                            lens_c[:, None],
                            topk_blocks=indexer.candidate_topk_blocks,
                            block_size=indexer.candidate_block_size,
                        )
                    )
                elif consume is not None:
                    s = s.masked_fill(~consume[b][rows], -torch.inf)
                idx = s.topk(k, dim=-1, sorted=False).indices
                if consume is not None and masks is None:
                    idx = mask_topk_scores(s, idx)
                    idx = idx.masked_fill(idx < 0, lc)
                idx = idx.sort(dim=-1).values
                reach = idx < lens_c[:, None]
                page_indices[tok_c, :k] = torch.where(
                    reach, slots_j[idx.clamp_max(lc - 1)], -1
                ).to(torch.int32)
                if raw_indices is not None:
                    raw_indices[tok_c, :k] = torch.where(reach, idx, -1).to(torch.int32)
            if masks is not None:
                publish.append(torch.cat(masks) if len(masks) > 1 else masks[0])
        if publish is not None:
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=publish
            )
