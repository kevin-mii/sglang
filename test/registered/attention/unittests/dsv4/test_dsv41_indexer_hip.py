"""The V4.1 low-ratio indexer on ROCm (gfx950): the FlyDSL fp4 paged logits
kernels (decode, target-verify rows and ragged prefill) plus the AOT paged top-k
transform against a torch golden of the reference scoring on the production
layout (64-slot indexer-K pages in the split payload / scale layout behind the
FULL 256-token page table); the backend entry points `low_ratio_index_topk_hip_*`
against the torch oracle `_low_ratio_index_topk_torch` on the same batch, pool and
metadata, including the candidate-mask flow from the source layer to a consumer
and the request-group split of the prefill logits; level one of the two-level
top-k (the length-bounded HIP helpers against the reference
`select_candidate_blocks` under kernel garbage past the reach, graph replay of the
source -> consumer hand-off, the decode body beyond the 16384-position candidate
span and the span skip below it); identity requests, whose visible compressed
context fits index_topk and are written without scoring; and the split-K Triton
GEMV route of the indexer head weights against the served aiter chain and an fp64
reference.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4.candidate_torch import CandidateMasks
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=420, suite="stage-b-test-1-gpu-small-amd-mi35x")

FULL_PAGE_SIZE = 256
INDEX_PAGE_SIZE = 64
HEAD_DIM = 128
N_HEADS = 32
TOPK = 512
# Released V4.1 config; the span 2048 x 8 = 16384 is where level one starts to bind.
TOPK_BLOCKS, BLOCK_SIZE = 2048, 8
SPAN = TOPK_BLOCKS * BLOCK_SIZE

# Head-weights route: the released checkpoint's hidden size and
# softmax_scale * n_heads ** -0.5 = 128 ** -0.5 * 32 ** -0.5, 2 ** -6 once rounded
# to fp32, so the aten multiply is an exact exponent shift.
HIDDEN = 5120
SCALE = 128**-0.5 * 32**-0.5


def golden_scores(q, k, weights):
    """Reference scoring (einsum(q, k).relu() * weights).sum(heads) in fp32: q [b, H, d],
    weights [b, H], k [n, d] shared by every row or [b, n, d] per row -> [b, n]."""
    eq = "bhd,bnd->bhn" if k.dim() == 3 else "bhd,nd->bhn"
    s = torch.einsum(eq, q.float(), k.float())
    return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)


def compressed_slot(full_page_table, ratio, j):
    """Slot of compressed position j through the FULL page table at `ratio`."""
    slots_per_page = FULL_PAGE_SIZE // ratio
    phys_page = full_page_table.gather(1, (j // slots_per_page).to(torch.int64))
    return phys_page.to(torch.int64) * slots_per_page + j % slots_per_page


def index_slots(page_table, pos):
    """Slot of compressed position `pos` through the expanded indexer page table."""
    return (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )


def sorted_rows(x):
    """Rows as sorted sets with the -1 padding last: the AOT top-k emits a row's set
    in arrival order."""
    return x.masked_fill(x < 0, torch.iinfo(x.dtype).max).sort(dim=-1).values


def sorted_by_raw(ri, pi):
    """Raw indices sorted by position with the -1 padding last, and the page indices
    carried along, so two selections of the same set compare column by column."""
    key = ri.masked_fill(ri == -1, torch.iinfo(ri.dtype).max)
    order = key.argsort(dim=-1)
    return ri.gather(-1, order), pi.gather(-1, order)


def reference_position_mask(logits, lens, topk_blocks, block_size):
    """The reference's level one on logits whose tail past the reach is -inf."""
    from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

    col = torch.arange(logits.shape[1], device=logits.device)
    pre = logits.masked_fill(col >= lens[:, None], -torch.inf)
    return select_candidate_blocks(
        pre, lens[:, None], topk_blocks=topk_blocks, block_size=block_size
    )


def ids_to_position_mask(ids, block_size, width):
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        candidate_block_ids_to_mask,
    )

    num_blocks = (width + block_size - 1) // block_size
    keep = candidate_block_ids_to_mask(ids, num_blocks)
    return keep.repeat_interleave(block_size, dim=-1)[:, :width]


def reference_consumer_rows(logits, lens, pos_mask, topk):
    """Per row: the set the reference consumer selects among the reachable candidates
    and its valid count; tie-free logits make it exact."""
    col = torch.arange(logits.shape[1], device=logits.device)
    out = []
    for b in range(logits.shape[0]):
        n = int(lens[b])
        cand = pos_mask[b] & (col < n)
        n_cand = int(cand.sum())
        k = min(topk, n_cand)
        s = logits[b].masked_fill(~cand, -torch.inf)
        out.append((set(s.topk(k).indices.tolist()), k))
    return out


class _StubIndexer:
    """Stand-in indexer: precomputed fp4-grid queries and head weights per token,
    fp32 golden scores, and the candidate-block config of the layer."""

    weights_proj_hip_max_tokens = -1  # the linear serves the head weights

    def __init__(
        self,
        q,
        weights,
        *,
        candidate_source=False,
        uses_candidates=False,
        candidate_blocks=(TOPK_BLOCKS, BLOCK_SIZE),
    ):
        self.q, self.w = q, weights
        self.index_topk = TOPK
        self.is_candidate_source = candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks, self.candidate_block_size = candidate_blocks
        self.owns_k = False
        self.n_local_heads = self.n_heads = q.shape[1]

    def queries(self, q_lora, freqs, positions=None):
        # the HIP path hands over the whole freqs table plus positions; the stub's
        # queries are precomputed per token so both are ignored
        return self.q[q_lora]

    def head_weights(self, x):
        return self.w[x]

    def scores(self, q, k, weights):
        return golden_scores(q, k, weights)


# -- the kernels: FlyDSL paged logits and the AOT paged top-k transform ---------


@unittest.skipUnless(is_hip(), "FlyDSL fp4 indexer kernels are ROCm only")
class TestFp4PagedLogitsKernels(CustomTestCase):
    def _setup(self, ratio):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            pack_fp4_query_flydsl,
            store_fp4_index_k_cache_split,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend import (
            _expand_index_page_table,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(ratio)
        bs, n_full_pages = 4, 8
        slots_per_page = FULL_PAGE_SIZE // ratio
        max_slots = n_full_pages * slots_per_page
        n_phys_pages = bs * n_full_pages + 3
        total_slots = n_phys_pages * slots_per_page
        n_index_pages = total_slots // INDEX_PAGE_SIZE

        full_page_table = (
            torch.randperm(n_phys_pages, device="cuda")[: bs * n_full_pages]
            .view(bs, n_full_pages)
            .to(torch.int32)
        )
        seq_lens = torch.tensor(
            [slots_per_page, slots_per_page + 5, 37, max_slots],
            dtype=torch.int32,
            device="cuda",
        )
        q = fake_quant_fp4(
            torch.randn(bs, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        k_all = fake_quant_fp4(
            torch.randn(total_slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.rand(bs, N_HEADS, device="cuda", dtype=torch.bfloat16)

        payload = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, 16, dtype=torch.uint8, device="cuda"
        ).view(torch.float4_e2m1fn_x2)
        scale = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, dtype=torch.uint8, device="cuda"
        )
        loc = torch.arange(total_slots, dtype=torch.int32, device="cuda")
        store_fp4_index_k_cache_split(
            k_all, payload, scale, loc, page_size=INDEX_PAGE_SIZE, rne=True
        )
        page_table = _expand_index_page_table(
            full_page_table,
            full_page_size=FULL_PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        q_fp4, q_scale = pack_fp4_query_flydsl(q)

        j = torch.arange(max_slots, device="cuda")
        slots = compressed_slot(full_page_table, ratio, j.expand(bs, -1))
        ref = golden_scores(q, k_all[slots], weights)
        visible = j[None, :] < seq_lens[:, None]
        return dict(
            bs=bs,
            max_slots=max_slots,
            k_all=k_all,
            full_page_table=full_page_table,
            seq_lens=seq_lens,
            payload=payload,
            scale=scale,
            page_table=page_table,
            q_fp4=q_fp4,
            q_scale=q_scale,
            weights=weights,
            ref=ref,
            visible=visible,
        )

    def _check_logits(self, st, logits):
        diff = (logits.float()[:, : st["max_slots"]] - st["ref"]).abs()
        rel = diff / st["ref"].abs().clamp_min(1.0)
        self.assertLess(
            rel[st["visible"]].max().item(),
            2e-2,
            msg=f"kernel logits diverged: max rel diff {rel[st['visible']].max()}",
        )

    def _check_topk(self, st, ratio, page_indices, raw_indices):
        for b in range(st["bs"]):
            n_valid = min(TOPK, int(st["seq_lens"][b]))
            sel_raw = raw_indices[b, :n_valid]
            sel_slot = page_indices[b, :n_valid]
            self.assertTrue(
                bool((sel_raw >= 0).all()) and bool((sel_raw < st["seq_lens"][b]).all())
            )
            self.assertTrue(bool((raw_indices[b, n_valid:] == -1).all()))
            self.assertTrue(bool((page_indices[b, n_valid:] == -1).all()))
            expect = compressed_slot(
                st["full_page_table"][b : b + 1], ratio, sel_raw[None].to(torch.int64)
            )[0]
            self.assertTrue(torch.equal(sel_slot.to(torch.int64), expect))
            ref_sel = (
                st["ref"][b]
                .masked_fill(~st["visible"][b], -torch.inf)
                .topk(n_valid)
                .indices
            )
            overlap = len(set(ref_sel.tolist()) & set(sel_raw.tolist())) / n_valid
            self.assertGreaterEqual(overlap, 0.9, msg=f"top-k overlap {overlap:.3f}")

    def _expand_to_verify_rows(self, st, ratio, block):
        """Target-verify rows: each request repeated `block` times, row j seeing one
        more compressed slot than row j - 1 (clamped to 1), with its own query."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            pack_fp4_query_flydsl,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend import (
            _expand_index_page_table,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(100 + ratio)
        bs = st["bs"]
        rows = bs * block
        offsets = torch.arange(block, device="cuda", dtype=torch.int32) - (block - 1)
        seq_lens = (st["seq_lens"][:, None] + offsets[None, :]).clamp_min(1).view(-1)
        full_page_table = st["full_page_table"].repeat_interleave(block, dim=0)
        page_table = _expand_index_page_table(
            full_page_table,
            full_page_size=FULL_PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        q = fake_quant_fp4(
            torch.randn(rows, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.rand(rows, N_HEADS, device="cuda", dtype=torch.bfloat16)
        q_fp4, q_scale = pack_fp4_query_flydsl(q)
        j = torch.arange(st["max_slots"], device="cuda")
        slots = compressed_slot(full_page_table, ratio, j.expand(rows, -1))
        ref = golden_scores(q, st["k_all"][slots], weights)
        visible = j[None, :] < seq_lens[:, None]
        return dict(
            st,
            bs=rows,
            full_page_table=full_page_table,
            seq_lens=seq_lens,
            page_table=page_table,
            q_fp4=q_fp4,
            q_scale=q_scale,
            weights=weights,
            ref=ref,
            visible=visible,
        )

    def _run_decode(self, ratio, verify_block=None):
        from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_fp4_paged_mqa_logits,
            prepare_fp4_decode_workspace,
        )

        st = self._setup(ratio)
        if verify_block is not None:
            st = self._expand_to_verify_rows(st, ratio, verify_block)
        workspace = prepare_fp4_decode_workspace(st["page_table"], st["seq_lens"])
        logits = aiter_fp4_paged_mqa_logits(
            q_fp4=st["q_fp4"],
            q_scale=st["q_scale"],
            k_payload=st["payload"],
            k_scale=st["scale"],
            weights=st["weights"],
            page_table=st["page_table"],
            c4_seq_lens=st["seq_lens"],
            weight_scale=1.0,
            is_decode=True,
            decode_workspace=workspace,
        )
        self._check_logits(st, logits)
        bs = st["bs"]
        page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        topk_transform_paged(
            logits,
            st["seq_lens"],
            st["page_table"],
            page_indices,
            INDEX_PAGE_SIZE,
            raw_indices,
        )
        self._check_topk(st, ratio, page_indices, raw_indices)

    def _run_prefill(self, ratio):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_fp4_paged_mqa_logits,
            prepare_fp4_prefill_workspace,
        )

        st = self._setup(ratio)
        for workspace in (
            None,
            prepare_fp4_prefill_workspace(st["page_table"], st["seq_lens"]),
        ):
            # One row per request stands in for one row per token: the prefill
            # kernel is row-wise, so the shapes are the same contract.
            logits = aiter_fp4_paged_mqa_logits(
                q_fp4=st["q_fp4"],
                q_scale=st["q_scale"],
                k_payload=st["payload"],
                k_scale=st["scale"],
                weights=st["weights"],
                page_table=st["page_table"],
                c4_seq_lens=st["seq_lens"],
                weight_scale=1.0,
                is_decode=False,
                prefill_workspace=workspace,
            )
            self._check_logits(st, logits)

    def test_decode_ratio1_matches_golden(self):
        """The decode logits and the paged transform must reproduce the reference
        scores and slots through a permuted FULL page table at ratio 1."""
        self._run_decode(ratio=1)

    def test_decode_ratio2_matches_golden(self):
        """Ratio 2 halves the slots per FULL page; the expanded table must still
        resolve every compressed position."""
        self._run_decode(ratio=2)

    def test_decode_body_on_target_verify_rows_ratio1(self):
        """Verify rows (one per draft token) must route through the decode body with
        per-row page tables."""
        self._run_decode(ratio=1, verify_block=6)

    def test_decode_body_on_target_verify_rows_ratio2(self):
        """Verify rows at ratio 2, where neighbouring rows share a compressed slot."""
        self._run_decode(ratio=2, verify_block=6)

    def test_prefill_ratio1_matches_golden(self):
        """The row-wise prefill kernel must match the reference with and without a
        prepared workspace."""
        self._run_prefill(ratio=1)

    def test_prefill_ratio2_matches_golden(self):
        """Prefill at ratio 2, with and without a prepared workspace."""
        self._run_prefill(ratio=2)


# -- the backend entry points against the torch oracle -------------------------


class _LowRatioBackendCase(CustomTestCase):
    """Shared fixture: a pool of fp4 indexer K behind a permuted FULL page table, one
    batch of requests, and the backend entry points run on a bare
    `DeepseekV4HipRadixBackend` with a stub indexer."""

    # (candidate_topk_blocks, candidate_block_size) of the stub indexer.
    CANDIDATE_BLOCKS = (TOPK_BLOCKS, BLOCK_SIZE)

    def _setup(self, ratio, seq_lens, extend_lens, seed=0):
        """Requests of `seq_lens` tokens, the last `extend_lens` of each in the batch;
        decode when every request extends by one."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            store_fp4_index_k_cache_split,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend import (
            _expand_index_page_table,
            _low_ratio_sparse_buffers,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            refresh_low_ratio_prefill_workspaces,
        )
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(1000 * ratio + seed)
        dev = "cuda"
        bs = len(seq_lens)
        slots_per_page = FULL_PAGE_SIZE // ratio
        n_full_pages = max((s + FULL_PAGE_SIZE - 1) // FULL_PAGE_SIZE for s in seq_lens)
        n_phys_pages = bs * n_full_pages + 2
        total_slots = n_phys_pages * slots_per_page
        full_page_table = (
            torch.randperm(n_phys_pages, device=dev)[: bs * n_full_pages]
            .view(bs, n_full_pages)
            .to(torch.int32)
        )
        req_to_token = torch.zeros(
            bs, n_full_pages * FULL_PAGE_SIZE, dtype=torch.int32, device=dev
        )
        p = torch.arange(n_full_pages * FULL_PAGE_SIZE, device=dev)
        for b in range(bs):
            req_to_token[b] = (
                full_page_table[b, p // FULL_PAGE_SIZE].to(torch.int64) * FULL_PAGE_SIZE
                + p % FULL_PAGE_SIZE
            ).to(torch.int32)

        k_all = fake_quant_fp4(
            torch.randn(total_slots, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        )
        n_index_pages = total_slots // INDEX_PAGE_SIZE
        payload = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, 16, dtype=torch.uint8, device=dev
        ).view(torch.float4_e2m1fn_x2)
        scale = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, dtype=torch.uint8, device=dev
        )
        store_fp4_index_k_cache_split(
            k_all,
            payload,
            scale,
            torch.arange(total_slots, dtype=torch.int32, device=dev),
            page_size=INDEX_PAGE_SIZE,
            rne=True,
        )
        pool = SimpleNamespace(
            get_index_k_fp4_payload_buffer=lambda layer_id: payload,
            get_index_k_fp4_scale_buffer=lambda layer_id: scale,
            get_low_ratio_index_k_dequant=lambda layer_id, slots: k_all[
                slots.to(torch.int64)
            ],
        )

        pos_list, req_list = [], []
        for b, (s, e) in enumerate(zip(seq_lens, extend_lens)):
            pos_list += list(range(s - e, s))
            req_list += [b] * e
        pos = torch.tensor(pos_list, dtype=torch.int64, device=dev)
        req = torch.tensor(req_list, dtype=torch.int64, device=dev)
        num_tokens = pos.numel()
        q = fake_quant_fp4(
            torch.randn(num_tokens, N_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        )
        weights = torch.rand(num_tokens, N_HEADS, device=dev, dtype=torch.bfloat16)
        tok_ids = torch.arange(num_tokens, device=dev)
        is_decode = all(e == 1 for e in extend_lens)

        # Per-request FULL page table in decode, per-token in prefill, as the
        # backend builds them.
        base_table = (
            full_page_table
            if is_decode
            else (req_to_token[req, ::FULL_PAGE_SIZE] // FULL_PAGE_SIZE).to(torch.int32)
        )
        page_table = _expand_index_page_table(
            base_table,
            full_page_size=FULL_PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        lens_raw = ((pos + 1) // ratio).to(torch.int32)
        lens_clamp1 = lens_raw.clamp_min(1)

        def metadata():
            """Prefill rows carry the raw visible count, decode rows the clamp-1 count,
            as the backend builds it."""
            _, page_indices, raw_indices = _low_ratio_sparse_buffers(
                lens_clamp1, TOPK, is_prefill=True
            )
            core = SimpleNamespace(
                sparse_page_indices=lambda r: page_indices,
                sparse_raw_indices=lambda r: raw_indices,
            )
            indexer_metadata = PagedIndexerMetadata(
                page_size=FULL_PAGE_SIZE,
                page_table=page_table,
                c4_seq_lens=lens_clamp1 if is_decode else lens_raw,
                use_topk_v2=False,
                compress_ratio=ratio,
                index_page_size=INDEX_PAGE_SIZE,
            )
            meta = SimpleNamespace(
                core_metadata=core,
                low_ratio_indexer_metadata=lambda r: indexer_metadata,
                fp4_low_ratio_decode_workspaces={},
                fp4_low_ratio_prefill_workspaces=(
                    {}
                    if is_decode
                    else refresh_low_ratio_prefill_workspaces(
                        {ratio: indexer_metadata}, None
                    )
                ),
            )
            return meta, page_indices, raw_indices

        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: is_decode,
                is_extend=lambda: not is_decode,
                is_target_verify=lambda: False,
            ),
            seq_lens_cpu=list(seq_lens),
            extend_seq_lens_cpu=list(extend_lens),
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32, device=dev),
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device=dev),
        )
        return SimpleNamespace(
            ratio=ratio,
            dev=dev,
            bs=bs,
            pool=pool,
            req_to_token=req_to_token,
            pos=pos,
            req=req,
            q=q,
            weights=weights,
            tok_ids=tok_ids,
            page_table=page_table,
            lens=lens_raw.to(torch.int64),
            metadata=metadata,
            forward_batch=forward_batch,
            is_decode=is_decode,
        )

    @staticmethod
    def _layer_inputs(st):
        """Another index-source layer's queries and head weights over the same K."""
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        return fake_quant_fp4(torch.randn_like(st.q)), torch.rand_like(st.weights)

    @staticmethod
    def _decode_batch(seq_lens, decode=True):
        """The two fields the decode predicates read."""
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: decode),
            seq_lens_cpu=torch.tensor(seq_lens),
        )

    def _run(
        self,
        st,
        path,
        *,
        inputs=None,
        skip=False,
        candidate_source=False,
        uses_candidates=False,
        masks=None,
        candidate_span=None,
        forward_batch=None,
    ):
        """Run one indexer layer through `path` ("extend", "decode" or "torch") on a
        bare backend; returns (page_indices, raw_indices, published candidate masks).
        `inputs` overrides the layer's (queries, head weights); `forward_batch`
        overrides the fixture's batch on the decode path."""
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_index_topk_hip_decode,
            low_ratio_index_topk_hip_extend,
        )

        meta, page_indices, raw_indices = st.metadata()
        backend = object.__new__(DeepseekV4HipRadixBackend)
        backend.forward_metadata = meta
        backend.token_to_kv_pool = st.pool
        backend.req_to_token = st.req_to_token
        backend.candidate_masks = masks
        backend.low_ratio_identity_skip = skip
        backend.low_ratio_candidate_span = candidate_span
        backend.index_topk = TOPK
        q, w = inputs if inputs is not None else (st.q, st.weights)
        indexer = _StubIndexer(
            q,
            w,
            candidate_source=candidate_source,
            uses_candidates=uses_candidates,
            candidate_blocks=self.CANDIDATE_BLOCKS,
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=st.ratio,
            indexer=indexer,
            freqs_cis=torch.zeros(int(st.pos.max()) + 1, device=st.dev),
        )
        if path == "extend":
            low_ratio_index_topk_hip_extend(
                backend, layer, st.tok_ids, st.tok_ids, st.pos, st.forward_batch
            )
        elif path == "decode":
            low_ratio_index_topk_hip_decode(
                backend,
                layer,
                st.tok_ids,
                st.tok_ids,
                st.pos,
                forward_batch=(
                    forward_batch if forward_batch is not None else st.forward_batch
                ),
            )
        else:
            # the shared torch oracle publishes and consumes through the metadata
            meta.candidate_metadata = (
                None if masks is None else CandidateMasks(request_masks=masks)
            )
            backend._low_ratio_index_topk_torch(
                layer, st.tok_ids, st.tok_ids, st.req, st.pos
            )
            published = meta.candidate_metadata
            return (
                page_indices,
                raw_indices,
                None if published is None else published.request_masks,
            )
        return page_indices, raw_indices, backend.candidate_masks

    def _dense_masks(self, st, masks, seq_lens, extend_lens):
        """The HIP publication (one Optional[CandidateBlocks] per request) as the torch
        oracle's dense bool [t_len, lc] masks: None keeps every reachable block."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            candidate_block_ids_to_mask,
        )

        out, row = [], 0
        for cb, s, e in zip(masks, seq_lens, extend_lens):
            lc = s // st.ratio
            lens = ((st.pos[row : row + e] + 1) // st.ratio)[:, None]
            row += e
            j = torch.arange(lc, device=st.dev)
            block = self.CANDIDATE_BLOCKS[1]
            if cb is None:
                out.append((j[None, :] // block) <= ((lens - 1) // block))
                continue
            num_blocks = (lc + block - 1) // block
            keep = candidate_block_ids_to_mask(cb.ids, num_blocks)
            out.append(keep.repeat_interleave(block, dim=1)[:, :lc])
        return out

    def _assert_candidates_equal(self, a_masks, b_masks, msg):
        """Two HIP publications keep the same blocks per row (ids are unordered). None
        stands for every reachable block, which a scored publication spells out as the
        blocks 0..n-1 of each row."""

        def keeps_every_block(cb):
            n = (cb.compact_lens // cb.block_size)[:, None]
            pad = 1 << 30  # -1 padding sorts after every block id
            ids = cb.ids.masked_fill(cb.ids < 0, pad).sort(dim=1).values
            j = torch.arange(ids.shape[1], device=ids.device)[None, :]
            return bool(torch.equal(ids, torch.where(j < n, j, pad).to(ids.dtype)))

        self.assertEqual(len(a_masks), len(b_masks), msg)
        for x, y in zip(a_masks, b_masks):
            if x is None or y is None:
                other = y if x is None else x
                self.assertTrue(other is None or keeps_every_block(other), msg)
                continue
            self.assertTrue(torch.equal(x.compact_lens, y.compact_lens), msg)
            self.assertTrue(
                torch.equal(x.ids.sort(dim=1).values, y.ids.sort(dim=1).values),
                f"{msg}: candidate blocks differ",
            )

    def _assert_selection(self, a, b, msg, *, exact_rows=None, masks=True):
        """Two (page_indices, raw_indices, masks) results agree: `exact_rows` (default
        all) exactly; the other rows by their -1 pattern and tie-robust selected
        positions (torch.topk breaks ties in no fixed order)."""
        a_pi, a_ri, a_masks = a
        b_pi, b_ri, b_masks = b
        if exact_rows is None:
            exact_rows = torch.ones(a_ri.shape[0], dtype=torch.bool, device=a_ri.device)
        self.assertTrue(
            torch.equal(a_ri[exact_rows], b_ri[exact_rows]),
            f"{msg}: raw indices differ",
        )
        self.assertTrue(
            torch.equal(a_pi[exact_rows], b_pi[exact_rows]),
            f"{msg}: page indices differ",
        )
        scored = ~exact_rows
        if bool(scored.any()):
            a_ri_s, a_pi_s = sorted_by_raw(a_ri[scored], a_pi[scored])
            b_ri_s, b_pi_s = sorted_by_raw(b_ri[scored], b_pi[scored])
            self.assertTrue(torch.equal(a_ri_s == -1, b_ri_s == -1), msg)
            self.assertTrue(torch.equal(a_pi_s == -1, b_pi_s == -1), msg)
            agree = ((a_ri_s == b_ri_s) | (a_ri_s == -1)).float().mean().item()
            self.assertGreaterEqual(
                agree, 0.95, f"{msg}: scored rows agreement {agree}"
            )
        if not masks:
            return
        if a_masks is None or b_masks is None:
            self.assertIs(a_masks, b_masks, msg)
            return
        self._assert_candidates_equal(a_masks, b_masks, msg)

    def _assert_agrees(self, a, b, msg):
        """Every row scored: the same -1 pattern and 95% of the selected positions."""
        none = torch.zeros(a[1].shape[0], dtype=torch.bool, device=a[1].device)
        self._assert_selection(a, b, msg, exact_rows=none, masks=False)


@unittest.skipUnless(is_hip(), "FlyDSL fp4 indexer kernels are ROCm only")
class TestLowRatioIndexerHipPaths(_LowRatioBackendCase):
    """The decode and ragged-prefill entry points against the torch oracle."""

    # Toy candidate blocks so the mask binds on requests of a few hundred tokens.
    CANDIDATE_BLOCKS = (2, 32)

    @staticmethod
    def _drop_filler(st, result, masks):
        """Keep only the candidate selections: a row short of top-k is padded with tied
        -inf positions, and torch.topk's pick among them is nondeterministic on ROCm."""
        pi, ri, _ = result
        ri, pi = ri.clone(), pi.clone()
        rows = 0
        lens = (st.pos + 1) // st.ratio
        for mask in masks:
            t_len = mask.shape[0]
            r = ri[rows : rows + t_len].long()
            valid = r != -1
            cand = torch.zeros_like(valid)
            cand[valid] = mask[
                torch.arange(t_len, device=r.device)[:, None].expand_as(r)[valid],
                r[valid],
            ] & (r[valid] < lens[rows : rows + t_len, None].expand_as(r)[valid])
            ri[rows : rows + t_len] = ri[rows : rows + t_len].masked_fill(~cand, -1)
            pi[rows : rows + t_len] = pi[rows : rows + t_len].masked_fill(~cand, -1)
            rows += t_len
        # Compact the kept candidates so two paths that kept the same set compare
        # equal column by column.
        ri, pi = sorted_by_raw(ri, pi)
        return pi, ri, None

    def test_prefill_path_agrees_with_torch_path(self):
        """The prefill kernel path must select the oracle's set per token, publish the
        oracle's candidate masks as a source, and honour them as a consumer."""
        for ratio in (1, 2):
            seq_lens, extend_lens = [300, 45, 700], [300, 45, 200]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=extend_lens)
            k = self._run(st, "extend", candidate_source=True)
            t = self._run(st, "torch", candidate_source=True)
            self._assert_agrees(k, t, f"{ratio=} source")
            k_dense, t_masks = self._dense_masks(st, k[2], seq_lens, extend_lens), t[2]
            self.assertEqual(len(k_dense), len(t_masks))
            for a, b in zip(k_dense, t_masks):
                self.assertEqual(a.shape, b.shape)
                self.assertGreaterEqual((a == b).float().mean().item(), 0.95)
            c = self._run(st, "extend", uses_candidates=True, masks=k[2])
            d = self._run(st, "torch", uses_candidates=True, masks=k_dense)
            self._assert_agrees(
                self._drop_filler(st, c, k_dense),
                self._drop_filler(st, d, k_dense),
                f"{ratio=} consumer",
            )

    def test_prefill_request_groups_are_neutral(self):
        """Splitting the logits into request groups must not change the selection or the
        candidate masks."""
        import sglang.srt.layers.attention.dsv4.low_ratio_backend_hip as hip

        for ratio in (1, 2):
            seq_lens, extend_lens = [300, 45, 700], [300, 45, 200]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=extend_lens)
            w_pi, w_ri, w_masks = self._run(st, "extend", candidate_source=True)
            with mock.patch.object(hip, "logits_rows_per_chunk", return_value=1):
                s_pi, s_ri, s_masks = self._run(st, "extend", candidate_source=True)
                c = self._run(st, "extend", uses_candidates=True, masks=s_masks)
            w_ri, w_pi = sorted_by_raw(w_ri, w_pi)
            s_ri, s_pi = sorted_by_raw(s_ri, s_pi)
            self.assertTrue(torch.equal(w_ri, s_ri) and torch.equal(w_pi, s_pi))
            self._assert_candidates_equal(w_masks, s_masks, f"{ratio=}")
            d = self._run(st, "extend", uses_candidates=True, masks=w_masks)
            w_dense = self._dense_masks(st, w_masks, seq_lens, extend_lens)
            c_pi, c_ri, _ = self._drop_filler(st, c, w_dense)
            d_pi, d_ri, _ = self._drop_filler(st, d, w_dense)
            self.assertTrue(torch.equal(c_ri, d_ri) and torch.equal(c_pi, d_pi))

    def test_decode_path_agrees_with_torch_path(self):
        """The decode path must select the same set per row as the torch path (the
        transform returns rows unsorted) and resolve the same slots for it."""
        for ratio in (1, 2):
            st = self._setup(
                ratio, seq_lens=[300, 45, 700, 1], extend_lens=[1, 1, 1, 1]
            )
            k_pi, k_ri, _ = self._run(st, "decode")
            t_pi, t_ri, _ = self._run(st, "torch")
            for b in range(k_ri.shape[0]):
                lc = int((st.pos[b] + 1) // ratio)
                if lc == 0:
                    # The decode kernel scores the clamp-1 length and selects the
                    # dummy slot 0 where torch leaves the row at -1.
                    self.assertTrue(bool((t_ri[b] == -1).all()))
                    self.assertTrue(bool((k_ri[b, 1:] == -1).all()))
                    continue
                n_valid = min(TOPK, lc)
                self.assertEqual(int((k_ri[b] >= 0).sum()), n_valid, f"{ratio=} {b=}")
                self.assertEqual(int((t_ri[b] >= 0).sum()), n_valid, f"{ratio=} {b=}")
                got, ref = (
                    set(k_ri[b, :n_valid].tolist()),
                    set(t_ri[b, :n_valid].tolist()),
                )
                overlap = len(got & ref) / n_valid
                self.assertGreaterEqual(
                    overlap, 0.95, f"{ratio=} {b=} overlap {overlap}"
                )
                # Slots resolve the same raw positions through the same page table.
                order = torch.argsort(k_ri[b, :n_valid])
                sorted_k_ri = k_ri[b, :n_valid][order]
                sorted_k_pi = k_pi[b, :n_valid][order]
                common = torch.isin(sorted_k_ri, t_ri[b, :n_valid])
                t_pos = {
                    int(r): int(p)
                    for r, p in zip(
                        t_ri[b, :n_valid].tolist(), t_pi[b, :n_valid].tolist()
                    )
                }
                for r, p in zip(
                    sorted_k_ri[common].tolist(), sorted_k_pi[common].tolist()
                ):
                    self.assertEqual(p, t_pos[r])


@unittest.skipUnless(is_hip(), "FlyDSL fp4 indexer kernels are ROCm only")
class TestTwoLevelDecodeHip(_LowRatioBackendCase):
    """Level one of the two-level top-k on the decode path: the candidate-source layer
    keeps TOPK_BLOCKS x BLOCK_SIZE positions and the later ratio-1 sources select
    inside them, so the selection only diverges from the plain paged top-k once a
    request has more than SPAN compressed positions."""

    def _garbage_tail(self, logits, lens):
        """Kernel garbage past each row's reach (large positives on even rows, NaN on
        odd), so a helper reading the tail fails loudly."""
        col = torch.arange(logits.shape[1], device=logits.device)
        tail = col[None, :] >= lens[:, None]
        odd = (torch.arange(logits.shape[0], device=logits.device) % 2 == 1)[:, None]
        logits = logits.masked_fill(tail & ~odd, 1e4)
        return logits.masked_fill(tail & odd, torch.nan)

    def _check_consumer(self, logits, seq, cands, page_table, msg):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            topk_within_candidate_blocks_hip,
        )

        rows, width = logits.shape
        page_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        raw_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        topk_within_candidate_blocks_hip(
            logits,
            seq,
            cands,
            page_table=page_table,
            page_size=INDEX_PAGE_SIZE,
            page_indices=page_indices,
            raw_indices=raw_indices,
        )
        pos_mask = ids_to_position_mask(cands.ids, cands.block_size, width)
        for b, (want, k) in enumerate(
            reference_consumer_rows(logits, seq, pos_mask, TOPK)
        ):
            got = raw_indices[b]
            self.assertTrue(bool((got[:k] >= 0).all()), f"{msg}: prefix row {b}")
            self.assertTrue(bool((got[k:] == -1).all()), f"{msg}: padding row {b}")
            self.assertEqual(set(got[:k].tolist()), want, f"{msg}: selection row {b}")
            sel = got[:k].to(torch.int64)
            expect = index_slots(page_table[b : b + 1], sel[None])[0]
            self.assertTrue(
                torch.equal(page_indices[b, :k].to(torch.int64), expect),
                f"{msg}: slots row {b}",
            )
            self.assertTrue(bool((page_indices[b, k:] == -1).all()))

    def test_level_one_matches_reference_under_garbage_tail(self):
        """The HIP block top-k (AOT row-split and torch fallback) must publish the
        reference's blocks and never read past a row's reach."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            select_candidate_blocks_hip,
        )

        torch.manual_seed(11)
        cases = (
            # Released blocks, a rectangle just wider than the longest row.
            (TOPK_BLOCKS, BLOCK_SIZE, 40000, [3, 8, 16384, 16385, 20000, 40000, 1]),
            # Released blocks on a 1M-wide rectangle, the page table's capacity on
            # a 1M-context server: the block top-k takes the AOT row-split path.
            (TOPK_BLOCKS, BLOCK_SIZE, 1 << 20, [16385, 131072, 7, 600]),
            # Toy blocks, as the backend path tests use: the torch.topk fallback.
            (2, 32, 700, [1, 300, 45, 700, 33]),
        )
        for topk_blocks, block_size, width, lens in cases:
            with self.subTest(topk_blocks=topk_blocks, width=width, lens=lens):
                seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
                raw = torch.randn(len(lens), width, device="cuda")
                raw = self._garbage_tail(raw, seq)
                expected = reference_position_mask(raw, seq, topk_blocks, block_size)

                cands = select_candidate_blocks_hip(
                    raw, seq, topk_blocks=topk_blocks, block_size=block_size
                )
                ids = cands.ids
                self.assertEqual(ids.shape, (len(lens), topk_blocks))
                self.assertTrue(
                    torch.equal(
                        cands.compact_lens.cpu(),
                        torch.tensor(
                            [
                                min((n + block_size - 1) // block_size, topk_blocks)
                                * block_size
                                for n in lens
                            ],
                            dtype=torch.int32,
                        ),
                    )
                )
                self.assertEqual(ids.dtype, torch.int32)
                got = ids_to_position_mask(ids, block_size, width)
                self.assertTrue(torch.equal(got, expected), "published blocks")
                for b, n in enumerate(lens):
                    row = ids[b]
                    n_ids = int((row >= 0).sum())
                    self.assertTrue(bool((row[:n_ids] >= 0).all()), "padding last")
                    self.assertLessEqual(
                        int(got[b, :n].sum()), topk_blocks * block_size
                    )

                n_pages = (width + INDEX_PAGE_SIZE - 1) // INDEX_PAGE_SIZE
                page_table = torch.stack(
                    [torch.randperm(n_pages, device="cuda") for _ in lens]
                ).to(torch.int32)
                self._check_consumer(raw, seq, cands, page_table, "consumer")

    def test_graph_replay_hand_off(self):
        """Source then consumer captured once must replay on new lengths and logits with
        no host sync."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            select_candidate_blocks_hip,
            topk_within_candidate_blocks_hip,
        )

        torch.manual_seed(3)
        bs, width = 2, 1 << 17
        n_pages = width // INDEX_PAGE_SIZE
        logits = torch.empty(bs, width, device="cuda")
        seq = torch.empty(bs, dtype=torch.int32, device="cuda")
        page_table = torch.stack(
            [torch.randperm(n_pages, device="cuda") for _ in range(bs)]
        ).to(torch.int32)
        page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")

        def step():
            cands = select_candidate_blocks_hip(
                logits, seq, topk_blocks=TOPK_BLOCKS, block_size=BLOCK_SIZE
            )
            topk_within_candidate_blocks_hip(
                logits,
                seq,
                cands,
                page_table=page_table,
                page_size=INDEX_PAGE_SIZE,
                page_indices=page_indices,
                raw_indices=raw_indices,
            )
            return cands.ids

        def fill(lens):
            seq.copy_(torch.tensor(lens, dtype=torch.int32))
            logits.copy_(self._garbage_tail(torch.randn(bs, width, device="cuda"), seq))

        fill([20000, 5])
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ids = step()
        torch.cuda.synchronize()

        for lens in ([131072, 16385], [16384, 1], [70001, 33]):
            with self.subTest(lens=lens):
                fill(lens)
                graph.replay()
                torch.cuda.synchronize()
                got_pi, got_ri, got_ids = (
                    page_indices.clone(),
                    raw_indices.clone(),
                    ids.clone(),
                )
                expected = reference_position_mask(logits, seq, TOPK_BLOCKS, BLOCK_SIZE)
                self.assertTrue(
                    torch.equal(
                        ids_to_position_mask(got_ids, BLOCK_SIZE, width), expected
                    )
                )
                # The eager run of the same kernels on the same inputs.
                eager_ids = step()
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(sorted_rows(eager_ids), sorted_rows(got_ids))
                )
                self.assertTrue(
                    torch.equal(sorted_rows(raw_indices), sorted_rows(got_ri))
                )
                self.assertTrue(
                    torch.equal(sorted_rows(page_indices), sorted_rows(got_pi))
                )
                for b, (want, k) in enumerate(
                    reference_consumer_rows(logits, seq, expected, TOPK)
                ):
                    self.assertEqual(set(got_ri[b, :k].tolist()), want, f"row {b}")
                    self.assertTrue(bool((got_ri[b, k:] == -1).all()))

    @staticmethod
    def _row_set(ri, b):
        sel = ri[b]
        return set(sel[sel >= 0].tolist())

    @staticmethod
    def _inside(sel, pos_mask_row):
        if not sel:
            return 1.0
        idx = torch.tensor(sorted(sel), dtype=torch.int64, device=pos_mask_row.device)
        return pos_mask_row[idx].float().mean().item()

    def test_decode_body_selects_inside_candidates_beyond_16k(self):
        """Past 16384 compressed positions the consumer must select inside the published
        candidate blocks and match the torch oracle's two-level selection."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            CandidateBlocks,
        )

        seq_lens = [20000, 16384, 16385, 33000, 300]
        st = self._setup(1, seq_lens=seq_lens, extend_lens=[1] * len(seq_lens))
        consumer = self._layer_inputs(st)
        width = st.page_table.shape[1] * INDEX_PAGE_SIZE
        self.assertGreater(width, SPAN)

        # The source publishes the candidate blocks ...
        s_pi, s_ri, cands = self._run(st, "decode", candidate_source=True)
        self.assertIsInstance(cands, CandidateBlocks)
        ids = cands.ids
        self.assertEqual(ids.shape, (st.bs, TOPK_BLOCKS))
        h_mask = ids_to_position_mask(ids, BLOCK_SIZE, width)
        # ... against the oracle's on its own (golden fp32) logits.
        _, _, t_masks = self._run(st, "torch", candidate_source=True)
        # The consumer layer selects inside the published blocks ...
        c_pi, c_ri, _ = self._run(
            st, "decode", inputs=consumer, uses_candidates=True, masks=cands
        )
        # ... the same layer without the filter (the path before this change) ...
        u_pi, u_ri, _ = self._run(st, "decode", inputs=consumer)
        # ... and the oracle consumer restricted to the same blocks, so the
        # comparison isolates the top-k from the logits kernels' small differences.
        o_masks = [h_mask[b : b + 1, : int(st.lens[b])] for b in range(st.bs)]
        _, o_ri, _ = self._run(
            st, "torch", inputs=consumer, uses_candidates=True, masks=o_masks
        )

        bound = False
        for b in range(st.bs):
            lc = int(st.lens[b])
            n_valid = min(TOPK, lc)
            for name, ri in (
                ("source", s_ri),
                ("consumer", c_ri),
                ("unfiltered", u_ri),
            ):
                self.assertEqual(int((ri[b] >= 0).sum()), n_valid, f"{name} row {b}")
                self.assertTrue(bool((ri[b, :n_valid] >= 0).all()), f"{name} row {b}")
            agree = (h_mask[b, :lc] == t_masks[b][0, :lc]).float().mean().item()
            self.assertGreaterEqual(agree, 0.95, f"mask agreement {agree} row {b}")
            self.assertLessEqual(int(h_mask[b, :lc].sum()), SPAN)

            consumer_set = self._row_set(c_ri, b)
            unfiltered = self._row_set(u_ri, b)
            oracle = self._row_set(o_ri, b)
            overlap = len(consumer_set & oracle) / n_valid
            self.assertGreaterEqual(
                overlap, 0.95, f"consumer vs oracle {overlap} row {b}"
            )
            self.assertEqual(
                self._inside(consumer_set, h_mask[b]),
                1.0,
                f"selection escaped, row {b}",
            )
            # Slots resolve the selected positions through the request's page table.
            sel = c_ri[b, :n_valid].to(torch.int64)
            expect = index_slots(st.page_table[b : b + 1], sel[None])[0]
            self.assertTrue(torch.equal(c_pi[b, :n_valid].to(torch.int64), expect))
            if lc <= SPAN:
                # Every block is a candidate: the filter changes nothing.
                self.assertEqual(consumer_set, unfiltered, f"row {b}")
                continue
            bound = True
            escaped = 1.0 - self._inside(unfiltered, h_mask[b])
            if lc >= 1.1 * SPAN:
                # With 10% or more of the blocks dropped the consumer's own top-k
                # lands in them (a row of 16385 drops one block of eight).
                self.assertGreater(
                    escaped,
                    0.0,
                    f"unfiltered selection stayed inside the blocks, row {b}",
                )
        self.assertTrue(bound, "no request exercised a binding candidate mask")

    # -- the candidate span: both levels skipped when the batch fits it --------

    def test_decode_body_skips_both_levels_when_batch_fits_span(self):
        """With the batch inside the span every block is a candidate, so source and
        consumer must select the identical set and slots as the filtered path."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            CandidateBlocks,
        )

        seq_lens = [1, 300, 4096, 8192, 16383, 16384]
        st = self._setup(1, seq_lens=seq_lens, extend_lens=[1] * len(seq_lens))
        consumer = self._layer_inputs(st)

        f_s_pi, f_s_ri, cands = self._run(st, "decode", candidate_source=True)
        self.assertIsInstance(cands, CandidateBlocks)
        f_c_pi, f_c_ri, _ = self._run(
            st, "decode", inputs=consumer, uses_candidates=True, masks=cands
        )
        s_pi, s_ri, published = self._run(
            st, "decode", candidate_source=True, candidate_span=SPAN
        )
        self.assertIsNone(published, "the source published under the skip")
        c_pi, c_ri, _ = self._run(
            st,
            "decode",
            inputs=consumer,
            uses_candidates=True,
            masks=None,
            candidate_span=SPAN,
        )
        for name, a, b in (
            ("source raw", s_ri, f_s_ri),
            ("source slots", s_pi, f_s_pi),
            ("consumer raw", c_ri, f_c_ri),
            ("consumer slots", c_pi, f_c_pi),
        ):
            self.assertTrue(torch.equal(sorted_rows(a), sorted_rows(b)), name)
        for b, n in enumerate(seq_lens):
            self.assertEqual(int((c_ri[b] >= 0).sum()), min(TOPK, n), f"row {b}")

        # A consumer that did not skip while the source did fails loudly.
        with self.assertRaises(AssertionError):
            self._run(st, "decode", inputs=consumer, uses_candidates=True, masks=None)

    def test_decode_body_keeps_filter_when_a_row_exceeds_span(self):
        """One row past the span must keep the whole batch on the filtered path, with
        the same blocks and selection as a backend without the span."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            CandidateBlocks,
        )

        seq_lens = [16385, 5]
        st = self._setup(1, seq_lens=seq_lens, extend_lens=[1] * len(seq_lens))
        consumer = self._layer_inputs(st)
        _, s_ri, cands = self._run(
            st, "decode", candidate_source=True, candidate_span=SPAN
        )
        self.assertIsInstance(cands, CandidateBlocks)
        _, f_ri, f_cands = self._run(st, "decode", candidate_source=True)
        self.assertTrue(torch.equal(sorted_rows(s_ri), sorted_rows(f_ri)))
        self.assertTrue(
            torch.equal(
                ids_to_position_mask(cands.ids, BLOCK_SIZE, 16392),
                ids_to_position_mask(f_cands.ids, BLOCK_SIZE, 16392),
            )
        )
        _, c_ri, _ = self._run(
            st,
            "decode",
            inputs=consumer,
            uses_candidates=True,
            masks=cands,
            candidate_span=SPAN,
        )
        _, fc_ri, _ = self._run(
            st, "decode", inputs=consumer, uses_candidates=True, masks=f_cands
        )
        self.assertTrue(torch.equal(sorted_rows(c_ri), sorted_rows(fc_ri)))

    def test_candidate_span_predicate_and_config(self):
        """The span comes from the model config only with a candidate source layer, and
        the decode predicate must follow the batch maximum eagerly and the captured
        variant inside a graph."""
        import sglang.srt.model_executor.runner_utils.capture_mode as cm
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_candidate_skip_span,
            low_ratio_decode_rows_fit_candidate_span,
        )

        cfg = SimpleNamespace(
            candidate_source_layer_id=20,
            candidate_topk_blocks=TOPK_BLOCKS,
            candidate_block_size=BLOCK_SIZE,
        )
        self.assertEqual(low_ratio_candidate_skip_span(cfg), SPAN)
        self.assertIsNone(
            low_ratio_candidate_skip_span(
                SimpleNamespace(
                    candidate_source_layer_id=-1,
                    candidate_topk_blocks=TOPK_BLOCKS,
                    candidate_block_size=BLOCK_SIZE,
                )
            )
        )
        self.assertIsNone(low_ratio_candidate_skip_span(SimpleNamespace()))

        backend = SimpleNamespace(low_ratio_candidate_span=SPAN)
        off = SimpleNamespace(low_ratio_candidate_span=None)
        fits = self._decode_batch([1, 300, SPAN])
        self.assertTrue(low_ratio_decode_rows_fit_candidate_span(backend, fits))
        self.assertFalse(low_ratio_decode_rows_fit_candidate_span(off, fits))
        self.assertFalse(low_ratio_decode_rows_fit_candidate_span(backend, None))
        self.assertFalse(
            low_ratio_decode_rows_fit_candidate_span(
                backend, self._decode_batch([1, SPAN + 1])
            )
        )
        self.assertFalse(
            low_ratio_decode_rows_fit_candidate_span(
                backend, self._decode_batch([1, 300], decode=False)
            )
        )
        self.assertFalse(
            low_ratio_decode_rows_fit_candidate_span(
                backend,
                SimpleNamespace(forward_mode=fits.forward_mode, seq_lens_cpu=None),
            )
        )
        # Capture batches carry the fill length; only the variant may decide.
        with mock.patch.object(cm, "get_is_capture_mode", return_value=True):
            for variant, expect in (
                ("candidate_all", True),
                ("candidate_c2_all", True),
                ("candidate_unfiltered", True),
                ("candidate_filtered", False),
                (None, False),
            ):
                with mock.patch.object(
                    cm, "get_capture_dsa_variant", return_value=variant
                ):
                    self.assertEqual(
                        low_ratio_decode_rows_fit_candidate_span(backend, fits),
                        expect,
                        variant,
                    )


@unittest.skipUnless(is_hip(), "FlyDSL fp4 indexer kernels are ROCm only")
class TestLowRatioIndexerIdentitySkip(_LowRatioBackendCase):
    """Identity requests: a request whose visible compressed context fits index_topk
    selects every position on every row, so the backend writes that selection
    without scoring. The skipped path must produce exactly the indices (page and
    raw) and candidate masks of the scored path, for whole-batch skips, mixed
    batches of short and long requests, and decode rows."""

    def _identity_row_mask(self, st, seq_lens, extend_lens):
        """Rows of the requests the skip writes without scores."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            is_identity_request,
        )

        mask = torch.zeros(st.pos.numel(), dtype=torch.bool, device=st.dev)
        row = 0
        for s, e in zip(seq_lens, extend_lens):
            mask[row : row + e] = is_identity_request(s, st.ratio, TOPK)
            row += e
        return mask

    def _assert_identity_rows(self, st, ri, pi, n_rows, msg):
        """Identity rows hold 0..lens-1 then -1, with slots resolved through the
        request's page table."""
        j = torch.arange(TOPK, device=st.dev)
        reach = j[None, :] < st.lens[:n_rows, None]
        self.assertTrue(
            torch.equal(
                ri[:n_rows], torch.where(reach, j[None, :], -1).to(torch.int32)
            ),
            msg,
        )
        self.assertTrue(torch.equal(pi[:n_rows] == -1, ~reach), msg)

    # Prefill --------------------------------------------------------------

    def test_prefill_all_identity_batch(self):
        """A batch where every request fits the top-k must produce the scored path's
        rows without running a kernel."""
        for ratio in (1, 2):
            seq_lens = [300, 45, 512 * ratio, 1, 2 * ratio + 1]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=seq_lens)
            full = self._run(st, "extend", skip=False)
            fast = self._run(st, "extend", skip=True)
            self._assert_selection(fast, full, f"{ratio=} plain")
            self._assert_identity_rows(
                st, fast[1], fast[0], st.pos.numel(), f"{ratio=}"
            )

            # Candidate masks are published per request with a visible position
            # (only ratio-1 layers use candidates in the model, where every
            # request has one); keep such requests out of the mask cases.
            seq_lens = [300, 45, 512 * ratio, 2 * ratio + 1]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=seq_lens)
            full = self._run(st, "extend", skip=False)
            full_src = self._run(st, "extend", skip=False, candidate_source=True)
            fast_src = self._run(st, "extend", skip=True, candidate_source=True)
            self._assert_selection(fast_src, full_src, f"{ratio=} source")
            for skip_masks in (full_src[2], fast_src[2]):
                full_c = self._run(
                    st, "extend", skip=False, uses_candidates=True, masks=skip_masks
                )
                fast_c = self._run(
                    st, "extend", skip=True, uses_candidates=True, masks=skip_masks
                )
                self._assert_selection(fast_c, full_c, f"{ratio=} consumer")
                self._assert_selection(
                    fast_c, full, f"{ratio=} consumer vs plain", masks=False
                )

    def test_prefill_mixed_short_and_long_requests(self):
        """Identity requests (whole and chunked) are written directly while long
        requests take the scored path, and the published masks must line up."""
        cases = {
            1: ([300, 1500, 45, 700, 512], [300, 1500, 45, 700, 200]),
            2: ([300, 1500, 3, 2100, 1024], [300, 1500, 3, 2100, 300]),
        }
        for ratio, (seq_lens, extend_lens) in cases.items():
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=extend_lens)
            exact = self._identity_row_mask(st, seq_lens, extend_lens)
            self.assertTrue(bool(exact.any()) and not bool(exact.all()))
            full = self._run(st, "extend", skip=False)
            fast = self._run(st, "extend", skip=True)
            self._assert_selection(fast, full, f"{ratio=} plain", exact_rows=exact)

            full_src = self._run(st, "extend", skip=False, candidate_source=True)
            fast_src = self._run(st, "extend", skip=True, candidate_source=True)
            self._assert_selection(
                fast_src, full_src, f"{ratio=} source", exact_rows=exact
            )
            full_c = self._run(
                st, "extend", skip=False, uses_candidates=True, masks=full_src[2]
            )
            fast_c = self._run(
                st, "extend", skip=True, uses_candidates=True, masks=fast_src[2]
            )
            self._assert_selection(
                fast_c, full_c, f"{ratio=} consumer", exact_rows=exact
            )

    def test_prefill_identity_rows_match_torch_oracle_exactly(self):
        """Identity rows inside a mixed batch agree with the torch oracle without any
        tie tolerance."""
        for ratio in (1, 2):
            seq_lens = [300, 45, 1500, 512 * ratio]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=seq_lens)
            exact = self._identity_row_mask(st, seq_lens, seq_lens)
            self.assertEqual(int(exact.sum()), sum(seq_lens) - 1500)
            fast_pi, fast_ri, _ = self._run(st, "extend", skip=True)
            t_pi, t_ri, _ = self._run(st, "torch", skip=False)
            self.assertTrue(torch.equal(fast_ri[exact], t_ri[exact]), f"{ratio=}")
            self.assertTrue(torch.equal(fast_pi[exact], t_pi[exact]), f"{ratio=}")

    def test_prefill_request_groups_are_neutral_with_skip(self):
        """Splitting a mixed batch into request groups must leave the identity rows and
        the scored rows' masks unchanged."""
        import sglang.srt.layers.attention.dsv4.low_ratio_backend_hip as hip

        for ratio in (1, 2):
            seq_lens = [300, 1500, 45, 700]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=seq_lens)
            exact = self._identity_row_mask(st, seq_lens, seq_lens)
            whole = self._run(st, "extend", skip=True, candidate_source=True)
            with mock.patch.object(hip, "logits_rows_per_chunk", return_value=1):
                split = self._run(st, "extend", skip=True, candidate_source=True)
            self._assert_selection(split, whole, f"{ratio=}", exact_rows=exact)

    def test_skip_disabled_by_small_candidate_topk(self):
        """A candidate top-k too small for every reachable block of a 512-position row
        makes the mask score-dependent, so the skip must stay off."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_identity_skip_enabled,
        )

        self.assertTrue(
            low_ratio_identity_skip_enabled(
                index_topk=512, candidate_topk_blocks=2048, candidate_block_size=8
            )
        )
        self.assertTrue(
            low_ratio_identity_skip_enabled(
                index_topk=512, candidate_topk_blocks=64, candidate_block_size=8
            )
        )
        self.assertFalse(
            low_ratio_identity_skip_enabled(
                index_topk=512, candidate_topk_blocks=2, candidate_block_size=32
            )
        )

    # Decode ---------------------------------------------------------------

    def test_decode_identity_batch_matches_scored_path(self):
        """A decode batch inside the top-k must skip to the transform's sequential
        branch (0..len-1 in order, then -1) and match the scored path exactly."""
        for ratio in (1, 2):
            seq_lens = [1, 37, 300, 512 * ratio, 512 * ratio + ratio - 1]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=[1] * len(seq_lens))
            full = self._run(st, "decode", skip=False)
            fast = self._run(st, "decode", skip=True)
            self._assert_selection(fast, full, f"{ratio=}")
            lens = torch.tensor([max(1, s // ratio) for s in seq_lens], device=st.dev)
            j = torch.arange(TOPK, device=st.dev)
            reach = j[None, :] < lens[:, None]
            self.assertTrue(
                torch.equal(fast[1], torch.where(reach, j[None, :], -1).to(torch.int32))
            )

    def test_decode_long_row_keeps_scored_path(self):
        """One decode row past the top-k must keep the whole batch on the scored path."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_decode_rows_are_identity,
        )

        for ratio in (1, 2):
            seq_lens = [1, 300, 512 * ratio + ratio]
            st = self._setup(ratio, seq_lens=seq_lens, extend_lens=[1] * len(seq_lens))
            backend = SimpleNamespace(low_ratio_identity_skip=True, index_topk=TOPK)
            self.assertFalse(
                low_ratio_decode_rows_are_identity(backend, st.forward_batch, ratio)
            )
            self.assertTrue(
                low_ratio_decode_rows_are_identity(
                    backend, self._decode_batch([1, 300, 512 * ratio]), ratio
                )
            )
            full = self._run(st, "decode", skip=False)
            fast = self._run(st, "decode", skip=True)
            exact = torch.tensor([True, True, False], device=st.dev)
            self._assert_selection(fast, full, f"{ratio=}", exact_rows=exact)

    def test_decode_predicate_follows_capture_variant(self):
        """Inside a captured graph only the variant may decide the skip; a disabled
        backend and target-verify rows never skip."""
        import sglang.srt.model_executor.runner_utils.capture_mode as cm
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_decode_rows_are_identity,
        )

        backend = SimpleNamespace(low_ratio_identity_skip=True, index_topk=TOPK)
        # Capture batches carry the fill length; only the variant may decide.
        fb = self._decode_batch([1, 1])
        with mock.patch.object(cm, "get_is_capture_mode", return_value=True):
            for variant, ratio, expect in (
                ("candidate_all", 1, True),
                ("candidate_all", 2, True),
                ("candidate_c2_all", 2, True),
                ("candidate_c2_all", 1, False),
                ("candidate_unfiltered", 1, False),
                ("candidate_filtered", 1, False),
                (None, 1, False),
            ):
                with mock.patch.object(cm, "_capture_dsa_variant", variant):
                    self.assertEqual(
                        low_ratio_decode_rows_are_identity(backend, fb, ratio),
                        expect,
                        (variant, ratio),
                    )
        self.assertFalse(
            low_ratio_decode_rows_are_identity(
                SimpleNamespace(low_ratio_identity_skip=False, index_topk=TOPK), fb, 1
            )
        )
        self.assertFalse(
            low_ratio_decode_rows_are_identity(
                backend, self._decode_batch([1, 1], decode=False), 1
            )
        )


# -- the head weights: split-K Triton GEMV route --------------------------------


def _served_chain(x, w, scale):
    from aiter.tuned_gemm import tgemm

    return (tgemm.mm(x, w, None, otype=x.dtype) * scale).contiguous()


def _ordered_bits(t):
    """bf16 -> integers ordered like the values, so differences count ulps."""
    i = t.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _ulp_distance(a, b):
    return (_ordered_bits(a) - _ordered_bits(b)).abs()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "split-K MFMA GEMV pair is gfx95 only"
)
class TestIndexerHeadWeightsHip(CustomTestCase):
    """The ROCm decode route of the head weights (split-K Triton GEMV plus a
    fixed-order reduce that applies `head_weight_scale`) must be
    `bf16(bf16(fp32 split-K sum) * scale)` bit for bit, give every row the same
    weights whether alone or inside a 16-row batch, replay from a graph, and stand
    within one bf16 ulp of the exactly rounded product on every element. Against
    the served chain (aiter's tuned GEMM, then the aten bf16 multiply) the two
    differ only where the fp32 sums round to different bf16 values."""

    def setUp(self):
        torch.manual_seed(20260909)
        self.w = (torch.randn(N_HEADS, HIDDEN, device="cuda") * 0.02).bfloat16()

    def _x(self, m):
        # Hidden states of varying magnitude per row.
        return (
            torch.randn(m, HIDDEN, device="cuda")
            * (0.5 + 3 * torch.rand(m, 1, device="cuda"))
        ).bfloat16()

    def test_max_tokens(self):
        """The route serves exactly the router GEMV's shapes (n a multiple of 8, k a
        multiple of 512, bf16) and declines everything else."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights_max_tokens,
        )
        from sglang.kernels.ops.moe.rocm_router_gate import ROCM_ROUTER_MAX_TOKENS

        self.assertEqual(
            rocm_indexer_head_weights_max_tokens(N_HEADS, HIDDEN, torch.bfloat16),
            ROCM_ROUTER_MAX_TOKENS,
        )
        self.assertEqual(
            rocm_indexer_head_weights_max_tokens(64, HIDDEN, torch.bfloat16),
            ROCM_ROUTER_MAX_TOKENS,
        )
        for n, k, dtype in (
            (N_HEADS, HIDDEN, torch.float16),
            (N_HEADS, HIDDEN + 128, torch.bfloat16),
            (24, HIDDEN, torch.bfloat16),
            (N_HEADS, 512 * 33, torch.bfloat16),
        ):
            self.assertEqual(rocm_indexer_head_weights_max_tokens(n, k, dtype), -1)

    def test_bitwise_definition_and_batch_invariance(self):
        """The route must equal the fixed-order split-K definition bit for bit and give
        a row the same weights alone as inside the batch."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights,
        )
        from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gemv_split_k

        for m in (1, 2, 5, 8, 13, 16):
            with self.subTest(m=m):
                x = self._x(m)
                got = rocm_indexer_head_weights(x, self.w, SCALE)
                self.assertEqual(got.shape, (m, N_HEADS))
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(got.is_contiguous())
                partials = rocm_router_gemv_split_k(x, self.w)
                acc = partials[0].clone()
                for s in range(1, partials.shape[0]):
                    acc += partials[s]
                expect = (acc.bfloat16().float() * SCALE).bfloat16()
                self.assertTrue(torch.equal(got, expect), "fixed-order definition")
                for r in range(m):
                    self.assertTrue(
                        torch.equal(
                            rocm_indexer_head_weights(x[r : r + 1], self.w, SCALE)[0],
                            got[r],
                        ),
                        f"row {r} depends on the batch",
                    )
                # Repeatable.
                self.assertTrue(
                    torch.equal(rocm_indexer_head_weights(x, self.w, SCALE), got)
                )

    def test_within_one_ulp_of_exact_and_against_served_chain(self):
        """Every element within one bf16 ulp of the exactly rounded product, and only a
        small fraction differing from the served chain."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights,
        )

        total = differ = 0
        for trial in range(120):
            m = (1, 2, 4, 8, 12, 16)[trial % 6]
            x = self._x(m)
            exact = (x.double() @ self.w.double().T * SCALE).bfloat16()
            got = rocm_indexer_head_weights(x, self.w, SCALE)
            served = _served_chain(x, self.w, SCALE)
            self.assertLessEqual(
                int(_ulp_distance(got, exact).max()),
                1,
                f"{m=}: more than 1 ulp from exact",
            )
            total += got.numel()
            differ += int((got != served).sum())
        # the chains disagree only where their fp32 sums straddle a bf16 rounding boundary
        self.assertLess(
            differ / total,
            2e-3,
            f"{differ} of {total} elements differ from the served chain",
        )

    def test_graph_replay(self):
        """A captured route must replay on new hidden states with no host sync."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights,
        )

        x = self._x(8)
        rocm_indexer_head_weights(x, self.w, SCALE)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = rocm_indexer_head_weights(x, self.w, SCALE)
        for _ in range(3):
            x.copy_(self._x(8))
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(out, rocm_indexer_head_weights(x, self.w, SCALE))
            )

    def test_indexer_inputs_route_and_fallback(self):
        """`_indexer_inputs` must take the GEMV pair only for decode rows of a real-
        shaped indexer and the linear otherwise."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            _indexer_head_weights,
        )

        calls = []

        class Indexer:
            weights_proj_hip_max_tokens = 16
            head_weight_scale = SCALE

            def __init__(self, w):
                self.weights_proj = SimpleNamespace(weight=w)

            def head_weights(self, x):
                calls.append(x.shape[0])
                return _served_chain(x, self.weights_proj.weight, SCALE)

        indexer = Indexer(self.w)
        x = self._x(16)
        got = _indexer_head_weights(indexer, x)
        self.assertEqual(calls, [])
        exact = (x.double() @ self.w.double().T * SCALE).bfloat16()
        self.assertLessEqual(int(_ulp_distance(got, exact).max()), 1)
        wide = self._x(17)
        self.assertTrue(
            torch.equal(
                _indexer_head_weights(indexer, wide), indexer.head_weights(wide)
            )
        )
        self.assertEqual(calls, [17, 17])
        # A non-contiguous row view and an indexer below the route's range fall back.
        strided = self._x(4)[:, ::2]
        self.assertEqual(strided.stride(1), 2)
        indexer_wide = Indexer(self.w[:, : HIDDEN // 2].contiguous())
        _indexer_head_weights(indexer_wide, strided)
        self.assertEqual(calls[-1], 4)
        stub = SimpleNamespace(
            head_weights=lambda x: _served_chain(x, self.w, SCALE),
            weights_proj_hip_max_tokens=-1,
        )
        self.assertTrue(
            torch.equal(_indexer_head_weights(stub, x), stub.head_weights(x))
        )


if __name__ == "__main__":
    unittest.main()
