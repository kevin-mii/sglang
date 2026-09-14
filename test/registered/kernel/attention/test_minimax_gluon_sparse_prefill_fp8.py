"""Gluon sparse prefill (MiniMax-M3, ROCm gfx950) on bf16 and fp8 K/V pools.

The Gluon paged-attention prefill must match a pure-torch block-sparse
softmax reference for fresh prompts, batched prompts,
prefix-extend with non-block-aligned chunk boundaries, and a second chunked
prefill chunk, on bf16 pools and on fp8 pools with unit and non-unit
per-tensor K/V scales. Before the fp8 support the dispatch silently fell back
to Triton on fp8 pools, which is the configuration the fp8 indexer cache
targets.
"""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

if not envs.SGLANG_USE_AITER.is_set():
    envs.SGLANG_USE_AITER.set(True)

try:
    from sglang.srt.layers.attention.minimax_sparse_ops.gluon_prefill import (
        can_use_gluon_prefill,
        gluon_sparse_prefill,
    )

    _HAS_DEPS = True
except Exception:  # pragma: no cover - aiter / triton missing
    _HAS_DEPS = False


def _gfx950() -> bool:
    if not (is_hip() and torch.cuda.is_available()):
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


HQ, HKV, D, BLK, TOPK = 16, 1, 128, 128, 18  # TP4 rank: 16 q heads, 1 kv head


@unittest.skipUnless(_HAS_DEPS and _gfx950(), "gfx950 + aiter Gluon path required")
class TestGluonSparsePrefillFp8(CustomTestCase):
    def _run_case(
        self, kv_dtype, seqs, prefixes, block_size_q, k_scale=1.0, v_scale=1.0
    ):
        torch.manual_seed(0)
        dev = "cuda"
        B = len(seqs)
        ext = [s - p for s, p in zip(seqs, prefixes)]
        total_q = sum(ext)
        max_len = max(seqs)
        max_slots = sum(seqs) + 1024
        perm = torch.randperm(max_slots, device=dev)
        req_to_token = torch.zeros((B, max_len), dtype=torch.int32, device=dev)
        off = 0
        for b, s in enumerate(seqs):
            req_to_token[b, :s] = perm[off : off + s].to(torch.int32)
            off += s
        slot_ids = torch.arange(B, dtype=torch.int32, device=dev)
        kf = torch.randn(max_slots, HKV, D, device=dev, dtype=torch.bfloat16)
        vf = torch.randn(max_slots, HKV, D, device=dev, dtype=torch.bfloat16)
        is_fp8 = kv_dtype != torch.bfloat16
        if is_fp8:
            k_cache = (kf / k_scale).to(kv_dtype)
            v_cache = (vf / v_scale).to(kv_dtype)
        else:
            k_cache, v_cache = kf, vf
        q = torch.randn(total_q, HQ, D, device=dev, dtype=torch.bfloat16)
        cu = torch.tensor(
            [0] + torch.cumsum(torch.tensor(ext), 0).tolist(),
            dtype=torch.int32,
            device=dev,
        )
        seq_lens = torch.tensor(seqs, dtype=torch.int32, device=dev)
        prefix_lens = torch.tensor(prefixes, dtype=torch.int32, device=dev)
        # Synthetic causal top-k: ascending block ids up to the query's own
        # block, -1 tail.
        topk = torch.full((HKV, total_q, TOPK), -1, dtype=torch.int32, device=dev)
        for b in range(B):
            for i in range(ext[b]):
                pos = prefixes[b] + i
                sb = pos // BLK
                cand = torch.randperm(sb + 1)[: TOPK - 1].tolist()
                if sb not in cand:
                    cand.append(sb)
                cand = sorted(set(cand))[:TOPK]
                topk[0, int(cu[b]) + i, : len(cand)] = torch.tensor(
                    cand, dtype=torch.int32
                )
        sm_scale = D**-0.5
        self.assertTrue(
            can_use_gluon_prefill(
                q, k_cache, v_cache, None, BLK, seq_lens.cpu(), None, k_scale, v_scale
            ),
            "dispatch must accept this pool dtype / scale combination",
        )
        out = gluon_sparse_prefill(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            topk_idx=topk,
            req_to_token=req_to_token,
            req_pool_indices=slot_ids,
            cu_seqlens=cu,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            seq_lens_cpu=seq_lens.cpu(),
            block_size_k=BLK,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # Gluon vs a pure-torch block-sparse softmax on sampled queries: first
        # rows, rows across a block-size_q boundary, chunk boundaries, request
        # boundaries, last rows.
        sample = sorted(
            {0, 1, 5, 7, 8, 9, 130, 200, total_q - 1, total_q - 2, total_q // 2}
            | ({int(cu[1]) - 1, int(cu[1]), int(cu[1]) + 3} if B > 1 else set())
        )
        for gi in (g for g in sample if 0 <= g < total_q):
            b = int((cu[1:] <= gi).sum())
            pos = prefixes[b] + (gi - int(cu[b]))
            blks = [t for t in topk[0, gi].tolist() if t >= 0 and t * BLK <= pos]
            kpos = torch.cat(
                [torch.arange(t * BLK, min((t + 1) * BLK, pos + 1)) for t in blks]
            )
            slots = req_to_token[b, kpos.to(dev)].long()
            kk = k_cache[slots, 0].float() * (k_scale if is_fp8 else 1.0)
            vv = v_cache[slots, 0].float() * (v_scale if is_fp8 else 1.0)
            pr = torch.softmax((q[gi].float() @ kk.T) * sm_scale, dim=-1)
            torch_ref = pr @ vv
            err = (out[gi].float() - torch_ref).abs().max().item()
            self.assertLess(err, 2e-2, f"query {gi}: gluon vs torch max abs {err}")
            self.assertFalse(torch.isnan(out[gi]).any().item())

    def test_bf16_pool(self):
        self._run_case(torch.bfloat16, [700], [0], 8)
        self._run_case(torch.bfloat16, [5000, 3100], [0, 0], 8)
        self._run_case(torch.bfloat16, [9000, 4200], [7000, 300], 4)
        self._run_case(torch.bfloat16, [20000], [8192], 8)

    def test_fp8_pool_unit_scales(self):
        fp8 = torch.float8_e4m3fn
        self._run_case(fp8, [700], [0], 8)
        self._run_case(fp8, [5000, 3100], [0, 0], 8)
        self._run_case(fp8, [9000, 4200], [7000, 300], 4)
        self._run_case(fp8, [20000], [8192], 8)

    def test_fp8_pool_non_unit_scales(self):
        self._run_case(
            torch.float8_e4m3fn, [6000, 2500], [1000, 0], 8, k_scale=0.5, v_scale=2.0
        )


if __name__ == "__main__":
    unittest.main()
