"""Triton backend extends over a long cached prefix (gfx950 long-prefix route).

The eager case runs the full backend against the HF-style reference with a
prefix long enough to take the split-prefix sweep. The ASM case feeds an fp8
cache with non-unit scales through the large-chunk branch and checks it
against an fp32 reference built from the quantized cache.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    run_dense_fixture_eager,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 long-prefix route"
)
class TestTritonLongPrefixExtend(CustomTestCase):
    def test_split_prefix_route_matches_reference(self):
        # 16 q heads over one KV head, as on MiniMax-M3 dense layers per TP4 rank.
        case = DenseAttentionCase(
            name="long_prefix_mqa_extend",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=16,
            num_kv_heads=1,
            page_size=1,
            prefix_lens=(9000, 12000),
            extend_lens=(3, 200),
        )
        fixture = build_dense_attention_fixture(
            self,
            case,
            head_dim=128,
            hidden_size=256,
            max_context_len=12288,
            dtype=torch.bfloat16,
        )
        backend = fixture.backend
        with patch.object(
            backend,
            "extend_attention_fwd_long_prefix",
            wraps=backend.extend_attention_fwd_long_prefix,
        ) as split:
            actual = run_dense_fixture_eager(fixture)
        self.assertEqual(split.call_count, 1)
        torch.testing.assert_close(
            actual,
            expected_dense_fixture_output(fixture),
            atol=DENSE_ATOL,
            rtol=DENSE_RTOL,
        )


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 ASM fp8 prefill"
)
class TestTritonLongPrefixAsmPrefill(CustomTestCase):
    def test_large_chunk_reads_quantized_cache(self):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        try:
            from aiter import flash_attn_varlen_fp8_pertensor_func
        except ImportError:
            self.skipTest("aiter is not available")
        from sglang.kernels.ops.attention.extend_attention import (
            build_unified_kv_indices,
        )

        torch.manual_seed(7)
        device = "cuda"
        fp8 = torch.float8_e4m3fn
        h_q, d = 16, 128
        prefix_lens, extend_lens = [9000, 20000], [2048, 300]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        slots = sum(seq_lens) + 64
        k_cache = torch.randn(slots, 1, d, device=device).to(fp8)
        v_cache = torch.randn(slots, 1, d, device=device).to(fp8)
        k_scale, v_scale = 0.25, 1.5
        perm = torch.randperm(slots, device=device)
        prefix_idx, chunk_loc, off = [], [], 0
        for p, e in zip(prefix_lens, extend_lens):
            prefix_idx.append(perm[off : off + p])
            chunk_loc.append(perm[off + p : off + p + e])
            off += p + e
        kv_indices = torch.cat(prefix_idx)
        out_cache_loc = torch.cat(chunk_loc)
        kv_indptr = torch.tensor(
            [0, *torch.tensor(prefix_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        # int64, as the backend keeps it outside dllm.
        qo_indptr = torch.tensor(
            [0, *torch.tensor(extend_lens).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        )
        n = sum(extend_lens)
        q = torch.randn(n, h_q * d, device=device, dtype=torch.bfloat16)
        # Raw K/V deliberately differ from the cache: the route must read the
        # quantized cache, which already holds this chunk.
        k = torch.zeros(n, 1, d, device=device, dtype=torch.bfloat16)
        v = torch.zeros_like(k)

        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.long_prefix_asm_prefill = flash_attn_varlen_fp8_pertensor_func
        backend.build_unified_kv_indices = build_unified_kv_indices
        backend.page_size = 1
        backend.unit_descale = torch.ones(1, dtype=torch.float32, device=device)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: k_cache, get_value_buffer=lambda _: v_cache
        )
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=qo_indptr,
            max_extend_len=max(extend_lens),
            out_cache_loc_full_physical=None,
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=h_q,
            tp_k_head_num=1,
            qk_head_dim=d,
            v_head_dim=d,
            scaling=d**-0.5,
            k_scale=torch.tensor([k_scale], device=device),
            v_scale=torch.tensor([v_scale], device=device),
            k_scale_float=k_scale,
            v_scale_float=v_scale,
        )
        batch = SimpleNamespace(
            batch_size=len(seq_lens),
            out_cache_loc=out_cache_loc,
            extend_start_loc=qo_indptr[:-1].to(torch.int32),
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=extend_lens,
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        )
        o = torch.empty_like(q)
        with patch.object(
            backend,
            "long_prefix_asm_prefill",
            wraps=backend.long_prefix_asm_prefill,
        ) as asm:
            backend._forward_extend_long_prefix(
                q, k, v, o, layer, batch, kv_indptr, kv_indices
            )
        self.assertEqual(asm.call_count, 1)

        o = o.view(n, h_q, d)
        q = q.view(n, h_q, d)
        for i, (p, e) in enumerate(zip(prefix_lens, extend_lens)):
            slots_i = torch.cat([prefix_idx[i], chunk_loc[i]])
            keys = k_cache[slots_i, 0].float() * k_scale
            values = v_cache[slots_i, 0].float() * v_scale
            rows = sorted({0, e // 2, e - 1})
            q0 = int(qo_indptr[i])
            for r in rows:
                query = q[q0 + r].to(fp8).float()  # Q is cast to fp8 at unit scale.
                scores = (query @ keys[: p + r + 1].T) * layer.scaling
                ref = scores.softmax(-1) @ values[: p + r + 1]
                diff = (o[q0 + r].float() - ref).abs().max().item()
                self.assertLess(diff, 2e-2 * v_scale, f"req {i} row {r}: {diff}")


if __name__ == "__main__":
    unittest.main()
