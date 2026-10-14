import unittest

import torch

from sglang.kernels.ops.attention.mla_decode_epilogue import mla_decode_epilogue
from sglang.kernels.ops.attention.mla_split_reduce import mla_split_reduce
from sglang.kernels.ops.attention.mla_v_up_gate import mla_v_up_gate
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: 12 heads padded to the asm kernel's 16, latent 512, V 128.
H, HP, L, V = 12, 16, 512, 128


def _make(splits, max_q, uncovered=(), seed=0):
    """aiter persistent-decode partials: request r has splits[r] splits (0, or listed in
    uncovered: stage 1 finalized its rows in o); a tile covers a request's max_q rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    indptr, final_map, pmap, base = [0], [], [], 0
    for r, n in enumerate(splits):
        if n == 0 or r in uncovered:
            continue
        for _ in range(n):
            pmap.append(base)
            base += max_q
        indptr.append(len(pmap))
        final_map.append([r * max_q, (r + 1) * max_q])
    rows = len(splits) * max_q
    i32 = dict(device="cuda", dtype=torch.int32)
    return dict(
        partial=torch.randn(max(base, 1), 1, HP, L, generator=g, device="cuda"),
        lse=torch.randn(max(base, 1), 1, HP, 1, generator=g, device="cuda") * 3,
        indptr=torch.tensor(indptr, **i32),
        fmap=(torch.tensor(final_map, **i32) if final_map else torch.zeros(0, 2, **i32)),
        pmap=torch.tensor(pmap or [0], **i32),
        o=torch.randn(rows, HP, L, generator=g, device="cuda").bfloat16(),
        w=(torch.randn(H, L, V, generator=g, device="cuda") * 0.05).bfloat16(),
        gate=torch.randn(rows, H * V, generator=g, device="cuda").bfloat16(),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMLADecodeEpilogue(unittest.TestCase):
    @torch.inference_mode()
    def test_matches_split_reduce_then_v_up_gate(self):
        """One launch must reproduce the backend's split combine followed by the
        W_vc + gate kernel within one bf16 ulp: single-query and 4-row verify
        tiles, rows stage 1 finalized, and rows with more splits than the
        program's tile (combined in chunks)."""
        cases = (
            ([37, 0, 128, 5], 1, ()),
            ([128, 3], 4, ()),
            ([64, 128, 7, 1], 4, (1,)),
            ([128] * 16, 4, ()),
        )
        for splits, max_q, uncovered in cases:
            t = _make(splits, max_q, uncovered, seed=len(splits) + max_q)
            for gate in (t["gate"], None):
                o_ref = t["o"].clone()
                mla_split_reduce(
                    t["partial"], t["lse"], t["indptr"], t["fmap"], t["pmap"], max_q, o_ref, H
                )
                ref = mla_v_up_gate(o_ref[:, :H, :], t["w"], gate).float()
                out = mla_decode_epilogue(
                    t["partial"], t["lse"], t["indptr"], t["fmap"], t["pmap"], t["o"], t["w"], gate
                ).float()
                beyond_ulp = ((out - ref).abs() > ref.abs() * 2**-7 + 1e-3).sum().item()
                with self.subTest(splits=splits[:4], max_q=max_q, gated=gate is not None):
                    self.assertLessEqual(beyond_ulp, max(out.numel() // 1000, 1))


if __name__ == "__main__":
    unittest.main()
