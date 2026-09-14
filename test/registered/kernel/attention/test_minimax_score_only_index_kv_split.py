"""The KV-block split of the score-only indexer must leave the top-k indices bit-identical."""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx as M

    _HAS_DEPS = True
except Exception:  # pragma: no cover
    _HAS_DEPS = False

H, KH, D = 1, 1, 128  # one TP4 rank of MiniMax-M3: one index head


@unittest.skipUnless(
    _HAS_DEPS and torch.cuda.is_available(), "GPU + Triton indexer required"
)
class TestScoreOnlyIndexKvSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dev = "cuda"
        cls.max_slots = 260_000
        cls.pool_len = 210_000
        cls.k_cache = (torch.randn(cls.max_slots, KH, D, device=cls.dev) * 0.5).to(
            torch.bfloat16
        )
        cls.req_to_token = torch.zeros(
            (4, cls.pool_len), dtype=torch.int32, device=cls.dev
        )

    def _topk(self, kc, lens, exts, target_programs):
        bs = len(lens)
        prefix = [L - e for L, e in zip(lens, exts)]
        cu = torch.tensor(
            [0] + torch.cumsum(torch.tensor(exts), 0).tolist(),
            dtype=torch.int32,
            device=self.dev,
        )
        torch.manual_seed(1)
        q = (torch.randn(int(cu[-1]), H, D, device=self.dev) * 0.3).to(torch.bfloat16)
        old = M._SCORE_ONLY_TARGET_PROGRAMS
        M._SCORE_ONLY_TARGET_PROGRAMS = target_programs
        try:
            _, out = M.flash_prefill_with_topk_index(
                q=q,
                k_cache=kc,
                v_cache=None,
                sink=None,
                req_to_token=self.req_to_token,
                slot_ids=torch.arange(bs, dtype=torch.int32, device=self.dev),
                cu_seqlens=cu,
                seq_lens=torch.tensor(lens, dtype=torch.int32, device=self.dev),
                prefix_lens=torch.tensor(prefix, dtype=torch.int32, device=self.dev),
                max_seqlen_q=max(exts),
                max_seqlen_k=max(lens),
                block_size_q=1,
                block_size_k=128,
                topk=16,
                init_blocks=1,
                local_blocks=2,
                score_type="max",
                disable_index_value=True,
                page_size=1,
            )
        finally:
            M._SCORE_ONLY_TARGET_PROGRAMS = old
        torch.cuda.synchronize()
        return out

    def _check(self, kc, cases):
        for lens, exts in cases:
            # one slot mapping per case, shared by both launches
            for b, L in enumerate(lens):
                self.req_to_token[b, :L] = torch.randperm(
                    self.max_slots, device=self.dev
                )[:L].to(torch.int32)
            ref = self._topk(kc, lens, exts, 1)
            out = self._topk(kc, lens, exts, 2048)
            self.assertTrue(
                torch.equal(ref, out), f"top-k differs for lens={lens} exts={exts}"
            )

    def test_bf16_index_cache_topk_is_identical(self):
        """A (row, block) written by two programs or by none changes the top-k."""
        self._check(
            self.k_cache,
            [
                ([198000], [10]),
                ([198000], [512]),
                ([198000, 70000], [1536, 300]),
                ([30000], [8192]),
                ([8192], [8192]),
                ([6000, 5000, 4000], [64, 5000, 1]),
            ],
        )

    def test_fp8_index_cache_topk_is_identical(self):
        """The fp8 index cache takes the per-page gather path; same invariant."""
        kc = self.k_cache.to(torch.float8_e4m3fn)
        self._check(
            kc, [([198000], [10]), ([198000], [1536]), ([120000, 200000], [2048, 4096])]
        )


if __name__ == "__main__":
    unittest.main()
