"""Under attention DP every rank must replay the same context-length graph variant, so the DP-wide maximum wins over the rank's own lengths."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.graph_variants import (
    DSA_DENSE,
    DSA_SPARSE,
    DsaGraphVariants,
    Dsv41CandidateGraphVariants,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _batch(local, dp_max):
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor(local), seq_lens=None, dp_max_seq_len=dp_max
    )


class TestDpMaxSeqLenPicksTheVariant(CustomTestCase):
    def test_dsa(self):
        policy = DsaGraphVariants(index_topk=2048)
        for dp_max, local, expected in (
            (4096, [1024], DSA_SPARSE),
            (1024, [4096], DSA_DENSE),
            (None, [4096], DSA_SPARSE),
        ):
            with self.subTest(dp_max_seq_len=dp_max, local=local):
                self.assertEqual(policy.select(_batch(local, dp_max)), expected)

    def test_dsv41_candidate(self):
        policy = Dsv41CandidateGraphVariants(
            graph_limits=(("candidate_unfiltered", 16384),),
            capture_labels=("candidate_unfiltered", "candidate_filtered"),
            verify_extra_tokens=6,
        )
        for dp_max, local, expected in (
            (16379, [4096], "candidate_filtered"),
            (4096, [16379], "candidate_unfiltered"),
        ):
            with self.subTest(dp_max_seq_len=dp_max, local=local):
                self.assertEqual(policy.select(_batch(local, dp_max)), expected)


if __name__ == "__main__":
    unittest.main()
