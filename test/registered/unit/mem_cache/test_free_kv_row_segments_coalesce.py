"""Unit test for _coalesce_contiguous_segments (mem_cache/common.py).

Under a DCP-widened allocator page (page_size * dcp_size), the hybrid radix
cache frees a request's tail as several adjacent pieces (the unaligned key
tail and the KDA-checkpoint-truncated tail). Adjacent pieces share a widened
page, which the page-once free contract rejects, while their union is one
page-aligned segment. The helper merges abutting segments before the free.
"""

import unittest

import torch

from sglang.srt.mem_cache.common import _coalesce_contiguous_segments
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _seg(start: int, n: int, start_pos: int):
    return (torch.arange(start, start + n, dtype=torch.int64), start_pos)


class TestCoalesceContiguousSegments(unittest.TestCase):
    def test_abutting_segments_merge(self):
        # positions [0, 8) and [8, 12) on the same row -> one segment [0, 12)
        merged = _coalesce_contiguous_segments([_seg(100, 8, 0), _seg(108, 4, 8)])
        self.assertEqual(len(merged), 1)
        idx, start = merged[0]
        self.assertEqual(start, 0)
        self.assertEqual(idx.tolist(), list(range(100, 112)))

    def test_gap_keeps_segments_apart(self):
        merged = _coalesce_contiguous_segments([_seg(100, 8, 0), _seg(120, 4, 16)])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][1], 0)
        self.assertEqual(merged[1][1], 16)

    def test_empty_segments_are_dropped(self):
        empty = (torch.empty(0, dtype=torch.int64), 8)
        merged = _coalesce_contiguous_segments(
            [_seg(100, 8, 0), empty, _seg(108, 4, 8)]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][0].numel(), 12)

    def test_three_way_chain(self):
        merged = _coalesce_contiguous_segments(
            [_seg(0, 3, 65873), _seg(3, 1, 65876), _seg(4, 4, 65877)]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][1], 65873)
        self.assertEqual(merged[0][0].numel(), 8)


if __name__ == "__main__":
    unittest.main()
