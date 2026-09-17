"""FlashMLA stores must address cache bytes beyond signed-int32 boundaries."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE_SIZE = 256
PAGE_BYTES = ((PAGE_SIZE * 584 + 575) // 576) * 576
POISON = 0xA5
# A broken int32 store stays inside this owned allocation, including the BF16
# view's -2**31-element address. No invalid GPU access is needed for the test.
GUARD_BYTES = (1 << 32) + 4096


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 FlashMLA cache")
class TestFlashmlaStoreLargeOffsets(CustomTestCase):
    def _check_page(self, cache, page, slots, value):
        expected = torch.full((PAGE_BYTES,), POISON, dtype=torch.uint8)
        for slot in slots:
            start = slot * 576
            # 448 -> e4m3fn 448, scale 1; 224 -> e4m3fn 448, scale 1/2.
            expected[start : start + 448] = 0x7E
            expected[start + 448 : start + 576] = torch.full(
                (64,), value, dtype=torch.bfloat16
            ).view(torch.uint8)
            scale = PAGE_SIZE * 576 + slot * 8
            expected[scale : scale + 7] = 127 if value == 448 else 126
        self.assertTrue(
            torch.equal(cache[page].cpu(), expected),
            "Payload, RoPE, scales, or unwritten page padding changed",
        )

    def test_public_store_large_offsets_and_graph_replay(self):
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache

        for boundary in (2**31, 2**32):
            for index_dtype in (torch.int32, torch.int64):
                with self.subTest(boundary=boundary, index_dtype=index_dtype):
                    page = (boundary + PAGE_BYTES - 1) // PAGE_BYTES
                    storage = torch.full(
                        (GUARD_BYTES + (page + 2) * PAGE_BYTES,),
                        POISON,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    cache = storage[GUARD_BYTES:].view(page + 2, PAGE_BYTES)
                    slots = [0, 3, PAGE_SIZE - 1]
                    indices = torch.tensor(
                        [page * PAGE_SIZE + slot for slot in slots],
                        dtype=index_dtype,
                        device="cuda",
                    )
                    data = torch.full(
                        (len(slots), 512), 448, dtype=torch.bfloat16, device="cuda"
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        fused_store_cache(
                            data, cache, indices, page_size=PAGE_SIZE, type="flashmla"
                        )
                    stream.synchronize()
                    self._check_page(cache, page, slots, 448)
                    self.assertTrue(torch.all(data == 448).item())

                    with torch.cuda.stream(stream):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            fused_store_cache(
                                data,
                                cache,
                                indices,
                                page_size=PAGE_SIZE,
                                type="flashmla",
                            )
                    stream.synchronize()
                    slots = [1, 4, PAGE_SIZE - 2]
                    indices.copy_(
                        torch.tensor(
                            [page * PAGE_SIZE + slot for slot in slots],
                            dtype=index_dtype,
                            device="cuda",
                        )
                    )
                    data.fill_(224)
                    storage.fill_(POISON)
                    graph.replay()
                    torch.cuda.synchronize()
                    self._check_page(cache, page, slots, 224)
                    self.assertTrue(torch.all(data == 224).item())
                    # Candidate must preserve the allocation before the cache
                    # and every earlier page. This also detects wrapped writes.
                    self.assertTrue(torch.all(storage[:GUARD_BYTES] == POISON).item())
                    self.assertTrue(torch.all(cache[:page] == POISON).item())
                    del graph, data, indices, cache, storage


if __name__ == "__main__":
    unittest.main()
