# [Hybrid cache] Coalesce abutting freed segments before the page-once free under DCP

## Motivation
With `--dcp-size > 1` the allocator page is widened to `page_size * dcp_size`. The unified radix cache frees a finished request's tail as several adjacent pieces (the unaligned key tail and the KDA-checkpoint-truncated tail on Kimi-K3's hybrid KDA + MLA cache). Two adjacent pieces can share one widened page, and `free_segments` rejects that:

```
AssertionError: segment at 65876 shares a page with the one ending at 65876
  unified_radix_cache.cache_finished_req -> free_kv_row -> free_kv_row_segments -> allocator.free_segments
```

Reproduces on current main with Kimi-K3 on 8x MI350X, `--attention-backend aiter --dcp-size 8 --dcp-comm-backend a2a`, on the first multi-turn request completion (InferenceX AgentX trace replay), with and without DSPARK.

## Modifications
`python/sglang/srt/mem_cache/common.py`: `_coalesce_contiguous_segments` merges abutting segments (same row, contiguous positions) before `free_kv_row_segments` hands them to the allocator; segments with a gap are left alone and empty pieces are dropped. CPU unit test `test/registered/unit/mem_cache/test_free_kv_row_segments_coalesce.py`.

## Accuracy / performance
8x MI350X Kimi-K3 DCP8: AgentX replay completes; GSM8K-200 0.995 (no spec) / 0.985 (DSPARK block 3). No change on non-DCP paths (segments are page-aligned there and never abut across a page).

## Checklist
- [x] Format your code according to the Code Formatting with Pre-Commit.
- [x] Add unit tests as outlined in the Running Unit Tests.
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
