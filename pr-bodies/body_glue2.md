## Motivation

Two pieces of a DeepSeek-V4.1 DSpark verify step on ROCm move data or launch collectives that one call can cover:

- The target logits cross TP ranks with `all_gather(dim=-1)`: an all-gather on dim 0 (aiter's custom all-gather), then a `movedim` relayout copy of the whole `[rows, vocab]` result, then `_copy_logits_to_buffer` copies it again into the logits buffer. At 8 requests (48 rows) the gather and the two copies take about 94 µs a step. aiter's all-gather has a last-dim kernel (`allgather_lastdim`) that writes the `[rows, world * n]` layout itself.
- The accept broadcasts `correct_len`, `bonus` and `cap_trim_lens` from rank 0 (`SGLANG_SPEC_TP_SYNC`), three RCCL launches of about 11 µs each.

## Modifications

- **`GroupCoordinator.all_gather_last_dim`**: aiter's last-dim kernel straight into `out` (or a new tensor), or None when aiter's custom all-gather does not apply (or a rank's row chunk is not a 16-byte multiple). `all_gather(dim=last)` takes it, so on ROCm no gather pays the relayout copy. The aiter dispatch (graph capture, warmup, unregistered inputs) moves from `_all_gather_into_tensor` into `_aiter_all_gather(output, input, dim)`, unchanged.
- **The logits processor gathers into its logits buffer** on ROCm when the buffer holds the gathered shape and dtype. `_copy_logits_to_buffer` then copies nothing (`copy_` onto itself is a no-op).
- **`SpecTpSync.sync_many`** broadcasts several tensors in one NCCL group (`GroupCoordinator.broadcast_many`). The accept's three syncs use it: one launch, and every rank still takes rank 0's values.

## Accuracy Tests

MI350X, this branch on main 5e79333c95:

- `test/registered/kernels/ops/communication/test_amd_all_gather_last_dim.py` (new; TP2 and TP4): 13 passed each. The last-dim gather against torch's all-gather bit for bit, float32 and bfloat16, 1 / 6 / 48 rows of 32320 columns, eager and in a CUDA graph; an unaligned row chunk declines.
- `test/registered/unit/spec/test_dspark_accept_tail.py`, `test_spec_tp_sync.py`, `test/registered/unit/distributed/test_parallel_state.py`: 36 passed.
- sgl-amd-mono's `test/e2e/test_deepseek_v41_server.py` (TP4, DSpark with `--monokernel-backend amd`, on top of sgl-project/sglang#45658 and the DSpark draft launch cuts): 9 passed, GSM8K 0.895 / 0.905.

## Speed Tests and Profiling

SPEED_TABLE

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-and-profile).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
