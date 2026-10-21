## Motivation

Four host-side pieces of a DeepSeek-V4.1 mono step that ROCm/ATOM does inside its kernels or avoids: the DSpark drafter's captures (a torch mean a captured layer), the int64 copy of SGLang's int32 SWA slots before the first K1, token parallel, a gather of every row's four residual streams for the head, and an all-reduce for each Engram layer's lookup. Stacked on #14's follow-ups (PR A, PR B).

## Modifications

- **K1 writes DSpark's captures.** A layer DSpark captures needs the mean of its input residual's streams, which the spec took with `residual.mean(dim=1)`, a launch a captured layer. K1's seam already holds the four bf16 streams of each column in LDS for the layer input; it sums them in order and scales by 1/4, as torch's mean does, into a buffer the spec keeps per captured layer (`aux_out`, flagged at run time, so no K1 variant more). ROCm/ATOM's K1 writes its drafter taps the same way.
- **K1 and K2 read int32 SWA slots.** SGLang's HIP backend hands out int32 slots; the kernels read a slot's low word at the slot's stride (`slot_words`), so the forward's first K1 needs no int64 copy.
- **Token parallel gathers the head's rows, not the residual.** With DSpark capturing, nothing past the head reads the residual: each rank combines and norms its own rows, and the head's output rows cross ranks with the captures (10 KB a row instead of 40 KB), as ROCm/ATOM gathers them.
- **One all-reduce for every Engram layer's rows.** SGLang's Engram lookup reassembles each Engram layer's TP-sharded rows with its own all-reduce (an RCCL int32 all-reduce, so the bf16 bits survive). The spec now gathers every Engram layer's rows of the step up front (each rank its shards' rows, zeros elsewhere, through SGLang's `engram_gather`) and reassembles them with one int32 all-reduce, as ROCm/ATOM stages its Engram lookups. One shard owns each row, so the integer sum keeps every bit.

## Accuracy Tests

MI350X, TP4, plugin commit 75626ee (on PR B; the tree tested as 76ff1d7), SGLang main 5e79333c95:

- `check_k1` (widths 1 / 6 / 12 / 24 / 48, token parallel included): every output bit for bit as before, on int32 slots; the capture bit for bit against `torch.mean`. All PASS.
- `test/kernels/test_deepseek_v41_kernels.py`: 4 passed.
- `test/kernels/test_deepseek_v41_k2i.py`: 14 passed.
- `test/unit`: 120 passed.
- `test/e2e/test_deepseek_v41_server.py`, with sgl-project/sglang#45658 and #45663 on SGLang main: 9 passed, GSM8K 0.915 / 0.905.

## Speed Tests and Profiling

Per-user decode tok/s:
- Setup: MI350X, TP4, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens, SGLang main 5e79333c95.
- Arms: PR B against this PR, ABAB.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| PR B, round 1 / round 2 | 694.9 / 700.4 | 570.7 / 570.1 | 466.1 / 468.1 | 349.8 / 351.8 |
| this PR, round 1 / round 2 | 705.6 / 712.3 | 576.0 / 572.4 | 460.2 / 468.2 | 351.3 / 348.8 |
| change of the means | +1.6% | +0.7% | -0.6% | -0.2% |

c4 and c8 move within session noise (about 1%); the launches and bytes it removes count most at c1. Acceptance length was 3.51-3.52 in every arm.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
