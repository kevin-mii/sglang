## Motivation

Three host-side pieces of a DeepSeek-V4.1 mono step that ROCm/ATOM does inside its kernels or avoids: the DSpark drafter's captures (a torch mean a captured layer), the int64 copy of SGLang's int32 SWA slots before the first K1, and, token parallel, a gather of every row's four residual streams for the head. Stacked on #14's follow-ups (PR A, PR B).

## Modifications

- **K1 writes DSpark's captures.** A layer DSpark captures needs the mean of its input residual's streams, which the spec took with `residual.mean(dim=1)`, a launch a captured layer. K1's seam already holds the four bf16 streams of each column in LDS for the layer input; it sums them in order and scales by 1/4, as torch's mean does, into a buffer the spec keeps per captured layer (`aux_out`, flagged at run time, so no K1 variant more). ROCm/ATOM's K1 writes its drafter taps the same way.
- **K1 and K2 read int32 SWA slots.** SGLang's HIP backend hands out int32 slots; the kernels read a slot's low word at the slot's stride (`slot_words`), so the forward's first K1 needs no int64 copy.
- **Token parallel gathers the head's rows, not the residual.** With DSpark capturing, nothing past the head reads the residual: each rank combines and norms its own rows, and the head's output rows cross ranks with the captures (10 KB a row instead of 40 KB), as ROCm/ATOM gathers them.

## Accuracy Tests

MI350X, TP4, plugin commit 4700131, SGLang 5ed1b36047 (main 617c8b67b1 with the HIP DeepSeek-V4 metadata launch cuts and the DSpark draft launch cuts, upcoming SGLang PRs):

- `check_k1` (widths 1 / 6 / 12 / 24 / 48, token parallel included): every output bit for bit as before, on int32 slots; the capture bit for bit against `torch.mean`. All PASS.
- `test/kernels/test_deepseek_v41_kernels.py`: 4 passed.
- `test/kernels/test_deepseek_v41_k2i.py`: 14 passed.
- `test/e2e/test_deepseek_v41_server.py`: 9 passed.

## Speed Tests and Profiling

Per-user decode tok/s, MI350X, TP4, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens, ABAB against PR B's head (c90e940), the same SGLang tree:

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| PR B, round 1 / round 2 | 715.2 / 716.6 | 575.4 / 578.6 | 473.2 / 473.6 | 365.2 / 361.2 |
| this PR, round 1 / round 2 | 716.6 / 716.6 | 578.7 / 580.5 | 476.7 / 473.7 | 365.1 / 364.9 |
| change of the means | +0.1% | +0.5% | +0.4% | +0.5% |

Within session noise (~1%): the launches and bytes it removes are small next to a step.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
