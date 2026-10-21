## Motivation

An index source of DeepSeek-V4.1 computed its index query in four SGLang launches between K1 and K2: the MXFP8 quant of qr, the group-32 FP8 wq_b GEMM, the head-weights GEMV and the rope / FP4 pack. ROCm/ATOM computes the index query inside its mono launches (5e8d913). Stacked on the follow-up to #14.

## Modifications

Kernels (`deepseek_v41/gfx950/attention`):

- **K1 computes the index query** (`index_query.py`), bit for bit SGLang's chain (`_mxfp8_quant`, aiter's packed group-32 FP8 GEMM, `rocm_router_gemv_split_k`, `index_q_rope_pack_weights_flydsl`). The wq_b index tiles run after K1's own wq_b, with K chains split by width as aiter's packed kernel splits them. The head weights run in the seam's gate slot, and the rope and FP4 pack once both are in. Token parallel pushes each owner's head weights to the other ranks. Below 5 rows, SGLang's chain stays.
- **The index weights load before K1's wait on the q latent**, as ROCm/ATOM issues its `iq_ops`: every batch is in flight through the wait instead of two round trips after it.
- **One CTA per 16-head tile of the head weights.** One CTA ran all 20 head-weight chains in three rounds over its 8 waves. The splits are still summed in split order, so the bits are the same.

Host side (`deepseek_v41/spec.py`, `gfx950/runner.py`, `layer.py`): K1's index-query arguments and buffers. On an index source, SGLang's index-query launches are skipped.

ROCm/ATOM's in-K2 indexer (its scorer and `lds_topk` selection tree) is not included. On MI350X, its top-k took about twice SGLang's `topk_v2` at every context we measured:

| rows, context | 8k | 64k | 128k |
|---|---|---|---|
| 6 rows: score + ATOM's tree / + SGLang's top-k (µs) | 18.6 / 13.1 | 31.1 / 21.7 | 35.9 / 24.0 |
| 48 rows | 28.4 / 13.5 | 64.9 / 39.7 | 93.5 / 61.1 |

## Accuracy Tests

MI350X, TP4, plugin commit c90e940, SGLang 22b95da1e7 (main 617c8b67b1 with four HIP metadata launch cuts, an upcoming SGLang PR):

- `check_k1` (index K1, widths 6 / 12 / 24 / 48, with and without token parallel): the index query, its scales and head weights equal SGLang's chain bit for bit.
- `test/kernels/test_deepseek_v41_kernels.py`: 4 passed.
- `test/kernels/test_deepseek_v41_k2i.py`: 14 passed.
- `test/e2e/test_deepseek_v41_server.py`: 9 passed.

## Speed Tests and Profiling

K1 of an index source (`check_k1 --timeline`, the last CTA's end):

| | 6 rows | 48 rows |
|---|---|---|
| index query in K1 | 42.2 µs | 64.4 µs |
| + weights before the q wait | 30.6 µs | 60.9 µs |
| + a CTA a head tile | 24.7 µs | 51.6 µs |

Per-user decode tok/s:
- Setup: MI350X, TP4, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens, SGLang main 617c8b67b1.
- Arms: the follow-up's head (7c6e527) against this PR, ABAB.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| base, round 1 / round 2 | 683.1 / 679.0 | 554.0 / 557.1 | 439.3 / 439.4 | 326.5 / 326.0 |
| this PR, round 1 / round 2 | 686.3 / 692.2 | 544.4 / 547.3 | 438.0 / 444.4 | 329.3 / 334.2 |
| change of the means | +1.2% | -1.7% | +0.4% | +1.7% |

c2 moves about 2% between sessions. A c2-only ABAB in another session, this PR against this PR with SGLang's chain at 12 rows: 558.5 / 552.0 against 548.6 / 552.3, so K1's index query is not slower there. Acceptance length was 3.50-3.52 in every arm.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
