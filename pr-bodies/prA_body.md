## Motivation

The rest of ROCm/ATOM's DeepSeek-V4.1 decode work at 5e8d913 (ROCm/ATOM#2545) that sgl-project/sgl-amd-mono#14 left out: token parallel from 12 rows, the MoE down pipeline and K2's tile tasks.

## Modifications

Kernels (`deepseek_v41/gfx950`):

- **MoE, down weights a round ahead.** Round r+1's weights and scales load while round r multiplies, and round 0's before the shared expert's down.
- **K2, wo_a and wo_b over (task, token tile) units.** Past one token tile, each (task, tile) gets its own CTA instead of one CTA running every tile in turn; a tile's repeat weight reads hit L2. One tile keeps the original path.
- **MoE, single-tile ug off the shared expert's CTAs.** Those CTAs reach ug about 10 µs after the others and ended last.
- **Token parallel** (`token_par.py`), from 12 rows when the rows are a multiple of TP: the attention and FFN all-reduces become reduce-scatters to each row's owner, and the seam, norm, qkv and gates run on the rank's own rows. The q latent, window records, MoE input and an index source's normed rows and qr reach the other ranks through the peer buffer's gather regions. After the last layer one all-gather brings the residual, the last pre mix and the DSpark captures from their owners. The sums keep the all-reduce's rank order, so outputs are the same bits. `SGLANG_AMD_MONO_TOKEN_PAR=0` turns it off.

Host side (`deepseek_v41/spec.py`, `gfx950/runner.py`): the token-parallel buffers and K1 / K2 arguments, and the final row gather.

## Accuracy Tests

MI350X, TP4 unless noted. SGLang main 5e79333c95, plugin commit 83cd0a6 (on #14 at d668364):

- `test/kernels/test_deepseek_v41_kernels.py`: 4 passed. `check_k1` token parallel at 12 / 24 / 48 rows, TP2 and TP4, every variant: bitwise. `check_k2` token parallel: this rank's rows bitwise against the plain K2, and a sentinel check that no rank writes another's rows.
- `test/unit`: 120 passed.
- `test/e2e/test_deepseek_v41_server.py`: 9 passed, GSM8K 0.925 / 0.895.

## Speed Tests and Profiling

Per-user decode tok/s:
- Setup: MI350X, TP4, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens, SGLang main 5e79333c95.
- Arms: #14 against this PR, ABAB.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| #14, round 1 / round 2 | 681.1 / 675.8 | 568.5 / 565.0 | 453.4 / 450.0 | 331.5 / 331.6 |
| this PR, round 1 / round 2 | 692.2 / 693.9 | 568.5 / 569.4 | 464.5 / 461.5 | 349.5 / 344.2 |
| change of the means | +2.2% | +0.4% | +2.5% | +4.6% |

Acceptance length was 3.51-3.52 in every arm. Token parallel starts at 12 rows (c2), where the rows split 3 a rank.

## Checklist

- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
