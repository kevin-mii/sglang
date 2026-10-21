## Motivation

At TP4 a DeepSeek-V4.1 DSpark draft runs its Markov head over a TP-sharded vocab, five positions a block. Each position ran an embedding lookup, the W2-shard GEMV, a partial launch, a full-vocab all-gather of the step logits, a second partial launch over the gathered row and a finish; the block's step logits were only read by the sampling accept, which a greedy batch never runs. Around it, the draft parameters took four launches, the draft KV commit copied each stage's KV slice, and the in-graph accept copied its six results into its static buffers. On MI350X with `--monokernel-backend amd` these launches are a visible share of a step, and the step-logits gathers grow with the batch (5 MB a rank a block at 8 rows).

## Modifications

- **The draft sampling parameters in one launch** (`fill_draft_params`): the temperature clamp, the greedy mask and the step temperatures of `resolve_draft_params`.
- **The Markov step's bias GEMV in the launch that adds it** (`_markov_bias_partial_kernel`, `MarkovBiasGemv`, `BaseDSparkMarkovHead.bias_gemv`). A CTA a 64-column tile reads its W2 rows once for a 16-row block, rounds the bias to BF16 as `F.linear` does, adds it and stores the step logits and/or the tile's candidate. Sampled rows draw the same Philox noise, keyed by global token id; a block of greedy rows draws none. The candidate count is even, so a row of candidates is a 16-byte multiple, which aiter's custom all-gather takes. The GEMV sums in another order than hipBLASLt, so near ties can draft differently, as `MarkovGreedyStep` already documents; the accept verifies the drafts.
- **A batch that does not sample leaves the draft logits sharded.** Each step keeps its shard's logits and picks through the per-block candidates, which draws the full vocab's token (the noise is keyed by global id). `gather_step_logits` crosses the block's logits once: after an eager block, and in a folded proposal only for a batch that samples (`DsparkDraftSampler.corrected_logits`; the NVIDIA megakernel keeps writing its whole-vocab buffer). The step op's whole-logits mode goes with its last caller.
- **The draft KV commit takes strided slices on HIP**, as on Blackwell: `fused_k_norm_rope_flashmla` reads an explicit row stride.
- **The in-graph accept writes its static buffers.** The finalize kernel stores its three lengths there and copies `correct_len` and `bonus` through; the out-token kernel fills its buffer and widens `correct_len` as it loads it. Seven launches a step go.

## Accuracy Tests

MI350X, TP4, DeepSeek-V4.1-Flash, this branch on main 5e79333c95; the e2e with sgl-project/sgl-amd-mono main b27d604 plus sgl-project/sgl-amd-mono#21 (`--monokernel-backend amd`):

- `test/registered/kernels/ops/speculative/test_dspark_markov_sample.py`: 10 passed. New: the fused bias against its materialized bias (integer weights keep the GEMV exact, so bit for bit, NaN and tie picks included), and three shards keeping their logits against the full vocab, both biases.
- `test/registered/unit/spec/test_dspark_draft_block.py`: 6 passed. New: a sharded head keeps the whole vocab's logits; the gather's rank-major layout, cropped.
- `test/registered/spec/dspark/test_dspark_draft_graph.py`: 4 passed. New: `fill_draft_params` against the torch chain.
- `test/registered/spec/dspark/test_dspark_kernel_parity.py`: passed (20 subtests). New: the finalize and out-token kernels into static buffers.
- sgl-amd-mono's `test/e2e/test_deepseek_v41_server.py` (DSpark with the mono kernels, GSM8K included): 9 passed, GSM8K 0.905 / 0.900.

## Speed Tests and Profiling

Per-user decode tok/s:
- Setup: MI350X, TP4, DeepSeek-V4.1-Flash, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens.
- Arms, ABAB: main 5e79333c95 against this branch with sgl-amd-mono main b27d604 plus #21; then, with the sgl-amd-mono ATOM-parity stack (sgl-amd-mono#14 and its follow-ups), sgl-project/sglang#45658 against #45658 plus this branch.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| main, round 1 / round 2 | 656.7 / 659.7 | 540.9 / 540.8 | 437.0 / 437.9 | 322.9 / 326.7 |
| this PR, round 1 / round 2 | 675.2 / 670.4 | 551.7 / 550.2 | 441.2 / 446.8 | 335.1 / 338.4 |
| change of the means | +2.2% | +1.9% | +1.5% | +3.7% |
| #45658, ATOM-parity plugin, round 1 / round 2 | 724.3 / 721.5 | 593.4 / 587.3 | 477.4 / 484.1 | 359.9 / 354.8 |
| #45658 + this PR, ATOM-parity plugin, round 1 / round 2 | 737.4 / 741.2 | 600.6 / 591.5 | 483.5 / 490.7 | 371.7 / 367.5 |
| change of the means | +2.3% | +1.0% | +1.3% | +3.4% |

Acceptance length was 3.51-3.52 in every arm.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-and-profile).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
