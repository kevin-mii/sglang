## Motivation

On HIP, every DeepSeek-V4.1 decode / target-verify metadata build ran small launches whose results V4.1 never reads, or that one existing launch can cover. With the mono kernels (`--monokernel-backend amd`), these launches are a visible share of a batch-1 DSpark step: about 15 of them go.

## Modifications

All in the HIP DeepSeek-V4 attention backend and its glue (`deepseek_v4_backend_hip_radix.py`, `kernels/ops/attention/dsv4/attn_glue_hip.py`):

- **SWA window indices in one launch.** The plain causal window went through `get_swa_page_indices` (eight torch ops) and a -1 pad. `BuildCausalSwaPageIndices` already builds the same int32 rows, -1 padded, for the bounded-replay floor; with no floor it is the same window.
- **No ratio 4 / 128 metadata when the pool has neither.** DeepSeek-V4.1's pool holds ratios 0 / 1 / 2, yet each step ran `_init_compressed_attn_metadata_kernel` and filled a ratio-4 top-k buffer. The HIP `DSV4AttnMetadata` now carries the pool's `present_ratios`, as the CUDA metadata does: `init_compression_metadata` skips the launch and `sparse_buffers` skips ratio 4 when they are absent. DeepSeek-V4 is unchanged.
- **Ratio 1 / 2 raw indices in the `sparse_buffers` launch.** `init_flashmla_related` filled `c1` / `c2` `sparse_raw_indices` with -1 in two torch launches next to `sparse_buffers`, which already fills the same-shaped page indices.
- **One launch for the same-length row expansion.** `expand_extend_with_same_length` built each request's rows' causal lengths and slots in four torch launches, and the DSpark draft block added its block size in a fifth. One Triton launch (`expand_same_length`) does both, the block size as an offset, in the torch ops' dtypes.
- **The rows' positions and window lengths in that launch.** `make_core_attn_metadata` derived each row's position (length - 1) and window length (clamped to `SWA_WINDOW`) in two torch launches; `expand_same_length` writes both, int32 as before, for the target-verify and DSpark draft-block metadata.
- **Target-verify metadata skips the lengths nothing reads.** `make_forward_metadata_from_raw_verify` built the block-extended and extend lengths (an add, a fill, a minimum and a clamp) every step. Only the unified KV's prefill meta and the c4 / c128 compressors read them; without those, the expansion adds the draft block as its offset.
- **Target verify's low-ratio slots and positions in one launch.** Decode already widens both in `widen_pair_i64`; it now repeats each slot a draft token, so target verify takes that launch instead of `token_req_indices` and a cast.
- **The full-to-SWA lookup and its int32 cast in one launch** (`gather_i32`), wrapping a -1 location to the mapping's trailing -1 as torch indexing does. A pool with its own lookup keeps it.
- **No copy for SWA page indices already aligned.** `init_compression_metadata` padded `swa_page_indices` to `PAGE_INDEX_ALIGNED_SIZE` with `F.pad`, which copies even when it adds nothing; it pads only a misaligned width now.

## Accuracy Tests

MI350X, TP4, this branch on main 5e79333c95, with sgl-project/sgl-amd-mono main b27d604 plus sgl-project/sgl-amd-mono#21:

- `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py`: 17 passed. New cases, each against the torch chain it replaces: `expand_same_length` (int32 / int64 lengths and slots, the block offset, the positions and window lengths), `sparse_buffers`' ratio 1 / 2 raw indices, `widen_pair_i64` with repeated slots, and `gather_i32` (int32 / int64 indices, -1 locations).
- sgl-amd-mono's `test/e2e/test_deepseek_v41_server.py` (DeepSeek-V4.1-Flash served with DSpark and the mono kernels, GSM8K included): 9 passed, GSM8K 0.895 / 0.905.

## Speed Tests and Profiling

Per-user decode tok/s:
- Setup: MI350X, TP4, DeepSeek-V4.1-Flash, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens.
- Arms: main 5e79333c95 against this PR, the same sgl-amd-mono build in both (main b27d604 plus #21), ABAB.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| main, round 1 / round 2 | 656.7 / 659.7 | 540.9 / 540.8 | 437.0 / 437.9 | 322.9 / 326.7 |
| this PR, round 1 / round 2 | 677.0 / 681.3 | 558.2 / 556.9 | 444.9 / 444.5 | 330.6 / 330.6 |
| change of the means | +3.2% | +3.1% | +1.7% | +1.8% |

Acceptance length was 3.51 in every arm.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-and-profile).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
