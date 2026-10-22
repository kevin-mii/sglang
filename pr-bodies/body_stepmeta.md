## Motivation

ROCm/ATOM builds a DeepSeek-V4.1 decode step's attention metadata in one launch (`write_step_meta`). SGLang's HIP backend built a target-verify step's metadata in ten:

- eight in `make_forward_metadata_from_raw_verify`: the same-length expansion, the int32 lengths, the causal SWA window indices, the page table, the ratio 1 / 2 compression metadata, the sparse buffers, and the two index page tables;
- two in `init_forward_metadata_in_graph`: the SWA locations and the low-ratio inputs.

The DSpark draft block's metadata took six more: the same-length expansion, the int32 lengths, the window gather, the block's SWA indices, the page table, and the SWA locations.

With the mono kernels (`--monokernel-backend amd`) these sixteen launches are about 70 µs of a batch-1 DSpark step. Stacked on sgl-project/sglang#45658, whose kernels it fuses.

## Modifications

- **`verify_step_meta`** (`attn_glue_hip.py`) builds all of them in one launch. Every value is a function of its row (the lengths are uniform: no prefix sums), so no program waits on another:
  - program `(r, 0)` handles request `r`'s rows: lengths, positions and window lengths, the SWA window gather, the ratio 1 / 2 locations and lengths, the -1 filled top-k buffers, and with the step's inputs the SWA locations and low-ratio inputs;
  - program `(r, 1 + i)` handles page tile `i`: it reads the request's page row once for the page table and both index page tables (the chain read it once a row, six times).
- **`DeepseekV4HipRadixBackend._fused_verify_metadata`** takes it for a pool whose compressed caches are ratios 1 / 2 only (V4.1) with no request window, no unified KV, and an index page smaller than a compressed page. It sets the same fields as the chain. Everything else keeps the chain.
- **`draft_step_meta`** does the same for the DSpark draft block. Program `(r, 0)` builds request `r`'s block rows: every row sees the window up to the committed prefix, then the whole block. Program `(r, 1 + i)` builds page tile `i`. **`_fused_draft_block_metadata`** takes it for V4.1's paged draft window (no ring, no request window).
- `init_forward_metadata_in_graph` skips the SWA-location and low-ratio launches when the fused launch built them (`DSV4Metadata.step_inputs_fused`).

## Accuracy Tests

MI350X, this branch on #45658 (main 5e79333c95):

- `test/registered/kernels/ops/attention/dsv4/test_dsv41_decode_glue_hip.py`: 41 passed. New:
  - `verify_step_meta` against the launch chain, bit for bit in every output and dtype (1 / 3 / 8 requests; ratios 1 + 2 and 2 alone; with and without the step's inputs; a window shorter than 128, negative locations, -1 locations);
  - the backend's fused target-verify metadata against the chain's, every field of `DSV4Metadata`, `DSV4AttnMetadata` and the index metadata, with the fused launch asserted to run;
  - `draft_step_meta` against its chain, bit for bit (1 / 4 requests; with and without `swa_valid_from`; with and without the SWA locations; a window still filling);
  - the backend's fused draft-block metadata against the chain's, every field.
- sgl-amd-mono's `test/e2e/test_deepseek_v41_server.py` (TP4, DSpark, `--monokernel-backend amd`): 9 passed, GSM8K 0.905 / 0.900 on this branch; 9 passed, GSM8K 0.900 / 0.890 with sgl-project/sglang#45663 and the sgl-amd-mono ATOM-parity stack on top. With the draft-block fusion as well, 9 passed, GSM8K 0.895 / 0.875.

## Speed Tests and Profiling

Per-user decode tok/s:
- Setup: MI350X, TP4, DeepSeek-V4.1-Flash, DSpark with simulated acceptance 3.51, 5 rounds of 3000 tokens.
- Arms, ABAB:
  - with sgl-amd-mono main b27d604 plus #21: #45658 against this branch;
  - with the sgl-amd-mono ATOM-parity stack (sgl-amd-mono#14 and its follow-ups): #45658, sgl-project/sglang#45663 and the logits-gather cuts, against those plus this branch.

| Arm | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| #45658, round 1 / round 2 | 684.0 / 682.4 | 558.0 / 554.6 | 445.3 / 448.8 | 328.3 / 326.9 |
| this branch, round 1 / round 2 | 681.8 / 687.0 | 554.7 / 559.3 | 446.3 / 445.5 | 329.6 / 327.9 |
| change of the means | +0.2% | +0.1% | -0.3% | +0.4% |
| ATOM-parity stack, round 1 / round 2 | 741.3 / 741.7 | 604.7 / 598.9 | 489.0 / 487.5 | 375.9 / 370.9 |
| ATOM-parity stack + this branch, round 1 / round 2 | 748.9 / 751.9 | 602.2 / 603.2 | 490.8 / 489.7 | 374.1 / 370.6 |
| change of the means | +1.2% | +0.1% | +0.4% | -0.3% |
| ... + the target-verify fusion (another session), round 1 / round 2 | 746.6 / 744.4 | 604.6 / 591.4 | 495.0 / 495.2 | 375.4 / 376.0 |
| ... + the draft-block fusion, round 1 / round 2 | 757.2 / 750.6 | 596.5 / 597.9 | 493.0 / 500.3 | 367.6 / 374.6 |
| change of the means | +1.1% | -0.1% | +0.3% | -1.2% |

Acceptance length was 3.50-3.51 in every arm.

Profile on the ATOM-parity stack (20 decode steps), at c1: 243 kernels a step, then 234 with the target-verify fusion, then 229 with the draft-block fusion. The target-verify launch takes 4.4 µs (5.7 µs at c8), where the ten launches took about 45 µs. The draft-block launch takes 4.5 µs, where its six took about 27 µs. The gain shows where launches are the larger share of a step, c1 on the faster stack; elsewhere it is within session noise (about 1%).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-and-profile).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
