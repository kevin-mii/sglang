# [Cookbook][AMD] Kimi-K3 MI350X/MI355X: DCP8 on the aiter backend, prefill/decode interleave, bf16 KDA state

## Motivation
The MI350X/MI355X balanced cell serves Kimi-K3 as flat TP8 on the Triton attention backend. With #34432 the aiter backend carries DCP (Gluon MLA decode with LSE, DSPARK verify under DCP), and on the InferenceX AgentX trace (multi-turn agentic sessions, 100k-450k tokens, 95%+ prefix reuse) three things decide the curve on 8x MI350X:

1. **DCP8** shards the MLA KV across the ranks: sessions stay GPU-resident above concurrency 4 instead of being re-prefilled.
2. **Prefill/decode interleave.** Chunked prefill runs its chunks back-to-back, so every resident session stops streaming for the whole prefill of a cold 100k-400k turn (39 s in a stall test). `--chunked-prefill-size 8192 --prefill-decode-interval 16` bounds the stall to one 8k chunk and keeps per-request mean ITL near the decode step.
3. **KDA state pool.** The default 300 fp32 slots (5 per running request) cap how many session states the radix cache can keep; at concurrency 16 the cache lost 8 points of hit rate to cold re-prefills while the MLA KV pool was 35% used. `--mamba-ssm-dtype bfloat16` halves the state so the same memory holds twice the slots (`--mamba-full-memory-ratio` / the calculator sizes it).

## Modifications
- `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`: MI350X and MI355X balanced cells: `--dcp-size 8 --dcp-comm-backend a2a --attention-backend aiter --chunked-prefill-size 8192 --prefill-decode-interval 16 --mamba-ssm-dtype bfloat16`.
- `docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx`: hardware table row.

## Results (8x MI350X, InferenceX AgentX trace, 1200 s window, per GPU; interactivity = 1/p90 per-request mean ITL)
| point | tok/s per GPU | P90 interactivity | notes |
| --- | ---: | ---: | --- |
| c8, previous cell (TP8 triton, no DCP) | capacity-bound above c4 | | |
| c8, DCP8 aiter, no spec | 2762 | 29.3 | |
| c8, DCP8 aiter, DSPARK 3 (accept length pinned 3.0) | 3842 | 43.2 | |
| c16, DCP8 aiter, no spec | 4154 | 18.1 | bf16 KDA state, 600 slots |
| c16, DCP8 aiter, DSPARK 3 (pinned 3.0), interval 6 | 4490 | 19.1 | |
| c32, DCP8 aiter + HiCache L1/L2, no interleave | 5866 | 11.2 | interleave off above c16 (prefill-bound) |

Accuracy: GSM8K-200 0.995 (no spec) / 0.985 (DSPARK 3) at parallel 32 under DCP8.

Requires the two fixes in the companion PRs (deferred f_b on the fused KDA in-proj path when `SGLANG_K3_KDA_FUSED_BACKEND=aiter` is set; free-segment coalescing on request completion under DCP).

## Checklist
- [x] Format your code according to the Code Formatting with Pre-Commit.
- [ ] Add unit tests as outlined in the Running Unit Tests. (docs/config only)
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
