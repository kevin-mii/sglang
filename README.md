# mla_fp8: FP8 MLA prefill attention for gfx950 (MI350X / MI355X)

Kimi-K3's MLA prefill attention (Q/K head dim 192, V 128) in FP8 on gfx950.

- **Inputs and outputs:** bf16 Q, fp8 e4m3fn K/V with per-tensor dequant scales, bf16 O, optional LSE in aiter's
  layout.
- **Supported calls:** varlen batches, causal (bottom-right) or not, MHA/GQA, KV splits with a merge, and plans that
  can be captured in a CUDA graph.

Built on [moonmath-ai/amd-kernels](https://github.com/moonmath-ai/amd-kernels)' gfx950 bf16 MLA kernel (MIT,
`LICENSE.moonmath`). Kept from it:
- one CTA of 8 waves per CU;
- K/V moved by DMA (`buffer_load ... lds`) into swizzled LDS rings;
- lazy-rescale softmax frames;
- the host planner (work list, KV splits, merge, varlen plans).

## Measured on 8x MI350X (gfx950, 1000 W), one GPU, ROCm 10 / torch 2.11

Kimi-K3 at TP8: 12 heads per GPU. Times are ms per attention call, one layer. All three kernels use the same causal
inputs and are timed back to back. Relative error is max |out − ref| / max |ref| against an fp32 reference over the same
fp8 K/V.

| Shape | aiter fp8 (server path today) | moonmath bf16 | **mla_fp8** | vs aiter |
|---|---|---|---|---|
| First 8K prefill chunk | 0.318 | 0.253 | **0.187** | 1.70x |
| 8K chunk at 32K | 1.502 | 1.669 | **1.091** | 1.38x |
| Last 8K chunk of a 50K prefill | 2.356 | 2.621 | **1.684** | 1.40x |
| Agentic turn: 1.9K new on 50K | 0.695 | 0.736 | **0.529** | 1.31x |
| 4 agentic turns batched | 2.904 | 2.951 | **1.901** | 1.53x |
| Extend 512 on 50K | 0.218 | 0.235 | **0.151** | 1.44x |
| Extend 128 on 50K | 0.122 | 0.185 | **0.079** | 1.54x |
| 64 new on 50K | 0.118 | 0.170 | **0.079** | 1.49x |
| 256 new on 100K | 0.219 | 0.338 | **0.177** | 1.24x |
| No prefix, 2K | 0.088 | 0.039 | **0.038** | 2.36x |
| No prefix, 16K | 1.066 | 0.873 | **0.666** | 1.60x |
| No prefix, 32K | 3.774 | 3.628 | **2.408** | 1.57x |
| 16 x 128 new on 4K | 0.182 | 0.161 | **0.110** | 1.65x |
| Mixed batch (1.9K/50K, 300/8K, 64/30K, 4K/4K) | 1.047 | 1.004 | **0.671** | 1.56x |

**Error:** mla_fp8 is at 3–5%. aiter's fp8 path is at 4–6%; it pads K3's 12 heads to 16 and casts Q to fp8.

**How the aiter column was timed:** it includes the server's Q padding, fp8 cast and reduce, as the K3 server runs
them.

**Reproduce:**
- `/scratch/wt/tools/bench_mla_prefill_3way.py` for the full table;
- `bench_mla_fp8_quick.py` for this kernel alone.

## Design (csrc/mla_fp8.hip)

- **[32x32]** `v_mfma_f32_32x32x64_f8f6f4`, at 2x the bf16 rate. Each wave takes one 32-row q tile.
  - QK S^T = K Q^T runs 3 k-steps of 64 over the 192 head dims, so there is no padding.
  - PV O^T = V^T P^T is one k-step over 64 keys.
- **[p8]** P is S^T's C registers packed to fp8 in register order. The V^T fragments follow the same key order
  (`ds_read_b64_tr_b8`), so P needs no lane shuffles.
- **[log2]** The softmax scale and K's dequant scale are folded into Q, so S comes out in log2 units minus the frame
  and needs no per-score multiply.
  - The power-of-two part goes into the MFMA's E8M0 operand scale, so Q keeps fp8's normal range.
  - The scale operands must be VGPRs: SGPR scales miscompile on this ROCm.
- **[frame]** fp8 P has 3 mantissa bits over 2^-9 .. 448. A frame puts the row max at P = 2^7 and moves, upward only,
  when a score passes 2^8.5.
- **[rsum]** Row sums are VALU adds over the fp32 P.
- **[lds8]** Swizzled LDS tiles keep the 16-B K reads and 8-B transposed V reads free of bank conflicts.
- **Fragment pipeline:** each block reads every MFMA's fragment two MFMAs ahead, with one loop copy per wave group.
- **Planner:** `MaxParts` is 16 (moonmath has 8), so small-q extends spread over all CUs.

## Hardware findings that shaped it (gfx950, MI350X)

- **A full MFMA stream starves the other wave on the SIMD.** While one wave streams 32x32x64 MFMAs back to back, the
  other wave on its SIMD gets almost no VALU issue. Moonmath's two-wave ping-pong (one wave's MFMA block beside the
  other's softmax) therefore serializes on fp8.
- **A wave's own VALU runs beside its own in-flight MFMAs.** About 6 VALU ops per MFMA keep the MFMA pipe ~80% busy.
- **The kernel is power-bound.** At 1000 W the GPU drops from ~2.1 to ~1.6 GHz under this kernel. Removing all LDS
  fragment reads raises the clock by 16%, so LDS traffic per FLOP is the next lever: two q tiles per wave halve it. That
  is `csrc/mla_fp8_v2.hip`, a work in progress that is still limited by register allocation.

## Files

| File | What it is |
|---|---|
| `csrc/mla_fp8.hip` | The kernel (v1, the default) |
| `csrc/mla_fp8_v3.hip` | Self-pipelined waves with one barrier per tile pair; on par with v1 |
| `csrc/mla_fp8_v2.hip` | One wave per SIMD, two q tiles; WIP |
| `csrc/attention_plan.hpp`, `attention_prims.hpp`, `attention_config.hpp`, `opus/` | moonmath's, patched: 32-row waves, fp8 K/V bytes, dequant scales, per-kernel thread count, `MaxParts` 16 |
| `csrc/mla_fp8_api.cpp` | Torch bindings: `fwd_varlen`, `VarlenPlan` |
| `tests/test_mla_fp8.py` | 23 tests: accuracy over shapes, causal, varlen, scales, LSE, splits, graph replay |

## Build and use

```bash
MAX_JOBS=8 python setup.py build_ext --inplace   # MLA8_KERNEL=csrc/mla_fp8_v3.hip picks another kernel
```

```python
import mla_fp8
plan = mla_fp8.VarlenPlan().plan(cu_q, cu_k, max_q, max_k, total_q, total_k, heads, kv_heads, causal,
                                 host_cu_q, host_cu_k)        # once per forward pass
o = plan.run(q_bf16, k_fp8, v_fp8, softmax_scale, k_scale, v_scale)   # every layer
```
