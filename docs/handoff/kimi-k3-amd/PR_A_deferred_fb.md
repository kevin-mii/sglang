# [AMD][Kimi-K3] Honor the deferred f_b handoff on the ROCm fused KDA in-proj path

## Motivation
`forward_qkvbfg_fused`'s ROCm single-GEMM in-proj branch (`SGLANG_ROCM_K3_FUSE_KDA_INPROJ`) always applies `f_b` to the `[f_a | b]` tail, ignoring `defer_f_b`. With `SGLANG_K3_KDA_FUSED_BACKEND=aiter` the model flags the gate as deferred while handing the backend a full-width `(bs, 1536)` gate; the aiter fused decode kernel's `covered()` check rejects it (it expects `f_a` of width 128) and the fallback re-applies `f_b` with the HIP stash layout, which fails decode CUDA-graph capture on 8x MI350X:

```
Capture cuda graph failed: mat1 and mat2 shapes cannot be multiplied (74x1536 and 128x1536)
```

The same failure reproduces on current main with `--dcp-size 8` (`kda_backend.forward_decode -> kimi_k3_tiny_gemm`, `32x1536 and 128x1536`).

## Modifications
Return `f_a` when `defer_f_b` is set so the fused decode kernel (f_b + conv + recurrence + gated RMSNorm) engages on the fused in-proj path. One line in `python/sglang/srt/models/kimi_k3.py`.

## Accuracy / performance
8x MI350X, Kimi-K3 MXFP4, TP8 (with and without DCP8), `SGLANG_K3_KDA_FUSED_BACKEND=aiter`: server captures decode graphs; GSM8K-200 0.985 (DSPARK block 3 + DCP8) / 0.995 (no spec). Without the env the path is unchanged.

## Checklist
- [x] Format your code according to the Code Formatting with Pre-Commit.
- [x] Add unit tests as outlined in the Running Unit Tests. (n/a: kernel-path fix validated on hardware; no CI runner for MI350X)
- [x] Update documentation / docstrings / example tutorials as needed.
- [x] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
