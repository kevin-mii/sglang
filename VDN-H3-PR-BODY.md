## [diffusion] feat: support VDN-H3 (hybrid window-softmax + Video Delta Attention MiniMax-H3) with a hybrid-window-h3 attention backend

### Motivation
[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3) (Video DeltaNet) replaces every MiniMax-H3 DiT block's dense self-attention with a hybrid of an exact chunk-aligned window softmax (chunk 5, radius 1, anchor frames, gated) and a frame-wise linear-attention branch (Video Delta rule) over the window's complement, distilled to 8 NFE. The checkpoint ships as deltas (4.3 GB linear branch + two LoRAs) on the released H3.

### Modifications
- **Distribution**: registered model overlay (`kevin-mi/VDN-H3-overlay`, same mechanism as FastH3) whose materializer prefuses both LoRA adapters into the transformer shards (fp32 accumulate, verified bit-exact against base + Σ B@A), attaches the linear branch as an extra shard, writes `hybrid_attention` into `transformer/config.json`, links the Qwen3-VL conditioner from `MiniMaxAI/MiniMax-H3`, and reuses FastH3's VAE/RoPE conversion.
- **Arch config**: `VDNHybridAttentionArchConfig` + name-mapping rules for the branch keys; a plain MiniMax-H3 load is byte-identical to today.
- **Attention backend** `hybrid_window_attn_h3`: window softmax as a union of dense FlashAttention varlen calls (default) or the in-tree VSA-H3 tile-64 Triton kernel with index lists from the window (`vdn_window_kernel=tiles`); request-static metadata built once per request in the denoising stage; per-head gate epilogue; dense fallback for the refiner / auxiliary components.
- **DiT**: `MiniMaxH3Attention` grows `softmax_gate`, `linear_attention` (`MiniMaxH3VDNLinearBranch`: features, TF32 statistics, Cholesky delta rule, forward/reverse scans, alpha-bridged boundary gather, gated RMSNorm readout, text-state seed) and `to_out_linear`; a dedicated eager hybrid core keeps raw q/k for the branch and supports Ulysses (one packed all-to-all of raw q/k/v, QK-norm+RoPE after it with a full-sequence RoPE cache, beta/gates by head, frame-mean all-reduce, head-range slicing of per-head params).
- **Kernels**: out-of-place fused QK-norm+RoPE JIT variant (bitwise equal to the in-place kernel); fused Triton branch kernels (temporal conv + SiLU + L2 norm, SiLU + L2 norm on strided q, statistics prologue, RMSNorm × gate epilogue) via the `sglang.kernels.ops.diffusion` facade.
- **Pipeline wiring**: `VDNH3Pipeline`, `VDNH3PipelineConfig` (forces the hybrid backend; rejects `--model-variant`, `quality="high"`, ring, torch.compile, BCG), `VDNH3SamplingParams` (9 sigma grid points = 8 NFE, t2va only), registry + overlay registry, bench presets, CI case `vdn_h3_t2va_4gpu_h100`, docs (cookbook §7, attention backends, compatibility matrix, catalog).

### Correctness
- Unit tests: window backend vs masked-dense reference on a ragged layout (both kernels), full cover == dense, gate, refiner fallback; branch arithmetic vs step-by-step references; head-slice == full run; fused kernels vs eager; registry/admission; materializer prefuse on synthetic tensors.
- Block-level parity against OpenVDN's `HybridAttention` with the real weights on identical inputs (blocks 0/25/49): relative L2 ≈ 5e-3 to 7e-3 for both the dense base+LoRA smoke and the full hybrid path (bf16 rounding; the hybrid tracks the dense path's own deviation).
- 1-GPU vs 4-GPU Ulysses outputs agree to the same PSNR as the base dense H3 path does (≈23–25 dB after 8 steps, chaotic amplification of reduction order).

### Review
A medium-effort code review (8 finder angles, 14 verified findings) was applied: pinned overlay revision, request-static index tensors (no per-block host syncs), single admission guard, repo parallel linears instead of a private linear, msgspec.Struct containers, dead-code removal, registered test env var, shared JIT launch tail. Two pre-existing qknorm-rope kernel test failures on this torch 2.13 / B200 box (`test_qknorm_rope_preserves_split_bf16_rounding`, `test_ernie_qknorm_rope_is_bit_exact`) reproduce with the untouched kernel source and are unrelated.

### Also in this PR
- MiniMax-H3 synthetic warmup honors `--warmup-num-frames` / `--warmup-resolutions` (it was hard-coded to a 5 s clip), removing the 2-3 s first-forward penalty for any longer served clip on base H3, FastH3 and VDN-H3.

### Benchmarks (same 4×B200 box, 1344×768, 345 frames / 14.375 s, t2va, 8 NFE, seed 1000, denoise stage, steady-state per NFE)
| Config | s/NFE | Denoise (8 NFE) | Peak/GPU |
| --- | ---: | ---: | ---: |
| VDN reference (OpenVDN stack), fp8, 1×B200 | 6.47 | 51.7 s | – |
| VDN reference, fp8, 4×B200 Ulysses (standard / 3+1 branch-parallel) | 2.68 / 2.61 | 21.4 / 20.9 s | – |
| SGLang VDN-H3, bf16, 1×B200 | 7.86 | 64.1 s | 147.6 GB |
| SGLang VDN-H3, bf16, 4×B200 Ulysses (served-shape warmup) | 2.53 | 20.3 s | 97.9 GB |
| SGLang VDN-H3, fp8, 1×B200 (DiT resident) | 7.49 | 61.1 s | 63.9 GB |
| SGLang VDN-H3, fp8, 4×B200 Ulysses (served-shape warmup) | **2.34** | **18.7 s** | 63.0 GB |
| OpenVDN published, fp8, 8×B200 Ulysses 5+3 | 1.40 | 11.2 s | – |
| OpenVDN reference, bf16 tuned, 1×B200 | 7.91 | 63.3 s | – |

Per GPU-second per NFE: SGLang 4×B200 fp8 = 9.4 vs the published 8×B200 = 11.2. Same host, same GPU count: SGLang is 6–9% faster than the released stack. The single-GPU fp8 gap is the online fp8 GEMM path (bf16 matches the released tuned bf16 stack).

Per-block hybrid attention at the paper shape (1 GPU, bf16): 169 ms eager → 117 ms with the fused kernels; the tile kernel path measures 226 ms vs 169 ms for the decomposed FlashAttention path on SM100, so decomposed is the default.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01M1KA1GB6xFrCp2qbMy9u23
