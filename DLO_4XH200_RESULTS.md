# DLO 4x H200 validation results — MiniMax-H3

Machine: kevin-mi-dlo-4xh200 (4x NVIDIA H200 141GB, NVLink NV18 all-pairs, 2TB RAM,
192 cores). Branch `dlo/distributed-layerwise-offload` = upstream main (edff717ef)
+ db06ede0c, editable install `sglang 0.5.19.dev411+gdb06ede0c`, torch 2.13.0+cu130.
Checkpoint: `MiniMaxAI/MiniMax-H3` snapshot `42ed227e` (FL2VA transformer 62 GiB /
13 safetensors shards) served from GPFS at `HF_HUB_CACHE=/cluster-storage/models`.
Raw logs, perf dumps, MP4s, and memory samples: `outputs/dlo_4xh200/` (untracked).

Closes DLO_HANDOFF.md items 1 (NCCL AllGather correctness), 2 (perf/memory
matrix, adapted SP8 -> SP4 for this box), 3 (mmap preflight on the real
checkpoint), plus a code-level parity audit against the vLLM-Omni reference.

## 1. Environment preflight

- GPFS page-cache prewarm of the 62 GiB FL2VA transformer: **8.8 s (~7 GB/s)** —
  shards were already cache-resident; storage is a non-factor for load times.
- GPUs verified idle (`nvidia-smi --query-compute-apps` empty) at every run
  boundary; fixed GPU set 0-3; eager only (`--enable-torch-compile=false`); no
  diffusers-fallback marker in any accepted run.

## 2. CI smoke — real NCCL AllGather boots and generates (SP2)

```
CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
  python/sglang/multimodal_gen/test/server/test_server_2_gpu.py \
  -k minimax_h3_t2va_2gpu_dlo_sp2 -x -s
```

**PASSED** (1 passed in 344 s). Both ranks logged the checkpoint-mmap acceptance:

```
Bound 535 tensors to checkpoint-mmap views across 13 file(s); host weight pages
are node-shared via the OS page cache.
```

That is HANDOFF item 3: **the fail-closed loader preflight accepts the real
MiniMax-H3 checkpoint** (the grouped-QKV reorder rides the deferred-transform
path); no fallback reason line was emitted.

## 3. Correctness A/B — same seed, byte-compared MP4s

Config: t2va, short_edge 768 / 16:9 / 4.0 s, 8 steps, flow_shift 12.0,
audio_flow_shift 3.0, seed 42, TP1, prefetch 1, eager (the CI case's request,
driven through `sglang generate`). Gate: `sp2_plain` run twice first — SHA256
must match before cross-arm comparison means anything.

| run | topology | offload mode | SHA256 (mp4) |
|---|---|---|---|
| sp2_plain_r1 | 2 GPU, SP2 | `--dit-layerwise-offload true` | `3c8a9ea632ea...64d2cd8f` |
| sp2_plain_r2 | 2 GPU, SP2 | same (determinism gate) | `3c8a9ea632ea...64d2cd8f` |
| sp2_dlo_ag | 2 GPU, SP2 | DLO AllGather | `3c8a9ea632ea...64d2cd8f` |
| sp4_plain | 4 GPU, SP4 | `--dit-layerwise-offload true` | `d9c178ada5b2...9cbcf178` |
| sp4_dlo_ag | 4 GPU, SP4 | DLO AllGather | `d9c178ada5b2...9cbcf178` |
| sp4_dlo_rl | 4 GPU, SP4 | DLO rank-local | `d9c178ada5b2...9cbcf178` |

Result: **PASS at both SP degrees.** The determinism gate holds (sp2_plain r1 ==
r2), and within each SP degree every offload mode produced a byte-identical MP4:
DLO AllGather and DLO rank-local match plain layerwise offload exactly over real
NCCL at SP2 and SP4. (SP2 vs SP4 differ from each other, as expected — different
sequence-parallel decomposition.) This closes HANDOFF item 1.

## 4. Parity audit vs vLLM-Omni PR #5397 (merge 744c65b73)

Full audit compared `layerwise_offload_distributed.py` / `layerwise_offload_sharding.py`
/ `checkpoint_mmap.py` / `fsdp_load.py` / `server_args.py` against the reference
`distributed_layerwise_backend.py` (1491 lines), `base.py`, `diffusers_loader.py`,
`diffusion_model_runner.py`, `diffusion_engine.py`, and `cpu_offload_diffusion.md`.

**Verdict: no correctness regression relative to the reference.** Every
mechanism is SAME, EQUIVALENT, or intentionally stricter (fail-closed).

| mechanism | verdict | note |
|---|---|---|
| shard split (ceil/world, zero-pad) | EQUIVALENT | ours adds 32-byte alignment on shard boundaries; self-consistent both ends |
| per-tensor flat-buffer offsets | EQUIVALENT | both reconstruct with the offsets they packed with |
| AllGather stream/event topology | SAME | H2D on copy stream -> comm stream waits -> `all_gather_into_tensor`; compute waits on comm event |
| gather output sizing | ours stronger | ref gathers into an unsliced max-size buffer — runtime size-mismatch error with heterogeneous layer sizes; ours slices to the layer's padded numel |
| device slots (count/assignment/wraparound) | EQUIVALENT | ours `max(2, 2*prefetch)` + monotone ring counter == ref double buffer at depth 1, without ref's slot-contamination state machine |
| NCCL buffer address stability | SAME | persistent prefix-sliced buffers on both sides |
| prefetch scheduling | SAME at depth 1 | ours generalizes to deeper prefetch |
| rank-local mode | EQUIVALENT core | ours adds checkpoint-mmap host backing (~1 host copy/node); ref pays N x host RSS |
| mmap preflight | ours stronger | ref silently bypasses TP1 `weight_loader` transforms (would load wrong weights for H3-style grouped QKV) and warns past unloaded meta params; ours fail-closes or defers the transform, and raises on leftover meta |
| warmup/dummy-run under AllGather | EQUIVALENT | ref must skip its startup dummy run when DLO+AG and shard size > 1 (`diffusion_engine.py:815-836` — the dummy reaches one rank only); ours needs no skip: a warmup request spans the whole Ulysses SP group in lockstep, and DP-group sharding is a hard error (`server_args.py:3289-3296`). Verified empirically by the SP2 smoke test passing with warmup |
| CUDA graph interaction | N/A / stricter | ref stack never captures graphs; ours hard-errors AllGather+BCG |
| non-contiguous params | ours stronger | ref flattens (would break stride-dependent kernels); ours keeps strided rank-local copies |
| FSDP/HSDP combos | stricter | ours forbids DLO+FSDP in both modes; ref allows HSDP+rank-local (untested DTensor surface) |

Reference defects the port fixes (evidence the divergences are deliberate):
unsliced AllGather output vs max-size shared buffer; silent `weight_loader`
bypass on the mmap path; warning-only unloaded meta params.

Hardening note (ours, unreachable in supported configs): `prefetch_layer`
indexes `self._flat_sizes[layer_idx]` unguarded
(`layerwise_offload_distributed.py:408`); a layer whose every tensor is
non-contiguous would KeyError at prefetch instead of streaming its strided
copies. Loud failure, not corruption; base manager uses `.get`. Worth a
one-line follow-up.

## 5. Perf/memory matrix — TP1 x SP4, 50 steps, 5.0 s 768p T2VA (seed 1101)

Presets `minimax-h3-t2va-sp4-{noofld,plain,dlo-ag,dlo-rl}` (this commit), run
order noofld -> plain -> dlo-ag -> dlo-rl -> noofld(repeat, drift sentinel),
fixed GPUs 0-3, memory/power sampler at 1 s cadence.

| arm | request s (warmup excl.) | denoise s | peak HBM GiB (torch reserved / nvidia-smi) | worker VmHWM GiB/rank | mmap | arm energy Wh |
|---|---|---|---|---|---|---|
| no offload | 77.36 | 74.52 | 87.3 / 93.7 | 66.3 | n/a | 71.8 |
| plain layerwise | 77.94 | 75.11 | 26.0 / 32.4 | 113.7 | n/a | 77.0 |
| DLO AllGather | **79.61** | 76.77 | 30.2 / 36.7 | **60.4** | accepted | 81.2 |
| DLO rank-local | 320.35 | 317.53 | 29.2 / 35.7 | 71.4 | accepted | 113.9 |
| no offload (sentinel) | 77.29 | 74.52 | 87.3 / 93.7 | 66.3 | n/a | 71.5 |

All four arms produced a **byte-identical MP4** (`754dbf8bc046...960a3ff3`) —
at 50 steps DLO AllGather, DLO rank-local, and plain layerwise match the fully
resident model exactly. Drift sentinel: 77.29 s vs 77.36 s first run (0.1%),
matrix valid. GPFS/page cache stayed warm throughout.

Findings:

- **DLO AllGather costs +2.1% request latency vs plain layerwise (+2.9% vs no
  offload) while cutting per-rank host memory roughly in half** (VmHWM 60.4 vs
  113.7 GiB; the load-time transient dominates both numbers — steady-state
  pinned is the 1/sp shard, ~15.5 GiB/rank vs 62 GiB/rank for plain). Peak HBM
  is within 4 GiB of plain (persistent device slots + gather buffers) and ~2.6x
  below no-offload.
- **DLO rank-local is 4.1x slower (320 s)** at this topology: every rank
  streams the full 62 GiB layer set per step over PCIe and pays the
  checkpoint-mmap staging pack plus the deferred grouped-QKV transform per
  layer per step. This matches the reference's direction (vLLM-Omni measured
  no-AG ~1.8x slower on Ascend) but is steeper here because the mmap-backed
  host path adds host-side packing. Rank-local's win is host memory across
  same-node ranks (one page-cache copy per node instead of N pinned copies) and
  topology freedom (no collectives) — it is a throughput/DP-concurrency route,
  not a latency route; at DP concurrency the transfer cost amortizes across
  concurrent requests as in the vLLM DP8 reference.
- Measurement notes: `cudaHostAlloc` pinned pages are not accounted in
  `/proc/meminfo` Unevictable/Mlocked on this kernel — the meminfo-delta pinned
  estimate reads 0; per-PID VmHWM (sampled at 1 s) is the host-memory signal
  used above, and it includes shared mmap page-cache pages for the mmap arms
  (so dlo arms' true private usage is lower than shown). Arm energy integrates
  `power.draw` over the whole arm window (load + warmup + measured request on 4
  GPUs), not per-video energy; use it only to compare arms of equal structure.

## 6. DP2 x SP2: AllGather under concurrent DP replicas

Question: does per-SP-group weight sharding coexist with concurrent requests
across DP replicas (the failure mode vLLM-Omni guards with dummy-run skipping)?

**MiniMax-H3, DP2 x SP2 DLO AllGather (server mode):** the server-warmup
request completed its DLO-AllGather denoise on replica 0 while replica
1 sat fully idle — direct proof the shard/collective group is per-replica (a
group spanning DP ranks would deadlock exactly there). The request then hung in
the **H3 decoding stage**, not in DLO: `minimax_h3/stages/decoding.py:395-411`
broadcasts the audio-VAE decode over the **world group** under a stale "DP is
currently rejected by ServerArgs validation" assumption, so H3 T2VA hangs at
decode under any dp_size>1 on upstream main, offload or not. Pre-existing
pipeline limitation, independent of this branch — worth an upstream fix/issue.

**Z-Image-Turbo, DP2 x SP2 DLO AllGather (server mode): PASS.** Server warmed
up healthy; 2 concurrent same-seed requests + 1 sequential returned
byte-identical PNGs (`1f40b1eebebd...`), and a 4-way concurrent volley
completed in 2.9 s wall vs ~1.5 s per single request — both replicas denoising
simultaneously, each running AllGather on its own communicator, with no
interference, deadlock, or output divergence (all 4 byte-identical).

## 7. H3 serving determinism: first-request audio divergence (root-caused, fixed)

Symptom (found during DP validation, reproduced on unmodified main at dp=1,
warmup off): the first same-seed T2VA request of a server process produces one
MP4 (`3c8a9ea6...`), every later request a different stable one
(`70d42fc8...`). Localization: video streams are pixel-identical (SSIM 1.0);
only the audio stream differs. Instrumentation showed the audio latent entering
the audio-VAE decode, the VAE's full state dict, and every global numeric flag
(TF32, matmul precision, cudnn.benchmark, deterministic) are identical across
requests — same input, same weights, different output. Cause: cuDNN selects
convolution algorithms from free-workspace/allocator state, which differs
between the process's first decode and all later ones; the BigVGAN audio
vocoder is sensitive to the resulting kernel change (cudnn.allow_tf32 was also
enabled). The ref2va audio *encode* already guards against exactly this with a
determinism context (TF32 off, cuDNN disabled, math SDP); the decode path had
no such guard.

Fix (branch `fix/minimax-h3-dp-replica-group`): wrap the audio-VAE decode in a
deterministic-algorithm scope local to the H3 decoding stage (cudnn
deterministic on, TF32 off, benchmark off — cuDNN stays enabled). Measured
A/B: four sequential same-seed server requests byte-identical at an unchanged
2.1 s decoding stage; the heavier encode-style cudnn-off scope also passes but
costs 4.7 s, so it is kept only as the documented escalation path. The
canonical H3 audio bytes change once with this fix (deterministic across
requests and processes thereafter); video is untouched (SSIM 1.0).

## 8. Reference numbers (vLLM-Omni blog — different hardware, NOT comparable)

MiniMax-H3 768x1344, 124 frames, 50 steps on **8x B300**: DP1xSP8 AllGather wave
P50 34.55 s, 26.37 GiB peak/GPU, 68.08 Wh; DP8xSP1 rank-local 183.78 videos/h,
20.05 GiB, 43.97 Wh. This box is 4x H200 at SP4 — half the ranks, previous-gen
silicon; use for qualitative shape only (AllGather should cut pinned host
memory ~4x vs plain at similar latency; rank-local should match plain latency
with ~1 host copy per node when mmap is accepted).
