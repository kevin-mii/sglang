# DLO vs FSDP throughput — Cosmos3-Super on 4x H200

Question: at best-throughput settings, does DLO beat the best FSDP
configuration when the model does not fit resident? Model:
`nvidia/Cosmos3-Super` (transformer 120 GiB bf16 — pure DP-resident is
impossible on 141 GiB H200s, unlike MiniMax-H3's 62 GiB DiT which fits).
Workload: T2V 832x480, 29 frames, default steps, seed 7 — the shape from the
vLLM-Omni DLO blog's Cosmos3 tables. Method: per arm, server mode
(`--warmup-mode off`), one discarded warm round at full concurrency, then 8
timed videos at the arm's concurrency; 1 s memory/power sampler. Branch:
`dlo/distributed-layerwise-offload` @ `387fa8a3e7` (DLO + the H3 DP fixes;
concurrency uses the per-replica scheduler, one request stream per DP
replica).

## Results (8 warmed videos per arm)

| arm | topology | conc | videos/hour | steady denoise s | peak GPU GiB | worker VmHWM GiB |
|---|---|---|---|---|---|---|
| **dp2 x (FSDP2+USP2)** | weights sharded in HBM | 2 | **359.4** | 13.1 | 128.7 | 158 |
| HSDP4+USP4 | weights sharded in HBM | 1 | 256.8 | ~10 | 126.5 | 158 |
| DLO rank-local DP4xSP1 | full host copy/rank | 4 | 175.4 | 79.8 | 27.6 | 207 |
| plain layerwise DP4xSP1 | full host copy/rank | 4 | 175.4 | 79.8 | 13.4 | 207 |
| DLO AllGather DP2xSP2 | host shard 1/2 per rank | 2 | 138.3 | 50.7 | 20.0 | 168 |

All arms byte-identical within their SP decomposition; dp2x(FSDP2+USP2) and
DLO-AG DP2xSP2 produce the SAME output hash (same SP2 math, weights on GPU vs
host) — a cross-strategy correctness check. All 8 timed outputs identical in
every arm. FSDP without USP (`--use-fsdp-inference` alone) fails upstream in
the Cosmos3 denoise (`Tensors must be contiguous` in `all_gather`); FSDP must
be paired with Ulysses.

## Findings

1. **When sharded weights fit in HBM, FSDP + DP concurrency dominates DLO on
   raw throughput (2.05x)**: dp2x(FSDP2+USP2) keeps 60 GiB/rank resident and
   gathers over NVLink (~450 GB/s); DLO re-uploads host shards over PCIe
   (~64 GB/s) every denoise step. The steady denoise times decompose exactly:
   FSDP 13.1 s (compute-bound); DLO-AG 50.7 s = ~13 s compute + ~35 s PCIe
   (60 GiB/rank/step); rank-local 79.8 s = ~13 s + ~70 s PCIe
   (120 GiB/rank/step).
2. **DLO's price for throughput buys HBM headroom**: the FSDP winner peaks at
   128.7 GiB/GPU — 13 GiB from the H200's ceiling; it cannot run this model on
   96 GiB or smaller cards at all (FSDP2 shard = 60 GiB/rank). The offload arms
   peak at 13-28 GiB. Per GiB of peak HBM, plain layerwise DP4 delivers 13.1
   videos/h/GiB vs FSDP's 2.8. DLO/offload is the throughput answer exactly
   when sharded weights do NOT fit (consumer GPUs; models > world x HBM), or
   when the freed HBM converts into more work per request (longer/higher-res
   videos that would OOM the FSDP config).
3. **DP concurrency is already delivered by per-replica scheduling.** At equal
   concurrency AllGather beats rank-local per-request (50.7 vs 79.8 s denoise),
   but halving concurrency cost more than the PCIe saving here (138 vs 175
   v/h). vLLM-Omni's `dp_concurrent` mode (weight shards across ALL ranks,
   synchronized request waves) would cut host memory to 1/world (~30 GiB/rank
   vs today's 207 GiB VmHWM at DP4 rank-local, since Cosmos3 declines
   checkpoint-mmap: "the model preprocesses its loaded state dict") and cut
   rank-local's PCIe volume 4x — its throughput ceiling would be DLO-AG-style
   ~13+9 s/step-window at concurrency 4, i.e. potentially ~2x today's DLO
   numbers. That is the case for implementing DP-group sharding — but on
   hardware where FSDP fits, FSDP still wins; the mode matters for the
   host-memory-constrained and HBM-constrained regimes together.
4. DLO rank-local == plain layerwise on throughput here (mmap declined for
   Cosmos3, so both stream full pinned copies); DLO's slot reuse showed a
   higher GPU peak (27.6 vs 13.4 GiB) because slots are sized to the largest
   layer and Cosmos3's two layer groups (`gen_layers`, `language_model.layers`)
   are heterogeneous.

## Reference points

vLLM-Omni blog (different hardware): B300 T2I — HSDP+USP4 wave 15.19 s vs
DLO+AG DP4 43.69 s, but DLO won their throughput column via 4-way concurrency
(0.0915 vs 0.0658 outputs/s) because they did not test FSDP+DP-concurrency
hybrids. Our dp2x(FSDP2+USP2) is that hybrid, and it wins. Their MiniMax-H3
DP8xSP1 rank-local: 183.8 videos/h on 8x B300 — our DP4xSP1 gets 175.4 on 4x
H200 (Cosmos3-Super, similar shape).

Raw logs, per-request times, and memory samples: `outputs/dlo_throughput/`
(untracked).
