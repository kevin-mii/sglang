# MiniMax-M3 on 4x MI350X/MI355X (TP4): launch and AIPerf AgentX scripts

Scripts used for the measurements in `M3_MI350X_STATUS.md` (repo root). Paths assume the benchmark host layout
(`/scratch/models/MiniMax-M3-MXFP4`, `/scratch/models/MiniMax-M3-EAGLE3-GQA`, `/scratch/aiperf-sa-venv` = SemiAnalysis AIPerf fork b7b16cf).

- `launch.sh` / `launch_v2.sh`: server launcher (MXFP4 quark checkpoint, EAGLE3, fp8 KV, custom all-reduce, Gluon prefill).
- `best_config.sh`: recommended real-acceptance config (source it, then run `launch_v2.sh`). GSM8K-1000 0.854.
- `best_lossy_config.sh`: ATOM-parity performance-only config (forced acceptance length 2.78 over 3 draft tokens,
  equivalent to ATOM's `--spec-decode-acceptance-rate 0.5933`). Outputs are not the model's; never use for accuracy.
- `run_sa_point.sh` + `bench_any.sh` + `atom_client_env.sh`: one AIPerf `inferencex-agentx-mvp` point with ATOM's client flags.
- `summarize.py`, `needle.py`, `steady.py`: result summary, needle-in-haystack sanity, steady-state decode harness.

Example:
```bash
source benchmark/minimax_m3_mi355x/best_config.sh
TAG=best GPUS=0,1,2,3 PORT=30000 SPEC_ATTN=decode EXTRA2="--max-running-requests 48 $EXTRA2" \
  ENVS2="NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1 $ENVS2" bash benchmark/minimax_m3_mi355x/launch_v2.sh
TAG=best_SAclient CONC=24 DURATION=3600 PORT=30000 bash benchmark/minimax_m3_mi355x/run_sa_point.sh
```
