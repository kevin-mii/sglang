# MiniMax-M3 MXFP4 on 8x MI355X: handoff (2026-09-23)

Goal: reproduce, then improve, this workload on 8x MI355X with `amd/MiniMax-M3-MXFP4` + EAGLE3.
The workload is ISL 74,176 tokens, OSL 650, 90% cached prefix, GSM8K prompts, c = 64/80/128 with 5 x c requests each.
Target: TTFT p50 < 3 s. The throughput target ("2.5TPS") still needs to be pinned down: 2.5x the baseline output throughput, or something else?

## TL;DR

- Branch **`M3-perf-rebase`** is M3-perf merged onto sglang main `4e60d70ba4`. That is the exact sglang commit of the latest image
  `lmsysorg/sglang-rocm:v0.5.20-rocm724-mi35x-20260923` (digest `sha256:02108de8f9418a12425fb56415794fe82d180cb985488725d485e96d97a9a26b`).
  It serves correctly on that image: **GSM8K-500 = 0.872** (the branch reference is 0.85-0.89) and EAGLE3 accept length 2.8 of 4.
- First reproduction, 2 x TP4 behind the cache-aware router. It beats the given baseline on every row:

  | conc | TTFT p50 ms (given -> here) | TTFT p90 ms | out tok/s/GPU | in tok/s/GPU | acc.len |
  |---:|---:|---:|---:|---:|---:|
  | 64 | 5,147 -> **4,921** | 22,294 -> 8,149 | 186 -> **246** | 21.2K -> 28.1K | 2.77 |
  | 80 | 10,433 -> **6,567** | 21,436 -> 10,267 | 207 -> **353** | 23.6K -> 40.3K | 2.81 |
  | 128 | 9,057 -> **8,541** | 43,705 -> 16,358 | 238 -> **439** | 27.2K -> 50.1K | 2.80 |

  Raw data: `results/baseline_2xtp4_real.{json,log}`. This client's ITL column is per token (spec-decode bursts spread
  evenly), so it is not comparable to the given 29 ms ITL. Compare TPS/user instead (44-51 here).
- **The original node went bad.** After a few server restarts, every sglang server on it keeps its KFD queues evicted about 90% of the time.
  Details are in "Why the original node was abandoned" below. Continue on a clean node.
- **2026-09-24, node 2 (8x MI350X VF): all targets met** once routing was fixed. The default `cache_aware` router sent 100% of c=64
  traffic to one replica. With `--policy round_robin` and `MAXRUN=64` (details in "Node attempt 2"):

  | conc | TTFT p50 ms (cache_aware, maxrun 48 -> round_robin, maxrun 64) | TTFT p90 ms | out tok/s/GPU | acc.len |
  |---:|---:|---:|---:|---:|
  | 64 | 5,557 -> **317** | 9,400 -> 3,713 | 218 -> **390** | 2.76 |
  | 80 | 7,280 -> **410** | 11,886 -> 4,713 | 306 -> **417** | 2.78 |
  | 128 | 17,170 -> **680** | 21,487 -> 7,555 | 363 -> **469** (3.75K total) | 2.81 |

  GSM8K-500 = 0.878 on the final config. All numbers use `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4`, which corrupts long-context
  tool-call output (`endpoint/ENDPOINT.md`). Raw data: `results/node2_0924/`.

## Node attempt 2 (2026-09-24): MI350X VF node, no PCIe atomics, runs after hostcall workarounds

Node: k3s pod, 8x **"AMD Instinct MI350X VF"**, ROCm 7.0 / sglang 0.5.14 / glibc 2.35 in the pod, no docker. The image runs through
proot (next section). Healthy otherwise: `evicted_ms` stayed 0, weights load in 24.5 s per rank, H2D 57 GB/s, bf16 GEMM 1.14 PFLOP/s,
P2P works between all 8 GPUs. One isolated 7.4K-token prefill on a 66.8K cached prefix takes 235 ms here (213 ms on the MI355X node).

### Hostcall: the node-level blocker and its workarounds

On these VFs **PCIe atomics are off**. The HIP runtime refuses to dispatch any kernel whose code object carries a `hidden_hostcall_buffer`
argument (device printf/assert, or implicit-arg reads the compiler can't analyze). The symptom is `hipErrorIllegalState` from
`hipModuleLaunchKernel`, usually surfacing later as a bare `HIP error` in some other op (first seen in `qr_all_reduce`). Run with `AMD_LOG_LEVEL=1`;
the runtime then logs `rocvirtual.cpp:3650 Pcie atomics not enabled, hostcall not supported` / `AQL dispatch failed!`.
**Preflight on any new node:** start one replica with `AMD_LOG_LEVEL=1` and grep for that line.
To find the kernel: `grep -a -l hidden_hostcall_buffer` over the kernel caches (Triton `/root/.cache/sglang/triton`, aiter `aiter/jit/*.so` and
`flydsl_cache`, sglang JIT `/root/.cache/sglang/jit`).

Four kernels on the M3 path hit this. Each has a fix, and none changes what the kernel computes:

| Kernel | Cause | Fix |
|---|---|---|
| aiter FlyDSL bf16 hgemm (`flydsl_cache/gemm_a16w16_gfx950_*`; the only FlyDSL family with hostcall, MoE kernels are clean) | FlyDSL codegen doesn't mark kernels hostcall-free | `AITER_CONFIG_GEMM_BF16=` a copy of the bf16 tuned table without `flydsl` rows (`nodeenv/aiter_bf16_nohostcall_csv.py`; 1,620 of 3,701 rows dropped, those shapes use aiter's torch fallback) |
| Triton `_index_block_score_only_kernel` (M3 sparse prefill; uses `tl.num_programs`) | implicit-arg read keeps the hostcall slot | patch Triton's AMD backend to add `amdgpu-no-hostcall-ptr` to every kernel that doesn't print (`nodeenv/triton_amd_no_hostcall.patch`, applies to `/opt/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py`; env `TRITON_AMD_NO_HOSTCALL_PTR=0` disables it) |
| sglang JIT `build_tree_kernel_efficient` (EAGLE tree; `printf` warning in `sgl_kernel/speculative/eagle.cuh`) | hostcall printf | `SGLANG_ROCM_NO_HOSTCALL=1` builds sglang JIT kernels with `-mprintf-kind=buffered` (on this branch, `kernels/jit/utils/arch.py`) |
| aiter CK `mha_batch_prefill_fp8bf16_*` (long-prefix extend, `SGLANG_USE_AITER_EXTEND_LONG_PREFIX`) | device `assert()` -> `__assert_fail` | `AITER_NO_HOSTCALL=1` builds aiter JIT modules with `-mprintf-kind=buffered -DNDEBUG` (`nodeenv/aiter_jit_no_hostcall.patch`, apply in `/sgl-workspace/aiter`) |

Delete an already-built module/cache entry after applying a fix so it gets rebuilt. Prebuilt aiter modules that still contain hostcall kernels
(`module_custom`, `module_moe_asm`, `module_moe_ck2stages_*`, `module_topk_plain`, `module_top_k_per_row`, `module_moe_opus`) are not loaded by this
config. If one fails later, delete its `.so` so aiter rebuilds it with `AITER_NO_HOSTCALL=1`. Quick-reduce and custom AR are fine (standalone
4-rank eager + graph test `nodeenv/qr_test.py`).

Server knobs used on this node (both replicas):
`AMD_LOG_LEVEL=1 SGLANG_ROCM_NO_HOSTCALL=1 AITER_NO_HOSTCALL=1 AITER_CONFIG_GEMM_BF16=/scratch/m3/aiter_cfg/bf16_tuned_gemm_nohostcall.csv MAXRUN=64`,
everything else as `serve_m3.sh` (git `2f0da89e4b` plus the `arch.py`/`environ.py` change in the commit that added this section).
Router: `python -m sglang_router.launch_router --worker-urls http://127.0.0.1:30001 http://127.0.0.1:30002 --policy round_robin --port 30000`.
GSM8K-500: 0.892 (maxrun 48) and 0.878 (maxrun 64).

### Results on this node (TOPK_FREQ=4; see the note in TL;DR)

| config | c=64 TTFT p50 / p90 ms, out/s/GPU | c=80 | c=128 |
|---|---|---|---|
| cache_aware router (default), maxrun 48 | 5,557 / 9,400, 218 | 7,280 / 11,886, 306 | 17,170 / 21,487, 363 |
| round_robin, maxrun 48 | 327 / 3,737, 391 | 434 / 4,737, 416 | 5,381 / 9,491, 434 |
| **round_robin, maxrun 64** | **317** / 3,713, 390 | **410** / 4,713, 417 | **680** / 7,555, **469** |
| round_robin, maxrun 64, **TOPK_FREQ=1** (production-safe) | 306 / 4,016, 347 | 389 / 5,053, 372 | 699 / 8,255, 423 |

With TOPK_FREQ=1, TTFT is unchanged and throughput drops about 10% (acc.len 2.67-2.78). Total tokens/min/GPU (input incl. cached + output): 2.39M / 2.57M / 2.92M at c=64/80/128, vs 2.69M / 2.88M / 3.24M with TOPK_FREQ=4.

Files: `results/node2_0924/bench_{baseline,rr,rr_maxrun64,rr_maxrun64_topk1}.{json,log}` and per-replica load every 2 s in `load_*.csv` (`load_sampler.py`).

- **Why the router mattered:** under `cache_aware`, c=64 put all requests on 30002 (48 running + 16 queued) while 30001 sat idle. c=80 gave 30001
  about 8 requests, and c=128 about 32 (30002: 47 running + 48 queued). The defaults `--balance-abs-threshold 64 --balance-rel-threshold 1.5` never
  trigger at these concurrencies, and every request matches the same 90% prefix. The handoff's MI355X baseline table was measured the same
  way and probably has the same imbalance. `round_robin` keeps both replicas at c/2 (the bench warms the prefix on both).
- **MAXRUN=64** removes the per-replica queue at c=128 (64 clients per replica vs 48 slots): p50 5.4 s -> 0.68 s, output +8%.
- TTFT p90 is still 3.7-7.6 s: the first wave of each level is c/2 simultaneous prefills per replica, drained one per forward
  (see "Where TTFT goes"). Chunk size, the fairness reserve and PD disaggregation from that list were not tried; they target p90 now.
- Untried: `power_of_two` or `cache_aware --balance-abs-threshold 1` (load-aware). Round-robin is load-oblivious, but closed-loop load stayed even here.
- Pitfall: rerunning the bench with the same seed after an aborted run inflates `cache%` (95% instead of 90%) and flatters TTFT.
  `POST /flush_cache` on each replica first.

### Setup on a pod without docker, mounts or unshare (proot)

This pod had `CAP_SYS_ADMIN`/`CAP_SYS_CHROOT`, but AppArmor (`cri-containerd.apparmor.d`) blocks `mount` and `unshare`, so chroot has no
`/proc`, `/sys` or `/dev`. 254 of the image's objects need glibc >= 2.38, so they can't run natively on the pod's 2.35 either.
Swapping the pod's glibc (what attempt 1 did) was not needed. Everything below lives under `/scratch/m3` and leaves the pod untouched.

```bash
N=benchmark/minimax_m3_mi355x/rebase_0923/nodeenv   # in a host-side clone of this branch
mkdir -p /scratch/m3/{image,bin,models,data,logs}
python $N/image_pull.py              # pulls the pinned digest from Docker Hub by manifest (26.6 GB, 47 gzip layers)
bash   $N/image_extract.sh           # applies layers with OCI whiteouts into /scratch/m3/image/rootfs (68 GB)
curl -sSL -o /scratch/m3/bin/proot https://proot.gitlab.io/proot/bin/proot && chmod +x /scratch/m3/bin/proot
cp $N/inimg $N/up_host.sh /scratch/m3/bin/
/scratch/m3/bin/inimg python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 2.11.0+rocm7.2 8
```

Then follow "Reproduce on a clean node" steps 1-3 against the rootfs (`/scratch/m3/image/rootfs/sgl-workspace/{sglang,aiter}`; host-side
`git` on those paths is fine). Use `M3_WORK=/scratch/m3` (inimg sets it). Start servers from the host with
`/scratch/m3/bin/up_host.sh 30001 0,1,2,3 [KNOB=...]`, not `inimg ./up.sh`: proot waits for every traced child, so the latter never returns.
Run other tools as `/scratch/m3/bin/inimg python ...`.
`stop_port.sh` works from the host (same PID namespace).

- `inimg` defaults its cwd to `/tmp`. From `/sgl-workspace`, the `sglang/` directory shadows the editable install as a namespace package.
- proot overhead (seccomp fast path active): kernel launch 3.7 us (native 4.5), graph replay identical, `getpid` unchanged, `stat` 19 us (native 0.9).
  Steady-state serving is unaffected. Startup pays about 100 s extra for Python imports, because one tracer process serves all TP ranks.
- `/opt/rocm` in the rootfs is an absolute symlink. Inspect it through inimg (`/opt/rocm-7.2.4/.info/version` = 7.2.4), not from the host.

- Stop servers with `nodeenv/stop_graceful.sh PORT` (SIGTERM to the process group, waits up to 180 s, never SIGKILLs; `stop_port.sh` SIGKILLs after 20 s).
  The router renames itself `sglang::router`; find its PID by that name, not by `launch_router`.

## Relation to the cookbook recipe

`docs/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx` recommends, for MI350X/MI355X, **MXFP8 at `--tp 8`, `--mem-fraction-static 0.80`, no speculative decoding**.
This work deliberately diverges, as requested: the MXFP4 quark checkpoint, EAGLE3 (3 steps / 4 tokens), and 2 x TP4 behind the cache-aware router
instead of TP8. M3 has 4 KV heads, so TP8 replicates KV, and two replicas double the prefill lanes for this burst-bound workload. It also uses mem 0.85 and the
M3-perf env knobs in `serve_m3.sh`. The cookbook TP8 MXFP8 config has not been measured on this workload; it is worth one run as a reference point.

## Where TTFT goes (measured, for the next person optimizing)

- One request's prefill on its own (7.4K new tokens on a 66.8K cached prefix, TP4) takes **213 ms**.
- The closed-loop client plus fixed OSL makes arrivals **synchronized waves** of about c/2 requests per replica. They start together and finish together.
  With `--chunked-prefill-size 8192`, one 7.4K prefill fits per forward pass, so a wave drains serially: burst of 8 = 1.66 s for the last request,
  burst of 32 = 5.0 s for the last request (157 ms/request). That is the TTFT. Decode also stalls during these bursts (ITL p90 about 57 ms), because
  SGLang disables mixed prefill+decode chunks under spec decode.
- At c=128, `--max-running-requests 48` per replica (96 total) makes a third of the requests queue at the server.

Levers, in the order I'd try them:
1. `CHUNK=16384 EXTRA="--max-prefill-tokens 32768"` (two prefills per pass), then 32768/65536. Measure with `burst_ttft.py`.
   A 32768 attempt on the bad node failed its server warmup; that is not yet separated from the node problem.
2. `MAXRUN=64` so c=128 doesn't queue. This captures new graph batch sizes (52-64), so expect a longer startup.
3. `FAIRNESS=0` versus 0.5 (the chunked-prefill fairness reserve splits chunks; it helps p99 but may hurt p50 under bursts).
4. Prefill/decode disaggregation (1 prefill TP4 + 1 decode TP4, the final config of the vLLM MI355X blog). It keeps prefill bursts off the decode path.
   It halves prefill capacity per wave, so check it with `burst_ttft.py` numbers first.
5. `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ` is 4 here (the fast AgentX setting). `endpoint/ENDPOINT.md` shows 4 corrupts tool-call tags past ~60K context.
   Use 1 for anything user-facing; it costs about 11% decode.

## Reproduce on a clean node (8x MI355X, docker available)

```bash
docker run -it --network host --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
  --security-opt seccomp=unconfined -v /scratch:/scratch \
  lmsysorg/sglang-rocm@sha256:02108de8f9418a12425fb56415794fe82d180cb985488725d485e96d97a9a26b bash
```

Inside the container:

```bash
export M3_WORK=/scratch; mkdir -p $M3_WORK/models $M3_WORK/data $M3_WORK/logs
# 1. sglang: the image's editable tree sits at exactly 4e60d70ba4, so check the branch out there (keeps the compiled rust extension)
cd /sgl-workspace/sglang && git remote add kevin https://github.com/kevin-mii/sglang && git fetch kevin M3-perf-rebase \
  && git checkout -b M3-perf-rebase FETCH_HEAD
R=/sgl-workspace/sglang/benchmark/minimax_m3_mi355x
# 2. aiter (image ships acf8fdf9): FlyDSL XCD-swizzle fix, rebased for this aiter, plus the M3 tuned MoE rows
cd /sgl-workspace/aiter && git apply $R/rebase_0923/aiter_acf8fdf9_xcd_swizzle.patch \
  && test "$(grep -c 'XCD remap is a bijection' aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py)" = 2 \
  && cp $R/tuned_fmoe_m3_gfx950.csv aiter/configs/model_configs/minimax_m3_gfx950_perf_tuned_fmoe.csv
# 3. weights (~250 GB) and GSM8K
python -c "from huggingface_hub import snapshot_download as d; d('Inferact/MiniMax-M3-EAGLE3-GQA', local_dir='$M3_WORK/models/MiniMax-M3-EAGLE3-GQA'); d('amd/MiniMax-M3-MXFP4', local_dir='$M3_WORK/models/MiniMax-M3-MXFP4', max_workers=32)"
python -c "from huggingface_hub import hf_hub_download as h; [h('openai/gsm8k', f'main/{s}-00000-of-00001.parquet', repo_type='dataset', local_dir='$M3_WORK/data/gsm8k') for s in ('train','test')]"
# 4. servers, ONE AT A TIME, then the router
cd $R/rebase_0923
./up.sh 30001 0,1,2,3 && ./up.sh 30002 4,5,6,7
setsid nohup python -m sglang_router.launch_router --worker-urls http://127.0.0.1:30001 http://127.0.0.1:30002 \
  --policy cache_aware --host 0.0.0.0 --port 30000 > $M3_WORK/logs/router.log 2>&1 &
# 5. quality gate (expect 0.85-0.89), then the benchmark (about 6 min)
python -m sglang.test.few_shot_gsm8k --port 30001 --num-questions 500 --num-shots 5 --parallel 48
python m3_prefix_bench.py --url http://127.0.0.1:30000 --warm-urls http://127.0.0.1:30001,http://127.0.0.1:30002 \
  --conc 64,80,128 --gpus 8 --out $M3_WORK/logs/bench.json
```

Server config = `serve_m3.sh`, which is the `reproduce.sh real` config: MXFP4, fp8 KV, EAGLE3 GQA 3 steps / 4 draft tokens, custom AR, INT4 quick-reduce,
Gluon sparse prefill, index top-k freq 4, long-prefix extend kernels, breakable prefill graphs, fairness reserve 0.5, mem 0.85, max-running 48.
Knobs are environment variables: `CHUNK MAXRUN MEMFRAC STEPS DRAFT TOPK_FREQ FAIRNESS NOSPEC=1 EXTRA="..."`, e.g.
`./up.sh 30001 0,1,2,3 CHUNK=16384 EXTRA="--max-prefill-tokens 32768"`.

Probes: `single_ttft.py URL` (isolated prefill latency), `burst_ttft.py URL N` (N simultaneous prefills). Stop a server with `./stop_port.sh PORT`.

## What changed on the branch (on top of M3-perf)

- `294049c7d3`: `moe_sorting_small.py` accepts aiter's new `output=` buffer. Newer aiter passes it to `_moe_sorting_impl`, which the branch
  monkeypatches. Without this fix, prefill graph capture dies with `TypeError: unexpected keyword argument 'output'`.
- `22cbbbb1db`: merge of main `4e60d70ba4`, 13 conflicts. The non-trivial resolutions (details in the commit message):
  - `schedule_policy.py`: main's PrefillAdder/prefill-budget refactor taken whole; the fairness reserve re-ported onto it.
  - `topk.py`: the branch's shared-slot gate fold kept alongside main's JIT-grouped-topk append guard; the fold is disabled on main's new `dynamic_expert_bias` path.
  - `minimax_sparse.py`: the Gluon/MSA prefill paths are skipped when HiSparse `loc_mapping` is set.
  - `topk_sparse.py`: the HiSparse slot remap added to both the SUB_K and dense paths.
  - `kvcache.cuh`: main's warp load/store with the branch's ROCm out-of-bounds write guard.
- `rebase_0923/` (this directory): scripts, results, the aiter patch and this handoff.
- Not yet done: flattening the merge into a linear rebase for upstreaming, and a full GSM8K-1000 / needle / 257K check on the merged branch
  (only GSM8K-500 has been run).

## Why the original node was abandoned

Symptoms: weight loading at about 40 MB/s per GPU, prefill graph capture at 100-300 s per shape (versus 71 s for all 58), host-to-device copies at
0.1-0.7 GB/s (53 GB/s when healthy), and kernel launch at 55-140 us (4.7 us when healthy). While any sglang server was up, this hit every
GPU process on the box, including ones on GPUs the server did not use.

Evidence: `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` grows about 9 s per 10 s for every sglang process, from its first seconds,
before weights load. A standalone torch process (1 or 4 GPUs, compute + H2D, with and without fork) shows 0 evictions on the same node. There was no VRAM
leak, no GTT use, no cgroup limit or reclaim, and the PCIe links were Gen5 x16. It started after the first servers were stopped with SIGKILL (and
one run with `TORCHINDUCTOR_COMPILE_THREADS` at default, where 128 inductor workers inherited `/dev/kfd`). Stuck teardowns sat in `D` state in
`synchronize_srcu`. Root cause not found. It needs host access (dmesg, amdgpu driver state) or a reboot/driver reload.
Later on 2026-09-24, even a Qwen2.5-0.5B sglang server on one GPU (no IPC, no custom AR) failed to become healthy within 10 min, and the full M3
server's prefill-graph capture ran at about 2 min per shape. So the problem is node-level, not M3-specific. A GPU reset was impossible from the container
(`/sys` is read-only, no debugfs; the `amd-smi`/`rocm-smi` reset was not attempted because it was blocked pending approval). The node was released for reacquisition.

The container was also unusual: no docker, seccomp blocked unshare, and ROCm 7.0. The latest image was extracted into `/sgl-workspace/scratch/rootfs`
and run natively after upgrading the container's glibc to 2.39 (backup in `/sgl-workspace/scratch/glibc_backup/`). A normal docker node needs none of that.

## Operational rules learned the hard way

- **Start replicas sequentially.** Two simultaneous 243 GB loads collapse H2D bandwidth.
- **Never SIGKILL a live GPU server.** Use `stop_port.sh` (SIGTERM to the process group), then wait until `/sys/class/kfd/kfd/proc/` is empty
  and no sglang process is in `D` state before starting the next one.
- Keep `TORCHINDUCTOR_COMPILE_THREADS=1` (already set in `serve_m3.sh`): inductor's forked compile pool otherwise inherits `/dev/kfd`.
- Health check for a node: `for f in /sys/class/kfd/kfd/proc/*/stats_*/evicted_ms; do cat $f; done` twice, 10 s apart, while a server runs.
  If it grows by seconds, the node is in the bad state.
- `pkill -f <pattern>` from a shell whose own command line contains the pattern kills that shell. Match PIDs by port instead.
