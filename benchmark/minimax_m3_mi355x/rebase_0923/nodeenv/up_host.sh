#!/bin/bash
# up_host.sh PORT GPUS [VAR=value ...]: host-side twin of rebase_0923/up.sh; runs serve_m3.sh inside the image via proot
PORT=$1; GPUS=$2; shift 2
D=/sgl-workspace/sglang/benchmark/minimax_m3_mi355x/rebase_0923; M3_WORK=/scratch/m3
(setsid nohup "$(dirname "$0")"/inimg env GPUS=$GPUS PORT=$PORT "$@" bash $D/serve_m3.sh > $M3_WORK/logs/nohup_$PORT.out 2>&1 &)
L=$M3_WORK/logs/server_$PORT.log; sleep 5; t0=$(date +%s)
until curl -sf -m 2 localhost:$PORT/health >/dev/null; do
  if grep -qE "kill_process_tree called|OutOfMemoryError|Initialization failed|Scheduler hit an exception" "$L" 2>/dev/null; then echo "$PORT FAILED, see $L"; exit 1; fi; sleep 5
done
echo "$PORT UP $(date +%T) after $(( $(date +%s)-t0 ))s"; grep "max_total_num_tokens" "$L" | cut -c1-200
