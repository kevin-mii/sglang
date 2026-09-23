#!/bin/bash
# up.sh PORT GPUS [VAR=value ...]: start one TP4 replica with serve_m3.sh and block until /health is 200.
# Start replicas one after another (concurrent 243 GB loads collapse host-to-device bandwidth).
PORT=$1; GPUS=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd); : "${M3_WORK:=/scratch}"; mkdir -p "$M3_WORK/logs"
(setsid nohup env GPUS=$GPUS PORT=$PORT "$@" bash "$HERE/serve_m3.sh" > "$M3_WORK/logs/nohup_$PORT.out" 2>&1 &)
L=$M3_WORK/logs/server_$PORT.log; sleep 5
until curl -sf -m 2 localhost:$PORT/health >/dev/null; do
  if grep -qE "kill_process_tree called|OutOfMemoryError|Initialization failed" "$L" 2>/dev/null; then echo "$PORT FAILED, see $L"; exit 1; fi; sleep 5
done
echo "$PORT UP $(date +%T)"; grep "max_total_num_tokens" "$L" | cut -c1-160
