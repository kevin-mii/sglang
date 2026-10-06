#!/bin/bash
# stop_graceful.sh PORT: SIGTERM the server's process group, wait up to 180 s, never SIGKILL
pid=$(ps -eo pid,args | awk -v p="--port $1" 'index($0, "sglang.launch_server") && index($0, p) && !index($0, "awk") && !index($0, "proot") {print $1; exit}')
[ -z "$pid" ] && { echo "no server on $1"; exit 0; }
pgid=$(ps -o pgid= -p $pid | tr -d ' '); kill -TERM -- -$pgid
for i in $(seq 1 180); do pgrep -g $pgid >/dev/null || { echo "stopped $1 after ${i}s"; exit 0; }; sleep 1; done
echo "WARNING: $1 still alive after 180 s (pgid $pgid)"; ps -o pid,stat,args -g $pgid | cut -c1-100
