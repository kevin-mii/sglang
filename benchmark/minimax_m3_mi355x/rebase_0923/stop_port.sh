#!/bin/bash
# kill the sglang server listening on port $1 (its whole process group)
pid=$(ps -eo pid,args | awk -v p="--port $1" 'index($0, "sglang.launch_server") && index($0, p) && !index($0, "awk") {print $1; exit}')
[ -z "$pid" ] && { echo "no server on $1"; exit 0; }
pgid=$(ps -o pgid= -p $pid | tr -d ' ')
kill -TERM -- -$pgid 2>/dev/null; for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 -- -$pgid 2>/dev/null
echo "stopped server on $1 (pgid $pgid)"
