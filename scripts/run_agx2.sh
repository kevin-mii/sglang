#!/bin/bash
# AgentX at TP $TP, concurrency $CONC for 900 s, monokernels off then on (ARMS), on sglang $TREE with
# sgl-amd-mono $OOTTREE, forced acceptance as the AgentX contract sets it; GPUs 2-5
S=/scratch/tmp/claude-0/-sgl-workspace/4bb9f594-e169-4b38-90ad-4a5cb2483512/scratchpad
OUT=$S/oot_val; TAG=${TAG:-agx}; CONC=${CONC:-1}; export TP=${TP:-4}
note() { echo "$*" >> $OUT/status; }
exec 9>/scratch/agentx/gpu5.lock; flock 9
if pgrep -f "sglang.cli.main import main; sys.exit(main()) serve|sglang.launch_server" > /dev/null; then
  note "=== $TAG ABORT $(date -u +%H:%M:%S): an SGLang server is already running"; exit 1
fi
bash /scratch/agentx/preflight.sh > $OUT/$TAG.preflight.log 2>&1 || { note "=== $TAG ABORT: aiperf preflight failed"; exit 1; }
note "=== $TAG start $(date -u +%H:%M:%S) sglang $(git -C $TREE rev-parse --short HEAD) oot $(git -C $OOTTREE rev-parse --short HEAD) tp$TP c$CONC"
for arm in ${ARMS:-off on}; do
  E=""; [ "$arm" = "on" ] && E="--monokernel-backend amd"
  srv=$OUT/$TAG.srv-$arm.log
  SGLTREE=$TREE OOT=$OOTTREE OOT_SITE=${OOT_SITE:-$S/oot_site_plugin} EXTRA="$E" MAXRUN=${MAXRUN:-32} \
  SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
  DEEPSEEK_V41_FLASH_MODEL_PATH=/scratch/models/DeepSeek-V4.1-Flash HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid nohup bash $S/oot_val/serve_agx.sh > $srv 2>&1 &
  SP=$!
  for i in $(seq 1 360); do grep -q 'The server is fired up' $srv && break; kill -0 $SP 2>/dev/null || break; sleep 10; done
  if grep -q 'The server is fired up' $srv; then
    run=$TAG-$arm-tp$TP-c$CONC
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash /scratch/agentx/client_agentx.sh $CONC 900 $run > $OUT/$TAG.cli-$arm.log 2>&1
    res=$(python3 - "/scratch/agentx/run/$run/profile_export_aiperf.json" "$TP" <<'PY'
import json, sys
try:
    j = json.load(open(sys.argv[1]))
    p90, p50 = j["inter_token_latency"]["p90"], j["inter_token_latency"]["p50"]
    ttt = j["total_token_throughput"]["avg"] / int(sys.argv[2])
    print(f"TTT/GPU {ttt:.0f}, P90 X {1000 / p90:.1f} tok/s/user, P50 {1000 / p50:.1f}, requests {int(j['request_count']['avg'])}")
except Exception as e:
    print(f"no result: {e}")
PY
)
    accept=$(grep -a -o 'accept len: [0-9.]*' $srv | tail -200 | awk '{s+=$3; n++} END {if (n) printf "%.2f", s/n}')
    runs=$(grep -a -o 'monokernel [a-z0-9_]* on gfx[0-9]* runs [^;]*' $srv | sort -u | tr '\n' '|')
    note "$TAG $arm :: $res; accept len (last 200 logs) $accept; ${runs:-no monokernel runs line}"
  else
    note "$TAG $arm :: BOOT FAILED (see $srv)"
  fi
  kill -TERM -- -$SP 2>/dev/null; sleep 20; kill -KILL -- -$SP 2>/dev/null; sleep 5
done
note "=== $TAG DONE $(date -u +%H:%M:%S)"
