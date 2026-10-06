#!/bin/bash
# agentx.sh PORT CONC DURATION_S OUTDIR: AgentX AIPerf client (reproduce.sh bench flags) against one TP4 server; prints a summary line
PORT=$1; C=$2; DUR=$3; out=$4; mkdir -p "$out"
A=/scratch/m3/aiperf-sa-venv/bin/aiperf; MODEL=/scratch/m3/models/MiniMax-M3-MXFP4
AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 AIPERF_FAILED_REQUEST_THRESHOLD=0.10 \
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10 AIPERF_WARMUP_REQUESTS_PER_LANE=10 AIPERF_BENCHMARK_GRACE_PERIOD=30 \
$A profile --scenario inferencex-agentx-mvp --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat \
  --streaming --model MiniMax-M3 --tokenizer $MODEL --tokenizer-trust-remote-code --apply-chat-template \
  --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393 --concurrency "$C" --benchmark-duration "$DUR" \
  --random-seed 42 --use-server-token-count --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --trace-idle-gap-cap-seconds 300 --warmup-requests-per-lane 10 --warmup-grace-period 1800 --failed-request-threshold 0.10 \
  --stats-interval 30 --slice-duration 1.0 --ui simple --output-artifact-dir "$out" > "$out/aiperf_stdout.log" 2>&1
python3 /scratch/m3/bin/agentx_summary.py "$out" | tee "$out/summary.txt"
