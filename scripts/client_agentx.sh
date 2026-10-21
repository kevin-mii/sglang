# AgentX MVP client, the ATOM DeepSeek-V4.1-Flash-Agentic recipe's client block
# (github.com/ROCm/ATOM/recipes/DeepSeek-V4.1-Flash-Agentic.md) with the duration
# set by DUR. Identical flags to the InferenceX srt_agentic.sh build_replay_cmd.
# Usage: bash client_agentx.sh <CONC> <DUR> <tag>
set -u
CONC=$1; DUR=$2; TAG=$3
M=/scratch/models/DeepSeek-V4.1-Flash
export HF_HOME=/scratch/agentx/hf
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
exec /scratch/agentx/aiperf-venv/bin/aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:30000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$M" --tokenizer "$M" --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration "$DUR" \
  --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 5 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:${METRICS_PORT:-30000}/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --ui simple \
  --output-artifact-dir /scratch/agentx/run/$TAG
