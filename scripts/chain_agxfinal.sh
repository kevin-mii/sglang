#!/bin/bash
# AgentX TP4 at c1 and c4, 900 s each: the current best stack, plugin en (6727e41) on SGLang sglang-markov4
S=/scratch/tmp/claude-0/-sgl-workspace/4bb9f594-e169-4b38-90ad-4a5cb2483512/scratchpad
export TREE=/scratch/agentx/sglang-markov4 OOTTREE=/scratch/agentx/sgl-amd-mono-en ARMS=on TP=4
TAG=agxfinal CONC=1 bash $S/oot_val/run_agx2.sh
TAG=agxfinal4 CONC=4 bash $S/oot_val/run_agx2.sh
echo "=== chain_agxfinal DONE $(date -u +%H:%M:%S)" >> $S/oot_val/status
