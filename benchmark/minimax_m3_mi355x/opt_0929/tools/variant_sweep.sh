#!/bin/bash
# variant_sweep.sh PORT GPUS TAG "CONCS" [KNOB=... ...] (TREE=sglang checkout): (re)start a server with knobs, GSM8K gate, AgentX at CONCS
PORT=$1; GPUS=$2; TAG=$3; CONCS=$4; shift 4
R=/scratch/m3/results/opt0929; L=/scratch/m3/logs/server_$PORT.log
/scratch/m3/bin/stop_graceful.sh $PORT
until [ $(ls /sys/class/kfd/kfd/proc/ | wc -l) -le 7 ]; do sleep 3; done
[ -f $L ] && mv $L /scratch/m3/logs/server_${PORT}_before_$TAG.log
(setsid nohup python3 /scratch/m3/bin/drop_weight_cache.py /scratch/m3/models/MiniMax-M3-MXFP4 /scratch/m3/models/MiniMax-M3-EAGLE3-GQA > /dev/null 2>&1 &)
K=$(cat /scratch/m3/bin/knobs.env)
INIMG_BINDS="-b ${TREE:-/sgl-workspace/opt-combined}:/tree_comb" /scratch/m3/bin/up_host.sh $PORT $GPUS $K PYTHONPATH=/tree_comb/python SGLANG_MINIMAX_M3_INDEXER_CP=1 "$@" || exit 1
/scratch/m3/bin/inimg python -m sglang.test.few_shot_gsm8k --port $PORT --num-questions 500 --num-shots 5 --parallel 48 2>&1 | grep -E "^Accuracy" > $R/${TAG}_gsm8k.txt
for c in $CONCS; do
  /scratch/m3/bin/inimg bash /scratch/m3/bin/agentx.sh $PORT $c 900 $R/${TAG}_c$c
done
echo DONE > $R/${TAG}_sweep.done
