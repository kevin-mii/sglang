#!/bin/bash
# the DSpark PR branch on main f8fd41c36e with the plugin shimmed for SGLang's kernel regroup: e2e, then ABAB vs main
S=/scratch/tmp/claude-0/-sgl-workspace/4bb9f594-e169-4b38-90ad-4a5cb2483512/scratchpad
OUT=$S/oot_val; TAG=dsp3; P=/scratch/agentx/sglang-dspark-pr; MAIN=/scratch/agentx/sglang-main2; IW=/scratch/agentx/sgl-amd-mono-iwm
note() { echo "$*" >> $OUT/status; }
exec 9>/scratch/agentx/gpu5.lock; flock 9
note "=== $TAG start $(date -u +%H:%M:%S) sglang $(git -C $P rev-parse --short HEAD) oot $(git -C $IW rev-parse --short HEAD)"
( W=$P OOT=$IW OOT_SITE=$S/oot_site_plugin source $S/oot_env.sh
  export PATH=$S/bin:$PATH HIP_VISIBLE_DEVICES=2,3,4,5 HF_ENDPOINT=http://127.0.0.1:9; cd $IW
  timeout 5400 python -m pytest -q -p no:cacheprovider -rP test/e2e/test_deepseek_v41_server.py > $OUT/$TAG.e2e.log 2>&1
  rc=$?; git checkout -q -- tmp_output_gsm8k.txt 2>/dev/null
  echo "$TAG e2e rc=$rc :: $(grep -aE '[0-9]+ (passed|failed)' $OUT/$TAG.e2e.log | tail -1)" >> $OUT/status
  exit $rc ); rc=$?
flock -u 9
if [ $rc -eq 0 ]; then
  export TREE=$MAIN AB_ROUNDS=5
  TAG=dspab CONCURRENCY=1,2,4,8 AB_TOKENS=3000 ARMS="main:$IW@$MAIN:on dsp:$IW@$P:on mainb:$IW@$MAIN:on dspb:$IW@$P:on" OOT_SITE=$S/oot_site_plugin $OUT/run_perf.sh
fi
echo "=== chain_dsp3 DONE $(date -u +%H:%M:%S)" >> $OUT/status
