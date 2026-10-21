#!/bin/bash
# decode A/B arms on frozen trees, one server each: ARMS="label:oot_tree:on|off ..." with sglang $TREE.
# PRE_CHECK="tree:module[:args]" runs a kernel check (TP4) before the arms.
S=/scratch/tmp/claude-0/-sgl-workspace/4bb9f594-e169-4b38-90ad-4a5cb2483512/scratchpad
OUT=$S/oot_val; TAG=${TAG:-perf}
note() { echo "$*" >> $OUT/status; }
exec 9>/scratch/agentx/gpu5.lock; flock 9
# a server still up under the lock is an orphan: stop rather than test against it
if pgrep -f "sglang.cli.main import main; sys.exit(main()) serve" > /dev/null; then
  note "=== $TAG ABORT $(date -u +%H:%M:%S): an SGLang server is already running"; exit 1
fi
export PATH=$S/bin:$PATH
# REQUIRE: a line the status file must hold first (an earlier step's pass)
if [ -n "${REQUIRE:-}" ] && ! grep -qE "$REQUIRE" $OUT/status; then
  note "=== $TAG skipped $(date -u +%H:%M:%S): no '$REQUIRE' in the status"; exit 0
fi
note "=== $TAG start $(date -u +%H:%M:%S) sglang $(git -C $TREE rev-parse --short HEAD)"
if [ -n "${PRE_CHECK:-}" ]; then
  IFS=: read -r ctree cmod cargs <<< "$PRE_CHECK"
  ( W=$TREE OOT=$ctree source $S/oot_env.sh; export PATH=$S/bin:$PATH HIP_VISIBLE_DEVICES=2,3,4,5; cd /tmp
    timeout 3600 python -m torch.distributed.run --nproc-per-node=4 --master-port=29604 \
      $ctree/test/kernels/deepseek_v41/$cmod.py $cargs > $OUT/$TAG.precheck.log 2>&1 )
  rc=$?; note "$TAG precheck $cmod rc=$rc :: $(grep -aE 'FAIL|ALL PASS|replay' $OUT/$TAG.precheck.log | tail -2 | tr '\n' '|')"
  [ $rc -ne 0 ] && { note "=== $TAG ABORT (precheck)"; exit 1; }
fi
for arm in $ARMS; do
  # label:tree:on|off[:VAR=VALUE,...] -- the fourth field, the arm's own environment
  IFS=: read -r label tree mono envspec <<< "$arm"
  # tree may be oot_tree@sglang_tree to override $TREE for this arm
  IFS=@ read -r tree sgtree <<< "$tree"
  ( W=${sgtree:-$TREE} OOT=$tree OOT_SITE=${OOT_SITE:-$S/oot_site} source $S/oot_env.sh; export PATH=$S/bin:$PATH HIP_VISIBLE_DEVICES=2,3,4,5; cd /tmp
    [ -n "$envspec" ] && export ${envspec//,/ }
    OOT_E2E_DIR=$tree/test/e2e timeout 5400 python $S/oot_val/decode_ab.py $label $mono $EXTRA > $OUT/$TAG.$label.log 2>&1 )
  note "$TAG $label rc=$? :: $(grep -a '^DECODE_AB' $OUT/$TAG.$label.log)"
done
note "=== $TAG DONE $(date -u +%H:%M:%S)"
