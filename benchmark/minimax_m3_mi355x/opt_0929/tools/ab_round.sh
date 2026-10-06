#!/bin/bash
# one lever, side by side at c=48, hicache 150 GB on both arms: ab_round.sh TAG "ENV_B..."
TAG=$1; shift; C=${CONC:-48}; H="--enable-hierarchical-cache --hicache-size 150 --radix-eviction-policy lru"; R=/scratch/m3/results/opt0929
timeout 900 python3 /scratch/m3/bin/drop_page_cache.py /scratch /tmp /root >/dev/null 2>&1
TREE=/sgl-workspace/opt-combined /scratch/m3/bin/variant_sweep.sh 30001 0,1,2,3 ${TAG}_base "$C" EXTRA="$H" &
sleep 420
TREE=/sgl-workspace/opt-combined /scratch/m3/bin/variant_sweep.sh 30002 4,5,6,7 ${TAG}_lever "$C" "$@" EXTRA="$H" &
wait; echo DONE > $R/${TAG}.done
