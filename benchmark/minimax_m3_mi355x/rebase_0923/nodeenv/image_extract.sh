#!/bin/bash
# apply image layers in order into rootfs, honoring OCI whiteouts
set -e
cd ${M3_IMAGE:-/scratch/m3/image}; R=$PWD/rootfs; mkdir -p $R
for d in $(python3 -c "import json;print(' '.join(l['digest'].split(':')[1] for l in json.load(open('manifest.json'))['layers']))"); do
  f=blobs/$d
  # whiteouts first
  tar -tzf $f | grep -E '(^|/)\.wh\.' | while read -r w; do
    dir=$(dirname "$w"); base=$(basename "$w")
    if [ "$base" = ".wh..wh..opq" ]; then find "$R/$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    else rm -rf "$R/$dir/${base#.wh.}"; fi
  done
  tar -xzpf $f -C $R --numeric-owner --exclude='.wh.*' --exclude='*/.wh.*'
  echo "applied $d"
done
echo EXTRACTED
