#!/bin/bash
# After generation: tile -> normalize -> measure -> faces -> read, as registered
# in doc/prereg_closeup_20261106.md (the third pool's path, unchanged).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/sh1/deepl/lineart/venv/bin/python
D=data/external_standard
O=results/closeup_20261106
mkdir -p $O
for m in animagine_xl_31 illustrious_xl_v01; do
  $P tools/external_standard/build_external_tiles.py --src $D/gen_${m}_closeup --out $D/tiles_${m}_closeup > $O/tiles_${m}.log 2>&1
  $P tools/external_standard/prepare_normalized.py --src $D/tiles_${m}_closeup --out $D/norm_${m}_closeup > $O/norm_${m}.log 2>&1
done
$P tools/evaluation/measure_lineart_profile.py \
  --source e6c=$D/norm_animagine_xl_31_closeup \
  --source i6c=$D/norm_illustrious_xl_v01_closeup \
  --output-csv $O/profile_closeup.csv > $O/profile_run.log 2>&1
$P tools/external_standard/face_scale_probe.py --out $O --pools e6c,i6c --csv face_px_est_closeup.csv --report face_closeup.txt > $O/face_run.log 2>&1
$P tools/external_standard/closeup_angle.py > $O/read_run.log 2>&1
echo done
