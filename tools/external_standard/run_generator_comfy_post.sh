#!/bin/bash
# After ComfyUI generation: tile -> normalize -> measure -> read, as the third pool did.
# Registered in doc/prereg_generator_comfy_20261107.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/sh1/deepl/lineart/venv/bin/python
D=data/external_standard
O=results/generator_comfy_20261107
mkdir -p $O
for m in animagine_xl_31 illustrious_xl_v01; do
  $P tools/external_standard/build_external_tiles.py --src $D/gen_${m}_comfy --out $D/tiles_${m}_comfy > $O/tiles_${m}.log 2>&1
  $P tools/external_standard/prepare_normalized.py --src $D/tiles_${m}_comfy --out $D/norm_${m}_comfy > $O/norm_${m}.log 2>&1
done
$P tools/evaluation/measure_lineart_profile.py \
  --source ec=$D/norm_animagine_xl_31_comfy \
  --source ic=$D/norm_illustrious_xl_v01_comfy \
  --source s1e=$D/norm_animagine_xl_31 \
  --source s1i=$D/norm_illustrious_xl_v01 \
  --output-csv $O/profile_comfy.csv > $O/profile_run.log 2>&1
$P tools/external_standard/generator_angle.py > $O/read_run.log 2>&1
echo done
