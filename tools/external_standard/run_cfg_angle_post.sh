#!/bin/bash
# After generation: tile -> normalize -> measure -> read, exactly as the third pool did.
# Registered in doc/prereg_generation_setting_cfg_20261103.md (Arms).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=/home/sh1/deepl/lineart/venv/bin/python
D=data/external_standard
for t in cfg4p0 cfg9p0; do
  $P tools/external_standard/build_external_tiles.py --src $D/gen_animagine_xl_31_$t --out $D/tiles_animagine_xl_31_$t
  $P tools/external_standard/prepare_normalized.py --src $D/tiles_animagine_xl_31_$t --out $D/norm_animagine_xl_31_$t
done
mkdir -p results/cfg_angle_20261103
$P tools/evaluation/measure_lineart_profile.py \
  --source e6_remeasure=$D/norm_animagine_xl_31_seed2 \
  --source e4=$D/norm_animagine_xl_31_cfg4p0 \
  --source e9=$D/norm_animagine_xl_31_cfg9p0 \
  --source s1=$D/norm_animagine_xl_31 \
  --output-csv results/cfg_angle_20261103/profile_cfg.csv > results/cfg_angle_20261103/profile_run.log 2>&1
$P tools/external_standard/cfg_angle.py
