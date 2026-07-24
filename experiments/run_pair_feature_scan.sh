#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-pair_feature_scan_current_no_ako5_badrough}
PY=${PY:-./venv/bin/python}

"$PY" tools/evaluation/score_pair_features.py \
  --list base_unique=dataset/pairs_480/valid_train_base_clean_unique.txt \
  --list milddup800=dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt \
  --list kurip_f1_cham_ink=dataset/pairs_480/valid_train_kurip_strict_f1_cham_ink.txt \
  --list kurip_precision=dataset/pairs_480/valid_train_kurip_strict_precision.txt \
  --list ako5=dataset/pairs_480/valid_train_ako5_clean_no_ako5_badrough.txt \
  --output-csv "results/${TAG}.csv" \
  --summary-csv "results/${TAG}_summary.csv" \
  --split-dir "dataset/pairs_480/${TAG}_splits"
