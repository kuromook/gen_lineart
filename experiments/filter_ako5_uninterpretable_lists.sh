#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
EXCLUDE=${EXCLUDE:-dataset/pairs_480/agreement_low_repair_categories/agreement_low_ako5_uninterpretable_rough_rebuild_or_exclude.txt}
TAG=${TAG:-no_ako5_uninterp}

filter_one() {
  local input=$1
  local stem=${input%.txt}
  local output=${stem}_${TAG}.txt
  local removed=${stem}_${TAG}_removed.txt
  "$PY" tools/evaluation/filter_pair_list.py \
    --input-list "$input" \
    --exclude-list "$EXCLUDE" \
    --output-list "$output" \
    --removed-out "$removed"
}

filter_one dataset/pairs_480/valid_train_std15.txt
filter_one dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt
filter_one dataset/pairs_480/valid_train_milddup800_clean.txt
filter_one dataset/pairs_480/valid_train_ako5_clean.txt
filter_one dataset/pairs_480/valid_train_warm_plan1.txt
filter_one dataset/pairs_480/valid_train_warm_plan2.txt
