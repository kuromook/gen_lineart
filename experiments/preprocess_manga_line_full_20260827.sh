#!/bin/bash
# Full-dataset preprocessing for candidate #3 (MangaLineExtraction-hf).
# Must run with the isolated venv (see tools/preprocess_manga_line_extraction_
# condition.py docstring for why -- the shared ../lineart/venv's transformers
# is too new for this repo's custom modeling code).
set -euo pipefail
cd "$(dirname "$0")/.."

MLE_PY=/tmp/claude-1000/-home-sh1-deepl-lineart-controlnet-realpairs/bc420faf-b7d1-49e0-8c69-c5d74a386d6d/scratchpad/mle_venv/bin/python
LOG=logs/preprocess_manga_line_full_20260827.log

mkdir -p logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] manga_line full preprocessing start"
"$MLE_PY" tools/preprocess_manga_line_extraction_condition.py \
  --file-list data/train_list.txt \
  --rough-dir data/rough \
  --output-dir data/rough_manga_line

WRITTEN=$(find data/rough_manga_line -type f | wc -l)
EXPECTED=$(wc -l < data/train_list.txt)
echo "written=$WRITTEN expected=$EXPECTED"
if [ "$WRITTEN" -ne "$EXPECTED" ]; then
  echo "ERROR: file count mismatch, aborting chain" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] manga_line full preprocessing complete"
