#!/bin/bash
# Exhaustive caption-word ablation sweep (inbox/initial_notice.md,
# 2026-09-05): after "manga panel" removal (hypothesis #5) made
# orientation_entropy *worse* rather than better, inventory every other
# tag appearing in >= 0.5% of data/captions.csv's training rows (63 tags,
# results/word_ablation_20260905/candidate_tags.py) by generating diag5
# outputs with each tag individually prepended to the plain baseline
# caption, holding the manga_nomangaword_trained checkpoint fixed. No
# retraining involved -- this is inference-only, ~30 min total, to find
# which single words (if any) most reintroduce cross-hatch hallucination
# when present in the prompt.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=word_ablation_sweep_20260905
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

mkdir -p logs

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] word ablation sweep start"
"$PY" results/word_ablation_20260905/generate_sweep.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "output_dir=results/word_ablation_20260905/outputs"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "Word ablation sweep complete" \
  "Review results/word_ablation_20260905/outputs/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] word ablation sweep complete"
