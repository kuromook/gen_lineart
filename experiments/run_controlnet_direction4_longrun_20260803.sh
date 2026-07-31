#!/bin/bash
# Direction 4 long training run to test the ControlNet "sudden convergence
# phenomenon" hypothesis: the first 10-epoch/1860-step run (2026-07-31)
# showed the SD1.5 anime prior still dominating over the rough conditioning
# (crisp binary ink, but hallucinated content unrelated to the input rough;
# an inference-time conditioning-scale/guidance-scale sweep did not fix
# it). This run goes 10x further (18600 steps, ~19h at the observed
# ~3.7s/step) to see whether the conditioning influence sharpens once
# training progresses further.
#
# Scheduled via crontab for Monday 2026-08-03 00:00 JST per user request
# (weekdays free for long GPU runs, weekends reserved for other PC use).
# Safe to re-run manually: --resume-from-checkpoint latest continues from
# the last saved state instead of restarting, so this script also works to
# extend the run on a later weekday if 18600 steps isn't enough.
set -euo pipefail
cd "$(dirname "$0")/.."

# Self-remove the one-shot crontab entry now that it has fired.
crontab -l 2>/dev/null | grep -v CONTROLNET_LONGRUN_20260803_MARKER | crontab - || true

PY=./venv/bin/python
TAG=controlnet_koma_direction4_longrun_20260803
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

echo "[$(date --iso-8601=seconds)] starting/resuming ${TAG}" >> "$LOG"

"$PY" scripts/train_controlnet.py \
  --output-dir "checkpoints/${TAG}" \
  --batch-size 2 \
  --grad-accum 4 \
  --max-train-steps 18600 \
  --save-steps 300 \
  --eval-snapshot-steps 1860 \
  --log-steps 100 \
  --num-workers 4 \
  --resume-from-checkpoint latest \
  >> "$LOG" 2>&1

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "max_train_steps=18600"
  echo "output_dir=checkpoints/${TAG}"
  echo "log=$LOG"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart Direction 4 long run complete" \
  "18600-step ControlNet run finished. Check ${LOG} and run scripts/infer_controlnet.py against checkpoints/${TAG}/final to see if conditioning influence sharpened." || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] ${TAG} complete"
