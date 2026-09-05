#!/bin/bash
# Waits for the running hypothesis-6 training
# (controlnet_lora_manga_unetlora_20260905, started 2026-09-05 ~14:05,
# ~15.3h, ETA ~09-06 05:30) to finish, then runs the cs re-evaluation of
# all models. The GPU only has 12GB and the training is using ~6.8GB, so
# these cannot overlap without risking an OOM that would kill a 15h job.
#
# Follows the unattended-chain precedent of
# experiments/run_overnight_20260827.sh, with one addition: that script
# invoked each stage itself, whereas here the first stage is an
# already-detached process, so this polls for its .done marker. A plain
# wait-for-file would hang forever if the training crashed, so the loop
# also watches the training process and aborts if it disappears without
# producing the marker.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_DONE=logs/controlnet_lora_manga_unetlora_20260905.done
TRAIN_PATTERN="train_controlnet_unet_lora.py"
LOG=logs/cs_reeval_chained_20260906.log

mkdir -p logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for $TRAIN_DONE"

while [ ! -f "$TRAIN_DONE" ]; do
  if ! pgrep -f "$TRAIN_PATTERN" > /dev/null; then
    # The training script also runs a short smoke test and then an eval
    # phase, during which no train_controlnet_unet_lora.py process exists
    # briefly. Re-check after a grace period before declaring failure.
    sleep 120
    if [ ! -f "$TRAIN_DONE" ] && ! pgrep -f "$TRAIN_PATTERN" > /dev/null \
       && ! pgrep -f "run_controlnet_lora_manga_unetlora" > /dev/null; then
      echo "ERROR: training process gone and no $TRAIN_DONE -- it failed or was killed." >&2
      echo "Not running the re-eval. Check logs/controlnet_lora_manga_unetlora_20260905.log" >&2
      exit 1
    fi
  fi
  sleep 60
done

echo "[$(date --iso-8601=seconds)] training complete, starting cs re-eval"
bash experiments/run_cs_reeval_20260906.sh
