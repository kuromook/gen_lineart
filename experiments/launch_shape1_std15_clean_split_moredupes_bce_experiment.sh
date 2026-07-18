#!/bin/bash
# Launch experiment 1 as a user systemd transient service.
set -euo pipefail
cd "$(dirname "$0")/.."

UNIT=${UNIT:-lineart-exp1-moredupes-bce}
LOG=/home/sh1/deepl/lineart/logs/train_shape1_std15_clean_split_moredupes_bce.service.log

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property=WorkingDirectory=/home/sh1/deepl/lineart \
  --property=StandardOutput=append:"$LOG" \
  --property=StandardError=append:"$LOG" \
  /home/sh1/deepl/lineart/experiments/run_shape1_std15_clean_split_moredupes_bce_experiment.sh

echo "launched: $UNIT.service"
echo "status: systemctl --user status $UNIT.service"
echo "log: tail -f $LOG"
