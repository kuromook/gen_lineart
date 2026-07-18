#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

UNIT=${UNIT:-lineart-milddup800-ft}
LOG=/home/sh1/deepl/lineart/logs/train_shape1_clean_split_bce_milddup800_ft10_lr1e5.service.log

systemd-run --user --unit="$UNIT" --collect \
  --property=WorkingDirectory=/home/sh1/deepl/lineart \
  --property=StandardOutput=append:"$LOG" \
  --property=StandardError=append:"$LOG" \
  /home/sh1/deepl/lineart/experiments/run_shape1_clean_split_bce_milddup800_ft_experiment.sh
