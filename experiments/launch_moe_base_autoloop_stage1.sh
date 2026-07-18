#!/bin/bash
# Launch the MoE base autoloop stage1 as a user systemd transient service.
set -euo pipefail
cd "$(dirname "$0")/.."

UNIT=${UNIT:-lineart-moe-base-autoloop-stage1}
LOG=/home/sh1/deepl/lineart/logs/moe_base_autoloop_stage1.service.log

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property=WorkingDirectory=/home/sh1/deepl/lineart \
  --property=StandardOutput=append:"$LOG" \
  --property=StandardError=append:"$LOG" \
  /home/sh1/deepl/lineart/experiments/run_moe_base_autoloop_stage1.sh

echo "launched: $UNIT.service"
echo "status: systemctl --user status $UNIT.service"
echo "log: tail -f $LOG"
