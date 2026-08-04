#!/bin/bash
# Waits for the self-anchor training script (run_combined_koma_selfanchor_ep25_msgan_20260804.sh)
# to exit and GPU memory to free up, then launches the clDice fine-grid
# probe, matching this project's established queued-multi-hour-job
# hand-off pattern (see run_after_controlnet_longrun_queue_unet_skip0_20260803.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

SELFANCHOR_PID=1165274
LOG=logs/queue_cldice_after_selfanchor_20260804.log
mkdir -p logs
: > "$LOG"

echo "[$(date --iso-8601=seconds)] waiting for self-anchor PID ${SELFANCHOR_PID} to finish, polling every 300s" >> "$LOG"
while kill -0 "$SELFANCHOR_PID" 2>/dev/null; do
  sleep 300
done
echo "[$(date --iso-8601=seconds)] self-anchor PID exited" >> "$LOG"

waited=0
while [ "$waited" -lt 600 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-99999}" -lt 3000 ]; then
    echo "[$(date --iso-8601=seconds)] GPU memory freed (${used} MiB used), launching cldice run" >> "$LOG"
    break
  fi
  sleep 15
  waited=$((waited + 15))
done

bash experiments/run_combined_koma_unet_skip0_cldice_finegrid_20260804.sh >> "$LOG" 2>&1
