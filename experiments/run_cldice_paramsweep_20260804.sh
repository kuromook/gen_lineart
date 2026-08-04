#!/bin/bash
# Runs the two clDice follow-up variants sequentially (GPU is free now,
# no prior job to wait on -- see run_after_selfanchor_queue_cldice_20260804.sh
# for the wait-then-launch pattern used when a prior job was still running).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/queue_cldice_paramsweep_20260804.log
mkdir -p logs
: > "$LOG"

echo "[$(date --iso-8601=seconds)] starting cldice lightweight run" >> "$LOG"
bash experiments/run_combined_koma_unet_skip0_cldice_lightweight_20260804.sh >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] lightweight run done, starting cldice inkpenalty run" >> "$LOG"
bash experiments/run_combined_koma_unet_skip0_cldice_inkpenalty_20260804.sh >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] both cldice param-sweep runs complete" >> "$LOG"
