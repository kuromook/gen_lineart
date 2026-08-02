#!/bin/bash
# Waits for the diffusion-controlnet worktree's active long ControlNet run
# (scripts/train_controlnet.py, PID given below, started 2026-08-03 00:00
# JST via cron, ~18600 steps / ~19h) to finish, then launches this worktree's
# unet_skip0 skip-connection ablation
# (run_combined_koma_unet_skip0_finegrid_20260803.sh). The two jobs share one
# 12GB GPU and cannot run concurrently without OOM risk, so this is a
# sequential hand-off, not a parallel launch.
#
# Meant to be started fully detached (nohup ... & disown) so it survives
# this CLI session ending, matching this project's established pattern for
# multi-hour unattended chains (see doc/work_log.md, 2026-07-29 combined
# koma training launch: a separate detached wrapper waiting on a training
# PID before firing the completion notification).
set -uo pipefail

CONTROLNET_PID=1049695
POLL_INTERVAL=300
GPU_FREE_WAIT_MAX=600
GPU_FREE_THRESHOLD_MIB=3000

LOG=/home/sh1/deepl/lineart-cleanup-refiner/logs/queue_unet_skip0_after_controlnet_20260803.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec >> "$LOG" 2>&1

is_controlnet_running() {
  ps -p "$CONTROLNET_PID" -o cmd= 2>/dev/null | grep -q "train_controlnet.py"
}

echo "[$(date --iso-8601=seconds)] waiting for ControlNet PID $CONTROLNET_PID (train_controlnet.py) to finish, polling every ${POLL_INTERVAL}s"
while is_controlnet_running; do
  sleep "$POLL_INTERVAL"
done
echo "[$(date --iso-8601=seconds)] ControlNet PID $CONTROLNET_PID no longer matches train_controlnet.py"

echo "[$(date --iso-8601=seconds)] waiting for GPU memory to free (threshold ${GPU_FREE_THRESHOLD_MIB} MiB used, max ${GPU_FREE_WAIT_MAX}s)"
waited=0
while [ "$waited" -lt "$GPU_FREE_WAIT_MAX" ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "[$(date --iso-8601=seconds)] gpu memory used: ${used} MiB"
  if [ "${used:-99999}" -lt "$GPU_FREE_THRESHOLD_MIB" ]; then
    break
  fi
  sleep 30
  waited=$((waited + 30))
done

echo "[$(date --iso-8601=seconds)] launching unet_skip0 finegrid run"
cd /home/sh1/deepl/lineart-cleanup-refiner
bash experiments/run_combined_koma_unet_skip0_finegrid_20260803.sh
status=$?
echo "[$(date --iso-8601=seconds)] unet_skip0 finegrid run script exited with code $status"
