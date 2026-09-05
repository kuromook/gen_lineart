#!/bin/bash
# Overnight unattended chain (2026-08-27, run with explicit user authorization
# to proceed without approval): candidate #2 (control_v11p_sd15_lineart
# ControlNet init) then candidate #3 (MangaLineExtraction-hf preprocessing),
# from the alternative-model survey in inbox/initial_notice.md. #1 (SDXL
# migration) deliberately excluded -- user wants to do that themselves
# tomorrow.
#
# Each stage is gated on the previous one succeeding (set -e propagates
# through this script since each `bash <subscript>` call's exit status is
# checked). If a stage fails, the chain stops there rather than proceeding
# into a corrupted/meaningless next stage; already-completed stages' results
# remain on disk regardless.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/overnight_20260827.log
mkdir -p logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] overnight chain start: #2 (sd15_lineart) -> #3 preprocess -> #3 (manga_line)"

echo "=== stage 1/3: candidate #2 (control_v11p_sd15_lineart) ==="
bash experiments/run_controlnet_lora_lineartsd15_20260827.sh

echo "=== stage 2/3: candidate #3 preprocessing (manga_line, full dataset) ==="
bash experiments/preprocess_manga_line_full_20260827.sh

echo "=== stage 3/3: candidate #3 (manga_line preprocessing) ==="
bash experiments/run_controlnet_lora_manga_20260827.sh

echo "[$(date --iso-8601=seconds)] overnight chain complete: both candidates trained and evaluated"
/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "Overnight chain complete (candidates #2 and #3)" \
  "Both control_v11p_sd15_lineart and manga_line variants trained and evaluated. Review results/." || true
