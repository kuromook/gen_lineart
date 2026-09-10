#!/bin/bash
# Single entry point for the batch prepared 2026-09-07/08 while the GPU was
# occupied by the sibling SDXL track (lineart-controlnet-sdxl-fidelity,
# train_controlnet_sdxl.py, ETA Wednesday). Runs, in order:
#   (A) run_grey_source_cross_sd15_20260908.sh  -- cheap, inference-only,
#       ~tens of minutes. Run first per doc/initial_notice.md's "Track Bから
#       の申し送り": resolving whether the grey residual is ControlNet- or
#       preprocessor-attributable is much cheaper than the training sweep,
#       and its answer may reshape how (B) is read.
#   (B) run_consistency_weight_sweep_20260908.sh -- the actual priority #1
#       from doc/initial_notice.md, ~44h across 4 new training runs.
#
# Safety guard: polls for the sibling track's training process before
# starting, so this is safe to launch a bit early without risking two jobs
# fighting over the single 12GB GPU (same precedent as
# experiments/run_cs_reeval_chained_20260906.sh, which did the same for an
# in-track job via its .done marker -- this one is a different track's
# process with no shared .done marker to poll, so it checks the process
# pattern directly instead).
#
# This script is NOT run as part of the 2026-09-07/08 prep -- launch it only
# once the GPU is actually expected to be free, via:
#   nohup ./experiments/run_wednesday_batch_20260908.sh \
#     > logs/wednesday_batch_20260908.log 2>&1 & disown
# then verify detachment with `ps -o ppid= -p $!` (expect 1).
set -euo pipefail
cd "$(dirname "$0")/.."

OTHER_TRACK_PATTERN="train_controlnet_sdxl.py"
LOG=logs/wednesday_batch_20260908.log

mkdir -p logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] checking for other track's GPU job ($OTHER_TRACK_PATTERN)"
while pgrep -f "$OTHER_TRACK_PATTERN" > /dev/null; do
  echo "  still running, waiting 60s"
  sleep 60
done
echo "[$(date --iso-8601=seconds)] GPU appears free, starting"

echo "=== (A) grey-source cross (SD1.5, 2x3) ==="
bash experiments/run_grey_source_cross_sd15_20260908.sh

echo "=== (B) consistency_weight sweep ==="
bash experiments/run_consistency_weight_sweep_20260908.sh

echo "[$(date --iso-8601=seconds)] Wednesday batch complete: both (A) and (B) done"
