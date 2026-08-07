#!/bin/bash
# Detached watcher: waits for the rough-domain WD14 tagging pass
# (logs/tag_wd14_rough_20260807.log, ~2.6h for 2,014 images) to finish,
# then launches run_domain_lora_rough_sd15base_sksv2_20260807.sh
# immediately with no interactive gate in between -- same pattern as
# experiments/run_domain_lora_chain_roughclean_then_roughfull_20260805.sh.
#
# Safety: only launches training if the tags CSV has the expected row
# count (2014 images + 1 header = 2015 lines); if the tagging process
# exits without producing a complete CSV, this logs the failure and does
# not launch training on an incomplete caption set.
set -uo pipefail
cd "$(dirname "$0")/.."

TAGS_CSV="results/domain_lora_rough_captiontags_20260807/tags.csv"
EXPECTED_LINES=2015
WATCH_PROC_PATTERN="tag_wd14.py.*rough_domain_filelist"
CHAIN_LOG="logs/domain_lora_chain_roughtag_then_train_20260807.log"

mkdir -p logs
echo "[$(date --iso-8601=seconds)] chain watcher started (pid $$), waiting for tagging to finish" >> "$CHAIN_LOG"

while true; do
  if ! pgrep -f "$WATCH_PROC_PATTERN" > /dev/null; then
    lines=$(wc -l < "$TAGS_CSV" 2>/dev/null || echo 0)
    if [ "$lines" -eq "$EXPECTED_LINES" ]; then
      echo "[$(date --iso-8601=seconds)] tagging process gone, $TAGS_CSV has $lines lines (expected) -- launching training" >> "$CHAIN_LOG"
      bash experiments/run_domain_lora_rough_sd15base_sksv2_20260807.sh >> "$CHAIN_LOG" 2>&1
      echo "[$(date --iso-8601=seconds)] chain complete" >> "$CHAIN_LOG"
      exit 0
    else
      echo "[$(date --iso-8601=seconds)] tagging process gone but $TAGS_CSV has $lines lines (expected $EXPECTED_LINES) -- treating as failed/incomplete, NOT launching training. Review logs/tag_wd14_rough_20260807.log manually." >> "$CHAIN_LOG"
      exit 1
    fi
  fi
  sleep 60
done
