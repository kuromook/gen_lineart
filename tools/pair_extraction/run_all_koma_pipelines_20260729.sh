#!/bin/bash
# Master driver: runs the panel->tile pipeline for ako5ver2, hamlabi, fitness,
# and housei sequentially (gakuen already completed separately as the pilot
# test run with identical parameters), then sends a completion notification.
# Launched fully detached (nohup + disown) so it survives the CLI session
# ending; progress and errors are all in this log file.
set -uo pipefail
cd /home/sh1/deepl/lineart
DATE=20260729

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==== master pipeline run starting ===="

echo "[$(date +%H:%M:%S)] ---- ako5ver2 ----"
bash tools/pair_extraction/run_koma_tile_pipeline.sh \
  ako5ver2 results/ako5ver2_koma_panels_20260728_v2.csv \
  dataset/raw_zips/dataset_ako5_koma_v2.zip "" "$DATE" 1.0

echo "[$(date +%H:%M:%S)] ---- hamlabi ----"
bash tools/pair_extraction/run_koma_tile_pipeline.sh \
  hamlabi results/hamlabi_koma_panels_20260728_v2.csv \
  dataset/raw_zips/dataset_hamlabi_koma_v2.zip "" "$DATE" 1.0

echo "[$(date +%H:%M:%S)] ---- fitness ----"
bash tools/pair_extraction/run_koma_tile_pipeline.sh \
  fitness results/fitness_koma_panels_20260728_v2.csv \
  dataset/raw_zips/dataset_fitness_koma_v2.zip "" "$DATE" 1.0

echo "[$(date +%H:%M:%S)] ---- housei (max-soft-ink-ratio 0.50, established source-specific relaxation) ----"
bash tools/pair_extraction/run_koma_tile_pipeline.sh \
  housei results/housei_koma_panels_20260728_v4.csv \
  dataset/raw_zips/dataset_housei_v4.zip dataset_housei "$DATE" 0.50

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==== master pipeline run finished ===="

SUMMARY=""
for src in ako5ver2 hamlabi fitness gakuen housei; do
  list="dataset/pairs_480/valid_train_${src}_koma_${DATE}.txt"
  if [[ -f "$list" ]]; then
    count=$(wc -l < "$list")
    SUMMARY="${SUMMARY}${src}: ${count} tiles\n"
  else
    SUMMARY="${SUMMARY}${src}: MISSING (check log)\n"
  fi
done

bash experiments/send_autoloop_notification.sh \
  "Lineart koma panel-to-tile pipeline complete" \
  "$(printf '%b' "All 5 sources finished panel-to-tile extraction.\n${SUMMARY}")"
