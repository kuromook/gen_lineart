#!/bin/bash
# Track J stage 1: materialise the two model arms on the held-out fill pools.
# Pre-registered in doc/work_log.md 2026-10-09. No training happens here.
#
# The msgan arm is not a one-step inference: the adopted checkpoint has
# in_channels=2 and needs the lucy_mild aux channel, so this reproduces the
# three stages of run_combined_koma_lucy_mild_msgan_20260729.sh exactly
# (atari aux -> preprocess_atari_aux --mode lucy_mild -> inference with
# --aux-dir), all with --autocontrast as that run used. Running the checkpoint
# single-channel would be a different model.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
D=/home/sh1/deepl/lineart/dataset/pairs_480
C=/home/sh1/deepl/lineart/checkpoints
R=results/baseline_fill_20261009
ATARI=$C/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth
MSGAN=$C/combined_koma_lucy_mild_msgan_20260729/best.pth

run_pool () {
  local name=$1 split=$2
  local L=$R/lists/$name.txt
  echo "=== [$name] split=$split n=$(wc -l < "$L") ==="
  echo "--- preprocessor (LineartAnimeDetector) ---"
  $PY tools/pair_extraction/preprocess_lineart_anime_condition.py \
    --file-list "$L" --rough-dir "$D/$split/rough" --output-dir "$R/preproc/$name"
  echo "--- atari aux ---"
  $PY scripts/inference_i2i_batch.py --checkpoint "$ATARI" \
    --file-list "$L" --rough-dir "$D/$split/rough" \
    --output-dir "$R/atari/$name" --autocontrast
  echo "--- lucy_mild aux ---"
  $PY scripts/preprocess_atari_aux.py \
    --input-dir "$R/atari/$name" --output-dir "$R/lucy/$name" --mode lucy_mild
  echo "--- msgan (combined_koma_lucy_mild_msgan_20260729) ---"
  $PY scripts/inference_i2i_batch.py --checkpoint "$MSGAN" \
    --file-list "$L" --rough-dir "$D/$split/rough" --aux-dir "$R/lucy/$name" \
    --output-dir "$R/msgan/$name" --autocontrast
}

run_pool housei_test test
run_pool ako5_held   train
run_pool ako5r_held  train
echo "=== done $(date --iso-8601=seconds) ==="
