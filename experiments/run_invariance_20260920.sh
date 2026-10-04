#!/bin/bash
# Track F: invariance-trained codebook, 2026-09-20. See doc/work_log.md (entry of this date).
# Each training cluster is duplicated; the copy has one random stroke of arc-rank >= 3
# (the "minor" definition of word_eval.py, 0-indexed 4th-or-lower) dropped. The perturbed
# code must still reconstruct the ORIGINAL cluster, and the pre-rounding FSQ values b of
# the full and perturbed views are pulled together (MSE, --consistency-weight).
# Bars unchanged: P(change | minor) <= 25%, then maximise D.
set -e
cd /home/sh1/deepl/lineart-stroke-grammar
PY=/home/sh1/deepl/lineart/venv/bin/python
for cfg in "125 5,5,5" "500 5,5,5,4" "1000 8,5,5,5"; do
  set -- $cfg
  w=$1; lv=$2
  $PY tools/stroke/train_codebook2.py --out results/invariance_20260920/w$w --epochs 20 \
      --levels $lv --spread-weight 1.0 --consistency-weight 1.0 \
      > results/invariance_20260920/train_w$w.out 2>&1
  $PY tools/stroke/word_eval.py --ckpt results/invariance_20260920/w$w/codebook2.pt \
      --out results/invariance_20260920/eval_w$w.json \
      > results/invariance_20260920/eval_w$w.out 2>&1
done
echo DONE
