#!/bin/bash
set -e
cd /home/sh1/deepl/lineart-stroke-grammar
PY=/home/sh1/deepl/lineart/venv/bin/python
for cfg in "125 5,5,5" "500 5,5,5,4" "1000 8,5,5,5"; do
  set -- $cfg
  w=$1; lv=$2
  $PY tools/stroke/train_codebook2.py --out results/invariance_20260920/l4_w$w --epochs 20 \
      --levels $lv --spread-weight 1.0 --consistency-weight 4.0 \
      > results/invariance_20260920/train_l4_w$w.out 2>&1
  $PY tools/stroke/word_eval.py --ckpt results/invariance_20260920/l4_w$w/codebook2.pt \
      --out results/invariance_20260920/eval_l4_w$w.json \
      > results/invariance_20260920/eval_l4_w$w.out 2>&1
done
echo DONE
