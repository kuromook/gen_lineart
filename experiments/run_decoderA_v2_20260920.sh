#!/bin/bash
# Track F decoder A redesign #2: 60 epochs, 2px bins (36). Same bars as pre-registered.
set -e
cd /home/sh1/deepl/lineart-stroke-grammar
PY=/home/sh1/deepl/lineart/venv/bin/python
$PY tools/stroke/train_codebook2.py --out results/decoderA_20260920/v2_w500 --epochs 60 \
    --levels 5,5,5,4 --spread-weight 1.0 --consistency-weight 4.0 --coord-bins 36 \
    > results/decoderA_20260920/train_v2_w500.out 2>&1
$PY tools/stroke/word_eval.py --ckpt results/decoderA_20260920/v2_w500/codebook2.pt \
    --out results/decoderA_20260920/eval_v2_w500.json \
    > results/decoderA_20260920/eval_v2_w500.out 2>&1
$PY tools/stroke/codebook_diag.py --ckpt results/decoderA_20260920/v2_w500/codebook2.pt \
    > results/decoderA_20260920/diag_v2_w500.out 2>&1
echo DONE
