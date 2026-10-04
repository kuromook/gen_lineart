#!/bin/bash
# Track F decoder A (categorical coordinates), 2026-09-20. See doc/work_log.md (entry of this date).
# Same recipe as the adopted l4_w500 (500 words, levels 5,5,5,4, consistency 4.0, spread 1.0,
# 20 epochs) but the decoder head classifies each point coordinate into 72 bins of 1px over [-36,36].
set -e
cd /home/sh1/deepl/lineart-stroke-grammar
PY=/home/sh1/deepl/lineart/venv/bin/python
$PY tools/stroke/train_codebook2.py --out results/decoderA_20260920/w500 --epochs 20 \
    --levels 5,5,5,4 --spread-weight 1.0 --consistency-weight 4.0 --coord-bins 72 \
    > results/decoderA_20260920/train_w500.out 2>&1
$PY tools/stroke/word_eval.py --ckpt results/decoderA_20260920/w500/codebook2.pt \
    --out results/decoderA_20260920/eval_w500.json \
    > results/decoderA_20260920/eval_w500.out 2>&1
$PY tools/stroke/codebook_diag.py --ckpt results/decoderA_20260920/w500/codebook2.pt \
    > results/decoderA_20260920/diag_w500.out 2>&1
echo DONE
