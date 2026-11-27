#!/bin/bash
# 登録 S4 の本番(T1 × C θ0.50): 作品ごとに results/s4_full/<work>/(済んだ作品は飛ばす)。4並列
cd "$(dirname "$0")/.."
ls results/s1_full | xargs -P4 -I{} bash -c '
  [ -f results/s4_full/{}/done ] && exit 0
  mkdir -p results/s4_full/{}
  ../lineart/venv/bin/python experiments/s2_panel_delta.py --work {} --out results/s4_full/{} --sketch-arm T1 --split C > results/s4_full/{}/log.txt 2>&1 \
    && touch results/s4_full/{}/done || echo "失敗 {}" >> results/s4_full_failures.txt'
echo "終了 $(date)" >> results/s4_full_run.log
