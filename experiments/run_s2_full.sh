#!/bin/bash
# 登録 S2/S2a の本番: 作品ごとに results/s2_full/<work>/(済んだ作品は飛ばす)。4並列
cd "$(dirname "$0")/.."
ls results/s1_full | xargs -P4 -I{} bash -c '
  [ -f results/s2_full/{}/done ] && exit 0
  mkdir -p results/s2_full/{}
  ../lineart/venv/bin/python experiments/s2_panel_delta.py --work {} --out results/s2_full/{} > results/s2_full/{}/log.txt 2>&1 \
    && touch results/s2_full/{}/done || echo "失敗 {}" >> results/s2_full_failures.txt'
echo "終了 $(date)" >> results/s2_full_run.log
