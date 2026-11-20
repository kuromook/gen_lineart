#!/bin/bash
# 登録 S1(S1a・S1b)の本番: 厳しい門の全作品。作品ごとに results/s1_full/<work>/s1_pages.csv
set -u
cd "$(dirname "$0")/.."
PY=../lineart/venv/bin/python
WORKS=$($PY -c "
import sys; sys.path.insert(0,'experiments')
from s1_cleanup_delta import population
print(' '.join(sorted({r['work_id'] for r in population()})))")
echo "works: $(echo $WORKS | wc -w)"
for w in $WORKS; do
  if [ -f results/s1_full/$w/s1_pages.csv ]; then echo "skip $w"; continue; fi
  echo "=== $w $(date +%H:%M:%S)"
  $PY experiments/s1_cleanup_delta.py --work $w --out results/s1_full/$w --crops || echo "FAILED $w"
done
echo "=== DONE $(date)"
