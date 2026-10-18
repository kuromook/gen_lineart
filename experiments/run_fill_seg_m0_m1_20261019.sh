#!/usr/bin/env bash
# Stage 2 slot 2026-10-19..10-22: train M0 then M1 (doc/work_log.md, the
# 2026-10-10 registration and amendment J1 of 2026-10-18).
#
# Both arms, not just M1, because the registered question C2 is **M1 - M0 >=
# 0.03 region-F1** plus NC4 dropping by >= 0.03: M1 alone leaves C2 unreadable.
# Sequential rather than parallel -- the 4-day window is far larger than the
# two runs need, and a parallel OOM on day three is the expensive failure.
#
# Plain supervised segmentation, U-Net, BCE + Dice, **no adversarial term**.
# The falsifier stands verbatim: 敵対項を足した時点、あるいは出力を「マスク」
# ではなく「線画」として読んだ時点で、禁止2の違反です。
#
# Discipline (lesson 7): PYTHONUNBUFFERED=1, the holdout is scored at every
# snapshot, training loss is never read as evidence.
set -u
cd /home/sh1/deepl/lineart-solid-fill
PY=/home/sh1/deepl/lineart/venv/bin/python
OUT=results/fill_seg_20261010
STEPS=20000
mkdir -p "$OUT"
for ARM in M0 M1; do
  echo "=== $ARM start $(date '+%F %T') ==="
  PYTHONUNBUFFERED=1 "$PY" scripts/train_fill_seg.py \
      --arm "$ARM" --steps "$STEPS" --eval-snapshot-steps 2000 --eval-panels 40
  echo "=== $ARM exit $? at $(date '+%F %T') ==="
done
echo "=== slot run finished $(date '+%F %T') ==="
