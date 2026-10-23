"""Anchor (handoff section 6 / discipline 2): reproduce blank region-F1 = 0.0084
on the 956 normal-stratum held panels. Borrowed instruments, read-only."""
import sys, csv
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of, fill_ratio_cv2
from region_metrics import regions, MIN_REGION_PX
from koma_split import ROOT, load as load_split

NEAR_BLANK_INK = 0.002
s = load_split()
rows = []
for name, work, fam in sorted(s['held']):
    g = cv2.imread(str(Path(ROOT)/'line'/name), cv2.IMREAD_GRAYSCALE)
    if g is None:
        rows.append((name, work, fam, None, None)); continue
    ink = ink_of(g)
    ink_ratio = float(ink.mean())
    gt = fill_mask(ink)
    _, gids, _ = regions(gt, MIN_REGION_PX)
    rows.append((name, work, fam, ink_ratio, len(gids)))

ok = [r for r in rows if r[3] is not None]
normal = [r for r in ok if r[3] >= NEAR_BLANK_INK]
nearb  = [r for r in ok if r[3] <  NEAR_BLANK_INK]
print(f'held panels read: {len(ok)} / {len(rows)}   normal {len(normal)}  near-blank {len(nearb)}')
def blank_f1(rs): return [1.0 if r[4]==0 else 0.0 for r in rs]
import collections
for label, rs in (('normal', normal), ('near-blank', nearb)):
    v = blank_f1(rs)
    per_fam = collections.defaultdict(list)
    for r, x in zip(rs, v): per_fam[r[2]].append(x)
    macro = float(np.mean([np.mean(x) for x in per_fam.values()])) if per_fam else float('nan')
    print(f'  {label:10s} n={len(rs):4d}  blank region-F1  panel-mean {np.mean(v):.4f}   family-macro {macro:.4f}')
    if label=='normal':
        for f in sorted(per_fam): print(f'      {f:14s} n={len(per_fam[f]):3d}  {np.mean(per_fam[f]):.4f}')
