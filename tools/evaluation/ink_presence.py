"""Is the beta-named production fill absent from the line raster, or present but
missed by the opening? The discriminator: what share of those pixels are INK at
all in the panel raster. Absent -> near 0. Present but thin -> near 1."""
import csv, json, sys, collections
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import ink_of, fill_mask
KOMA=Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
D=Path('/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill')
man=json.load(open(D/'beta_manifest.json')); pages={p['page']:p for p in man['pages']}
panels=collections.defaultdict(list)
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    if r['work_id']=='018_2018_ako6': panels[r['pid'].replace('__','_')].append(r)
acc=collections.Counter()
for pid in sorted(panels):
    p=pages[pid]
    if not p['beta']: continue
    std={}
    for b in p['beta']:
        fam=b['name_family'] if b['name_family'] in ('beta','line') else 'unknown'
        ink=cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE)<128
        std[fam]=ink if fam not in std else (std[fam]|ink)
    for r in sorted(panels[pid], key=lambda r:r['name']):
        x0,y0,x1,y1=(int(r[k]) for k in ('x0','y0','x1','y1'))
        g=cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        gi=ink_of(g)
        if float(gi.mean())<0.002: continue
        for fam,m in std.items():
            pm=m[y0:y1,x0:x1]
            acc[(fam,'px')]+=int(pm.sum()); acc[(fam,'ink')]+=int((pm&gi).sum())
print(f'{"name_family":>12} {"prod px":>14} {"of which ink in raster":>24} {"share":>8}')
for fam in ('beta','line','unknown'):
    px=acc[(fam,'px')]; ik=acc[(fam,'ink')]
    if px: print(f'{fam:>12} {px:>14,} {ik:>24,} {ik/px:>8.4f}')
