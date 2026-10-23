"""L4: which axis actually decides whether production fill is visible to the mask?

L2 read the tone folder as the cause. L3 refutes that reading: the beta-named
NON-tone layers also score ~0.04. The competing explanation is that the line
raster was composited by DISCARDING layers classified as fill, so the operative
axis is the extraction side's line/fill call -- i.e. the NAME -- and the tone
folder merely correlates with it.

The 2x2 separates them. If naming is operative: both beta cells near zero, both
line cells high. If the folder is operative: both tone cells near zero.
"""
import csv, json, sys, collections
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
KOMA=Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
D=Path('/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill')
man=json.load(open(D/'beta_manifest.json')); pages={p['page']:p for p in man['pages']}
panels=collections.defaultdict(list)
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    if r['work_id']=='018_2018_ako6': panels[r['pid'].replace('__','_')].append(r)
CELLS=[('beta','tone'),('beta','nontone'),('line','tone'),('line','nontone'),
       ('unknown','tone'),('unknown','nontone')]
acc={c:{'inter':0,'prod':0,'rad':0} for c in CELLS}
nl=collections.Counter()
for pid in sorted(panels):
    p=pages[pid]
    if not p['beta']: continue
    std={c:None for c in CELLS}
    for b in p['beta']:
        fam=b['name_family'] if b['name_family'] in ('beta','line') else 'unknown'
        c=(fam, 'tone' if 'トーン' in b['ancestors'] else 'nontone')
        nl[c]+=1
        ink=cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE)<128
        std[c]=ink.copy() if std[c] is None else (std[c]|ink)
    for r in sorted(panels[pid], key=lambda r:r['name']):
        x0,y0,x1,y1=(int(r[k]) for k in ('x0','y0','x1','y1'))
        g=cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
        if g is None or float(ink_of(g).mean())<0.002: continue
        rad=fill_mask(ink_of(g)); rs=int(rad.sum())
        for c in CELLS:
            if std[c] is None: continue
            pm=std[c][y0:y1,x0:x1]
            if not pm.any(): continue
            acc[c]['inter']+=int((rad&pm).sum()); acc[c]['prod']+=int(pm.sum()); acc[c]['rad']+=rs
print(f'{"name":>8} {"folder":>9} {"layers":>7} {"prod_px":>13} {"cover_prod":>11} {"purity":>8}')
for c in CELLS:
    a=acc[c]
    if a['prod']==0: print(f'{c[0]:>8} {c[1]:>9} {nl[c]:>7} {0:>13} {"-":>11} {"-":>8}'); continue
    print(f'{c[0]:>8} {c[1]:>9} {nl[c]:>7} {a["prod"]:>13,} {a["inter"]/a["prod"]:>11.4f} {a["inter"]/a["rad"]:>8.4f}')
