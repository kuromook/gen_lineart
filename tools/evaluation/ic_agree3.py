"""L3: the clean standard.

The problem L2 exposed: `prod_inline` is the union of whole LAYERS, and IC-L2
showed a line-named layer is only 86.3% area by pixel mass (median layer 0.626),
so that standard carries strokes and understates the mask in both directions.

The clean standard has to be selected on an axis that is NOT the radius-4
opening, or the comparison is circular. Taken: name_family == 'beta' AND not
under a tone folder -- selected by NAME (the extraction side's axis) and
independently verified by FORM (IC-L2: that class keeps 0.991 of itself, p10
0.955), but never filtered by it. 59 layers.

Reported beside it, as the explicit upper/lower bracket on one quantity:
    prod_inline      whole fill-bearing layers, strokes included  (lower bound)
    prod_clean       beta-named, non-tone                         (clean)
"""
import csv, json, sys, collections
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
from region_metrics import match, MIN_REGION_PX

KOMA = Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
D    = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill')
OUT  = Path('/home/sh1/deepl/lineart-fill-region/results/l1_mask_vs_production')
man  = json.load(open(D/'beta_manifest.json'))
pages = {p['page']: p for p in man['pages']}
panels = collections.defaultdict(list)
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    if r['work_id']=='018_2018_ako6': panels[r['pid'].replace('__','_')].append(r)

nlay=collections.Counter(); rows=[]
for pid in sorted(panels):
    p=pages[pid]
    if not p['beta']: continue
    clean=None
    for b in p['beta']:
        if b['name_family']=='beta' and 'トーン' not in b['ancestors']:
            nlay[pid]+=1
            ink=cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE)<128
            clean = ink.copy() if clean is None else (clean|ink)
    for r in sorted(panels[pid], key=lambda r:r['name']):
        x0,y0,x1,y1=(int(r[k]) for k in ('x0','y0','x1','y1'))
        g=cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        gi=ink_of(g); rad=fill_mask(gi)
        pm=np.zeros(g.shape,bool) if clean is None else clean[y0:y1,x0:x1]
        inter=int((rad&pm).sum())
        mm=match(rad,pm,MIN_REGION_PX,0.5)
        rows.append({'name':r['name'],'slug':r['slug'],'ink_ratio':float(gi.mean()),
            'rad4_px':int(rad.sum()),'clean_px':int(pm.sum()),'inter':inter,
            'union':int(rad.sum())+int(pm.sum())-inter,
            'reg_n_rad4':mm['n_pred'],'reg_n_prod':mm['n_gt'],'reg_matched':mm['matched'],
            'reg_f1':mm['region_f1']})
with open(OUT/'per_panel_clean.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f'beta-named non-tone layers: {sum(nlay.values())} over {len(nlay)} pages')
norm=[r for r in rows if r['ink_ratio']>=0.002]
withstd=[r for r in norm if r['clean_px']>0]
print(f'normal panels: {len(norm)}; those carrying this standard: {len(withstd)}')
def line(lab, rs):
    i=sum(r['inter'] for r in rs); u=sum(r['union'] for r in rs)
    pr=sum(r['clean_px'] for r in rs); ra=sum(r['rad4_px'] for r in rs)
    print(f'{lab:>22} {len(rs):>4} {i/u if u else 0:>8.4f} {i/pr if pr else float("nan"):>11.4f}'
          f' {i/ra if ra else 0:>12.4f} {pr:>12,} {ra:>12,}')
print(f'\n{"pool":>22} {"n":>4} {"IoU":>8} {"cover_prod":>11} {"purity_rad4":>12} {"prod_px":>12} {"rad4_px":>12}')
line('clean (all panels)', withstd)
g=collections.defaultdict(list)
for r in withstd: g[r['slug']].append(r)
for k in sorted(g): line(k, g[k])
import statistics as st
mr=sum(r['reg_matched'] for r in withstd); npd=sum(r['reg_n_rad4'] for r in withstd); ngt=sum(r['reg_n_prod'] for r in withstd)
print(f'\nregion level vs clean standard, >=500px, IoU>=0.5:')
print(f'  region-F1 mean {st.mean([r["reg_f1"] for r in withstd]):.4f}  median {st.median([r["reg_f1"] for r in withstd]):.4f}')
print(f'  regions: rad4 {npd}  prod_clean {ngt}  matched {mr}')
print(f'  pooled region recall {mr/max(1,ngt):.4f}  pooled region precision {mr/max(1,npd):.4f}')
