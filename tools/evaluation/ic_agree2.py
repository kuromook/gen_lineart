"""L2: split the production standard on the FOLDER axis -- a third axis,
independent of both the name (extraction side) and the form (L's instrument).

Hypothesis under test, stated before the rows are read: the beta-named class
overlaps the radius-4 mask almost not at all (cover_prod 0.0529) because two
thirds of it sits under a `トーン` folder, and the line raster this project
holds was composited without tone. If that is the cause, then
    prod_tone     (ancestor contains トーン)  -> near zero, and
    prod_nontone  (everything else)          -> real overlap.
If instead BOTH are near zero, the tone folder is not the explanation and the
mask simply does not see painted areas.

Also emits the region-level rows against `prod_inline` := everything not under
a tone folder, i.e. the production fill that should be present in the raster
the mask is computed from.
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
    if r['work_id'] == '018_2018_ako6':
        panels[r['pid'].replace('__','_')].append(r)

rows=[]
for pid in sorted(panels):
    p = pages[pid]
    if not p['beta']: continue
    std={'prod_tone':None,'prod_inline':None}
    for b in p['beta']:
        ink = cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE) < 128
        k = 'prod_tone' if 'トーン' in b['ancestors'] else 'prod_inline'
        std[k] = ink.copy() if std[k] is None else (std[k] | ink)
    for r in sorted(panels[pid], key=lambda r: r['name']):
        x0,y0,x1,y1 = (int(r[k]) for k in ('x0','y0','x1','y1'))
        g = cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        gi = ink_of(g); rad = fill_mask(gi)
        row={'pid':pid,'name':r['name'],'slug':r['slug'],'ink_ratio':float(gi.mean()),
             'rad4_px':int(rad.sum())}
        for k,m in std.items():
            pm = np.zeros(g.shape, bool) if m is None else m[y0:y1, x0:x1]
            inter=int((rad&pm).sum())
            row[f'{k}_px']=int(pm.sum()); row[f'{k}_inter']=inter
            row[f'{k}_union']=int(rad.sum())+int(pm.sum())-inter
        mm = match(rad, (np.zeros(g.shape,bool) if std['prod_inline'] is None
                         else std['prod_inline'][y0:y1,x0:x1]), MIN_REGION_PX, 0.5)
        row.update({'reg_n_rad4':mm['n_pred'],'reg_n_prod':mm['n_gt'],
                    'reg_matched':mm['matched'],'reg_f1':mm['region_f1']})
        rows.append(row)
with open(OUT/'per_panel_folderaxis.csv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

norm=[r for r in rows if float(r['ink_ratio'])>=0.002]
print(f'panels: {len(rows)}  normal: {len(norm)}\n')
def line(label, rs, k):
    i=sum(int(r[f'{k}_inter']) for r in rs); u=sum(int(r[f'{k}_union']) for r in rs)
    pr=sum(int(r[f'{k}_px']) for r in rs); ra=sum(int(r['rad4_px']) for r in rs)
    print(f'{label:>26} {len(rs):>4} {i/u if u else 0:>8.4f} {i/pr if pr else float("nan"):>11.4f}'
          f' {i/ra if ra else 0:>12.4f} {pr:>13,} {ra:>12,}')
print(f'{"standard":>26} {"n":>4} {"IoU":>8} {"cover_prod":>11} {"purity_rad4":>12} {"prod_px":>13} {"rad4_px":>12}')
line('prod_tone (under トーン)', norm, 'prod_tone')
line('prod_inline (not tone)', norm, 'prod_inline')
print('\n-- prod_inline by slug (drawing stage), pools kept apart --')
g=collections.defaultdict(list)
for r in norm: g[r['slug']].append(r)
for k in sorted(g): line(k, g[k], 'prod_inline')
print('\n-- region level vs prod_inline, >=500px, IoU>=0.5 --')
import statistics as st
f1=[float(r['reg_f1']) for r in norm]
mr=sum(int(r['reg_matched']) for r in norm); npd=sum(int(r['reg_n_rad4']) for r in norm); ngt=sum(int(r['reg_n_prod']) for r in norm)
print(f'  region-F1 mean {st.mean(f1):.4f} median {st.median(f1):.4f}')
print(f'  regions: rad4 {npd}  prod_inline {ngt}  matched {mr}')
print(f'  pooled region recall {mr/max(1,ngt):.4f}  pooled region precision {mr/max(1,npd):.4f}')
