"""L6: how many koma panels give a CLEAN line/fill pair?

The user's direction (2026-10-24): build the fill research on data where line
and fill are actually separated, rather than measuring mixed data as mixed.

A panel is a clean pair when both hold:
  (a) its production fill sits in dedicated fill layers
      share_mixed = area in keep4<0.95 layers / total fill area   < MIX_MAX
  (b) that fill is NOT in the dataset's line raster, so the raster is pure line
      leak = |clean_fill AND raster_ink| / |clean_fill|           < LEAK_MAX
Then: line = the existing dataset raster, fill = the production layers,
rough = the existing rough. All three already exist.
"""
import csv, json, sys, collections
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import ink_of
KOMA=Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
BASE=Path('/home/sh1/deepl/lineart/data/incoming_manuscript')
OUT=Path('/home/sh1/deepl/lineart-fill-region/results/l6_clean_subset'); OUT.mkdir(parents=True,exist_ok=True)
WORKS=[('ako6','018_2018_ako6'),('ako7','026_2019_ako7')]
MIX_MAX, LEAK_MAX = 0.10, 0.10

form={}
for r in csv.DictReader(open('/home/sh1/deepl/lineart-fill-region/results/l5_ako7_generation/layer_form.csv')):
    form[(r['work'],r['page'],r['name'],r['n_px'])]=float(r['keep4'])

panels=collections.defaultdict(list); bywork={}
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    for w,wid in WORKS:
        if r['work_id']==wid:
            k=r['pid'].replace('__','_'); panels[k].append(r); bywork[k]=w

rows=[]
for w,_ in WORKS:
    man=json.load(open(BASE/w/'fill'/'beta_manifest.json'))
    pages={p['page']:p for p in man['pages']}
    for pid,prs in panels.items():
        if bywork[pid]!=w or pid not in pages or not pages[pid]['beta']: continue
        clean=None; mixed=None
        for b in pages[pid]['beta']:
            im=cv2.imread(str(BASE/w/'fill'/b['file']), cv2.IMREAD_GRAYSCALE)<128
            k=form.get((w,pid,b['name'],str(int(im.sum()))))
            if k is None: continue
            if k>=0.95: clean = im if clean is None else (clean|im)
            else:       mixed = im if mixed is None else (mixed|im)
        for r in prs:
            x0,y0,x1,y1=(int(v) for v in (r['x0'],r['y0'],r['x1'],r['y1']))
            g=cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
            if g is None: continue
            gi=ink_of(g)
            if float(gi.mean())<0.002: continue
            c = np.zeros(g.shape,bool) if clean is None else clean[y0:y1,x0:x1]
            m = np.zeros(g.shape,bool) if mixed is None else mixed[y0:y1,x0:x1]
            cp, mp = int(c.sum()), int(m.sum())
            if cp+mp==0: continue
            leak = int((c&gi).sum())/cp if cp else 0.0
            rows.append({'work':w,'pid':pid,'name':r['name'],'clean_px':cp,'mixed_px':mp,
                         'share_mixed':mp/(cp+mp),'leak':leak,
                         'clean_frac_of_panel':cp/g.size})
with open(OUT/'per_panel.csv','w',newline='') as fh:
    wtr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)

print(f'{"作品":6s} {"コマ":>5} {"(a)分離":>8} {"(b)漏れ無":>9} {"両方=綺麗":>10} {"ベタ面積中央":>12}')
for w,_ in WORKS:
    v=[r for r in rows if r['work']==w]
    a=[r for r in v if r['share_mixed']<MIX_MAX]
    b=[r for r in v if r['leak']<LEAK_MAX]
    both=[r for r in v if r['share_mixed']<MIX_MAX and r['leak']<LEAK_MAX]
    import statistics as st
    med=st.median([r['clean_frac_of_panel'] for r in both]) if both else float('nan')
    print(f'{w:6s} {len(v):>5} {len(a):>8} {len(b):>9} {len(both):>10} {100*med:>11.1f}%')
tot=[r for r in rows if r['share_mixed']<MIX_MAX and r['leak']<LEAK_MAX]
print(f'\n綺麗な組の合計: {len(tot)} コマ  (測れた {len(rows)} コマ中)')
print(f'  ラフも線画も既に在るので、この {len(tot)} コマは追加作業なしで使える')
