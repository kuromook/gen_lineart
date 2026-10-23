"""L5: is ako7 the same production generation as ako6?

Pre-registered 2026-10-23 BEFORE ako7 was delivered (doc/work_log.md, section 5
of the evening entry). Primary values, both of them:

    (1) `black` keep4 median      -- is the black layer a mass or a stroke?
    (2) `beta`-name ink presence  -- does that fill reach the line raster?

ako6's recorded values are 0.994 and 0.1421. Both works are measured in THIS
run and ako6 is printed first: **if ako6 does not reproduce, nothing else in
the run is read** (discipline 2).

Form test carries its own identity (disk r=20 -> ~1.0, 3px line -> 0.0).
Pools are never mixed: every row is per work.
"""
import csv, json, sys, collections, statistics as st
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of

KOMA = Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
BASE = Path('/home/sh1/deepl/lineart/data/incoming_manuscript')
OUT  = Path('/home/sh1/deepl/lineart-fill-region/results/l5_ako7_generation')
OUT.mkdir(parents=True, exist_ok=True)
WORKS = [('ako6', '018_2018_ako6'), ('ako7', '026_2019_ako7')]
PAD = 32

# ---- identity ----
d = np.zeros((200,200), bool); yy,xx = np.ogrid[:200,:200]
d[(yy-100)**2+(xx-100)**2 <= 400] = True
l = np.zeros((200,200), bool); l[100:103, 20:180] = True
kd, kl = fill_mask(d).sum()/d.sum(), fill_mask(l).sum()/l.sum()
assert kd > 0.95 and kl == 0.0, (kd, kl)
print(f'identity: disk r=20 {kd:.4f} / 3px line {kl:.4f} -> PASS\n')

rows = []
for wname, work_id in WORKS:
    man = json.load(open(BASE/wname/'fill'/'beta_manifest.json'))
    for p in man['pages']:
        for b in p['beta']:
            im = cv2.imread(str(BASE/wname/'fill'/b['file']), cv2.IMREAD_GRAYSCALE)
            ink = im < 128; n = int(ink.sum())
            if n == 0:
                continue
            ys, xs = np.where(ink)
            sub = ink[max(0,ys.min()-PAD):ys.max()+1+PAD, max(0,xs.min()-PAD):xs.max()+1+PAD]
            dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
            rows.append({'work': wname, 'page': p['page'], 'name': b['name'],
                         'name_family': b['name_family'], 'ancestors': b['ancestors'],
                         'n_px': n, 'keep4': float(fill_mask(sub).sum())/n,
                         'rmax': float(dist.max()), 'drawn_ratio': b['drawn_ratio']})
with open(OUT/'layer_form.csv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print('=== PRIMARY 1: `black` keep4 median (ako6 recorded 0.994) ===')
for wname,_ in WORKS:
    v=[r['keep4'] for r in rows if r['work']==wname and r['name']=='black']
    tag = '  <- KNOWN CELL' if wname=='ako6' else ''
    print(f'  {wname}  n={len(v):>4}  median {st.median(v):.3f}  p10 {sorted(v)[len(v)//10]:.3f}'
          f'  rmax median {st.median([r["rmax"] for r in rows if r["work"]==wname and r["name"]=="black"]):.1f}{tag}')

print('\n=== layer form by name family, per work (pools kept apart) ===')
for wname,_ in WORKS:
    rs=[r for r in rows if r['work']==wname]
    tot=sum(r['n_px']*r['keep4'] for r in rs)
    print(f'  -- {wname}: {len(rs)} layers, painted px {sum(r["n_px"] for r in rs):,}, '
          f'area px {int(tot):,} ({100*tot/sum(r["n_px"] for r in rs):.1f}%)')
    for fam in ('beta','line','unknown'):
        v=[r for r in rs if r['name_family']==fam]
        if not v: continue
        a=sum(r['n_px']*r['keep4'] for r in v)
        print(f'     {fam:8s} n={len(v):>4} keep4 med {st.median([r["keep4"] for r in v]):.3f}'
              f'  area share {100*a/tot:>5.1f}%')

# ---- ink presence on the koma panels ----
print('\n=== PRIMARY 2: ink presence of production fill in the line raster ===')
print('    (ako6 `beta` recorded 0.1421)')
panels = collections.defaultdict(list)
bywork = {}
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    for wname, work_id in WORKS:
        if r['work_id']==work_id:
            panels[r['pid'].replace('__','_')].append(r); bywork[r['pid'].replace('__','_')]=wname
acc=collections.Counter(); npan=collections.Counter()
for wname, _ in WORKS:
    man = json.load(open(BASE/wname/'fill'/'beta_manifest.json'))
    pages={p['page']:p for p in man['pages']}
    for pid, prs in panels.items():
        if bywork[pid]!=wname or pid not in pages or not pages[pid]['beta']:
            continue
        std={}
        for b in pages[pid]['beta']:
            fam=b['name_family']
            ink=cv2.imread(str(BASE/wname/'fill'/b['file']), cv2.IMREAD_GRAYSCALE)<128
            std[fam]= ink if fam not in std else (std[fam]|ink)
        for r in prs:
            x0,y0,x1,y1=(int(r[k]) for k in ('x0','y0','x1','y1'))
            g=cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
            if g is None: continue
            gi=ink_of(g)
            if float(gi.mean())<0.002: continue
            npan[wname]+=1
            for fam,m in std.items():
                pm=m[y0:y1,x0:x1]
                acc[(wname,fam,'px')]+=int(pm.sum()); acc[(wname,fam,'ink')]+=int((pm&gi).sum())
for wname,_ in WORKS:
    tag='  <- KNOWN CELL' if wname=='ako6' else ''
    print(f'  -- {wname}: {npan[wname]} panels{tag}')
    tp=ti=0
    for fam in ('beta','line','unknown'):
        px=acc[(wname,fam,'px')]; ik=acc[(wname,fam,'ink')]
        tp+=px; ti+=ik
        if px: print(f'     {fam:8s} prod px {px:>12,}  ink {ik:>12,}  share {ik/px:.4f}')
    if tp: print(f'     {"TOTAL":8s} prod px {tp:>12,}  ink {ti:>12,}  share {ti/tp:.4f}'
                 f'   -> raster misses {100*(1-ti/tp):.1f}%')
