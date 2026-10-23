"""IC-L2: the form test on the production layer side.

Registered quantity, fixed before any row is read:
    form_keep(layer) := |fill_mask(layer_ink)| / |layer_ink|
the share of a layer's own painted pixels that survive the radius-4 opening
(a disk of radius 4 fits). A painted area keeps nearly all of itself; a 1-bit
stroke layer keeps almost none. The extraction side tested colour; this tests
form, and the two are built from different material.

Self-check (discipline 2 -- this quantity has no value on record, so the
instrument carries its identity inside itself): a disk of radius 20 must keep
~1.0, a 3px line must keep 0.0.

Pools are not mixed: rows are printed per name family and per source folder.
"""
import csv, json, sys, collections, statistics as st
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, FILL_HALF_WIDTH_PX

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill')
PAD = 32

# ---- self-check ----
d = np.zeros((200,200), bool); yy,xx = np.ogrid[:200,:200]
d[(yy-100)**2+(xx-100)**2 <= 20**2] = True
k_disk = fill_mask(d).sum()/d.sum()
l = np.zeros((200,200), bool); l[100:103, 20:180] = True
k_line = fill_mask(l).sum()/l.sum()
assert k_disk > 0.95, k_disk
assert k_line == 0.0, k_line
print(f'self-check: disk r=20 keeps {k_disk:.4f} (>0.95), 3px line keeps {k_line:.4f} (==0) -> PASS\n')

man = json.load(open(D/'beta_manifest.json'))
rows=[]
for p in man['pages']:
    folder = p['source'].replace('\\','/').split('/')[-2]
    for b in p['beta']:
        im = cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE)
        ink = im < 128
        n = int(ink.sum())
        if n == 0:
            rows.append(dict(b, page=p['page'], folder=folder, n_px=0, keep4=None, keep3=None, keep6=None, rmax=0.0)); continue
        ys,xs = np.where(ink)
        y0,y1 = max(0,ys.min()-PAD), min(ink.shape[0], ys.max()+1+PAD)
        x0,x1 = max(0,xs.min()-PAD), min(ink.shape[1], xs.max()+1+PAD)
        sub = ink[y0:y1, x0:x1]
        dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
        keep = {}
        for r in (3.0,4.0,6.0):
            keep[r] = float(fill_mask(sub, radius=r).sum())/n
        rows.append(dict(b, page=p['page'], folder=folder, n_px=n,
                         keep4=keep[4.0], keep3=keep[3.0], keep6=keep[6.0],
                         rmax=float(dist.max())))
print('layers measured:', len(rows))
out = Path('/home/sh1/deepl/lineart-fill-region/results/ic_l2_layer_form')
out.mkdir(parents=True, exist_ok=True)
with open(out/'layer_form.csv','w',newline='') as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(len(v)*p))]
def line(label, rs):
    k=[r['keep4'] for r in rs if r['keep4'] is not None]
    if not k: print(f'  {label:26s} n={len(rs):4d}  (no ink)'); return
    rm=[r['rmax'] for r in rs]
    print(f'  {label:26s} n={len(rs):4d}  keep4 p10 {q(k,.10):.3f} med {st.median(k):.3f} p90 {q(k,.90):.3f}'
          f'   keep4<0.05: {100*sum(1 for x in k if x<0.05)/len(k):4.1f}%   r_max med {st.median(rm):5.1f}')

print('\n-- all delivered layers --'); line('all', rows)
print('\n-- by name_family --')
g=collections.defaultdict(list)
for r in rows: g[r['name_family']].append(r)
for k in sorted(g, key=lambda x:-len(g[x])): line(k, g[k])
print('\n-- by layer name (>=15 layers) --')
g=collections.defaultdict(list)
for r in rows: g[r['name']].append(r)
for k in sorted(g, key=lambda x:-len(g[x])):
    if len(g[k])>=15: line(k, g[k])
print('\n-- by source folder (drawing stage) --')
g=collections.defaultdict(list)
for r in rows: g[r['folder']].append(r)
for k in sorted(g): line(k, g[k])
