"""L1: the radius-4 inferred fill mask against the production beta layers,
in both directions. 225 ako6 koma panels, train split, version-exact by pid.

Quantities fixed before any row is read, per panel, at radius 4:
    cover_prod   = |rad4 & prod| / |prod|    how much production fill the mask recovers
    purity_rad4  = |rad4 & prod| / |rad4|    how much of the mask is production fill
    iou          = |rad4 & prod| / |rad4 | prod|
Micro values are pixel-pooled WITHIN a pool and never across pools.
Region level: components >= 500px both sides, greedy match at IoU >= 0.5
(region_metrics, borrowed unchanged) -> region recall / precision / F1.

Three production standards, cut on axes that are NOT the instrument under test:
    prod_all       the delivered composite (781 layers; the extraction side's
                   own selection, which tested colour)
    prod_betaname  name_family == 'beta'  (193 layers)
    prod_linename  name_family == 'line'  (549 layers)
Name is independent of form, so this cut cannot be circular. The form axis is
reported separately by IC-L2 and is never used to filter the standard here.

Negative controls, one named error each. If these do not collapse, the
agreement is a quantity effect and not a position one (the preproc@budget trap):
    NC-L1 offbyone  the production mask of the next panel in sorted order,
                    both centre-cropped to the common overlap; the arm's own
                    row on the SAME crop is emitted as @nc2crop and NC-L1 is
                    read only against that row.
    NC-L2 hflip     the production mask mirrored left-right.

Identity check carried inside: the union of a page's delivered layer PNGs must
equal that page's delivered composite, byte for byte.
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
OUT.mkdir(parents=True, exist_ok=True)
NEAR_BLANK_INK = 0.002
NC_MIN_OVERLAP = 0.25

man = json.load(open(D/'beta_manifest.json'))
pages = {p['page']: p for p in man['pages']}
panels = collections.defaultdict(list)
for r in csv.DictReader(open(KOMA/'manifest.csv')):
    if r['work_id'] == '018_2018_ako6':
        panels[r['pid'].replace('__','_')].append(r)

def centre_crop(m,h,w):
    y=(m.shape[0]-h)//2; x=(m.shape[1]-w)//2
    return m[y:y+h, x:x+w]

rows=[]; id_checked=0
for pid in sorted(panels):
    p = pages[pid]
    if not p['beta']:
        continue
    std = {'prod_all': None, 'prod_betaname': None, 'prod_linename': None}
    for b in p['beta']:
        ink = cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE) < 128
        for k, want in (('prod_all',True), ('prod_betaname', b['name_family']=='beta'),
                        ('prod_linename', b['name_family']=='line')):
            if not want: continue
            std[k] = ink.copy() if std[k] is None else (std[k] | ink)
    # identity: union of layers == delivered composite
    comp = cv2.imread(str(D/p['composite']), cv2.IMREAD_GRAYSCALE) < 128
    assert np.array_equal(comp, std['prod_all']), f'layer union != composite on {pid}'
    id_checked += 1
    for r in sorted(panels[pid], key=lambda r: r['name']):
        x0,y0,x1,y1 = (int(r[k]) for k in ('x0','y0','x1','y1'))
        g = cv2.imread(str(KOMA/'line'/r['name']), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        gi = ink_of(g)
        row = {'pid':pid, 'name':r['name'], 'slug':r['slug'],
               'w':g.shape[1], 'h':g.shape[0], 'ink_ratio':float(gi.mean())}
        rad = {rr: fill_mask(gi, radius=rr) for rr in (3.0,4.0,6.0)}
        for k,m in std.items():
            pm = np.zeros(g.shape, bool) if m is None else m[y0:y1, x0:x1]
            row[f'{k}_px'] = int(pm.sum())
            for rr in (3.0,4.0,6.0):
                a = rad[rr]
                inter = int((a & pm).sum())
                row[f'{k}_inter_r{int(rr)}'] = inter
                row[f'{k}_union_r{int(rr)}'] = int(a.sum()) + int(pm.sum()) - inter
            if k == 'prod_all':
                mm = match(rad[4.0], pm, MIN_REGION_PX, 0.5)
                row.update({'reg_n_rad4':mm['n_pred'], 'reg_n_prod':mm['n_gt'],
                            'reg_matched':mm['matched'], 'reg_prec':mm['precision'],
                            'reg_rec':mm['recall'], 'reg_f1':mm['region_f1']})
                # NC-L2 hflip
                fm = pm[:, ::-1]
                row['nc_hflip_inter'] = int((rad[4.0] & fm).sum())
                row['nc_hflip_union'] = int(rad[4.0].sum()) + int(fm.sum()) - row['nc_hflip_inter']
        row['rad4_px'] = int(rad[4.0].sum())
        row['rad3_px'] = int(rad[3.0].sum())
        row['rad6_px'] = int(rad[6.0].sum())
        row['_box'] = (x0,y0,x1,y1)
        rows.append(row)

print(f'identity check (layer union == composite): {id_checked} pages, all equal -> PASS')
print(f'panels measured: {len(rows)}')

# ---- NC-L1 off-by-one, consecutive panels in sorted name order ----
ordered = sorted(rows, key=lambda r: r['name'])
nc1 = []; skipped = 0
for a, b in zip(ordered, ordered[1:]):
    h, w = min(a['h'],b['h']), min(a['w'],b['w'])
    ov = h*w / float(min(a['h']*a['w'], b['h']*b['w']))
    if ov < NC_MIN_OVERLAP:
        skipped += 1; continue
    ga = ink_of(cv2.imread(str(KOMA/'line'/a['name']), cv2.IMREAD_GRAYSCALE))
    ra = centre_crop(fill_mask(ga), h, w)
    pb = pages[b['pid']]
    mb = cv2.imread(str(D/pb['composite']), cv2.IMREAD_GRAYSCALE) < 128
    x0,y0,x1,y1 = b['_box']
    pmb = centre_crop(mb[y0:y1, x0:x1], h, w)
    pa = pages[a['pid']]
    ma = cv2.imread(str(D/pa['composite']), cv2.IMREAD_GRAYSCALE) < 128
    x0,y0,x1,y1 = a['_box']
    pma = centre_crop(ma[y0:y1, x0:x1], h, w)
    def sc(m):
        i = int((ra & m).sum()); return i, int(ra.sum())+int(m.sum())-i, int(m.sum())
    i1,u1,n1 = sc(pmb); i0,u0,n0 = sc(pma)
    nc1.append({'name':a['name'],'nc_inter':i1,'nc_union':u1,'nc_prod_px':n1,
                'own_inter':i0,'own_union':u0,'own_prod_px':n0,'rad4_px':int(ra.sum())})
print(f'NC-L1 pairs: {len(nc1)}, skipped by the {NC_MIN_OVERLAP:.0%} overlap guard: {skipped}')

for r in rows: r.pop('_box')
with open(OUT/'per_panel.csv','w',newline='') as fh:
    w_=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
with open(OUT/'nc_offbyone.csv','w',newline='') as fh:
    w_=csv.DictWriter(fh, fieldnames=list(nc1[0].keys())); w_.writeheader(); w_.writerows(nc1)
print('wrote', OUT/'per_panel.csv', 'and', OUT/'nc_offbyone.csv')
