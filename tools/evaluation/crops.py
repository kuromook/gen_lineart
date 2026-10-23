"""Native-resolution cut-outs, 1:1, no downscaling (discipline 6 / L section 7-14:
do not argue from a reduced montage). Four strips side by side at native scale:
line raster | radius-4 mask | production fill in the raster | production tone-folder fill."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0,'/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of

KOMA = Path('/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826')
D    = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill')
OUT  = Path('/home/sh1/deepl/lineart-fill-region/results/l1_mask_vs_production/crops')
OUT.mkdir(parents=True, exist_ok=True)
man  = json.load(open(D/'beta_manifest.json'))
pages = {p['page']: p for p in man['pages']}
rows = {r['name']: r for r in csv.DictReader(open(KOMA/'manifest.csv'))}
WIN = 1100

def build(panel_name, tag):
    r = rows[panel_name]
    pid = r['pid'].replace('__','_')
    x0,y0,x1,y1 = (int(r[k]) for k in ('x0','y0','x1','y1'))
    g = cv2.imread(str(KOMA/'line'/panel_name), cv2.IMREAD_GRAYSCALE)
    rad = fill_mask(ink_of(g))
    tone = np.zeros(g.shape, bool); inl = np.zeros(g.shape, bool)
    for b in pages[pid]['beta']:
        ink = (cv2.imread(str(D/b['file']), cv2.IMREAD_GRAYSCALE) < 128)[y0:y1, x0:x1]
        if 'トーン' in b['ancestors']: tone |= ink
        else: inl |= ink
    # centre the window on the biggest production region the mask does not cover
    miss = inl & ~rad
    n,lab,st,cent = cv2.connectedComponentsWithStats(miss.astype(np.uint8), connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cy, cx = int(cent[i][1]), int(cent[i][0])
    else:
        cy, cx = g.shape[0]//2, g.shape[1]//2
    h = min(WIN, g.shape[0]); w = min(WIN, g.shape[1])
    ty = max(0, min(g.shape[0]-h, cy-h//2)); tx = max(0, min(g.shape[1]-w, cx-w//2))
    sl = (slice(ty,ty+h), slice(tx,tx+w))
    def strip(a, invert=True):
        v = (~a if invert else a).astype(np.uint8)*255
        return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    parts = [cv2.cvtColor(g[sl], cv2.COLOR_GRAY2BGR), strip(rad[sl]), strip(inl[sl]), strip(tone[sl])]
    gap = np.full((h, 6, 3), 128, np.uint8)
    out = parts[0]
    for p in parts[1:]: out = np.hstack([out, gap, p])
    path = OUT/f'{tag}_{panel_name.replace(".png","")}_native_{tx}_{ty}_{w}x{h}.png'
    cv2.imwrite(str(path), out)
    print(f'{path}\n   panel {g.shape[1]}x{g.shape[0]}, window {w}x{h} at ({tx},{ty}) 1:1 native; '
          f'strips: line | rad4 | prod_inline | prod_tone')

build('koma_01676_018_2018_ako6__ako6__page0030_p1.png', 'fragmentation')
build('koma_01717_018_2018_ako6__ako6__page0065_p2.png', 'tonechannel')
