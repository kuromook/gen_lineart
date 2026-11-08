"""Smoke (no correlations): sketch-side ink threshold at native 480 tiles.

One page per year label (seed 0) from v4 ok+primary pairs. Picks the tile
maximising sketch<200 among tiles whose GT<128 ratio is in [0.02, 0.15]
(first version maximised min(sketch, GT) and picked solid-fill tiles), writes a 1:1 strip
[sketch | <128 | <160 | <200 | <220 | GT] and prints per-tile ink ratios and
GT line width (line_width_stats, FILL_HALF_WIDTH_PX=4 fill_ratio) at native scale.
"""
import csv, glob, os, re, random, sys, collections as C
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools/evaluation')
import measure_lineart_profile as M
from tile_region_manifest_480 import line_width_stats

ROOT = 'data/clip_pairs_v4/'
OUT = 'results/pair_feature_correlation_20261108/smoke_threshold/'
T = 480
THR = (128, 160, 200, 220)
rows = list(csv.DictReader(open(ROOT + 'clip_pairs/clip_pairs_qc_v4.csv', encoding='utf-8')))
idx = {os.path.basename(p): p for p in glob.glob(ROOT + 'clip_pairs/**/*.jpg', recursive=True)}
pop = [x for x in rows if x['pair_quality'] == 'ok' and x['is_primary'] == 'True']
by = C.defaultdict(list)
for x in pop:
    by[re.match(r"\d+_(\d{4}\w?)_", x['work_id']).group(1)].append(x)
random.seed(0)
print('year work page_shape tile | sk<128 sk<160 sk<200 sk<220 | gt<128 gt_w50 gt_w95 gt_fill | sk200_w50 sk200_fill')
for y in sorted(by):
    x = random.choice(by[y])
    s = np.asarray(Image.open(idx[x['sketch']]).convert('L'))
    g = np.asarray(Image.open(idx[x['line']]).convert('L'))
    if s.shape != g.shape:
        print(y, 'SHAPE MISMATCH', s.shape, g.shape); continue
    best, bt = -1, None
    for r0 in range(0, s.shape[0] - T + 1, T):
        for c0 in range(0, s.shape[1] - T + 1, T):
            gr = (g[r0:r0+T, c0:c0+T] < 128).mean()
            v = (s[r0:r0+T, c0:c0+T] < 200).mean() if 0.02 <= gr <= 0.15 else -1
            if v > best: best, bt = v, (r0, c0)
    r0, c0 = bt
    st, gt = s[r0:r0+T, c0:c0+T], g[r0:r0+T, c0:c0+T]
    gi = gt < 128
    w50, w95 = line_width_stats(gi)
    si = st < 200
    sw50, _ = line_width_stats(si)
    strip = [st] + [np.where(st < t, 0, 255).astype(np.uint8) for t in THR] + [gt]
    sep = np.full((T, 8), 128, np.uint8)
    im = np.concatenate(sum([[a, sep] for a in strip], [])[:-1], axis=1)
    fn = f"{OUT}{y}_{x['work_id']}_{os.path.splitext(x['page'])[0]}_r{r0}_c{c0}.png"
    Image.fromarray(im).save(fn)
    # page-level alignment check at 1/8: overlap of ink at zero shift vs best shift (+-40 px at 1/8)
    def small(a, t):
        h, w = a.shape[0] // 8, a.shape[1] // 8
        return np.asarray(Image.fromarray(((a < t) * 255).astype(np.uint8)).resize((w, h), Image.BOX)) > 32
    ss, gs = small(s, 200), small(g, 128)
    def ov(dy, dx):
        a = np.roll(np.roll(ss, dy, 0), dx, 1)
        return (a & gs).sum() / max(gs.sum(), 1)
    z = ov(0, 0)
    bz, bd = max((ov(dy, dx), (dy, dx)) for dy in range(-40, 41, 2) for dx in range(-40, 41, 2))
    Image.fromarray(np.stack([np.where(gs, 0, 255), np.where(ss, 0, 255), np.full(gs.shape, 255)], -1).astype(np.uint8)
                    ).save(f"{OUT}overlay_{y}_{x['work_id']}_{os.path.splitext(x['page'])[0]}.png")
    print(y, x['work_id'], x['page'], 'align0', round(z, 3), 'best', round(bz, 3), 'shift8', bd, x['aligned'], x['sketch_name'])
    print(y, x['work_id'], s.shape, (r0, c0), '|', *[round((st < t).mean(), 3) for t in THR], '|',
          round(gi.mean(), 3), round(w50, 2), round(w95, 2), round(M.fill_ratio(gi), 3), '|',
          round(sw50, 2), round(M.fill_ratio(si), 3))
