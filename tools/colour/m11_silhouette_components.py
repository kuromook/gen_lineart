"""M11: the author decided (2026-10-24) that the silhouette layer is the
target, and said that a silhouette layer is MOSTLY A COMPOSITE OF SEVERAL
AREAS. So "one silhouette layer = one area" is not what the author means by
面, and the standard has to be the areas inside the layer, not the layer.

This file measures, per silhouette layer, the number of 8-connected
components of its coverage above the registered area floor.

Registered floor (追補5 E6, as a FRACTION of the page per 追補2 B4):
    1e-4 of the page area, 8-connected.
Here the layers are native size, so the floor is 1e-4 of the canvas.

A component count is a LOWER BOUND on the author's 面 count: two areas that
touch (a collar and the body of a shirt) are one component. That limit is
registered in 追補6 and is not something this file can resolve.

MEASURES ONLY. No arm is scored. Silhouette layers are identified
structurally (the base of a clip chain, 追補5 E4 B3'), never by name; the
name columns are written to the CSV for the falsifier step and are not used
to select anything.
"""
import json, collections, csv, sys, time
from pathlib import Path
import cv2, numpy as np

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
OUT = Path('results/m11_silhouette'); OUT.mkdir(parents=True, exist_ok=True)
PAPER_TYPE = 1584
ALPHA_DRAWN = 26
AREA_FLOOR = 1e-4          # registered in 追補5 E6
FLOOR_SWEEP = (1e-5, 1e-4, 1e-3, 1e-2)


def clip_bases(page):
    """The base of each maximal clip run, within one parent. 0 = bottom."""
    byp = collections.defaultdict(list)
    for l in page['layers']:
        if l['layer_type'] != PAPER_TYPE:
            byp[l['parent_id']].append(l)
    out = {}
    for sib in byp.values():
        sib.sort(key=lambda l: l['order_in_parent'])
        base = None
        for l in sib:
            if l['clip'] == 0:
                base = l
            elif base is not None:
                out[base['layer_id']] = base
    return out


def main():
    pages = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    rows, t0 = [], time.time()
    for p in pages:
        bases = clip_bases(p)
        for lid, l in sorted(bases.items()):
            im = cv2.imread(str(D / l['file']), cv2.IMREAD_UNCHANGED)
            m = (im[..., 3] >= ALPHA_DRAWN).astype(np.uint8)
            n_px = m.size
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            row = {'page': p['page'], 'layer_id': lid, 'name': l['name'],
                   'coverage': float(m.mean()), 'components_all': int(n - 1)}
            for fl in FLOOR_SWEEP:
                row[f'n@{fl:.0e}'] = int((areas >= max(1, round(fl * n_px))).sum())
            keep = areas[areas >= max(1, round(AREA_FLOOR * n_px))]
            row['largest_share'] = float(keep.max() / m.sum()) if len(keep) and m.sum() else 0.0
            rows.append(row)
        print(f'{p["page"]:44s} {len(bases):>4} silhouettes  {time.time()-t0:6.1f}s', flush=True)

    with open(OUT / 'components.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {OUT/"components.csv"} ({len(rows)} silhouette layers)')

    key = f'n@{AREA_FLOOR:.0e}'
    n = [r[key] for r in rows]
    print(f'\n== areas per silhouette layer, floor {AREA_FLOOR:.0e} of the page, 8-connected ==')
    print(f'  silhouette layers        : {len(n)}')
    print(f'  with 0 areas above floor : {sum(1 for x in n if x == 0)}')
    print(f'  with exactly 1           : {sum(1 for x in n if x == 1)}')
    print(f'  with 2 or more           : {sum(1 for x in n if x >= 2)}'
          f'   ({sum(1 for x in n if x >= 2)/len(n):.3f} of all)')
    v = sorted(n)
    print(f'  count   min {v[0]}  p10 {v[len(v)//10]}  med {v[len(v)//2]}  '
          f'p90 {v[-max(1,len(v)//10)]}  max {v[-1]}')
    print(f'  total areas in the registration set: {sum(n)}  '
          f'(against {len(n)} layers)')

    print('\n== how the count moves with the floor (the floor is registered at 1e-4) ==')
    for fl in FLOOR_SWEEP:
        k = f'n@{fl:.0e}'
        vv = [r[k] for r in rows]
        print(f'  floor {fl:.0e}: total {sum(vv):>6}  median {sorted(vv)[len(vv)//2]:>3}  '
              f'>=2 areas {sum(1 for x in vv if x >= 2):>4}  '
              f'max {max(vv):>5}')
    print('  (components_all, no floor at all): total '
          f'{sum(r["components_all"] for r in rows)}')

    print('\n== per page ==')
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page']].append(r)
    for pg, rs in byp.items():
        nn = [r[key] for r in rs]
        print(f'  {pg:44s} {len(rs):>4} layers -> {sum(nn):>5} areas   '
              f'>=2 areas in {sum(1 for x in nn if x >= 2):>3} layers')
    print('\nNo arm is scored in this file.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
