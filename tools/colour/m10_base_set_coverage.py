"""M10: B2 was written for ONE base layer. Measure whether that is the shape
of these pages, or whether the area division is a SET of flat layers.

Registration 3.1 B2 says "the base covers a large fraction of the page,
because it is the layer that shows every area". M8 measured the clip chains
and found 167 clip bases across 8 pages with median coverage 0.0012 -- per
part silhouettes, not one page-wide layer. So the quantity B2 should cut on
may be the coverage of a SET, not of a layer.

This file measures, per page, the union coverage of four sets on pixels:
    extent      every non-paper layer, clip ignored          the drawn extent
    colour      every format-5 (colour) layer                 "
    flat        format-5, not clipped, top_colour_share >= FLAT
    clipbase    the clip bases M8 found
Union needs pixels: drawn_ratio sums overlap, and for a clipped layer it
overstates coverage by a median factor of 24 (M8, trap T-M3).

MEASURES ONLY. FLAT below is a sweep, not a registered cut; the cut goes in
追補5. No arm is scored.
"""
import json, collections, csv, sys, time
from pathlib import Path
import cv2, numpy as np

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
OUT = Path('results/m10_base_set'); OUT.mkdir(parents=True, exist_ok=True)
PAPER_TYPE = 1584
ALPHA_DRAWN = 26
FLAT_SWEEP = (0.5, 0.75, 0.9, 0.99)


def alpha_of(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return im[..., 3] >= ALPHA_DRAWN


def clip_bases(page):
    byp = collections.defaultdict(list)
    for l in page['layers']:
        if l['layer_type'] != PAPER_TYPE:
            byp[l['parent_id']].append(l)
    ids = set()
    for sib in byp.values():
        sib.sort(key=lambda l: l['order_in_parent'])
        base = None
        for l in sib:
            if l['clip'] == 0:
                base = l
            elif base is not None:
                ids.add(base['layer_id'])
    return ids


def main():
    pages = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    rows = []
    t0 = time.time()
    for p in pages:
        L = [l for l in p['layers'] if l['layer_type'] != PAPER_TYPE]
        bases = clip_bases(p)
        acc = {'extent': None, 'colour': None, 'clipbase': None}
        acc.update({f'flat{f}': None for f in FLAT_SWEEP})
        for l in L:
            a = alpha_of(D / l['file'])
            def add(k):
                acc[k] = a.copy() if acc[k] is None else (acc[k] | a)
            add('extent')
            if '5' in l['formats']:
                add('colour')
                if not l['clip']:
                    for f in FLAT_SWEEP:
                        if l['top_colour_share'] >= f:
                            add(f'flat{f}')
            if l['layer_id'] in bases:
                add('clipbase')
        n = next(v for v in acc.values() if v is not None).size
        row = {'page': p['page'], 'canvas': f'{p["canvas"][0]}x{p["canvas"][1]}',
               'n_layers': len(L), 'n_clipbase': len(bases)}
        for k, v in acc.items():
            row[k] = 0.0 if v is None else float(v.mean())
        ext = acc['extent']
        for k, v in acc.items():
            if k == 'extent' or v is None:
                continue
            row[f'{k}_of_extent'] = float((v & ext).sum()) / max(1, int(ext.sum()))
        rows.append(row)
        print(f'{p["page"]:44s} extent {row["extent"]:.4f}  '
              f'clipbase/extent {row.get("clipbase_of_extent", 0):.4f}  '
              f'flat0.9/extent {row.get("flat0.9_of_extent", 0):.4f}  '
              f'{time.time()-t0:6.1f}s', flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'set_coverage.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['page', 'canvas', 'n_layers', 'n_clipbase'] +
                           [k for k in keys if k not in ('page', 'canvas', 'n_layers', 'n_clipbase')])
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {OUT/"set_coverage.csv"}')
    print('\n== union coverage as a fraction of the drawn extent, per page ==')
    print(f'{"page":44s} ' + ' '.join(f'{k:>12}' for k in
          ('colour', 'clipbase', 'flat0.5', 'flat0.75', 'flat0.9', 'flat0.99')))
    for r in rows:
        print(f'{r["page"]:44s} ' + ' '.join(
            f'{r.get(k+"_of_extent", 0):>12.4f}' for k in
            ('colour', 'clipbase', 'flat0.5', 'flat0.75', 'flat0.9', 'flat0.99')))
    print('\nNo cut is registered here. No arm is scored.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
