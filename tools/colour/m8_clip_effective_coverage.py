"""M8: what a clipped layer actually covers, measured on pixels.

Why this file exists. The manifest's drawn_ratio was verified against the
exported alpha and agrees exactly (M7: 6 spot checks, |diff| <= 1e-5). But
303 of the pilot's 698 non-paper layers carry clip=1, and 146 of those have
drawn_ratio == 1.0 -- they paint the whole canvas in their own raster, and
CSP restricts them to the layer below only when it composites. The export
does not apply the clip (the extraction side applies it in their own
recomposition check, and says so).

So for a clipped layer:
    drawn_ratio      = coverage of its OWN raster          (what we were given)
    effective        = |alpha_clipped & alpha_clipbase|    (what it paints)
and B2 (coverage) read off drawn_ratio is wrong for 43% of the layers.
B1 (flatness) is contaminated the same way, because top_colour_share and
colours_q16 are computed over the layer's own drawn pixels, including the
ones the clip throws away.

This file measures the size of that contamination on the registration set
(8 pilot pages) so that 追補5 can register cuts on a quantity that exists.
MEASURES ONLY -- no judge_*, no arm, no cut applied.

Clip chain: inside one parent, layers sorted by order_in_parent (0 = bottom,
the direction the extraction side verified by recomposition on all 8 pages).
A maximal run of clip=1 hangs off the nearest clip=0 layer below it.
"""
import json, collections, csv, sys, time
from pathlib import Path
import cv2, numpy as np

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
OUT = Path('results/m8_clip_coverage'); OUT.mkdir(parents=True, exist_ok=True)
PAPER_TYPE = 1584
ALPHA_DRAWN = 26          # the extraction side's own threshold for "drawn"


def alpha_of(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        raise RuntimeError(f'not RGBA: {path}')
    return im[..., 3] >= ALPHA_DRAWN


def chains(page):
    """(clip_base, [clipped...]) per maximal clip run, within one parent."""
    byp = collections.defaultdict(list)
    for l in page['layers']:
        if l['layer_type'] != PAPER_TYPE:
            byp[l['parent_id']].append(l)
    out = []
    for sib in byp.values():
        sib.sort(key=lambda l: l['order_in_parent'])
        base, run = None, []
        for l in sib:
            if l['clip'] == 0:
                if base is not None and run:
                    out.append((base, run))
                base, run = l, []
            elif base is not None:
                run.append(l)
        if base is not None and run:
            out.append((base, run))
    return out


def measure_page(page):
    rows = []
    for base, run in chains(page):
        ab = alpha_of(D / base['file'])
        nb = int(ab.sum()); tot = ab.size
        for l in run:
            al = alpha_of(D / l['file'])
            inter = int((al & ab).sum())
            rows.append({
                'page': page['page'], 'base_id': base['layer_id'],
                'base_name': base['name'], 'base_dr': base['drawn_ratio'],
                'base_cov': nb / tot, 'base_tcs': base['top_colour_share'],
                'base_q16': base['colours_q16'], 'base_fmt': '|'.join(base['formats']),
                'layer_id': l['layer_id'], 'name': l['name'],
                'drawn_ratio': l['drawn_ratio'], 'effective': inter / tot,
                'inside_share': (inter / int(al.sum())) if al.sum() else 0.0,
                'tcs': l['top_colour_share'], 'q16': l['colours_q16'],
                'sat_mean': l['sat_mean'], 'fmt': '|'.join(l['formats']),
                'composite': l['composite'], 'opacity': l['opacity'],
            })
    return rows


def main():
    pages = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    rows, t0 = [], time.time()
    for p in pages:
        n0 = len(rows)
        rows += measure_page(p)
        print(f'{p["page"]:44s} {len(rows)-n0:>4} clipped layers   '
              f'{time.time()-t0:6.1f}s', flush=True)
    with open(OUT / 'clip_pairs.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {OUT/"clip_pairs.csv"} ({len(rows)} rows)')

    print('\n== how wrong is drawn_ratio for a clipped layer? (registration set) ==')
    full = [r for r in rows if r['drawn_ratio'] == 1.0]
    print(f'  clipped layers measured          : {len(rows)}')
    print(f'  of those with drawn_ratio == 1.0 : {len(full)}')
    for name, sel in (('all clipped', rows), ('drawn_ratio == 1.0', full)):
        v = sorted(r['effective'] for r in sel)
        if not v: continue
        print(f'  effective coverage, {name:20s} '
              f'min {v[0]:.6f}  p10 {v[len(v)//10]:.6f}  med {v[len(v)//2]:.6f}  '
              f'p90 {v[-max(1,len(v)//10)]:.6f}  max {v[-1]:.6f}')
    ratio = sorted(r['effective'] / r['drawn_ratio'] for r in rows if r['drawn_ratio'] > 0)
    print(f'  effective / drawn_ratio          : min {ratio[0]:.6f}  '
          f'med {ratio[len(ratio)//2]:.6f}  max {ratio[-1]:.6f}')
    print(f'  layers where drawn_ratio overstates coverage by >= 10x: '
          f'{sum(1 for x in ratio if x <= 0.1)}')
    ins = sorted(r['inside_share'] for r in rows)
    print(f'  share of the clipped layer own pixels that survive the clip: '
          f'min {ins[0]:.4f}  med {ins[len(ins)//2]:.4f}  max {ins[-1]:.4f}')

    print('\n== the clip bases (candidates for "the layer that shows the area") ==')
    bases = {}
    for r in rows:
        bases[(r['page'], r['base_id'])] = r
    print(f'  distinct clip bases: {len(bases)}')
    v = sorted(b['base_cov'] for b in bases.values())
    print(f'  base coverage   min {v[0]:.6f}  p10 {v[len(v)//10]:.6f}  '
          f'med {v[len(v)//2]:.6f}  p90 {v[-max(1,len(v)//10)]:.6f}  max {v[-1]:.6f}')
    v = sorted(b['base_tcs'] for b in bases.values())
    print(f'  base top_colour_share  min {v[0]:.4f}  med {v[len(v)//2]:.4f}  max {v[-1]:.4f}')
    print(f'  base drawn_ratio == 1.0: {sum(1 for b in bases.values() if b["base_dr"]==1.0)}')
    print('  base names (printed last, selector for nothing):',
          collections.Counter(b['base_name'] for b in bases.values()).most_common(8))
    return 0


if __name__ == '__main__':
    sys.exit(main())
