"""M7: read the distribution of the pilot's 706 layers, so the cuts for
B1..B4 can be registered (registration 3.1 licenses exactly this: "read the
distribution of the delivered layers, then register where to cut").

This file MEASURES ONLY. It contains no judge_* function and scores no arm.
The cuts it feeds into are registered in 追補5 and are then frozen for the
test set (the 782 + 82 pages still being exported).

Registration set: the 8 pilot pages / 4 works. Anything read here is read on
the set where the cuts are chosen, so no claim may be made from it.

Names are NOT used to select anything here. `name` is carried only so that the
later falsifier step (identify by B1..B4, THEN print the name-mismatch rate)
has something to compare against, and the name columns are printed last.

Configuration: native-size RGBA layer exports, alpha = coverage, colour = the
layer's own colour. drawn_ratio / sat_hist / sat_mean / colours_q16 /
top_colour_share are the extraction side's per-layer statistics over drawn
pixels (alpha >= 26). sat_hist is 32 bins of 8 over saturation max-min.
"""
import json, collections, statistics, sys
from pathlib import Path

MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
PAPER_TYPE = 1584


def load():
    m = json.load(open(MANIFEST, encoding='utf-8'))
    return m['pages']


def q(vals, name, fmt='{:.5f}'):
    v = sorted(vals)
    if not v:
        return f'  {name:28s} (none)'
    f = lambda x: fmt.format(x)
    return (f'  {name:28s} n={len(v):<4} min {f(v[0])}  p10 {f(v[len(v)//10])}  '
            f'med {f(statistics.median(v))}  p90 {f(v[-max(1,len(v)//10)])}  max {f(v[-1])}')


def measure_shape(pages):
    """What the 8 pages are made of, per page. Never pooled."""
    print('== per page ==')
    print(f'{"page":44s} {"layers":>6} {"paper":>5} {"zerodrw":>7} {"clip":>5} '
          f'{"hidden":>6} {"grey":>5} {"rgb":>4} {"alpha":>5}')
    for p in pages:
        L = p['layers']
        fmt = collections.Counter(f for l in L for f in l['formats'])
        n2, n5, n1 = fmt.get('2', 0), fmt.get('5', 0), fmt.get('1', 0)
        zd = sum(1 for l in L if l['drawn_ratio'] == 0)
        print(f'{p["page"]:44s} {len(L):>6} '
              f'{sum(1 for l in L if l["layer_type"]==PAPER_TYPE):>5} '
              f'{zd:>7} {sum(1 for l in L if l["clip"]):>5} '
              f'{sum(1 for l in L if not l["visibility"]):>6} '
              f'{n2:>5} {n5:>4} {n1:>5}')


def measure_fields(pages, label, keep):
    """Distribution of the per-layer quantities, per page (no pooling)."""
    print(f'\n== {label} ==')
    for p in pages:
        L = [l for l in p['layers'] if keep(l)]
        if not L:
            print(f'\n{p["page"]}  (no layer passes the filter)')
            continue
        print(f'\n{p["page"]}  ({len(L)} layers)')
        print(q([l['drawn_ratio'] for l in L], 'drawn_ratio'))
        print(q([l['top_colour_share'] for l in L], 'top_colour_share'))
        print(q([l['colours_q16'] for l in L], 'colours_q16', '{:.0f}'))
        print(q([l['sat_mean'] for l in L], 'sat_mean', '{:.2f}'))
        print(q([l['opacity'] / 256 for l in L], 'opacity/256', '{:.3f}'))


def measure_crosstab(pages, keep, label):
    """Pooled ONLY across layers within the registration set, and said so:
    this is the shape the cuts are chosen from, not a claim."""
    L = [l for p in pages for l in p['layers'] if keep(l)]
    print(f'\n== {label} (pooled over the 8 registration pages, {len(L)} layers) ==')
    print('  composite raw value:', dict(collections.Counter(l['composite'] for l in L).most_common()))
    print('  opacity != 256     :', sum(1 for l in L if l['opacity'] != 256))
    print('  clip = 1           :', sum(1 for l in L if l['clip']))
    print('  visibility = 0     :', sum(1 for l in L if not l['visibility']))
    print('  has_mask           :', sum(1 for l in L if l['has_mask']))
    print('  draw_color_enable  :', sum(1 for l in L if l['draw_color_enable']))
    print('  formats            :', dict(collections.Counter(f for l in L for f in l['formats'])))

    print('\n  joint of flatness and coverage (the B1 x B2 plane):')
    sh = [0.0, 0.5, 0.75, 0.9, 0.99]
    dr = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]
    print('     top_colour_share \\ drawn_ratio  ' + ''.join(f'{d:>9}' for d in dr))
    for i, s in enumerate(sh):
        s_hi = sh[i + 1] if i + 1 < len(sh) else 1.01
        row = []
        for j, d in enumerate(dr):
            d_hi = dr[j + 1] if j + 1 < len(dr) else 1.01
            row.append(sum(1 for l in L if s <= l['top_colour_share'] < s_hi
                           and d <= l['drawn_ratio'] < d_hi))
        print(f'     [{s:.2f},{s_hi:.2f})                    ' + ''.join(f'{c:>9}' for c in row))

    print('\n  saturation of drawn pixels, by format (sat_mean):')
    for f in ('2', '5'):
        v = [l['sat_mean'] for l in L if f in l['formats']]
        print(q(v, f'format {f}', '{:.2f}'))

    print('\n  colours_q16 against top_colour_share:')
    for lo, hi in ((1, 2), (2, 5), (5, 17), (17, 65), (65, 257), (257, 10**9)):
        v = [l['top_colour_share'] for l in L if lo <= l['colours_q16'] < hi]
        print(q(v, f'colours_q16 in [{lo},{hi})', '{:.4f}'))


def measure_order(pages, keep):
    """B3: is the flat layer BELOW the gradient layer inside the same folder?
    order_in_parent: 0 is the bottom (the extraction side verified the
    direction by recomposition on all 8 pages). Reported as the raw joint of
    order and flatness per folder, with no cut applied."""
    print('\n== B3 input: order inside a folder, against flatness ==')
    pairs = below = above = ties = 0
    for p in pages:
        byparent = collections.defaultdict(list)
        for l in p['layers']:
            if keep(l):
                byparent[l['parent_id']].append(l)
        for pid, sib in byparent.items():
            if len(sib) < 2:
                continue
            for a in sib:
                for b in sib:
                    if a['order_in_parent'] >= b['order_in_parent']:
                        continue
                    # a is below b. is the lower one flatter?
                    pairs += 1
                    da = a['top_colour_share'] - b['top_colour_share']
                    if da > 0.05:
                        below += 1
                    elif da < -0.05:
                        above += 1
                    else:
                        ties += 1
    print(f'  sibling pairs: {pairs}')
    print(f'    lower layer flatter by >0.05 : {below}')
    print(f'    upper layer flatter by >0.05 : {above}')
    print(f'    within 0.05                  : {ties}')
    print('  (no cut registered from this line yet -- see 追補5 B3)')


def measure_names_last(pages, keep):
    """Printed LAST and used for nothing above. This is the falsifier material
    for the step after identification, not a selector."""
    L = [l for p in pages for l in p['layers'] if keep(l)]
    print('\n== name material (selector for nothing; falsifier only) ==')
    for token in ('ベース', '基本', '下塗', '線', 'グラデ', '影', 'ハイライト', 'トーン', '肌', '髪'):
        n = sum(1 for l in L if token in l['name'])
        print(f'  name contains {token:8s}: {n:4d}')
    print('  distinct names:', len({l['name'] for l in L}))


def main():
    pages = load()
    tot = sum(len(p['layers']) for p in pages)
    print(f'pilot: {len(pages)} pages, {tot} layers '
          f'(skipped by the extraction side: '
          f'{sum(len(p.get("skipped") or []) for p in pages)})')
    print('REGISTRATION SET -- cuts are chosen here, so no claim comes from it.\n')
    measure_shape(pages)

    drawn = lambda l: l['layer_type'] != PAPER_TYPE and l['drawn_ratio'] > 0
    measure_fields(pages, 'per-layer quantities, paper excluded', drawn)
    measure_crosstab(pages, drawn, 'shape of the registration set')
    measure_order(pages, drawn)
    measure_names_last(pages, drawn)
    print('\nNo cut is applied and no arm is scored in this file.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
