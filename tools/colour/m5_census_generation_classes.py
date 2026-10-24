"""M5: the full census (7,634 pages) read per generation, and the 790-page
population for request 3.

Separation required by the discipline notice (f112ccf, point 3):
    measure_*   returns numbers only
    verify_*    recomputes a number the extraction side reported (reconciliation)
    judge_*     PASS/FAIL against a stated prediction, and says whether that
                prediction was registered BEFORE the rows were read

For M, each work is a generation. Nothing here is pooled across works except
where a line says "pooled" in so many words.

Inputs (read-only):
    ../lineart/inbox/page_saturation_full.csv   7,634 x 47, delivered 11:51
    inbox/work_color_mode_pages.csv             the old 5-page sample, 1,348 rows
    ../lineart/inbox/colour_layers_manifest_pilot.json   the 8 pilot pages

Configuration of every colour number below: measured on CanvasPreview (CSP's
own reduced composite, scale 1/1..1/16 and varying WITHIN a work), over ALL
pixels of the page including paper white, saturation = max(R,G,B)-min(R,G,B)
in sRGB (not HSV S). The 0.02 bar is the old table's definition and is NOT
calibrated -- see registration C5: its stated basis (JPEG bleed) does not hold,
every preview is PNG.
"""
import csv, json, collections, sys
from pathlib import Path

CENSUS = Path('/home/sh1/deepl/lineart/inbox/page_saturation_full.csv')
SAMPLE = Path('inbox/work_color_mode_pages.csv')
PILOT  = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
COLOUR_BAR = 0.02          # the old table's bar, uncalibrated
FAINT_BAR  = 0.0           # population floor for request 3: strictly greater


# ---------------------------------------------------------------- measure
def measure_census(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    ok = [r for r in rows if r['status'] == 'ok']
    hist_bad = 0
    for r in ok:
        if sum(int(r[k]) for k in r if k.startswith('sat_')) != int(r['pixels']):
            hist_bad += 1
    return {'rows': len(rows), 'ok': len(ok),
            'works': len({(r['年'], r['作品']) for r in rows}),
            'dup_page_path': len(rows) - len({r['page_path'] for r in rows}),
            'hist_sum_mismatch': hist_bad,
            'errors': [(r['年'], r['作品'], r['ページ'], r['status'])
                       for r in rows if r['status'] != 'ok']}, ok


def classify(ratios, bar=COLOUR_BAR):
    """C = every page at/above the bar, M = none, X = mixed. One generation."""
    n = sum(1 for x in ratios if x >= bar)
    return 'C' if n == len(ratios) else 'M' if n == 0 else 'X'


def measure_work_classes(rows, ratio_key, bar=COLOUR_BAR):
    byw = collections.defaultdict(list)
    for r in rows:
        byw[(r['年'], r['作品'])].append(float(r[ratio_key]))
    return {k: classify(v, bar) for k, v in byw.items()}, byw


def measure_reclass(old_cls, new_cls):
    """Works whose generation class moved from the 5-page sample to the census.
    'toward_mixed' = lost purity, 'toward_pure' = gained it."""
    moved, toward_pure = [], []
    for k, a in old_cls.items():
        b = new_cls.get(k)
        if b is None or a == b:
            continue
        moved.append((k, a, b))
        if b in ('C', 'M'):
            toward_pure.append((k, a, b))
    return moved, toward_pure


def measure_population(ok, new_cls, pilot_paths):
    """The request-3 population: status=ok and color_ratio > 0, banded.
    The bands only order the run; they are not a bar on anything."""
    pos = [r for r in ok if float(r['color_ratio']) > FAINT_BAR]
    out = []
    for r in pos:
        cr = float(r['color_ratio'])
        band = ('D_faint' if cr < COLOUR_BAR
                else 'B_allcolour' if new_cls[(r['年'], r['作品'])] == 'C'
                else 'C_mixed')
        out.append((r, band, r['page_path'] in pilot_paths))
    return out


# ---------------------------------------------------------------- verify
def verify_reported(census):
    """Recompute what note_m_page_saturation_20261024.md reported. These are
    reconciliations of someone else's numbers, not predictions of ours."""
    want = {'rows': 7634, 'ok': 7630, 'works': 289,
            'dup_page_path': 0, 'hist_sum_mismatch': 0}
    return {k: (census[k], want[k], census[k] == want[k]) for k in want}


# ---------------------------------------------------------------- judge
def judge_reclass_one_way(toward_pure, moved):
    """Prediction: no work can look PURER in the census than in a 5-page
    sample of itself, because a sample can only miss colour pages, never
    invent them. Bar: len(toward_pure) == 0.

    NOT registered before the rows were read -- it was observed first and is
    recorded here as a measurement with a mechanism attached (lesson 15: the
    mechanism is a registered prediction for the NEXT table, not evidence
    from this one).
    """
    return {'name': 'judge_reclass_one_way', 'registered_before': False,
            'moved': len(moved), 'toward_pure': len(toward_pure),
            'verdict': 'PASS' if not toward_pure else 'FAIL'}


def judge_known_cell_pilot(pop, pilot_paths):
    """Registered BEFORE the request-3 delivery (registration C3/C4): the 8
    pilot pages are re-run inside the 790, and the delivered layer rows must
    reproduce the pilot manifest. Here we can only check the 8 are in the
    population at all; the reproduction itself is scored on delivery.
    Bar: all 8 present.
    """
    got = sum(1 for _, _, k in pop if k)
    return {'name': 'judge_known_cell_pilot', 'registered_before': True,
            'pilot_pages_in_population': got, 'expected': len(pilot_paths),
            'verdict': 'PASS' if got == len(pilot_paths) else 'FAIL',
            'coverage_note': 'known cell covers 4 of 118 generations (C4)'}


def main():
    census, ok = measure_census(CENSUS)
    print('== census (delivered 2026-10-24 11:51) ==')
    for k, (got, want, good) in verify_reported(census).items():
        print(f'  verify {k:20s} {got:>6} (reported {want}) {"OK" if good else "MISMATCH"}')
    for e in census['errors']:
        print('  error row:', ' '.join(e))

    new_cls, new_by = measure_work_classes(ok, 'color_ratio')
    old_rows = list(csv.DictReader(open(SAMPLE, encoding='utf-8-sig')))
    old_cls, old_by = measure_work_classes(old_rows, '色画素率')

    print(f'\n== generation classes, bar color_ratio >= {COLOUR_BAR} (uncalibrated) ==')
    print(f'{"":12s} {"sample(5p/work)":>16s} {"census":>8s} {"census pages":>13s}')
    for c, name in (('C', 'all_colour'), ('X', 'mixed'), ('M', 'all_mono')):
        pg = sum(len(v) for k, v in new_by.items() if new_cls[k] == c)
        print(f'{name:12s} {sum(1 for x in old_cls.values() if x==c):>16} '
              f'{sum(1 for x in new_cls.values() if x==c):>8} {pg:>13}')

    moved, toward_pure = measure_reclass(old_cls, new_cls)
    print(f'\n== reclassified works: {len(moved)} ==')
    for (y, w), a, b in sorted(moved, key=lambda t: (t[1], t[2], t[0])):
        print(f'  {y:6s} {w:22s} {a}->{b}   sample {len(old_by[(y,w)])}p / census {len(new_by[(y,w)])}p')

    pilot = {p['source'] for p in json.load(open(PILOT))['pages']}
    pop = measure_population(ok, new_cls, pilot)
    band = collections.Counter(b for _, b, _ in pop)
    print(f'\n== request-3 population: color_ratio > {FAINT_BAR}, {len(pop)} pages ==')
    for b in ('B_allcolour', 'C_mixed', 'D_faint'):
        works = len({(r['年'], r['作品']) for r, bb, _ in pop if bb == b})
        print(f'  {b:14s} {band[b]:>5} pages  {works:>4} works')
    print(f'  works total  {len({(r["年"],r["作品"]) for r,_,_ in pop}):>5}')

    print('\n== judgements (separate from the measurements above) ==')
    for j in (judge_reclass_one_way(toward_pure, moved),
              judge_known_cell_pilot(pop, pilot)):
        reg = 'registered before' if j.pop('registered_before') else 'NOT registered before'
        print(f'  {j.pop("name")}: {j.pop("verdict")}  ({reg})')
        for k, v in j.items():
            print(f'      {k} = {v}')
    print('\nNo arm is scored in this file. A0 and N1/N2/N3 wait for 追補4.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
