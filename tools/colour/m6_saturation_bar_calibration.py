"""M6: calibrate the saturation bar from the delivered 32-bin histograms,
per generation (one work = one generation).

What the old bar was: "fraction of pixels with saturation >= 24, over 2% of
the page". Registration C5 retired its stated basis: the basis was JPEG bleed
reaching saturation ~10, and the census shows every one of the 7,630 previews
is PNG. A lossless preview of a grey page has no bleed floor at all.

PREDICTIONS REGISTERED BEFORE THE ROWS WERE READ (2026-10-24 13:40, this
docstring was written and committed before main() was ever run):

  P7 (no bleed floor): in works the census calls all-monochrome, pages carry
      essentially no saturated pixels at all, because there is no lossy codec
      to invent them. Bar: median per-page fraction of pixels at sat >= 8
      is <= 0.001 across the all-mono works, read as the median of per-WORK
      medians (never pooled across works).
  P8 (the threshold does not matter): between s = 8 and s = 24 at f = 0.02,
      the page classification changes for <= 1% of the census (<= 76 pages).
      If P8 holds, the arbitrariness of 24 touches almost nothing and the
      threshold is not the parameter worth arguing about; f is.
  P9 (the fraction does matter, and the valley is per-generation): inside the
      works the census calls mixed, the per-page fraction at s = 8 is bimodal
      -- at least 80% of their pages fall outside the band 0.002 .. 0.05.
      This is the quantity 追補2 B1 left undecided ("position pending the
      saturation histogram"). Bar: >= 0.80, computed per work and reported as
      the median of per-work values.

A judgement is PASS/FAIL against the bar above and nothing else. The numbers
printed outside judge_* are measurements and carry no verdict.

Configuration of every number: CanvasPreview (CSP's own reduced composite,
scale 1/1..1/16, varying within a work), ALL pixels of the page including
paper white, saturation = max(R,G,B) - min(R,G,B) in sRGB, 32 bins of 8.
Bin k covers saturation 8k .. 8k+7, so "sat >= 8" means bins 1..31.
"""
import csv, collections, statistics, sys
from pathlib import Path

CENSUS = Path('/home/sh1/deepl/lineart/inbox/page_saturation_full.csv')
BINS = [f'sat_{k*8:03d}_{k*8+7:03d}' for k in range(32)]
OLD_S, OLD_F = 24, 0.02          # the old, uncalibrated bar
BAND_LO, BAND_HI = 0.002, 0.05   # P9's band


def load():
    rows = [r for r in csv.DictReader(open(CENSUS, encoding='utf-8'))
            if r['status'] == 'ok']
    for r in rows:
        r['_h'] = [int(r[b]) for b in BINS]
        r['_px'] = int(r['pixels'])
    return rows


def measure_frac_at(row, s):
    """Fraction of the page's pixels at saturation >= s. s must be a bin edge."""
    assert s % 8 == 0, 'only bin edges are measurable from a 8-wide histogram'
    k = s // 8
    return sum(row['_h'][k:]) / row['_px']


def measure_work_classes(rows, s=OLD_S, f=OLD_F):
    byw = collections.defaultdict(list)
    for r in rows:
        byw[(r['年'], r['作品'])].append(r)
    cls = {}
    for k, v in byw.items():
        n = sum(1 for r in v if measure_frac_at(r, 8 * (s // 8)) >= f)
        cls[k] = 'C' if n == len(v) else 'M' if n == 0 else 'X'
    return cls, byw


def measure_per_work_median(byw, works, s):
    return {k: statistics.median(measure_frac_at(r, s) for r in byw[k])
            for k in works}


def measure_switch_count(rows, s_a, s_b, f):
    return sum(1 for r in rows
               if (measure_frac_at(r, s_a) >= f) != (measure_frac_at(r, s_b) >= f))


def measure_outside_band(byw, works, s, lo=BAND_LO, hi=BAND_HI):
    out = {}
    for k in works:
        v = [measure_frac_at(r, s) for r in byw[k]]
        out[k] = sum(1 for x in v if x < lo or x > hi) / len(v)
    return out


# ------------------------------------------------------------------ judge
def judge_no_bleed_floor(per_work_median):
    """P7, registered before the rows were read."""
    med = statistics.median(per_work_median.values())
    return {'name': 'P7 judge_no_bleed_floor', 'registered_before': True,
            'median_of_work_medians_at_sat8': round(med, 8), 'bar': '<= 0.001',
            'verdict': 'PASS' if med <= 0.001 else 'FAIL'}


def judge_threshold_insensitive(switch, n):
    """P8, registered before the rows were read."""
    return {'name': 'P8 judge_threshold_insensitive', 'registered_before': True,
            'pages_switching_between_s8_and_s24': switch, 'of': n,
            'fraction': round(switch / n, 5), 'bar': '<= 0.01',
            'verdict': 'PASS' if switch / n <= 0.01 else 'FAIL'}


def judge_valley_in_mixed(outside):
    """P9, registered before the rows were read."""
    med = statistics.median(outside.values())
    return {'name': 'P9 judge_valley_in_mixed', 'registered_before': True,
            'median_per_work_fraction_outside_band': round(med, 4),
            'band': f'{BAND_LO} .. {BAND_HI}', 'bar': '>= 0.80',
            'verdict': 'PASS' if med >= 0.80 else 'FAIL'}


def main():
    rows = load()
    cls, byw = measure_work_classes(rows)
    mono = [k for k, c in cls.items() if c == 'M']
    mixed = [k for k, c in cls.items() if c == 'X']
    colour = [k for k, c in cls.items() if c == 'C']
    print(f'pages {len(rows)}  works {len(byw)}  '
          f'(all-mono {len(mono)} / mixed {len(mixed)} / all-colour {len(colour)})')
    print(f'bar used for the classes above: sat >= {OLD_S}, fraction >= {OLD_F} (uncalibrated)')

    print('\n== measurements (no verdicts) ==')
    for s in (8, 16, 24, 32, 64):
        for name, works in (('all_mono', mono), ('mixed', mixed), ('all_colour', colour)):
            m = measure_per_work_median(byw, works, s)
            v = sorted(m.values())
            print(f'  sat>={s:3d}  {name:11s} per-work median frac: '
                  f'p10 {v[len(v)//10]:.6f}  median {statistics.median(v):.6f}  '
                  f'p90 {v[-max(1,len(v)//10)]:.6f}')
        print()

    print('  pages switching class as the threshold moves, at f = 0.02:')
    for a, b in ((8, 24), (16, 24), (24, 32), (24, 64)):
        print(f'    s {a:3d} -> {b:3d}: {measure_switch_count(rows, a, b, OLD_F):>5} pages')
    print('\n  pages switching as the FRACTION moves, at s = 8:')
    for f in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        n = sum(1 for r in rows if measure_frac_at(r, 8) >= f)
        print(f'    f = {f:<7}: {n:>5} pages at/above')

    print('\n== judgements ==')
    outside = measure_outside_band(byw, mixed, 8)
    for j in (judge_no_bleed_floor(measure_per_work_median(byw, mono, 8)),
              judge_threshold_insensitive(measure_switch_count(rows, 8, 24, OLD_F), len(rows)),
              judge_valley_in_mixed(outside)):
        reg = 'registered before' if j.pop('registered_before') else 'NOT registered'
        print(f'  {j.pop("name")}: {j.pop("verdict")}  ({reg})')
        for k, v in j.items():
            print(f'      {k} = {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
