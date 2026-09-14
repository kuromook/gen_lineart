"""Track D, hypotheses 3/4 step (a): how well aligned is each training pair?

Stroke-level tracking (`doc/work_log.md` 2026-09-15) showed the generator
inherits its stroke set from the condition map, and that part of the
condition-vs-GT offset is already in the raw rough/GT pairs. Separating
hypothesis 3 (the mapping is too loose to learn placement from) from
hypothesis 4 (a generative objective cannot express selection) needs a model
trained on well-aligned pairs only -- which first needs every training pair
scored, with the same definition used on the holdout so the numbers compare.

Per pair, GT line art is thinned to 1px (skimage) and split into segments at
crossing-number junctions (same unit as `stroke_churn.py`). For each source
-- the raw rough (gray < 200), `manga_line` (> 32, white on black) and
`lineart_coarse` (> 32, white on black) -- the length-weighted share of GT
segments whose median distance to the source skeleton is <= 3px, 3-8px, or
> 8px.

Denser sources cover more GT by chance, and density varies tile to tile, so
each tile also gets its own chance level: the same measurement against the
source rotated 180 degrees (same density and orientation statistics,
position destroyed except for structure symmetric about the centre).
`*_aligned` = matched <= 3px share minus rotated <= 3px share. A left-right
flip was tried first and rejected: panel borders and vertical rules survive a
flip in place, so a tile could score 43% by "chance".

Outputs `per_pair.csv` and prints distributions per source and per data
source prefix, plus how many pairs clear a few alignment thresholds.
"""

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_churn import dist_to, ink_skeleton, load_gray, segments  # noqa: E402

TRACK_A_DATA = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data")
SOURCES = {
    "rough": ("rough", "dark", 200),
    "manga_line": ("rough_manga_line", "bright", 32),
    "lineart_coarse": ("rough_lineart_coarse", "bright", 32),
}


def source_skeleton(gray, polarity, threshold):
    from skimage.morphology import skeletonize

    mask = gray < threshold if polarity == "dark" else gray > threshold
    n = int(mask.sum())
    if n < 20 or mask.size - n < 20:
        return np.zeros_like(mask)
    return skeletonize(mask)


def shares(dist, segs):
    acc = np.zeros(3)
    for rr, cc in segs:
        m = float(np.median(np.minimum(dist[rr, cc], 99)))
        acc[0 if m <= 3 else (1 if m <= 8 else 2)] += len(rr)
    total = acc.sum()
    return acc / total if total > 0 else np.full(3, np.nan)


def score_pair(args):
    name, data_dir, min_seg = args
    data_dir = Path(data_dir)
    gt = load_gray(data_dir / "line" / name)
    segs = segments(ink_skeleton(gt), min_seg)
    row = {"name": name, "source": name.split("_")[0], "gt_len": int(sum(len(r) for r, _ in segs)),
           "gt_ink": float((gt < 128).mean())}
    for key, (sub, polarity, thr) in SOURCES.items():
        sk = source_skeleton(load_gray(data_dir / sub / name), polarity, thr)
        m = shares(dist_to(sk), segs)
        f = shares(dist_to(np.ascontiguousarray(np.rot90(sk, 2))), segs)
        row.update({
            f"{key}_le3": m[0], f"{key}_3to8": m[1], f"{key}_gt8": m[2],
            f"{key}_chance_le3": f[0], f"{key}_aligned": m[0] - f[0],
            f"{key}_density": float(sk.mean()),
        })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(TRACK_A_DATA))
    parser.add_argument("--list", default=str(TRACK_A_DATA / "train_list.txt"))
    parser.add_argument("--min-seg-px", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", default="results/pair_alignment_strata_20260915")
    args = parser.parse_args()

    names = [l.strip() for l in open(args.list) if l.strip()]
    if args.limit:
        names = names[: args.limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(score_pair, [(n, args.data_dir, args.min_seg_px) for n in names], chunksize=8)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(names)}", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["name"])

    with open(out_dir / "per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["gt_len"] > 0 and not np.isnan(r["rough_le3"])]
    print(f"\n{len(rows)} pairs scored, {len(valid)} with GT strokes")
    print("\nper source (length-weighted share of each pair's GT strokes; distribution over pairs p10 / p50 / p90)")
    for key in SOURCES:
        for col in ("le3", "3to8", "gt8", "chance_le3", "aligned"):
            v = np.array([r[f"{key}_{col}"] for r in valid])
            print(f"  {key:<15} {col:<11} mean {v.mean():.3f}   p10 {np.percentile(v, 10):.3f}  p50 {np.percentile(v, 50):.3f}  p90 {np.percentile(v, 90):.3f}")

    print("\nby data source prefix: n, mean rough_aligned, mean manga_line_aligned, mean lineart_coarse_aligned, mean rough 3-8px")
    by = defaultdict(list)
    for r in valid:
        by[r["source"]].append(r)
    for src, rs in sorted(by.items()):
        print(f"  {src:<15} n={len(rs):>5}  rough {np.mean([r['rough_aligned'] for r in rs]):.3f}  "
              f"manga_line {np.mean([r['manga_line_aligned'] for r in rs]):.3f}  "
              f"lineart_coarse {np.mean([r['lineart_coarse_aligned'] for r in rs]):.3f}  "
              f"rough 3-8px {np.mean([r['rough_3to8'] for r in rs]):.3f}")

    print("\npairs clearing alignment thresholds (count of valid pairs)")
    for key in SOURCES:
        v = np.array([r[f"{key}_aligned"] for r in valid])
        le3 = np.array([r[f"{key}_le3"] for r in valid])
        print(f"  {key:<15} aligned>=0.3: {int((v >= 0.3).sum())}  >=0.5: {int((v >= 0.5).sum())}  >=0.7: {int((v >= 0.7).sum())}"
              f"   | matched<=3px >=0.6: {int((le3 >= 0.6).sum())}  >=0.8: {int((le3 >= 0.8).sum())}")

    a = np.array([[r["rough_aligned"], r["manga_line_aligned"], r["lineart_coarse_aligned"]] for r in valid])
    c = np.corrcoef(a.T)
    print(f"\ncorrelation of per-pair aligned scores: rough~manga_line {c[0, 1]:.3f}, rough~coarse {c[0, 2]:.3f}, manga_line~coarse {c[1, 2]:.3f}")
    print(f"\nsaved: {out_dir / 'per_pair.csv'}")


if __name__ == "__main__":
    main()
