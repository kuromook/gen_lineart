"""The region (面) decomposition of a koma panel's GT, and how much of the
solid fill it actually accounts for.

If the unit of judgement is the region, something has to say what the regions
are. The decomposition used here:

    ink       = gray < 128
    fill      = fill_mask(ink)              # the solid areas, radius 4
    line_ink  = ink & ~fill                 # the strokes, fills removed
    regions   = connected components of ~line_ink, area >= MIN_REGION_PX
    filled(r) = mean(fill inside r) >= 0.5

The known weak point, measured here rather than assumed: a ベタ whose *own
edge* is the boundary -- no separate contour drawn around it -- loses that
boundary with the fill, so its region merges into the surrounding paper and
the fill becomes `orphan` area, sitting inside a region labelled not-filled.
`orphan_frac` is that quantity. It is the ceiling on what a region classifier
can reach however good it is, so it is read before anything is trained.

Usage:
  region_decompose.py --dir <line dir> --label ako5_koma [--limit N]
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402

R = Path("results/region_closure_20261010")
MIN_REGION_PX = 500
FILLED_FRAC = 0.5


def one(gray, name, label):
    ink = ink_of(gray)
    fill = fill_mask(ink)
    line_ink = ink & ~fill
    n, lab, st, _ = cv2.connectedComponentsWithStats((~line_ink).astype(np.uint8), 4)
    fill_px = int(fill.sum())
    row = {"set": label, "image": name, "h": gray.shape[0], "w": gray.shape[1],
           "fill_px": fill_px, "regions": 0, "filled": 0,
           "captured_px": 0, "paper_px": 0, "big_filled": 0}
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < MIN_REGION_PX:
            continue
        m = lab == i
        f = int((fill & m).sum())
        row["regions"] += 1
        if f >= FILLED_FRAC * area:
            row["filled"] += 1
            row["captured_px"] += f
            if area >= 20 * MIN_REGION_PX:
                row["big_filled"] += 1
        else:
            row["paper_px"] += area - f
    row["orphan_px"] = fill_px - row["captured_px"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if args.limit:
        paths = paths[:args.limit]
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            rows.append(one(g, p.name, args.label))

    R.mkdir(parents=True, exist_ok=True)
    out = R / f"regions_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    reg = sum(r["regions"] for r in rows)
    fil = sum(r["filled"] for r in rows)
    fpx = sum(r["fill_px"] for r in rows)
    orp = sum(r["orphan_px"] for r in rows)
    print(f"{'set':14}{'panels':>8}{'regions':>9}{'/panel':>8}{'filled':>9}"
          f"{'baserate':>10}{'orphanfill':>12}")
    print(f"{args.label:14}{len(rows):8d}{reg:9d}{reg / len(rows):8.1f}{fil:9d}"
          f"{(100.0 * fil / reg if reg else 0):9.1f}%"
          f"{(100.0 * orp / fpx if fpx else 0):11.1f}%")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
