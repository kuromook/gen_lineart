"""Bridge the gaps instead of shrinking the image.

The downscale reading of "closes when you step back" was measured first and
does not work (see work_log 2026-10-10): INTER_AREA downscaling of a thin
stroke *breaks* it into dashes rather than merging its two sides, so the
barrier gets worse as the image shrinks, not better.

This is the other reading of the same idea: a gap narrower than 2k closes
under a morphological closing by a disk of radius k. The barrier is built
from the ROUGH, because that is the image the task has, and because the
silhouette a fill will occupy is still drawn as an outline there -- the GT
with its fills removed has lost that outline with the fill.

Reported per closing radius:
  enclosed   share of GT fill area whose region is not the panel-sized blob
  tightness  median fill area / region area -- a region ten times the fill
             is "closed" on a technicality
  cost       share of the panel the closing turns into barrier; at some k the
             drawing is solid ink and everything closes for the wrong reason

Usage: closure_by_closing.py --root <panel dir> --label ako5_koma [--limit N]
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
RADII = (0, 2, 4, 8, 16, 32)
MIN_FILL_PX = 500
BLOB_FRAC = 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(p for p in (root / "line").iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if args.limit:
        paths = paths[:args.limit]

    rows = []
    for p in paths:
        line = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rp = root / "rough" / p.name
        rough = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if line is None or rough is None:
            continue
        if rough.shape != line.shape:
            rough = cv2.resize(rough, (line.shape[1], line.shape[0]),
                               interpolation=cv2.INTER_AREA)
        fill = fill_mask(ink_of(line))
        nf, labf, stf, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
        keep = [i for i in range(1, nf) if stf[i, cv2.CC_STAT_AREA] >= MIN_FILL_PX]
        if not keep:
            continue
        ink = ink_of(rough)
        panel_px = ink.size
        for k in RADII:
            if k == 0:
                bar = ink
            else:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                bar = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)
            n, lab, st, _ = cv2.connectedComponentsWithStats((~bar).astype(np.uint8), 4)
            cost = float(bar.mean())
            for i in keep:
                m = labf == i
                ids, cnt = np.unique(lab[m], return_counts=True)
                ok = ids > 0
                ra = int(st[ids[ok][np.argmax(cnt[ok])], cv2.CC_STAT_AREA]) if ok.any() else 0
                rows.append({"image": p.name, "k": k, "cost": round(cost, 4),
                             "fill_px": int(m.sum()), "region_px": ra,
                             "panel_px": panel_px,
                             "enclosed": int(0 < ra < BLOB_FRAC * panel_px)})

    R.mkdir(parents=True, exist_ok=True)
    out = R / f"closing_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "k", "cost", "fill_px",
                                          "region_px", "panel_px", "enclosed"])
        w.writeheader()
        w.writerows(rows)

    print(f"{args.label}: {len({r['image'] for r in rows})} panels, "
          f"{len([r for r in rows if r['k'] == RADII[0]])} fill regions >= {MIN_FILL_PX}px")
    print(f"{'close k':>9}{'barrier cost':>14}{'enclosed(area)':>16}{'median tightness':>18}")
    for k in RADII:
        sel = [r for r in rows if r["k"] == k]
        tot = sum(r["fill_px"] for r in sel)
        enc = sum(r["fill_px"] for r in sel if r["enclosed"])
        tight = [r["fill_px"] / r["region_px"] for r in sel if r["enclosed"]]
        print(f"{k:9d}{100 * np.mean([r['cost'] for r in sel]):13.1f}%"
              f"{100.0 * enc / tot:15.1f}%"
              f"{(np.median(tight) if tight else float('nan')):18.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
