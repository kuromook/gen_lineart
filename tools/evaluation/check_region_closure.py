"""Does a solid-fill region close inside the input window?

A design check for stage 2, not a stage-1 claim. Stage 1 scored on 480px
windows because that is the unit every existing pool and the adopted
checkpoint use. If the unit of judgement is the region (面), the window has
to be large enough for a region's own boundary to be inside it -- otherwise
the model is being asked about an object it is never shown whole.

Measured per image, on the GT line art only (no model, no arm):

  comp_total       fill-mask connected components of >= MIN_COMP_PX
  comp_border      those touching the image border
  frame_like       border-touching components whose bbox spans >= FRAME_SPAN
                   of both sides -- on a koma panel this is usually the panel
                   frame itself, which is thick ink and legitimately enters
                   the fill mask. Reported separately because it inflates the
                   koma side, i.e. against the hypothesis being tested.
  area_border      share of fill AREA sitting in border-touching components

Scale is not assumed equal between the two sets: `stroke_px` is printed so a
reader can see whether the two units are being compared at the same drawing
scale. It is measured over ink *outside* the fill mask -- over all ink it is
not a stroke width at all but a reading of the solid areas, which is the very
thing under test, and on a koma panel it comes out 5x too large.

Usage:
  check_region_closure.py --tiles housei_test --split test
  check_region_closure.py --dir /path/to/koma_panels/line --label ako5_koma
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
R = Path("results/region_closure_20261010")
MIN_COMP_PX = 200      # ~ HAS_FILL_AREA (0.001 of a 480px window)
FRAME_SPAN = 0.80


def stats(gray, label, name):
    ink = ink_of(gray)
    fill = fill_mask(ink)
    h, w = gray.shape
    row = {"set": label, "image": name, "h": h, "w": w,
           "ink_ratio": round(float(ink.mean()), 6),
           "fill_area_frac": round(float(fill.mean()), 6),
           "stroke_px": 0.0, "comp_total": 0, "comp_border": 0,
           "frame_like": 0, "fill_px": int(fill.sum()), "border_px": 0}
    if not fill.any():
        if ink.any():
            d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
            row["stroke_px"] = round(float(2 * np.median(d[ink])), 3)
        return row
    line_ink = ink & ~fill
    if line_ink.any():
        d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        row["stroke_px"] = round(float(2 * np.median(d[line_ink])), 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < MIN_COMP_PX:
            continue
        row["comp_total"] += 1
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches:
            row["comp_border"] += 1
            row["border_px"] += int(area)
            if bw >= FRAME_SPAN * w and bh >= FRAME_SPAN * h:
                row["frame_like"] += 1
    return row


def summarise(rows, label):
    n = len(rows)
    ct = sum(r["comp_total"] for r in rows)
    cb = sum(r["comp_border"] for r in rows)
    fl = sum(r["frame_like"] for r in rows)
    fp = sum(r["fill_px"] for r in rows)
    bp = sum(r["border_px"] for r in rows)
    px = np.median([r["stroke_px"] for r in rows if r["stroke_px"] > 0] or [0])
    side = np.median([min(r["h"], r["w"]) for r in rows])
    print(f"{label:16}{n:6d}{side:8.0f}{px:10.2f}{ct:9d}"
          f"{(100.0 * cb / ct if ct else 0):10.1f}%"
          f"{(100.0 * (cb - fl) / ct if ct else 0):12.1f}%"
          f"{(100.0 * bp / fp if fp else 0):11.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", help="list basename under baseline_fill_20261009/lists")
    ap.add_argument("--split", choices=("train", "test"))
    ap.add_argument("--dir", help="directory of GT line images at native scale")
    ap.add_argument("--label", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor, for matching stroke scale")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    if args.tiles:
        lst = Path("results/baseline_fill_20261009/lists") / f"{args.tiles}.txt"
        tiles = [l.strip() for l in open(lst) if l.strip()]
        if args.limit:
            tiles = tiles[:args.limit]
        for t in tiles:
            stem = t[:-4] if t.endswith(".jpg") else t
            g = cv2.imread(str(SHARED / args.split / "line" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            rows.append(stats(g, args.label, stem))
    else:
        paths = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
        if args.limit:
            paths = paths[:args.limit]
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if args.scale != 1.0:
                g = cv2.resize(g, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
            rows.append(stats(g, args.label, p.name))

    R.mkdir(parents=True, exist_ok=True)
    out = R / f"closure_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'set':16}{'n':>6}{'side':>8}{'stroke':>10}{'comps':>9}"
          f"{'border':>11}{'-frame':>12}{'areaborder':>12}")
    summarise(rows, args.label)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
