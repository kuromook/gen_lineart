"""Diagnostic D1: is a solid fill's OUTLINE as curve-like as a drawn stroke?

Registered 2026-10-10 in doc/work_log.md. A fill's medial axis is an artefact
of skeletonizing a filled region -- nobody drew the lattice -- and it is 51.6%
of all skeleton length and 60.3% of everything the curve model fails on. The
drawn mark at a fill is its outline. So the repair is to drop the lattice and
take the contour, which only works if the contour is a curve of ordinary
complexity.

Measure, identical for both populations: split recursively, each piece fitted
in its own PCA frame, until every piece is within RMS 1.0px (cap 20 pieces),
and report **pieces per 100px**.

  bar: 対処可能 if the median for fill contours is within 2x the median for
       drawn chains outside the fill. Fixed before the first run.

NC-d1: a random closed curve with the SAME perimeter and the same
isoperimetric ratio P^2/(4 pi A) as the real region. Named error: *pieces per
100px is just a function of how long and how blobby a closed curve is, and
any closed region of this size would score the same.*

Usage:
  fill_contour_curves.py --root <panel dir> --limit 40
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402
from stroke_chain import assemble, build_links, path_len, seg_rms  # noqa: E402
from stroke_curves import SPLIT_MARGIN, SPLIT_MIN_SIDE, SPLIT_STEPS  # noqa: E402

R = Path("results/fill_contour_20261010")
RESID_PX = 1.0
CAP = 20
MIN_PX = 30.0
SEED = 20261010


def count_pieces(p, budget):
    """Pieces needed to bring every piece within RESID_PX. `budget` caps the
    recursion so a pathological curve cannot cost unbounded time; a curve that
    hits the cap is recorded as having hit it rather than quietly truncated."""
    if len(p) < 2 * SPLIT_MIN_SIDE or seg_rms(p) <= RESID_PX or budget <= 1:
        return 1
    lo = max(int(SPLIT_MARGIN * len(p)), SPLIT_MIN_SIDE)
    hi = min(int((1 - SPLIT_MARGIN) * len(p)), len(p) - SPLIT_MIN_SIDE)
    if hi <= lo:
        return 1
    step = max(1, (hi - lo) // SPLIT_STEPS)
    best, at = float("inf"), -1
    for b in range(lo, hi, step):
        r = max(seg_rms(p[:b]), seg_rms(p[b:]))
        if r < best:
            best, at = r, b
    if at < 0:
        return 1
    left = count_pieces(p[:at], budget - 1)
    return left + count_pieces(p[at:], max(1, budget - left))


def iso_ratio(per, area):
    return per * per / (4 * np.pi * area) if area > 0 else float("inf")


def random_closed(per, iso, rng, k_max=8):
    """A closed curve with the given perimeter and isoperimetric ratio.
    r(theta) = R (1 + sum_k a_k cos(k theta + phi_k)); the roughness is scaled
    until the ratio matches, then the whole curve is scaled to the perimeter."""
    ks = np.arange(2, k_max + 1)
    phis = np.array([rng.uniform(0, 2 * np.pi) for _ in ks])
    amp = np.array([1.0 / k for k in ks])
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def shape(s):
        r = 1.0 + (s * amp[:, None] * np.cos(ks[:, None] * th[None, :] + phis[:, None])).sum(0)
        r = np.clip(r, 0.05, None)
        xy = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        d = np.diff(np.vstack([xy, xy[:1]]), axis=0)
        P = float(np.hypot(d[:, 0], d[:, 1]).sum())
        A = 0.5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                            - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))
        return xy, P, A

    lo, hi = 0.0, 2.0
    for _ in range(30):                      # bisect the roughness to match iso
        mid = 0.5 * (lo + hi)
        xy, P, A = shape(mid)
        if iso_ratio(P, A) < iso:
            lo = mid
        else:
            hi = mid
    xy, P, A = shape(0.5 * (lo + hi))
    xy = xy * (per / P)
    n = max(int(per), 16)                    # resample at ~1px spacing
    idx = np.linspace(0, len(xy), n, endpoint=False).astype(int) % len(xy)
    return xy[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    rng = random.Random(SEED)
    paths = sorted(p for p in (Path(args.root) / args.which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[:args.limit or None]

    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        fill = fill_mask(ink)

        # drawn chains outside the fill -- the reference population, built by
        # exactly the chainer the stroke measurement used
        frags, link, absorbed, _ = build_links(ink, "real", random.Random(SEED))
        for pts, cyc, nf in assemble(frags, link, skip=absorbed):
            L = path_len(pts)
            if L < MIN_PX or cyc:
                continue
            xy = np.round(pts).astype(int)
            h, w = fill.shape
            ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            if not ok.any() or fill[xy[ok, 1], xy[ok, 0]].mean() >= 0.5:
                continue
            rows.append({"image": p.name, "kind": "drawn_chain", "len_px": round(L, 1),
                         "pieces": count_pieces(pts.astype(np.float64), CAP)})

        # fill contours, and a size-and-blobbiness-matched random control
        cs, _ = cv2.findContours(fill.astype(np.uint8), cv2.RETR_CCOMP,
                                 cv2.CHAIN_APPROX_NONE)
        for c in cs:
            q = c.reshape(-1, 2).astype(np.float64)
            L = path_len(np.vstack([q, q[:1]]))
            if L < MIN_PX:
                continue
            A = abs(float(cv2.contourArea(c)))
            rows.append({"image": p.name, "kind": "fill_contour", "len_px": round(L, 1),
                         "pieces": count_pieces(q, CAP)})
            if A > 4:
                rc = random_closed(L, iso_ratio(L, A), rng)
                rows.append({"image": p.name, "kind": "nc_d1_random", "len_px": round(L, 1),
                             "pieces": count_pieces(rc, CAP)})

    R.mkdir(parents=True, exist_ok=True)
    out = R / "pieces.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "kind", "len_px", "pieces"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(paths)} images\n")
    print(f"{'population':>16}{'n':>8}{'px':>11}{'med len':>9}"
          f"{'med pieces/100px':>18}{'mean':>8}{'hit cap %':>11}")
    med = {}
    for kind in ("drawn_chain", "fill_contour", "nc_d1_random"):
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        per = sorted(100.0 * r["pieces"] / r["len_px"] for r in sel)
        lens = sorted(r["len_px"] for r in sel)
        m = per[len(per) // 2]
        med[kind] = m
        print(f"{kind:>16}{len(sel):8d}{int(sum(lens)):11d}{lens[len(lens) // 2]:9.0f}"
              f"{m:18.2f}{sum(per) / len(per):8.2f}"
              f"{100.0 * sum(1 for r in sel if r['pieces'] >= CAP) / len(sel):10.1f}%")
    if "drawn_chain" in med and "fill_contour" in med:
        ratio = med["fill_contour"] / med["drawn_chain"]
        print(f"\nfill contour / drawn chain = {ratio:.2f}x   bar is 2.00x  "
              f"-> {'対処可能 PASS' if ratio <= 2.0 else '対処不能 FAIL'}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
