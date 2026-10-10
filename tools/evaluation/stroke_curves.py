"""What curve is a stroke following?

Registered 2026-10-10 in doc/work_log.md (items 2 and 4), from the user's
hypothesis: most lines are a straight line or a parabolic arch, and some
change to a different curve partway. The fit is the shared instrument behind
both of that day's open items -- cutting a line that runs out of a solid fill
(it stops following the fill's boundary) and bridging a gap the eye closes
(extend the stroke along its own curve, which adds ink only where the drawing
was already going; isotropic closing swallows the neighbourhood instead).

Pipeline per image:

  skeletonize -> cut at junction pixels (8-neighbour degree >= 3)
              -> order each remaining component into a polyline
              -> fit in the branch's OWN PCA frame, never in image axes
                 (image-axis fitting passes every other check and fails
                 IC-c3, the 90-degree rotation)

Models, both fitted on (s, t) = coordinates along and across the principal
axis:

  line      t = 0            residual = RMS(t); PCA already minimises it
  parabola  t = a s^2+b s+c  residual = RMS(t - fit)
  split     one breakpoint searched between SPLIT_MARGIN and 1-SPLIT_MARGIN
            of the branch, a parabola each side; accepted only if the RMS
            falls below SPLIT_GAIN x the single-parabola RMS, so a split is
            never free

A branch is `line` if its line residual is within RESID_PX, else `arc` if the
parabola is, else `split` if the split is accepted and within RESID_PX, else
`unexplained`. Thresholds are fixed here and in the registration before the
first fit was run.

Usage:
  stroke_curves.py --root <panel dir> --which rough --label ako5 --limit 40
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402

R = Path("results/stroke_curves_20261010")
MIN_BRANCH_PX = 10     # shorter than this fits anything; see NC-c1
LONG_BRANCH_PX = 30    # the subset the headline must also hold on
RESID_PX = 1.0         # "explained", in pixels of RMS cross-axis error
SPLIT_MARGIN = 0.2     # a breakpoint may not sit in the outer fifth
SPLIT_GAIN = 0.7       # a split must cut the RMS to this share, or it is not taken
SPLIT_MIN_SIDE = 10
SPLIT_STEPS = 60      # breakpoints tried per branch; a cap on cost, fixed before any run


def branch_in_fill(p, fill):
    """Share of a branch's pixels inside the solid-fill mask. A branch wholly
    inside one is the fill's medial axis, not a drawn stroke -- the two must
    not be pooled, and item 4 is about exactly this boundary."""
    xy = np.round(p).astype(int)
    h, w = fill.shape
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if not ok.any():
        return 0.0
    return float(fill[xy[ok, 1], xy[ok, 0]].mean())


def skeleton_branches(ink):
    """Skeleton polylines, cut at junctions. Returns a list of (N,2) int arrays
    ordered along the stroke."""
    skel = skeletonize(ink)
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    seg = skel & ~(skel & (nb >= 3))
    n, lab = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    out = []
    ys, xs = np.nonzero(lab)
    by = defaultdict(list)
    for y, x, i in zip(ys, xs, lab[ys, xs]):
        by[i].append((int(y), int(x)))
    for i, pix in by.items():
        if len(pix) < MIN_BRANCH_PX:
            continue
        p = order_path(pix)
        if p is not None:
            out.append(p)
    return out, skel


def order_path(pix, min_px=None):
    """Walk a thin component into a single ordered path. Returns None for a
    component that is not a simple path (a residual blob), rather than
    guessing an order for it."""
    s = set(pix)
    nbr = {}
    for (y, x) in s:
        a = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (y + dy, x + dx) in s]
        nbr[(y, x)] = a
        if len(a) > 2:
            return None
    ends = [p for p in s if len(nbr[p]) <= 1]
    start = ends[0] if ends else next(iter(s))
    path, prev, cur = [start], None, start
    while True:
        nxt = [q for q in nbr[cur] if q != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        path.append(cur)
        if len(path) > len(s):
            return None
    if len(path) < (MIN_BRANCH_PX if min_px is None else min_px):
        return None
    return np.array([(x, y) for (y, x) in path], dtype=np.float64)


def pca_frame(p):
    c = p.mean(0)
    q = p - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    s = q @ vt[0]
    t = q @ vt[1]
    return s, t


def poly_rms(s, t, deg):
    if len(s) < deg + 2:
        return float("inf"), None
    co = np.polyfit(s, t, deg)
    return float(np.sqrt(np.mean((np.polyval(co, s) - t) ** 2))), co


def fit_branch(p):
    s, t = pca_frame(p)
    line_rms = float(np.sqrt(np.mean(t ** 2)))
    par_rms, par_co = poly_rms(s, t, 2)
    row = {"n": len(p), "line_rms": line_rms, "par_rms": par_rms,
           "split_rms": float("nan"), "split_at": -1,
           "curvature": abs(par_co[0]) if par_co is not None else float("nan")}
    best_split, best_at = float("inf"), -1
    lo, hi = int(SPLIT_MARGIN * len(p)), int((1 - SPLIT_MARGIN) * len(p))
    step = max(1, (hi - lo) // SPLIT_STEPS)
    for b in range(max(lo, SPLIT_MIN_SIDE), max(hi, SPLIT_MIN_SIDE + 1), step):
        if b < SPLIT_MIN_SIDE or len(p) - b < SPLIT_MIN_SIDE:
            continue
        r1, _ = poly_rms(s[:b], t[:b], 2)
        r2, _ = poly_rms(s[b:], t[b:], 2)
        if not np.isfinite(r1) or not np.isfinite(r2):
            continue
        r = float(np.sqrt((b * r1 ** 2 + (len(p) - b) * r2 ** 2) / len(p)))
        if r < best_split:
            best_split, best_at = r, b
    row["split_rms"], row["split_at"] = best_split, best_at

    if line_rms <= RESID_PX:
        row["cls"] = "line"
    elif par_rms <= RESID_PX:
        row["cls"] = "arc"
    elif best_split <= SPLIT_GAIN * par_rms and best_split <= RESID_PX:
        row["cls"] = "split"
    else:
        row["cls"] = "unexplained"
    return row


def summarise(rows, label):
    def share(sel):
        tot = sum(r["n"] for r in sel)
        out = {}
        for c in ("line", "arc", "split", "unexplained"):
            out[c] = 100.0 * sum(r["n"] for r in sel if r["cls"] == c) / tot if tot else 0.0
        return tot, out
    print(f"{'subset':>22}{'branches':>10}{'px':>10}{'line':>9}{'arc':>8}"
          f"{'split':>8}{'explained':>11}{'unexpl.':>9}")
    subsets = [("all >=%dpx" % MIN_BRANCH_PX, rows),
               (">=%dpx" % LONG_BRANCH_PX, [r for r in rows if r["n"] >= LONG_BRANCH_PX])]
    if any("in_fill" in r for r in rows):
        subsets += [(">=%dpx outside fill" % LONG_BRANCH_PX,
                     [r for r in rows if r["n"] >= LONG_BRANCH_PX and r.get("in_fill", 0) < 0.5]),
                    (">=%dpx inside fill" % LONG_BRANCH_PX,
                     [r for r in rows if r["n"] >= LONG_BRANCH_PX and r.get("in_fill", 0) >= 0.5])]
    for name, sel in subsets:
        if not sel:
            continue
        tot, sh = share(sel)
        print(f"{name:>22}{len(sel):10d}{tot:10d}{sh['line']:8.1f}%{sh['arc']:7.1f}%"
              f"{sh['split']:7.1f}%{sh['line'] + sh['arc'] + sh['split']:10.1f}%"
              f"{sh['unexplained']:8.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="rough", choices=("rough", "line"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--rotate", action="store_true", help="IC-c3: rotate 90 deg first")
    args = ap.parse_args()

    paths = sorted(p for p in (Path(args.root) / args.which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[:args.limit or None]
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if args.rotate:
            g = np.rot90(g).copy()
        ink = ink_of(g)
        fill = fill_mask(ink)
        for br in skeleton_branches(ink)[0]:
            r = fit_branch(br)
            r["image"] = p.name
            r["in_fill"] = round(branch_in_fill(br, fill), 3)
            r["len_px"] = round(float(np.hypot(*(br[-1] - br[0]))), 1)
            rows.append(r)

    R.mkdir(parents=True, exist_ok=True)
    tag = args.label + ("_rot90" if args.rotate else "")
    out = R / f"branches_{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n", "cls", "line_rms", "par_rms",
                                          "split_rms", "split_at", "curvature",
                                          "in_fill", "len_px"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{tag}: {len(paths)} images, {len(rows)} branches")
    summarise(rows, tag)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
