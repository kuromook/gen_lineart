"""Instrument checks and negative controls for stroke_curves.py.

Every expectation below was written into this file before it was first run,
as the registration requires (doc/work_log.md 2026-10-10).

IC-c1  synthetic strokes with known equations, rendered as ink at the
       corpus's own stroke width (3px) and recovered through the same
       skeleton -> branch -> fit path as real data:
         a straight segment      -> class `line`,  rms <= 0.5px
         a parabola a=0.004      -> class `arc`,   rms <= 0.5px, and the
                                    recovered a within 20% of 0.004
         a 120-degree circular arc, R=200 -> its parabola rms must be at
                                    least 3x the true parabola's, or the
                                    fit does not distinguish shapes and
                                    nothing downstream may be read
IC-c3  frame invariance of the FIT. Amended 2026-10-10 after the first
       reading: rotating the *image* and re-running the whole pipeline
       compares 11,166 branches against 11,313, because skeletonization is
       not exactly rotation-equivariant, so a 0.2-point difference says
       nothing about the fit. The check that isolates what it is for rotates
       the branch COORDINATES -- the same points, a different frame -- and
       requires the residuals to match to 1e-9. An image-axis fit fails this
       and passes everything else. The image-level run is kept as a looser
       end-to-end sanity row.
NC-c1  random walks with the same branch-length distribution as the real
       data. If real strokes are not explained far better than these, the
       70% in the hypothesis is a property of short segments, which fit
       anything, and the hypothesis is unsupported.

Usage:
  check_stroke_curves.py --ic
  check_stroke_curves.py --nc results/stroke_curves_20261010/branches_<tag>.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import ink_of  # noqa: E402
from stroke_curves import fit_branch, skeleton_branches, summarise  # noqa: E402

TRUE_A = 0.004
TOL_RMS = 0.5
TOL_A = 0.20
ARC_FACTOR = 3.0
PAD = 60


def render(pts, size):
    img = np.full(size, 255, np.uint8)
    p = np.round(pts).astype(np.int32)
    cv2.polylines(img, [p], False, 0, 3, cv2.LINE_AA)
    return img


def longest(ink):
    br, _ = skeleton_branches(ink)
    if not br:
        return None
    return max(br, key=len)


def ic1():
    ok = True
    s = np.linspace(-150, 150, 600)

    cases = {}
    cases["line"] = np.stack([s + 200, np.full_like(s, 160)], 1)
    cases["parabola"] = np.stack([s + 200, TRUE_A * s ** 2 + 60], 1)
    th = np.linspace(-np.radians(60), np.radians(60), 600)
    cases["arc120"] = np.stack([200 * np.sin(th) + 220, -200 * np.cos(th) + 260], 1)

    size = (420, 420)
    res = {}
    for name, pts in cases.items():
        br = longest(ink_of(render(pts, size)))
        if br is None:
            print(f"  {name:10} NO BRANCH -- FAIL")
            ok = False
            continue
        r = fit_branch(br)
        res[name] = r
        print(f"  {name:10} n={r['n']:4d} cls={r['cls']:11} line_rms={r['line_rms']:7.3f} "
              f"par_rms={r['par_rms']:7.3f} a={r['curvature']:.5f}")

    if res.get("line", {}).get("cls") != "line" or res["line"]["line_rms"] > TOL_RMS:
        print(f"  -> FAIL: straight segment must be `line` with rms <= {TOL_RMS}")
        ok = False
    p = res.get("parabola")
    if not p or p["cls"] != "arc" or p["par_rms"] > TOL_RMS:
        print(f"  -> FAIL: parabola must be `arc` with rms <= {TOL_RMS}")
        ok = False
    elif abs(p["curvature"] - TRUE_A) / TRUE_A > TOL_A:
        print(f"  -> FAIL: recovered a={p['curvature']:.5f} is further than "
              f"{TOL_A:.0%} from {TRUE_A}")
        ok = False
    a = res.get("arc120")
    if a and p:
        ratio = a["par_rms"] / max(p["par_rms"], 1e-9)
        verdict = "ok" if ratio >= ARC_FACTOR else "FAIL"
        print(f"  -> circular arc fits the parabola model {ratio:.1f}x worse than a "
              f"parabola does (bar {ARC_FACTOR}x): {verdict}")
        if ratio < ARC_FACTOR:
            ok = False
    print(f"IC-c1: {'PASS' if ok else 'FAIL'}")
    return ok


def nc1(path):
    rows = list(csv.DictReader(open(path)))
    lens = [int(r["n"]) for r in rows]
    rng = np.random.default_rng(20261010)
    out = []
    for n in lens:
        # an 8-connected walk that never immediately backtracks -- the same
        # kind of object the skeleton produces, with the structure removed
        p = [(0, 0)]
        d = rng.integers(0, 8)
        for _ in range(n - 1):
            d = (d + rng.integers(-1, 2)) % 8
            ang = d * np.pi / 4
            p.append((p[-1][0] + np.cos(ang), p[-1][1] + np.sin(ang)))
        out.append(fit_branch(np.array(p)))
    print(f"NC-c1 random walks matched to {len(lens)} real branch lengths:")
    summarise(out, "nc1")
    return out


def ic3(root, which, limit):
    """Rotate the branch coordinates, not the image: same points, new frame."""
    paths = sorted(q for q in (Path(root) / which).iterdir()
                   if q.suffix.lower() in (".png", ".jpg"))[:limit]
    worst_line = worst_par = 0.0
    n = 0
    for q in paths:
        g = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        for br in skeleton_branches(ink_of(g))[0]:
            a = fit_branch(br)
            for deg in (90.0, 37.0):
                th = np.radians(deg)
                rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                b = fit_branch(br @ rot.T)
                worst_line = max(worst_line, abs(a["line_rms"] - b["line_rms"]))
                worst_par = max(worst_par, abs(a["par_rms"] - b["par_rms"]))
            n += 1
    ok = worst_line < 1e-9 and worst_par < 1e-9
    print(f"IC-c3 coordinate rotation, {n} branches x (90deg, 37deg): "
          f"worst line_rms delta {worst_line:.3e}, worst par_rms delta {worst_par:.3e}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--nc", default=None)
    ap.add_argument("--ic3", default=None, help="panel root for the coordinate-rotation check")
    ap.add_argument("--which", default="rough")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    if args.ic:
        print("IC-c1 synthetic strokes with known equations")
        ic1()
    if args.ic3:
        ic3(args.ic3, args.which, args.limit)
    if args.nc:
        nc1(args.nc)


if __name__ == "__main__":
    main()
