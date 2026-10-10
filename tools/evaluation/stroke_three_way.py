"""H-stroke-3: the three-way re-run, registered 2026-10-10 in doc/work_log.md.

The first H-stroke failed, and its reading was wrong, because a threshold on
mask overlap (`in_fill < 0.5`) was used as a population filter: 64% of the
scored length had fill in it, and the explained share fell from 94.8% at zero
fill to 45.3% at 0.3-0.5. So the populations are rebuilt so that **every
member is something a person drew**:

  A  line ink with no fill contact -- skeletonize `ink & ~fill`, so the medial
     axis does not exist to be filtered out, and neither does the 4px rim. A
     line running into a blacked region ends about 4px early; that is where B
     takes over.
  B  fill contours -- findContours on the fill mask (RETR_CCOMP, holes count),
     cut into open pieces where another stroke meets them.
  C  the medial axis -- excluded by construction, its length printed so the
     exclusion is visible rather than assumed.

A and B are scored separately and never pooled.

Where a contour is cut: a contour point is a cut point if the de-filled line
skeleton has a pixel within # AMENDED 2026-10-10 before the measurement run, with the reason measured.
# 2.0 was set by eye and missed one of four known meeting points in IC-t1.
# Cause, measured: where a thin line meets a blacked region the fill mask
# swallows the last few pixels of the line, so the DE-FILLED skeleton can
# begin that far from the contour -- the four known points sit at 0.96, 0.96,
# 1.91 and 2.87px. The principled value is the fill mask's own reach, since
# that is what did the swallowing; it is not fitted. The next-nearest
# non-meeting point is 22px away, so the synthetic answer is exact for any
# threshold in [3, 8] and the choice is not delicate. Sensitivity on real
# panels is printed every run.
CUT_NEAR_PX = FILL_HALF_WIDTH_PX of it; adjacent cut points
collapse to one; between consecutive cut points is one open piece. A contour
with no cut point stays closed, cannot be fitted by a model that needs a
principal axis, and is counted as `closed_uncut` -- never dropped quietly.

Controls. A has a chaining decision, so it keeps NC-j1 (shuffled tangents) and
NC-j2 (random matching). B has none, so it gets its own:
  NC-t2  the cut points shuffled to random positions, same count. Named error:
         *the pieces fit merely because the contour was cut into that many
         pieces, not because the cuts are where strokes meet.*
  NC-d1  a random closed curve with the same perimeter and isoperimetric
         ratio, cut the same way. Named error: *any closed curve of this size
         and blobbiness would score the same.*

Pass, unchanged: >= 70% of length explained (line / single parabola /
one-breakpoint split, RMS <= 1.0px) with every control for that population at
least 10 points below.

Usage:
  stroke_three_way.py --root <panel dir> --which line --label ako5 --limit 40
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_contour_curves import iso_ratio, random_closed  # noqa: E402
from fill_mask import FILL_HALF_WIDTH_PX, fill_mask, ink_of  # noqa: E402
from stroke_chain import (LONG_BRANCH_PX, SEED, assemble, build_links,  # noqa: E402
                          fit_stroke, path_len)

R = Path("results/stroke_three_way_20261010")
# AMENDED 2026-10-10 before the measurement run, with the reason measured.
# 2.0 was set by eye and missed one of four known meeting points in IC-t1.
# Cause, measured: where a thin line meets a blacked region the fill mask
# swallows the last few pixels of the line, so the DE-FILLED skeleton can
# begin that far from the contour -- the four known points sit at 0.96, 0.96,
# 1.91 and 2.87px. The principled value is the fill mask's own reach, since
# that is what did the swallowing; it is not fitted. The next-nearest
# non-meeting point is 22px away, so the synthetic answer is exact for any
# threshold in [3, 8] and the choice is not delicate. Sensitivity on real
# panels is printed every run.
CUT_NEAR_PX = FILL_HALF_WIDTH_PX
NEAR_FILL_PX = 4.0
EXPLAINED = ("line", "arc", "split")


# --------------------------------------------------------------------------
# contour cutting

def cut_indices(contour, dist_to_line, near_px=CUT_NEAR_PX):
    """Indices where another stroke meets the contour, runs collapsed to one."""
    xy = np.round(contour).astype(int)
    h, w = dist_to_line.shape
    x = np.clip(xy[:, 0], 0, w - 1)
    y = np.clip(xy[:, 1], 0, h - 1)
    near = dist_to_line[y, x] <= near_px
    n = len(near)
    if not near.any():
        return []
    if near.all():
        return [0]
    start = int(np.argmax(~near))          # begin at a non-cut point
    rolled = np.roll(near, -start)
    out, i = [], 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            out.append((start + (i + j - 1) // 2) % n)
            i = j
        else:
            i += 1
    return sorted(set(out))


def open_pieces(contour, cuts):
    """Pieces between consecutive cut points, walking the closed contour."""
    n = len(contour)
    if not cuts:
        return []
    if len(cuts) == 1:
        c = cuts[0]
        return [np.roll(contour, -c, axis=0)]
    out = []
    for a, b in zip(cuts, cuts[1:] + cuts[:1]):
        idx = (np.arange(a, b if b > a else b + n) % n)
        if len(idx) >= 4:
            out.append(contour[idx])
    return out


# --------------------------------------------------------------------------

def score(pts, kind, extra=None):
    L = path_len(pts)
    r = fit_stroke(pts) if len(pts) >= 4 else {
        "n": len(pts), "cls": "tooshort", "line_rms": "", "par_rms": "",
        "split_rms": "", "split_at": -1, "curvature": "", "a_coef": ""}
    r["path_px"] = round(L, 1)
    r["end_px"] = round(float(np.hypot(*(pts[-1] - pts[0]))), 1)
    r["straight"] = round(r["end_px"] / L, 3) if L > 0 else ""
    r["kind"] = kind
    r.update(extra or {})
    return r


def population_a(ink, fill, mode, rng):
    """Chains from the de-filled line ink. `near_fill` is recorded for every
    chain and its distribution inside the kept set is printed -- the trap that
    broke the last run, applied to this one."""
    line_ink = ink & ~fill
    d_fill = cv2.distanceTransform((~fill).astype(np.uint8), cv2.DIST_L2, 3)
    frags, link, absorbed, stats = build_links(line_ink, mode, rng)
    rows = []
    for pts, cyc, nf in assemble(frags, link, skip=absorbed):
        if cyc or len(pts) < 4:
            continue
        xy = np.round(pts).astype(int)
        h, w = d_fill.shape
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        nearf = (float((d_fill[xy[ok, 1], xy[ok, 0]] <= NEAR_FILL_PX).mean())
                 if ok.any() else 0.0)
        rows.append(score(pts, "A_line", {"n_frags": nf,
                                          "near_fill": round(nearf, 3)}))
    return rows, stats


def population_b(ink, fill, mode, rng):
    """Fill contours, cut where a stroke meets them."""
    line_skel = skeletonize(ink & ~fill)
    d_line = cv2.distanceTransform((~line_skel).astype(np.uint8), cv2.DIST_L2, 3)
    cs, _ = cv2.findContours(fill.astype(np.uint8), cv2.RETR_CCOMP,
                             cv2.CHAIN_APPROX_NONE)
    rows, uncut, uncut_px = [], 0, 0.0
    sens = {t: 0 for t in (2.0, 4.0, 8.0)}
    for c in cs:
        q = c.reshape(-1, 2).astype(np.float64)
        per = path_len(np.vstack([q, q[:1]]))
        if per < LONG_BRANCH_PX or len(q) < 8:
            continue
        cuts = cut_indices(q, d_line)
        for t in sens:
            sens[t] += len(cut_indices(q, d_line, t))
        if mode == "nc_t2":
            cuts = sorted(rng.sample(range(len(q)), min(len(cuts), len(q)))) \
                if cuts else []
        elif mode == "nc_d1":
            area = abs(float(cv2.contourArea(c)))
            if area <= 4:
                continue
            q = random_closed(per, iso_ratio(per, area), rng)
            cuts = sorted(rng.sample(range(len(q)), min(len(cuts), len(q)))) \
                if cuts else []
        if not cuts:
            uncut += 1
            uncut_px += per
            continue
        for p in open_pieces(q, cuts):
            if path_len(p) >= LONG_BRANCH_PX:
                rows.append(score(p, "B_contour", {"n_cuts": len(cuts),
                                                   "near_fill": 1.0,
                                                   "n_frags": 1}))
    return rows, {"closed_uncut": uncut, "closed_uncut_px": uncut_px,
                  **{f"cuts@{int(t)}px": v for t, v in sens.items()}}


def summarise(rows, label, stat, bar=70.0):
    def block(sel):
        t = sum(r["n"] for r in sel)
        if not t:
            return None
        sh = {c: 100.0 * sum(r["n"] for r in sel if r["cls"] == c) / t
              for c in EXPLAINED + ("unexplained",)}
        return len(sel), t, sh
    print(f"\n== {label} ==  " + "  ".join(f"{k} {v}" for k, v in stat.items()))
    print(f"{'subset':>26}{'items':>8}{'px':>9}{'line':>8}{'arc':>7}{'split':>7}"
          f"{'EXPL':>8}{'unexpl':>8}")
    out = {}
    for name, sel in (("A line ink >=30px",
                       [r for r in rows if r["kind"] == "A_line"
                        and r["path_px"] >= LONG_BRANCH_PX]),
                      ("B fill contour >=30px",
                       [r for r in rows if r["kind"] == "B_contour"])):
        b = block(sel)
        if b is None:
            continue
        n, t, sh = b
        e = sum(sh[c] for c in EXPLAINED)
        out[name[0]] = (e, t)
        print(f"{name:>26}{n:8d}{t:9d}{sh['line']:7.1f}%{sh['arc']:6.1f}%"
              f"{sh['split']:6.1f}%{e:7.1f}%{sh['unexplained']:7.1f}%")
    a = [r for r in rows if r["kind"] == "A_line" and r["path_px"] >= LONG_BRANCH_PX]
    if a:
        print(f"  near_fill inside the KEPT set A (the check the last run lacked):",
              end=" ")
        tot = sum(r["n"] for r in a)
        for lo, hi, lab in ((0, 1e-9, "0"), (1e-9, .1, "0-.1"),
                            (.1, .5, ".1-.5"), (.5, 1.01, ">=.5")):
            s = [r for r in a if (lo <= r["near_fill"] < hi if lo == 0
                                  else lo < r["near_fill"] < hi)]
            print(f"{lab} {100.0 * sum(r['n'] for r in s) / tot:.0f}%", end="  ")
        print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line", choices=("rough", "line"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--modes", default="real,nc_j1,nc_j2,nc_t2,nc_d1")
    args = ap.parse_args()

    paths = sorted(p for p in (Path(args.root) / args.which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[:args.limit or None]
    R.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in args.modes.split(","):
        rng = random.Random(SEED)
        rows, agg = [], defaultdict(float)
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            ink = ink_of(g)
            fill = fill_mask(ink)
            agg["medial_axis_px_excluded"] += int((skeletonize(ink) & fill).sum())
            if mode in ("real", "nc_j1", "nc_j2"):
                rs, st = population_a(ink, fill, mode, rng)
            else:
                rs, st = [], {}
            if mode in ("real", "nc_t2", "nc_d1"):
                rb, sb = population_b(ink, fill, mode, rng)
                rs += rb
                st = {**st, **sb}
            for r in rs:
                r["image"] = p.name
            rows += rs
            for k, v in st.items():
                agg[k] += v
        tag = f"{args.label}_{args.which}_{mode}"
        with open(R / f"three_way_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "image", "kind", "n", "cls", "line_rms", "par_rms", "split_rms",
                "split_at", "curvature", "a_coef", "path_px", "end_px",
                "straight", "n_frags", "n_cuts", "near_fill"],
                extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        results[mode] = summarise(rows, f"{tag}  ({len(paths)} images)",
                                  {k: (int(v) if v == int(v) else round(v, 1))
                                   for k, v in agg.items()})

    print(f"\n{'=' * 78}\nVERDICT  (bar: >=70% explained AND every control >=10 points below)")
    for pop, label, ctrls in (("A", "A  line ink, no fill contact", ("nc_j1", "nc_j2")),
                              ("B", "B  fill contours, cut at strokes", ("nc_t2", "nc_d1"))):
        if "real" not in results or pop not in results["real"]:
            continue
        base = results["real"][pop][0]
        gaps = {c: base - results[c][pop][0] for c in ctrls
                if c in results and pop in results[c]}
        ok = base >= 70.0 and all(g >= 10.0 for g in gaps.values()) and gaps
        print(f"  {label}: {base:.1f}%  " +
              "  ".join(f"{c} {base - g:.1f}% (gap {g:+.1f})" for c, g in gaps.items()) +
              f"   -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
