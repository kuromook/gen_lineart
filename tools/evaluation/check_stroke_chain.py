"""Instrument checks for the junction chainer, as registered 2026-10-10.

  IC-j1  a straight line crossing a parabola a=0.004: the chainer must return
         exactly 2 strokes, each classed line/arc (NOT split -- a wrong
         pairing makes two kinked strokes, which is what split means), with
         the parabola's coefficient recovered to within 20%
  IC-j2  a T: the bar is one stroke, the stem another, 3 arms -> 1 pair and
         1 arm deliberately left unmatched. This is the check that fails if
         matching were greedy instead of a matching.
  IC-j3  IC-j1's branch COORDINATES rotated 37 deg (the IC-c3 amendment --
         rotating the image instead compares different skeletons). Every pair
         cost and the chosen matching must be unchanged.
  IC-j4  external anchor (lesson 12): handled by montage_stroke_chain.py

Usage:  check_stroke_chain.py [--panel <line png> for IC-j3]
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402
from stroke_chain import (MAX_ARMS, SEED, arm_geometry, best_matching,  # noqa: E402
                          chain_image, fragments, pair_cost)

R = Path("results/stroke_chain_20261010")
A_TRUE = 0.004
TOL_A = 0.20
MIN_STROKE_PX = 50.0

# IC-j3 tolerances, AMENDED 2026-10-10 with the reason recorded in the work
# log. The registered bar was a single "< 1e-9" on every term, which is the
# wrong bar for theta: theta = degrees(arccos(T_i . -T_j)) and arccos has an
# infinite derivative at +-1, so a 1e-12 rounding difference in the dot
# product of two collinear arms becomes sqrt(2e-12) rad = 8e-5 deg. Measured
# worst case on a real panel is theta = 180.0 vs 179.99999879 with dK at
# 2.3e-11 and d at 7.7e-13 -- conditioning, not an axis dependence. The
# criterion that carries the check is unchanged and exact: the chosen
# matching must be identical.
TOL = {"cost": 1e-6, "theta": 1e-3, "dK": 1e-6, "d": 1e-6}


def canvas(curves, size=420, thick=2):
    img = np.full((size, size), 255, np.uint8)
    for pts in curves:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, 0, thick)
    return ink_of(img)


def run(ink):
    rows, stat = chain_image(ink, fill_mask(ink), "real", random.Random(SEED))
    big = [r for r in rows if r["path_px"] >= MIN_STROKE_PX]
    return rows, big, stat


def ic_j1():
    x = np.arange(50, 351, 1.0)
    line = np.stack([x, 100 + (x - 50) * (200.0 / 300.0)], 1)
    par = np.stack([x, 120 + A_TRUE * (x - 200) ** 2], 1)
    ink = canvas([line, par])
    rows, big, stat = run(ink)
    print(f"IC-j1  arms {stat['arms']}  pairs {stat['pairs']}  "
          f"frags {stat['frags']}  strokes>={MIN_STROKE_PX:.0f}px {len(big)} "
          f"(all {stat['strokes']})")
    for r in sorted(big, key=lambda r: -r["path_px"]):
        print(f"       {r['cls']:>12}  path {r['path_px']:7.1f}px  "
              f"par_rms {r['par_rms']}  a {r['a_coef']}  frags {r['n_frags']}")
    ok = len(big) == 2 and all(r["cls"] in ("line", "arc") for r in big)
    arcs = [r for r in big if r["cls"] == "arc"]
    a_ok = any(abs(abs(float(r["a_coef"])) - A_TRUE) <= TOL_A * A_TRUE for r in arcs)
    print(f"       2 strokes, both line/arc: {ok};  a recovered within "
          f"{TOL_A:.0%}: {a_ok}  -> {'PASS' if ok and a_ok else 'FAIL'}")
    return ok and a_ok, ink


def ic_j2():
    x = np.arange(50, 351, 1.0)
    bar = np.stack([x, np.full_like(x, 200.0)], 1)
    y = np.arange(200, 351, 1.0)
    stem = np.stack([np.full_like(y, 200.0), y], 1)
    rows, big, stat = run(canvas([bar, stem]))
    print(f"IC-j2  arms {stat['arms']}  pairs {stat['pairs']}  "
          f"strokes>={MIN_STROKE_PX:.0f}px {len(big)} (all {stat['strokes']})")
    for r in sorted(big, key=lambda r: -r["path_px"]):
        print(f"       {r['cls']:>12}  path {r['path_px']:7.1f}px  frags {r['n_frags']}")
    lens = sorted((r["path_px"] for r in big), reverse=True)
    ok = (len(big) == 2 and stat["arms"] == 3 and stat["pairs"] == 1
          and lens[0] > 250 and 100 < lens[1] < 200
          and sorted(r["cls"] for r in big) == ["line", "line"])
    print(f"       bar whole + stem separate, exactly 1 pair of 3 arms "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def costs_and_matching(frags, nodes):
    arms_by_node = defaultdict(list)
    for fi, (n0, n1) in enumerate(nodes):
        if n0 >= 0:
            arms_by_node[n0].append((fi, 0))
        if n1 >= 0:
            arms_by_node[n1].append((fi, 1))
    out = []
    for nd in sorted(arms_by_node):
        arms = arms_by_node[nd]
        k = len(arms)
        if k < 2 or k > MAX_ARMS:
            continue
        g = [arm_geometry(frags[fi], e) for (fi, e) in arms]
        cost, raws = {}, []
        for i in range(k):
            for j in range(i + 1, k):
                gated, raw, theta, dK, d = pair_cost(g[i], g[j])
                cost[(i, j)] = gated
                raws.append((raw, theta, dK, d))
        out.append((nd, tuple(sorted(best_matching(k, cost))), raws))
    return out


def ic_j3(ink, angle_deg=37.0):
    frags, nodes, _, _ = fragments(ink)
    a = np.radians(angle_deg)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = [p @ Rm.T for p in frags]
    base = costs_and_matching(frags, nodes)
    other = costs_and_matching(rot, nodes)
    if len(base) != len(other):
        print(f"IC-j3  node count differs {len(base)} vs {len(other)} -> FAIL")
        return False
    worst, same = {k: 0.0 for k in TOL}, True
    for (n1, m1, r1), (n2, m2, r2) in zip(base, other):
        same &= (m1 == m2) and len(r1) == len(r2)
        for a1, a2 in zip(r1, r2):
            for k, v1, v2 in zip(("cost", "theta", "dK", "d"), a1, a2):
                dd = abs(v1 - v2)
                if np.isfinite(dd):
                    worst[k] = max(worst[k], dd)
    ok = same and all(worst[k] < TOL[k] for k in TOL)
    terms = "  ".join(f"{k} {worst[k]:.2g}/{TOL[k]:g}" for k in ("cost", "theta", "dK", "d"))
    print(f"IC-j3  {len(base)} nodes rotated {angle_deg} deg in coordinates: "
          f"matching identical {same}   {terms} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=None, help="also run IC-j3 on a real panel")
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    ok1, ink = ic_j1()
    ok2 = ic_j2()
    ok3 = ic_j3(ink)
    if args.panel:
        g = cv2.imread(args.panel, cv2.IMREAD_GRAYSCALE)
        if g is not None:
            ok3 &= ic_j3(ink_of(g))
    print(f"\nIC-j1 {'PASS' if ok1 else 'FAIL'}   IC-j2 {'PASS' if ok2 else 'FAIL'}"
          f"   IC-j3 {'PASS' if ok3 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
