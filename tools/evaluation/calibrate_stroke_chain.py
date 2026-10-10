"""IC-j6: cut a real stroke and see whether the chainer rejoins it.

Why this exists. The chainer's constants were pre-registered by eyeball and
two of them are wrong, which the first real run showed: over all candidate
pairs the median tangent disagreement is 62 deg (so a 60 deg gate throws away
real continuations) and the median curvature term is 0.73 of a 2.0 budget
(curvature read over a 12px skeleton window is mostly quantisation noise, so
KAPPA_REF = 200px is far too small a floor). They have to be calibrated, and
calibrating them against the explained share would be fitting the instrument
to the hypothesis.

So: a labelled target that is not the hypothesis, on real ink (lesson 12 --
one leg anchored to something the pipeline did not produce).

  take a real line-art panel, crop a window around a long fragment F
  draw a straight occluding stroke across F at a random angle
  re-skeletonize: F is now cut into two parts by a new junction
  the ground truth is known by construction -- the two parts are one stroke

  rejoin = share of trials where both parts end in the same chain   (recall)
  wrong  = share of trials where a part of F is chained to a piece of the
           occluder instead                                        (precision)
  score  = rejoin - wrong

The panels used here are mechanically disjoint from the H-stroke panels (the
first 40 are reserved for the measurement; calibration takes 40 from index
100) so that nothing is tuned on the images the number is read from.

Usage:
  calibrate_stroke_chain.py --root <dir> --grid     # sweep and report
  calibrate_stroke_chain.py --root <dir>            # score current constants
"""

import argparse
import itertools
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import ink_of  # noqa: E402
import stroke_chain as sc  # noqa: E402

R_OUT = Path("results/stroke_chain_20261010/icj6_grid.csv")
CROP = 160
OCC_LEN = 36        # shorter than the first try: on dense line art a long
                    # occluder almost never finds a site clear of other ink
MIN_TARGET_PX = 40
ANGLE_LO, ANGLE_HI = 20.0, 90.0
PART_SHARE = 0.8
PART_MIN_PX = 8
SEED = 20261010
CAL_FROM, CAL_N = 100, 71      # panels [100:171]; the H-stroke number is read
                               # from [0:40], so the two sets are disjoint by
                               # construction and nothing is tuned on them


def path_len(p):
    return sc.path_len(p)


def mask_of(pts, shape, thick=1):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, thick)
    return m.astype(bool)


def pick_target(ink):
    """The longest fragment passing near the crop centre. The crop is built
    around a long fragment's midpoint, so one always exists; requiring a long
    fragment near the centre of a RANDOM crop dropped 61% of sites, because
    the median line-art fragment is only 31px."""
    frags, _, _, _ = sc.fragments(ink)
    cy, cx = ink.shape[0] / 2, ink.shape[1] / 2
    best, bl = None, 0.0
    for p in frags:
        d = np.hypot(p[:, 0] - cx, p[:, 1] - cy).min()
        L = path_len(p)
        if d <= 10 and L >= MIN_TARGET_PX and L > bl:
            best, bl = p, L
    return best


def make_trial(ink, rng, tries=8):
    """Returns (ink2, F_mask, occ_mask, angle) or None if no clean site.

    Several angles are tried, because on dense line art most directions put
    the occluder through other ink and the ground truth would stop being
    known. That biases the angle distribution toward whatever is clear, so
    the angle histogram is always reported next to the result."""
    F = pick_target(ink)
    if F is None:
        return None
    F_mask = mask_of(F, ink.shape, 1)
    k = np.ones((3, 3), np.uint8)
    near_F = cv2.dilate(F_mask.astype(np.uint8), k).astype(bool)
    for _ in range(tries):
        i = int(len(F) * rng.uniform(0.35, 0.65))
        j0, j1 = max(0, i - 5), min(len(F) - 1, i + 5)
        tan = F[j1] - F[j0]
        nrm = float(np.hypot(*tan))
        if nrm < 1e-6:
            continue
        tan = tan / nrm
        ang = np.radians(rng.uniform(ANGLE_LO, ANGLE_HI)) * rng.choice([-1, 1])
        d = np.array([tan[0] * np.cos(ang) - tan[1] * np.sin(ang),
                      tan[0] * np.sin(ang) + tan[1] * np.cos(ang)])
        c = F[i]
        occ = np.stack([c - 0.5 * OCC_LEN * d, c + 0.5 * OCC_LEN * d])
        occ_mask = mask_of(occ, ink.shape, 2)
        foot = cv2.dilate(occ_mask.astype(np.uint8), k).astype(bool)
        if (ink & foot & ~near_F).any():
            continue
        return ink | occ_mask, F_mask, occ_mask, abs(np.degrees(ang))
    return None


def parts_of(frags, ref_mask, other_mask):
    out = []
    ref = cv2.dilate(ref_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    oth = cv2.dilate(other_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for fi, p in enumerate(frags):
        xy = np.round(p).astype(int)
        inr = ref[xy[:, 1], xy[:, 0]].mean()
        ino = oth[xy[:, 1], xy[:, 0]].mean()
        if len(p) >= PART_MIN_PX and inr >= PART_SHARE and ino < inr:
            out.append(fi)
    return out


def roots(n, link):
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for (f, _), (g, _) in link.items():
        ra, rb = find(f), find(g)
        if ra != rb:
            par[ra] = rb
    return [find(i) for i in range(n)]


def build_trials(crops, rng_seed=SEED):
    """Trials are built ONCE: the cut, its ground truth, and the skeleton of
    the cut image do not depend on any swept constant."""
    rng = random.Random(rng_seed)
    out = []
    for ink in crops:
        t = make_trial(ink, rng)
        if t is None:
            continue
        ink2, F_mask, occ_mask, ang = t
        pre = sc.fragments(ink2)
        frags = pre[0]
        fp = parts_of(frags, F_mask, occ_mask)
        op = parts_of(frags, occ_mask, F_mask)
        if len(fp) != 2 or not op:
            continue
        out.append((ink2, pre, fp, op, ang))
    return out


def score(trials, rng_seed=SEED, by_angle=False):
    rng = random.Random(rng_seed)
    n = rejoin = wrong = 0
    buckets = {}
    for ink2, pre, fp, op, ang in trials:
        frags, link, absorbed, _ = sc.build_links(ink2, "real", rng, pre=pre)
        r = roots(len(frags), link)
        n += 1
        ok = r[fp[0]] == r[fp[1]]
        bad = any(r[f] == r[o] for f in fp for o in op)
        rejoin += ok
        wrong += bad
        if by_angle:
            b = int(ang // 15) * 15
            d = buckets.setdefault(b, [0, 0, 0])
            d[0] += 1
            d[1] += ok
            d[2] += bad
    if n == 0:
        return 0, 0.0, 0.0, -1.0, buckets
    return n, rejoin / n, wrong / n, rejoin / n - wrong / n, buckets


def load_crops(root, which, lo, cnt, rng, per_panel=12):
    """Crops centred on the midpoint of a long fragment, so every crop has a
    target to cut. The panel's fragments are found once per panel."""
    paths = sorted(p for p in (Path(root) / which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[lo:lo + cnt]
    out = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        h, w = ink.shape
        if h < CROP + 8 or w < CROP + 8:
            continue
        frags, _, _, _ = sc.fragments(ink)
        cand = [f for f in frags if path_len(f) >= MIN_TARGET_PX]
        if not cand:
            continue
        rng.shuffle(cand)
        for f in cand[:per_panel]:
            y, x = f[len(f) // 2][1], f[len(f) // 2][0]
            y = int(min(max(y, CROP // 2), h - CROP // 2 - 1))
            x = int(min(max(x, CROP // 2), w - CROP // 2 - 1))
            out.append(ink[y - CROP // 2:y + CROP // 2,
                           x - CROP // 2:x + CROP // 2].copy())
    return out


GRID = {
    "TAN_SKIP": [0, 2, 4],
    "TAN_WIN": [8, 12, 20],
    "KAPPA_REF": [20.0, 50.0, 200.0],
    "THETA_MAX": [45.0, 60.0, 75.0, 90.0],
    "D_MAX": [2.0, 3.0, 5.0],
    "W_KAPPA": [0.0, 0.5, 1.0],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line", choices=("rough", "line"))
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--from-index", type=int, default=CAL_FROM)
    ap.add_argument("--n-panels", type=int, default=CAL_N)
    ap.add_argument("--per-panel", type=int, default=12)
    args = ap.parse_args()

    crops = load_crops(args.root, args.which, args.from_index, args.n_panels,
                       random.Random(SEED), per_panel=args.per_panel)
    trials = build_trials(crops)
    print(f"{len(crops)} candidate sites from panels "
          f"[{args.from_index}:{args.from_index + args.n_panels}] of "
          f"{args.which} -> {len(trials)} usable trials")

    if not args.grid:
        n, rj, wr, sc_, buck = score(trials, by_angle=True)
        print(f"current constants: trials {n}  rejoin {rj:.3f}  wrong {wr:.3f}  "
              f"score {sc_:.3f}")
        for b in sorted(buck):
            t, o, w = buck[b]
            print(f"  crossing angle {b:3d}-{b + 15:3d} deg  n {t:4d}  "
                  f"rejoin {o / t:.3f}  wrong {w / t:.3f}")
        return

    base = {k: getattr(sc, k) for k in GRID}
    keys = list(GRID)
    results = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        for k, v in zip(keys, combo):
            setattr(sc, k, v)
        n, rj, wr, s, _ = score(trials)
        results.append((s, rj, wr, n, dict(zip(keys, combo))))
    for k, v in base.items():
        setattr(sc, k, v)
    import csv as _csv
    with open(R_OUT, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(keys + ["n", "rejoin", "wrong", "score"])
        for s_, rj, wr, n, cfg in results:
            w.writerow([cfg[k] for k in keys] + [n, round(rj, 4), round(wr, 4),
                                                 round(s_, 4)])
    print(f"saved every combination: {R_OUT}")
    results.sort(key=lambda r: -r[0])
    print(f"\ntop 12 of {len(results)} combinations, by rejoin - wrong:")
    for s, rj, wr, n, cfg in results[:12]:
        print(f"  score {s:.3f}  rejoin {rj:.3f}  wrong {wr:.3f}  n {n}  "
              + "  ".join(f"{k}={cfg[k]}" for k in keys))
    print("\nregistered (pre-calibration) constants:")
    for s, rj, wr, n, cfg in results:
        if cfg == base:
            print(f"  score {s:.3f}  rejoin {rj:.3f}  wrong {wr:.3f}  n {n}")
            break


if __name__ == "__main__":
    main()
