"""IC6: can region-F1 grade a partially correct prediction?

Registered 2026-10-16 in doc/work_log.md (commit 4e0e498) before this file
existed, with every expected value written down first. Read that entry; the
short form:

  leg 1  retention   keep a fraction k of the GT regions, pixel-exact.
                     An IDENTITY: region-F1 = 2*n_keep/(n+n_keep) provably,
                     so this leg checks the implementation, not the metric.
  leg 2  erosion     erode every GT region by a disc of radius e. Not a
                     subset of components, so this is the leg that measures.
                     Predicted region-F1 = the fraction of regions whose
                     0.2929*R >= e; the prediction is expected to OVERSHOOT
                     because real regions are elongated, can fall below the
                     500px filter, and can split.

`region_metrics.py` and `fill_mask.py` are used unmodified and their sha256
are printed. No GPU work and no training here.

Usage:
  check_region_f1_partial.py --check          # the assert cell only
  check_region_f1_partial.py                  # the full held-side run
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_mask as fm  # noqa: E402
import region_metrics as rm  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402

OUT = Path("results/fill_koma_20261010")
NEAR_BLANK_INK = 0.002          # the stage-2 stratum boundary, unchanged
KEEP = (0.25, 0.50, 0.75)
ERODE = (1, 2, 4, 6, 8, 12)
HALF_COEF = 1.0 - 1.0 / np.sqrt(2.0)      # 0.29289...
C1_TOL = 0.02
C2_SEP = 0.15
MIN_PANELS = 200
# the values the registration fixed, repeated here so the run compares against
# them rather than against whatever it computes
EXPECT_KEEP = {0.25: 0.3885, 0.50: 0.6552, 0.75: 0.8602}
EXPECT_ERODE = {1: 1.000, 2: 1.000, 4: 0.908, 6: 0.545, 8: 0.363, 12: 0.209}
EXPECT_HALF_PX = 6.40
BLANK_F1 = 0.0084               # stage-2, normal stratum
NC3_F1 = 0.0063                 # stage-2, nc3_hflip:rough_dark


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def keep_mask(gt, name, k):
    """The pixel-exact retention prediction, and n, n_keep.

    Selection: region ids sorted ascending, shuffled by Random("IC6:<name>"),
    first n_keep taken. n_keep = int(round(k*n)), round-half-to-even.
    """
    lab, ids, _ = rm.regions(gt, rm.MIN_REGION_PX)
    n = len(ids)
    order = sorted(ids)
    random.Random(f"IC6:{name}").shuffle(order)
    n_keep = int(round(k * n))
    pred = np.zeros(gt.shape, bool)
    for i in order[:n_keep]:
        pred |= (lab == i)
    return pred, n, n_keep


def erode_mask(gt, e):
    """Every GT region eroded by a disc of radius e, via the same Euclidean
    distance transform the fill mask itself is built on, so `radius` means
    the same thing on both legs."""
    if not gt.any():
        return np.zeros(gt.shape, bool)
    dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
    return dist > e


EFF_EROSION_SHORTFALL = 0.25   # E1, measured at R in {20,40,80}, e in 1..12


def predicted_still_matching(areas, e, calibrated=False):
    """Disc model: a region survives IoU>=0.5 while e <= 0.2929*R.

    E2: `dist > e` erodes by an effective e - 0.25px (E1), so the honest
    boundary is e - 0.25 <= 0.2929*R. The REGISTERED prediction is the
    uncalibrated one and stays the comparison of record; the calibrated
    column is printed beside it because "predicted vs measured" should not
    quietly absorb a known 0.25px offset.
    """
    if not len(areas):
        return float("nan")
    R = np.sqrt(np.asarray(areas, float) / np.pi)
    eff = e - EFF_EROSION_SHORTFALL if calibrated else e
    return float((HALF_COEF * R >= eff).mean())


def _self_check():
    """One assert cell, values computed by hand (registration, C-assert)."""
    # a 40x40 square is 1600px, above MIN_REGION_PX = 500
    g = np.zeros((200, 300), bool)
    g[20:60, 20:60] = True            # region A, 1600px
    g[20:60, 120:160] = True          # region B, 1600px
    lab, ids, stats = rm.regions(g, rm.MIN_REGION_PX)
    assert len(ids) == 2, ids
    assert int(stats[ids[0], cv2.CC_STAT_AREA]) == 1600
    # keep one of two: precision 1.0, recall 0.5, region-F1 = 2*1/(2+1) = 2/3
    one = np.zeros(g.shape, bool)
    one[20:60, 20:60] = True
    r = rm.match(one, g)
    assert r["matched"] == 1 and r["n_pred"] == 1 and r["n_gt"] == 2, r
    assert abs(r["region_f1"] - 2.0 / 3.0) < 1e-12, r["region_f1"]
    # keeping none: empty prediction against 2 regions is 0.0, not 1.0
    assert rm.match(np.zeros(g.shape, bool), g)["region_f1"] == 0.0
    # a disc R=20: eroded by 5 leaves (15/20)^2 = 0.5625 -> must match;
    # eroded by 8 leaves (12/20)^2 = 0.36 -> must not. E1: e=6 is NOT asserted
    # because it is a knife-edge, analytic 0.49 against measured 0.5099.
    assert abs((15 / 20) ** 2 - 0.5625) < 1e-9
    assert abs((12 / 20) ** 2 - 0.36) < 1e-9
    d = rm.disc((200, 200), 100, 100, 20)
    assert d.sum() >= rm.MIN_REGION_PX
    for e, want in ((5, 1), (8, 0)):
        er = erode_mask(d, e)
        got = rm.match(er, d)["matched"]
        iou = (er & d).sum() / max((er | d).sum(), 1)
        assert got == want, (e, got, want, iou)
    # E1's calibration, as a bounded claim rather than a point value: at
    # R = 40 the effective erosion of `dist > e` lies in [e-0.5, e].
    big = rm.disc((400, 400), 200, 200, 40)
    R0 = np.sqrt(big.sum() / np.pi)
    for e in (2, 6, 12):
        Re = np.sqrt(erode_mask(big, e).sum() / np.pi)
        eff = R0 - Re
        assert e - 0.5 <= eff <= e, (e, eff)
    # the analytic half-match coefficient itself
    assert abs(HALF_COEF - 0.2929) < 1e-3, HALF_COEF
    # 0.2929*R >= e at R=20 gives e <= 5.86, so 5 survives and 6 does not
    assert HALF_COEF * 20 >= 5 and HALF_COEF * 20 < 6
    return True


_self_check()


def montage(rows, path, cell=240):
    """One row per degradation level: GT beside the degraded prediction."""
    tiles = []
    for label, gt, pred in rows:
        pair = []
        for m in (gt, pred):
            v = np.full((*m.shape, 3), 255, np.uint8)
            v[m] = (40, 40, 40)
            h, w = m.shape
            s = cell / max(h, w)
            v = cv2.resize(v, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
            pad = np.full((cell, cell, 3), 255, np.uint8)
            pad[:v.shape[0], :v.shape[1]] = v
            pair.append(pad)
        row = np.concatenate(pair, axis=1)
        cv2.putText(row, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(row)
    if not tiles:
        return "no tiles"
    img = np.concatenate(tiles, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def run(limit=0, montage_panel=None):
    print(f"region_metrics.py sha256 {sha(rm.__file__)} (unmodified)")
    print(f"fill_mask.py      sha256 {sha(fm.__file__)} (unmodified)")
    print(f"MIN_REGION_PX {rm.MIN_REGION_PX}  MATCH_IOU {rm.MATCH_IOU}  "
          f"analytic half-match coefficient {HALF_COEF:.4f}")

    names = [n for n, _, _ in load_split()["held"]]
    if limit:
        names = names[:limit]

    keep_f1 = {k: [] for k in KEEP}
    keep_dev = {k: [] for k in KEEP}      # C1b: deviation from the identity
    er_f1 = {e: [] for e in ERODE}
    er_match = {e: [0, 0] for e in ERODE}  # [matched, n_gt]
    er_npred = {e: [0, 0] for e in ERODE}  # [n_pred, n_gt]
    areas_all = []
    halves = []
    n_zero = n_norm = n_nb = 0
    mont = []
    rows_csv = []

    for idx, name in enumerate(names):
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = fm.ink_of(g)
        if ink.mean() < NEAR_BLANK_INK:
            n_nb += 1
            continue                       # the stratum cannot support IC6
        n_norm += 1
        gt = fm.fill_mask(ink)
        lab, ids, stats = rm.regions(gt, rm.MIN_REGION_PX)
        n = len(ids)
        if n == 0:
            n_zero += 1
            continue                       # region-F1 1.0 here is an IDENTITY
        a = [int(stats[i, cv2.CC_STAT_AREA]) for i in ids]
        areas_all.extend(a)
        halves.extend(HALF_COEF * np.sqrt(np.array(a, float) / np.pi))

        row = {"panel": name, "n_regions": n}
        for k in KEEP:
            pred, _, n_keep = keep_mask(gt, name, k)
            f1 = rm.match(pred, gt)["region_f1"]
            exp = 2.0 * n_keep / (n + n_keep) if (n + n_keep) else 1.0
            keep_f1[k].append(f1)
            keep_dev[k].append(abs(f1 - exp))
            row[f"keep{k}_f1"] = round(f1, 6)
            row[f"keep{k}_exp"] = round(exp, 6)
            row[f"keep{k}_nkeep"] = n_keep
        for e in ERODE:
            pred = erode_mask(gt, e)
            r = rm.match(pred, gt)
            er_f1[e].append(r["region_f1"])
            er_match[e][0] += r["matched"]
            er_match[e][1] += r["n_gt"]
            er_npred[e][0] += r["n_pred"]
            er_npred[e][1] += r["n_gt"]
            row[f"er{e}_f1"] = round(r["region_f1"], 6)
            row[f"er{e}_matched"] = r["matched"]
            row[f"er{e}_npred"] = r["n_pred"]
        rows_csv.append(row)
        if montage_panel is None and 4 <= n <= 12 and not mont:
            montage_panel = name
        if name == montage_panel:
            for k in KEEP:
                p, _, nk = keep_mask(gt, name, k)
                mont.append((f"keep k={k:.2f} ({nk}/{n} regions)", gt, p))
            for e in ERODE:
                mont.append((f"erode e={e}px", gt, erode_mask(gt, e)))
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(names)}", flush=True)

    print(f"\npanels: normal {n_norm}, near-blank {n_nb} (SKIPPED, see below), "
          f"normal with 0 regions {n_zero} (region-F1 1.0 there is an "
          f"IDENTITY and is excluded)")
    n_used = len(keep_f1[KEEP[0]])
    print(f"panels used: {n_used} (>= {MIN_PANELS} required by C1 -> "
          f"{'ok' if n_used >= MIN_PANELS else 'INSUFFICIENT'})")
    print(f"regions: {len(areas_all)}, median area "
          f"{np.median(areas_all):.0f}px")

    print(f"\nLEG 1, retention -- an IDENTITY: region-F1 = 2*n_keep/(n+n_keep)")
    print(f"{'k':>6}{'measured':>11}{'registered':>12}{'nominal':>10}"
          f"{'|diff|':>9}{'C1 (<=0.02)':>13}{'C1b max dev':>13}")
    c1 = c1b = True
    means = {}
    for k in KEEP:
        m = float(np.mean(keep_f1[k]))
        means[k] = m
        d = abs(m - EXPECT_KEEP[k])
        mx = float(np.max(keep_dev[k]))
        c1 &= d <= C1_TOL
        c1b &= mx < 1e-9
        print(f"{k:>6.2f}{m:>11.4f}{EXPECT_KEEP[k]:>12.4f}"
              f"{2 * k / (1 + k):>10.3f}{d:>9.4f}"
              f"{'PASS' if d <= C1_TOL else 'FAIL':>13}"
              f"{mx:>13.2e}")
    c1 &= n_used >= MIN_PANELS
    print(f"  C1  within {C1_TOL} of the registered per-panel expectation, "
          f"over >= {MIN_PANELS} panels -> {'PASS' if c1 else 'FAIL'}")
    print(f"  C1b every panel exactly on the identity -> "
          f"{'PASS' if c1b else 'FAIL'}")

    ordered = all(means[KEEP[i]] + C2_SEP <= means[KEEP[i + 1]]
                  for i in range(len(KEEP) - 1))
    seps = [means[KEEP[i + 1]] - means[KEEP[i]] for i in range(len(KEEP) - 1)]
    print(f"\n  C2  strictly ordered and separated by >= {C2_SEP}: "
          f"gaps {' and '.join(f'{s:.3f}' for s in seps)} -> "
          f"{'PASS' if ordered else 'FAIL'}")
    lowest = min(means.values())
    print(f"      blank {BLANK_F1:.4f} and nc3_hflip:rough_dark {NC3_F1:.4f} "
          f"sit {lowest - BLANK_F1:.3f} and {lowest - NC3_F1:.3f} below the "
          f"lowest retention level")

    print(f"\nLEG 2, erosion -- the leg that measures. Prediction is the disc "
          f"model and is EXPECTED to overshoot.")
    print(f"{'e':>4}{'measured F1':>13}{'predicted':>11}{'calib(E2)':>11}"
          f"{'match rate':>12}{'pred/gt regions':>17}{'diff':>9}")
    prev = None
    mono = True
    for e in ERODE:
        m = float(np.mean(er_f1[e]))
        mr = er_match[e][0] / max(er_match[e][1], 1)
        npr = er_npred[e][0] / max(er_npred[e][1], 1)
        pred = predicted_still_matching(areas_all, e)
        cal = predicted_still_matching(areas_all, e, calibrated=True)
        if prev is not None and mr > prev + 1e-12:
            mono = False
        prev = mr
        print(f"{e:>4}{m:>13.4f}{pred:>11.3f}{cal:>11.3f}{mr:>12.3f}"
              f"{npr:>17.3f}{m - pred:>9.3f}")
    print(f"  match rate = matched/n_gt pooled; pred/gt regions shows whether "
          f"erosion made regions VANISH (<1) or SPLIT (>1).")
    half_meas = None
    for e in ERODE:
        mr = er_match[e][0] / max(er_match[e][1], 1)
        if mr <= 0.5:
            half_meas = e
            break
    print(f"  C3  monotonically non-increasing in e -> "
          f"{'PASS' if mono else 'FAIL'}")
    print(f"      half the regions stop matching at e <= {half_meas}px "
          f"(measured bracket), predicted {EXPECT_HALF_PX:.2f}px; "
          f"median 0.2929R over the regions actually used = "
          f"{np.median(halves):.2f}px")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ic6_per_panel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nper-panel: {OUT / 'ic6_per_panel.csv'}")
    print(f"montage ({montage_panel}): "
          f"{montage(mont, OUT / 'ic6_degradation.png')}")

    ok = c1 and c1b and ordered and mono
    verdict = ("CAN" if (ordered and mono) else "CANNOT")
    print(f"\nVERDICT: region-F1 at MIN_REGION_PX={rm.MIN_REGION_PX} / "
          f"MATCH_IOU={rm.MATCH_IOU} **{verdict}** grade a half-correct "
          f"prediction, measured on {n_used} normal-stratum panels: a "
          f"prediction holding half the regions scores "
          f"{means[0.50]:.3f} against blank's {BLANK_F1:.4f}.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage-panel", default=None)
    args = ap.parse_args()
    print("self-check: PASS (assert cell, hand-computed: 2/3 for 1 of 2 kept; "
          "a disc R=20 survives e=5 at IoU 0.5625 and fails e=6 at 0.49)")
    if args.check:
        sys.exit(0)
    sys.exit(0 if run(args.limit, args.montage_panel) else 1)


if __name__ == "__main__":
    main()
