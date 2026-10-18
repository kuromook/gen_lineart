"""IC7: does region-F1 grade a prediction that is too LARGE, and where is the
tolerance boundary?

Registered 2026-10-18 in doc/work_log.md (commit 324b2a4) before this file
existed, with every expected value naming its source. Read that entry; the
short form:

  leg      dilation   pred = (distance_to_gt <= d), NOT intersected with ink,
                      because the failure mode probed is a model painting past
                      a region's boundary onto white paper.
  the two grids
    sqrt(2)  d = e*sqrt2 for IC6's e. The registered predicted column at these
             radii IS IC6's registered erosion column, by an exact identity:
             erosion survives while A >= pi e^2 / 0.29289^2 = 36.62 e^2,
             dilation while A >= pi d^2 / 0.41421^2 = 18.31 d^2, and the ratio
             of thresholds is exactly 2 because 0.29289/0.41421 = 1/sqrt2.
             So predicted_dilate(d) = predicted_erode(d/sqrt2).
    integer  IC6's own radii, so over-painting and under-painting are compared
             at EQUAL radius (C4b). Its predicted column is computed in-run
             from the measured census and is labelled computed, not registered.

`region_metrics.py` and `fill_mask.py` are used unmodified and their sha256
are printed. No GPU work and no training here. IC7 cannot release and cannot
block the 2026-10-19..10-22 slot; it produces a reading caveat.

Usage:
  check_region_f1_overpaint.py --check      # the assert cell only
  check_region_f1_overpaint.py              # the full held-side run
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_mask as fm  # noqa: E402
import region_metrics as rm  # noqa: E402
from check_region_f1_partial import montage  # noqa: E402  (IC6's, unmodified)
from koma_split import ROOT, load as load_split  # noqa: E402

OUT = Path("results/fill_koma_20261010")
NEAR_BLANK_INK = 0.002            # the stage-2 stratum boundary, unchanged
DIL_COEF = np.sqrt(2.0) - 1.0     # 0.414213..., GT subset of pred: R^2/(R+d)^2
ERO_COEF = 1.0 - 1.0 / np.sqrt(2.0)

# the twelve radii, ascending. The sqrt(2) grid carries the registered
# prediction; the integer grid carries the equal-radius comparison with IC6.
SQRT2 = (1.414, 2.828, 5.657, 8.485, 11.314, 16.971)
INTEG = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
DILATE = tuple(sorted(SQRT2 + INTEG))
MIRROR_AT = (2.828, 8.485)        # one below, one at the predicted half-match

# registered, from results/fill_koma_20261010/ic6_run.txt (IC6's e = d/sqrt2)
EXPECT_DILATE = {1.414: 1.000, 2.828: 1.000, 5.657: 0.908,
                 8.485: 0.545, 11.314: 0.363, 16.971: 0.209}
EXPECT_HALF_PX = 9.05             # 0.41421*sqrt(1500/pi) = 6.40*sqrt2
IC6_MEDIAN_AREA = 1500            # ic6_run.txt census
IC6_HALF_PX = 6.40                # IC6's registered erosion half-match
# IC6's MEASURED erosion F1, used only to assert that ic6_per_panel.csv is the
# file this run thinks it is; C4b compares recomputed per-panel values.
IC6_ERODE_F1 = {1: 0.9091, 2: 0.8223, 4: 0.3821,
                6: 0.2546, 8: 0.1986, 12: 0.1438}
IC6_CSV = OUT / "ic6_per_panel.csv"
MONTAGE_PANEL = "koma_00003_001_2015_gakusai3__tmp3_gakusai3__page0034_p1.png"

BLANK_F1 = 0.0084                 # report_held_nc2repaired.txt, normal stratum
NC3_F1 = 0.0063                   # nc3_hflip:rough_dark, same file
C3_BAND = (0.20, 0.80)
C3_SEP = 0.15
C4_MIN_HALF_PX = 4.0              # IC6's MEASURED erosion bracket
C5_RATIO_BAR = 0.90
C5_AT = 8.485
NC_HFLIP_BAR = 0.05
MIN_PANELS = 200


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def dilate_mask(gt, d):
    """Every GT region dilated by a disc of radius d, through the SAME cv2
    maskSize-3 DIST_L2 transform IC6's erosion leg used, so that `radius`
    means the same thing on both legs. d = 0 returns the GT exactly: the
    distance to the GT is 0 on the GT itself."""
    if not gt.any():
        return np.zeros(gt.shape, bool)
    dist = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3)
    return dist <= d


def _calibrate_shift():
    """E1's form, for dilation: the effective dilation of `dist <= d` measured
    on a disc, as a bounded claim. Computed at import, never hard-coded."""
    big = rm.disc((500, 500), 250, 250, 40)
    r0 = np.sqrt(big.sum() / np.pi)
    shifts = []
    for d in (2, 6, 12):
        rd = np.sqrt(dilate_mask(big, d).sum() / np.pi)
        shifts.append((rd - r0) - d)
    return float(np.mean(shifts)), shifts


DIL_EFF_SHIFT, _SHIFTS = _calibrate_shift()


def predicted_matching(areas, d, calibrated=False):
    """Disc model: a region survives IoU>=0.5 while d <= 0.41421*R."""
    if not len(areas):
        return float("nan")
    R = np.sqrt(np.asarray(areas, float) / np.pi)
    eff = d + DIL_EFF_SHIFT if calibrated else d
    return float((DIL_COEF * R >= eff).mean())


def _self_check():
    """One assert cell, values computed by hand in the registration."""
    # the area-threshold identity the sqrt(2) grid rests on: exactly 2
    assert abs((np.pi / DIL_COEF ** 2) / (np.pi / ERO_COEF ** 2) - 0.5) < 1e-12
    assert abs(np.pi / DIL_COEF ** 2 - 18.31) < 0.01
    assert abs(np.pi / ERO_COEF ** 2 - 36.62) < 0.01
    # the registered half-match radius, derived twice, agreeing to 0.01px
    a = DIL_COEF * np.sqrt(IC6_MEDIAN_AREA / np.pi)
    b = IC6_HALF_PX * np.sqrt(2.0)
    assert abs(a - b) < 0.01, (a, b)
    assert abs(a - EXPECT_HALF_PX) < 0.01, a
    # the sqrt(2) grid really is IC6's radii scaled, to 0.001px
    for d, e in zip(SQRT2, (1, 2, 4, 6, 8, 12)):
        assert abs(d - e * np.sqrt(2.0)) < 0.001, (d, e)
    # d = 0 is the IDENTITY: two 40x40 squares, 1600px each, above the filter
    g = np.zeros((200, 300), bool)
    g[20:60, 20:60] = True
    g[20:60, 120:160] = True
    assert len(rm.regions(g, rm.MIN_REGION_PX)[1]) == 2
    assert np.array_equal(dilate_mask(g, 0), g)
    assert rm.match(dilate_mask(g, 0), g)["region_f1"] == 1.0
    assert dilate_mask(g, 1).sum() > g.sum()        # NC-d0: not a no-op
    # a disc R=20 (1257px, above the filter): dilated by 6 the IoU is
    # (20/26)^2 = 0.5917 and it MUST match; by 10 it is (20/30)^2 = 0.4444 and
    # it MUST NOT. Both are off the knife-edge on purpose (E1's lesson): d = 8
    # gives 0.5102, and a +-0.5px offset would cross 0.5, while 6 and 10
    # survive it (0.5694 and 0.4597).
    assert abs((20 / 26) ** 2 - 0.5917) < 1e-3
    assert abs((20 / 30) ** 2 - 0.4444) < 1e-3
    d20 = rm.disc((400, 400), 200, 200, 20)
    assert d20.sum() >= rm.MIN_REGION_PX
    for d, want in ((6, 1), (10, 0)):
        pr = dilate_mask(d20, d)
        got = rm.match(pr, d20)["matched"]
        iou = (pr & d20).sum() / max((pr | d20).sum(), 1)
        assert got == want, (d, got, want, iou)
    # NC-iou: the d = 10 case above is the detector -- a matcher thresholding
    # on coverage instead of IoU would have matched it, since GT is a subset.
    assert (dilate_mask(d20, 10) & d20).sum() == d20.sum()
    # the bounded calibration, as a claim rather than a point value
    for d, s in zip((2, 6, 12), _SHIFTS):
        assert abs(s) <= 0.5, (d, s)
    # dilation is monotone in the mask, so splitting is impossible
    assert rm.match(dilate_mask(g, 2), g)["n_pred"] <= 2
    return True


_self_check()


def load_ic6():
    """IC6's per-panel erosion F1, for the equal-radius comparison (C4b)."""
    if not IC6_CSV.exists():
        return {}
    out = {}
    with open(IC6_CSV) as f:
        for row in csv.DictReader(f):
            out[row["panel"]] = {e: float(row[f"er{e}_f1"])
                                 for e in (1, 2, 4, 6, 8, 12)}
    return out


def run(limit=0):
    print(f"region_metrics.py sha256 {sha(rm.__file__)} (unmodified)")
    print(f"fill_mask.py      sha256 {sha(fm.__file__)} (unmodified)")
    print(f"MIN_REGION_PX {rm.MIN_REGION_PX}  MATCH_IOU {rm.MATCH_IOU}  "
          f"dilation coefficient {DIL_COEF:.5f} (erosion's {ERO_COEF:.5f})")
    print(f"effective dilation of `dist <= d`: d {DIL_EFF_SHIFT:+.3f}px, "
          f"measured at R=40 for d in (2,6,12) -> "
          f"{', '.join(f'{s:+.3f}' for s in _SHIFTS)} (bounded assert: "
          f"|shift| <= 0.5; the REGISTERED column stays the comparison of "
          f"record)")

    ic6 = load_ic6()
    print(f"IC6 per-panel: {IC6_CSV} ({len(ic6)} panels)")

    names = [n for n, _, _ in load_split()["held"]]
    if limit:
        names = names[:limit]

    f1 = {d: [] for d in DILATE}
    piou = {d: [] for d in DILATE}
    mt = {d: [0, 0] for d in DILATE}          # [matched, n_gt]
    npr = {d: [0, 0] for d in DILATE}         # [n_pred, n_gt]
    area = {d: [] for d in DILATE}
    mir = {d: [] for d in MIRROR_AT}
    id_fail = []                              # C1: panels where d=0 is not 1.0
    d1_changed = 0
    areas_all = []
    rows_csv = []
    mont = []
    n_norm = n_nb = n_zero = 0

    for idx, name in enumerate(names):
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = fm.ink_of(g)
        if ink.mean() < NEAR_BLANK_INK:
            n_nb += 1
            continue                           # stratum unsupported (IC6)
        n_norm += 1
        gt = fm.fill_mask(ink)
        lab, ids, stats = rm.regions(gt, rm.MIN_REGION_PX)
        n = len(ids)
        if n == 0:
            n_zero += 1
            continue                           # F1 = 1.0 here is an IDENTITY
        areas_all.extend(int(stats[i, cv2.CC_STAT_AREA]) for i in ids)

        # C1, the identity half
        p0 = dilate_mask(gt, 0.0)
        if rm.match(p0, gt)["region_f1"] != 1.0 or not np.array_equal(p0, gt):
            id_fail.append(name)
        if dilate_mask(gt, 1.0).sum() > gt.sum():
            d1_changed += 1

        row = {"panel": name, "n_regions": n, "gt_area_frac": round(
            float(gt.mean()), 6)}
        for d in DILATE:
            pred = dilate_mask(gt, d)
            r = rm.match(pred, gt)
            pi = rm.pixel_iou(pred, gt)["iou"]
            f1[d].append(r["region_f1"])
            piou[d].append(pi)
            mt[d][0] += r["matched"]
            mt[d][1] += r["n_gt"]
            npr[d][0] += r["n_pred"]
            npr[d][1] += r["n_gt"]
            area[d].append(float(pred.mean()))
            key = f"d{d:g}"
            row[f"{key}_f1"] = round(r["region_f1"], 6)
            row[f"{key}_piou"] = round(pi, 6)
            row[f"{key}_matched"] = r["matched"]
            row[f"{key}_npred"] = r["n_pred"]
            if d in MIRROR_AT:
                mf = rm.match(pred[:, ::-1], gt)["region_f1"]
                mir[d].append(mf)
                row[f"{key}_mirror_f1"] = round(mf, 6)
            if name == MONTAGE_PANEL:
                mont.append((f"dilate d={d:g}px", gt, pred))
        rows_csv.append(row)
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(names)}", flush=True)

    n_used = len(rows_csv)
    print(f"\npanels: normal {n_norm}, near-blank {n_nb} (SKIPPED, stratum "
          f"unsupported), normal with 0 regions {n_zero} (region-F1 1.0 there "
          f"is an IDENTITY and is excluded)")
    print(f"panels used: {n_used} (>= {MIN_PANELS} required -> "
          f"{'ok' if n_used >= MIN_PANELS else 'INSUFFICIENT'})")
    print(f"regions: {len(areas_all)}, median area "
          f"{np.median(areas_all):.0f}px (IC6: 22602, 1500px)")

    # ---- C1
    c1 = (not id_fail) and d1_changed > 0 and n_used >= MIN_PANELS
    areas_mono = all(np.mean(area[DILATE[i]]) < np.mean(area[DILATE[i + 1]])
                     for i in range(len(DILATE) - 1))
    c1 &= areas_mono
    print(f"\nC1  IDENTITY at d = 0: region-F1 exactly 1.000, "
          f"{len(id_fail)} of {n_used} panels differ (bar: 0) -> "
          f"{'PASS' if not id_fail else 'FAIL'}")
    print(f"    NC-d0 the non-identity half: d = 1 changes the mask on "
          f"{d1_changed}/{n_used} panels (bar: >= 1), mean predicted area "
          f"strictly increasing in d -> {'yes' if areas_mono else 'NO'}")
    if not c1:
        print("\nC1 FAILED -> stop rule: no verdict is read, nothing below "
              "this line is quoted.")
        if id_fail[:5]:
            print(f"    first failing panels: {id_fail[:5]}")
        return False

    # ---- the table
    print(f"\nDILATION -- the leg that measures. The disc model is EXPECTED to "
          f"overshoot (merging, elongation).")
    print(f"{'d':>7}{'grid':>9}{'region-F1':>11}{'pixel IoU':>11}"
          f"{'match rate':>12}{'predicted':>11}{'calib':>8}"
          f"{'npred/ngt':>11}{'pred area':>11}")
    prev = None
    mono = True
    f1m = {}
    mrate = {}
    for d in DILATE:
        m = float(np.mean(f1[d]))
        f1m[d] = m
        pim = float(np.mean(piou[d]))
        mr = mt[d][0] / max(mt[d][1], 1)
        mrate[d] = mr
        ratio = npr[d][0] / max(npr[d][1], 1)
        reg = EXPECT_DILATE.get(d)
        pc = predicted_matching(areas_all, d)
        cal = predicted_matching(areas_all, d, calibrated=True)
        if prev is not None and mr > prev + 1e-12:
            mono = False
        prev = mr
        grid = "sqrt2*REG" if reg is not None else "integer"
        pstr = f"{reg:.3f}" if reg is not None else f"({pc:.3f})"
        print(f"{d:>7.3f}{grid:>9}{m:>11.4f}{pim:>11.4f}{mr:>12.3f}"
              f"{pstr:>11}{cal:>8.3f}{ratio:>11.3f}"
              f"{float(np.mean(area[d])):>11.4f}")
    print(f"    predicted: REGISTERED on the sqrt(2) grid (= IC6's erosion "
          f"column, ic6_run.txt); (parenthesised) = computed in-run from this "
          f"census, NOT registered. pixel IoU is printed beside every "
          f"region-F1 (stage-2 C5).")
    print(f"    npred/ngt < 1 means dilation MERGED regions; > 1 would be a "
          f"bug, since dilation is monotone in the mask.")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ic7_per_panel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\nper-panel: {OUT / 'ic7_per_panel.csv'}")
    print(f"montage ({MONTAGE_PANEL}): "
          f"{montage(mont, OUT / 'ic7_overpaint.png')}")

    # ---- C2
    print(f"\nC2  match rate monotonically non-increasing in d -> "
          f"{'PASS' if mono else 'FAIL'}")

    # ---- C3
    graded = [d for d in DILATE if C3_BAND[0] <= f1m[d] <= C3_BAND[1]]
    sep_ok = all(min(f1m[d] - BLANK_F1, f1m[d] - NC3_F1) >= C3_SEP
                 for d in graded)
    c3 = bool(graded) and sep_ok
    print(f"C3  (WEAK GATE, labelled as one) at least one radius in "
          f"[{C3_BAND[0]}, {C3_BAND[1]}] and >= {C3_SEP} above blank "
          f"{BLANK_F1:.4f} and nc3_hflip:rough_dark {NC3_F1:.4f}: "
          f"{len(graded)} radii ({', '.join(f'{d:g}' for d in graded)}) -> "
          f"{'PASS' if c3 else 'FAIL'}")

    # ---- NC-hflip
    nc_ok = True
    for d in MIRROR_AT:
        mm = float(np.mean(mir[d]))
        ok = mm <= NC_HFLIP_BAR
        nc_ok &= ok
        word = ("collapses, so C3 reads placement" if ok else
                "DOES NOT COLLAPSE -> C3 IS WITHDRAWN")
        print(f"    NC-hflip at d = {d:g}: the SAME prediction mirrored "
              f"left-right scores {mm:.4f} (bar: <= {NC_HFLIP_BAR}) -> "
              f"{word}")
    if not nc_ok:
        c3 = False

    # ---- C4
    half = next((d for d in DILATE if mrate[d] <= 0.5), None)
    c4 = half is not None and half >= C4_MIN_HALF_PX
    if half is None:
        print(f"\nC4  the match rate never reaches 0.5 inside the grid "
              f"(max d = {DILATE[-1]:g}px) -> tolerance exceeds the grid")
        c4 = True
    else:
        print(f"\nC4  measured dilation half-match bracket: d <= {half:g}px; "
              f"registered prediction {EXPECT_HALF_PX:.2f}px; IC6's MEASURED "
              f"erosion bracket {C4_MIN_HALF_PX:g}px")
        word = (f"ASYMMETRIC AS PREDICTED (dilation tolerated to >= "
                f"{C4_MIN_HALF_PX:g}px, erosion's measured bracket)" if c4
                else "REFUTED: over-painting is punished at a smaller radius "
                     "than under-painting")
        print(f"    -> {word}")

    # ---- C4b, per-panel on the shared panels
    print(f"\nC4b equal-radius comparison with IC6's erosion leg, on the "
          f"panels both runs used")
    c4b = True
    if ic6:
        shared = [r["panel"] for r in rows_csv if r["panel"] in ic6]
        print(f"    shared panels: {len(shared)}")
        # the assert with a known value: the recomputed IC6 means must
        # reproduce the registered table (ic6_run.txt) to 0.001
        for e, want in IC6_ERODE_F1.items():
            got = float(np.mean([ic6[p][e] for p in shared]))
            assert abs(got - want) < 1.5e-3, (e, got, want)
        print(f"    assert: recomputed IC6 erosion means reproduce "
              f"ic6_run.txt to <0.0015 at all six radii -> PASS")
        print(f"{'radius':>8}{'dilate F1':>11}{'erode F1':>11}"
              f"{'dilate-erode':>14}{'C4b':>8}")
        f1_by_panel = {r["panel"]: r for r in rows_csv}
        for e in (1, 2, 4, 6, 8, 12):
            dv = float(np.mean([f1_by_panel[p][f"d{e}_f1"] for p in shared]))
            ev = float(np.mean([ic6[p][e] for p in shared]))
            ok = dv >= ev
            c4b &= ok
            print(f"{e:>8}{dv:>11.4f}{ev:>11.4f}{dv - ev:>+14.4f}"
                  f"{'ok' if ok else 'VIOLATED':>8}")
        print(f"    C4b dilate >= erode at ALL six shared radii -> "
              f"{'PASS' if c4b else 'FAIL'}")
    else:
        c4b = False
        print(f"    {IC6_CSV} missing -> C4b NOT READ")

    # ---- C5
    ratio5 = npr[C5_AT][0] / max(npr[C5_AT][1], 1)
    c5 = ratio5 < C5_RATIO_BAR
    # AMENDMENT L2 (2026-10-18): the alternative this used to print --
    # "regions are far enough apart ... any overshoot is elongation alone" --
    # is WITHDRAWN. The ratio is moved UP by dilated sub-500px blobs crossing
    # the filter and DOWN by merging, so it cannot test either one alone.
    # ic7_npred_decomposition.py measured both: at d = 8.485, 57.5% of
    # predicted regions are crossers AND 14.2% of GT regions sit in a merged
    # carrier.
    word5 = ("MERGING FIRED" if c5 else
             "REFUTED as registered -- and the ratio cannot be read as "
             "evidence either way (L2): crossers push it up while merging "
             "pushes it down. See results/fill_koma_20261010/"
             "ic7_npred_decomposition.txt for the decomposition that can.")
    print(f"\nC5  merging at d = {C5_AT:g}: npred/ngt {ratio5:.3f} "
          f"(bar: < {C5_RATIO_BAR}) -> {word5}")

    print(f"\nC1 {'PASS' if c1 else 'FAIL'}  C2 {'PASS' if mono else 'FAIL'}  "
          f"C3 {'PASS' if c3 else 'FAIL'}  C4 {'PASS' if c4 else 'FAIL'}  "
          f"C4b {'PASS' if c4b else 'FAIL'}  C5 "
          f"{'PASS' if c5 else 'REFUTED'}")
    return c1 and mono and c3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print("self-check: PASS (assert cell, hand-computed: the area-threshold "
          "ratio is exactly 2; 9.05px derived twice; d=0 is the identity; a "
          "disc R=20 matches at d=6 (IoU 0.5917) and not at d=10 (0.4444))")
    if args.check:
        sys.exit(0)
    sys.exit(0 if run(args.limit) else 1)


if __name__ == "__main__":
    main()
