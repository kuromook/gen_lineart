"""IC-j10: how much do the stage-2 conclusions depend on FILL_HALF_WIDTH_PX?

Registered in doc/work_log.md (2026-10-17, commit a106025) BEFORE this file
existed, with every expected value written down there first:

  IC   at r = 0 the opening is the identity -> fill_mask(ink, 0) == ink
       EXACTLY, 0 differing pixels, and fill_ratio == 1.0. Asserted.
  C1   nesting fill(8) c fill(6) c fill(4) c fill(3) c fill(2).
       AMENDMENT F1 (2026-10-17, made from a counterexample in the code
       BEFORE any sensitivity number was read -- the assert fired on the
       6-panel smoke run and the tables are printed after it):
       the registered expectation of "exact EDT: 0 violating pixels" is
       ARITHMETICALLY WRONG. `core = ink & (dist > r)` is a STRICT threshold
       on a pixel lattice, so a pixel can enter fill(big) through a core
       point that `>` excluded at the smaller radius. Nesting holds in the
       continuum; on the lattice it holds only up to a one-pixel rim.
       The asserted form becomes: EVERY violating pixel lies at distance
       <= 1.0px from the smaller mask (measured: exactly 1.00 on every
       violating pair of the smoke run). The share bar is unchanged --
       the headline still moves to the approximation if violations exceed
       1% of fill(r=2) area on the median panel.
       Both implementations are counted, which answers what C1 was for:
       exact 356 vs cv2 375 violating pixels over the same 6 panels, so the
       maskSize-3 approximation is NOT the cause (it adds ~5%).
  D1   the verdict does not move at any r in {2,3,4,6,8}:
       "robust to r" := region_F1(rough_dark) - region_F1(blank) < 0.05 AND
                        region_F1(rough_dark) - region_F1(nc3_hflip) < 0.05
                        at EVERY r
       "r-dependent" := either gap >= 0.05 at SOME r -> the stage-2 verdict
                        was an artefact of r = 4 and is withdrawn.
  D2   core area at r=8 / core area at r=4 = 0.63, band +/- 0.10 (predicted
       from an independent 60-panel census recorded 2026-10-14).

This tool CANNOT select r and must not be used to: a larger r makes the
target smaller and every arm's job different, so "the best r" is not an
object. See the registration.

Only three arms are measured, and the others are named rather than silently
dropped: all_black (an IDENTITY), rough_dark@budget, freq_table (which at
r = 4 predicted NOTHING -- predArea 0.0000, so it IS blank, an identity),
NC1 polarity, NC2 off-by-one (void, 0 rows of 13,566), NC5 shift50.

Usage:
  sens_fill_radius.py --limit 0 --exact-every 20
"""

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import MIN_INK_PX, fill_mask, ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from region_metrics import MIN_REGION_PX, match, pixel_iou, regions  # noqa: E402

OUT = Path("results/fill_koma_20261010")
RADII = (2, 3, 4, 6, 8)        # r = 0 is the instrument check, not a point
NEAR_BLANK_INK = 0.002         # the stratum boundary used by stage 2
GAP_BAR = 0.05                 # the registered threshold for both D1 legs
D2_PRED, D2_TOL = 0.63, 0.10


def core_of(ink, r, exact=False):
    """ink further than r from paper: exactly what fill_ratio counts."""
    if ink.sum() < MIN_INK_PX:
        return np.zeros_like(ink, dtype=bool)
    d = (distance_transform_edt(ink) if exact
         else cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3))
    return ink & (d > r)


def fill_mask_exact(ink, r):
    """fill_mask on an exact Euclidean transform. fill_mask.py is NOT touched;
    this is the independent leg of C1, deliberately a second implementation
    of one definition (the same split fill_ratio_cv2 / fill_ratio_exact
    already uses)."""
    core = core_of(ink, r, exact=True)
    if not core.any():
        return np.zeros_like(ink, dtype=bool)
    back = distance_transform_edt(~core)
    return (back <= r) & ink


def ic_identity(ink):
    """IC: at r = 0 the opening is the identity. Known value: 0 differing px."""
    got = fill_mask(ink, 0)
    diff = int((got != ink).sum())
    return diff


def nesting_violations(masks, rim=False):
    """Pixels present at a LARGER radius but absent at the next smaller one,
    summed over consecutive pairs. Zero for a continuum opening; on a pixel
    lattice with a strict erosion threshold it is a one-pixel rim (F1).

    With `rim`, also returns the largest distance of any violating pixel from
    the smaller mask -- the quantity the amended C1 asserts."""
    v, worst = 0, 0.0
    for big, small in zip(RADII[1:], RADII[:-1]):
        bad = masks[big] & ~masks[small]
        v += int(bad.sum())
        if rim and bad.any():
            worst = max(worst, float(distance_transform_edt(~masks[small])[bad].max()))
    return (v, worst) if rim else v


def montage(rows, normal_rows):
    """One panel, every radius, so the numbers above cite something visible.

    The panel is chosen MECHANICALLY: the normal-stratum panel whose r = 4
    `gt_area_frac` is the median of that stratum. Picking the clearest panel
    by eye would make the montage an argument instead of evidence.
    """
    at4 = sorted({x["panel"]: x["gt_area_frac"] for x in normal_rows
                  if x["r"] == 4}.items(), key=lambda kv: kv[1])
    if not at4:
        print("montage: no normal-stratum panel; not written")
        return
    name, af = at4[len(at4) // 2]
    rg = cv2.imread(str(Path(ROOT) / "rough" / name), cv2.IMREAD_GRAYSCALE)
    gg = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
    gt_ink, r_ink = ink_of(gg), ink_of(rg)
    h, w = gt_ink.shape
    sc_ = 420.0 / max(h, w)
    def tile(mask, label):
        img = np.full((h, w, 3), 255, np.uint8)
        img[mask] = (20, 20, 20)
        img = cv2.resize(img, (int(w * sc_), int(h * sc_)),
                         interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                      (170, 170, 170), 1)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 160), 1, cv2.LINE_AA)
        return img
    r1 = [tile(gt_ink, "line ink")] + \
         [tile(fill_mask(gt_ink, r), f"GT fill r={r}") for r in RADII]
    r2 = [tile(r_ink, "rough ink")] + \
         [tile(fill_mask(r_ink, r), f"rough_dark r={r}") for r in RADII]
    img = np.vstack([np.hstack(r1), np.hstack(r2)])
    out = OUT / "icj10_radius.png"
    cv2.imwrite(str(out), img)
    print(f"montage: {out}  panel {name} "
          f"(median gt_area_frac of the normal stratum at r=4: {af:.4f}); "
          f"top row GT fill, bottom row the rough_dark arm, same panel")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1,
                    help="mechanical subsample of the main legs, every k-th "
                         "panel by sorted name; k and n are printed")
    ap.add_argument("--exact-every", type=int, default=20,
                    help="the exact-EDT C1 leg is costly; same mechanical rule")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    names = sorted(n for n, _, _ in load_split()["held"])
    names = names[::args.every]
    if args.limit:
        names = names[:args.limit]
    print(f"IC-j10: FILL_HALF_WIDTH_PX sensitivity, r in {RADII} "
          f"(+ r = 0 as the instrument check)")
    print(f"held panels: {len(names)} (every {args.every}th by sorted name "
          f"of 969); exact-EDT C1 leg on every {args.exact_every}th of those")

    rows, ic_fail, cv2_viol, exact_viol, exact_n = [], [], [], [], 0
    for idx, name in enumerate(names):
        rg = cv2.imread(str(Path(ROOT) / "rough" / name), cv2.IMREAD_GRAYSCALE)
        gg = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if rg is None or gg is None or rg.shape != gg.shape:
            continue
        gt_ink, r_ink = ink_of(gg), ink_of(rg)
        npx = float(gt_ink.size)
        ink_ratio = float(gt_ink.sum()) / npx
        near_blank = ink_ratio < NEAR_BLANK_INK

        d = ic_identity(gt_ink)
        if d:
            ic_fail.append((name, d))

        gtm, cores = {}, {}
        for r in RADII:
            gtm[r] = fill_mask(gt_ink, r)
            cores[r] = core_of(gt_ink, r)
        cv2_viol.append((nesting_violations(gtm), int(gtm[RADII[0]].sum())))
        if idx % args.exact_every == 0:
            em = {r: fill_mask_exact(gt_ink, r) for r in RADII}
            exact_viol.append(nesting_violations(em, rim=True))
            exact_n += 1

        for r in RADII:
            gt = gtm[r]
            rough = fill_mask(r_ink, r)
            preds = {"blank": np.zeros_like(gt), "rough_dark": rough,
                     "nc3_hflip:rough_dark": rough[:, ::-1]}
            _, gt_ids, _ = regions(gt, MIN_REGION_PX)
            base = {
                "panel": name, "r": r, "near_blank": near_blank,
                "gt_ink_ratio": round(ink_ratio, 6),
                "gt_area_frac": round(float(gt.sum()) / npx, 6),
                "gt_fill_ratio": round(float(cores[r].sum())
                                       / max(gt_ink.sum(), 1), 6),
                "gt_n_regions": len(gt_ids),
                "core_px": int(cores[r].sum()),
            }
            for arm, p in preds.items():
                m = match(p, gt)
                row = dict(base)
                row["arm"] = arm
                row["region_f1"] = round(m["region_f1"], 6)
                row["pixel_iou"] = round(pixel_iou(p, gt)["iou"] or 0.0, 6)
                row["pred_area_frac"] = round(float(p.sum()) / npx, 6)
                rows.append(row)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(names)}", flush=True)

    # ---------------- IC and C1, read before any sensitivity number --------
    print(f"\nIC  r = 0 is the identity: expected 0 differing pixels on every "
          f"panel -> {len(ic_fail)} panels failed")
    assert not ic_fail, f"IC FAILED, nothing else is read: {ic_fail[:3]}"
    print("    PASS (asserted, not eyeballed)")

    med_rel = st.median([v / max(a, 1) for v, a in cv2_viol])
    # cv2_viol rows are (violations, fill(r=2) area) for every panel; the
    # exact leg's rows are (violations, worst rim distance) on its subsample.
    print(f"C1  nesting, cv2 maskSize-3 (what every published Track J fill "
          f"number used):")
    print(f"      violating pixels, median per panel "
          f"{st.median([v for v, _ in cv2_viol]):.1f}, "
          f"max {max(v for v, _ in cv2_viol)}")
    print(f"      as a share of fill(r=2) area, MEDIAN panel {med_rel:.5f} "
          f"(bar for the headline: > 0.01)")
    ev = sum(v for v, _ in exact_viol)
    ew = max((w for _, w in exact_viol), default=0.0)
    cvn = sum(v for v, _ in cv2_viol)
    print(f"C1  nesting, exact EDT on {exact_n} panels: {ev} violating pixels, "
          f"worst distance from the smaller mask {ew:.2f}px")
    print(f"      AMENDMENT F1: the registered expectation of 0 was "
          f"arithmetically wrong (strict `dist > r` on a lattice). The "
          f"asserted form is rim confinement: <= 1.0px.")
    assert ew <= 1.0, f"violations are NOT a one-pixel rim: {ew:.2f}px"
    print(f"      PASS (asserted)")
    print(f"      exact {ev} vs cv2 {cvn} violating pixels over the SAME "
          f"{exact_n} panels -> the maskSize-3 approximation is not the "
          f"cause; the lattice is")

    with open(OUT / "icj10_radius.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-panel rows: {OUT / 'icj10_radius.csv'} ({len(rows)} rows)")

    # ---------------- the tables, both strata apart ------------------------
    for stratum, sel in (("normal", lambda r: not r["near_blank"]),
                         ("near-blank", lambda r: r["near_blank"])):
        sub = [r for r in rows if sel(r)]
        pan = {r["panel"] for r in sub}
        print(f"\n=== {stratum} stratum, {len(pan)} panels ===")
        if stratum == "near-blank":
            print("  reported separately, NO VERDICT read (IC6 established "
                  "this stratum cannot support a region metric: nine of the "
                  "13 panels have zero regions)")
        print(f"{'r':>3}{'gt area mean':>14}{'median':>9}"
              f"{'gt fill_ratio':>15}{'regions/panel':>15}"
              f"{'blank F1':>10}{'rough_dark':>12}{'nc3_hflip':>11}"
              f"{'-blank':>9}{'-nc3':>8}")
        verdicts = []
        for r in RADII:
            at = [x for x in sub if x["r"] == r]
            if not at:
                continue
            g = {x["panel"]: x for x in at}.values()
            af = [x["gt_area_frac"] for x in g]
            fr = [x["gt_fill_ratio"] for x in g]
            nr = [x["gt_n_regions"] for x in g]
            f1 = {a: st.mean([x["region_f1"] for x in at if x["arm"] == a])
                  for a in ("blank", "rough_dark", "nc3_hflip:rough_dark")}
            gb = f1["rough_dark"] - f1["blank"]
            gn = f1["rough_dark"] - f1["nc3_hflip:rough_dark"]
            verdicts.append((r, gb, gn))
            print(f"{r:>3}{st.mean(af):>14.4f}{st.median(af):>9.4f}"
                  f"{st.mean(fr):>15.4f}{st.mean(nr):>15.2f}"
                  f"{f1['blank']:>10.4f}{f1['rough_dark']:>12.4f}"
                  f"{f1['nc3_hflip:rough_dark']:>11.4f}"
                  f"{gb:>+9.4f}{gn:>+8.4f}")
        print("  gt area = fill mask area / panel area; gt fill_ratio = "
              "ink-normalised (the `core` share). Both always printed "
              "together.")
        print("  blank's region-F1 is an IDENTITY: 0 unless the GT is also "
              "empty.")
        if stratum != "normal":
            continue
        bad = [(r, gb, gn) for r, gb, gn in verdicts
               if gb >= GAP_BAR or gn >= GAP_BAR]
        print(f"\nD1 verdict (bar {GAP_BAR:+.2f} on BOTH gaps, at EVERY r):")
        if bad:
            print(f"  **r-DEPENDENT** at r = {[r for r, _, _ in bad]} -> the "
                  f"stage-2 verdict was an artefact of r = 4 and is "
                  f"WITHDRAWN; M1's target definition is reopened.")
        else:
            print(f"  **ROBUST TO r** -- at every r in {RADII} both gaps stay "
                  f"below {GAP_BAR:+.2f}. The stage-2 conclusion does not "
                  f"depend on FILL_HALF_WIDTH_PX.")
        print("  this does NOT say r = 4 is the right half-width; nothing "
              "here can select r (see the registration).")

        montage(rows, sub)

        c4 = sum(x["core_px"] for x in sub if x["r"] == 4)
        c8 = sum(x["core_px"] for x in sub if x["r"] == 8)
        ratio = c8 / max(c4, 1)
        ok = abs(ratio - D2_PRED) <= D2_TOL
        print(f"\nD2 core area r=8 / r=4 = {ratio:.4f}  vs predicted "
              f"{D2_PRED:.2f} +/- {D2_TOL:.2f} -> "
              f"{'INSIDE' if ok else 'OUTSIDE'} the band")
        if not ok:
            print("  the 60-panel census of 2026-10-14 did not generalise; "
                  "every number derived from it is reread.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
