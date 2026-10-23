"""Read the stage-2 learned arms in the order the registration requires.

Takes BOTH tables -- the non-learned one (`score_fill_koma.py`) and the model
one (`score_fill_seg_models.py`) -- because C1 compares the best arm of all
against `blank` / `all_black` / its own NC2, and C3 compares the best
non-learned arm against the best model. Reading them in two places is how a
model gets compared to a number from a different geometry.

Order, fixed 2026-10-10 and not re-chosen here:
  controls first; an arm that falls to NC2 or NC3 is excluded from every
  verdict (C4); then C1, then C2, then C3; strata apart and per family group,
  never pooled into one headline (C5); `npred/ngt` beside every region-F1
  (IC7: a low F1 with a high ratio is a speckle defect, not a "does not know
  where the fill is" defect).

Verdict words carry their thresholds, and the two baselines sit in the same
line as every number: `blank` ("don't paint") and the preprocessor, whose
readable ceiling for fill is pixel IoU 0.0000.

The trainer's own 40-panel holdout is NOT read here and cannot be: it is a
different panel set and a different geometry (it keeps the window pad). This
tool only reads the 956/969-panel table.

Usage:
  report_fill_seg.py --models-csv results/fill_koma_20261010/per_panel_held_models.csv \
                     --base-csv  results/fill_koma_20261010/per_panel_held_nc2repaired.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_fill_koma import C1_MARGIN, NC3_MARGIN, by_arm, load, mean  # noqa: E402

C2_MARGIN = 0.03          # M1 - M0, and NC4's required drop
C3_MARGIN = 0.05
MODEL_ARMS = ["M0", "M1"]
BASE_READ = ["blank", "all_black", "rough_dark", "rough_dark@budget", "freq_table"]
PREPROC_CEILING = "pixel IoU 0.0000 (IC1's 0.0002 is an arm that fell to a control)"


def ratio(rows):
    """npred/ngt over the stratum, from the summed counts rather than a mean
    of per-panel ratios (a panel with ngt = 0 has no ratio)."""
    npred = sum(int(r["n_pred"]) for r in rows)
    ngt = sum(int(r["n_gt"]) for r in rows)
    return npred, ngt, (npred / ngt if ngt else float("nan"))


def line(tag, rows, note=""):
    npred, ngt, rt = ratio(rows)
    print(f"{tag:>24}{mean(rows, 'region_f1'):11.4f}{mean(rows, 'iou'):10.4f}"
          f"{mean(rows, 'pred_area_frac'):10.4f}{mean(rows, 'gt_area_frac'):9.4f}"
          f"{npred:8d}{ngt:7d}{rt:9.3f}   {note}")


def head():
    print(f"{'arm':>24}{'region-F1':>11}{'pixelIoU':>10}{'predArea':>10}"
          f"{'gtArea':>9}{'npred':>8}{'ngt':>7}{'np/ngt':>9}")


def paired(a, b, key="region_f1"):
    """Per-panel pairing, so a difference is read on the same panels."""
    da = {r["panel"]: r[key] for r in a}
    db = {r["panel"]: r[key] for r in b}
    shared = sorted(set(da) & set(db))
    if not shared:
        return float("nan"), 0, 0, 0
    d = [da[p] - db[p] for p in shared]
    return sum(d) / len(d), len(shared), \
        sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-csv", required=True)
    ap.add_argument("--base-csv", required=True)
    args = ap.parse_args()
    mrows, brows = load(args.models_csv), load(args.base_csv)
    mpan = {r["panel"] for r in mrows}
    bpan = {r["panel"] for r in brows}
    print(f"models: {len(mpan)} panels, {len(mrows)} rows  {args.models_csv}")
    print(f"base  : {len(bpan)} panels, {len(brows)} rows  {args.base_csv}")
    shared = mpan & bpan
    print(f"panels in both: {len(shared)}  "
          f"models-only {len(mpan - bpan)}  base-only {len(bpan - mpan)}")
    if mpan != bpan:
        print(f"PANEL SETS DIFFER -- every verdict below is read on the "
              f"{len(shared)} shared panels only, and the two tables are "
              f"filtered to them first. A learned arm compared against a "
              f"non-learned arm measured on a different panel set is the "
              f"defect this guard exists to refuse.")
        mrows = [r for r in mrows if r["panel"] in shared]
        brows = [r for r in brows if r["panel"] in shared]

    # A1, checked INSIDE the table rather than against a printed figure: the
    # models tool recomputes `blank` through its own mask path, so the two
    # tables must agree on it panel by panel, not just in the mean.
    mb = {r["panel"]: r["region_f1"] for r in mrows if r["arm"] == "blank"}
    bb = {r["panel"]: r["region_f1"] for r in brows if r["arm"] == "blank"}
    both = sorted(set(mb) & set(bb))
    dis = [q for q in both if abs(mb[q] - bb[q]) > 1e-12]
    print(f"A1  blank agrees on {len(both) - len(dis)}/{len(both)} shared panels "
          f"between the two tools -> {'PASS' if both and not dis else 'FAIL'}"
          f"{'' if not dis else '  first disagreement: ' + dis[0]}")

    for stratum in ("normal", "near_blank"):
        m, b = by_arm(mrows, stratum), by_arm(brows, stratum)
        if not m:
            continue
        n = len({r["panel"] for r in m.get("M0", [])})
        print(f"\n{'=' * 110}\n{stratum} stratum, {n} panels"
              f"   (region-F1: 8-conn fill components >=500px, IoU>=0.5, greedy 1-1)")
        if stratum == "near_blank":
            print("IC6: this stratum cannot support a region metric (most panels "
                  "have zero regions). NO VERDICT is read here.")

        print("\nNEGATIVE CONTROLS FIRST -- an arm that falls to NC2 or NC3 is "
              "excluded from every verdict (C4)")
        head()
        for arm in MODEL_ARMS:
            for nc in ("nc2_offbyone", "nc3_hflip", "nc5_shift50"):
                k = f"{nc}:{arm}"
                if k in m:
                    line(k, m[k], {"nc2_offbyone": "panel i's prediction vs panel i+1's GT",
                                   "nc3_hflip": "the prediction mirrored left-right",
                                   "nc5_shift50": "rolled 50px (sensitivity only)"}[nc])
                if nc == "nc2_offbyone" and f"{arm}@nc2crop" in m:
                    line(f"{arm}@nc2crop", m[f"{arm}@nc2crop"],
                         "the arm itself on that same crop (H1)")
        if "nc4_ctxshuffle:M1" in m:
            line("nc4_ctxshuffle:M1", m["nc4_ctxshuffle:M1"],
                 "M1 with ANOTHER panel's context (C2's reading rule)")

        print(f"\nARMS   (every line beside \"don't paint\" = blank, and the "
              f"preprocessor ceiling {PREPROC_CEILING})")
        head()
        for arm in BASE_READ:
            if arm in b:
                line(arm, b[arm], "non-learned")
        for arm in MODEL_ARMS:
            if arm in m:
                line(arm, m[arm], "LEARNED, 40,000 steps")
        if "blank" in m:
            line("blank (models table)", m["blank"],
                 "anchor A1: must equal the base table's blank")

        if stratum != "normal":
            continue

        print("\nC4  controls, per model arm")
        excluded = set()
        for arm in MODEL_ARMS:
            if arm not in m:
                continue
            f1 = mean(m[arm], "region_f1")
            crop = mean(m.get(f"{arm}@nc2crop", []), "region_f1")
            nc2 = mean(m.get(f"nc2_offbyone:{arm}", []), "region_f1")
            nc3 = mean(m.get(f"nc3_hflip:{arm}", []), "region_f1")
            cr = {r["panel"]: r["region_f1"] for r in m.get(f"{arm}@nc2crop", [])}
            nr = {r["panel"]: r["region_f1"] for r in m.get(f"nc2_offbyone:{arm}", [])}
            shared = set(cr) & set(nr)
            n_dis = sum(1 for q in shared if cr[q] != nr[q])
            v = []
            if crop <= nc2:
                if shared and n_dis == 0:
                    v.append(f"NC2 tie carries NO INFORMATION: the arm and its "
                             f"NC2 agree on all {len(shared)} panels")
                else:
                    v.append(f"falls to NC2 ({crop:.4f}<={nc2:.4f}, {n_dis} panels differ)")
                    excluded.add(arm)
            if f1 - nc3 < NC3_MARGIN:
                v.append(f"falls to NC3 (gap {f1 - nc3:+.4f} < {NC3_MARGIN})")
                excluded.add(arm)
            print(f"  {arm}: region-F1 {f1:.4f}; on the NC2 crop {crop:.4f} vs its "
                  f"NC2 {nc2:.4f}; NC3 {nc3:.4f} -> {'; '.join(v) or 'clears both controls'}")
        print(f"  excluded from every verdict by C4: "
              f"{', '.join(sorted(excluded)) or 'none'}")

        print("\nC1  the best arm's region-F1 must exceed blank, all_black and "
              "its own NC2 by >= 0.05 absolute")
        cands = {a: mean(m[a], "region_f1") for a in MODEL_ARMS if a in m}
        cands.update({a: mean(b[a], "region_f1") for a in BASE_READ
                      if a in b and a not in ("blank", "all_black")})
        best = max(cands, key=lambda k: cands[k]) if cands else None
        blank = mean(b.get("blank", []), "region_f1")
        allb = mean(b.get("all_black", []), "region_f1")
        if best is not None:
            src = m if best in MODEL_ARMS else b
            f1 = cands[best]
            crop = mean(src.get(f"{best}@nc2crop", []), "region_f1")
            nc2 = mean(src.get(f"nc2_offbyone:{best}", []), "region_f1")
            print(f"  best arm of all: {best} at {f1:.4f}")
            for leg, lhs, rhs in (("vs blank", f1, blank), ("vs all_black", f1, allb),
                                  ("vs its own NC2 (same crop)", crop, nc2)):
                gap = lhs - rhs
                print(f"    {leg:>28}: {lhs:.4f} vs {rhs:.4f}  gap {gap:+.4f}  "
                      f"bar {C1_MARGIN}  -> {'PASS' if gap >= C1_MARGIN else 'FAIL'}")

        print("\nC2  M1 - M0 >= 0.03 AND M1's NC4 must drop by >= 0.03. BOTH.")
        if "M0" in m and "M1" in m:
            d, n_sh, up, dn = paired(m["M1"], m["M0"])
            print(f"  leg 1  M1 - M0 (paired on {n_sh} panels): {d:+.4f}  "
                  f"bar +{C2_MARGIN}  -> {'PASS' if d >= C2_MARGIN else 'FAIL'}"
                  f"   (M1 higher on {up}, lower on {dn})")
            if "nc4_ctxshuffle:M1" in m:
                d4, n4, u4, d4n = paired(m["M1"], m["nc4_ctxshuffle:M1"])
                print(f"  leg 2  M1 - NC4 (paired on {n4} panels): {d4:+.4f}  "
                      f"bar +{C2_MARGIN}  -> {'PASS' if d4 >= C2_MARGIN else 'FAIL'}"
                      f"   (M1 higher on {u4}, lower on {d4n})")
                both = d >= C2_MARGIN and d4 >= C2_MARGIN
                print(f"  C2 -> {'PASS' if both else 'FAIL'}"
                      f"{'' if both else '  (the koma unit does not earn its cost)'}")
            else:
                print("  leg 2  NC4 ROW MISSING: C2 is NOT READ")

        print("\nC3  if the best non-learned arm is within 0.05 region-F1 of the "
              "best model, the registered conclusion is \"no learned model is "
              "needed for this\" and that is the headline")
        nl = {a: mean(b[a], "region_f1") for a in BASE_READ if a in b}
        ml = {a: mean(m[a], "region_f1") for a in MODEL_ARMS if a in m}
        if nl and ml:
            bn, bm = max(nl, key=lambda k: nl[k]), max(ml, key=lambda k: ml[k])
            gap = ml[bm] - nl[bn]
            print(f"  best non-learned {bn} {nl[bn]:.4f}   best model {bm} "
                  f"{ml[bm]:.4f}   model - non-learned {gap:+.4f}  bar {C3_MARGIN}")
            print(f"  C3 -> {'the model earns its place' if gap >= C3_MARGIN else 'NO LEARNED MODEL IS NEEDED FOR THIS'}")

        print("\nPER FAMILY GROUP (never pooled into one average)")
        fams = sorted({r["family"] for r in mrows})
        arms = [a for a in MODEL_ARMS if a in m] + \
               [a for a in ("rough_dark", "rough_dark@budget", "blank") if a in b]
        print(f"{'family':>14}{'panels':>8}" + "".join(f"{a:>20}" for a in arms))
        for f in fams:
            sub_m = [r for r in mrows if r["family"] == f and not r["near_blank"]]
            sub_b = [r for r in brows if r["family"] == f and not r["near_blank"]]
            if not sub_m:
                continue
            np_ = len({r["panel"] for r in sub_m})
            cells = ""
            for a in arms:
                src = sub_m if a in MODEL_ARMS else sub_b
                cells += f"{mean([r for r in src if r['arm'] == a], 'region_f1'):20.4f}"
            print(f"{f:>14}{np_:8d}{cells}")


if __name__ == "__main__":
    main()
