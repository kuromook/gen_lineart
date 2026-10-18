"""Read the stage-2 non-learned table in the order the registration requires.

Negative controls FIRST; an arm that falls to one is not read at all (C4).
Then the arms, per family, with the near-blank and normal strata apart and
both readings beside each other -- no overall average is a headline (C5).

Verdict words always carry their threshold (second foundation notice):
  "falls to NC2"  := the arm's region-F1 ON THE NC2 CROP (`arm@nc2crop`)
                     <= its own nc2_offbyone value. Amendment H1 (2026-10-18)
                     makes NC2 a common-overlap centre crop, so the arm is
                     scored on that same crop and the control is read against
                     THAT row. Reading it against the full-panel row would
                     compare two geometries and is the defect H1 exists to
                     remove.
  "falls to NC3"  := region-F1 - nc3_hflip region-F1 < 0.05
  "beats blank"   := region-F1 - blank region-F1 >= 0.05   (C1's margin)

Identities are labelled as identities: `blank` scores 0 unless the GT is also
empty, and `all_black`'s pixel IoU is |G|/|P| by construction.

Usage: report_fill_koma.py --csv results/fill_koma_20261010/per_panel_held.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

C1_MARGIN = 0.05
NC3_MARGIN = 0.05
BASE_ARMS = ["blank", "all_black", "rough_dark", "rough_dark@budget",
             "freq_table", "nc1_polarity"]
NCS = ["nc2_offbyone", "nc3_hflip", "nc5_shift50"]


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k in ("region_f1", "iou", "pred_area_frac", "gt_area_frac",
                  "pred_fill_ratio", "gt_fill_ratio", "precision", "recall"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        r["near_blank"] = int(r["near_blank"])
        r["n_gt"] = int(r["n_gt"])
        rows.append(r)
    return rows


def mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), float)]
    return sum(v) / len(v) if v else float("nan")


def by_arm(rows, stratum):
    out = defaultdict(list)
    for r in rows:
        if stratum == "normal" and r["near_blank"]:
            continue
        if stratum == "near_blank" and not r["near_blank"]:
            continue
        out[r["arm"]].append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows = load(args.csv)
    panels = {r["panel"] for r in rows}
    print(f"{len(panels)} panels, {len(rows)} rows: {args.csv}")
    print(f"near-blank panels: {len({r['panel'] for r in rows if r['near_blank']})}\n")

    for stratum in ("normal", "near_blank"):
        a = by_arm(rows, stratum)
        if not a:
            continue
        n = len({r["panel"] for r in a.get("blank", [])})
        print(f"{'=' * 96}\n{stratum} stratum, {n} panels   "
              f"(region-F1: 8-conn fill components >=500px, IoU>=0.5, greedy 1-1)")

        print("\nNEGATIVE CONTROLS FIRST -- an arm that falls to one is not read")
        print(f"{'control':>22}{'region-F1':>11}{'pixel IoU':>11}{'pred area':>11}"
              f"{'named error':>46}")
        named = {"nc1_polarity": "the rough read as white-on-black",
                 "nc2_offbyone": "panel i's prediction vs panel i+1's GT",
                 "nc3_hflip": "the prediction mirrored left-right",
                 "nc5_shift50": "the prediction rolled 50px (sensitivity only)"}
        if "nc1_polarity" in a:
            s = a["nc1_polarity"]
            print(f"{'nc1_polarity':>22}{mean(s, 'region_f1'):11.4f}"
                  f"{mean(s, 'iou'):11.4f}{mean(s, 'pred_area_frac'):11.4f}"
                  f"{named['nc1_polarity']:>46}")
        for nc in NCS:
            for arm in ("rough_dark", "rough_dark@budget", "freq_table", "all_black"):
                k = f"{nc}:{arm}"
                if k in a:
                    s = a[k]
                    print(f"{k:>22}{mean(s, 'region_f1'):11.4f}"
                          f"{mean(s, 'iou'):11.4f}{mean(s, 'pred_area_frac'):11.4f}"
                          f"{named[nc]:>46}")
                    # H1: NC2 is read against the arm on the SAME crop, so
                    # that row is printed immediately under it. The arm's
                    # full-panel row is in the ARMS table and is NOT what the
                    # NC2 verdict compares against.
                    ck = f"{arm}@nc2crop"
                    if nc == "nc2_offbyone" and ck in a:
                        cs = a[ck]
                        print(f"{ck:>22}{mean(cs, 'region_f1'):11.4f}"
                              f"{mean(cs, 'iou'):11.4f}"
                              f"{mean(cs, 'pred_area_frac'):11.4f}"
                              f"{'the arm itself on that same crop':>46}")

        print(f"\nARMS   (area_frac and the ink-normalised fill_ratio always together)")
        print(f"{'arm':>20}{'region-F1':>11}{'pixelIoU':>10}{'prec':>8}{'rec':>7}"
              f"{'predArea':>10}{'gtArea':>9}{'predRatio':>11}{'gtRatio':>9}"
              f"{'verdict':>34}")
        blank_f1 = mean(a.get("blank", []), "region_f1")
        for arm in BASE_ARMS:
            if arm not in a:
                continue
            s = a[arm]
            f1 = mean(s, "region_f1")
            nc2 = mean(a.get(f"nc2_offbyone:{arm}", []), "region_f1")
            # H1: the arm on the SAME crop. Falls back to the full-panel row
            # only when no crop row exists (a pre-H1 CSV), and says so.
            crop_key = f"{arm}@nc2crop"
            f1_crop = mean(a.get(crop_key, []), "region_f1")
            same_geom = crop_key in a
            nc3 = mean(a.get(f"nc3_hflip:{arm}", []), "region_f1")
            v = []
            if arm == "blank":
                v.append("IDENTITY: 0 unless GT empty")
            elif arm == "all_black":
                v.append("IDENTITY: pixel IoU = |G|/|P|")
            elif arm == "nc1_polarity":
                v.append("control, not an arm")
            else:
                lhs = f1_crop if same_geom else f1
                # When the arm and its NC2 agree on EVERY panel, NC2 is a
                # tautology on this material: swapping panels changed no
                # score anywhere, so the tie carries no information and is
                # not evidence that the arm fell to a control. Measured
                # per panel rather than guessed from the mean, and the
                # disagreement count is printed either way.
                #
                # First written as `predArea == 0 or 1` (a panel-independent
                # mask). That test did not fire for `freq_table`, correctly:
                # it predicts on 1 of 956 panels, so it is NOT
                # panel-independent. Agreement per panel is the thing the
                # label is actually about.
                cr = {r["panel"]: r["region_f1"] for r in a.get(crop_key, [])}
                nr = {r["panel"]: r["region_f1"]
                      for r in a.get(f"nc2_offbyone:{arm}", [])}
                shared = set(cr) & set(nr)
                n_dis = sum(1 for q in shared if cr[q] != nr[q])
                panel_independent = bool(shared) and n_dis == 0
                if nc2 == nc2 and lhs <= nc2:
                    tag = "" if same_geom else " [NO CROP ROW: geometries differ]"
                    if panel_independent:
                        v.append(f"NC2 tie carries NO INFORMATION: the arm "
                                 f"and its NC2 agree on all {len(shared)} "
                                 f"panels ({lhs:.3f}=={nc2:.3f})")
                    else:
                        v.append(f"falls to NC2 ({lhs:.3f}<={nc2:.3f}, "
                             f"{n_dis} panels differ){tag}")
                elif nc2 == nc2 and not same_geom:
                    v.append("NC2 NOT READ: no @nc2crop row")
                if nc3 == nc3 and f1 - nc3 < NC3_MARGIN:
                    v.append(f"falls to NC3 (gap {f1 - nc3:+.3f}<{NC3_MARGIN})")
                if not v:
                    v.append("beats both controls" if f1 - blank_f1 >= C1_MARGIN
                             else f"no C1 margin (gap {f1 - blank_f1:+.3f}<{C1_MARGIN})")
            print(f"{arm:>20}{f1:11.4f}{mean(s, 'iou'):10.4f}"
                  f"{mean(s, 'precision'):8.3f}{mean(s, 'recall'):7.3f}"
                  f"{mean(s, 'pred_area_frac'):10.4f}{mean(s, 'gt_area_frac'):9.4f}"
                  f"{mean(s, 'pred_fill_ratio'):11.4f}{mean(s, 'gt_fill_ratio'):9.4f}"
                  f"{'; '.join(v):>34}")

        if stratum == "normal":
            print(f"\nPER FAMILY (never pooled into one average)")
            fams = sorted({r["family"] for r in rows})
            arms = [x for x in ("rough_dark", "rough_dark@budget", "freq_table") if x in a]
            print(f"{'family':>14}{'panels':>8}" + "".join(f"{x:>20}" for x in arms))
            for f in fams:
                sub = [r for r in rows if r["family"] == f and not r["near_blank"]]
                if not sub:
                    continue
                np_ = len({r["panel"] for r in sub})
                cells = "".join(
                    f"{mean([r for r in sub if r['arm'] == x], 'region_f1'):20.4f}"
                    for x in arms)
                print(f"{f:>14}{np_:8d}{cells}")
        print()


if __name__ == "__main__":
    main()
