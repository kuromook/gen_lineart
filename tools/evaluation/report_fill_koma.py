"""Read the stage-2 non-learned table in the order the registration requires.

Negative controls FIRST; an arm that falls to one is not read at all (C4).
Then the arms, per family, with the near-blank and normal strata apart and
both readings beside each other -- no overall average is a headline (C5).

Verdict words always carry their threshold (second foundation notice):
  "falls to NC2"  := region-F1 <= its own nc2_offbyone value
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
            nc3 = mean(a.get(f"nc3_hflip:{arm}", []), "region_f1")
            v = []
            if arm == "blank":
                v.append("IDENTITY: 0 unless GT empty")
            elif arm == "all_black":
                v.append("IDENTITY: pixel IoU = |G|/|P|")
            elif arm == "nc1_polarity":
                v.append("control, not an arm")
            else:
                if nc2 == nc2 and f1 <= nc2:
                    v.append(f"falls to NC2 ({f1:.3f}<={nc2:.3f})")
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
