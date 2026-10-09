"""Aggregate the per-tile scores into the pre-registered tables and verdicts.

The strata, the primary value and the pass criteria are those registered in
doc/work_log.md 2026-10-09 and are not re-chosen here:

  primary stratum  GT ink_ratio >= 0.002 AND gt_fill_area_frac > 0.001
                   ("normal layer, and there is a fill to find")
  primary value    micro IoU -- pixel-pooled, per pool, never pooled across
                   pools (lesson 6)
  always beside it `blank` (IoU 0 by construction) and `preproc`

Instrument checks IC2/IC3 and negative controls NC1-NC4 are evaluated first
and printed first. An arm that fails its own control is marked NOT READ.

Usage:  report_fill_baselines.py --lists housei_test ako5_held ako5r_held
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results/baseline_fill_20261009")
MAIN_ARMS = ["blank", "preproc@128", "preproc@32", "preproc@otsu",
             "preproc@budget", "msgan", "all_black"]
PREPROC_ARMS = [a for a in MAIN_ARMS if a.startswith("preproc")]


def load(lists):
    rows = []
    for name in lists:
        path = R / f"per_tile_{name}.csv"
        if not path.exists():
            print(f"(missing, skipped: {path})")
            continue
        for r in csv.DictReader(open(path)):
            r["iou"] = float(r["iou"]) if r["iou"] else None
            r["f1"] = float(r["f1"]) if r["f1"] else None
            for k in ("inter_px", "pred_px", "gt_px", "union_px"):
                r[k] = int(r[k])
            for k in ("gt_ink_ratio", "gt_fill_area_frac", "pred_fill_area_frac", "radius"):
                r[k] = float(r[k])
            for k in ("near_blank", "has_gt_fill", "full_fill"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def micro_iou(rows):
    u = sum(r["union_px"] for r in rows)
    return (sum(r["inter_px"] for r in rows) / u) if u else None


def micro_f1(rows):
    d = sum(r["pred_px"] + r["gt_px"] for r in rows)
    return (2 * sum(r["inter_px"] for r in rows) / d) if d else None


def macro_iou(rows):
    vals = [r["iou"] for r in rows if r["iou"] is not None]
    return float(np.mean(vals)) if vals else None


def fmt(v, w=8, p=4):
    return " " * (w - 3) + "n/a" if v is None else f"{v:{w}.{p}f}"


def stratify(rows, name):
    if name == "normal_fill":
        return [r for r in rows if not r["near_blank"] and r["has_gt_fill"]]
    if name == "normal_nofill":
        return [r for r in rows if not r["near_blank"] and not r["has_gt_fill"]]
    if name == "near_blank":
        return [r for r in rows if r["near_blank"]]
    return rows


def arm_table(rows, pool, arms, stratum, title):
    sel = [r for r in rows if r["pool"] == pool and r["radius"] == 4.0]
    sel = stratify(sel, stratum)
    by_arm = defaultdict(list)
    for r in sel:
        by_arm[r["arm"]].append(r)
    n_tiles = len({r["tile"] for r in sel})
    gt_area = np.mean([r["gt_fill_area_frac"] for r in by_arm.get("blank", [])]) \
        if by_arm.get("blank") else float("nan")
    print(f"\n{title}  [{pool} / {stratum}]  n={n_tiles} tiles, "
          f"GT fill area {gt_area:.4f}")
    print(f"{'arm':18}{'microIoU':>10}{'microF1':>10}{'macroIoU':>10}"
          f"{'predArea':>10}{'area/GT':>9}{'tiles':>7}")
    for arm in arms:
        rs = by_arm.get(arm)
        if not rs:
            continue
        pa = float(np.mean([r["pred_fill_area_frac"] for r in rs]))
        ratio = pa / gt_area if gt_area else float("nan")
        print(f"{arm:18}{fmt(micro_iou(rs), 10)}{fmt(micro_f1(rs), 10)}"
              f"{fmt(macro_iou(rs), 10)}{pa:10.4f}{ratio:9.2f}{len(rs):7d}")
    return by_arm, gt_area, n_tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lists", nargs="+", default=["housei_test", "ako5_held", "ako5r_held"])
    args = ap.parse_args()
    rows = load(args.lists)
    if not rows:
        raise SystemExit("nothing scored yet")
    pools = [p for p in args.lists if any(r["pool"] == p for r in rows)]

    print("=" * 78)
    print("INSTRUMENT CHECKS -- nothing below is read until these pass")
    print("=" * 78)
    ic_ok = True
    for pool in pools:
        sel = [r for r in rows if r["pool"] == pool and r["radius"] == 4.0]
        ab = [r for r in sel if r["arm"] == "all_black"]
        # IC2: all_black's prediction is the whole tile, so IoU == |G|/|P|
        bad = [r for r in ab if r["iou"] is not None
               and abs(r["iou"] - r["gt_px"] / max(r["pred_px"], 1)) > 0.01]
        bl = [r for r in sel if r["arm"] == "blank"]
        bad_blank = [r for r in bl if r["pred_px"] != 0
                     or (r["iou"] not in (None, 0.0))]
        print(f"IC2 [{pool}] all_black IoU == |G|/|P|: "
              f"{len(ab) - len(bad):d}/{len(ab):d} exact -> "
              f"{'PASS' if not bad else 'FAIL'}")
        print(f"IC3 [{pool}] blank paints nothing: "
              f"{len(bl) - len(bad_blank):d}/{len(bl):d} -> "
              f"{'PASS' if not bad_blank else 'FAIL'}")
        ic_ok = ic_ok and not bad and not bad_blank
    if not ic_ok:
        print("\nINSTRUMENT CHECK FAILED -- stopping, per the registration.")
        return 1

    print("\n" + "=" * 78)
    print("NEGATIVE CONTROLS -- each a named confusion, with its registered bar")
    print("=" * 78)
    nc_verdicts = {}
    for pool in pools:
        sel = stratify([r for r in rows if r["pool"] == pool and r["radius"] == 4.0],
                       "normal_fill")
        by_arm = defaultdict(list)
        for r in sel:
            by_arm[r["arm"]].append(r)
        print(f"\n[{pool}]")
        pol = by_arm.get("nc1_polarity", [])
        ab = by_arm.get("all_black", [])
        if pol and ab:
            pol_area = float(np.mean([r["pred_fill_area_frac"] for r in pol]))
            gap = abs((micro_iou(pol) or 0) - (micro_iou(ab) or 0))
            ok = pol_area >= 0.90 and gap <= 0.02
            print(f"  NC1 polarity not inverted: fill area {pol_area:.4f} "
                  f"(bar >=0.90), gap to all_black {gap:.4f} (bar <=0.02) -> "
                  f"{'as predicted' if ok else 'NOT as predicted'}")
        for arm in ("preproc@32", "msgan"):
            true_iou = micro_iou(by_arm.get(arm, [])) or 0.0
            line = [f"  {arm:12} true micro IoU {true_iou:.4f}"]
            for nc, bar in (("nc2_offbyone", 0.10), ("nc3_transpose", None),
                            ("nc4_shift50", None)):
                rs = by_arm.get(f"{nc}:{arm}", [])
                v = micro_iou(rs)
                line.append(f"{nc.split('_')[0]} {fmt(v, 7)}")
                if nc == "nc2_offbyone":
                    nc_verdicts[(pool, arm)] = (true_iou, v or 0.0)
            print("  ".join(line))
            t, n2 = nc_verdicts[(pool, arm)]
            print(f"               -> {'above' if t > n2 + 0.01 else 'NOT above'} "
                  f"its own off-by-one control"
                  f"{'' if t > n2 + 0.01 else '  [NOT READ as alignment]'}")

    print("\n" + "=" * 78)
    print("PRIMARY: normal layer, GT has a fill  (micro IoU, per pool, radius 4)")
    print("=" * 78)
    summary = {}
    for pool in pools:
        by_arm, gt_area, n = arm_table(rows, pool, MAIN_ARMS, "normal_fill", "PRIMARY")
        summary[pool] = {a: micro_iou(rs) for a, rs in by_arm.items()}
        summary[pool]["_gt_area"] = gt_area
        summary[pool]["_n"] = n

    print("\n" + "=" * 78)
    print("SECONDARY STRATA (reported, not the main value)")
    print("=" * 78)
    for pool in pools:
        for stratum in ("near_blank", "normal_nofill", "all"):
            arm_table(rows, pool, MAIN_ARMS, stratum, "stratum")

    print("\n" + "=" * 78)
    print("RADIUS SENSITIVITY (is the conclusion a property of radius 4?)")
    print("=" * 78)
    for pool in pools:
        print(f"\n[{pool} / normal_fill]  micro IoU by fill radius")
        print(f"{'arm':18}{'r=3':>10}{'r=4':>10}{'r=6':>10}")
        for arm in ("blank", "preproc@32", "msgan", "all_black"):
            cells = []
            for radius in (3.0, 4.0, 6.0):
                sel = stratify([r for r in rows if r["pool"] == pool
                                and r["radius"] == radius and r["arm"] == arm],
                               "normal_fill")
                cells.append(fmt(micro_iou(sel), 10))
            print(f"{arm:18}" + "".join(cells))

    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 78)
    for pool in pools:
        s = summary[pool]
        best_pp_arm = max(PREPROC_ARMS, key=lambda a: s.get(a) or -1)
        best_pp = s.get(best_pp_arm) or 0.0
        ms = s.get("msgan") or 0.0
        print(f"\n[{pool}]  n={s['_n']}  GT fill area={s['_gt_area']:.4f}")
        if best_pp >= 0.20:
            c1 = (f"C1 FAILED: the preprocessor DOES fill ({best_pp_arm} "
                  f"micro IoU {best_pp:.4f} >= 0.20). This track's premise is wrong.")
        elif best_pp < 0.05:
            c1 = (f"C1 as expected: best preprocessor variant is {best_pp_arm} at "
                  f"micro IoU {best_pp:.4f} (< 0.05) -- it cannot fill.")
        else:
            c1 = (f"C1 between the bars: {best_pp_arm} at {best_pp:.4f} "
                  f"(0.05-0.20). Reported as such; not called either way.")
        print("  " + c1)
        if ms >= 0.20:
            c2 = (f"C2: msgan micro IoU {ms:.4f} >= 0.20 -- already partly solves it. "
                  f"Stage 2's comparison target becomes msgan, not `blank`.")
        elif ms < 0.05:
            c2 = (f"C2: msgan micro IoU {ms:.4f} < 0.05 -- does nothing. "
                  f"Stage 2's three baselines stand as written.")
        else:
            c2 = (f"C2: msgan micro IoU {ms:.4f} is between the bars (0.05-0.20) -- "
                  f"HELD, decided on the montage before stage 2 is registered.")
        print("  " + c2)
        print(f"  C3: blank micro IoU {fmt(s.get('blank'), 6)} by construction; "
              f"its fill-area error is the whole GT ({s['_gt_area']:.4f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
