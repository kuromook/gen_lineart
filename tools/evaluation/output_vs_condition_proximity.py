"""Does a model's output sit on GT's strokes or on its condition map's?

Where GT and the condition map disagree, a model that learned placement from
the pairs should draw near GT; a model that only inherits its input should
draw near the condition map. Per snapshot, over every output skeleton pixel
(1px thinning, same as `stroke_churn.py`): the share within --tol px of GT
only, of the condition map only, of both, or of neither. Also the
GT-stroke view: length-weighted share of GT segments the output draws
(<= tol), cross-tabulated by whether the condition map has that stroke.

Used by Track D's hypothesis 3/4 probe (`experiments/run_h34_alignment_probe_20260915.sh`):
the pre-registered extension rule compares the "GT only" share between the
aligned and control arms. First used inline on the w=0.2 manga_line probe
(`doc/work_log.md` "Stroke-Level Churn": condition-only 31.9-36.3% vs GT-only
9.4-9.6%).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_churn import dist_to, ink_skeleton, load_gray, segments  # noqa: E402


def condition_skeleton(gray, threshold):
    from skimage.morphology import skeletonize

    mask = gray > threshold  # conditions are white strokes on black
    n = int(mask.sum())
    if n < 20 or mask.size - n < 20:
        return np.zeros_like(mask)
    return skeletonize(mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, help="dir with one subdir per snapshot of {base}_out.png")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--cond-dir", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--cond-threshold", type=int, default=32)
    parser.add_argument("--tol", type=float, default=3.0)
    parser.add_argument("--min-seg-px", type=int, default=8)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    names = [l.strip() for l in open(args.sample_list) if l.strip()]
    snaps = sorted((d for d in Path(args.results_root).iterdir() if d.is_dir() and d.name.startswith("step_")),
                   key=lambda d: int(d.name.split("_")[1]))
    acc = {d.name: {"gt_only": 0, "cond_only": 0, "both": 0, "neither": 0, "total": 0,
                    "drawn_given_cond": np.zeros(2), "len_given_cond": np.zeros(2)} for d in snaps}

    for i, name in enumerate(names):
        gt_skel = ink_skeleton(load_gray(Path(args.gt_dir) / name))
        cond_skel = condition_skeleton(load_gray(Path(args.cond_dir) / name), args.cond_threshold)
        gd, cd = dist_to(gt_skel), dist_to(cond_skel)
        segs = segments(gt_skel, args.min_seg_px)
        seg_cond = [float(np.median(np.minimum(cd[rr, cc], 99))) <= args.tol for rr, cc in segs]
        for d in snaps:
            out = ink_skeleton(load_gray(d / f"{name[:-4]}_out.png"))
            a = acc[d.name]
            if out.any():
                ng, nc = gd[out] <= args.tol, cd[out] <= args.tol
                a["gt_only"] += int((ng & ~nc).sum()); a["cond_only"] += int((nc & ~ng).sum())
                a["both"] += int((ng & nc).sum()); a["neither"] += int((~ng & ~nc).sum()); a["total"] += int(out.sum())
            od = dist_to(out)
            for (rr, cc), has_cond in zip(segs, seg_cond):
                k = 1 if has_cond else 0
                a["len_given_cond"][k] += len(rr)
                if float(np.median(np.minimum(od[rr, cc], 99))) <= args.tol:
                    a["drawn_given_cond"][k] += len(rr)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(names)}", file=sys.stderr, flush=True)

    rows = []
    print(f"{len(names)} tiles, tol {args.tol}px")
    print("snapshot      out near: GT only  cond only   both  neither | GT drawn: cond has it  cond lacks it")
    for d in snaps:
        a = acc[d.name]
        T = max(a["total"], 1)
        row = {
            "label": d.name, "step": int(d.name.split("_")[1]),
            "out_gt_only": a["gt_only"] / T, "out_cond_only": a["cond_only"] / T,
            "out_both": a["both"] / T, "out_neither": a["neither"] / T,
            "gt_drawn_where_cond_has": a["drawn_given_cond"][1] / max(a["len_given_cond"][1], 1),
            "gt_drawn_where_cond_lacks": a["drawn_given_cond"][0] / max(a["len_given_cond"][0], 1),
        }
        rows.append(row)
        print(f"{d.name:<13}{row['out_gt_only']:>18.3f}{row['out_cond_only']:>11.3f}{row['out_both']:>7.3f}{row['out_neither']:>9.3f}"
              f" |{row['gt_drawn_where_cond_has']:>21.3f}{row['gt_drawn_where_cond_lacks']:>15.3f}")
    output_csv = args.output_csv or str(Path(args.results_root) / "output_vs_condition_proximity.csv")
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
