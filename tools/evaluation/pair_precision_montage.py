"""Track D reopen (2026-10-19): the eyes for the precision numbers.

`pair_precision_report.py` says about half the rough's ink length has no GT
skeleton within 8px. This draws that claim: per pool, the pair whose
`rough_prec_aligned` sits closest to the pool median, with each skeleton
segment coloured by the stratum the measurement put it in -- green <= 3px,
amber 3-8px, red > 8px -- the other skeleton in grey behind it. Column 3 is
the precision side (rough segments, distance to GT), column 4 the recall side
(GT segments, distance to rough), so the two directions are the same picture
read from either end.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_alignment_strata import SOURCES, source_skeleton  # noqa: E402
from stroke_churn import dist_to, ink_skeleton, load_gray, segments  # noqa: E402

DATA = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data")
OUT = Path("results/pair_alignment_strata_20261019")
POOLS = ["clippairskoma", "ako5ver2koma", "fitnesskoma", "gakuenkoma", "hamlabikoma", "houseikoma"]
BAND = [(40, 140, 40), (0, 150, 230), (40, 40, 220)]  # BGR: <=3 green, 3-8 amber, >8 red


def paint(base_segs, dist, other_sk):
    img = np.full((480, 480, 3), 255, np.uint8)
    img[other_sk] = (205, 205, 205)
    for rr, cc in base_segs:
        m = float(np.median(np.minimum(dist[rr, cc], 99)))
        img[rr, cc] = BAND[0 if m <= 3 else (1 if m <= 8 else 2)]
    return cv2.erode(img, np.ones((2, 2), np.uint8))  # 1px skeletons are invisible at this scale


def label(img, lines):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    rows = defaultdict(list)
    with open(OUT / "per_pair.csv") as f:
        for r in csv.DictReader(f):
            rows[r["source"]].append(r)

    sub, polarity, thr = SOURCES["rough"]
    panels = []
    for pool in POOLS:
        rs = rows[pool]
        med = float(np.median([float(r["rough_prec_aligned"]) for r in rs]))
        pick = min(rs, key=lambda r: abs(float(r["rough_prec_aligned"]) - med))
        name = pick["name"]
        gt = load_gray(DATA / "line" / name)
        rough = load_gray(DATA / sub / name)
        gt_sk = ink_skeleton(gt)
        rough_sk = source_skeleton(rough, polarity, thr)
        gt_segs = segments(gt_sk, 8)
        rough_segs = segments(rough_sk, 8)
        prec = paint(rough_segs, dist_to(gt_sk), gt_sk)
        rec = paint(gt_segs, dist_to(rough_sk), rough_sk)
        strip = np.hstack([
            label(cv2.cvtColor(rough, cv2.COLOR_GRAY2BGR), [f"{pool}  rough", name[:46]]),
            label(cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), ["GT line"]),
            label(prec, ["PRECISION: rough segs -> GT",
                         f"prec<=3 {float(pick['rough_prec_le3']):.3f}  algn {float(pick['rough_prec_aligned']):.3f}",
                         f"3-8 {float(pick['rough_prec_3to8']):.3f}  >8 {float(pick['rough_prec_gt8']):.3f}"]),
            label(rec, ["RECALL: GT segs -> rough",
                        f"le3 {float(pick['rough_le3']):.3f}  algn {float(pick['rough_aligned']):.3f}",
                        f"3-8 {float(pick['rough_3to8']):.3f}  >8 {float(pick['rough_gt8']):.3f}"]),
        ])
        panels.append(cv2.copyMakeBorder(strip, 0, 3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        print(f"{pool:<15} median prec_aligned {med:.4f}  pair {name}")

    out = OUT / "montage_precision_vs_recall.png"
    cv2.imwrite(str(out), np.vstack(panels))
    print(f"saved: {out}  (green <=3px, amber 3-8px, red >8px; grey = the other skeleton)")


if __name__ == "__main__":
    main()
