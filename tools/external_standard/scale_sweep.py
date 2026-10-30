"""Sweep both pools over a fixed ratio list and read each axis as a function of scale.

doc/prereg_external_standard_v2_20261030.md section 6. The ratio lists and the
definition of "explained by size" were written before this ran, and the sweep
reports medians only -- no AUC.

An axis's difference is explained by size when the author's s=1.00 median falls
inside the external sweep's [min, max]. Within 5% of either end it is "borderline"
and counted as neither.
"""

import argparse
import csv
import glob
import os
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))

import normalize
from measure_lineart_profile import profile_metrics

EXTERNAL_RATIOS = [0.50, 0.625, 0.75, 1.00, 1.25, 1.50, 2.00]
AUTHOR_RATIOS = [1.00, 1.25, 1.50, 2.00]
TILE = 480
AXES = [
    "ink_ratio", "deep_black_ratio", "background_ratio",
    "faint_of_drawn_ratio", "faint_near_ink_ratio", "faint_mean_dist_to_ink",
    "long_component_ratio", "components_per_1k_ink_px",
    "line_width_p50", "width_consistency", "long_line_ratio", "orientation_entropy",
]


def one(path, ratio, tmp):
    im = Image.open(path).convert("L")
    side = int(round(im.size[0] * ratio))
    if side < TILE:
        return None
    if side != im.size[0]:
        im = im.resize((side, side), Image.LANCZOS)
    o = (side - TILE) // 2
    window = np.asarray(im.crop((o, o, o + TILE, o + TILE)))
    jpg = os.path.join(tmp, "s.jpg")
    normalize.save_jpeg(window, jpg)
    gray, _ = normalize.equalize_white_point(np.asarray(Image.open(jpg).convert("L")))
    png = os.path.join(tmp, "s.png")
    Image.fromarray(gray).save(png)
    m = profile_metrics(pathlib.Path(png), 0.08)
    m["line_width_p95"] = m.get("line_width_p95", float("nan"))
    return m


def sweep(label, paths, ratios, tmp, writer):
    for ratio in ratios:
        acc = {a: [] for a in AXES}
        acc["faint_dist_over_width"] = []
        skipped = 0
        for path in paths:
            m = one(path, ratio, tmp)
            if m is None:
                skipped += 1
                continue
            for a in AXES:
                acc[a].append(m[a])
            w = m["line_width_p50"]
            acc["faint_dist_over_width"].append(m["faint_mean_dist_to_ink"] / w if w > 0 else float("nan"))
        row = {"pool": label, "ratio": ratio, "n": len(acc[AXES[0]]), "skipped": skipped}
        for a in list(AXES) + ["faint_dist_over_width"]:
            row[a] = float(np.nanmedian(acc[a])) if acc[a] else float("nan")
        writer.writerow(row)
        print(f"  {label:22s} s={ratio:<6.3f} n={row['n']:5d} skipped={skipped:4d} "
              f"width={row['line_width_p50']:.3f} faint={row['faint_of_drawn_ratio']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/ic1_20261030/scale_sweep.csv")
    parser.add_argument("--sample", type=int, default=0, help="use every Nth image (0 = all)")
    args = parser.parse_args()

    pools = [
        ("author", sorted(glob.glob("/home/sh1/deepl/lineart/dataset/pairs_480/train/line_combined_koma_20260729/*.jpg")), AUTHOR_RATIOS),
        ("external_animagine", sorted(glob.glob("data/external_standard/gen_animagine_xl_31/*.png")), EXTERNAL_RATIOS),
        ("external_illustrious", sorted(glob.glob("data/external_standard/gen_illustrious_xl_v01/*.png")), EXTERNAL_RATIOS),
    ]
    if args.sample:
        pools = [(n, p[:: args.sample], r) for n, p, r in pools]

    tmp = tempfile.mkdtemp()
    fields = ["pool", "ratio", "n", "skipped"] + AXES + ["faint_dist_over_width"]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, paths, ratios in pools:
            print(f"{label}: {len(paths)} images x {len(ratios)} ratios", flush=True)
            sweep(label, paths, ratios, tmp, writer)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
