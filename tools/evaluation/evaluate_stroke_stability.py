"""Quantify stroke continuity/fragmentation ("wobble") in line-art outputs,
independent of GT coordinate alignment. Complements
evaluate_fixed_outputs.py's F1/chamfer/ink_ratio (alignment- and
darkness-focused) with metrics that specifically capture whether ink
forms long continuous strokes or many short disconnected fragments.

Motivation (doc/work_log.md 2026-08-01, single-stage direct-regression
"wobble" investigation): the existing metrics can't distinguish "more
training made the output more *confident* (darker/more ink)" from "more
training made the output more *stable* (fragments merging into
continuous strokes)" -- these are separate claims that happened to move
together so far, but aren't guaranteed to.

Metrics (all computed on the prediction alone, thresholded at 128 like
the other eval tools; also computed on GT for reference):

- component_count / mean_component_len / median_component_len /
  long_component_ratio: skeletonize the ink mask, take connected
  components (a continuous stroke skeletonizes to ~1 component spanning
  its length; a wobbly/fragmented stroke breaks into many short
  components). long_component_ratio is the fraction of skeleton pixels
  belonging to components at least LONG_COMPONENT_PX long.
- orientation_entropy: reused from
  tools/pair_extraction/tile_region_manifest_480.py -- entropy of local
  edge-direction histogram; low = locally coherent/directional strokes,
  high = chaotic/noisy edges.
- long_line_ratio: reused from the same module (Hough-line-based); a
  smaller --min-length-fraction than that module's panel-border default,
  since character strokes are much shorter than panel borders.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pair_extraction"))

import cv2
import numpy as np

from tile_region_manifest_480 import edge_map, long_line_ratio, orientation_entropy


IMAGE_SIZE = 480
THRESHOLD = 128
LONG_COMPONENT_PX = 20
DEFAULT_SAMPLES = [
    "housei_002_06_15",
    "housei_002_07_12",
    "housei_002_19_12",
    "lineart_004_002",
    "lineart_004_004",
    "lineart_004_006",
    "lineart_004_008",
    "lineart_004_010",
]


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def dataset_path(name, split):
    base = normalize_name(name)
    if split == "auto":
        split = "train" if base.startswith("housei") else "test"
    return f"dataset/pairs_480/{split}/line/{base}.jpg"


def load_ink_and_gray(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    ink = gray < THRESHOLD
    return ink, gray


def skeletonize(mask):
    """Morphological thinning (same approach as
    scripts/train_i2i_survey.py::skeletonize_ink)."""
    m = mask.astype(np.uint8) * 255
    skel = np.zeros_like(m)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(m) > 0:
        eroded = cv2.erode(m, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(m, opened))
        m = eroded
    return skel > 0


def component_stats(skeleton, long_threshold_px=LONG_COMPONENT_PX):
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    lengths = stats[1:, cv2.CC_STAT_AREA]  # skeleton is ~1px wide, so area ~= path length
    if len(lengths) == 0:
        return {
            "component_count": 0,
            "mean_component_len": 0.0,
            "median_component_len": 0.0,
            "long_component_ratio": 0.0,
        }
    total = float(lengths.sum())
    long_ratio = float(lengths[lengths >= long_threshold_px].sum()) / max(total, 1.0)
    return {
        "component_count": int(len(lengths)),
        "mean_component_len": float(lengths.mean()),
        "median_component_len": float(np.median(lengths)),
        "long_component_ratio": long_ratio,
    }


def stability_metrics(path, min_length_fraction):
    ink, gray = load_ink_and_gray(path)
    ink_px = int(ink.sum())
    if ink_px < 20:
        return {
            "ink_pixels": ink_px,
            "orientation_entropy": 0.0,
            "long_line_ratio": 0.0,
            "component_count": 0,
            "mean_component_len": 0.0,
            "median_component_len": 0.0,
            "long_component_ratio": 0.0,
            "components_per_1k_ink_px": 0.0,
        }
    edges = edge_map(gray)
    skel = skeletonize(ink)
    comp = component_stats(skel)
    return {
        "ink_pixels": ink_px,
        "orientation_entropy": orientation_entropy(edges),
        "long_line_ratio": long_line_ratio(ink, IMAGE_SIZE, min_length_fraction),
        **comp,
        "components_per_1k_ink_px": (comp["component_count"] / max(ink_px, 1)) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sample-list", default=None)
    parser.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    parser.add_argument("--output-csv", default="results/stroke_stability_metrics.csv")
    parser.add_argument(
        "--min-length-fraction",
        type=float,
        default=0.08,
        help="Hough long_line_ratio's min segment length as a fraction of tile size "
        "(0.08 ~= 38px for a 480 tile -- tuned for character strokes, not panel borders)",
    )
    args = parser.parse_args()

    samples = read_sample_list(args.sample_list) if args.sample_list else DEFAULT_SAMPLES

    rows = []
    for name in samples:
        base = normalize_name(name)
        gt_row = {
            "sample": base,
            "model": "GT",
            **stability_metrics(dataset_path(name, args.split), args.min_length_fraction),
        }
        rows.append(gt_row)
        for model in args.models:
            pred_path = f"results/{model}/{base}_out.png"
            row = {
                "sample": base,
                "model": model,
                **stability_metrics(pred_path, args.min_length_fraction),
            }
            rows.append(row)

    fields = list(rows[0])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "component_count",
        "mean_component_len",
        "long_component_ratio",
        "components_per_1k_ink_px",
        "orientation_entropy",
        "long_line_ratio",
    ]
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    header = "model".ljust(28) + "".join(k[:14].rjust(16) for k in metric_keys)
    print(header)
    for model in ["GT"] + [m for m in args.models]:
        model_rows = by_model[model]
        means = {k: np.mean([r[k] for r in model_rows]) for k in metric_keys}
        line = model.ljust(28) + "".join(f"{means[k]:16.3f}" for k in metric_keys)
        print(line)

    print(f"\nsaved: {args.output_csv}")


if __name__ == "__main__":
    main()
