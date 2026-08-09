"""Quantitatively compare fixed-sample line-art outputs against ground truth.

**2026-08-09 fix**: ink extraction used to be a raw grayscale threshold
(`gray < 128`). That is fine for clean scanned line art, but diffusion-model
outputs are frequently soft/antialiased rather than crisply binarized, and a
raw threshold picks up broad faint-gray regions as "ink" -- measured on a
real comparison this inflated `ink_ratio` to 60-165x the GT ink count (see
`doc/eval_metric_inventory.md`), which then confounds precision/recall/
chamfer on top of the separate truncate-radius saturation issue documented
in `doc/work_log.md`'s 2026-08-09 "Workstream C" entries. Default extraction
is now Canny-edge-based (`tile_region_manifest_480.edge_map`), matching the
extraction already validated for the conditioning-roundtrip metric.
`--extraction threshold` reproduces the old (buggy-for-diffusion-output,
fine-for-clean-scans) behavior for anyone needing to compare against
historical numbers that used it.

Also dropped the default truncate radius 20px->8px: at 20px, dense line art
saturates chamfer regardless of correspondence quality (see the same
work_log entries); 8px keeps the metric responsive without being so tight
that ordinary sub-pixel jitter between real scans dominates. Still a
CLI-overridable parameter, not a hardcoded assumption.

**2026-08-09 addition**: a same-day literature check
(`doc/eval_metric_literature_survey_20260809.md`) found that `f1_2px`/
`precision_2px`/`recall_2px` above are a many-to-one tolerance match (any
number of predicted pixels can each independently claim the same target
pixel), materially weaker than the BSDS boundary-detection benchmark's
one-to-one bipartite-matched F-score their names evoke. Added `bsds_f1`/
`bsds_precision`/`bsds_recall` (`tile_region_manifest_480.
bipartite_match_f1`) alongside them, not replacing them -- validated 6/6
against the 2026-08-09 hand fidelity ranking (vs. chamfer's 1/6), the best
result of any metric tried so far. Kept the old columns for continuity
with historical CSVs rather than breaking them silently.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pair_extraction"))
from tile_region_manifest_480 import bipartite_match_f1, edge_map  # noqa: E402


IMAGE_SIZE = 480
THRESHOLD = 128
TOLERANCE_PX = 2.0
TRUNCATE_PX = 8.0
DEFAULT_EXTRACTION = "edge"
DEFAULT_MODELS = ["std15", "warm2", "warm_regions"]
SAMPLES = [
    "housei_002_06_15",
    "housei_002_07_12",
    "housei_002_19_12",
    "lineart_004_002",
    "lineart_004_004",
    "lineart_004_006",
    "lineart_004_008",
    "lineart_004_010",
]
OUTPUT_CSV = "results/fixed_output_metrics.csv"


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


def load_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def extract_ink(path, extraction):
    gray = load_gray(path)
    if extraction == "threshold":
        return gray < THRESHOLD
    return edge_map(gray)


def distance_to_ink(ink):
    return cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def metrics(pred, target, truncate_px):
    pred_dist = distance_to_ink(pred)
    target_dist = distance_to_ink(target)
    pred_count = max(int(pred.sum()), 1)
    target_count = max(int(target.sum()), 1)

    precision = float((target_dist[pred] <= TOLERANCE_PX).sum() / pred_count)
    recall = float((pred_dist[target] <= TOLERANCE_PX).sum() / target_count)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    chamfer = 0.5 * (
        float(np.minimum(target_dist[pred], truncate_px).mean())
        + float(np.minimum(pred_dist[target], truncate_px).mean())
    )
    bsds_f1, bsds_precision, bsds_recall = bipartite_match_f1(pred, target, TOLERANCE_PX)
    pred_ink = float(pred.mean())
    target_ink = float(target.mean())
    return {
        "precision_2px": precision,
        "recall_2px": recall,
        "f1_2px": f1,
        "bsds_precision": bsds_precision,
        "bsds_recall": bsds_recall,
        "bsds_f1": bsds_f1,
        "chamfer_px": chamfer,
        "pred_ink": pred_ink,
        "target_ink": target_ink,
        "ink_ratio": pred_ink / max(target_ink, 1e-9),
    }


def main(models, samples, output_csv, split, extraction, truncate_px):
    rows = []
    for name in samples:
        target = extract_ink(dataset_path(name, split), extraction)
        for model in models:
            base = normalize_name(name)
            pred = extract_ink(f"results/{model}/{base}_out.png", extraction)
            row = {"sample": base, "model": model, **metrics(pred, target, truncate_px)}
            rows.append(row)

    fields = list(rows[0])
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("model         F1@2px  bsds_F1  chamfer  ink_ratio  precision  recall  bsds_prec  bsds_rec")
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        means = {
            key: np.mean([row[key] for row in model_rows])
            for key in (
                "f1_2px", "bsds_f1", "chamfer_px", "ink_ratio", "precision_2px", "recall_2px",
                "bsds_precision", "bsds_recall",
            )
        }
        print(
            f"{model:12s}  {means['f1_2px']:.4f}  {means['bsds_f1']:.4f}  {means['chamfer_px']:7.3f}"
            f"  {means['ink_ratio']:9.3f}  {means['precision_2px']:.4f}"
            f"  {means['recall_2px']:.4f}  {means['bsds_precision']:.4f}  {means['bsds_recall']:.4f}"
        )
    print(f"\nsaved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sample-list", default=None)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    parser.add_argument(
        "--extraction", choices=["edge", "threshold"], default=DEFAULT_EXTRACTION,
        help="'edge' (default, Canny-based, correct for diffusion output) or "
             "'threshold' (raw gray<128, reproduces pre-2026-08-09 numbers)",
    )
    parser.add_argument("--truncate-px", type=float, default=TRUNCATE_PX)
    args = parser.parse_args()
    samples = read_sample_list(args.sample_list) if args.sample_list else SAMPLES
    main(args.models, samples, args.output_csv, args.split, args.extraction, args.truncate_px)
