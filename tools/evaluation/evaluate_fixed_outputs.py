"""Quantitatively compare fixed-sample line-art outputs against ground truth."""

import argparse
import csv
import os

import cv2
import numpy as np


IMAGE_SIZE = 480
THRESHOLD = 128
TOLERANCE_PX = 2.0
TRUNCATE_PX = 20.0
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


def load_ink(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image < THRESHOLD


def distance_to_ink(ink):
    return cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def metrics(pred, target):
    pred_dist = distance_to_ink(pred)
    target_dist = distance_to_ink(target)
    pred_count = max(int(pred.sum()), 1)
    target_count = max(int(target.sum()), 1)

    precision = float((target_dist[pred] <= TOLERANCE_PX).sum() / pred_count)
    recall = float((pred_dist[target] <= TOLERANCE_PX).sum() / target_count)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    chamfer = 0.5 * (
        float(np.minimum(target_dist[pred], TRUNCATE_PX).mean())
        + float(np.minimum(pred_dist[target], TRUNCATE_PX).mean())
    )
    pred_ink = float(pred.mean())
    target_ink = float(target.mean())
    return {
        "precision_2px": precision,
        "recall_2px": recall,
        "f1_2px": f1,
        "chamfer_px": chamfer,
        "pred_ink": pred_ink,
        "target_ink": target_ink,
        "ink_ratio": pred_ink / max(target_ink, 1e-9),
    }


def main(models, samples, output_csv, split):
    rows = []
    for name in samples:
        target = load_ink(dataset_path(name, split))
        for model in models:
            base = normalize_name(name)
            pred = load_ink(f"results/{model}/{base}_out.png")
            row = {"sample": base, "model": model, **metrics(pred, target)}
            rows.append(row)

    fields = list(rows[0])
    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("model         F1@2px  chamfer  ink_ratio  precision  recall")
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        means = {
            key: np.mean([row[key] for row in model_rows])
            for key in ("f1_2px", "chamfer_px", "ink_ratio", "precision_2px", "recall_2px")
        }
        print(
            f"{model:12s}  {means['f1_2px']:.4f}  {means['chamfer_px']:7.3f}"
            f"  {means['ink_ratio']:9.3f}  {means['precision_2px']:.4f}"
            f"  {means['recall_2px']:.4f}"
        )
    print(f"\nsaved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sample-list", default=None)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    args = parser.parse_args()
    samples = read_sample_list(args.sample_list) if args.sample_list else SAMPLES
    main(args.models, samples, args.output_csv, args.split)
