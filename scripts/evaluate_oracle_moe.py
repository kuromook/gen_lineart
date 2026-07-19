import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 480
THRESHOLD = 128
TOLERANCE_PX = 2.0
TRUNCATE_PX = 20.0


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def parse_model(value):
    if "=" in value:
        label, directory = value.split("=", 1)
        return label, Path(directory)
    return value, Path("results") / value


def load_ink(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
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


def score(row, mode):
    if mode == "f1":
        return row["f1_2px"]
    if mode == "chamfer":
        return -row["chamfer_px"]
    if mode == "balanced":
        ink_penalty = abs(np.log(max(row["ink_ratio"], 1e-6)))
        return row["f1_2px"] - 0.025 * row["chamfer_px"] - 0.03 * ink_penalty
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--score-mode", default="balanced", choices=["balanced", "f1", "chamfer"])
    args = parser.parse_args()

    samples = read_sample_list(args.sample_list)
    models = [parse_model(value) for value in args.model]
    oracle_dir = Path(args.oracle_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    choices = []
    for sample in samples:
        base = normalize_name(sample)
        target = load_ink(Path("dataset/pairs_480") / args.split / "line" / f"{base}.jpg")
        sample_rows = []
        for label, directory in models:
            pred_path = directory / f"{base}_out.png"
            if not pred_path.exists():
                continue
            row = {
                "sample": base,
                "model": label,
                "path": str(pred_path),
                **metrics(load_ink(pred_path), target),
            }
            rows.append(row)
            sample_rows.append(row)
        if not sample_rows:
            raise RuntimeError(f"No expert outputs found for {base}")
        best = max(sample_rows, key=lambda row: score(row, args.score_mode))
        shutil.copyfile(best["path"], oracle_dir / f"{base}_out.png")
        choices.append({
            "sample": base,
            "expert": best["model"],
            "score": score(best, args.score_mode),
            "f1_2px": best["f1_2px"],
            "chamfer_px": best["chamfer_px"],
            "ink_ratio": best["ink_ratio"],
        })

    fields = [
        "sample",
        "model",
        "path",
        "precision_2px",
        "recall_2px",
        "f1_2px",
        "chamfer_px",
        "pred_ink",
        "target_ink",
        "ink_ratio",
    ]
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for choice in choices:
        counts[choice["expert"]] = counts.get(choice["expert"], 0) + 1
    summary = {
        "score_mode": args.score_mode,
        "oracle_dir": str(oracle_dir),
        "counts": counts,
        "choices": choices,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
