"""Quantify how well a model's predicted ink corresponds to the *rough*
input's own strokes, as distinct from correspondence to GT (evaluate_
fixed_outputs.py) or internal stroke continuity (evaluate_stroke_
stability.py).

Motivation (doc/work_log.md 2026-08-02, epoch-trajectory investigation):
user's visual read of the epoch-trajectory montage was that binarization/
crispness ("線画らしい白黒") looks adequate by ~epoch 55, while
faithfulness to the rough's actual content ("下絵に対する忠実さ") starts
degrading much earlier (~epoch 15) and keeps getting worse -- i.e. these
may be two different axes moving in opposite directions, not one
"confidence" axis. Reuses tools/evaluation/score_pair_agreement.py's
Canny-edge-based agreement_metrics(), originally built to score rough/GT
correspondence for data-split purposes, applied here to (rough,
prediction) instead of (rough, GT).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pair_agreement import agreement_metrics, load_gray


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--rough-dir", default="dataset/pairs_480/test/rough")
    parser.add_argument("--output-csv", default="results/rough_fidelity_metrics.csv")
    args = parser.parse_args()

    samples = read_list(args.sample_list)

    rows = []
    for name in samples:
        base = normalize_name(name)
        rough = load_gray(Path(args.rough_dir) / f"{base}.jpg")
        for model in args.models:
            pred_path = f"results/{model}/{base}_out.png"
            pred = load_gray(pred_path)
            row = {"sample": base, "model": model, **agreement_metrics(rough, pred)}
            rows.append(row)

    fields = list(rows[0])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = ["edge_f1", "edge_precision", "edge_recall", "chamfer", "density_ratio", "agreement_score"]
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    header = "model".ljust(28) + "".join(k[:14].rjust(16) for k in metric_keys)
    print(header)
    for model in args.models:
        model_rows = by_model[model]
        means = {k: np.mean([r[k] for r in model_rows]) for k in metric_keys}
        line = model.ljust(28) + "".join(f"{means[k]:16.3f}" for k in metric_keys)
        print(line)

    print(f"\nsaved: {args.output_csv}")


if __name__ == "__main__":
    main()
