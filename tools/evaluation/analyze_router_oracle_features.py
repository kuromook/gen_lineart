"""Join oracle expert choices with feature probe labels and summarize routing axes."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_labels(path):
    labels = defaultdict(list)
    for row in csv.DictReader(open(path)):
        labels[row["base"]].append(row)
    return labels


def load_metrics(path):
    metrics = defaultdict(dict)
    for row in csv.DictReader(open(path)):
        metrics[row["sample"]][row["model"]] = row
    return metrics


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--oracle-summary", required=True)
    parser.add_argument("--expert-metrics-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--choice-csv", required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels_csv)
    metrics = load_metrics(args.expert_metrics_csv)
    summary = json.loads(Path(args.oracle_summary).read_text())

    choice_rows = []
    for choice in summary["choices"]:
        sample = choice["sample"]
        for label in labels.get(sample, []):
            row = {
                **label,
                "expert": choice["expert"],
                "oracle_score": choice["score"],
                "oracle_f1_2px": choice["f1_2px"],
                "oracle_chamfer_px": choice["chamfer_px"],
                "oracle_ink_ratio": choice["ink_ratio"],
            }
            expert_metric = metrics.get(sample, {}).get(choice["expert"], {})
            for key in ("precision_2px", "recall_2px", "pred_ink", "target_ink"):
                row[key] = expert_metric.get(key, "")
            choice_rows.append(row)

    Path(args.choice_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.choice_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(choice_rows[0]))
        writer.writeheader()
        writer.writerows(choice_rows)

    summary_rows = []
    for group_key in ("probe_group", "source_prefix"):
        for value in sorted({row[group_key] for row in choice_rows}):
            part = [row for row in choice_rows if row[group_key] == value]
            counts = Counter(row["expert"] for row in part)
            summary_rows.append(
                {
                    "group_by": group_key,
                    "group": value,
                    "count": len(part),
                    "expert_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                    "mean_oracle_f1_2px": mean(part, "oracle_f1_2px"),
                    "mean_oracle_chamfer_px": mean(part, "oracle_chamfer_px"),
                    "mean_oracle_ink_ratio": mean(part, "oracle_ink_ratio"),
                    "mean_agreement_score": mean(part, "agreement_score"),
                    "mean_background_haze": mean(part, "rough_background_haze_ink"),
                    "mean_line_near_uncertainty": mean(part, "rough_line_near_uncertainty_ink"),
                    "mean_black_fill_score": mean(part, "black_fill_score"),
                }
            )

    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"saved: {args.choice_csv}")
    print(f"saved: {args.output_csv}")
    for row in summary_rows:
        if row["group_by"] == "probe_group":
            print(
                f"{row['group']} n={row['count']} "
                f"experts={row['expert_counts']} "
                f"f1={row['mean_oracle_f1_2px']:.4f}"
            )


if __name__ == "__main__":
    main()
