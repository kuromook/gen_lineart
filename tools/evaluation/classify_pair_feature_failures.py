"""Assign coarse data-repair/QC categories from pair feature metrics."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def value(row, key):
    return float(row[key])


def classify(row):
    if row["source_prefix"] == "ako5" and value(row, "agreement_score") < -1.2:
        return "ako5_uninterpretable_rough_rebuild_or_exclude"
    if value(row, "black_fill_score") >= 0.05 or value(row, "line_largest_dark_cc_area_ratio") >= 0.08:
        return "black_fill_or_solid_region"
    if value(row, "rough_edge_density") < 0.006 and value(row, "line_edge_density") >= 0.010:
        return "rough_too_sparse_vs_line"
    if value(row, "rough_background_haze_ink") >= 0.08:
        return "dirty_or_hazy_rough"
    if value(row, "line_edge_density") < 0.006:
        return "line_too_sparse_fragment"
    return "valid_low_correspondence_or_metric_failure"


def write_list(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(row["name"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--group", default="agreement_low")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--list-dir", required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    group_key = f"{args.group.rsplit('_', 1)[0]}_group"
    group_value = args.group.rsplit("_", 1)[1]
    for row in csv.DictReader(open(args.feature_csv)):
        if row["name"] in seen or row.get(group_key) != group_value:
            continue
        seen.add(row["name"])
        row["repair_category"] = classify(row)
        rows.append(row)

    fields = [
        "name",
        "base",
        "source_prefix",
        "list_label",
        "repair_category",
        "agreement_score",
        "edge_f1",
        "agreement_chamfer",
        "rough_edge_density",
        "line_edge_density",
        "rough_background_haze_ink",
        "rough_line_near_uncertainty_ink",
        "black_fill_score",
        "line_largest_dark_cc_area_ratio",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)

    summary_rows = []
    for group_by in ("repair_category", "source_prefix"):
        labels = sorted({row[group_by] for row in rows})
        for label in labels:
            part = [row for row in rows if row[group_by] == label]
            counts = Counter(row["repair_category"] for row in part)
            summary_rows.append(
                {
                    "group_by": group_by,
                    "group": label,
                    "count": len(part),
                    "category_counts": dict(sorted(counts.items())),
                }
            )
    with open(args.summary_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    list_dir = Path(args.list_dir)
    for category in sorted({row["repair_category"] for row in rows}):
        write_list(list_dir / f"{args.group}_{category}.txt", [row for row in rows if row["repair_category"] == category])

    print(f"rows={len(rows)}")
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.summary_csv}")
    for category, count in Counter(row["repair_category"] for row in rows).most_common():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
