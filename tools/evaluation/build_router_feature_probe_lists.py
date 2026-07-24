"""Build small feature-stratified probe lists for router/oracle evaluation."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


GROUPS = {
    "clean_seed": lambda row: row["clean_router_seed"],
    "agreement_high": lambda row: row["agreement_group"] == "high",
    "agreement_low": lambda row: row["agreement_group"] == "low",
    "background_haze_high": lambda row: row["background_haze_group"] == "high",
    "line_near_uncertainty_high": lambda row: row["line_near_uncertainty_group"] == "high",
    "black_fill_high": lambda row: row["black_fill_group"] == "high",
}


SORT_KEYS = {
    "clean_seed": ("agreement_score", True),
    "agreement_high": ("agreement_score", True),
    "agreement_low": ("agreement_score", False),
    "background_haze_high": ("rough_background_haze_ink", True),
    "line_near_uncertainty_high": ("rough_line_near_uncertainty_ink", True),
    "black_fill_high": ("black_fill_score", True),
}


def truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def add_virtual_flags(rows):
    for row in rows:
        row["clean_router_seed"] = (
            float(row["agreement_percentile"]) >= 0.55
            and float(row["background_haze_percentile"]) <= 0.65
            and float(row["black_fill_percentile"]) <= 0.75
        )


def unique_by_name(rows):
    out = {}
    for row in rows:
        out.setdefault(row["name"], row)
    return list(out.values())


def source_balanced_select(rows, count, sort_key, reverse):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["source_prefix"]].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: float(row[sort_key]), reverse=reverse)

    selected = []
    used = set()
    sources = sorted(buckets, key=lambda key: len(buckets[key]), reverse=True)
    while len(selected) < count:
        progressed = False
        for source in sources:
            bucket = buckets[source]
            while bucket and bucket[0]["name"] in used:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            selected.append(row)
            used.add(row["name"])
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def write_list(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(row["name"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="router_feature_probe")
    parser.add_argument("--count-per-group", type=int, default=12)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.feature_csv)))
    add_virtual_flags(rows)
    rows = unique_by_name(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_rows = []
    combined = []
    combined_seen = set()
    for group, keep in GROUPS.items():
        candidates = [row for row in rows if keep(row)]
        sort_key, reverse = SORT_KEYS[group]
        selected = source_balanced_select(candidates, args.count_per_group, sort_key, reverse)
        write_list(output_dir / f"{args.tag}_{group}.txt", selected)
        for row in selected:
            label_rows.append(
                {
                    "name": row["name"],
                    "base": row["base"],
                    "probe_group": group,
                    "source_prefix": row["source_prefix"],
                    "agreement_score": row["agreement_score"],
                    "rough_background_haze_ink": row["rough_background_haze_ink"],
                    "rough_line_near_uncertainty_ink": row["rough_line_near_uncertainty_ink"],
                    "black_fill_score": row["black_fill_score"],
                }
            )
            if row["name"] not in combined_seen:
                combined.append(row)
                combined_seen.add(row["name"])
        print(f"{group}: candidates={len(candidates)} selected={len(selected)}")

    combined_list = output_dir / f"{args.tag}_combined.txt"
    labels_csv = output_dir / f"{args.tag}_labels.csv"
    write_list(combined_list, combined)
    with open(labels_csv, "w", newline="") as file:
        fields = list(label_rows[0])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(label_rows)
    print(f"combined={len(combined)} saved: {combined_list}")
    print(f"labels={len(label_rows)} saved: {labels_csv}")


if __name__ == "__main__":
    main()
