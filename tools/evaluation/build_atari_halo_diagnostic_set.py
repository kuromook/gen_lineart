"""Build a shuffled diagnostic list for atari-halo source analysis."""

import argparse
import csv
import random
from pathlib import Path


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return Path(name).stem


def load_scores(path):
    rows = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            base = normalize_name(row["name"])
            if base in rows:
                continue
            rows[base] = {
                "agreement_score": float(row["agreement_score"]),
                "rough_edge_density": float(row["rough_edge_density"]),
                "line_edge_density": float(row["line_edge_density"]),
                "edge_f1": float(row["edge_f1"]),
                "chamfer": float(row["chamfer"]),
            }
    return rows


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def bucket(row, low_agreement, high_agreement, density_mid):
    if row["agreement_score"] <= low_agreement:
        agreement = "low"
    elif row["agreement_score"] >= high_agreement:
        agreement = "high"
    else:
        agreement = "mid"
    density = "dense" if row["rough_edge_density"] >= density_mid else "sparse"
    return f"{agreement}_{density}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-list", required=True)
    parser.add_argument("--agreement-csv", required=True)
    parser.add_argument("--exclude-list", action="append", default=[])
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-list", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    scores = load_scores(args.agreement_csv)
    excluded = set()
    for path in args.exclude_list:
        excluded.update(normalize_name(name) for name in read_list(path))

    seen = set()
    candidates = []
    for name in read_list(args.pool_list):
        base = normalize_name(name)
        if base in seen or base in excluded or base not in scores:
            continue
        seen.add(base)
        candidates.append({"name": f"{base}.jpg", **scores[base]})

    if not candidates:
        raise SystemExit("no candidates after filtering")

    low_agreement = quantile([row["agreement_score"] for row in candidates], 1 / 3)
    high_agreement = quantile([row["agreement_score"] for row in candidates], 2 / 3)
    density_mid = quantile([row["rough_edge_density"] for row in candidates], 0.5)

    buckets = {}
    for row in candidates:
        key = bucket(row, low_agreement, high_agreement, density_mid)
        row["bucket"] = key
        buckets.setdefault(key, []).append(row)

    rng = random.Random(args.seed)
    for rows in buckets.values():
        rng.shuffle(rows)

    selected = []
    keys = sorted(buckets)
    while len(selected) < args.count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < args.count:
                selected.append(buckets[key].pop())

    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_list, "w") as file:
        for row in selected:
            file.write(f"{row['name']}\n")

    fields = [
        "name",
        "bucket",
        "agreement_score",
        "rough_edge_density",
        "line_edge_density",
        "edge_f1",
        "chamfer",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(f"selected={len(selected)} from candidates={len(candidates)}")
    for key in keys:
        n = sum(1 for row in selected if row["bucket"] == key)
        print(f"{key}: {n}")
    print(f"saved: {args.output_list}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
