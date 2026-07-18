"""Build initial pair metadata for category-aware training experiments."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


DATASET_DIR = Path("dataset/pairs_480")
ROUGH_DIR = DATASET_DIR / "train/rough"
DEFAULT_LINE_DIR = DATASET_DIR / "train/line"
DEFAULT_OUTPUT = DATASET_DIR / "pair_metadata.csv"

SOURCE_LISTS = {
    "warm_regions": DATASET_DIR / "valid_train_warm_regions.txt",
    "ako5_regions": DATASET_DIR / "valid_train_ako5_regions.txt",
    "kurip_refined": DATASET_DIR / "valid_train_kurip_vlm_accept_refined.txt",
    "kurip_refined_mix": DATASET_DIR / "valid_train_warm_regions_kurip_vlm_accept_refined123.txt",
}

LINE_DIR_BY_SET = {
    "kurip_refined": DATASET_DIR / "train/line_kurip_vlm_accept_refined_clean_t192_cc8",
    "kurip_refined_mix": DATASET_DIR / "train/line_kurip_vlm_accept_refined123_mix_clean_t192_cc8",
}


def read_list(path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def dataset_source(name):
    if name.startswith("housei_"):
        return "housei"
    if name.startswith("lineart_") or name.startswith("orig_"):
        return "lineart"
    if name.startswith("ako5"):
        return "ako5"
    if name.startswith("kurip"):
        return "kurip"
    return "unknown"


def alignment_quality(name):
    if name.startswith("kuripr_"):
        return "locally_refined"
    if name.startswith("kuripm_") or name.startswith("ako5r_") or name.startswith("ako5a_"):
        return "rough_shifted"
    if name.startswith(("housei_", "lineart_")):
        return "same_coordinate"
    return "uncertain"


def initial_content_category(name, source, alignment):
    if source == "kurip" and alignment == "locally_refined":
        return "line_fragment"
    return "unknown"


def pair_quality(alignment):
    if alignment in {"same_coordinate", "locally_refined"}:
        return "high"
    if alignment == "rough_shifted":
        return "medium"
    return "low"


def image_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "", ""
    return float(image.std()), float((image < 128).mean())


def resolve_line_path(name, set_name):
    line_dir = LINE_DIR_BY_SET.get(set_name, DEFAULT_LINE_DIR)
    candidate = line_dir / name
    if candidate.exists():
        return candidate
    fallback = DEFAULT_LINE_DIR / name
    if fallback.exists():
        return fallback
    return candidate


def collect_rows(args):
    by_name = {}
    for set_name, list_path in SOURCE_LISTS.items():
        for name in read_list(list_path):
            if name in by_name:
                by_name[name]["source_set"] = ";".join(
                    sorted(set(by_name[name]["source_set"].split(";")) | {set_name})
                )
                continue
            rough_path = ROUGH_DIR / name
            line_path = resolve_line_path(name, set_name)
            source = dataset_source(name)
            alignment = alignment_quality(name)
            rough_std, _ = image_stats(rough_path)
            _, line_ink = image_stats(line_path)
            by_name[name] = {
                "name": name,
                "split": "train",
                "rough_path": str(rough_path),
                "line_path": str(line_path),
                "dataset_source": source,
                "content_category": initial_content_category(name, source, alignment),
                "pair_quality": pair_quality(alignment),
                "alignment_quality": alignment,
                "line_ink": line_ink,
                "rough_std": rough_std,
                "source_set": set_name,
                "notes": "",
            }
    rows = list(by_name.values())
    rows.sort(key=lambda row: (row["dataset_source"], row["name"]))
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "split",
        "rough_path",
        "line_path",
        "dataset_source",
        "content_category",
        "pair_quality",
        "alignment_quality",
        "line_ink",
        "rough_std",
        "source_set",
        "notes",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    def counts(key):
        out = {}
        for row in rows:
            out[row[key]] = out.get(row[key], 0) + 1
        return out

    print(f"rows: {len(rows)}")
    for key in ("dataset_source", "content_category", "pair_quality", "alignment_quality", "source_set"):
        print(key)
        for label, count in sorted(counts(key).items()):
            print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rows = collect_rows(args)
    write_csv(rows, Path(args.output))
    print_summary(rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
