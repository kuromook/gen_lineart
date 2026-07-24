"""Score paired training tiles for router/data-split feature design."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 480
GT_THRESHOLD = 128


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def source_prefix(name):
    return normalize_name(name).split("_", 1)[0]


def edges(image, low=45, high=135):
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.Canny(blur, low, high) > 0


def distance_to(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def agreement_metrics(rough, line, tolerance=3.0, truncate=20.0):
    rough_edge = edges(rough)
    line_edge = edges(line)
    rough_count = max(int(rough_edge.sum()), 1)
    line_count = max(int(line_edge.sum()), 1)
    rough_dist = distance_to(rough_edge)
    line_dist = distance_to(line_edge)
    precision = float((rough_dist[line_edge] <= tolerance).sum() / line_count)
    recall = float((line_dist[rough_edge] <= tolerance).sum() / rough_count)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    line_to_rough = (
        float(np.minimum(rough_dist[line_edge], truncate).mean())
        if line_edge.any()
        else truncate
    )
    rough_to_line = (
        float(np.minimum(line_dist[rough_edge], truncate).mean())
        if rough_edge.any()
        else truncate
    )
    chamfer = 0.5 * (line_to_rough + rough_to_line)
    rough_density = float(rough_edge.mean())
    line_density = float(line_edge.mean())
    density_ratio = line_density / max(rough_density, 1e-9)
    score = f1 - 0.025 * chamfer - 0.08 * abs(np.log(max(density_ratio, 1e-6)))
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "agreement_chamfer": chamfer,
        "rough_edge_density": rough_density,
        "line_edge_density": line_density,
        "edge_density_ratio": density_ratio,
        "agreement_score": score,
    }


def ink_stats(gray, prefix):
    ink = 1.0 - gray.astype(np.float32) / 255.0
    return {
        f"{prefix}_gray_mean": float(gray.mean()),
        f"{prefix}_gray_std": float(gray.std()),
        f"{prefix}_ink_mean": float(ink.mean()),
        f"{prefix}_faint_area_ratio": float(((ink > 0.03) & (ink < 0.35)).mean()),
        f"{prefix}_strong_area_ratio": float((ink >= 0.35).mean()),
        f"{prefix}_very_dark_area_ratio": float((ink >= 0.75).mean()),
    }


def line_black_region_metrics(line):
    dark = line < 64
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return {
            "line_dark_cc_count": 0,
            "line_largest_dark_cc_area_ratio": 0.0,
            "line_largest_dark_cc_bbox_area_ratio": 0.0,
            "line_largest_dark_cc_fill_ratio": 0.0,
            "line_large_dark_cc_count": 0,
            "black_fill_score": 0.0,
        }
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    bbox_areas = np.maximum(widths * heights, 1.0)
    fill = areas / bbox_areas
    largest = int(np.argmax(areas))
    image_area = float(IMAGE_SIZE * IMAGE_SIZE)
    largest_area_ratio = float(areas[largest] / image_area)
    largest_bbox_ratio = float(bbox_areas[largest] / image_area)
    largest_fill = float(fill[largest])
    large_count = int(((areas / image_area) >= 0.01).sum())
    black_fill_score = largest_area_ratio * max(largest_fill, 0.0)
    return {
        "line_dark_cc_count": int(num_labels - 1),
        "line_largest_dark_cc_area_ratio": largest_area_ratio,
        "line_largest_dark_cc_bbox_area_ratio": largest_bbox_ratio,
        "line_largest_dark_cc_fill_ratio": largest_fill,
        "line_large_dark_cc_count": large_count,
        "black_fill_score": black_fill_score,
    }


def haze_against_line(pred_gray, line, inner_px=2.0, outer_px=9.0):
    pred_ink = 1.0 - pred_gray.astype(np.float32) / 255.0
    gt_ink = line < GT_THRESHOLD
    dist = distance_to(gt_ink)
    core = gt_ink
    line_near = (dist > inner_px) & (dist <= outer_px)
    far_bg = dist > outer_px
    faint = (pred_ink > 0.03) & (pred_ink < 0.35)
    strong = pred_ink >= 0.35
    core_ink = float(pred_ink[core].mean()) if core.any() else 0.0
    line_near_ink = float(pred_ink[line_near].mean()) if line_near.any() else 0.0
    bg_haze = float(pred_ink[far_bg].mean()) if far_bg.any() else 0.0
    return {
        "rough_core_ink_mean": core_ink,
        "rough_line_near_uncertainty_ink": line_near_ink,
        "rough_line_near_faint_ratio": float(faint[line_near].mean()) if line_near.any() else 0.0,
        "rough_line_near_strong_ratio": float(strong[line_near].mean()) if line_near.any() else 0.0,
        "rough_line_near_to_core": float(line_near_ink / max(core_ink, 1e-6)) if core.any() else 0.0,
        "rough_background_haze_ink": bg_haze,
        "rough_background_faint_ratio": float(faint[far_bg].mean()) if far_bg.any() else 0.0,
        "rough_background_ink_area_ratio": float((far_bg & (pred_ink > 0.03)).mean()),
        "rough_background_haze_to_core": float(bg_haze / max(core_ink, 1e-6)) if core.any() else 0.0,
    }


def add_percentiles_and_groups(rows):
    specs = [
        ("agreement_score", "agreement", False),
        ("rough_background_haze_ink", "background_haze", True),
        ("rough_line_near_uncertainty_ink", "line_near_uncertainty", True),
        ("black_fill_score", "black_fill", True),
    ]
    for key, label, high_is_bad in specs:
        ordered = sorted(range(len(rows)), key=lambda idx: rows[idx][key])
        denom = max(len(rows) - 1, 1)
        for rank, idx in enumerate(ordered):
            pct = rank / denom
            rows[idx][f"{label}_percentile"] = pct
            if label == "agreement":
                if pct >= 0.70:
                    group = "high"
                elif pct <= 0.30:
                    group = "low"
                else:
                    group = "mid"
            else:
                if high_is_bad and pct >= 0.80:
                    group = "high"
                elif high_is_bad and pct <= 0.50:
                    group = "low"
                else:
                    group = "mid"
            rows[idx][f"{label}_group"] = group


def mean(rows, key):
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def summarize(rows, keys):
    summary = []
    for group_key in ("list_label", "source_prefix"):
        labels = sorted({row[group_key] for row in rows})
        for label in labels:
            part = [row for row in rows if row[group_key] == label]
            out = {"group_by": group_key, "group": label, "count": len(part)}
            for key in keys:
                out[key] = mean(part, key)
            summary.append(out)
    out = {"group_by": "all", "group": "all", "count": len(rows)}
    for key in keys:
        out[key] = mean(rows, key)
    summary.append(out)
    return summary


def write_name_list(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(path, "w") as file:
        for row in rows:
            name = row["name"]
            if name in seen:
                continue
            seen.add(name)
            file.write(name + "\n")


def write_feature_lists(split_dir, rows):
    split_dir = Path(split_dir)
    specs = {
        "agreement_high": lambda row: row["agreement_group"] == "high",
        "agreement_mid": lambda row: row["agreement_group"] == "mid",
        "agreement_low": lambda row: row["agreement_group"] == "low",
        "background_haze_high": lambda row: row["background_haze_group"] == "high",
        "line_near_uncertainty_high": lambda row: row["line_near_uncertainty_group"] == "high",
        "black_fill_high": lambda row: row["black_fill_group"] == "high",
        "high_agreement_low_haze": lambda row: (
            row["agreement_group"] == "high"
            and row["background_haze_group"] == "low"
            and row["black_fill_group"] != "high"
        ),
        "clean_router_seed": lambda row: (
            row["agreement_percentile"] >= 0.55
            and row["background_haze_percentile"] <= 0.65
            and row["black_fill_percentile"] <= 0.75
        ),
    }
    written = []
    for label, keep in specs.items():
        part = [row for row in rows if keep(row)]
        path = split_dir / f"{label}.txt"
        write_name_list(path, part)
        written.append((label, path, len({row["name"] for row in part})))
    return written


def parse_labeled_list(value):
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", dest="lists", action="append", required=True, help="LABEL=path or path")
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--split-dir")
    parser.add_argument("--dedupe-by-name", action="store_true")
    args = parser.parse_args()

    rows = []
    seen = set()
    for list_value in args.lists:
        label, list_path = parse_labeled_list(list_value)
        for name in read_list(list_path):
            base = normalize_name(name)
            if args.dedupe_by_name and base in seen:
                continue
            seen.add(base)
            rough = load_gray(Path(args.rough_dir) / f"{base}.jpg")
            line = load_gray(Path(args.line_dir) / f"{base}.jpg")
            row = {
                "name": f"{base}.jpg",
                "base": base,
                "list_label": label,
                "source_prefix": source_prefix(base),
            }
            row.update(agreement_metrics(rough, line))
            row.update(ink_stats(rough, "rough"))
            row.update(ink_stats(line, "line"))
            row.update(line_black_region_metrics(line))
            row.update(haze_against_line(rough, line))
            rows.append(row)

    if not rows:
        raise SystemExit("no rows scored")
    add_percentiles_and_groups(rows)

    feature_keys = [
        "agreement_score",
        "edge_f1",
        "agreement_chamfer",
        "rough_gray_std",
        "rough_ink_mean",
        "rough_faint_area_ratio",
        "rough_background_haze_ink",
        "rough_line_near_uncertainty_ink",
        "line_ink_mean",
        "line_very_dark_area_ratio",
        "line_largest_dark_cc_area_ratio",
        "line_largest_dark_cc_fill_ratio",
        "black_fill_score",
    ]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows, feature_keys)
    with open(args.summary_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"rows={len(rows)}")
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.summary_csv}")
    if args.split_dir:
        for label, path, count in write_feature_lists(args.split_dir, rows):
            print(f"saved: {path} rows={count} label={label}")
    for row in summary_rows:
        if row["group_by"] == "all":
            print(
                "all "
                f"agreement={row['agreement_score']:.4f} "
                f"bg_haze={row['rough_background_haze_ink']:.4f} "
                f"line_near={row['rough_line_near_uncertainty_ink']:.4f} "
                f"black_fill={row['black_fill_score']:.5f}"
            )


if __name__ == "__main__":
    main()
