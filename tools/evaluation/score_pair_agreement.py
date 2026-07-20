"""Score rough/line correspondence and write high/low agreement splits."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 480


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def edges(image, low, high):
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.Canny(blur, low, high) > 0


def distance_to(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def agreement_metrics(rough, line, tolerance=3.0, truncate=20.0):
    rough_edge = edges(rough, 45, 135)
    line_edge = edges(line, 45, 135)
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
        "chamfer": chamfer,
        "rough_edge_density": rough_density,
        "line_edge_density": line_density,
        "density_ratio": density_ratio,
        "agreement_score": score,
    }


def write_list(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(row["name"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--high-list", required=True)
    parser.add_argument("--low-list", required=True)
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()

    rows = []
    for name in read_list(args.file_list):
        rough = load_gray(Path(args.rough_dir) / name)
        line = load_gray(Path(args.line_dir) / name)
        rows.append({"name": name, **agreement_metrics(rough, line)})
    rows.sort(key=lambda row: row["agreement_score"], reverse=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    count = min(args.count, len(rows) // 2)
    write_list(args.high_list, rows[:count])
    write_list(args.low_list, rows[-count:])

    def mean(part, key):
        return float(np.mean([row[key] for row in part]))

    high = rows[:count]
    low = rows[-count:]
    print(f"rows={len(rows)} split_count={count}")
    print(
        "high "
        f"score={mean(high, 'agreement_score'):.4f} "
        f"f1={mean(high, 'edge_f1'):.4f} "
        f"chamfer={mean(high, 'chamfer'):.3f}"
    )
    print(
        "low  "
        f"score={mean(low, 'agreement_score'):.4f} "
        f"f1={mean(low, 'edge_f1'):.4f} "
        f"chamfer={mean(low, 'chamfer'):.3f}"
    )
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.high_list}")
    print(f"saved: {args.low_list}")


if __name__ == "__main__":
    main()
