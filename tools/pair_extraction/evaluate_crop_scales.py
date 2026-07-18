"""Evaluate source crop sizes while keeping 480px model input size.

This is a dry-run diagnostic for raw manuscript datasets. It reuses accepted
tile coordinates as anchors, crops larger source regions around the same
centers, resizes every crop to the model output size, and reports alignment and
content-density metrics by source crop size.
"""

import argparse
import csv
import io
import json
import os
import statistics
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ZIP = os.path.expanduser("~/dataset_kurip_v4.zip")
DEFAULT_ZIP_ROOT = "dataset_kurip_v4"
DEFAULT_COORDS = "results/kurip_vlm_accept_refined_tiles.csv"
DEFAULT_CSV_OUT = "results/kurip_crop_scale_eval.csv"
DEFAULT_SUMMARY_OUT = "results/kurip_crop_scale_summary.csv"
DEFAULT_MONTAGE_OUT = "results/kurip_crop_scale_eval.png"
OUTPUT_SIZE = 480


def page_id(entry):
    return Path(entry.get("file", entry["sketch"])).stem.replace("page", "")


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"{zip_root}/manifest.json"))


def load_pair(zf, zip_root, entry, autocontrast_rough):
    rough = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['line']}"))).convert("L")
    if rough.size != line.size:
        raise ValueError(f"size mismatch: rough={rough.size} line={line.size}")
    if autocontrast_rough:
        rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def read_coords(path, limit):
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            row = dict(row)
            for key in ("rank", "line_x", "line_y", "rough_x", "rough_y"):
                row[key] = int(row[key])
            rows.append(row)
    return rows[:limit] if limit else rows


def centered_crop(image, center_x, center_y, source_size, output_size):
    height, width = image.shape
    x0 = int(round(center_x - source_size / 2))
    y0 = int(round(center_y - source_size / 2))
    x0_clamped = min(max(x0, 0), width - source_size)
    y0_clamped = min(max(y0, 0), height - source_size)
    if x0_clamped < 0 or y0_clamped < 0:
        return None, True
    crop = image[y0_clamped:y0_clamped + source_size, x0_clamped:x0_clamped + source_size]
    if crop.shape != (source_size, source_size):
        return None, True
    resized = Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.LANCZOS)
    return np.asarray(resized), (x0 != x0_clamped or y0 != y0_clamped)


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def chamfer(rough_edge, line_edge, truncate):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 99.0
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def support_f1(rough_edge, line_edge, close_px):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0
    kernel = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def orientation_entropy(edges):
    image = edges.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    valid = magnitude > 0.25
    if valid.sum() < 40:
        return 0.0
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=magnitude[valid])
    probabilities = hist / max(hist.sum(), 1e-9)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(12))


def line_components(line, threshold, min_area):
    binary = (line < threshold).astype(np.uint8)
    if binary.sum() == 0:
        return 0, 0, 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= min_area]
    if not areas:
        return 0, 0, 0.0
    total = sum(areas)
    return len(areas), max(areas), max(areas) / max(total, 1)


def evaluate_pair(rough, line, args):
    rough_edge = edge_map(rough)
    line_edge = edge_map(line)
    component_count, largest_component, largest_component_share = line_components(
        line, args.line_threshold, args.min_component_area,
    )
    line_ink = float((line < args.line_threshold).mean())
    return {
        "line_ink": line_ink,
        "rough_std": float(rough.std()),
        "rough_edge_pixels": int(rough_edge.sum()),
        "line_edge_pixels": int(line_edge.sum()),
        "edge_f1": support_f1(rough_edge, line_edge, args.close_px),
        "chamfer": chamfer(rough_edge, line_edge, args.truncate_px),
        "line_entropy": orientation_entropy(line_edge),
        "component_count": component_count,
        "largest_component": largest_component,
        "largest_component_share": largest_component_share,
        "line_fragment_proxy": component_count / max(line_ink * 1000.0, 1e-9),
        "rough": rough,
        "line": line,
        "rough_edge": rough_edge,
        "line_edge": line_edge,
    }


def evaluate(rows, manifest, args):
    entries = {page_id(entry): entry for entry in manifest}
    out = []
    with zipfile.ZipFile(args.zip_path) as zf:
        cache = {}
        for index, row in enumerate(rows, 1):
            if row["page"] not in cache:
                cache[row["page"]] = load_pair(
                    zf, args.zip_root, entries[row["page"]], args.autocontrast_rough,
                )
            rough_page, line_page = cache[row["page"]]
            line_center_x = row["line_x"] + args.anchor_tile / 2
            line_center_y = row["line_y"] + args.anchor_tile / 2
            rough_center_x = row["rough_x"] + args.anchor_tile / 2
            rough_center_y = row["rough_y"] + args.anchor_tile / 2
            for source_size in args.source_sizes:
                line, line_clamped = centered_crop(
                    line_page, line_center_x, line_center_y, source_size, args.output_size,
                )
                rough, rough_clamped = centered_crop(
                    rough_page, rough_center_x, rough_center_y, source_size, args.output_size,
                )
                if line is None or rough is None:
                    continue
                metrics = evaluate_pair(rough, line, args)
                out.append({
                    "rank": row["rank"],
                    "name": row["name"],
                    "page": row["page"],
                    "source_size": source_size,
                    "output_size": args.output_size,
                    "scale_factor": source_size / args.output_size,
                    "line_clamped": int(line_clamped),
                    "rough_clamped": int(rough_clamped),
                    **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)},
                    "_images": metrics,
                })
            if index % 50 == 0:
                print(f"{index}/{len(rows)}", flush=True)
    return out


def write_eval_csv(rows, path):
    fields = [
        "rank", "name", "page", "source_size", "output_size", "scale_factor",
        "line_clamped", "rough_clamped", "line_ink", "rough_std",
        "rough_edge_pixels", "line_edge_pixels", "edge_f1", "chamfer",
        "line_entropy", "component_count", "largest_component",
        "largest_component_share", "line_fragment_proxy",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def median(values):
    return statistics.median(values) if values else 0.0


def write_summary(rows, path):
    fields = [
        "source_size", "rows", "clamped_rows", "median_line_ink",
        "median_rough_std", "median_edge_f1", "median_chamfer",
        "median_line_entropy", "median_component_count",
        "median_largest_component_share", "median_fragment_proxy",
    ]
    by_size = {}
    for row in rows:
        by_size.setdefault(row["source_size"], []).append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for source_size in sorted(by_size):
            group = by_size[source_size]
            writer.writerow({
                "source_size": source_size,
                "rows": len(group),
                "clamped_rows": sum(1 for row in group if row["line_clamped"] or row["rough_clamped"]),
                "median_line_ink": median([row["line_ink"] for row in group]),
                "median_rough_std": median([row["rough_std"] for row in group]),
                "median_edge_f1": median([row["edge_f1"] for row in group]),
                "median_chamfer": median([row["chamfer"] for row in group]),
                "median_line_entropy": median([row["line_entropy"] for row in group]),
                "median_component_count": median([row["component_count"] for row in group]),
                "median_largest_component_share": median([row["largest_component_share"] for row in group]),
                "median_fragment_proxy": median([row["line_fragment_proxy"] for row in group]),
            })


def make_montage(rows, path, count, source_sizes):
    selected_names = []
    for row in rows:
        if row["source_size"] == source_sizes[0] and row["name"] not in selected_names:
            selected_names.append(row["name"])
        if len(selected_names) >= count:
            break
    by_key = {(row["name"], row["source_size"]): row for row in rows}
    if not selected_names:
        return

    thumb = 150
    label_h = 32
    header_h = 24
    columns = len(source_sizes) * 3
    canvas = Image.new("RGB", (thumb * columns, header_h + (thumb + label_h) * len(selected_names)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except OSError:
        font = small_font = ImageFont.load_default()

    for scale_index, source_size in enumerate(source_sizes):
        for sub_index, title in enumerate(("rough", "line", "ovl")):
            draw.text(((scale_index * 3 + sub_index) * thumb + 4, 4), f"{source_size} {title}", fill=0, font=font)

    for row_index, name in enumerate(selected_names):
        y = header_h + row_index * (thumb + label_h)
        for scale_index, source_size in enumerate(source_sizes):
            row = by_key.get((name, source_size))
            if not row:
                continue
            images = row["_images"]
            overlay = np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 255, np.uint8)
            overlay[images["rough_edge"]] = (255, 60, 60)
            overlay[images["line_edge"]] = (40, 80, 255)
            for sub_index, image in enumerate((images["rough"], images["line"], overlay)):
                x = (scale_index * 3 + sub_index) * thumb
                canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (x, y))
        label = name[:80]
        draw.text((3, y + thumb + 2), label, fill=0, font=small_font)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_source_sizes(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--coords-csv", default=DEFAULT_COORDS)
    parser.add_argument("--source-sizes", type=parse_source_sizes, default=parse_source_sizes("480,720,960"))
    parser.add_argument("--anchor-tile", type=int, default=480)
    parser.add_argument("--output-size", type=int, default=OUTPUT_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--autocontrast-rough", action="store_true")
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=3)
    parser.add_argument("--truncate-px", type=int, default=30)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--montage-out", default=DEFAULT_MONTAGE_OUT)
    parser.add_argument("--montage-count", type=int, default=24)
    args = parser.parse_args()

    rows = read_coords(args.coords_csv, args.limit)
    manifest = load_manifest(args.zip_path, args.zip_root)
    evaluated = evaluate(rows, manifest, args)
    write_eval_csv(evaluated, args.csv_out)
    write_summary(evaluated, args.summary_out)
    make_montage(evaluated, args.montage_out, args.montage_count, args.source_sizes)
    print(f"input_rows={len(rows)} evaluated_rows={len(evaluated)}")
    print(f"wrote: {args.csv_out}")
    print(f"wrote: {args.summary_out}")
    print(f"wrote: {args.montage_out}")


if __name__ == "__main__":
    main()
