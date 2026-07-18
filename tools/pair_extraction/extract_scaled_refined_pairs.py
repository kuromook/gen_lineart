"""Save scaled raw-manuscript crops from refined coordinate anchors.

The model input stays 480x480, but the source crop can be larger, such as
960x960, to normalize semantic field of view across raw manuscript datasets.
"""

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ZIP = os.path.expanduser("~/dataset_kurip_v4.zip")
DEFAULT_ZIP_ROOT = "dataset_kurip_v4"
DEFAULT_COORDS = "results/kurip_vlm_accept_refined_tiles.csv"
DEFAULT_ROUGH_OUT = "dataset/pairs_480/train/rough"
DEFAULT_LINE_OUT = "dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_clean_t192_cc8"
DEFAULT_LIST_OUT = "dataset/pairs_480/valid_train_kurip_vlm_accept_refined_s960.txt"
DEFAULT_CSV_OUT = "results/kurip_vlm_accept_refined_s960_tiles.csv"
DEFAULT_QC_OUT = "results/kurip_vlm_accept_refined_s960_qc.png"
ANCHOR_TILE = 480
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
        return None, True, None
    crop = image[y0_clamped:y0_clamped + source_size, x0_clamped:x0_clamped + source_size]
    if crop.shape != (source_size, source_size):
        return None, True, None
    resized = Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.LANCZOS)
    return np.asarray(resized), (x0 != x0_clamped or y0 != y0_clamped), (x0_clamped, y0_clamped)


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def support_f1(rough_edge, line_edge, close_px):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0
    kernel = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def chamfer(rough_edge, line_edge, truncate):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 99.0
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def clean_line(line, threshold, min_component_area):
    binary = (line < threshold).astype(np.uint8)
    if min_component_area > 1 and binary.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = np.zeros_like(binary, dtype=bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                keep |= labels == label
        binary = keep.astype(np.uint8)
    return np.where(binary > 0, 0, 255).astype(np.uint8)


def extract(rows, manifest, args):
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
            line, line_clamped, line_origin = centered_crop(
                line_page, line_center_x, line_center_y, args.source_size, args.output_size,
            )
            rough, rough_clamped, rough_origin = centered_crop(
                rough_page, rough_center_x, rough_center_y, args.source_size, args.output_size,
            )
            if line is None or rough is None:
                continue
            line_clean = clean_line(line, args.line_threshold, args.min_component_area)
            rough_edge = edge_map(rough)
            line_edge = edge_map(line_clean)
            name = (
                f'kurips{args.source_size}_{row["page"]}_'
                f'l{line_origin[0]:04d}_{line_origin[1]:04d}_'
                f'r{rough_origin[0]:04d}_{rough_origin[1]:04d}.jpg'
            )
            out.append({
                **row,
                "name": name,
                "source_size": args.source_size,
                "output_size": args.output_size,
                "scale_factor": args.source_size / args.output_size,
                "line_crop_x": line_origin[0],
                "line_crop_y": line_origin[1],
                "rough_crop_x": rough_origin[0],
                "rough_crop_y": rough_origin[1],
                "line_clamped": int(line_clamped),
                "rough_clamped": int(rough_clamped),
                "line_ink": float((line_clean < 128).mean()),
                "rough_std": float(rough.std()),
                "edge_f1": support_f1(rough_edge, line_edge, args.close_px),
                "chamfer": chamfer(rough_edge, line_edge, args.truncate_px),
                "rough": rough,
                "line": line_clean,
                "rough_edge": rough_edge,
                "line_edge": line_edge,
            })
            if index % 50 == 0:
                print(f"{index}/{len(rows)} saved_rows={len(out)}", flush=True)
    return out


def write_csv(rows, path):
    fields = [
        "rank", "name", "page", "source_page", "source_size", "output_size",
        "scale_factor", "line_x", "line_y", "rough_x", "rough_y",
        "line_crop_x", "line_crop_y", "rough_crop_x", "rough_crop_y",
        "line_clamped", "rough_clamped", "line_ink", "rough_std",
        "edge_f1", "chamfer", "vlm_score", "vlm_reason",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})


def save_tiles(rows, args):
    Path(args.rough_out).mkdir(parents=True, exist_ok=True)
    Path(args.line_out).mkdir(parents=True, exist_ok=True)
    for row in rows:
        Image.fromarray(row["rough"]).save(Path(args.rough_out) / row["name"], quality=95)
        Image.fromarray(row["line"]).save(Path(args.line_out) / row["name"], quality=95)
    Path(args.list_out).write_text("\n".join(row["name"] for row in rows) + "\n")


def make_qc(rows, path, count):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 220, 32
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        top = index * (thumb + label_h)
        overlay = np.full((args_output_size(row), args_output_size(row), 3), 255, np.uint8)
        overlay[row["rough_edge"]] = (255, 60, 60)
        overlay[row["line_edge"]] = (40, 80, 255)
        for column, image in enumerate((row["rough"], row["line"], overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, top))
        text = (
            f'{row["name"]} s={row["source_size"]} '
            f'F1={row["edge_f1"]:.2f} ch={row["chamfer"]:.1f} ink={row["line_ink"]:.3f}'
        )
        draw.text((3, top + thumb + 2), text, fill=0, font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def args_output_size(row):
    return int(row["output_size"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--coords-csv", default=DEFAULT_COORDS)
    parser.add_argument("--source-size", type=int, default=960)
    parser.add_argument("--anchor-tile", type=int, default=ANCHOR_TILE)
    parser.add_argument("--output-size", type=int, default=OUTPUT_SIZE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--autocontrast-rough", action="store_true")
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=3)
    parser.add_argument("--truncate-px", type=int, default=30)
    parser.add_argument("--rough-out", default=DEFAULT_ROUGH_OUT)
    parser.add_argument("--line-out", default=DEFAULT_LINE_OUT)
    parser.add_argument("--list-out", default=DEFAULT_LIST_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--qc-out", default=DEFAULT_QC_OUT)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rows = read_coords(args.coords_csv, args.limit)
    manifest = load_manifest(args.zip_path, args.zip_root)
    extracted = extract(rows, manifest, args)
    write_csv(extracted, args.csv_out)
    make_qc(extracted, args.qc_out, args.qc_count)
    if args.save:
        save_tiles(extracted, args)
    print(f"input_rows={len(rows)} extracted={len(extracted)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}")
    if args.save:
        print(f"saved: {args.list_out}, {args.rough_out}, {args.line_out}")


if __name__ == "__main__":
    main()
