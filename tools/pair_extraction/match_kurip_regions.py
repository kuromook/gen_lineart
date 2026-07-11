"""Find locally aligned kurip rough/line regions before tile extraction.

This replaces same-coordinate extraction for kurip. For each line tile, it
searches a nearby rough window and records only candidates whose aligned rough
edges overlap the line edges well enough for visual review.
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


ZIP_PATH = os.path.expanduser("~/dataset_kurip_v4.zip")
ZIP_ROOT = "dataset_kurip_v4"
CSV_OUT = "results/kurip_region_matches.csv"
JSON_OUT = "results/kurip_region_matches.json"
QC_OUT = "results/kurip_region_match_qc.png"
TILE = 480


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"{zip_root}/manifest.json"))


def page_id(entry):
    return Path(entry.get("file", entry["sketch"])).stem.replace("page", "")


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['line']}"))).convert("L")
    rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def window_starts(limit, tile, stride):
    if limit < tile:
        return []
    starts = list(range(0, limit - tile + 1, stride))
    last = limit - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def support_f1(rough_edge, line_edge, tolerance=4):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def chamfer(rough_edge, line_edge, truncate=30):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 99.0
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def orientation_entropy(edges):
    if edges.sum() < 40:
        return 0.0
    image = edges.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    valid = mag > 0.25
    if valid.sum() < 40:
        return 0.0
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=mag[valid])
    probs = hist / max(hist.sum(), 1e-9)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / np.log(12))


def candidate_offsets(search, step):
    offsets = []
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return offsets


def refine_offsets(dx, dy, step):
    fine = max(step // 2, 1)
    offsets = []
    for yy in range(dy - step, dy + step + 1, fine):
        for xx in range(dx - step, dx + step + 1, fine):
            offsets.append((xx, yy))
    offsets.sort(key=lambda item: (item[0] - dx) ** 2 + (item[1] - dy) ** 2)
    return offsets


def extract_tile(image, x, y, tile):
    h, w = image.shape
    if x < 0 or y < 0 or x + tile > w or y + tile > h:
        return None
    return image[y:y + tile, x:x + tile]


def match_tile(rough, rough_edges_full, line_tile, line_edge, x, y, args):
    best = None
    for dx, dy in candidate_offsets(args.search, args.search_step):
        rough_tile = extract_tile(rough, x + dx, y + dy, args.tile)
        if rough_tile is None:
            continue
        rough_edge = extract_tile(rough_edges_full, x + dx, y + dy, args.tile)
        score = support_f1(rough_edge, line_edge, args.close_px)
        if best is None or score > best["coarse_f1"]:
            best = {"dx": dx, "dy": dy, "coarse_f1": score}
    if best is None:
        return None

    refined = None
    for dx, dy in refine_offsets(best["dx"], best["dy"], args.search_step):
        rough_tile = extract_tile(rough, x + dx, y + dy, args.tile)
        if rough_tile is None:
            continue
        rough_edge = extract_tile(rough_edges_full, x + dx, y + dy, args.tile)
        f1 = support_f1(rough_edge, line_edge, args.close_px)
        if refined is None or f1 > refined["edge_f1"]:
            c = chamfer(rough_edge, line_edge, args.truncate_px)
            ent = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
            refined = {
                "dx": dx,
                "dy": dy,
                "edge_f1": f1,
                "chamfer": c,
                "orientation_entropy": ent,
                "rough_edges": int(rough_edge.sum()),
                "rough_std": float(rough_tile.std()),
                "rough_tile": rough_tile,
            }
    if refined is None:
        return None
    score = (
        5.0 * refined["edge_f1"]
        - 0.20 * refined["chamfer"]
        + 0.8 * refined["orientation_entropy"]
    )
    return {**refined, "match_score": score, "line_tile": line_tile}


def process_page(entry, rough, line, args):
    height = min(rough.shape[0], line.shape[0])
    width = min(rough.shape[1], line.shape[1])
    rough = rough[:height, :width]
    line = line[:height, :width]
    rough_edges_full = edge_map(rough)
    line_edges_full = edge_map(line)
    rows = []
    page = page_id(entry)
    for y in window_starts(height, args.tile, args.stride):
        for x in window_starts(width, args.tile, args.stride):
            line_tile = line[y:y + args.tile, x:x + args.tile]
            line_ink = float((line_tile < 128).mean())
            if not args.ink_min <= line_ink <= args.ink_max:
                continue
            line_edge = line_edges_full[y:y + args.tile, x:x + args.tile]
            if line_edge.sum() < args.min_edge_pixels:
                continue
            match = match_tile(rough, rough_edges_full, line_tile, line_edge, x, y, args)
            if match is None:
                continue
            if (
                match["rough_std"] < args.min_rough_std
                or match["rough_edges"] < args.min_edge_pixels
                or match["edge_f1"] < args.min_f1
                or match["chamfer"] > args.max_chamfer
            ):
                continue
            row = {
                "page": page,
                "source_page": entry.get("file", ""),
                "line_x": x,
                "line_y": y,
                "rough_x": x + match["dx"],
                "rough_y": y + match["dy"],
                "dx": match["dx"],
                "dy": match["dy"],
                "line_ink": line_ink,
                "line_edges": int(line_edge.sum()),
                "rough_edges": match["rough_edges"],
                "rough_std": match["rough_std"],
                "edge_f1": match["edge_f1"],
                "chamfer": match["chamfer"],
                "orientation_entropy": match["orientation_entropy"],
                "match_score": match["match_score"],
                "rough_tile": match["rough_tile"],
                "line_tile": line_tile,
            }
            rows.append(row)
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    return rows[: args.max_per_page] if args.max_per_page else rows


def serializable(row):
    return {k: v for k, v in row.items() if k not in {"rough_tile", "line_tile"}}


def write_outputs(rows, args):
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "rank", "page", "source_page", "line_x", "line_y", "rough_x", "rough_y",
        "dx", "dy", "line_ink", "line_edges", "rough_edges", "rough_std",
        "edge_f1", "chamfer", "orientation_entropy", "match_score",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **serializable(row)})
    with open(args.json_out, "w") as file:
        json.dump([{"rank": rank, **serializable(row)} for rank, row in enumerate(rows, 1)], file, indent=2)
        file.write("\n")


def make_qc(rows, args):
    picks = rows[: args.qc_count]
    if not picks:
        return
    thumb, label_h = 220, 30
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        y = index * (thumb + label_h)
        rough = row["rough_tile"]
        line = row["line_tile"]
        overlay = np.full((args.tile, args.tile, 3), 255, np.uint8)
        overlay[edge_map(rough)] = (255, 60, 60)
        overlay[edge_map(line)] = (40, 80, 255)
        for col, image in enumerate((rough, line, overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (col * thumb, y))
        text = (
            f'{row["page"]} line=({row["line_x"]},{row["line_y"]}) '
            f'd=({row["dx"]},{row["dy"]}) F1={row["edge_f1"]:.2f} '
            f'ch={row["chamfer"]:.1f} score={row["match_score"]:.2f}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    os.makedirs(os.path.dirname(args.qc_out) or ".", exist_ok=True)
    canvas.save(args.qc_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--stride", type=int, default=480)
    parser.add_argument("--search", type=int, default=128)
    parser.add_argument("--search-step", type=int, default=16)
    parser.add_argument("--close-px", type=int, default=4)
    parser.add_argument("--truncate-px", type=int, default=30)
    parser.add_argument("--min-rough-std", type=float, default=10.0)
    parser.add_argument("--ink-min", type=float, default=0.01)
    parser.add_argument("--ink-max", type=float, default=0.16)
    parser.add_argument("--min-edge-pixels", type=int, default=250)
    parser.add_argument("--min-f1", type=float, default=0.25)
    parser.add_argument("--max-chamfer", type=float, default=20.0)
    parser.add_argument("--max-per-page", type=int, default=20)
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--qc-count", type=int, default=40)
    args = parser.parse_args()

    manifest = load_manifest(args.zip_path, args.zip_root)
    entries = manifest[: args.pages] if args.pages else manifest
    rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for index, entry in enumerate(entries, 1):
            rough, line = load_pair(zf, args.zip_root, entry)
            page_rows = process_page(entry, rough, line, args)
            rows.extend(page_rows)
            print(f"{index}/{len(entries)} {page_id(entry)} matches={len(page_rows)}", flush=True)
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    write_outputs(rows, args)
    make_qc(rows, args)
    print(f"matches={len(rows)}")
    print(f"saved: {args.csv_out}, {args.json_out}, {args.qc_out}")


if __name__ == "__main__":
    main()
