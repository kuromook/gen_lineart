"""Extract high-confidence 480px training pairs from ako5 local region matches.

The input transforms map full-resolution sketch coordinates to line coordinates.
Each accepted local region is scanned with overlapping 480px windows. Every
window is independently checked before it can be written to the training set.

Dry-run is the default. Use --save only after reviewing the QC montage.
"""
import argparse
import csv
import io
import json
import math
import os
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
ZIP_PATH = os.path.expanduser("~/dataset_ako5.zip")
MATCHES = "results/ako5_region_matches.json"
CSV_OUT = "results/ako5_region_tiles.csv"
QC_OUT = "results/ako5_region_tiles_qc.png"
QC_TAIL_OUT = "results/ako5_region_tiles_qc_tail.png"
QC_SAMPLE_OUT = "results/ako5_region_tiles_qc_sample.png"
ROUGH_OUT = "dataset_480/train/rough"
LINE_OUT = "dataset_480/train/line"
LIST_OUT = "dataset_480/valid_train_ako5_regions.txt"
TILE = 480


def page_id(filename):
    return filename.split("_")[1]


def load_pages(zip_path, needed_sketches, needed_lines):
    sketches, lines = {}, {}
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("dataset_ako5/manifest.json"))
        sketch_files = {page_id(entry["sketch"]): entry["sketch"] for entry in manifest}
        line_files = {page_id(entry["line"]): entry["line"] for entry in manifest}
        for pid in sorted(needed_sketches):
            raw = zf.read(f"dataset_ako5/{sketch_files[pid]}")
            image = Image.open(io.BytesIO(raw)).convert("L")
            sketches[pid] = np.asarray(ImageOps.autocontrast(image, cutoff=0))
        for pid in sorted(needed_lines):
            raw = zf.read(f"dataset_ako5/{line_files[pid]}")
            lines[pid] = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
    return sketches, lines


def window_starts(lo, hi, limit, tile, stride):
    lo = max(0, int(math.ceil(lo)))
    hi = min(limit, int(math.floor(hi)))
    if hi - lo < tile:
        return []
    starts = list(range(lo, hi - tile + 1, stride))
    last = hi - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def warp_tile(sketch, matrix, x, y, tile):
    local = matrix.copy()
    local[:, 2] -= (x, y)
    rough = cv2.warpAffine(
        sketch, local, (tile, tile), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    height, width = sketch.shape
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    transformed = cv2.transform(corners[None, :, :], local)[0]
    support = np.zeros((tile, tile), np.uint8)
    cv2.fillConvexPoly(support, np.rint(transformed).astype(np.int32), 1)
    return rough, support.astype(bool)


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


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


def tile_statistics(rough, line, support, close_px, truncate_px, thresholds=None):
    stats = {
        "rough_std": float(rough.std()),
        "line_ink": float((line < 128).mean()),
        "support": float(support.mean()),
    }
    if thresholds is not None and (
        stats["rough_std"] < thresholds.min_rough_std
        or not thresholds.ink_min <= stats["line_ink"] <= thresholds.ink_max
        or stats["support"] < thresholds.min_support
    ):
        return None

    rough_edge = edge_map(rough) & support
    line_edge = edge_map(line) & support
    stats["rough_edges"] = int(rough_edge.sum())
    stats["line_edges"] = int(line_edge.sum())
    min_edges = thresholds.min_edge_pixels if thresholds is not None else 20
    if stats["rough_edges"] < min_edges or stats["line_edges"] < min_edges:
        return None

    stats["orientation_entropy"] = min(
        orientation_entropy(rough_edge), orientation_entropy(line_edge),
    )
    if thresholds is not None and stats["orientation_entropy"] < thresholds.min_entropy:
        return None

    d_to_rough = cv2.distanceTransform(
        (~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )
    d_to_line = cv2.distanceTransform(
        (~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )
    line_dist = d_to_rough[line_edge]
    rough_dist = d_to_line[rough_edge]
    precision = float((rough_dist <= close_px).mean())
    recall = float((line_dist <= close_px).mean())
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    chamfer = float((np.minimum(line_dist, truncate_px).mean()
                     + np.minimum(rough_dist, truncate_px).mean()) / 2)
    return {
        **stats,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "chamfer": chamfer,
    }


def tile_score(stats, region_score):
    return (
        5.0 * stats["edge_f1"]
        - 0.20 * stats["chamfer"]
        + 0.8 * stats["orientation_entropy"]
        + 0.15 * min(region_score, 6.0)
    )


def passes(row, args):
    return (
        row["region_score"] >= args.min_region_score
        and row["tile_score"] >= args.min_tile_score
        and row["rough_std"] >= args.min_rough_std
        and args.ink_min <= row["line_ink"] <= args.ink_max
        and row["support"] >= args.min_support
        and row["line_edges"] >= args.min_edge_pixels
        and row["rough_edges"] >= args.min_edge_pixels
        and row["edge_f1"] >= args.min_f1
        and row["chamfer"] <= args.max_chamfer
        and row["orientation_entropy"] >= args.min_entropy
    )


def overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / (TILE * TILE)


def deduplicate(rows, overlap):
    kept = []
    for row in sorted(rows, key=lambda item: item["tile_score"], reverse=True):
        duplicate = any(
            row["sketch_page"] == other["sketch_page"]
            and row["line_page"] == other["line_page"]
            and overlap_ratio(row["tile_bbox"], other["tile_bbox"]) >= overlap
            for other in kept
        )
        if not duplicate:
            kept.append(row)
    return kept


def make_qc(rows, path, count):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 240, 28
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        y = index * (thumb + label_h)
        overlay = np.full((TILE, TILE, 3), 255, np.uint8)
        overlay[edge_map(row["rough"])] = (255, 60, 60)
        overlay[edge_map(row["line"])] = (40, 80, 255)
        for column, image in enumerate((row["rough"], row["line"], overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, y))
        text = (
            f'{row["name"]} score={row["tile_score"]:.2f} F1={row["edge_f1"]:.2f} '
            f'cham={row["chamfer"]:.1f} ink={row["line_ink"]:.2f} '
            f'ent={row["orientation_entropy"]:.2f}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    canvas.save(path)


def evenly_spaced(rows, count):
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--matches", default=MATCHES)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--stride", type=int, default=240)
    parser.add_argument("--min-region-score", type=float, default=4.0)
    parser.add_argument("--min-tile-score", type=float, default=3.0)
    parser.add_argument("--min-rough-std", type=float, default=15.0)
    parser.add_argument("--ink-min", type=float, default=0.015)
    parser.add_argument("--ink-max", type=float, default=0.15)
    parser.add_argument("--min-support", type=float, default=0.98)
    parser.add_argument("--min-edge-pixels", type=int, default=300)
    parser.add_argument("--min-f1", type=float, default=0.35)
    parser.add_argument("--max-chamfer", type=float, default=8.0)
    parser.add_argument("--min-entropy", type=float, default=0.55)
    parser.add_argument("--close-px", type=float, default=6.0)
    parser.add_argument("--truncate-px", type=float, default=24.0)
    parser.add_argument("--duplicate-overlap", type=float, default=0.60)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--qc-tail-out", default=QC_TAIL_OUT)
    parser.add_argument("--qc-sample-out", default=QC_SAMPLE_OUT)
    parser.add_argument("--rough-out", default=ROUGH_OUT)
    parser.add_argument("--line-out", default=LINE_OUT)
    parser.add_argument("--list-out", default=LIST_OUT)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if args.tile != TILE:
        raise ValueError("Only 480px output is currently supported")

    matches = [
        row for row in json.load(open(args.matches))
        if row["score"] >= args.min_region_score
    ]
    sketches, lines = load_pages(
        args.zip_path,
        {row["sketch_page"] for row in matches},
        {row["line_page"] for row in matches},
    )

    candidates = []
    for region_index, row in enumerate(matches, 1):
        sid, lid = row["sketch_page"], row["line_page"]
        matrix = np.asarray(row["full_matrix"], dtype=np.float64)
        x0, y0, x1, y1 = row["bbox_full"]
        line_page = lines[lid]
        xs = window_starts(x0, x1, line_page.shape[1], TILE, args.stride)
        ys = window_starts(y0, y1, line_page.shape[0], TILE, args.stride)
        for y in ys:
            for x in xs:
                rough, support = warp_tile(sketches[sid], matrix, x, y, TILE)
                line = line_page[y:y + TILE, x:x + TILE]
                stats = tile_statistics(
                    rough, line, support, args.close_px, args.truncate_px, args,
                )
                if stats is None:
                    continue
                candidate = {
                    **stats,
                    "sketch_page": sid,
                    "line_page": lid,
                    "region_rank": row["rank"],
                    "region_score": row["score"],
                    "tile_bbox": (x, y, x + TILE, y + TILE),
                    "rough": rough,
                    "line": line,
                }
                candidate["tile_score"] = tile_score(candidate, row["score"])
                if passes(candidate, args):
                    candidates.append(candidate)
        if region_index % 20 == 0 or region_index == len(matches):
            print(
                f"regions: {region_index}/{len(matches)}, passing tiles: {len(candidates)}",
                flush=True,
            )

    accepted = deduplicate(candidates, args.duplicate_overlap)
    accepted.sort(key=lambda item: item["tile_score"], reverse=True)
    for index, row in enumerate(accepted, 1):
        x0, y0, _, _ = row["tile_bbox"]
        row["name"] = f'ako5r_{row["sketch_page"][4:]}_{row["line_page"][4:]}_{x0:04d}_{y0:04d}.jpg'
        row["rank"] = index

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "rank", "name", "sketch_page", "line_page", "region_rank", "region_score",
        "tile_score", "edge_f1", "edge_precision", "edge_recall", "chamfer",
        "orientation_entropy", "rough_std", "line_ink", "support",
        "rough_edges", "line_edges", "tile_bbox", "decision", "notes",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in accepted:
            writer.writerow({**row, "decision": "", "notes": ""})
    make_qc(accepted, args.qc_out, args.qc_count)
    make_qc(accepted[-args.qc_count:], args.qc_tail_out, args.qc_count)
    make_qc(evenly_spaced(accepted, args.qc_count), args.qc_sample_out, args.qc_count)

    if args.save:
        os.makedirs(args.rough_out, exist_ok=True)
        os.makedirs(args.line_out, exist_ok=True)
        for row in accepted:
            Image.fromarray(row["rough"]).save(os.path.join(args.rough_out, row["name"]), quality=95)
            Image.fromarray(row["line"]).save(os.path.join(args.line_out, row["name"]), quality=95)
        with open(args.list_out, "w") as file:
            file.write("\n".join(row["name"] for row in accepted) + "\n")

    action = "saved" if args.save else "dry-run"
    print(f"{action}: regions={len(matches)} raw_tiles={len(candidates)} accepted={len(accepted)}")
    print(
        f"wrote: {args.csv_out}, {args.qc_out}, {args.qc_tail_out}, "
        f"{args.qc_sample_out}"
    )
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
