"""Extract 480px training pairs from page-aligned sketch/line images."""

import argparse
import csv
import io
import json
import math
import os
import zipfile
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ZIP_PATH = os.path.expanduser("~/dataset_kurip_v4.zip")
ZIP_ROOT = "dataset_kurip_v4"
ROUGH_OUT = "dataset/pairs_480/train/rough"
LINE_OUT = "dataset/pairs_480/train/line"
LIST_OUT = "dataset/pairs_480/valid_train_kurip.txt"
CSV_OUT = "results/kurip_tiles.csv"
QC_OUT = "results/kurip_tiles_qc.png"
QC_TAIL_OUT = "results/kurip_tiles_qc_tail.png"
QC_SAMPLE_OUT = "results/kurip_tiles_qc_sample.png"
TILE = 480


def read_zip_member(zf, root, name):
    candidates = [
        f"{root}/{name}",
        f"{root}\\{name}",
        name,
    ]
    for candidate in candidates:
        try:
            return zf.read(candidate)
        except KeyError:
            continue
    raise KeyError(f"missing zip member for {name!r}; tried {candidates}")


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(read_zip_member(zf, zip_root, "manifest.json"))


def page_id(entry):
    return os.path.splitext(entry["file"])[0].replace("page", "")


def load_page_pair(zf, entry, zip_root):
    rough_raw = read_zip_member(zf, zip_root, entry["sketch"])
    line_raw = read_zip_member(zf, zip_root, entry["line"])
    rough = Image.open(io.BytesIO(rough_raw)).convert("L")
    line = Image.open(io.BytesIO(line_raw)).convert("L")
    if rough.size != line.size:
        raise ValueError(f"size mismatch: {entry['file']} rough={rough.size} line={line.size}")
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


def tile_score(stats):
    return (
        5.0 * stats["edge_f1"]
        - 0.20 * stats["chamfer"]
        + 0.8 * stats["orientation_entropy"]
    )


def collect_page_tiles(entry, rough, line, args):
    height, width = line.shape
    rough_edges_full = edge_map(rough)
    line_edges_full = edge_map(line)
    d_to_rough = cv2.distanceTransform(
        (~rough_edges_full).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )
    d_to_line = cv2.distanceTransform(
        (~line_edges_full).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )

    rows = []
    pid = page_id(entry)
    for y in window_starts(height, args.tile, args.stride):
        for x in window_starts(width, args.tile, args.stride):
            rough_tile = rough[y:y + args.tile, x:x + args.tile]
            line_tile = line[y:y + args.tile, x:x + args.tile]
            line_ink = float((line_tile < 128).mean())
            rough_std = float(rough_tile.std())
            if rough_std < args.min_rough_std or not args.ink_min <= line_ink <= args.ink_max:
                continue

            rough_edge = rough_edges_full[y:y + args.tile, x:x + args.tile]
            line_edge = line_edges_full[y:y + args.tile, x:x + args.tile]
            rough_edges = int(rough_edge.sum())
            line_edges = int(line_edge.sum())
            if rough_edges < args.min_edge_pixels or line_edges < args.min_edge_pixels:
                continue

            entropy = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
            if entropy < args.min_entropy:
                continue

            rough_dist = d_to_line[y:y + args.tile, x:x + args.tile][rough_edge]
            line_dist = d_to_rough[y:y + args.tile, x:x + args.tile][line_edge]
            precision = float((rough_dist <= args.close_px).mean())
            recall = float((line_dist <= args.close_px).mean())
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
            chamfer = float(
                (
                    np.minimum(rough_dist, args.truncate_px).mean()
                    + np.minimum(line_dist, args.truncate_px).mean()
                )
                / 2.0
            )
            row = {
                "source_page": entry["file"],
                "page": pid,
                "x": x,
                "y": y,
                "tile_bbox": (x, y, x + args.tile, y + args.tile),
                "edge_f1": f1,
                "edge_precision": precision,
                "edge_recall": recall,
                "chamfer": chamfer,
                "orientation_entropy": entropy,
                "rough_std": rough_std,
                "line_ink": line_ink,
                "rough_edges": rough_edges,
                "line_edges": line_edges,
            }
            row["tile_score"] = tile_score(row)
            if (
                row["tile_score"] >= args.min_tile_score
                and row["edge_f1"] >= args.min_f1
                and row["chamfer"] <= args.max_chamfer
            ):
                rows.append(row)
    return rows


def overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / float(TILE * TILE)


def deduplicate(rows, overlap, max_per_page):
    kept = []
    per_page = {}
    for row in sorted(rows, key=lambda item: item["tile_score"], reverse=True):
        page = row["page"]
        if max_per_page and per_page.get(page, 0) >= max_per_page:
            continue
        duplicate = any(
            page == other["page"] and overlap_ratio(row["tile_bbox"], other["tile_bbox"]) >= overlap
            for other in kept
        )
        if duplicate:
            continue
        kept.append(row)
        per_page[page] = per_page.get(page, 0) + 1
    return kept


def evenly_spaced(rows, count):
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


class PageCache:
    def __init__(self, zip_path, zip_root, manifest, max_items=4):
        self.zip_path = zip_path
        self.zip_root = zip_root
        self.entries = {page_id(entry): entry for entry in manifest}
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, page):
        if page in self.cache:
            self.cache.move_to_end(page)
            return self.cache[page]
        with zipfile.ZipFile(self.zip_path) as zf:
            pair = load_page_pair(zf, self.entries[page], self.zip_root)
        self.cache[page] = pair
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return pair


def make_qc(rows, path, count, cache):
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
        rough_page, line_page = cache.get(row["page"])
        x, y = row["x"], row["y"]
        rough = rough_page[y:y + TILE, x:x + TILE]
        line = line_page[y:y + TILE, x:x + TILE]
        overlay = np.full((TILE, TILE, 3), 255, np.uint8)
        overlay[edge_map(rough)] = (255, 60, 60)
        overlay[edge_map(line)] = (40, 80, 255)
        top = index * (thumb + label_h)
        for column, image in enumerate((rough, line, overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, top))
        text = (
            f'{row["name"]} score={row["tile_score"]:.2f} F1={row["edge_f1"]:.2f} '
            f'cham={row["chamfer"]:.1f} ink={row["line_ink"]:.2f} '
            f'ent={row["orientation_entropy"]:.2f}'
        )
        draw.text((3, top + thumb + 2), text, fill="black", font=font)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)


def save_tiles(rows, cache, rough_out, line_out, list_out):
    os.makedirs(rough_out, exist_ok=True)
    os.makedirs(line_out, exist_ok=True)
    rows_by_page = {}
    for row in rows:
        rows_by_page.setdefault(row["page"], []).append(row)
    for page in sorted(rows_by_page):
        rough_page, line_page = cache.get(page)
        for row in rows_by_page[page]:
            x, y = row["x"], row["y"]
            rough = rough_page[y:y + TILE, x:x + TILE]
            line = line_page[y:y + TILE, x:x + TILE]
            Image.fromarray(rough).save(os.path.join(rough_out, row["name"]), quality=95)
            Image.fromarray(line).save(os.path.join(line_out, row["name"]), quality=95)
    with open(list_out, "w") as file:
        file.write("\n".join(row["name"] for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--name-prefix", default="kurip")
    parser.add_argument("--exclude-page", action="append", default=[])
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--stride", type=int, default=240)
    parser.add_argument("--min-tile-score", type=float, default=2.0)
    parser.add_argument("--min-rough-std", type=float, default=10.0)
    parser.add_argument("--ink-min", type=float, default=0.01)
    parser.add_argument("--ink-max", type=float, default=0.16)
    parser.add_argument("--min-edge-pixels", type=int, default=250)
    parser.add_argument("--min-f1", type=float, default=0.25)
    parser.add_argument("--max-chamfer", type=float, default=12.0)
    parser.add_argument("--min-entropy", type=float, default=0.45)
    parser.add_argument("--close-px", type=float, default=6.0)
    parser.add_argument("--truncate-px", type=float, default=24.0)
    parser.add_argument("--duplicate-overlap", type=float, default=0.60)
    parser.add_argument("--max-per-page", type=int, default=80)
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

    manifest = load_manifest(args.zip_path, args.zip_root)
    candidates = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for index, entry in enumerate(manifest, 1):
            if page_id(entry) in set(args.exclude_page):
                print(f"pages: {index}/{len(manifest)} skipped={page_id(entry)}", flush=True)
                continue
            rough, line = load_page_pair(zf, entry, args.zip_root)
            rows = collect_page_tiles(entry, rough, line, args)
            candidates.extend(rows)
            print(
                f"pages: {index}/{len(manifest)} page_tiles={len(rows)} "
                f"candidates={len(candidates)}",
                flush=True,
            )

    accepted = deduplicate(candidates, args.duplicate_overlap, args.max_per_page)
    accepted.sort(key=lambda item: item["tile_score"], reverse=True)
    for rank, row in enumerate(accepted, 1):
        row["rank"] = rank
        row["name"] = f'{args.name_prefix}_{row["page"]}_{row["x"]:04d}_{row["y"]:04d}.jpg'

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "rank", "name", "source_page", "page", "x", "y", "tile_score",
        "edge_f1", "edge_precision", "edge_recall", "chamfer",
        "orientation_entropy", "rough_std", "line_ink", "rough_edges",
        "line_edges", "tile_bbox", "decision", "notes",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in accepted:
            writer.writerow({**row, "decision": "", "notes": ""})

    cache = PageCache(args.zip_path, args.zip_root, manifest)
    make_qc(accepted, args.qc_out, args.qc_count, cache)
    make_qc(accepted[-args.qc_count:], args.qc_tail_out, args.qc_count, cache)
    make_qc(evenly_spaced(accepted, args.qc_count), args.qc_sample_out, args.qc_count, cache)
    if args.save:
        save_tiles(accepted, cache, args.rough_out, args.line_out, args.list_out)

    action = "saved" if args.save else "dry-run"
    print(f"{action}: pages={len(manifest)} raw_tiles={len(candidates)} accepted={len(accepted)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}, {args.qc_tail_out}, {args.qc_sample_out}")
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
