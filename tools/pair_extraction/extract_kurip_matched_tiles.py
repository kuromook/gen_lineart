"""Extract kurip tiles from locally matched rough/line coordinates.

Dry-run is the default. Use --save after checking the QC montage.
"""

import argparse
import csv
import io
import json
import os
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ZIP_PATH = os.path.expanduser("~/dataset_kurip_v4.zip")
ZIP_ROOT = "dataset_kurip_v4"
MATCHES = "results/kurip_region_matches_strict.csv"
ROUGH_OUT = "dataset/pairs_480/train/rough"
LINE_OUT = "dataset/pairs_480/train/line_kurip_matched_clean_t192_cc8"
LIST_OUT = "dataset/pairs_480/valid_train_kurip_matched_strict.txt"
CSV_OUT = "results/kurip_matched_tiles.csv"
QC_OUT = "results/kurip_matched_tiles_qc.png"
TILE = 480


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"{zip_root}/manifest.json"))


def page_id(entry):
    return Path(entry.get("file", entry["sketch"])).stem.replace("page", "")


def load_pair(zf, zip_root, entry, autocontrast_rough):
    rough = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['line']}"))).convert("L")
    if autocontrast_rough:
        rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def read_matches(path, args):
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            row = {
                **row,
                "rank": int(row["rank"]),
                "line_x": int(row["line_x"]),
                "line_y": int(row["line_y"]),
                "rough_x": int(row["rough_x"]),
                "rough_y": int(row["rough_y"]),
                "dx": int(row["dx"]),
                "dy": int(row["dy"]),
                "edge_f1": float(row["edge_f1"]),
                "chamfer": float(row["chamfer"]),
                "match_score": float(row["match_score"]),
                "line_ink": float(row["line_ink"]),
            }
            if (
                row["edge_f1"] >= args.min_f1
                and row["chamfer"] <= args.max_chamfer
                and row["match_score"] >= args.min_match_score
            ):
                rows.append(row)
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    return rows[: args.limit] if args.limit else rows


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


def crop(image, x, y, tile):
    height, width = image.shape
    if x < 0 or y < 0 or x + tile > width or y + tile > height:
        return None
    return image[y:y + tile, x:x + tile]


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


class PageCache:
    def __init__(self, zip_path, zip_root, manifest, autocontrast_rough, max_items=64):
        self.zip_path = zip_path
        self.zip_root = zip_root
        self.entries = {page_id(entry): entry for entry in manifest}
        self.autocontrast_rough = autocontrast_rough
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, page):
        if page in self.cache:
            self.cache.move_to_end(page)
            return self.cache[page]
        with zipfile.ZipFile(self.zip_path) as zf:
            pair = load_pair(zf, self.zip_root, self.entries[page], self.autocontrast_rough)
        self.cache[page] = pair
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return pair


def materialize(rows, cache, args):
    out = []
    for row in rows:
        rough_page, line_page = cache.get(row["page"])
        rough = crop(rough_page, row["rough_x"], row["rough_y"], args.tile)
        line = crop(line_page, row["line_x"], row["line_y"], args.tile)
        if rough is None or line is None:
            continue
        if args.clean_line:
            line = clean_line(line, args.line_threshold, args.min_component_area)
        name = (
            f'kuripm_{row["page"]}_l{row["line_x"]:04d}_{row["line_y"]:04d}_'
            f'r{row["rough_x"]:04d}_{row["rough_y"]:04d}.jpg'
        )
        out.append({**row, "name": name, "rough": rough, "line": line})
    return out


def make_qc(rows, path, count, tile):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 240, 30
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        top = index * (thumb + label_h)
        overlay = np.full((tile, tile, 3), 255, np.uint8)
        overlay[edge_map(row["rough"])] = (255, 60, 60)
        overlay[edge_map(row["line"])] = (40, 80, 255)
        for column, image in enumerate((row["rough"], row["line"], overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, top))
        text = (
            f'{row["name"]} F1={row["edge_f1"]:.2f} ch={row["chamfer"]:.1f} '
            f'score={row["match_score"]:.2f} d=({row["dx"]},{row["dy"]})'
        )
        draw.text((3, top + thumb + 2), text, fill="black", font=font)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "rank", "name", "page", "source_page", "line_x", "line_y", "rough_x", "rough_y",
        "dx", "dy", "line_ink", "edge_f1", "chamfer", "match_score", "decision", "notes",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({**row, "rank": index, "decision": "", "notes": ""})


def save_tiles(rows, rough_out, line_out, list_out):
    os.makedirs(rough_out, exist_ok=True)
    os.makedirs(line_out, exist_ok=True)
    for row in rows:
        Image.fromarray(row["rough"]).save(os.path.join(rough_out, row["name"]), quality=95)
        Image.fromarray(row["line"]).save(os.path.join(line_out, row["name"]), quality=95)
    with open(list_out, "w") as file:
        file.write("\n".join(row["name"] for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--matches", default=MATCHES)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--min-f1", type=float, default=0.48)
    parser.add_argument("--max-chamfer", type=float, default=10.5)
    parser.add_argument("--min-match-score", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--clean-line", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autocontrast-rough", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--qc-count", type=int, default=60)
    parser.add_argument("--rough-out", default=ROUGH_OUT)
    parser.add_argument("--line-out", default=LINE_OUT)
    parser.add_argument("--list-out", default=LIST_OUT)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.tile != TILE:
        raise ValueError("Only 480px output is currently supported")
    manifest = load_manifest(args.zip_path, args.zip_root)
    cache = PageCache(args.zip_path, args.zip_root, manifest, args.autocontrast_rough)
    rows = materialize(read_matches(args.matches, args), cache, args)
    write_csv(rows, args.csv_out)
    make_qc(rows, args.qc_out, args.qc_count, args.tile)
    if args.save:
        save_tiles(rows, args.rough_out, args.line_out, args.list_out)

    action = "saved" if args.save else "dry-run"
    print(f"{action}: matched_tiles={len(rows)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}")
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
