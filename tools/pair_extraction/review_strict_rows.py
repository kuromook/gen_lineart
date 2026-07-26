"""Render selected rows of a filter_matched_region_tiles.py strict CSV at full
native resolution for manual review (rough | line side by side, one row per
pair). Works for any source using the `match_kurip_regions.py` /
`filter_matched_region_tiles.py` route (name predates use beyond its original
source).

The QC contact sheets produced by filter_matched_region_tiles.py downscale
tiles into a grid, which can hide small-scale content mismatches. This script
re-crops the exact rough/line tiles named by --ranks (1-based rank in the
strict CSV, matching its default sort order: descending tile_score) directly
from the source zip at full 480x480 resolution, stacked vertically so nothing
is downscaled.

Usage:
  venv/bin/python tools/pair_extraction/review_strict_rows.py \
    --csv results/fitness_native_tiles_480_strict.csv \
    --zip dataset/raw_zips/dataset_fitness_v4.zip \
    --zip-root dataset_fitness_v4 \
    --ranks 1 5 10 ... \
    --out /path/to/review_sheet.png
"""

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

TILE = 480


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def load_manifest(zip_path, zip_root):
    import json

    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read(f"{zip_root}/manifest.json"))


def page_lookup(manifest):
    return {Path(entry.get("file", entry["sketch"])).stem.replace("page", ""): entry for entry in manifest}


def load_page_pair(archive, zip_root, entry):
    rough = Image.open(io.BytesIO(archive.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(archive.read(f"{zip_root}/{entry['line']}"))).convert("L")
    return np.asarray(ImageOps.autocontrast(rough, cutoff=0)), np.asarray(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--zip-root", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    by_rank = {int(row["rank"]): row for row in rows}

    manifest = load_manifest(args.zip_path, args.zip_root)
    pages = page_lookup(manifest)

    label_h = 28
    gap = 8
    cell_w = TILE * 2 + gap
    cell_h = TILE + label_h

    canvas = Image.new("RGB", (cell_w, cell_h * len(args.ranks)), "white")
    draw = ImageDraw.Draw(canvas)

    with zipfile.ZipFile(args.zip_path) as archive:
        cache = {}
        for i, rank in enumerate(args.ranks):
            row = by_rank.get(rank)
            y0 = i * cell_h
            if row is None:
                draw.text((4, y0 + 4), f"rank {rank}: NOT FOUND", fill="red")
                continue
            page = row["page"]
            if page not in cache:
                cache[page] = load_page_pair(archive, args.zip_root, pages[page])
            rough_page, line_page = cache[page]
            lx, ly = int(row["line_x"]), int(row["line_y"])
            rx, ry = int(row["rough_x"]), int(row["rough_y"])
            rough_tile = rough_page[ry : ry + TILE, rx : rx + TILE]
            line_tile = line_page[ly : ly + TILE, lx : lx + TILE]
            rough_img = Image.fromarray(rough_tile).convert("RGB")
            line_img = Image.fromarray(line_tile).convert("RGB")
            canvas.paste(rough_img, (0, y0 + label_h))
            canvas.paste(line_img, (TILE + gap, y0 + label_h))
            score = row.get("tile_score", "?")
            draw.text(
                (4, y0 + 4),
                f"rank {rank}  page={page}  score={score}  line=({lx},{ly}) rough=({rx},{ry})",
                fill="black",
            )

    canvas.save(args.out)
    print(f"wrote {args.out} ({len(args.ranks)} rows)")


if __name__ == "__main__":
    main()
