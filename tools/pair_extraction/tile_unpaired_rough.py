"""Tile a directory of full-page rough (no paired line art) images into
480x480 tiles for the unpaired-rough pool (see memory
project_unpaired_data_pools.md). No alignment/pairing is needed since there
is no line-art counterpart -- just a grid split plus a blank-tile filter,
following the same autocontrast+std gate as prepare_ako5.py's rough side.

Usage:
    ./venv/bin/python tools/pair_extraction/tile_unpaired_rough.py \
        --source-name skima \
        --input-dir dataset/unpaired_rough/skima/cleaned \
        --output-dir dataset/pairs_480/train/rough_unpaired_skima \
        --file-list dataset/pairs_480/valid_train_unpaired_skima.txt
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageOps

TILE_SIZE = 480
STD_THRESH = 15


def tile_image(img, tile_size=480):
    w, h = img.size
    n_cols = w // tile_size
    n_rows = h // tile_size
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tile_size
            y = row * tile_size
            tiles.append((row, col, img.crop((x, y, x + tile_size, y + tile_size))))
    return tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-name", required=True, help="prefix for output filenames, e.g. skima")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--std-thresh", type=float, default=STD_THRESH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    existing = set()
    if os.path.exists(args.file_list):
        with open(args.file_list) as f:
            existing = {line.strip() for line in f if line.strip()}
    print(f"existing {args.file_list}: {len(existing)}")

    pages = sorted(
        name for name in os.listdir(args.input_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"pages: {len(pages)}")

    new_entries = []
    stats = {"pages": 0, "tiles_total": 0, "tiles_pass": 0}

    for page_name in pages:
        prefix = f"{args.source_name}_{os.path.splitext(page_name)[0]}"
        existing_for_page = [e for e in existing if e.startswith(prefix + "_")]
        if existing_for_page:
            continue

        rough_img = Image.open(os.path.join(args.input_dir, page_name)).convert("L")
        tiles = tile_image(rough_img, args.tile_size)
        stats["pages"] += 1
        stats["tiles_total"] += len(tiles)

        page_pass = 0
        for row, col, tile in tiles:
            rough_ac = ImageOps.autocontrast(tile, cutoff=0)
            std = float(np.array(rough_ac).std())
            if std < args.std_thresh:
                continue
            fname = f"{prefix}_{row:02d}_{col:02d}.jpg"
            tile.save(os.path.join(args.output_dir, fname), quality=95)
            new_entries.append(fname)
            page_pass += 1

        stats["tiles_pass"] += page_pass

    if new_entries:
        with open(args.file_list, "a") as f:
            for fname in sorted(new_entries):
                f.write(fname + "\n")

    total_now = len(existing) + len(new_entries)
    print("=== done ===")
    print(f"pages processed : {stats['pages']}")
    print(f"tiles total     : {stats['tiles_total']}")
    print(f"tiles passed    : {stats['tiles_pass']} (std>={args.std_thresh})")
    print(f"{args.file_list}: {len(existing)} -> {total_now}")


if __name__ == "__main__":
    main()
