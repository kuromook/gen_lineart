"""Build a full-resolution rough/line review sheet from a tile CSV.

The QC montages written by `tools/pair_extraction/tile_region_manifest_480.py`
shrink each tile to a thumbnail, which hides stroke quality. This tool pastes
tiles at their native tile size so stroke width, gray fringe, and rough-to-line
correspondence stay judgeable.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


LABEL_KEYS = (
    "tile_score",
    "strict_edge_f1",
    "strict_edge_recall",
    "strict_edge_precision",
    "line_width_p50",
    "soft_ink_ratio",
    "long_line_ratio",
    "line_ink",
    "support",
    "src_per_out",
)


def resolve_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return manifest_path.parent / path


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def pick_indices(count, picks, mode):
    if count <= picks:
        return list(range(count))
    if mode == "top":
        return list(range(picks))
    if mode == "tail":
        return list(range(count - picks, count))
    return list(np.linspace(0, count - 1, picks, dtype=int))


def label_for(row):
    parts = [f'rank{row.get("rank", "?")}']
    for key in LABEL_KEYS:
        if row.get(key):
            try:
                parts.append(f"{key.replace('_', '')[:9]}={float(row[key]):.3g}")
            except ValueError:
                pass
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rough-key", default="masked_rough_path")
    parser.add_argument("--line-key", default="masked_line_path")
    parser.add_argument("--tile", type=int, default=480)
    parser.add_argument("--picks", type=int, default=10)
    parser.add_argument("--mode", choices=("spread", "top", "tail"), default="spread")
    parser.add_argument("--autocontrast-rough", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tiles = read_rows(args.tile_csv)
    if not tiles:
        raise SystemExit("tile CSV is empty")
    manifest_path = Path(args.manifest)
    manifest = read_rows(manifest_path)
    by_review_index = {}
    for row_index, row in enumerate(manifest, 1):
        key = row.get("_review_index") or str(row_index)
        by_review_index[key] = row

    picks = [tiles[index] for index in pick_indices(len(tiles), args.picks, args.mode)]
    tile = args.tile
    label_h = 26
    canvas = Image.new("RGB", (tile * 2, (tile + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    cache = {}
    for index, row in enumerate(picks):
        source = by_review_index[row["source_review_index"]]
        key = source[args.rough_key]
        if key not in cache:
            cache.clear()
            rough = Image.open(resolve_path(source[args.rough_key], manifest_path)).convert("L")
            if args.autocontrast_rough:
                rough = ImageOps.autocontrast(rough, cutoff=0)
            line = Image.open(resolve_path(source[args.line_key], manifest_path)).convert("L")
            cache[key] = (np.asarray(rough), np.asarray(line))
        rough_page, line_page = cache[key]
        x, y = int(row["x"]), int(row["y"])
        top = index * (tile + label_h)
        canvas.paste(Image.fromarray(rough_page[y : y + tile, x : x + tile]).convert("RGB"), (0, top))
        canvas.paste(Image.fromarray(line_page[y : y + tile, x : x + tile]).convert("RGB"), (tile, top))
        draw.text((4, top + tile + 4), label_for(row), fill="black", font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote: {args.out} tiles={len(picks)} of {len(tiles)} mode={args.mode}")


if __name__ == "__main__":
    main()
