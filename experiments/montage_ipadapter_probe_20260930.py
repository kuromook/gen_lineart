"""Contact sheet for the IP-Adapter probe: conditioning | each arm | GT.

Per the track's operating rules, numbers are never reported without a montage
whose exact path is quoted beside them. Tiles are picked by spread over the
baseline arm's f1 rather than cherry-picked, so the sheet shows the range the
means come from: worst, 25th, median, 75th, best.

Columns are labelled and every panel is drawn at GT polarity (black ink on
white paper), including the conditioning map, which is stored inverted.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

TRACK = Path(__file__).resolve().parents[1]
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
CELL = 240
LABEL_H = 22


def pick_tiles(per_tile_csv, n=5):
    by_tile = {}
    with open(per_tile_csv) as f:
        for row in csv.DictReader(f):
            if row["arm"] == "baseline":
                by_tile[row["tile"]] = float(row["gt_bsds_f1"])
    if not by_tile:
        raise SystemExit("no baseline rows in per-tile CSV")
    ordered = sorted(by_tile, key=by_tile.get)
    idx = [0, len(ordered) // 4, len(ordered) // 2, (3 * len(ordered)) // 4, len(ordered) - 1]
    return [(ordered[i], by_tile[ordered[i]]) for i in sorted(set(idx))][:n]


def load_cell(path, invert=False):
    img = Image.open(path).convert("L").resize((CELL, CELL))
    if invert:
        img = Image.eval(img, lambda p: 255 - p)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="results/ipadapter_probe_20260930")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = TRACK / args.probe_root if not Path(args.probe_root).is_absolute() else Path(args.probe_root)
    per_tile = root / "scores_per_tile.csv"
    tiles = pick_tiles(per_tile)

    arms = ["baseline"] + sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name != "baseline"
    )
    cols = ["condition"] + arms + ["GT"]

    width = CELL * len(cols)
    height = LABEL_H + len(tiles) * (CELL + LABEL_H)
    sheet = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(sheet)

    for c, name in enumerate(cols):
        draw.text((c * CELL + 4, 5), name[:34], fill=0)

    for r, (tile, f1) in enumerate(tiles):
        y = LABEL_H + r * (CELL + LABEL_H)
        stem = Path(tile).stem
        draw.text((4, y + CELL + 4), f"{tile}  baseline f1={f1:.4f}", fill=0)
        for c, name in enumerate(cols):
            if name == "condition":
                cell = load_cell(COND_DIR / tile, invert=True)
            elif name == "GT":
                cell = load_cell(GT_DIR / tile)
            else:
                p = root / name / f"{stem}_out.png"
                if not p.exists():
                    continue
                cell = load_cell(p)
            sheet.paste(cell, (c * CELL, y))

    out = Path(args.out) if args.out else root / "montage_ipadapter_probe.png"
    sheet.save(out)
    print(f"wrote {out}")
    print("columns:", " | ".join(cols))
    for tile, f1 in tiles:
        print(f"  {tile}  baseline f1={f1:.4f}")


if __name__ == "__main__":
    main()
