"""Contact sheets for the fill masks: rough, GT, GT fill, and each arm.

Two modes, and the mode is printed into the sheet so a reader always knows
which one they are looking at:

  --mode random   tiles drawn with a seed fixed in the registration, before
                  any score was read. This is the sheet a claim may rest on.
  --mode extreme  the best and worst tiles by one arm's IoU. Useful, and
                  selected on the thing being judged, so the title says so.

The fill mask is drawn as a red overlay on the grey image rather than as a
separate binary panel, because the question a reader has is "is that red
region the blacked-in area", and two side-by-side binaries do not answer it.

Usage:
  montage_fill.py --list housei_test --split test --mode random --out <png>
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of, load_gray  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
R = Path("results/baseline_fill_20261009")
SEED = 20261009  # registered before any score was read
CELL = 150
LABEL = 13


def overlay(gray, mask, color=(220, 40, 40)):
    rgb = np.dstack([gray] * 3).astype(np.uint8)
    if mask is not None and mask.any():
        rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def panels(tile, split, pool):
    """Whatever is on disk. An arm that has not been materialised is left out
    rather than drawn blank, so an empty column can never be misread as an arm
    that painted nothing."""
    stem = tile[:-4] if tile.endswith(".jpg") else tile
    rough = load_gray(SHARED / split / "rough" / f"{stem}.jpg")
    gt = load_gray(SHARED / split / "line" / f"{stem}.jpg")
    out = [("rough", overlay(rough, None)),
           ("GT", overlay(gt, None)),
           ("GT fill", overlay(gt, fill_mask(ink_of(gt))))]
    pp = R / "preproc" / pool / f"{stem}.jpg"
    if pp.exists():
        pp_raw = load_gray(pp)
        pp_vis = 255 - pp_raw  # show it the way a person reads line art
        out += [("preproc@32", overlay(pp_vis, None)),
                ("preproc fill", overlay(pp_vis, fill_mask(pp_raw > 32)))]
    ms = R / "msgan" / pool / f"{stem}_out.png"
    if ms.exists():
        ms_gray = load_gray(ms)
        out += [("msgan", overlay(ms_gray, None)),
                ("msgan fill", overlay(ms_gray, fill_mask(ink_of(ms_gray))))]
    return out


def build(tiles, split, pool, title, out_path, captions=None):
    font = ImageFont.load_default()
    rows = len(tiles)
    cols = len(panels(tiles[0], split, pool))
    head = 18
    sheet = Image.new("RGB", (cols * CELL + 150, head + rows * (CELL + LABEL)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill=(0, 0, 0), font=font)
    for r, tile in enumerate(tiles):
        y = head + r * (CELL + LABEL)
        for c, (name, img) in enumerate(panels(tile, split, pool)):
            x = c * CELL
            if r == 0:
                d.text((x + 2, y - 11), name, fill=(0, 0, 0), font=font)
            sheet.paste(img.resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(200, 200, 200))
        cap = tile if captions is None else f"{tile}\n{captions[r]}"
        d.text((cols * CELL + 4, y + 2), cap.replace(".jpg", ""), fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", required=True)
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument("--mode", default="random", choices=("random", "extreme"))
    ap.add_argument("--arm", default="msgan", help="extreme mode: rank by this arm's IoU")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stratum", default="normal_fill",
                    choices=("normal_fill", "near_blank", "no_fill", "any"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prof = {r["tile"]: r for r in csv.DictReader(open(R / "gt_fill_profile.csv"))}
    tiles = [l.strip() for l in open(R / "lists" / f"{args.list}.txt") if l.strip()]

    def keep(t):
        p = prof[t[:-4] if t.endswith(".jpg") else t]
        if args.stratum == "any":
            return True
        if args.stratum == "near_blank":
            return p["near_blank"] == "1"
        if args.stratum == "no_fill":
            return p["near_blank"] == "0" and p["has_fill"] == "0"
        return p["near_blank"] == "0" and p["has_fill"] == "1"

    tiles = [t for t in tiles if keep(t)]
    if not tiles:
        raise SystemExit(f"no tiles in stratum {args.stratum}")

    if args.mode == "random":
        rng = random.Random(SEED)
        picked = rng.sample(tiles, min(args.n, len(tiles)))
        title = (f"{args.list} / {args.stratum} / RANDOM seed={SEED} "
                 f"(registered before any score was read) / fill mask in red, radius 4")
        caps = None
    else:
        per_tile = R / f"per_tile_{args.list}.csv"
        iou = {}
        for row in csv.DictReader(open(per_tile)):
            if row["arm"] == args.arm and row["radius"] == "4.0" and row["iou"]:
                iou[row["tile"]] = float(row["iou"])
        ranked = sorted((t for t in tiles if t[:-4] in iou),
                        key=lambda t: iou[t[:-4]])
        half = max(args.n // 2, 1)
        picked = ranked[:half] + ranked[-half:]
        caps = [f"{args.arm} IoU {iou[t[:-4]]:.3f}" for t in picked]
        title = (f"{args.list} / {args.stratum} / SELECTED: worst {half} and best {half} "
                 f"tiles by {args.arm} IoU -- NOT a random sample")

    build(picked, args.split, args.list, title, Path(args.out), captions=caps)


if __name__ == "__main__":
    main()
