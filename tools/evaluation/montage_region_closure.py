"""Visual backing for check_region_closure.py: red = solid fill whose region
closes inside the window, blue = fill in a component touching the border, i.e.
a region the window does not show whole.

Tiles are drawn with a seed, not picked after reading the numbers.

Usage:
  montage_region_closure.py --dir <line dir> --label ako5_koma --out <png>
  montage_region_closure.py --tiles ako5_held --split train --label ako5_480 --out <png>
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
SEED = 20261010
CELL = 230
MIN_COMP_PX = 200


def render(gray):
    ink = ink_of(gray)
    fill = fill_mask(ink)
    h, w = gray.shape
    rgb = np.dstack([gray] * 3).astype(np.uint8)
    if fill.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
        border = np.zeros_like(fill)
        inside = np.zeros_like(fill)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < MIN_COMP_PX:
                continue
            m = lab == i
            if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                border |= m
            else:
                inside |= m
        rgb[inside] = (0.35 * rgb[inside] + 0.65 * np.array([220, 40, 40])).astype(np.uint8)
        rgb[border] = (0.35 * rgb[border] + 0.65 * np.array([40, 80, 230])).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir")
    ap.add_argument("--tiles")
    ap.add_argument("--split", choices=("train", "test"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(SEED)
    if args.dir:
        items = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    else:
        lst = Path("results/baseline_fill_20261009/lists") / f"{args.tiles}.txt"
        items = [SHARED / args.split / "line" / f"{l.strip().replace('.jpg', '')}.jpg"
                 for l in open(lst) if l.strip()]
    picked = rng.sample(items, min(args.n, len(items)))

    font = ImageFont.load_default()
    head = 20
    sheet = Image.new("RGB", (len(picked) * CELL, head + CELL + 14), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{args.label} / RANDOM seed={SEED} / red = fill region closed inside the "
                   f"window, blue = component touching the border", fill=(0, 0, 0), font=font)
    for c, p in enumerate(picked):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = render(g)
        sheet.paste(img.resize((CELL, CELL)), (c * CELL, head))
        d.rectangle([c * CELL, head, c * CELL + CELL - 1, head + CELL - 1], outline=(190, 190, 190))
        d.text((c * CELL + 2, head + CELL + 1), f"{p.stem[:30]} {g.shape[1]}x{g.shape[0]}",
               fill=(0, 0, 0), font=font)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
