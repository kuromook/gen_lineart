"""Contact sheet for the stage-2 koma arms. No number is reported without one
of these paths beside it (initial notice; /tmp is forbidden).

Columns: the rough, the GT line art, the GT fill in red, then each arm's fill
in red over the rough -- so the reader's actual question, "is that red region
the blacked-in area", is answerable by looking.

Panels are drawn with a seed fixed here, before any score was read, and the
sheet says so in its own title. The near-blank stratum gets its own sheet:
pooling the two is what C5 forbids.

Usage: montage_fill_koma.py --side held --stratum normal --n 8
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
from koma_split import ROOT, load as load_split  # noqa: E402
from score_fill_koma import NEAR_BLANK_INK, arms_for, fit_freq_table  # noqa: E402

OUT = Path("results/fill_koma_20261010")
SEED = 20261010
CELL = 190


def overlay(gray, mask, color=(40, 40, 220)):
    rgb = np.dstack([gray] * 3).astype(np.uint8)
    if mask is not None and mask.any():
        rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", default="held", choices=("held", "train"))
    ap.add_argument("--stratum", default="normal", choices=("normal", "near_blank"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pool", type=int, default=120, help="panels to screen for the stratum")
    args = ap.parse_args()

    split = load_split()
    names = [n for n, _, _ in split[args.side]]
    rng = random.Random(SEED)
    rng.shuffle(names)
    table, _ = fit_freq_table([n for n, _, _ in split["train"]], every=40)

    show_arms = ["rough_dark", "rough_dark@budget", "freq_table"]
    picked = []
    for name in names[:args.pool]:
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        nb = ink_of(g).mean() < NEAR_BLANK_INK
        if (args.stratum == "near_blank") != nb:
            continue
        got = arms_for(name, table)
        if got is None:
            continue
        picked.append((name, got))
        if len(picked) >= args.n:
            break
    if not picked:
        raise SystemExit(f"no panels in stratum {args.stratum}")

    cols = 3 + len(show_arms)
    font = ImageFont.load_default()
    head = 20
    sheet = Image.new("RGB", (cols * CELL + 260, head + len(picked) * (CELL + 14)),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 5), f"{args.side} / {args.stratum} / RANDOM seed={SEED} "
                   f"(fixed before any score was read) / fill mask in red, radius 4 / "
                   f"region = 8-conn component >= 500px", fill=(0, 0, 0), font=font)
    for r, (name, (preds, gt, gt_ink, r_ink)) in enumerate(picked):
        rough = cv2.imread(str(Path(ROOT) / "rough" / name), cv2.IMREAD_GRAYSCALE)
        line = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        panels = [("rough", overlay(rough, None)), ("GT line", overlay(line, None)),
                  ("GT fill", overlay(line, gt))]
        panels += [(a, overlay(rough, preds[a])) for a in show_arms]
        y = head + r * (CELL + 14)
        for c, (lbl, img) in enumerate(panels):
            x = c * CELL
            if r == 0:
                d.text((x + 2, y - 11), lbl, fill=(0, 0, 0), font=font)
            sheet.paste(img.resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(200, 200, 200))
        d.text((cols * CELL + 4, y + 2),
               f"{name[:42]}\nGT fill area {gt.mean():.4f}", fill=(0, 0, 0), font=font)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"montage_{args.side}_{args.stratum}_random.png"
    sheet.save(p)
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
