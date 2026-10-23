"""Contact sheet for the stage-2 LEARNED arms. No number is reported without
one of these paths beside it (initial notice; /tmp is forbidden).

Columns: the rough, the GT line art, the GT fill in red, then M0, M1 and
M1-with-another-panel's-context (NC4) in red over the rough -- so the
reader's actual question, "is that red region the blacked-in area", is
answerable by looking, and so that "predicts almost nothing" can be told
apart from "predicts speckles everywhere" by eye as well as by `npred/ngt`.

The panel seed is fixed here, in the commit that precedes any model score
being read, and the sheet says so in its own title. The near-blank stratum
gets its own sheet: pooling the two is what C5 forbids.

Usage: montage_fill_seg.py --stratum normal --n 8
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
from fill_mask import ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from montage_fill_koma import CELL, overlay  # noqa: E402
from region_metrics import regions, MIN_REGION_PX  # noqa: E402
from score_fill_koma import NEAR_BLANK_INK  # noqa: E402
from score_fill_seg_models import (CKPT_DIR, ctx_of, derangement,  # noqa: E402
                                   load_model, predict)
from train_fill_seg import panel_tensors  # noqa: E402

OUT = Path("results/fill_koma_20261010")
SEED = 20261010      # the same seed the non-learned sheet used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stratum", default="normal", choices=("normal", "near_blank"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pool", type=int, default=160)
    ap.add_argument("--step", type=int, default=40000)
    args = ap.parse_args()

    split = load_split()
    names = [n for n, _, _ in split["held"]]
    shuffle = derangement(names)
    rng = random.Random(SEED)
    order = names[:]
    rng.shuffle(order)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {a: load_model(CKPT_DIR / f"{a}_step{args.step}.pt", a, device)[0]
              for a in ("M0", "M1")}

    picked = []
    for name in order[:args.pool]:
        line = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if line is None:
            continue
        if (ink_of(line).mean() < NEAR_BLANK_INK) != (args.stratum == "near_blank"):
            continue
        t = panel_tensors(name)
        if t is None:
            continue
        rough, fill, ctx, (h0, w0) = t
        other = ctx_of(shuffle[name])
        cells = {
            "M0": predict(models["M0"], "M0", rough, ctx, device)[:h0, :w0],
            "M1": predict(models["M1"], "M1", rough, ctx, device)[:h0, :w0],
            "NC4 ctx-shuffled M1": predict(models["M1"], "M1", rough, other,
                                           device)[:h0, :w0],
        }
        picked.append((name, line, fill[:h0, :w0], cells))
        print(f"{len(picked)}/{args.n} {name}", flush=True)
        if len(picked) >= args.n:
            break
    if not picked:
        raise SystemExit(f"no panels in stratum {args.stratum}")

    labels = ["rough", "GT line", "GT fill", "M0", "M1", "NC4 ctx-shuffled M1"]
    font = ImageFont.load_default()
    head = 20
    sheet = Image.new("RGB", (len(labels) * CELL + 300,
                              head + len(picked) * (CELL + 14)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 5), f"held / {args.stratum} / RANDOM seed={SEED} (fixed before any "
                   f"model score was read) / step {args.step} / fill mask in red, "
                   f"radius 4 / region = 8-conn component >= {MIN_REGION_PX}px",
           fill=(0, 0, 0), font=font)
    for r, (name, line, gt, cells) in enumerate(picked):
        rough = cv2.imread(str(Path(ROOT) / "rough" / name), cv2.IMREAD_GRAYSCALE)
        imgs = [("rough", overlay(rough, None)), ("GT line", overlay(line, None)),
                ("GT fill", overlay(line, gt))]
        imgs += [(k, overlay(rough, v)) for k, v in cells.items()]
        y = head + r * (CELL + 14)
        for c, (lbl, img) in enumerate(imgs):
            x = c * CELL
            if r == 0:
                d.text((x + 2, y - 11), lbl, fill=(0, 0, 0), font=font)
            sheet.paste(img.resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(200, 200, 200))
        note = []
        for k, v in cells.items():
            n_reg = len(regions(v, MIN_REGION_PX)[1])
            note.append(f"{k.split()[0]} area {v.mean():.4f} / {n_reg} regions")
        d.text((len(labels) * CELL + 4, y + 2),
               f"{name[:40]}\nGT fill area {gt.mean():.4f} / "
               f"{len(regions(gt, MIN_REGION_PX)[1])} regions\n" + "\n".join(note),
               fill=(0, 0, 0), font=font)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"montage_models_held_{args.stratum}_step{args.step}.png"
    sheet.save(p)
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
