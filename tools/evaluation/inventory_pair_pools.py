"""Inventory of every GT line-art pool, by source, on the axes that decide
whether two pools are the same task.

Why (2026-09-11): this project has been training on one source family and
evaluating on another without that being written down anywhere, and it read
one pool's line_width_p50 of 7.59 as "thick strokes" when it was solid fill.
Both mistakes come from not having a per-source profile on record.

The axes are chosen to answer the questions actually asked of a pool:

  fill_ratio        is this pool solid-black heavy? (hair, clothing)
  line_width_p50    read only together with fill_ratio -- see
                    measure_lineart_profile.fill_ratio
  ink_ratio         how much is drawn at all
  near_white_frac   how much of the page is paper
  blank_tile_frac   what share of tiles are effectively empty; a pool with
                    many blanks makes per-tile f1 unstable and rewards a
                    model that draws nothing
  grid_ink_cv       how unevenly ink is spread across an 8x8 grid -- a
                    character on white paper is uneven, an all-over
                    background texture is even. This is the cheap proxy for
                    "character art vs. scenery/pattern"; it does not replace
                    the contact sheet.

Character-vs-natural-subject cannot be settled by these numbers, so the
script also writes one contact sheet per pool for a human read, and the two
are meant to be used together.

Usage:
    inventory_pair_pools.py --out-dir results/pool_inventory_20260911
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
from measure_lineart_profile import profile_metrics  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
TRACK = Path(__file__).resolve().parents[2]  # tools/evaluation/ -> track root
IMAGE_SIZE = 480
BLANK_INK_RATIO = 0.002  # below this a tile is effectively an empty page

AXES = ["ink_ratio", "fill_ratio", "line_width_p50", "near_white_frac",
        "midtone_frac", "orientation_entropy", "grid_ink_cv"]


def source_of(name):
    """Strip the trailing tile coordinates to get the source id."""
    stem = name[:-4] if name.endswith(".jpg") else name
    parts = stem.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) or stem


def gather():
    """All GT line tiles, grouped by pool. Two universes are kept apart:
    the shared train/test splits, and this track's own train_list (which is
    what the fine-tunes actually saw)."""
    pools = defaultdict(list)
    for split in ("train", "test"):
        d = SHARED / split / "line"
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix == ".jpg":
                pools[f"shared/{split}:{source_of(p.name)}"].append(p)

    train_list = TRACK / "data/train_list.txt"
    if train_list.exists():
        line_dir = TRACK / "data/line"
        for name in (l.strip() for l in open(train_list)):
            if name:
                p = line_dir / name
                if p.exists():
                    pools[f"trainlist:{source_of(name)}"].append(p)
    return pools


def profile_pool(paths, sample, rng):
    picked = paths if len(paths) <= sample else rng.sample(paths, sample)
    acc = defaultdict(list)
    blanks = 0
    for p in picked:
        m = profile_metrics(p)
        for a in AXES:
            acc[a].append(float(m[a]))
        if m["ink_ratio"] < BLANK_INK_RATIO:
            blanks += 1
    out = {a: float(np.mean(acc[a])) for a in AXES}
    out["blank_tile_frac"] = blanks / len(picked)
    out["n_total"] = len(paths)
    out["n_sampled"] = len(picked)
    return out


def contact_sheet(name, paths, out_path, rng, cols=8, rows=4):
    picked = paths if len(paths) <= cols * rows else rng.sample(paths, cols * rows)
    cell, lab = 130, 12
    font = ImageFont.load_default()
    sheet = Image.new("L", (cols * cell, rows * (cell + lab) + lab), 255)
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), name, fill=0, font=font)
    for i, p in enumerate(picked):
        x, y = (i % cols) * cell, lab + (i // cols) * (cell + lab)
        d.text((x + 2, y), p.name[:18], fill=0, font=font)
        sheet.paste(Image.open(p).convert("L").resize((cell, cell)), (x, y + lab))
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/pool_inventory_20260911")
    ap.add_argument("--sample", type=int, default=120,
                    help="tiles profiled per pool (pools are large; the axes are means)")
    ap.add_argument("--min-tiles", type=int, default=20,
                    help="pools smaller than this are folded into an 'other' row")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "contact").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pools = gather()
    small = [k for k, v in pools.items() if len(v) < args.min_tiles]
    merged = defaultdict(list)
    for k in small:
        universe = k.split(":")[0]
        merged[f"{universe}:(other <{args.min_tiles} tiles)"] += pools.pop(k)
    pools.update(merged)

    rows = []
    for name in sorted(pools, key=lambda k: -len(pools[k])):
        paths = pools[name]
        stats = profile_pool(paths, args.sample, rng)
        rows.append({"pool": name, **stats})
        contact_sheet(name, paths, out / "contact" / f"{name.replace('/', '_').replace(':', '_')}.png", rng)

    csv_path = out / "inventory.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pool", "n_total", "n_sampled", "blank_tile_frac"] + AXES)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'pool':34}{'n':>7}{'ink':>8}{'fill%':>8}{'line_w':>8}"
          f"{'白%':>7}{'空白tile%':>10}{'grid_cv':>9}")
    for r in rows:
        print(f"{r['pool']:34}{r['n_total']:7d}{r['ink_ratio']:8.4f}"
              f"{r['fill_ratio']*100:7.1f}%{r['line_width_p50']:8.2f}"
              f"{r['near_white_frac']*100:6.1f}%{r['blank_tile_frac']*100:9.1f}%"
              f"{r['grid_ink_cv']:9.2f}")
    print(f"\nsaved: {csv_path}")
    print(f"contact sheets: {out/'contact'}  (numbers cannot tell character art from"
          f" scenery -- read these beside the table)")


if __name__ == "__main__":
    main()
