"""Build a stratified visual QC contact sheet for the unpaired-rough
candidate pool (dataset/unpaired_rough_candidates/<source>/rough/*),
gathered by run_unpaired_rough_drops_measure_20260804.sh as domain-only
training material for the `diffusion` branch's rough-domain generation
work (not ControlNet conditioning material, per 2026-08-04/05 direction
change -- see doc/work_log.md).

Motivation: an earlier automated-filter attempt this session (rough-side
dark-pixel ratio, rough-vs-line pixel correlation, exact-duplicate check)
did not surface a genuine "finished ink mistakenly labeled as rough"
cluster -- outliers on both metrics turned out to be legitimate simple/
confident rough strokes, not contamination, on visual inspection. This
tool produces a larger stratified-random sample (proportional to each
source's pool size) for direct human visual review instead of relying on
a metric-based pre-filter, since the metrics didn't cleanly separate
signal from legitimate content.

Thumbnails (not native tile resolution, unlike
tools/compare/make_tile_review_sheet.py) are enough for this specific
judgment call -- "does this look like graphite rough or finished ink" is
a gross visual distinction, unlike fine positional rough/line
correspondence, which does need full resolution.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def list_sources(root):
    return sorted(p.name for p in Path(root).iterdir() if (p / "rough").is_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset/unpaired_rough_candidates")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--cell", type=int, default=200)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/qc_unpaired_rough_candidates_20260804.png")
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    sources = list_sources(root)

    per_source_files = {}
    for source in sources:
        files = sorted((root / source / "rough").glob("*"))
        per_source_files[source] = files
    total_pool = sum(len(files) for files in per_source_files.values())

    picks = []  # list of (source, path)
    for source in sources:
        files = per_source_files[source]
        share = max(1, round(args.total * len(files) / total_pool))
        share = min(share, len(files))
        for path in random.sample(files, share):
            picks.append((source, path))

    cell, cols = args.cell, args.cols
    label_h = 18
    rows = (len(picks) + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols, (cell + label_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for index, (source, path) in enumerate(picks):
        img = Image.open(path).convert("L").resize((cell, cell))
        row, col = divmod(index, cols)
        top, left = row * (cell + label_h), col * cell
        canvas.paste(img.convert("RGB"), (left, top))
        draw.text((left + 3, top + cell + 2), source, fill="red", font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    counts = {source: sum(1 for s, _ in picks if s == source) for source in sources}
    print(f"wrote: {args.out} tiles={len(picks)} of pool={total_pool}")
    print("per-source picks:", counts)


if __name__ == "__main__":
    main()
