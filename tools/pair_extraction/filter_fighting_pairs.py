"""Apply the ako5ver2-derived strict stroke-scale filter to the original
pre-cropped `fighting` source (`dataset_fighting.zip`, formerly named
`lineart.zip`; renamed because "lineart" was too generic and collided with an
unrelated legacy `lineart_`-prefixed source category in
`lineart/dataset_gate.py`'s `DEFAULT_LABELS`. Layout:
`rough/rough_<page>-<index>.jpg` paired with `2/line_<page>-<index>.jpg`,
already fixed 480x480 native tiles.

Unlike ako5ver2/fitness, this source needs no region matching or alignment
search: pairs are already 1:1 by filename suffix and visually well aligned.
What it has not been through is the strict content-quality filter (stroke
width, gray/soft-ink fringe, long straight lines, tight-tolerance edge
correspondence, tile-score cutoff) built while reviewing ako5ver2 native
tiles. This script reuses that filter directly from `tile_region_manifest_480.py`
so this source gets the same quality bar. See
`doc/region_dataset_extraction_policy.md`.

Dry-run is the default; use --save after reviewing QC.
"""

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_region_valid_masks import build_mask
from tile_region_manifest_480 import TILE, analyze_tile, edge_map

SUFFIX_RE = re.compile(r"rough_(\d+-\d+)\.jpg$")


def make_mask_args(args):
    class MaskArgs:
        pass

    mask_args = MaskArgs()
    mask_args.support_px = args.mask_support_px
    mask_args.window = args.mask_window
    mask_args.edge_density = args.mask_edge_density
    mask_args.expand_ignore = args.mask_expand_ignore
    mask_args.close_ignore = args.mask_close_ignore
    mask_args.blur_sigma = 1.0
    mask_args.include_black_fill = False
    mask_args.include_rough_extra = False
    mask_args.black_threshold = 40
    mask_args.rough_black_threshold = 96
    mask_args.black_window = 45
    mask_args.black_density = 0.16
    mask_args.min_valid_ratio = args.mask_min_valid_ratio
    return mask_args


def make_qc(rows, path, count):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 240, 34
    canvas = Image.new("RGB", (thumb * 4, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        y = index * (thumb + label_h)
        rough_tile, line_tile, mask_tile = row["_rough"], row["_line"], row["_mask"]
        overlay = np.full((TILE, TILE, 3), 255, np.uint8)
        overlay[edge_map(rough_tile)] = (255, 60, 60)
        overlay[edge_map(line_tile)] = (40, 80, 255)
        mask_vis = np.where(mask_tile > 127, line_tile, 220).astype(np.uint8)
        for column, image in enumerate((rough_tile, line_tile, mask_vis, overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, y))
        text = (
            f'{row["name"]} score={row["tile_score"]:.2f} sF1={row["strict_edge_f1"]:.2f} '
            f'sRec={row["strict_edge_recall"]:.2f} w50={row["line_width_p50"]:.1f} '
            f'soft={row["soft_ink_ratio"]:.2f} long={row["long_line_ratio"]:.2f} ink={row["line_ink"]:.3f}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(rows, path):
    fields = [
        "rank", "name", "suffix", "tile_score", "edge_f1", "chamfer", "strict_edge_f1",
        "strict_edge_precision", "strict_edge_recall", "orientation_entropy",
        "rough_std", "line_ink", "line_width_p50", "line_width_p95", "long_line_ratio",
        "soft_ink_ratio", "largest_black_component_ratio", "thick_ink_ratio", "support",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_tiles(rows, rough_out, line_out, mask_out, list_out):
    rough_out, line_out, mask_out = Path(rough_out), Path(line_out), Path(mask_out)
    for path in (rough_out, line_out, mask_out):
        path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        Image.fromarray(row["_rough"]).save(rough_out / row["name"])
        Image.fromarray(row["_line"]).save(line_out / row["name"])
        Image.fromarray(row["_mask"]).save(mask_out / row["name"])
    Path(list_out).parent.mkdir(parents=True, exist_ok=True)
    with open(list_out, "w") as file:
        file.write("\n".join(row["name"] for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--rough-prefix", default="rough/rough_")
    parser.add_argument("--line-prefix", default="2/line_")
    parser.add_argument("--name-prefix", default="fighting")

    parser.add_argument("--mask-support-px", type=int, default=20)
    parser.add_argument("--mask-window", type=int, default=61)
    parser.add_argument("--mask-edge-density", type=float, default=0.030)
    parser.add_argument("--mask-expand-ignore", type=int, default=16)
    parser.add_argument("--mask-close-ignore", type=int, default=16)
    parser.add_argument("--mask-min-valid-ratio", type=float, default=0.35)

    # Style/content gate: per-source, tune freely. Alignment gate (chamfer,
    # strict edge correspondence) is NOT CLI-configurable here; it uses the
    # fixed ALIGNMENT_* constants in tile_region_manifest_480.py, shared by
    # every source. See that module's docstring for the split rationale.
    parser.add_argument("--min-support", type=float, default=0.80)
    parser.add_argument("--ink-min", type=float, default=0.012)
    parser.add_argument("--ink-max", type=float, default=0.08)
    parser.add_argument("--max-black-component-ratio", type=float, default=0.025)
    parser.add_argument("--max-thick-ink-ratio", type=float, default=0.015)
    parser.add_argument("--thick-ink-kernel", type=int, default=9)
    parser.add_argument("--min-rough-std", type=float, default=8.0)
    parser.add_argument("--min-edge-pixels", type=int, default=120)
    parser.add_argument("--min-entropy", type=float, default=0.45)
    parser.add_argument("--min-tile-score", type=float, default=2.5)
    parser.add_argument("--score-mode", choices=("legacy", "strict"), default="strict")
    parser.add_argument("--max-line-width-p50", type=float, default=6.0)
    parser.add_argument("--long-line-fraction", type=float, default=0.5)
    parser.add_argument("--max-long-line-ratio", type=float, default=0.25)
    parser.add_argument("--max-soft-ink-ratio", type=float, default=0.40)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qc-count", type=int, default=40)
    parser.add_argument("--csv-out", default="results/fighting_native_tiles_480_strict.csv")
    parser.add_argument("--qc-out", default="results/fighting_native_tiles_480_strict_qc.png")
    parser.add_argument("--qc-tail-out", default="results/fighting_native_tiles_480_strict_qc_tail.png")
    parser.add_argument("--rough-out", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-out", default="dataset/pairs_480/train/line_fighting_native_strict_20260725")
    parser.add_argument("--mask-out", default="dataset/regions_fighting_native_20260725_masks")
    parser.add_argument("--list-out", default="dataset/pairs_480/valid_train_fighting_native_strict_20260725.txt")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    args.tile = TILE
    mask_args = make_mask_args(args)

    with zipfile.ZipFile(args.zip_path) as archive:
        names = archive.namelist()
        rough_names = sorted(n for n in names if n.startswith(args.rough_prefix) and n.endswith(".jpg"))
        if args.limit:
            rough_names = rough_names[: args.limit]

        accepted = []
        rejected = 0
        for index, rough_name in enumerate(rough_names, 1):
            suffix = rough_name[len(args.rough_prefix):]
            line_name = f"{args.line_prefix}{suffix}"
            if line_name not in names:
                rejected += 1
                continue
            rough_tile = np.asarray(
                ImageOps.autocontrast(Image.open(io.BytesIO(archive.read(rough_name))).convert("L"), cutoff=0)
            )
            line_tile = np.asarray(Image.open(io.BytesIO(archive.read(line_name))).convert("L"))
            if rough_tile.shape != (TILE, TILE) or line_tile.shape != (TILE, TILE):
                rejected += 1
                continue

            mask_arr, _ = build_mask(Image.fromarray(rough_tile), Image.fromarray(line_tile), mask_args)
            region_stats = {
                "source_long_side": float(TILE),
                "src_per_out": 1.0,
                "src_px_per_tile": float(TILE),
                "source_origin_x": 0.0,
                "source_origin_y": 0.0,
            }
            stats = analyze_tile(rough_tile, line_tile, mask_arr, args, region_stats)
            if stats is None:
                rejected += 1
                continue
            suffix_clean = suffix.replace(".jpg", "")
            name = f"{args.name_prefix}_{suffix_clean}.jpg"
            accepted.append({
                **stats,
                "name": name,
                "suffix": suffix_clean,
                "_rough": rough_tile,
                "_line": line_tile,
                "_mask": mask_arr,
            })
            if index % 50 == 0 or index == len(rough_names):
                print(f"pairs: {index}/{len(rough_names)} accepted={len(accepted)}", flush=True)

    accepted.sort(key=lambda row: row["tile_score"], reverse=True)
    for rank, row in enumerate(accepted, 1):
        row["rank"] = rank
    write_csv(accepted, args.csv_out)
    make_qc(accepted, args.qc_out, args.qc_count)
    make_qc(accepted[-args.qc_count :], args.qc_tail_out, args.qc_count)
    if args.save:
        save_tiles(accepted, args.rough_out, args.line_out, args.mask_out, args.list_out)

    action = "saved" if args.save else "dry-run"
    print(f"{action}: pairs={len(rough_names)} rejected={rejected} accepted={len(accepted)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}, {args.qc_tail_out}")
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
