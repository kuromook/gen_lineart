"""Apply the ako5ver2-derived strict stroke-scale filter to
`match_kurip_regions.py`-style matches, for any source using that route.

`match_kurip_regions.py` (name predates its use beyond its original source;
fully generic via `--zip`/`--zip-root`) already performs the region-matching
step this project requires (line-anchored, rough-side local offset search; see
`doc/EXTRACTION_RULES.md`, route `needs_global_or_local_alignment`). Its
candidate tiles are already native-resolution fixed 480x480 crops, so none of
the source-scale normalization work done for ako5ver2 applies here (`src_per_out`
is always 1.0 by construction).

What that matcher's output lacks on its own is the strict content-quality
filter (stroke width, gray/soft-ink fringe, long straight lines, tight-tolerance
edge correspondence, and a tile-score cutoff) discovered while reviewing
ako5ver2 native tiles: metrics that individually pass every gate can still
describe an unrelated rough/line pair. This script reuses that filter directly
from `tile_region_manifest_480.py` instead of re-implementing it, so every
source using this route gets the same quality bar. See
`doc/region_dataset_extraction_policy.md`.

Also supports `--offset`/`--limit`/`--append` for chunked runs: this
environment has been observed to silently kill long-running background
processes somewhere around 10-13 minutes with no traceback, so a slow full
pass should be split into several short, resumable invocations rather than run
as one long job.

Dry-run is the default; use --save after reviewing QC.
"""

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_region_valid_masks import build_mask
from tile_region_manifest_480 import TILE, analyze_tile, edge_map


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def read_zip_member(archive, zip_root, name):
    for candidate in (f"{zip_root}/{name}", f"{zip_root}\\{name}", name):
        try:
            return archive.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_manifest(zip_path, zip_root):
    import json

    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(read_zip_member(archive, zip_root, "manifest.json"))


def page_lookup(manifest):
    return {Path(entry.get("file", entry["sketch"])).stem.replace("page", ""): entry for entry in manifest}


def load_page_pair(archive, zip_root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(archive, zip_root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(archive, zip_root, entry["line"]))).convert("L")
    return np.asarray(ImageOps.autocontrast(rough, cutoff=0)), np.asarray(line)


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


CSV_FIELDS = [
    "rank", "name", "page", "line_x", "line_y", "rough_x", "rough_y", "dx", "dy",
    "match_score", "tile_score", "edge_f1", "chamfer", "strict_edge_f1",
    "strict_edge_precision", "strict_edge_recall", "orientation_entropy",
    "rough_std", "line_ink", "line_width_p50", "line_width_p95", "long_line_ratio",
    "soft_ink_ratio", "largest_black_component_ratio", "thick_ink_ratio", "support",
]


def write_csv(rows, path, append=False):
    """`append=True` re-numbers `rank` to continue after any existing rows.

    Used for chunked runs (`--offset`/`--limit`) so a slow full pass can be
    split into several short, resumable invocations. See
    `doc/raw_dataset_extraction_knowledge.md` for why this was needed: this
    environment silently kills background jobs somewhere around 10-13 minutes
    regardless of the requested timeout, with no traceback.
    """
    path = Path(path)
    start_rank = 0
    if append and path.exists():
        with open(path, newline="") as file:
            existing = list(csv.DictReader(file))
        start_rank = len(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for offset, row in enumerate(rows, 1):
            writer.writerow({**row, "rank": start_rank + offset})


def save_tiles(rows, rough_out, line_out, mask_out, list_out, append=False):
    rough_out, line_out, mask_out = Path(rough_out), Path(line_out), Path(mask_out)
    for path in (rough_out, line_out, mask_out):
        path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        Image.fromarray(row["_rough"]).save(rough_out / row["name"])
        Image.fromarray(row["_line"]).save(line_out / row["name"])
        Image.fromarray(row["_mask"]).save(mask_out / row["name"])
    list_path = Path(list_out)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and list_path.exists() else "w"
    with open(list_path, mode) as file:
        file.write("\n".join(row["name"] for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-csv", required=True, help="output of match_kurip_regions.py")
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--zip-root", required=True)
    parser.add_argument("--name-prefix", default="matched")

    # Mask generation (native-scale calibration validated on ako5ver2 native tiles).
    parser.add_argument("--mask-support-px", type=int, default=20)
    parser.add_argument("--mask-window", type=int, default=61)
    parser.add_argument("--mask-edge-density", type=float, default=0.030)
    parser.add_argument("--mask-expand-ignore", type=int, default=16)
    parser.add_argument("--mask-close-ignore", type=int, default=16)
    parser.add_argument("--mask-min-valid-ratio", type=float, default=0.35)

    # Style/content gate: per-source, tune freely (see module docstring in
    # tile_region_manifest_480.py for the alignment-vs-style split rationale).
    # Alignment gate (chamfer, strict edge correspondence) is NOT
    # CLI-configurable here; it uses the fixed ALIGNMENT_* constants in
    # tile_region_manifest_480.py, shared by every source.
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

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="append to csv-out/list-out instead of overwriting")
    parser.add_argument("--qc-count", type=int, default=40)
    parser.add_argument("--csv-out", default="results/matched_native_tiles_480_strict.csv")
    parser.add_argument("--qc-out", default="results/matched_native_tiles_480_strict_qc.png")
    parser.add_argument("--qc-tail-out", default="results/matched_native_tiles_480_strict_qc_tail.png")
    parser.add_argument("--rough-out", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-out", default="dataset/pairs_480/train/line_matched_native_strict")
    parser.add_argument("--mask-out", default="dataset/regions_matched_native_masks")
    parser.add_argument("--list-out", default="dataset/pairs_480/valid_train_matched_native_strict.txt")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    args.tile = TILE
    mask_args = make_mask_args(args)

    matches = read_rows(args.matches_csv)
    if args.offset:
        matches = matches[args.offset :]
    if args.limit:
        matches = matches[: args.limit]
    manifest = load_manifest(args.zip_path, args.zip_root)
    pages = page_lookup(manifest)

    accepted = []
    rejected = 0
    with zipfile.ZipFile(args.zip_path) as archive:
        cache = {}
        for index, match in enumerate(matches, 1):
            page = match["page"]
            if page not in pages:
                rejected += 1
                continue
            if page not in cache:
                cache.clear()
                cache[page] = load_page_pair(archive, args.zip_root, pages[page])
            rough_page, line_page = cache[page]
            lx, ly = int(match["line_x"]), int(match["line_y"])
            rx, ry = int(match["rough_x"]), int(match["rough_y"])
            if (
                rx < 0 or ry < 0 or rx + TILE > rough_page.shape[1] or ry + TILE > rough_page.shape[0]
                or lx + TILE > line_page.shape[1] or ly + TILE > line_page.shape[0]
            ):
                rejected += 1
                continue
            rough_tile = rough_page[ry : ry + TILE, rx : rx + TILE]
            line_tile = line_page[ly : ly + TILE, lx : lx + TILE]

            mask_arr, _ = build_mask(Image.fromarray(rough_tile), Image.fromarray(line_tile), mask_args)

            region_stats = {
                "source_long_side": float(TILE),
                "src_per_out": 1.0,
                "src_px_per_tile": float(TILE),
                "source_origin_x": float(lx),
                "source_origin_y": float(ly),
            }
            stats = analyze_tile(rough_tile, line_tile, mask_arr, args, region_stats)
            if stats is None:
                rejected += 1
                continue
            name = f"{args.name_prefix}_{page}_{lx:04d}_{ly:04d}.jpg"
            accepted.append({
                **stats,
                "name": name,
                "page": page,
                "line_x": lx,
                "line_y": ly,
                "rough_x": rx,
                "rough_y": ry,
                "dx": match["dx"],
                "dy": match["dy"],
                "match_score": match["match_score"],
                "_rough": rough_tile,
                "_line": line_tile,
                "_mask": mask_arr,
            })
            if index % 100 == 0 or index == len(matches):
                print(f"matches: {index}/{len(matches)} accepted={len(accepted)}", flush=True)

    accepted.sort(key=lambda row: row["tile_score"], reverse=True)
    for rank, row in enumerate(accepted, 1):
        row["rank"] = rank
    write_csv(accepted, args.csv_out, append=args.append)
    make_qc(accepted, args.qc_out, args.qc_count)
    make_qc(accepted[-args.qc_count :], args.qc_tail_out, args.qc_count)
    if args.save:
        save_tiles(accepted, args.rough_out, args.line_out, args.mask_out, args.list_out, append=args.append)

    action = "saved" if args.save else "dry-run"
    print(f"{action}: matches={len(matches)} rejected={rejected} accepted={len(accepted)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}, {args.qc_tail_out}")
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
