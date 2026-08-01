"""Extract fixed 480px tiles from a reviewed region manifest.

This is intended for reviewed square-padded masked region manifests such as
ako5ver2 keep281. Dry-run is the default; use --save after checking QC montages.

Two selection modes exist:

- `legacy`: the original ink/black-fill oriented filters and score.
- `strict`: adds source-scale gating, tight-tolerance edge correspondence, and
  stroke-likeness metrics, to reward local stroke correspondence and reject
  panel-scale composition crops.

## Alignment Gate vs Style Gate

`analyze_tile()` deliberately separates two kinds of rejection, per
`doc/preprocess/region_dataset_extraction_policy.md`:

- **Alignment gate** (`ALIGNMENT_*` constants below): whether the rough and
  line edges actually sit on top of each other (chamfer distance, edge
  correspondence at a fixed pixel tolerance). This is a purely geometric
  question, so the criteria are fixed constants shared by every source, not
  CLI arguments. Do not add a per-source override for these; if the constants
  need revisiting, that is a decision made once for all sources, deliberately,
  with evidence (see the policy doc's alignment section), not a per-dataset
  tuning knob.
- **Style/content gate** (the existing `--ink-*`, `--max-soft-ink-ratio`,
  `--max-line-width-p50`, etc. CLI arguments): stroke width, gray fringe, ink
  density, black-fill, panel-border rejection. These encode a source's drawing
  style and are expected to differ per dataset; keep tuning them per source.
"""

import argparse
import ast
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


TILE = 480

# Alignment gate: fixed, universal criteria for "rough and line agree on
# position." Native-scale calibrated (see doc/preprocess/raw_dataset_extraction_knowledge.md).
# Change these only as a single deliberate cross-dataset decision, never per source.
ALIGNMENT_CLOSE_PX = 22.0
ALIGNMENT_STRICT_CLOSE_PX = 8.0
ALIGNMENT_TRUNCATE_PX = 60.0
ALIGNMENT_MIN_F1 = 0.0
ALIGNMENT_MAX_CHAMFER = 200.0
ALIGNMENT_MIN_STRICT_F1 = 0.0
ALIGNMENT_MIN_STRICT_RECALL = 0.55
ALIGNMENT_MIN_STRICT_PRECISION = 0.15


def read_manifest(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def resolve_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return manifest_path.parent / path


def window_starts(limit, tile, stride):
    if limit < tile:
        return []
    starts = list(range(0, limit - tile + 1, stride))
    last = limit - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def orientation_entropy(edges):
    image = edges.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    valid = magnitude > 0.25
    if valid.sum() < 40:
        return 0.0
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=magnitude[valid])
    probabilities = hist / max(hist.sum(), 1e-9)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(12))


def support_f1(rough_edge, line_edge, close_px):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0, 0.0, 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px) * 2 + 1, int(close_px) * 2 + 1))
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, precision, recall


def chamfer(rough_edge, line_edge, truncate):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return float(truncate)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(
        (
            np.minimum(d_to_line[rough_edge], truncate).mean()
            + np.minimum(d_to_rough[line_edge], truncate).mean()
        )
        / 2
    )


def line_width_stats(ink_mask):
    """Stroke thickness from the distance transform ridge of the ink mask."""
    if ink_mask.sum() < 20:
        return 0.0, 0.0
    distance = cv2.distanceTransform(ink_mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[ink_mask]
    return float(2 * np.median(values)), float(2 * np.percentile(values, 95))


def long_line_ratio(ink_mask, tile, min_length_fraction):
    """Ink fraction explained by long straight segments, such as panel borders."""
    if ink_mask.sum() < 20:
        return 0.0
    segments = cv2.HoughLinesP(
        ink_mask.astype(np.uint8) * 255,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(tile * min_length_fraction),
        maxLineGap=6,
    )
    if segments is None:
        return 0.0
    canvas = np.zeros(ink_mask.shape, np.uint8)
    for x0, y0, x1, y1 in segments[:, 0]:
        cv2.line(canvas, (x0, y0), (x1, y1), 1, 3)
    return float(((canvas > 0) & ink_mask).sum() / max(int(ink_mask.sum()), 1))


def soft_ink_ratio(line, support, ink_mask):
    """Gray/antialiased share of the drawn area.

    Heavily downscaled composition crops turn solid strokes into gray mush, so a
    high soft share is a scale-abuse signal rather than real line art.
    """
    soft = (line >= 128) & (line < 220) & support
    drawn = int(ink_mask.sum()) + int(soft.sum())
    if drawn < 20:
        return 0.0
    return float(soft.sum() / drawn)


def legacy_tile_score(stats):
    return (
        4.0 * stats["edge_f1"]
        - 0.14 * stats["chamfer"]
        + 0.8 * stats["orientation_entropy"]
        + 0.2 * min(stats["rough_std"] / 20.0, 2.0)
        + 0.2 * min(stats["line_ink"] / 0.05, 2.0)
    )


def strict_tile_score(stats):
    """Reward tight local stroke correspondence on crisp stroke-scale targets.

    Source scale is handled by the `--min-src-per-out` / `--max-src-per-out`
    band gate rather than by the score. Measurement on keep281 showed raw edge
    correspondence rises with downscaling, because a shrunken page turns both
    rough and line into dense edge mush that matches everywhere, so scoring
    correspondence alone would rank page-scale composition crops highest.
    """
    return (
        4.0 * stats["strict_edge_f1"]
        + 2.0 * stats["strict_edge_recall"]
        + 0.6 * stats["orientation_entropy"]
        + 0.3 * min(stats["rough_std"] / 20.0, 2.0)
        - 2.0 * stats["long_line_ratio"]
        - 1.5 * stats["soft_ink_ratio"]
        - 0.25 * max(stats["line_width_p50"] - 3.0, 0.0)
    )


def tile_score(stats, mode):
    return strict_tile_score(stats) if mode == "strict" else legacy_tile_score(stats)


def alignment_metrics(rough_edge, line_edge):
    """Geometric agreement between rough and line edges. Universal: same
    formula and same fixed tolerances (`ALIGNMENT_*`) for every source.
    """
    f1, precision, recall = support_f1(rough_edge, line_edge, ALIGNMENT_CLOSE_PX)
    ch = chamfer(rough_edge, line_edge, ALIGNMENT_TRUNCATE_PX)
    strict_f1, strict_precision, strict_recall = support_f1(rough_edge, line_edge, ALIGNMENT_STRICT_CLOSE_PX)
    return {
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "chamfer": ch,
        "strict_edge_f1": strict_f1,
        "strict_edge_precision": strict_precision,
        "strict_edge_recall": strict_recall,
    }


def alignment_gate_pass(metrics):
    """Fixed, cross-dataset pass/fail on the geometric agreement alone."""
    return (
        metrics["edge_f1"] >= ALIGNMENT_MIN_F1
        and metrics["chamfer"] <= ALIGNMENT_MAX_CHAMFER
        and metrics["strict_edge_f1"] >= ALIGNMENT_MIN_STRICT_F1
        and metrics["strict_edge_recall"] >= ALIGNMENT_MIN_STRICT_RECALL
        and metrics["strict_edge_precision"] >= ALIGNMENT_MIN_STRICT_PRECISION
    )


def style_metrics(rough, line, support, ink_mask, args):
    """Stroke-appearance statistics. Per-source: every threshold here comes
    from `args`, expected to be tuned per dataset.
    """
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8), 8)
    if component_count > 1:
        largest_black_component_ratio = float(component_stats[1:, cv2.CC_STAT_AREA].max() / args.tile / args.tile)
    else:
        largest_black_component_ratio = 0.0
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.thick_ink_kernel, args.thick_ink_kernel))
    thick_ink_ratio = float((cv2.erode(ink_mask.astype(np.uint8), thick_kernel) > 0).mean())
    rough_valid = rough[support]
    rough_std = float(rough_valid.std()) if rough_valid.size else 0.0
    width_p50, width_p95 = line_width_stats(ink_mask)
    long_lines = long_line_ratio(ink_mask, args.tile, args.long_line_fraction)
    soft_ink = soft_ink_ratio(line, support, ink_mask)
    return {
        "largest_black_component_ratio": largest_black_component_ratio,
        "thick_ink_ratio": thick_ink_ratio,
        "rough_std": rough_std,
        "line_width_p50": width_p50,
        "line_width_p95": width_p95,
        "long_line_ratio": long_lines,
        "soft_ink_ratio": soft_ink,
    }


def style_gate_pass(metrics, args):
    """Per-source pass/fail on stroke appearance. Tune `args` per dataset."""
    return (
        metrics["largest_black_component_ratio"] <= args.max_black_component_ratio
        and metrics["thick_ink_ratio"] <= args.max_thick_ink_ratio
        and metrics["rough_std"] >= args.min_rough_std
        and metrics["long_line_ratio"] <= args.max_long_line_ratio
        and metrics["soft_ink_ratio"] <= args.max_soft_ink_ratio
        and (not args.max_line_width_p50 or metrics["line_width_p50"] <= args.max_line_width_p50)
    )


def analyze_tile(rough, line, mask, args, region_stats):
    support = mask > 127
    support_ratio = float(support.mean())
    if support_ratio < args.min_support:
        return None
    support_pixels = max(int(support.sum()), 1)
    line_ink = float(((line < 128) & support).sum() / support_pixels)
    if not args.ink_min <= line_ink <= args.ink_max:
        return None
    ink_mask = (line < 128) & support

    style = style_metrics(rough, line, support, ink_mask, args)
    if not style_gate_pass(style, args):
        return None

    rough_edge = edge_map(rough) & support
    line_edge = edge_map(line) & support
    rough_edges = int(rough_edge.sum())
    line_edges = int(line_edge.sum())
    if rough_edges < args.min_edge_pixels or line_edges < args.min_edge_pixels:
        return None

    alignment = alignment_metrics(rough_edge, line_edge)
    entropy = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
    if entropy < args.min_entropy:
        return None
    if not alignment_gate_pass(alignment):
        return None

    stats = {
        "support": support_ratio,
        "line_ink": line_ink,
        "rough_edges": rough_edges,
        "line_edges": line_edges,
        "orientation_entropy": entropy,
        **alignment,
        **style,
        **region_stats,
    }
    stats["tile_score"] = tile_score(stats, args.score_mode)
    if stats["tile_score"] < args.min_tile_score:
        return None
    return stats


def region_scale_stats(row, tile):
    """Source-resolution context for one manifest region.

    `src_per_out` is source pixels per normalized output pixel, so
    `src_px_per_tile` is how much of the original manuscript a single tile covers.
    Large values mean the tile is a downscaled panel-scale composition crop.
    """
    long_side = 0.0
    for key in ("native_long_side", "materialized_long_side", "mask_image_size"):
        try:
            long_side = float(row.get(key) or 0)
        except ValueError:
            long_side = 0.0
        if long_side > 0:
            break
    long_side = long_side or float(tile)
    try:
        box = ast.literal_eval(row["line_box"])
        source_long_side = float(max(box[2] - box[0], box[3] - box[1]))
    except (KeyError, ValueError, SyntaxError, TypeError, IndexError):
        source_long_side = long_side
    src_per_out = source_long_side / max(long_side, 1.0)
    try:
        box = ast.literal_eval(row["line_box"])
        origin = (float(box[0]), float(box[1]))
    except (KeyError, ValueError, SyntaxError, TypeError, IndexError):
        origin = (0.0, 0.0)
    return {
        "source_long_side": source_long_side,
        "src_per_out": src_per_out,
        "src_px_per_tile": src_per_out * tile,
        "source_origin_x": origin[0],
        "source_origin_y": origin[1],
    }


def region_rejection(row, scale, args):
    """Return a reason string when a whole manifest region should be skipped."""
    if args.max_src_per_out and scale["src_per_out"] > args.max_src_per_out:
        return "src_per_out_high"
    if args.min_src_per_out and scale["src_per_out"] < args.min_src_per_out:
        return "src_per_out_low"
    if args.exclude_feature_tags:
        excluded = {tag.strip() for tag in args.exclude_feature_tags.split(",") if tag.strip()}
        tags = {tag.strip() for tag in (row.get("feature_tags") or "").split(";") if tag.strip()}
        if tags & excluded:
            return "feature_tag"
    if args.min_align_f1:
        try:
            if float(row.get("align_f1") or 0.0) < args.min_align_f1:
                return "align_f1_low"
        except ValueError:
            return "align_f1_missing"
    if args.max_unsupported_line_edge_ratio < 1.0:
        try:
            if float(row.get("unsupported_line_edge_ratio") or 0.0) > args.max_unsupported_line_edge_ratio:
                return "unsupported_line_edge_high"
        except ValueError:
            return "unsupported_line_edge_missing"
    return ""


def overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return intersection / min(area_a, area_b)


def deduplicate(rows, overlap, max_per_region, scope):
    """Drop overlapping tiles, keeping the highest scoring one.

    `scope="page"` compares tiles in source-page coordinates, so overlapping
    parent and child regions of the same page cannot contribute the same content
    twice. `scope="region"` keeps the original per-region behaviour.
    """
    kept = []
    per_region = {}
    by_group = {}
    for row in sorted(rows, key=lambda item: item["tile_score"], reverse=True):
        if scope == "page":
            group = row["source_page"]
            box = row["page_bbox"]
        else:
            group = row["source_review_index"]
            box = row["tile_bbox"]
        region = row["source_review_index"]
        if max_per_region and per_region.get(region, 0) >= max_per_region:
            continue
        neighbours = by_group.setdefault(group, [])
        if any(overlap_ratio(box, other) >= overlap for other in neighbours):
            continue
        neighbours.append(box)
        per_region[region] = per_region.get(region, 0) + 1
        kept.append(row)
    return kept


def load_region_images(row, args, manifest_path):
    rough = np.asarray(
        ImageOps.autocontrast(Image.open(resolve_path(row[args.rough_key], manifest_path)).convert("L"), cutoff=0)
    )
    line = np.asarray(Image.open(resolve_path(row[args.line_key], manifest_path)).convert("L"))
    mask = np.asarray(Image.open(resolve_path(row[args.mask_key], manifest_path)).convert("L"))
    return rough, line, mask


def with_tiles(rows, manifest_rows, args, manifest_path):
    """Yield (row, rough, line, mask) re-cropping on demand.

    Native-resolution regions produce far too many candidates to keep every tile
    in memory, so tiles are grouped by source region and re-read when needed.
    """
    by_region = {}
    for row in rows:
        by_region.setdefault(row["source_row_index"], []).append(row)
    for region_index in sorted(by_region):
        source = manifest_rows[region_index - 1]
        rough, line, mask = load_region_images(source, args, manifest_path)
        for row in by_region[region_index]:
            x, y, tile = row["x"], row["y"], args.tile
            yield (
                row,
                rough[y : y + tile, x : x + tile],
                line[y : y + tile, x : x + tile],
                mask[y : y + tile, x : x + tile],
            )


def make_qc(rows, path, count, manifest_rows, args, manifest_path):
    picks = rows[:count]
    if not picks:
        return
    tiles = {id(row): images for row, *images in with_tiles(picks, manifest_rows, args, manifest_path)}
    thumb, label_h = 240, 30
    canvas = Image.new("RGB", (thumb * 4, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(picks):
        y = index * (thumb + label_h)
        rough_tile, line_tile, mask_tile = tiles[id(row)]
        overlay = np.full((TILE, TILE, 3), 255, np.uint8)
        overlay[edge_map(rough_tile)] = (255, 60, 60)
        overlay[edge_map(line_tile)] = (40, 80, 255)
        mask_vis = np.where(mask_tile > 127, line_tile, 220).astype(np.uint8)
        for column, image in enumerate((rough_tile, line_tile, mask_vis, overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (column * thumb, y))
        text = (
            f'{row["name"]} score={row["tile_score"]:.2f} F1={row["edge_f1"]:.2f} '
            f'sF1={row["strict_edge_f1"]:.2f} sRec={row["strict_edge_recall"]:.2f} '
            f'cham={row["chamfer"]:.1f} ink={row["line_ink"]:.3f} w50={row["line_width_p50"]:.1f} '
            f'soft={row["soft_ink_ratio"]:.2f} long={row["long_line_ratio"]:.2f} '
            f'src/out={row["src_per_out"]:.2f} support={row["support"]:.2f}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evenly_spaced(rows, count):
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def strip_images(row):
    return {key: value for key, value in row.items() if key not in {"rough", "line", "mask"}}


def save_tiles_streaming(rows, rough_out, line_out, list_out, manifest_rows, args, manifest_path):
    rough_out, line_out = Path(rough_out), Path(line_out)
    rough_out.mkdir(parents=True, exist_ok=True)
    line_out.mkdir(parents=True, exist_ok=True)
    written = 0
    for row, rough_tile, line_tile, _ in with_tiles(rows, manifest_rows, args, manifest_path):
        Image.fromarray(rough_tile).save(rough_out / row["name"], quality=95)
        Image.fromarray(line_tile).save(line_out / row["name"], quality=95)
        written += 1
    Path(list_out).parent.mkdir(parents=True, exist_ok=True)
    with open(list_out, "w") as file:
        file.write("\n".join(row["name"] for row in rows) + "\n")
    return written


def write_csv(rows, path):
    fields = [
        "rank",
        "name",
        "source_review_index",
        "source_row_index",
        "source_page",
        "source_name",
        "source_rank",
        "x",
        "y",
        "tile_bbox",
        "page_bbox",
        "tile_score",
        "edge_f1",
        "edge_precision",
        "edge_recall",
        "chamfer",
        "strict_edge_f1",
        "strict_edge_precision",
        "strict_edge_recall",
        "orientation_entropy",
        "rough_std",
        "line_ink",
        "line_width_p50",
        "line_width_p95",
        "long_line_ratio",
        "soft_ink_ratio",
        "src_per_out",
        "src_px_per_tile",
        "source_long_side",
        "largest_black_component_ratio",
        "thick_ink_ratio",
        "support",
        "rough_edges",
        "line_edges",
        "decision",
        "notes",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**strip_images(row), "decision": "", "notes": ""})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rough-key", default="masked_rough_path")
    parser.add_argument("--line-key", default="masked_line_path")
    parser.add_argument("--mask-key", default="valid_mask_path")
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--stride", type=int, default=144)
    parser.add_argument("--min-support", type=float, default=0.85)
    parser.add_argument("--ink-min", type=float, default=0.005)
    parser.add_argument("--ink-max", type=float, default=0.18)
    parser.add_argument("--max-black-component-ratio", type=float, default=1.0)
    parser.add_argument("--max-thick-ink-ratio", type=float, default=1.0)
    parser.add_argument("--thick-ink-kernel", type=int, default=9)
    parser.add_argument("--min-rough-std", type=float, default=8.0)
    parser.add_argument("--min-edge-pixels", type=int, default=120)
    parser.add_argument("--min-entropy", type=float, default=0.25)
    parser.add_argument("--min-tile-score", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=("legacy", "strict"), default="legacy")
    parser.add_argument("--max-line-width-p50", type=float, default=0.0)
    parser.add_argument("--long-line-fraction", type=float, default=0.5)
    parser.add_argument("--max-long-line-ratio", type=float, default=1.0)
    parser.add_argument("--max-soft-ink-ratio", type=float, default=1.0)
    parser.add_argument("--max-src-per-out", type=float, default=0.0)
    parser.add_argument("--min-src-per-out", type=float, default=0.0)
    parser.add_argument("--exclude-feature-tags", default="")
    parser.add_argument("--min-align-f1", type=float, default=0.0)
    parser.add_argument("--max-unsupported-line-edge-ratio", type=float, default=1.0)
    parser.add_argument("--duplicate-overlap", type=float, default=0.60)
    parser.add_argument("--dedup-scope", choices=("region", "page"), default="region")
    parser.add_argument("--max-per-region", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--name-prefix", default="ako5k281")
    parser.add_argument("--csv-out", default="results/ako5ver2_keep281_tiles_480.csv")
    parser.add_argument("--qc-out", default="results/ako5ver2_keep281_tiles_480_qc.png")
    parser.add_argument("--qc-tail-out", default="results/ako5ver2_keep281_tiles_480_qc_tail.png")
    parser.add_argument("--qc-sample-out", default="results/ako5ver2_keep281_tiles_480_qc_sample.png")
    parser.add_argument("--rough-out", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-out", default="dataset/pairs_480/train/line_ako5ver2_keep281_tiles_20260725")
    parser.add_argument("--list-out", default="dataset/pairs_480/valid_train_ako5ver2_keep281_tiles_20260725.txt")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if args.tile != TILE:
        raise ValueError("Only 480px output is currently supported")

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    candidates = []
    region_rejects = {}
    used_regions = 0
    for row_index, row in enumerate(rows, 1):
        scale = region_scale_stats(row, args.tile)
        reason = region_rejection(row, scale, args)
        if reason:
            region_rejects[reason] = region_rejects.get(reason, 0) + 1
            continue
        used_regions += 1
        rough, line, mask = load_region_images(row, args, manifest_path)
        if rough.shape != line.shape or rough.shape != mask.shape:
            raise ValueError(f"shape mismatch at row {row_index}: {rough.shape} {line.shape} {mask.shape}")
        height, width = rough.shape
        for y in window_starts(height, args.tile, args.stride):
            for x in window_starts(width, args.tile, args.stride):
                rough_tile = rough[y : y + args.tile, x : x + args.tile]
                line_tile = line[y : y + args.tile, x : x + args.tile]
                mask_tile = mask[y : y + args.tile, x : x + args.tile]
                stats = analyze_tile(rough_tile, line_tile, mask_tile, args, scale)
                if stats is None:
                    continue
                review_index = row.get("_review_index") or str(row_index)
                name = f"{args.name_prefix}_{int(review_index):04d}_{x:04d}_{y:04d}.jpg"
                page_bbox = (
                    round(scale["source_origin_x"] + x * scale["src_per_out"]),
                    round(scale["source_origin_y"] + y * scale["src_per_out"]),
                    round(scale["source_origin_x"] + (x + args.tile) * scale["src_per_out"]),
                    round(scale["source_origin_y"] + (y + args.tile) * scale["src_per_out"]),
                )
                candidates.append(
                    {
                        **stats,
                        "name": name,
                        "source_review_index": review_index,
                        "source_row_index": row_index,
                        "source_page": row.get("page") or review_index,
                        "source_name": row.get("masked_name") or row.get("name") or "",
                        "source_rank": row.get("rank") or "",
                        "x": x,
                        "y": y,
                        "tile_bbox": (x, y, x + args.tile, y + args.tile),
                        "page_bbox": page_bbox,
                    }
                )
        if row_index % 50 == 0 or row_index == len(rows):
            print(
                f"regions: {row_index}/{len(rows)} used={used_regions} candidates={len(candidates)}",
                flush=True,
            )

    accepted = deduplicate(candidates, args.duplicate_overlap, args.max_per_region, args.dedup_scope)
    if args.limit:
        accepted = accepted[: args.limit]
    for rank, row in enumerate(accepted, 1):
        row["rank"] = rank
    write_csv(accepted, args.csv_out)
    qc_args = (rows, args, manifest_path)
    make_qc(accepted, args.qc_out, args.qc_count, *qc_args)
    make_qc(accepted[-args.qc_count :], args.qc_tail_out, args.qc_count, *qc_args)
    make_qc(evenly_spaced(accepted, args.qc_count), args.qc_sample_out, args.qc_count, *qc_args)
    if args.save:
        save_tiles_streaming(accepted, args.rough_out, args.line_out, args.list_out, rows, args, manifest_path)

    action = "saved" if args.save else "dry-run"
    print(
        f"{action}: regions={len(rows)} used_regions={used_regions} "
        f"raw_tiles={len(candidates)} accepted={len(accepted)}"
    )
    if region_rejects:
        summary = " ".join(f"{key}={value}" for key, value in sorted(region_rejects.items()))
        print(f"region rejects: {summary}")
    print(f"wrote: {args.csv_out}, {args.qc_out}, {args.qc_tail_out}, {args.qc_sample_out}")
    if args.save:
        print(f"training list: {args.list_out}")


if __name__ == "__main__":
    main()
