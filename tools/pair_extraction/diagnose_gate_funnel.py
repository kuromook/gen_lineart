"""Diagnose which strict-filter gate is the yield bottleneck for a matches CSV.

Unlike filter_matched_region_tiles.py's analyze_tile(), which returns None as
soon as any gate fails (so a rejected tile's later-stage metrics are unknown),
this script evaluates every gate independently per tile and tabulates a
funnel: how many tiles fail at each stage, in gate-check order. Read-only /
diagnostic; does not save tiles.
"""

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_region_valid_masks import build_mask
from tile_region_manifest_480 import (
    TILE,
    chamfer,
    edge_map,
    line_width_stats,
    long_line_ratio,
    orientation_entropy,
    soft_ink_ratio,
    support_f1,
)
from filter_matched_region_tiles import load_manifest, load_page_pair, make_mask_args, page_lookup, read_rows


def evaluate(rough, line, mask, args):
    support = mask > 127
    support_ratio = float(support.mean())
    support_pixels = max(int(support.sum()), 1)
    line_ink = float(((line < 128) & support).sum() / support_pixels)
    ink_mask = (line < 128) & support
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8), 8)
    black_ratio = float(component_stats[1:, cv2.CC_STAT_AREA].max() / TILE / TILE) if component_count > 1 else 0.0
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.thick_ink_kernel, args.thick_ink_kernel))
    thick_ratio = float((cv2.erode(ink_mask.astype(np.uint8), thick_kernel) > 0).mean())
    rough_valid = rough[support]
    rough_std = float(rough_valid.std()) if rough_valid.size else 0.0
    rough_edge = edge_map(rough) & support
    line_edge = edge_map(line) & support
    rough_edges, line_edges = int(rough_edge.sum()), int(line_edge.sum())

    gates = {
        "support": support_ratio >= args.min_support,
        "ink_range": args.ink_min <= line_ink <= args.ink_max,
        "black_component": black_ratio <= args.max_black_component_ratio,
        "thick_ink": thick_ratio <= args.max_thick_ink_ratio,
        "rough_std": rough_std >= args.min_rough_std,
        "edge_pixels": rough_edges >= args.min_edge_pixels and line_edges >= args.min_edge_pixels,
    }
    result = {
        "support": support_ratio, "line_ink": line_ink, "black_ratio": black_ratio,
        "thick_ratio": thick_ratio, "rough_std": rough_std,
        "rough_edges": rough_edges, "line_edges": line_edges,
    }
    if rough_edges < 10 or line_edges < 10:
        gates.update({"entropy": False, "strict_recall": False, "strict_precision": False, "soft_ink": False, "width": False, "long_line": False})
        return gates, result

    entropy = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
    _, strict_precision, strict_recall = support_f1(rough_edge, line_edge, args.strict_close_px)
    width_p50, _ = line_width_stats(ink_mask)
    long_lines = long_line_ratio(ink_mask, TILE, args.long_line_fraction)
    soft_ink = soft_ink_ratio(line, support, ink_mask)

    gates.update({
        "entropy": entropy >= args.min_entropy,
        "strict_recall": strict_recall >= args.min_strict_line_recall,
        "strict_precision": strict_precision >= args.min_strict_rough_precision,
        "soft_ink": soft_ink <= args.max_soft_ink_ratio,
        "width": width_p50 <= args.max_line_width_p50,
        "long_line": long_lines <= args.max_long_line_ratio,
    })
    result.update({
        "entropy": entropy, "strict_recall": strict_recall, "strict_precision": strict_precision,
        "soft_ink": soft_ink, "width_p50": width_p50, "long_line": long_lines,
    })
    return gates, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-csv", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--zip-root", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--append", action="store_true")

    parser.add_argument("--mask-support-px", type=int, default=20)
    parser.add_argument("--mask-window", type=int, default=61)
    parser.add_argument("--mask-edge-density", type=float, default=0.030)
    parser.add_argument("--mask-expand-ignore", type=int, default=16)
    parser.add_argument("--mask-close-ignore", type=int, default=16)
    parser.add_argument("--mask-min-valid-ratio", type=float, default=0.35)

    parser.add_argument("--min-support", type=float, default=0.80)
    parser.add_argument("--ink-min", type=float, default=0.012)
    parser.add_argument("--ink-max", type=float, default=0.08)
    parser.add_argument("--max-black-component-ratio", type=float, default=0.025)
    parser.add_argument("--max-thick-ink-ratio", type=float, default=0.015)
    parser.add_argument("--thick-ink-kernel", type=int, default=9)
    parser.add_argument("--min-rough-std", type=float, default=8.0)
    parser.add_argument("--min-edge-pixels", type=int, default=120)
    parser.add_argument("--min-entropy", type=float, default=0.45)
    parser.add_argument("--strict-close-px", type=float, default=8.0)
    parser.add_argument("--min-strict-line-recall", type=float, default=0.55)
    parser.add_argument("--min-strict-rough-precision", type=float, default=0.15)
    parser.add_argument("--max-line-width-p50", type=float, default=6.0)
    parser.add_argument("--long-line-fraction", type=float, default=0.5)
    parser.add_argument("--max-long-line-ratio", type=float, default=0.25)
    parser.add_argument("--max-soft-ink-ratio", type=float, default=0.40)
    args = parser.parse_args()

    mask_args = make_mask_args(args)
    matches = read_rows(args.matches_csv)
    if args.offset:
        matches = matches[args.offset :]
    if args.limit:
        matches = matches[: args.limit]
    manifest = load_manifest(args.zip_path, args.zip_root)
    pages = page_lookup(manifest)

    fieldnames = [
        "page", "line_x", "line_y", "rough_x", "rough_y",
        "gate_support", "gate_ink_range", "gate_black_component", "gate_thick_ink",
        "gate_rough_std", "gate_edge_pixels", "gate_entropy", "gate_strict_recall",
        "gate_strict_precision", "gate_soft_ink", "gate_width", "gate_long_line",
        "all_pass",
        "support", "line_ink", "black_ratio", "thick_ratio", "rough_std",
        "rough_edges", "line_edges", "entropy", "strict_recall", "strict_precision",
        "soft_ink", "width_p50", "long_line",
    ]
    out_path = Path(args.out_csv)
    mode = "a" if args.append and out_path.exists() else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, mode, newline="")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    with zipfile.ZipFile(args.zip_path) as archive:
        cache = {}
        for index, match in enumerate(matches, 1):
            page = match["page"]
            if page not in pages:
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
                continue
            rough_tile = rough_page[ry : ry + TILE, rx : rx + TILE]
            line_tile = line_page[ly : ly + TILE, lx : lx + TILE]
            mask_arr, _ = build_mask(Image.fromarray(rough_tile), Image.fromarray(line_tile), mask_args)
            gates, result = evaluate(rough_tile, line_tile, mask_arr, args)
            row = {f"gate_{k}": v for k, v in gates.items()}
            row["all_pass"] = all(gates.values())
            row.update(result)
            row.update({"page": page, "line_x": lx, "line_y": ly, "rough_x": rx, "rough_y": ry})
            writer.writerow(row)
            if index % 100 == 0 or index == len(matches):
                print(f"{index}/{len(matches)}", flush=True)
    out_file.close()
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
