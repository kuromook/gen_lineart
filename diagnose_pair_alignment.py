"""Diagnose whether paired rough/line pages support same-coordinate tiling.

This is a preflight check for new datasets. It samples same-coordinate tiles,
measures direct edge overlap, searches a local translation, and recommends one
of:

- same_coordinate_ok
- needs_global_or_local_alignment
- needs_region_matching
- hold_for_manual_review
"""

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ZIP_ROOT = "dataset_kurip_v4"
TILE = 480


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"{zip_root}/manifest.json"))


def page_label(entry):
    return Path(entry.get("file", entry["sketch"])).stem.replace("page", "")


def load_zip_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['line']}"))).convert("L")
    return rough, line


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


def edge_score(rough_edge, line_edge, tolerance=4):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0, 99.0
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), np.ones((tolerance * 2 + 1,) * 2, np.uint8)) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), np.ones((tolerance * 2 + 1,) * 2, np.uint8)) > 0
    precision = float((line_support[rough_edge]).mean())
    recall = float((rough_support[line_edge]).mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    chamfer = 0.5 * (
        float(np.minimum(d_to_line[rough_edge], 30).mean())
        + float(np.minimum(d_to_rough[line_edge], 30).mean())
    )
    return f1, chamfer


def shift_edge(line_edge, dx, dy):
    h, w = line_edge.shape
    shifted = np.zeros_like(line_edge)
    y0 = max(0, dy)
    y1 = min(h, h + dy)
    x0 = max(0, dx)
    x1 = min(w, w + dx)
    yy0 = max(0, -dy)
    yy1 = min(h, h - dy)
    xx0 = max(0, -dx)
    xx1 = min(w, w - dx)
    if y1 > y0 and x1 > x0:
        shifted[y0:y1, x0:x1] = line_edge[yy0:yy1, xx0:xx1]
    return shifted


def support_f1(rough_support, line_support, rough_edge, line_edge):
    precision = float((line_support[rough_edge]).mean()) if rough_edge.any() else 0.0
    recall = float((rough_support[line_edge]).mean()) if line_edge.any() else 0.0
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def best_shift(rough_edge, line_edge, max_shift, step):
    best = {"shift_f1": -1.0, "shift_chamfer": 99.0, "dx": 0, "dy": 0}
    kernel = np.ones((9, 9), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            shifted = shift_edge(line_edge, dx, dy)
            line_support = cv2.dilate(shifted.astype(np.uint8), kernel) > 0
            f1 = support_f1(rough_support, line_support, rough_edge, shifted)
            if f1 > best["shift_f1"]:
                best = {"shift_f1": f1, "shift_chamfer": 99.0, "dx": dx, "dy": dy}
    shifted = shift_edge(line_edge, best["dx"], best["dy"])
    _, best["shift_chamfer"] = edge_score(rough_edge, shifted)
    return best


def sample_rows(rough, line, page, args):
    rough = ImageOps.autocontrast(rough.convert("L"), cutoff=0)
    line = line.convert("L")
    width, height = rough.size
    if line.size != rough.size:
        width = min(width, line.size[0])
        height = min(height, line.size[1])
        rough = rough.crop((0, 0, width, height))
        line = line.crop((0, 0, width, height))
    rough_np = np.asarray(rough)
    line_np = np.asarray(line)
    rows = []
    for y in window_starts(height, args.tile, args.stride):
        for x in window_starts(width, args.tile, args.stride):
            rt = rough_np[y:y + args.tile, x:x + args.tile]
            lt = line_np[y:y + args.tile, x:x + args.tile]
            rough_std = float(rt.std())
            line_ink = float((lt < 128).mean())
            if rough_std < args.min_rough_std or not args.ink_min <= line_ink <= args.ink_max:
                continue
            re = edge_map(rt)
            le = edge_map(lt)
            if re.sum() < args.min_edge_pixels or le.sum() < args.min_edge_pixels:
                continue
            same_f1, same_chamfer = edge_score(re, le)
            shift = best_shift(re, le, args.max_shift, args.shift_step)
            rows.append({
                "page": page,
                "x": x,
                "y": y,
                "rough_std": rough_std,
                "line_ink": line_ink,
                "rough_edges": int(re.sum()),
                "line_edges": int(le.sum()),
                "same_f1": same_f1,
                "same_chamfer": same_chamfer,
                **shift,
            })
    rows.sort(key=lambda row: row["same_f1"], reverse=True)
    if args.max_tiles_per_page and len(rows) > args.max_tiles_per_page:
        rows = rows[:args.max_tiles_per_page]
    return rows


def summarize(rows):
    def percentile(key, p):
        vals = sorted(row[key] for row in rows)
        if not vals:
            return 0.0
        return float(vals[int((len(vals) - 1) * p)])

    same_median = percentile("same_f1", 0.5)
    same_good = sum(row["same_f1"] >= 0.35 for row in rows) / max(len(rows), 1)
    shift_median = percentile("shift_f1", 0.5)
    shift_good = sum(row["shift_f1"] >= 0.35 for row in rows) / max(len(rows), 1)
    median_gain = shift_median - same_median
    if same_median >= 0.35 and same_good >= 0.70:
        recommendation = "same_coordinate_ok"
    elif shift_median >= 0.35 and median_gain >= 0.08:
        recommendation = "needs_global_or_local_alignment"
    elif shift_good >= 0.20 or median_gain >= 0.05:
        recommendation = "needs_region_matching"
    else:
        recommendation = "hold_for_manual_review"
    return {
        "tile_count": len(rows),
        "same_f1_median": same_median,
        "same_f1_p90": percentile("same_f1", 0.9),
        "same_good_ratio": same_good,
        "shift_f1_median": shift_median,
        "shift_f1_p90": percentile("shift_f1", 0.9),
        "shift_good_ratio": shift_good,
        "median_shift_gain": median_gain,
        "recommendation": recommendation,
    }


def make_qc(rows, path, args):
    picks = sorted(rows, key=lambda row: row["shift_f1"], reverse=True)[: args.qc_count]
    if not picks:
        return
    thumb = 180
    label_h = 30
    canvas = Image.new("RGB", (thumb * 2, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    # Reload page cache lazily from saved tile paths is not available, so draw only labels here.
    # The detailed overlay QC is generated by diagnose_kurip_alignment_debug-style followups.
    for i, row in enumerate(picks):
        y = i * (thumb + label_h)
        draw.rectangle((0, y, thumb * 2 - 1, y + thumb - 1), outline=(200, 200, 200))
        text = (
            f'{row["page"]} x={row["x"]} y={row["y"]} '
            f'same={row["same_f1"]:.2f} shift={row["shift_f1"]:.2f} '
            f'dx={row["dx"]} dy={row["dy"]}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=os.path.expanduser("~/dataset_kurip_v4.zip"), dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--stride", type=int, default=480)
    parser.add_argument("--max-tiles-per-page", type=int, default=24)
    parser.add_argument("--min-rough-std", type=float, default=10.0)
    parser.add_argument("--ink-min", type=float, default=0.01)
    parser.add_argument("--ink-max", type=float, default=0.16)
    parser.add_argument("--min-edge-pixels", type=int, default=250)
    parser.add_argument("--max-shift", type=int, default=100)
    parser.add_argument("--shift-step", type=int, default=4)
    parser.add_argument("--csv-out", default="results/pair_alignment_diagnosis.csv")
    parser.add_argument("--json-out", default="results/pair_alignment_diagnosis.json")
    parser.add_argument("--qc-out", default="results/pair_alignment_diagnosis_qc.png")
    parser.add_argument("--qc-count", type=int, default=24)
    args = parser.parse_args()

    manifest = load_manifest(args.zip_path, args.zip_root)
    entries = manifest[: args.pages] if args.pages else manifest
    all_rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for index, entry in enumerate(entries, 1):
            rough, line = load_zip_pair(zf, args.zip_root, entry)
            rows = sample_rows(rough, line, page_label(entry), args)
            all_rows.extend(rows)
            print(f"{index}/{len(entries)} {page_label(entry)} tiles={len(rows)}")

    summary = summarize(all_rows)
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "page", "x", "y", "rough_std", "line_ink", "rough_edges", "line_edges",
        "same_f1", "same_chamfer", "shift_f1", "shift_chamfer", "dx", "dy",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(args.json_out, "w") as file:
        json.dump({"summary": summary, "args": vars(args)}, file, indent=2)
        file.write("\n")
    make_qc(all_rows, args.qc_out, args)
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.csv_out}, {args.json_out}, {args.qc_out}")


if __name__ == "__main__":
    main()
