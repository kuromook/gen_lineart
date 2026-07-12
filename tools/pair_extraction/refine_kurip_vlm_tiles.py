"""Refine kurip VLM-accepted tile pairs by re-aligning rough crops to fixed line crops."""

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


ZIP_PATH = os.path.expanduser("~/dataset_kurip_v4.zip")
ZIP_ROOT = "dataset_kurip_v4"
REVIEW_CSV = "results/kurip_vlm_candidates_top500_review_qwen3vl.csv"
ROUGH_OUT = "dataset/pairs_480/train/rough"
LINE_OUT = "dataset/pairs_480/train/line_kurip_vlm_accept_refined_clean_t192_cc8"
LIST_OUT = "dataset/pairs_480/valid_train_kurip_vlm_accept_refined.txt"
CSV_OUT = "results/kurip_vlm_accept_refined_tiles.csv"
QC_OUT = "results/kurip_vlm_accept_refined_qc.png"
TILE = 480


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"{zip_root}/manifest.json"))


def page_id(entry):
    return Path(entry.get("file", entry["sketch"])).stem.replace("page", "")


def read_accepts(path, limit):
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            if row.get("vlm_decision") != "accept":
                continue
            rows.append({
                **row,
                "rank": int(row["rank"]),
                "line_x": int(row["line_x"]),
                "line_y": int(row["line_y"]),
                "rough_x": int(row["rough_x"]),
                "rough_y": int(row["rough_y"]),
                "dx": int(row["dx"]),
                "dy": int(row["dy"]),
                "line_ink": float(row["line_ink"]),
                "edge_f1": float(row["edge_f1"]),
                "chamfer": float(row["chamfer"]),
                "match_score": float(row["match_score"]),
            })
    return rows[:limit] if limit else rows


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['sketch']}"))).convert("L")
    line = Image.open(io.BytesIO(zf.read(f"{zip_root}/{entry['line']}"))).convert("L")
    return np.asarray(rough), np.asarray(line)


def crop(image, x, y, tile):
    h, w = image.shape
    if x < 0 or y < 0 or x + tile > w or y + tile > h:
        return None
    return image[y:y + tile, x:x + tile]


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def clean_line(line, threshold, min_component_area):
    binary = (line < threshold).astype(np.uint8)
    if min_component_area > 1 and binary.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = np.zeros_like(binary, dtype=bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                keep |= labels == label
        binary = keep.astype(np.uint8)
    return np.where(binary > 0, 0, 255).astype(np.uint8)


def support_f1(rough_edge, line_edge, rough_support, line_support):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 0.0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def chamfer(rough_edge, line_edge, truncate):
    if rough_edge.sum() < 10 or line_edge.sum() < 10:
        return 99.0
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def iter_offsets(center_x, center_y, radius, step):
    offsets = []
    for y in range(center_y - radius, center_y + radius + 1, step):
        for x in range(center_x - radius, center_x + radius + 1, step):
            offsets.append((x, y))
    offsets.sort(key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
    return offsets


def best_offset(rough_edges, rough_support_full, line_edge, line_support, rough_x, rough_y, args):
    best = None
    for x, y in iter_offsets(rough_x, rough_y, args.search_radius, args.coarse_step):
        rough_edge = crop(rough_edges, x, y, args.tile)
        if rough_edge is None:
            continue
        rough_support = crop(rough_support_full, x, y, args.tile)
        f1 = support_f1(rough_edge, line_edge, rough_support, line_support)
        if best is None or f1 > best["f1"]:
            best = {"x": x, "y": y, "f1": f1}
    if best is None:
        return None

    refined = None
    for x, y in iter_offsets(best["x"], best["y"], args.fine_radius, args.fine_step):
        rough_edge = crop(rough_edges, x, y, args.tile)
        if rough_edge is None:
            continue
        rough_support = crop(rough_support_full, x, y, args.tile)
        f1 = support_f1(rough_edge, line_edge, rough_support, line_support)
        if refined is None or f1 > refined["f1"]:
            refined = {"x": x, "y": y, "f1": f1, "rough_edge": rough_edge}
    return refined


def refine(rows, manifest, args):
    entries = {page_id(entry): entry for entry in manifest}
    out = []
    with zipfile.ZipFile(args.zip_path) as zf:
        cache = {}
        for index, row in enumerate(rows, 1):
            if row["page"] not in cache:
                rough, line = load_pair(zf, args.zip_root, entries[row["page"]])
                match_rough = ImageOps.autocontrast(Image.fromarray(rough), cutoff=0)
                rough_edges = edge_map(np.asarray(match_rough))
                kernel = np.ones((args.close_px * 2 + 1, args.close_px * 2 + 1), np.uint8)
                cache[row["page"]] = {
                    "rough": rough,
                    "line": line,
                    "rough_edges": rough_edges,
                    "rough_support": cv2.dilate(rough_edges.astype(np.uint8), kernel) > 0,
                    "line_edges": edge_map(line),
                }
            page = cache[row["page"]]
            line = crop(page["line"], row["line_x"], row["line_y"], args.tile)
            line_edge = crop(page["line_edges"], row["line_x"], row["line_y"], args.tile)
            if line is None or line_edge is None:
                continue
            kernel = np.ones((args.close_px * 2 + 1, args.close_px * 2 + 1), np.uint8)
            line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
            best = best_offset(
                page["rough_edges"],
                page["rough_support"],
                line_edge,
                line_support,
                row["rough_x"],
                row["rough_y"],
                args,
            )
            if best is None:
                continue
            rough = crop(page["rough"], best["x"], best["y"], args.tile)
            if rough is None:
                continue
            c = chamfer(best["rough_edge"], line_edge, args.truncate_px)
            shift = abs(best["x"] - row["rough_x"]) + abs(best["y"] - row["rough_y"])
            if best["f1"] < args.min_f1 or c > args.max_chamfer:
                continue
            line_clean = clean_line(line, args.line_threshold, args.min_component_area)
            name = (
                f'kuripr_{row["page"]}_l{row["line_x"]:04d}_{row["line_y"]:04d}_'
                f'r{best["x"]:04d}_{best["y"]:04d}.jpg'
            )
            out.append({
                **row,
                "name": name,
                "old_rough_x": row["rough_x"],
                "old_rough_y": row["rough_y"],
                "rough_x": best["x"],
                "rough_y": best["y"],
                "dx": best["x"] - row["line_x"],
                "dy": best["y"] - row["line_y"],
                "refine_shift_l1": shift,
                "refined_edge_f1": best["f1"],
                "refined_chamfer": c,
                "rough": rough,
                "line": line_clean,
                "rough_edge": best["rough_edge"],
                "line_edge": line_edge,
            })
            if index % 50 == 0:
                print(f"{index}/{len(rows)} accepted={len(out)}", flush=True)
    out.sort(key=lambda r: (r["refined_edge_f1"], -r["refined_chamfer"]), reverse=True)
    return out


def write_csv(rows, path):
    fields = [
        "rank", "name", "page", "source_page", "line_x", "line_y",
        "old_rough_x", "old_rough_y", "rough_x", "rough_y", "dx", "dy",
        "refine_shift_l1", "line_ink", "edge_f1", "chamfer", "match_score",
        "refined_edge_f1", "refined_chamfer", "vlm_score", "vlm_reason",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})


def make_qc(rows, path, count, tile):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 240, 34
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for idx, row in enumerate(picks):
        top = idx * (thumb + label_h)
        overlay = np.full((tile, tile, 3), 255, np.uint8)
        overlay[row["rough_edge"]] = (255, 60, 60)
        overlay[row["line_edge"]] = (40, 80, 255)
        for col, image in enumerate((row["rough"], row["line"], overlay)):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (col * thumb, top))
        text = (
            f'{row["name"]} F1 {row["edge_f1"]:.2f}->{row["refined_edge_f1"]:.2f} '
            f'ch {row["chamfer"]:.1f}->{row["refined_chamfer"]:.1f} '
            f'shift={row["refine_shift_l1"]}'
        )
        draw.text((3, top + thumb + 2), text, fill="black", font=font)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)


def save_tiles(rows, args):
    os.makedirs(args.rough_out, exist_ok=True)
    os.makedirs(args.line_out, exist_ok=True)
    for row in rows:
        Image.fromarray(row["rough"]).save(os.path.join(args.rough_out, row["name"]), quality=95)
        Image.fromarray(row["line"]).save(os.path.join(args.line_out, row["name"]), quality=95)
    Path(args.list_out).write_text("\n".join(row["name"] for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--review-csv", default=REVIEW_CSV)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--search-radius", type=int, default=64)
    parser.add_argument("--coarse-step", type=int, default=8)
    parser.add_argument("--fine-radius", type=int, default=8)
    parser.add_argument("--fine-step", type=int, default=2)
    parser.add_argument("--close-px", type=int, default=3)
    parser.add_argument("--truncate-px", type=int, default=30)
    parser.add_argument("--min-f1", type=float, default=0.50)
    parser.add_argument("--max-chamfer", type=float, default=8.0)
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--rough-out", default=ROUGH_OUT)
    parser.add_argument("--line-out", default=LINE_OUT)
    parser.add_argument("--list-out", default=LIST_OUT)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.tile != TILE:
        raise ValueError("Only 480px output is currently supported")
    rows = read_accepts(args.review_csv, args.limit)
    manifest = load_manifest(args.zip_path, args.zip_root)
    refined = refine(rows, manifest, args)
    write_csv(refined, args.csv_out)
    make_qc(refined, args.qc_out, args.qc_count, args.tile)
    if args.save:
        save_tiles(refined, args)
    print(f"input_accepts={len(rows)} refined={len(refined)}")
    print(f"wrote: {args.csv_out}, {args.qc_out}")
    if args.save:
        print(f"saved: {args.list_out}, {args.rough_out}, {args.line_out}")


if __name__ == "__main__":
    main()
