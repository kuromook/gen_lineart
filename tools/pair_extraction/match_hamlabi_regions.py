"""Generate content-matched hamlabi rough/line region candidates.

This is deliberately not a same-XY tile extractor. The line page provides
semantic region proposals, then the rough page is searched over nearby
translation and scale candidates. Output is a review manifest and QC montage;
no training list is written by default.
"""

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ZIP = "dataset_hamlabi.zip"
DEFAULT_ZIP_ROOT = "dataset_hamlabi"
DEFAULT_CSV = "results/hamlabi_region_candidates.csv"
DEFAULT_JSON = "results/hamlabi_region_candidates.json"
DEFAULT_QC = "results/hamlabi_region_candidates_qc.png"


def read_zip_member(zf, root, name):
    for candidate in (f"{root}/{name}", f"{root}\\{name}", name):
        try:
            return zf.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(read_zip_member(zf, zip_root, "manifest.json"))


def page_id(entry):
    return Path(entry.get("file", entry["line"])).stem.replace("page", "")


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["line"]))).convert("L")
    rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def line_ink_mask(line, threshold, min_component_area):
    mask = (line < threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label] = 1
    return cleaned


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def orientation_entropy(edges):
    if edges.sum() < 40:
        return 0.0
    image = edges.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    valid = mag > 0.25
    if valid.sum() < 40:
        return 0.0
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=mag[valid])
    probs = hist / max(hist.sum(), 1e-9)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / np.log(12))


def support_f1(rough_edge, line_edge, tolerance):
    if rough_edge.sum() < 20 or line_edge.sum() < 20:
        return 0.0, 0.0, 0.0
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[rough_edge].mean())
    recall = float(rough_support[line_edge].mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return f1, precision, recall


def chamfer(rough_edge, line_edge, truncate):
    if rough_edge.sum() < 20 or line_edge.sum() < 20:
        return 99.0
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def projection_corr(a_edge, b_edge):
    if a_edge.sum() < 20 or b_edge.sum() < 20:
        return 0.0
    scores = []
    for axis in (0, 1):
        a = a_edge.sum(axis=axis).astype(np.float32)
        b = b_edge.sum(axis=axis).astype(np.float32)
        if a.std() < 1e-6 or b.std() < 1e-6:
            scores.append(0.0)
        else:
            scores.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.nan_to_num(np.mean(scores), nan=0.0))


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def expand_box(box, margin, width, height):
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    pad = max(bw, bh) * margin
    return clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height)


def crop_resize(image, box, target_size):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return np.asarray(Image.fromarray(crop).resize(target_size, Image.Resampling.LANCZOS))


def region_proposals(line, args):
    h, w = line.shape
    mask = line_ink_mask(line, args.line_threshold, args.min_line_component_area)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.region_close, args.region_close))
    grouped = cv2.dilate(mask, close_kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    rows = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < args.min_region_area or bw < args.min_region_size or bh < args.min_region_size:
            continue
        if bw > args.max_region_ratio * w or bh > args.max_region_ratio * h:
            continue
        box = expand_box((x, y, x + bw, y + bh), args.line_margin, w, h)
        bx0, by0, bx1, by1 = box
        line_crop = line[by0:by1, bx0:bx1]
        line_ink = float((line_crop < args.line_threshold).mean())
        if line_ink < args.min_line_ink or line_ink > args.max_line_ink:
            continue
        rows.append({
            "line_box": box,
            "region_area": area,
            "line_ink": line_ink,
        })
    rows.sort(key=lambda item: (item["line_box"][2] - item["line_box"][0]) * (item["line_box"][3] - item["line_box"][1]), reverse=True)
    return rows[: args.max_regions_per_page] if args.max_regions_per_page else rows


def rough_candidate_boxes(line_box, rough_shape, args):
    h, w = rough_shape
    x0, y0, x1, y1 = line_box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0
    step = max(args.search_step, int(round(max(bw, bh) * args.search_step_ratio)))
    search = max(args.search_px, int(round(max(bw, bh) * args.search_ratio)))
    offsets = []
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    for scale in args.scales:
        sw = bw * scale
        sh = bh * scale
        for dx, dy in offsets:
            box = clamp_box((cx + dx - sw / 2, cy + dy - sh / 2, cx + dx + sw / 2, cy + dy + sh / 2), w, h)
            if box[2] - box[0] < args.min_region_size or box[3] - box[1] < args.min_region_size:
                continue
            yield scale, dx, dy, box


def match_region(rough, line, proposal, args):
    line_box = proposal["line_box"]
    lw = line_box[2] - line_box[0]
    lh = line_box[3] - line_box[1]
    target_size = (args.match_size, max(16, int(round(args.match_size * lh / lw)))) if lw >= lh else (
        max(16, int(round(args.match_size * lw / lh))), args.match_size,
    )
    line_norm = crop_resize(line, line_box, target_size)
    if line_norm is None:
        return None
    line_edge = edge_map(line_norm)
    if line_edge.sum() < args.min_match_edges:
        return None
    best = None
    for scale, dx, dy, rough_box in rough_candidate_boxes(line_box, rough.shape, args):
        rough_norm = crop_resize(rough, rough_box, target_size)
        if rough_norm is None:
            continue
        rough_edge = edge_map(rough_norm)
        if rough_edge.sum() < args.min_match_edges:
            continue
        f1, precision, recall = support_f1(rough_edge, line_edge, args.close_px)
        ch = chamfer(rough_edge, line_edge, args.truncate_px)
        pcorr = projection_corr(rough_edge, line_edge)
        ent = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
        rough_std = float(rough_norm.std())
        score = 4.0 * f1 - 0.08 * ch + 0.6 * pcorr + 0.4 * ent + min(rough_std / 80.0, 0.4)
        row = {
            **proposal,
            "rough_box": rough_box,
            "scale": scale,
            "dx": dx,
            "dy": dy,
            "edge_f1": f1,
            "edge_precision": precision,
            "edge_recall": recall,
            "chamfer": ch,
            "projection_corr": pcorr,
            "orientation_entropy": ent,
            "rough_std": rough_std,
            "line_edges": int(line_edge.sum()),
            "rough_edges": int(rough_edge.sum()),
            "match_score": score,
            "line_norm": line_norm,
            "rough_norm": rough_norm,
            "line_edge": line_edge,
            "rough_edge": rough_edge,
        }
        if best is None or row["match_score"] > best["match_score"]:
            best = row
    return best


def normalize_pair(rough, line, rough_box, line_box, args):
    lw = line_box[2] - line_box[0]
    lh = line_box[3] - line_box[1]
    long_side = max(lw, lh)
    out_w = max(32, int(round(args.output_long_side * lw / long_side)))
    out_h = max(32, int(round(args.output_long_side * lh / long_side)))
    return (
        crop_resize(rough, rough_box, (out_w, out_h)),
        crop_resize(line, line_box, (out_w, out_h)),
        out_w,
        out_h,
    )


def feature_tags(row, args):
    tags = []
    line_box = row["line_box"]
    width = line_box[2] - line_box[0]
    height = line_box[3] - line_box[1]
    area = width * height
    if max(width, height) >= args.large_region_long_side or area >= args.large_region_area:
        tags.append("large_parent")
    if row["black_fill_ratio"] >= args.black_fill_ratio:
        tags.append("black_fill")
    if width / max(height, 1) >= args.wide_region_ratio:
        tags.append("wide_panel")
    if height / max(width, 1) >= args.tall_region_ratio:
        tags.append("tall_panel")
    return ";".join(tags)


def process_page(entry, rough, line, args):
    rows = []
    proposals = region_proposals(line, args)
    page = page_id(entry)
    for index, proposal in enumerate(proposals, 1):
        match = match_region(rough, line, proposal, args)
        if match is None:
            continue
        if (
            match["match_score"] < args.min_match_score
            or match["edge_f1"] < args.min_f1
            or match["chamfer"] > args.max_chamfer
            or match["rough_std"] < args.min_rough_std
        ):
            decision = "review_low_score"
        else:
            decision = "candidate"
        rough_norm, line_norm, out_w, out_h = normalize_pair(
            rough, line, match["rough_box"], match["line_box"], args,
        )
        if rough_norm is None or line_norm is None:
            continue
        black_fill_ratio = float((line_norm < args.black_threshold).mean())
        row = {
            **match,
            "page": page,
            "source_page": entry.get("file", ""),
            "region_index": index,
            "output_width": out_w,
            "output_height": out_h,
            "black_fill_ratio": black_fill_ratio,
            "decision": decision,
            "rough_output": rough_norm,
            "line_output": line_norm,
        }
        row["feature_tags"] = feature_tags(row, args)
        rows.append({
            **row,
        })
        if args.verbose:
            print(
                f"  region {index}/{len(proposals)} decision={decision} "
                f"score={match['match_score']:.2f} f1={match['edge_f1']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    return rows


def serializable(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            continue
        if key.endswith("_box"):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def write_rows(rows, args):
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "page", "source_page", "region_index", "decision",
        "line_box", "rough_box", "output_width", "output_height",
        "scale", "dx", "dy", "line_ink", "region_area",
        "match_score", "edge_f1", "edge_precision", "edge_recall",
        "chamfer", "projection_corr", "orientation_entropy",
        "rough_std", "black_fill_ratio", "feature_tags",
        "line_edges", "rough_edges", "notes",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **serializable(row), "notes": ""})
    with open(args.json_out, "w") as file:
        json.dump([{"rank": rank, **serializable(row)} for rank, row in enumerate(rows, 1)], file, indent=2)
        file.write("\n")


def make_qc(rows, args):
    picks = rows[: args.qc_count]
    if not picks:
        return
    thumb, label_h = 220, 34
    canvas = Image.new("RGB", (thumb * 4, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = ImageFont.load_default()
    for idx, row in enumerate(picks):
        top = idx * (thumb + label_h)
        overlay = np.full((*row["line_edge"].shape, 3), 255, np.uint8)
        overlay[row["rough_edge"]] = (255, 60, 60)
        overlay[row["line_edge"]] = (40, 80, 255)
        images = (row["rough_norm"], row["line_norm"], overlay, row["rough_output"])
        for col, image in enumerate(images):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (col * thumb, top))
        text = (
            f'{row["page"]} r{row["region_index"]} {row["decision"]} '
            f'score={row["match_score"]:.2f} F1={row["edge_f1"]:.2f} '
            f'ch={row["chamfer"]:.1f} d=({row["dx"]},{row["dy"]}) sc={row["scale"]:.2f} '
            f'out={row["output_width"]}x{row["output_height"]} tags={row["feature_tags"]}'
        )
        draw.text((3, top + thumb + 2), text, fill="black", font=font)
    Path(args.qc_out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.qc_out)


def save_regions(rows, args):
    rough_dir = Path(args.rough_out)
    line_dir = Path(args.line_out)
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for rank, row in enumerate(rows, 1):
        name = f'hamlabir_{row["page"]}_{row["region_index"]:03d}_{rank:04d}.png'
        rough_path = rough_dir / name
        line_path = line_dir / name
        Image.fromarray(row["rough_output"]).save(rough_path)
        Image.fromarray(row["line_output"]).save(line_path)
        manifest_rows.append({
            "name": name,
            "rough_path": str(rough_path),
            "line_path": str(line_path),
            **serializable(row),
        })
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as file:
        json.dump(manifest_rows, file, indent=2)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--pages", type=int, default=0)
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-line-component-area", type=int, default=12)
    parser.add_argument("--region-close", type=int, default=121)
    parser.add_argument("--line-margin", type=float, default=0.12)
    parser.add_argument("--min-region-area", type=int, default=3500)
    parser.add_argument("--min-region-size", type=int, default=180)
    parser.add_argument("--max-region-ratio", type=float, default=0.85)
    parser.add_argument("--min-line-ink", type=float, default=0.004)
    parser.add_argument("--max-line-ink", type=float, default=0.22)
    parser.add_argument("--max-regions-per-page", type=int, default=12)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.85, 0.95, 1.0, 1.05, 1.15])
    parser.add_argument("--search-px", type=int, default=120)
    parser.add_argument("--search-ratio", type=float, default=0.18)
    parser.add_argument("--search-step", type=int, default=48)
    parser.add_argument("--search-step-ratio", type=float, default=0.07)
    parser.add_argument("--match-size", type=int, default=384)
    parser.add_argument("--output-long-side", type=int, default=768)
    parser.add_argument("--close-px", type=int, default=5)
    parser.add_argument("--truncate-px", type=int, default=36)
    parser.add_argument("--min-match-edges", type=int, default=150)
    parser.add_argument("--min-match-score", type=float, default=0.8)
    parser.add_argument("--min-f1", type=float, default=0.18)
    parser.add_argument("--max-chamfer", type=float, default=22.0)
    parser.add_argument("--min-rough-std", type=float, default=12.0)
    parser.add_argument("--black-threshold", type=int, default=32)
    parser.add_argument("--black-fill-ratio", type=float, default=0.08)
    parser.add_argument("--large-region-long-side", type=int, default=3000)
    parser.add_argument("--large-region-area", type=int, default=5_000_000)
    parser.add_argument("--wide-region-ratio", type=float, default=2.2)
    parser.add_argument("--tall-region-ratio", type=float, default=2.2)
    parser.add_argument("--csv-out", default=DEFAULT_CSV)
    parser.add_argument("--json-out", default=DEFAULT_JSON)
    parser.add_argument("--qc-out", default=DEFAULT_QC)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--rough-out", default="dataset/regions_hamlabi/rough")
    parser.add_argument("--line-out", default="dataset/regions_hamlabi/line")
    parser.add_argument("--manifest-out", default="dataset/regions_hamlabi/manifest.json")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.zip_path, args.zip_root)
    entries = manifest[: args.pages] if args.pages else manifest
    rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for index, entry in enumerate(entries, 1):
            rough, line = load_pair(zf, args.zip_root, entry)
            page_rows = process_page(entry, rough, line, args)
            rows.extend(page_rows)
            print(f"{index}/{len(entries)} page={page_id(entry)} regions={len(page_rows)} total={len(rows)}", flush=True)
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    write_rows(rows, args)
    make_qc(rows, args)
    if args.save:
        save_regions([row for row in rows if row["decision"] == "candidate"], args)
    print(f"candidates={sum(row['decision'] == 'candidate' for row in rows)} review={len(rows)}")
    print(f"wrote: {args.csv_out}, {args.json_out}, {args.qc_out}")
    if args.save:
        print(f"saved manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
