"""Post-align normalized rough/line region pairs by small translation search."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lineart.region_dataset import fit_square, read_manifest, resolve_path


def path_for(row, manifest_path, keys):
    for key in keys:
        if row.get(key):
            return resolve_path(row[key], manifest_path)
    raise KeyError(f"missing path key; tried {keys}")


def row_name(row, index):
    for key in ("v2_name", "final_name", "name", "materialized_name", "masked_name"):
        if row.get(key):
            return Path(row[key]).stem
    return f"region_{index:04d}"


def edge_map(gray, sigma):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.Canny(blurred, 45, 135) > 0


def shift_array(arr, dx, dy, fill=255):
    h, w = arr.shape
    shifted = np.full_like(arr, fill)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(w, w - dx)
    src_y1 = min(h, h - dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted


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
        return truncate
    d_to_rough = cv2.distanceTransform((~rough_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_to_line = cv2.distanceTransform((~line_edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 0.5 * (
        float(np.minimum(d_to_line[rough_edge], truncate).mean())
        + float(np.minimum(d_to_rough[line_edge], truncate).mean())
    )


def score_edges(rough_edge, line_edge, tolerance, truncate):
    f1, precision, recall = support_f1(rough_edge, line_edge, tolerance)
    ch = chamfer(rough_edge, line_edge, truncate)
    return 4.0 * f1 - 0.08 * ch, f1, precision, recall, ch


def fast_score_shifted(shifted_edge, line_edge, line_support, tolerance):
    if shifted_edge.sum() < 20 or line_edge.sum() < 20:
        return -999.0, 0.0, 0.0, 0.0
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    rough_support = cv2.dilate(shifted_edge.astype(np.uint8), kernel) > 0
    precision = float(line_support[shifted_edge].mean())
    recall = float(rough_support[line_edge].mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return 4.0 * f1, f1, precision, recall


def candidate_offsets(max_shift, step):
    offsets = []
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            offsets.append((dx, dy))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return offsets


def find_shift(rough, line, args):
    rough_arr = np.asarray(rough, dtype=np.uint8)
    line_arr = np.asarray(line, dtype=np.uint8)
    rough_edge = edge_map(rough_arr, args.blur_sigma)
    line_edge = edge_map(line_arr, args.blur_sigma)
    kernel = np.ones((args.tolerance * 2 + 1, args.tolerance * 2 + 1), np.uint8)
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0
    base_score, base_f1, base_precision, base_recall, base_chamfer = score_edges(
        rough_edge,
        line_edge,
        args.tolerance,
        args.truncate_px,
    )
    best = {
        "dx": 0,
        "dy": 0,
        "search_score": 4.0 * base_f1,
        "score": base_score,
        "f1": base_f1,
        "precision": base_precision,
        "recall": base_recall,
        "chamfer": base_chamfer,
    }
    for dx, dy in candidate_offsets(args.max_shift, args.step):
        shifted_edge = shift_array(rough_edge.astype(np.uint8), dx, dy, fill=0) > 0
        search_score, f1, precision, recall = fast_score_shifted(
            shifted_edge,
            line_edge,
            line_support,
            args.tolerance,
        )
        if search_score > best["search_score"]:
            best = {
                "dx": dx,
                "dy": dy,
                "search_score": search_score,
                "score": base_score,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "chamfer": base_chamfer,
            }
    if args.refine_step > 0 and (best["dx"] or best["dy"]):
        cx, cy = best["dx"], best["dy"]
        for dy in range(cy - args.step, cy + args.step + 1, args.refine_step):
            for dx in range(cx - args.step, cx + args.step + 1, args.refine_step):
                if abs(dx) > args.max_shift or abs(dy) > args.max_shift:
                    continue
                shifted_edge = shift_array(rough_edge.astype(np.uint8), dx, dy, fill=0) > 0
                search_score, f1, precision, recall = fast_score_shifted(
                    shifted_edge,
                    line_edge,
                    line_support,
                    args.tolerance,
                )
                if search_score > best["search_score"]:
                    best = {
                        "dx": dx,
                        "dy": dy,
                        "search_score": search_score,
                        "score": base_score,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "chamfer": base_chamfer,
                    }
    if best["dx"] or best["dy"]:
        shifted_edge = shift_array(rough_edge.astype(np.uint8), best["dx"], best["dy"], fill=0) > 0
        best["score"], best["f1"], best["precision"], best["recall"], best["chamfer"] = score_edges(
            shifted_edge,
            line_edge,
            args.tolerance,
            args.truncate_px,
        )
    best["base_score"] = base_score
    best["base_f1"] = base_f1
    best["base_chamfer"] = base_chamfer
    best["score_gain"] = best["score"] - base_score
    if best["score_gain"] < args.min_gain:
        best["dx"] = 0
        best["dy"] = 0
        best["score"] = base_score
        best["f1"] = base_f1
        best["chamfer"] = base_chamfer
        best["score_gain"] = 0.0
    best.pop("search_score", None)
    return best


def overlay_edges(rough, line):
    rough_edge = edge_map(np.asarray(rough, dtype=np.uint8), 1.0)
    line_edge = edge_map(np.asarray(line, dtype=np.uint8), 1.0)
    overlay = np.full((*rough_edge.shape, 3), 255, dtype=np.uint8)
    overlay[rough_edge] = (255, 60, 60)
    overlay[line_edge] = (40, 80, 255)
    both = rough_edge & line_edge
    overlay[both] = (40, 160, 40)
    return Image.fromarray(overlay)


def write_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_qc(items, output, thumb):
    if not items:
        return
    columns = ["rough original", "rough aligned", "line", "edge overlay"]
    header_h = 28
    label_h = 34
    canvas = Image.new("RGB", (thumb * len(columns), header_h + (thumb + label_h) * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = small_font = ImageFont.load_default()
    for col, title in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * thumb + (thumb - width) / 2, 6), title, fill=0, font=font)
    for row_idx, item in enumerate(items):
        y = header_h + row_idx * (thumb + label_h)
        for col, image in enumerate((item["rough"], item["aligned"], item["line"], item["overlay"])):
            canvas.paste(image.convert("RGB").resize((thumb, thumb)), (col * thumb, y))
        draw.text((4, y + thumb + 2), item["label"], fill=0, font=small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--fit-mode", choices=["square_pad", "resize_stretch"], default="square_pad")
    parser.add_argument("--autocontrast-rough", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-shift", type=int, default=24)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--refine-step", type=int, default=1)
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--truncate-px", type=int, default=36)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--qc-count", type=int, default=34)
    parser.add_argument("--qc-thumb", type=int, default=220)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    out_base = Path(args.out_base)
    rough_dir = out_base / "rough"
    line_dir = out_base / "line"
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    qc_items = []
    for index, row in enumerate(rows, 1):
        rough_path = path_for(
            row,
            manifest_path,
            ("aligned_rough_path", "v2_rough_path", "final_rough_path", "rough_path", "materialized_rough_path"),
        )
        line_path = path_for(
            row,
            manifest_path,
            ("aligned_line_path", "v2_line_path", "final_line_path", "line_path", "materialized_line_path"),
        )
        rough = Image.open(rough_path).convert("L")
        line = Image.open(line_path).convert("L")
        if args.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough_norm = fit_square(rough, args.image_size, args.fit_mode)
        line_norm = fit_square(line, args.image_size, args.fit_mode)
        shift = find_shift(rough_norm, line_norm, args)
        aligned_arr = shift_array(np.asarray(rough_norm, dtype=np.uint8), shift["dx"], shift["dy"], fill=255)
        aligned = Image.fromarray(aligned_arr)

        name = f"aligned_{index:04d}_{row_name(row, index)}.png"
        aligned_rough_path = rough_dir / name
        aligned_line_path = line_dir / name
        aligned.save(aligned_rough_path)
        line_norm.save(aligned_line_path)
        out_row = {
            **row,
            "aligned_name": name,
            "aligned_rough_path": str(aligned_rough_path),
            "aligned_line_path": str(aligned_line_path),
            "align_dx": shift["dx"],
            "align_dy": shift["dy"],
            "align_score": shift["score"],
            "align_base_score": shift["base_score"],
            "align_score_gain": shift["score_gain"],
            "align_f1": shift["f1"],
            "align_base_f1": shift["base_f1"],
            "align_chamfer": shift["chamfer"],
            "align_base_chamfer": shift["base_chamfer"],
            "align_image_size": args.image_size,
            "align_fit_mode": args.fit_mode,
        }
        out_rows.append(out_row)
        if len(qc_items) < args.qc_count:
            qc_items.append({
                "rough": rough_norm,
                "aligned": aligned,
                "line": line_norm,
                "overlay": overlay_edges(aligned, line_norm),
                "label": (
                    f"{index:03d} dx={shift['dx']} dy={shift['dy']} "
                    f"gain={shift['score_gain']:.3f} f1={shift['base_f1']:.2f}->{shift['f1']:.2f} "
                    f"{row_name(row, index)}"
                ),
            })

    (out_base / "manifest.json").write_text(json.dumps(out_rows, indent=2) + "\n")
    write_csv(out_rows, out_base / "manifest.csv")
    make_qc(qc_items, out_base / "post_align_qc.png", args.qc_thumb)
    shifted = sum(1 for row in out_rows if int(row["align_dx"]) or int(row["align_dy"]))
    mean_gain = sum(float(row["align_score_gain"]) for row in out_rows) / max(len(out_rows), 1)
    print(f"rows={len(out_rows)} shifted={shifted} mean_gain={mean_gain:.4f} wrote={out_base}")


if __name__ == "__main__":
    main()
