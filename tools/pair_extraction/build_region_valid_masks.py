"""Build valid-loss masks for partially mismatched region pairs."""

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
    for key in ("v2_name", "final_name", "name", "materialized_name"):
        if row.get(key):
            return Path(row[key]).stem
    return f"region_{index:04d}"


def edge_map(gray, blur_sigma):
    blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    return cv2.Canny(blurred, 45, 135) > 0


def dense_regions(mask, window, threshold):
    density = cv2.blur(mask.astype(np.float32), (window, window))
    return density >= threshold


def build_mask(rough, line, args):
    rough_arr = np.asarray(rough, dtype=np.uint8)
    line_arr = np.asarray(line, dtype=np.uint8)
    if rough_arr.shape != line_arr.shape:
        rough_arr = np.asarray(rough.resize(line.size, Image.Resampling.BICUBIC), dtype=np.uint8)

    rough_edge = edge_map(rough_arr, args.blur_sigma)
    line_edge = edge_map(line_arr, args.blur_sigma)
    kernel = np.ones((args.support_px * 2 + 1, args.support_px * 2 + 1), np.uint8)
    rough_support = cv2.dilate(rough_edge.astype(np.uint8), kernel) > 0
    line_support = cv2.dilate(line_edge.astype(np.uint8), kernel) > 0

    unsupported_line = line_edge & ~rough_support
    unsupported_rough = rough_edge & ~line_support
    mismatch = unsupported_line.copy()
    if args.include_rough_extra:
        mismatch |= unsupported_rough
    ignore = dense_regions(mismatch, args.window, args.edge_density)

    if args.include_black_fill:
        line_dark = line_arr < args.black_threshold
        rough_dark = rough_arr < args.rough_black_threshold
        unsupported_black = line_dark & ~cv2.dilate(rough_dark.astype(np.uint8), kernel).astype(bool)
        ignore |= dense_regions(unsupported_black, args.black_window, args.black_density)

    if args.expand_ignore > 0:
        expand_kernel = np.ones((args.expand_ignore * 2 + 1, args.expand_ignore * 2 + 1), np.uint8)
        ignore = cv2.dilate(ignore.astype(np.uint8), expand_kernel) > 0
    if args.close_ignore > 0:
        close_kernel = np.ones((args.close_ignore, args.close_ignore), np.uint8)
        ignore = cv2.morphologyEx(ignore.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0

    valid = ~ignore
    min_valid = args.min_valid_ratio
    if valid.mean() < min_valid:
        keep = dense_regions(line_edge | rough_edge, args.window, args.edge_density * 0.5)
        valid = ~keep
    return valid.astype(np.uint8) * 255, {
        "ignore_ratio": float((~valid).mean()),
        "unsupported_line_edge_ratio": float(unsupported_line.sum() / max(line_edge.sum(), 1)),
        "unsupported_rough_edge_ratio": float(unsupported_rough.sum() / max(rough_edge.sum(), 1)),
        "line_edge_pixels": int(line_edge.sum()),
        "rough_edge_pixels": int(rough_edge.sum()),
    }


def overlay_ignore(line, valid_mask):
    base = np.asarray(line.convert("RGB"), dtype=np.uint8).copy()
    ignore = np.asarray(valid_mask) < 128
    base[ignore] = (255 * 0.45 + base[ignore] * 0.55).astype(np.uint8)
    base[ignore, 0] = 255
    base[ignore, 1] = (base[ignore, 1] * 0.35).astype(np.uint8)
    base[ignore, 2] = (base[ignore, 2] * 0.35).astype(np.uint8)
    return Image.fromarray(base)


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
    columns = ["rough", "line", "valid mask", "ignored overlay"]
    label_h = 34
    header_h = 26
    canvas = Image.new("RGB", (thumb * len(columns), header_h + (thumb + label_h) * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = small_font = ImageFont.load_default()
    for col, title in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * thumb + (thumb - width) / 2, 5), title, fill=0, font=font)
    for row_idx, item in enumerate(items):
        y = header_h + row_idx * (thumb + label_h)
        for col, image in enumerate((item["rough"], item["line"], item["mask"], item["overlay"])):
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
    parser.add_argument("--support-px", type=int, default=5)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--edge-density", type=float, default=0.030)
    parser.add_argument("--expand-ignore", type=int, default=9)
    parser.add_argument("--close-ignore", type=int, default=9)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--include-black-fill", action="store_true")
    parser.add_argument("--include-rough-extra", action="store_true")
    parser.add_argument("--black-threshold", type=int, default=40)
    parser.add_argument("--rough-black-threshold", type=int, default=96)
    parser.add_argument("--black-window", type=int, default=45)
    parser.add_argument("--black-density", type=float, default=0.16)
    parser.add_argument("--min-valid-ratio", type=float, default=0.35)
    parser.add_argument("--qc-count", type=int, default=34)
    parser.add_argument("--qc-thumb", type=int, default=220)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    out_base = Path(args.out_base)
    rough_dir = out_base / "rough"
    line_dir = out_base / "line"
    mask_dir = out_base / "valid_mask"
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

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
        mask_arr, stats = build_mask(rough_norm, line_norm, args)
        mask = Image.fromarray(mask_arr)

        name = f"masked_{index:04d}_{row_name(row, index)}.png"
        rough_out = rough_dir / name
        line_out = line_dir / name
        mask_out = mask_dir / name
        rough_norm.save(rough_out)
        line_norm.save(line_out)
        mask.save(mask_out)
        out_row = {
            **row,
            "masked_name": name,
            "masked_rough_path": str(rough_out),
            "masked_line_path": str(line_out),
            "valid_mask_path": str(mask_out),
            "mask_method": "edge_support_v1_blackfill" if args.include_black_fill else "edge_support_v1",
            "mask_image_size": args.image_size,
            "mask_fit_mode": args.fit_mode,
            **stats,
        }
        out_rows.append(out_row)
        if len(qc_items) < args.qc_count:
            qc_items.append({
                "rough": rough_norm,
                "line": line_norm,
                "mask": mask,
                "overlay": overlay_ignore(line_norm, mask),
                "label": (
                    f"{index:03d} ignore={stats['ignore_ratio']:.2f} "
                    f"ul={stats['unsupported_line_edge_ratio']:.2f} "
                    f"ur={stats['unsupported_rough_edge_ratio']:.2f} {row_name(row, index)}"
                ),
            })

    (out_base / "manifest.json").write_text(json.dumps(out_rows, indent=2) + "\n")
    write_csv(out_rows, out_base / "manifest.csv")
    make_qc(qc_items, out_base / "valid_mask_qc.png", args.qc_thumb)
    print(f"rows={len(out_rows)} wrote={out_base}")


if __name__ == "__main__":
    main()
