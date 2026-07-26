"""Materialize accepted koma-panel candidates from `match_koma_panels.py` into
native-resolution rough/line image pairs, ready for the existing
`build_region_valid_masks.py` + `tile_region_manifest_480.py` pipeline.

Panel-level acceptance is a chamfer gate (`--max-chamfer`), applied per panel
rather than per page: a page with some bad panels (e.g. housei_010/011/012)
can still contribute its good panels. Pages that are a true asset mismatch
(housei_004) are dropped entirely via `--exclude-housei`.

Dry-run by default (writes a QC montage only); pass `--save` to write the
materialized rough/line PNGs and manifest.
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_koma_panels import ZIP_PATH, ZIP_ROOT, crop_page, extract_scaled_roi, load_gray, load_json_member, page_id


def materialize_rough(rough_gray, cx, cy, out_w, out_h, dx, dy, scale, pad=170):
    roi = extract_scaled_roi(rough_gray, cx, cy, out_w, out_h, pad, scale)
    y0, x0 = pad + dy, pad + dx
    return roi[y0:y0 + out_h, x0:x0 + out_w]


def make_qc(items, output_path, thumb=220):
    if not items:
        return
    columns = ["rough (aligned)", "line"]
    header_h, label_h = 28, 34
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
        for col, image in enumerate((item["rough"], item["line"])):
            canvas.paste(Image.fromarray(image).convert("RGB").resize((thumb, thumb)), (col * thumb, y))
        draw.text((4, y + thumb + 2), item["label"], fill=0, font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels-csv", default="results/housei_koma_panels_20260726.csv")
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--zip-root", default=ZIP_ROOT)
    parser.add_argument("--max-chamfer", type=float, default=20.0)
    parser.add_argument("--exclude-housei", default="housei_004")
    parser.add_argument("--out-dir", default="dataset/regions_housei_koma_panels_20260726")
    parser.add_argument("--qc-out", default="results/housei_koma_panels_20260726_materialized_qc.png")
    parser.add_argument("--qc-count", type=int, default=60)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    excluded = {h.strip() for h in args.exclude_housei.split(",") if h.strip()}

    with open(args.panels_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    accepted = [
        r for r in rows
        if r["housei"] not in excluded and float(r["chamfer"]) <= args.max_chamfer
    ]
    print(f"panels: total={len(rows)} excluded_pages={len(rows) - len([r for r in rows if r['housei'] not in excluded])} "
          f"accepted(chamfer<={args.max_chamfer})={len(accepted)}")

    out_dir = Path(args.out_dir)
    rough_dir, line_dir = out_dir / "rough", out_dir / "line"
    if args.save:
        rough_dir.mkdir(parents=True, exist_ok=True)
        line_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    qc_items = []
    with zipfile.ZipFile(args.zip_path) as zf:
        manifest = load_json_member(zf, args.zip_root, "manifest.json")
        by_id = {page_id(e): e for e in manifest}
        page_cache = {}
        for index, row in enumerate(accepted, 1):
            hid = row["housei"]
            if hid not in page_cache:
                entry = by_id[hid]
                line_gray = load_gray(zf, args.zip_root, entry["line"])
                rough_gray = load_gray(zf, args.zip_root, entry["sketch"], autocontrast=True)
                page_cache[hid] = (line_gray, rough_gray)
            line_gray, rough_gray = page_cache[hid]

            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            out_w, out_h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dx, dy, scale = int(row["dx"]), int(row["dy"]), float(row["scale"])

            line_crop = crop_page(line_gray, x0, y0, x1, y1)
            rough_aligned = materialize_rough(rough_gray, cx, cy, out_w, out_h, dx, dy, scale)

            name = f"koma_{index:04d}_{hid}_p{row['panel_index']}.png"
            native_long_side = max(out_w, out_h)
            out_row = {
                **row,
                "name": name,
                "native_long_side": native_long_side,
                "native_width": out_w,
                "native_height": out_h,
            }
            if args.save:
                rough_path = rough_dir / name
                line_path = line_dir / name
                Image.fromarray(rough_aligned).save(rough_path)
                Image.fromarray(line_crop).save(line_path)
                out_row["native_rough_path"] = str(rough_path)
                out_row["native_line_path"] = str(line_path)
            out_rows.append(out_row)

            if len(qc_items) < args.qc_count:
                qc_items.append({
                    "rough": rough_aligned, "line": line_crop,
                    "label": f'{hid} p{row["panel_index"]} chamfer={float(row["chamfer"]):.1f} scale={scale:.2f} d=({dx},{dy})',
                })

    make_qc(qc_items, Path(args.qc_out), thumb=220)

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(out_rows, indent=2) + "\n")
        fields = list(out_rows[0].keys()) if out_rows else []
        with open(out_dir / "manifest.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"saved {len(out_rows)} rows to {out_dir}/manifest.csv")
    else:
        print(f"dry-run: would save {len(out_rows)} rows to {out_dir}/manifest.csv; pass --save to write")
    print(f"QC: {args.qc_out}")


if __name__ == "__main__":
    main()
