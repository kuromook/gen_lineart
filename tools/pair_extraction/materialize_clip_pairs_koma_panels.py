"""Materialize accepted `clip_pairs` koma-panel candidates from
`match_clip_pairs_koma_panels.py`'s output into native-resolution rough/line
image pairs, ready for `build_region_valid_masks.py` + `tile_region_manifest_480.py`
-- the `clip_pairs` counterpart of `materialize_koma_panels.py`, adapted for
multiple zip roots (one per work_id/slug) instead of a single fixed one.

Panel-level acceptance is a chamfer gate (`--max-chamfer`), applied per panel
rather than per page, same convention as the 5 existing koma-pipeline sources
(`run_koma_tile_pipeline.sh` uses a generous `--max-chamfer 45` pre-filter
here and relies on `tile_region_manifest_480.py`'s fixed `ALIGNMENT_*`
constants for the real gate downstream).

Dry-run by default (writes a QC montage only); pass `--save` to write the
materialized rough/line PNGs and manifest.
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_koma_panels import crop_page, extract_scaled_roi, load_gray

ZIP_PATH = "dataset/raw_zips/dataset_clip_pairs_v2.zip"
QC_CSV_MEMBER = "clip_pairs/clip_pairs_qc.csv"


def materialize_rough(rough_gray, cx, cy, out_w, out_h, dx, dy, scale, pad=170):
    pad = max(pad, abs(dx) + 8, abs(dy) + 8)
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
    parser.add_argument("--panels-csv", default="results/clip_pairs_koma_panels_20260821.csv")
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--max-chamfer", type=float, default=45.0)
    parser.add_argument("--out-dir", default="dataset/regions_clip_pairs_koma_panels_20260822")
    parser.add_argument("--qc-out", default="results/clip_pairs_koma_panels_20260822_materialized_qc.png")
    parser.add_argument("--qc-count", type=int, default=60)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(args.panels_csv, newline="") as file:
        rows = list(csv.DictReader(file))
    accepted = [r for r in rows if float(r["chamfer"]) <= args.max_chamfer]
    print(f"panels: total={len(rows)} accepted(chamfer<={args.max_chamfer})={len(accepted)}")

    out_dir = Path(args.out_dir)
    rough_dir, line_dir = out_dir / "rough", out_dir / "line"
    if args.save:
        rough_dir.mkdir(parents=True, exist_ok=True)
        line_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    qc_items = []
    with zipfile.ZipFile(args.zip_path) as zf:
        qc_text = zf.read(QC_CSV_MEMBER).decode("utf-8")
        qc_by_key = {
            (r["work_id"], r["slug"], r["page"]): r
            for r in csv.DictReader(qc_text.splitlines())
        }

        # Single-entry cache keyed by (work_id, slug, page): accepted rows
        # are page-grouped (match_clip_pairs_koma_panels.py processes pages
        # in order), so this only reloads on an actual page change -- same
        # OOM-avoidance convention as materialize_koma_panels.py.
        page_cache = {}
        skipped = 0
        for index, row in enumerate(accepted, 1):
            work_id, slug, page = row["work_id"], row["slug"], row["page"]
            key = (work_id, slug, page)
            if key not in page_cache:
                page_cache.clear()
                qc_entry = qc_by_key.get(key)
                if qc_entry is None:
                    skipped += 1
                    continue
                zip_root = f"clip_pairs/{work_id}/{slug}"
                line_gray = load_gray(zf, zip_root, qc_entry["line"])
                rough_gray = load_gray(zf, zip_root, qc_entry["sketch"], autocontrast=True)
                page_cache[key] = (line_gray, rough_gray)
            if key not in page_cache:
                continue
            line_gray, rough_gray = page_cache[key]

            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            out_w, out_h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dx, dy, scale = int(row["dx"]), int(row["dy"]), float(row["scale"])

            line_crop = crop_page(line_gray, x0, y0, x1, y1)
            rough_aligned = materialize_rough(rough_gray, cx, cy, out_w, out_h, dx, dy, scale)

            name = f"koma_{index:05d}_{row['pid']}_p{row['panel_index']}.png"
            native_long_side = max(out_w, out_h)
            out_row = {
                **row,
                # alias so downstream tools built for the 5 existing koma
                # sources (split_koma_panel_subregions.py etc., which index
                # row["housei"] directly as the generic page-id field) work
                # unchanged against this clip_pairs manifest.
                "housei": row["pid"],
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
                    "label": f'{row["pid"]} p{row["panel_index"]} chamfer={float(row["chamfer"]):.1f} scale={scale:.2f} d=({dx},{dy})',
                })

        if skipped:
            print(f"skipped {skipped} rows: no matching clip_pairs_qc.csv entry")

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
