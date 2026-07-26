"""Materialize reviewed region candidate CSVs as variable-aspect pairs."""

import argparse
import ast
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_zip_member(zf, root, name):
    for candidate in (f"{root}/{name}", f"{root}\\{name}", name):
        try:
            return zf.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(read_zip_member(zf, zip_root, "manifest.json"))
        rows = {}
        for entry in manifest:
            try:
                read_zip_member(zf, zip_root, entry["sketch"])
                read_zip_member(zf, zip_root, entry["line"])
            except KeyError:
                continue
            page = Path(entry.get("file", entry["line"])).stem.replace("page", "")
            rows[page] = entry
    return rows


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["line"]))).convert("L")
    rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def parse_box(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in ast.literal_eval(value))


def crop_resize(image, box, long_side):
    x0, y0, x1, y1 = parse_box(box)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None, None
    height, width = crop.shape
    scale = long_side / max(width, height)
    out_w = max(32, int(round(width * scale)))
    out_h = max(32, int(round(height * scale)))
    resized = Image.fromarray(crop).resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(resized), out_w, out_h


def read_rows(paths, decision):
    rows = []
    for path in paths:
        source = Path(path).stem
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                if decision and row.get("decision") != decision:
                    continue
                rows.append({**row, "source_csv": str(path), "source_kind": source})
    rows.sort(key=lambda row: float(row.get("match_score", 0.0)), reverse=True)
    return rows


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = parse_box(a)
    bx0, by0, bx1, by1 = parse_box(b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(1, ax1 - ax0) * max(1, ay1 - ay0)
    area_b = max(1, bx1 - bx0) * max(1, by1 - by0)
    return inter / max(area_a + area_b - inter, 1)


def dedupe_rows(rows, duplicate_iou):
    if duplicate_iou <= 0:
        return rows
    kept = []
    for row in rows:
        duplicate = False
        for existing in kept:
            if row.get("page") != existing.get("page"):
                continue
            if (
                box_iou(row["line_box"], existing["line_box"]) >= duplicate_iou
                or box_iou(row["rough_box"], existing["rough_box"]) >= duplicate_iou
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def write_csv(rows, path):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize(args):
    manifest = load_manifest(args.zip_path, args.zip_root)
    rows = dedupe_rows(read_rows(args.csv, args.decision), args.duplicate_iou)
    if args.limit:
        rows = rows[: args.limit]

    rough_dir = Path(args.out_base) / "rough"
    line_dir = Path(args.out_base) / "line"
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        cache = {}
        for out_index, row in enumerate(rows, 1):
            page = row["page"]
            if page not in manifest:
                continue
            if page not in cache:
                cache[page] = load_pair(zf, args.zip_root, manifest[page])
            rough_page, line_page = cache[page]
            rough, out_w, out_h = crop_resize(rough_page, row["rough_box"], args.long_side)
            line, _, _ = crop_resize(line_page, row["line_box"], args.long_side)
            if rough is None or line is None:
                continue
            name = f'{args.name_prefix}_{page}_{int(row["rank"]):04d}_{out_index:04d}.png'
            rough_path = rough_dir / name
            line_path = line_dir / name
            Image.fromarray(rough).save(rough_path)
            Image.fromarray(line).save(line_path)
            output_rows.append({
                **row,
                "name": name,
                "rough_path": str(rough_path),
                "line_path": str(line_path),
                "materialized_width": out_w,
                "materialized_height": out_h,
                "materialized_long_side": args.long_side,
            })
    return output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--zip-root", required=True)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--decision", default="candidate")
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--name-prefix", default="region")
    parser.add_argument("--duplicate-iou", type=float, default=0.72)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = materialize(args)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as file:
        json.dump(rows, file, indent=2)
        file.write("\n")
    write_csv(rows, args.csv_out)
    print(f"materialized={len(rows)}")
    print(f"wrote: {args.manifest_out}, {args.csv_out}")


if __name__ == "__main__":
    main()
