"""Materialize reviewed hamlabi region candidates as variable-aspect pairs."""

import argparse
import ast
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_ZIP = "dataset_hamlabi.zip"
DEFAULT_ZIP_ROOT = "dataset_hamlabi"
DEFAULT_REVIEW = "results/hamlabi_region_codex_review_prelim.csv"
DEFAULT_BASE = "dataset/regions_hamlabi_review"


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
    return {Path(entry.get("file", entry["line"])).stem.replace("page", ""): entry for entry in manifest}


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["line"]))).convert("L")
    rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def parse_box(value):
    return tuple(int(v) for v in ast.literal_eval(value))


def crop_resize(image, box, long_side):
    x0, y0, x1, y1 = parse_box(box)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None, None
    h, w = crop.shape
    scale = long_side / max(w, h)
    out_w = max(32, int(round(w * scale)))
    out_h = max(32, int(round(h * scale)))
    resized = Image.fromarray(crop).resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(resized), out_w, out_h


def read_review_rows(path, decision):
    with open(path, newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("codex_decision") == decision]
    rows.sort(key=lambda row: int(row["rank"]))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize(args):
    rows = read_review_rows(args.review_csv, args.decision)
    manifest = load_manifest(args.zip_path, args.zip_root)
    rough_dir = Path(args.out_base) / "rough"
    line_dir = Path(args.out_base) / "line"
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        cache = {}
        for out_index, row in enumerate(rows, 1):
            page = row["page"]
            if page not in cache:
                cache[page] = load_pair(zf, args.zip_root, manifest[page])
            rough_page, line_page = cache[page]
            rough, out_w, out_h = crop_resize(rough_page, row["rough_box"], args.long_side)
            line, _, _ = crop_resize(line_page, row["line_box"], args.long_side)
            if rough is None or line is None:
                continue
            name = f'hamlabir_{page}_{int(row["rank"]):04d}_{out_index:04d}.png'
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
    parser.add_argument("--zip", default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW)
    parser.add_argument("--decision", default="accept_review")
    parser.add_argument("--out-base", default=DEFAULT_BASE)
    parser.add_argument("--manifest-out", default=f"{DEFAULT_BASE}/manifest.json")
    parser.add_argument("--csv-out", default=f"{DEFAULT_BASE}/manifest.csv")
    parser.add_argument("--long-side", type=int, default=768)
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
