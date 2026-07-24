"""Materialize a variable-aspect region manifest into fixed-size square pairs."""

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lineart.region_dataset import fit_square, read_manifest, resolve_path


def path_for(row, manifest_path, keys):
    for key in keys:
        if row.get(key):
            return resolve_path(row[key], manifest_path)
    raise KeyError(f"missing path key; tried {keys}")


def write_csv(rows, path):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--size", type=int, choices=[480, 768], required=True)
    parser.add_argument("--fit-mode", choices=["square_pad", "resize_stretch"], default="square_pad")
    parser.add_argument("--name-prefix", default="region")
    parser.add_argument("--autocontrast-rough", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    rough_dir = Path(args.out_base) / "rough"
    line_dir = Path(args.out_base) / "line"
    rough_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for index, row in enumerate(rows, 1):
        rough_path = path_for(row, manifest_path, ("v2_rough_path", "final_rough_path", "rough_path"))
        line_path = path_for(row, manifest_path, ("v2_line_path", "final_line_path", "line_path"))
        rough = Image.open(rough_path).convert("L")
        line = Image.open(line_path).convert("L")
        if args.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough_out = fit_square(rough, args.size, args.fit_mode)
        line_out = fit_square(line, args.size, args.fit_mode)
        source_name = Path(row.get("v2_name") or row.get("final_name") or row.get("name") or rough_path.name).stem
        name = f"{args.name_prefix}_{args.size}_{index:04d}_{source_name}.png"
        rough_out_path = rough_dir / name
        line_out_path = line_dir / name
        rough_out.save(rough_out_path)
        line_out.save(line_out_path)
        out_rows.append({
            **row,
            "materialized_name": name,
            "materialized_rough_path": str(rough_out_path),
            "materialized_line_path": str(line_out_path),
            "materialized_size": args.size,
            "materialized_fit_mode": args.fit_mode,
        })

    Path(args.out_base).mkdir(parents=True, exist_ok=True)
    (Path(args.out_base) / "manifest.json").write_text(json.dumps(out_rows, indent=2) + "\n")
    write_csv(out_rows, Path(args.out_base) / "manifest.csv")
    (Path(args.out_base) / "file_list.txt").write_text(
        "\n".join(row["materialized_name"] for row in out_rows) + "\n"
    )
    print(f"materialized={len(out_rows)} size={args.size} fit_mode={args.fit_mode}")
    print(f"wrote: {args.out_base}")


if __name__ == "__main__":
    main()
