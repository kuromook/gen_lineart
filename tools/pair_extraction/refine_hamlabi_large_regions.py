"""Find child region pairs inside large hamlabi parent candidates."""

import argparse
import ast
import csv
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import match_hamlabi_regions as mh


DEFAULT_PARENTS = "results/hamlabi_region_candidates.csv"
DEFAULT_CSV = "results/hamlabi_region_child_candidates.csv"
DEFAULT_JSON = "results/hamlabi_region_child_candidates.json"
DEFAULT_QC = "results/hamlabi_region_child_candidates_qc.png"


def parse_box(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in ast.literal_eval(value))


def translate_box(box, offset_x, offset_y):
    x0, y0, x1, y1 = box
    return x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y


def read_parents(path, args):
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            line_box = parse_box(row["line_box"])
            width = line_box[2] - line_box[0]
            height = line_box[3] - line_box[1]
            is_large = (
                max(width, height) >= args.parent_long_side
                or width * height >= args.parent_area
                or "large_parent" in row.get("feature_tags", "")
            )
            if is_large:
                rows.append(row)
    return rows[: args.limit_parents] if args.limit_parents else rows


def load_manifest(zip_path, zip_root):
    manifest = mh.load_manifest(zip_path, zip_root)
    return {mh.page_id(entry): entry for entry in manifest}


def child_args(args):
    return SimpleNamespace(
        line_threshold=args.line_threshold,
        min_line_component_area=args.min_line_component_area,
        region_close=args.region_close,
        line_margin=args.line_margin,
        min_region_area=args.min_region_area,
        min_region_size=args.min_region_size,
        max_region_ratio=args.max_region_ratio,
        min_line_ink=args.min_line_ink,
        max_line_ink=args.max_line_ink,
        max_regions_per_page=args.max_children_per_parent,
        scales=args.scales,
        search_px=args.search_px,
        search_ratio=args.search_ratio,
        search_step=args.search_step,
        search_step_ratio=args.search_step_ratio,
        match_size=args.match_size,
        output_long_side=args.output_long_side,
        close_px=args.close_px,
        truncate_px=args.truncate_px,
        min_match_edges=args.min_match_edges,
        min_match_score=args.min_match_score,
        min_f1=args.min_f1,
        max_chamfer=args.max_chamfer,
        min_rough_std=args.min_rough_std,
        black_threshold=args.black_threshold,
        black_fill_ratio=args.black_fill_ratio,
        large_region_long_side=args.large_region_long_side,
        large_region_area=args.large_region_area,
        wide_region_ratio=args.wide_region_ratio,
        tall_region_ratio=args.tall_region_ratio,
    )


def process_parent(parent, rough_page, line_page, args):
    ca = child_args(args)
    line_parent_box = parse_box(parent["line_box"])
    rough_parent_box = parse_box(parent["rough_box"])
    lx0, ly0, lx1, ly1 = line_parent_box
    rx0, ry0, rx1, ry1 = rough_parent_box
    line_parent = line_page[ly0:ly1, lx0:lx1]
    rough_parent = rough_page[ry0:ry1, rx0:rx1]
    if line_parent.size == 0 or rough_parent.size == 0:
        return []

    rows = []
    proposals = mh.region_proposals(line_parent, ca)
    for child_index, proposal in enumerate(proposals, 1):
        local_line_box = proposal["line_box"]
        local_width = local_line_box[2] - local_line_box[0]
        local_height = local_line_box[3] - local_line_box[1]
        parent_width = lx1 - lx0
        parent_height = ly1 - ly0
        if (
            local_width >= parent_width * args.max_child_parent_ratio
            and local_height >= parent_height * args.max_child_parent_ratio
        ):
            continue
        match = mh.match_region(rough_parent, line_parent, proposal, ca)
        if match is None:
            continue
        line_box = translate_box(match["line_box"], lx0, ly0)
        rough_box = translate_box(match["rough_box"], rx0, ry0)
        rough_norm, line_norm, out_w, out_h = mh.normalize_pair(rough_page, line_page, rough_box, line_box, ca)
        if rough_norm is None or line_norm is None:
            continue
        decision = "candidate"
        if (
            match["match_score"] < args.min_match_score
            or match["edge_f1"] < args.min_f1
            or match["chamfer"] > args.max_chamfer
            or match["rough_std"] < args.min_rough_std
        ):
            decision = "review_low_score"
        row = {
            **match,
            "page": parent["page"],
            "source_page": parent["source_page"],
            "parent_rank": parent["rank"],
            "parent_region_index": parent["region_index"],
            "parent_line_box": line_parent_box,
            "parent_rough_box": rough_parent_box,
            "parent_feature_tags": parent.get("feature_tags", ""),
            "child_index": child_index,
            "region_index": f'{parent["region_index"]}.{child_index}',
            "line_box": line_box,
            "rough_box": rough_box,
            "output_width": out_w,
            "output_height": out_h,
            "black_fill_ratio": float((line_norm < args.black_threshold).mean()),
            "decision": decision,
            "rough_output": rough_norm,
            "line_output": line_norm,
        }
        row["feature_tags"] = mh.feature_tags(row, ca)
        rows.append(row)
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    return rows[: args.max_children_per_parent] if args.max_children_per_parent else rows


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
        "rank", "page", "source_page", "parent_rank", "parent_region_index",
        "child_index", "decision", "line_box", "rough_box",
        "parent_line_box", "parent_rough_box", "output_width", "output_height",
        "scale", "dx", "dy", "line_ink", "region_area",
        "match_score", "edge_f1", "edge_precision", "edge_recall",
        "chamfer", "projection_corr", "orientation_entropy", "rough_std",
        "black_fill_ratio", "feature_tags", "parent_feature_tags",
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
            f'{row["page"]} p{row["parent_rank"]}/c{row["child_index"]} {row["decision"]} '
            f'score={row["match_score"]:.2f} F1={row["edge_f1"]:.2f} '
            f'black={row["black_fill_ratio"]:.2f} tags={row["feature_tags"]}'
        )
        draw.text((3, top + thumb + 2), text, fill="black", font=font)
    Path(args.qc_out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.qc_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=mh.DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=mh.DEFAULT_ZIP_ROOT)
    parser.add_argument("--parents", default=DEFAULT_PARENTS)
    parser.add_argument("--limit-parents", type=int, default=0)
    parser.add_argument("--parent-long-side", type=int, default=3000)
    parser.add_argument("--parent-area", type=int, default=5_000_000)
    parser.add_argument("--line-threshold", type=int, default=192)
    parser.add_argument("--min-line-component-area", type=int, default=10)
    parser.add_argument("--region-close", type=int, default=61)
    parser.add_argument("--line-margin", type=float, default=0.14)
    parser.add_argument("--min-region-area", type=int, default=1200)
    parser.add_argument("--min-region-size", type=int, default=120)
    parser.add_argument("--max-region-ratio", type=float, default=0.82)
    parser.add_argument("--min-line-ink", type=float, default=0.004)
    parser.add_argument("--max-line-ink", type=float, default=0.28)
    parser.add_argument("--max-children-per-parent", type=int, default=8)
    parser.add_argument("--max-child-parent-ratio", type=float, default=0.86)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.90, 1.0, 1.10])
    parser.add_argument("--search-px", type=int, default=80)
    parser.add_argument("--search-ratio", type=float, default=0.12)
    parser.add_argument("--search-step", type=int, default=48)
    parser.add_argument("--search-step-ratio", type=float, default=0.09)
    parser.add_argument("--match-size", type=int, default=320)
    parser.add_argument("--output-long-side", type=int, default=768)
    parser.add_argument("--close-px", type=int, default=5)
    parser.add_argument("--truncate-px", type=int, default=36)
    parser.add_argument("--min-match-edges", type=int, default=80)
    parser.add_argument("--min-match-score", type=float, default=0.45)
    parser.add_argument("--min-f1", type=float, default=0.12)
    parser.add_argument("--max-chamfer", type=float, default=26.0)
    parser.add_argument("--min-rough-std", type=float, default=8.0)
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
    args = parser.parse_args()

    parents = read_parents(args.parents, args)
    manifest = load_manifest(args.zip_path, args.zip_root)
    rows = []
    with zipfile.ZipFile(args.zip_path) as zf:
        cache = {}
        for index, parent in enumerate(parents, 1):
            page = parent["page"]
            if page not in cache:
                cache[page] = mh.load_pair(zf, args.zip_root, manifest[page])
            rough_page, line_page = cache[page]
            child_rows = process_parent(parent, rough_page, line_page, args)
            rows.extend(child_rows)
            print(
                f"{index}/{len(parents)} parent_rank={parent['rank']} "
                f"page={page} children={len(child_rows)} total={len(rows)}",
                flush=True,
            )
    rows.sort(key=lambda item: item["match_score"], reverse=True)
    write_rows(rows, args)
    make_qc(rows, args)
    print(f"parents={len(parents)} children={len(rows)}")
    print(f"wrote: {args.csv_out}, {args.json_out}, {args.qc_out}")


if __name__ == "__main__":
    main()
