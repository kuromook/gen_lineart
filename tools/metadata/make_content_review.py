"""Create a manual content-category review sheet from pair metadata."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_METADATA = "dataset/pairs_480/pair_metadata.csv"
DEFAULT_REVIEW_CSV = "dataset/pairs_480/reviews/kurip_refined_content_review.csv"
DEFAULT_MONTAGE = "results/kurip_refined_content_review.png"
IMAGE_SIZE = 480


def load_rows(path, dataset_source, alignment_quality):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if dataset_source:
        rows = [row for row in rows if row["dataset_source"] == dataset_source]
    if alignment_quality:
        rows = [row for row in rows if row["alignment_quality"] == alignment_quality]
    return rows


def load_gray(path):
    image = Image.open(path).convert("L")
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


def edge_overlay(rough, line):
    rough_np = np.asarray(rough)
    line_np = np.asarray(line)
    rough_edge = cv2.Canny(cv2.GaussianBlur(rough_np, (0, 0), 1.0), 45, 135) > 0
    line_edge = cv2.Canny(cv2.GaussianBlur(line_np, (0, 0), 1.0), 45, 135) > 0
    overlay = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, np.uint8)
    overlay[rough_edge] = (255, 60, 60)
    overlay[line_edge] = (40, 80, 255)
    return Image.fromarray(overlay)


def write_review_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "suggested_content_category",
        "content_category",
        "pair_quality",
        "alignment_quality",
        "line_ink",
        "rough_std",
        "rough_path",
        "line_path",
        "review_content_category",
        "review_keep",
        "review_notes",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "name": row["name"],
                "suggested_content_category": row["content_category"],
                "content_category": row["content_category"],
                "pair_quality": row["pair_quality"],
                "alignment_quality": row["alignment_quality"],
                "line_ink": row["line_ink"],
                "rough_std": row["rough_std"],
                "rough_path": row["rough_path"],
                "line_path": row["line_path"],
                "review_content_category": "",
                "review_keep": "",
                "review_notes": "",
            })


def make_montage(rows, path, count):
    picks = rows[:count]
    if not picks:
        return
    thumb = 220
    header_h = 24
    label_h = 34
    columns = ["rough", "line", "overlay"]
    canvas = Image.new("RGB", (thumb * len(columns), header_h + (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = small_font = ImageFont.load_default()

    for col, title in enumerate(columns):
        draw.text((col * thumb + 4, 4), title, fill=0, font=font)

    for row_index, row in enumerate(picks):
        y = header_h + row_index * (thumb + label_h)
        rough = load_gray(row["rough_path"])
        line = load_gray(row["line_path"])
        images = [rough, line, edge_overlay(rough, line)]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB").resize((thumb, thumb)), (col * thumb, y))
        label = (
            f'{row["name"]} {row["content_category"]} '
            f'ink={float(row["line_ink"]):.3f} std={float(row["rough_std"]):.1f}'
        )
        draw.text((3, y + thumb + 2), label, fill=0, font=small_font)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--dataset-source", default="kurip")
    parser.add_argument("--alignment-quality", default="locally_refined")
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--montage", default=DEFAULT_MONTAGE)
    parser.add_argument("--montage-count", type=int, default=80)
    args = parser.parse_args()

    rows = load_rows(args.metadata, args.dataset_source, args.alignment_quality)
    write_review_csv(rows, args.review_csv)
    make_montage(rows, args.montage, args.montage_count)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.review_csv}")
    print(f"wrote: {args.montage}")


if __name__ == "__main__":
    main()
