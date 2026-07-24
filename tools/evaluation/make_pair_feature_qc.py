"""Create rough/line/edge-overlay QC montages from pair feature scores."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SIZE = 480


def load_gray(path):
    image = Image.open(path).convert("L")
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))


def edge_array(image):
    gray = np.asarray(image, dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def overlay_image(rough, line):
    overlay = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    rough_edge = edge_array(rough)
    line_edge = edge_array(line)
    both = rough_edge & line_edge
    overlay[rough_edge] = (235, 55, 55)
    overlay[line_edge] = (55, 90, 235)
    overlay[both] = (45, 170, 80)
    return Image.fromarray(overlay)


def fit_text(draw, text, max_width, font_path):
    for size in range(15, 8, -1):
        font = ImageFont.truetype(font_path, size)
        if draw.textlength(text, font=font) <= max_width:
            return font
    return ImageFont.truetype(font_path, 9)


def make_montage(rows, output, rough_dir, line_dir, columns=2, thumb=220):
    label_h = 44
    tile_w = thumb * 3
    tile_h = thumb + label_h
    canvas = Image.new("RGB", (tile_w * columns, tile_h * ((len(rows) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 11)
    small_font = ImageFont.truetype(font_path, 10)

    for idx, row in enumerate(rows):
        grid_x = (idx % columns) * tile_w
        grid_y = (idx // columns) * tile_h
        base = row["base"]
        rough = load_gray(Path(rough_dir) / f"{base}.jpg")
        line = load_gray(Path(line_dir) / f"{base}.jpg")
        images = [
            ImageOps.autocontrast(rough),
            line,
            overlay_image(rough, line),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB").resize((thumb, thumb)), (grid_x + col * thumb, grid_y))
        title = (
            f"{row['name']} src={row['source_prefix']} list={row['list_label']} "
            f"score={float(row['agreement_score']):.3f} f1={float(row['edge_f1']):.3f}"
        )
        draw.text((grid_x + 4, grid_y + thumb + 2), title, fill="black", font=fit_text(draw, title, tile_w - 8, font_path))
        detail = (
            f"ch={float(row['agreement_chamfer']):.1f} "
            f"bg={float(row['rough_background_haze_ink']):.3f} "
            f"near={float(row['rough_line_near_uncertainty_ink']):.3f} "
            f"black={float(row['black_fill_score']):.3f}"
        )
        draw.text((grid_x + 4, grid_y + thumb + 23), detail, fill=(40, 40, 40), font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--name-list")
    parser.add_argument("--group", default="agreement_low")
    parser.add_argument("--source")
    parser.add_argument("--sort-key", default="agreement_score")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.feature_csv)))
    allowed = None
    if args.name_list:
        with open(args.name_list) as file:
            allowed = {line.strip() for line in file if line.strip()}
    seen = set()
    filtered = []
    group_key = f"{args.group.rsplit('_', 1)[0]}_group" if args.group.endswith(("_high", "_mid", "_low")) else None
    group_value = args.group.rsplit("_", 1)[1] if group_key else None
    for row in rows:
        if row["name"] in seen:
            continue
        if allowed is not None and row["name"] not in allowed:
            continue
        if group_key and row.get(group_key) != group_value:
            continue
        if args.source and row["source_prefix"] != args.source:
            continue
        seen.add(row["name"])
        filtered.append(row)
    filtered.sort(key=lambda row: float(row[args.sort_key]), reverse=args.reverse)
    make_montage(
        filtered[: args.count],
        Path(args.output),
        args.rough_dir,
        args.line_dir,
    )
    print(f"rows={len(filtered)} shown={min(args.count, len(filtered))}")


if __name__ == "__main__":
    main()
