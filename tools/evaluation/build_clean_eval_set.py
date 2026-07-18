"""Build a leak-free fixed evaluation set from test tiles with sane ink stats."""

import argparse
import csv
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_DIR = Path("dataset/pairs_480")
DEFAULT_TRAIN_LIST = DATASET_DIR / "valid_train_warm_regions_clean_split.txt"
DEFAULT_TEST_LIST = DATASET_DIR / "valid_test.txt"
DEFAULT_OUTPUT_LIST = DATASET_DIR / "eval_fixed_clean_balanced.txt"
DEFAULT_OUTPUT_CSV = Path("results/clean_eval_candidates.csv")
DEFAULT_OUTPUT_MONTAGE = Path("results/clean_eval_candidates.png")
IMAGE_SIZE = 240


def read_names(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_tile(name):
    base = Path(name).stem
    match = re.fullmatch(r"orig_(\d{3})-(\d+)", base)
    if match:
        page, tile = match.groups()
        return f"lineart_{page}_{int(tile):03d}"
    return base


def canonical_page(name):
    base = Path(name).stem
    match = re.fullmatch(r"orig_(\d{3})-\d+", base)
    if match:
        return f"lineart_{match.group(1)}"
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] in {"housei", "lineart"}:
        return "_".join(parts[:2])
    return parts[0]


def image_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return {
        "std": float(image.std()),
        "ink": float((image < 128).mean()),
    }


def collect_rows(args):
    train_names = read_names(Path(args.train_list))
    test_names = read_names(Path(args.test_list))
    train_tiles = {canonical_tile(name) for name in train_names}
    train_pages = {canonical_page(name) for name in train_names}

    rows = []
    for name in test_names:
        rough_path = DATASET_DIR / "test/rough" / name
        line_path = DATASET_DIR / "test/line" / name
        rough_stats = image_stats(rough_path)
        line_stats = image_stats(line_path)
        tile = canonical_tile(name)
        page = canonical_page(name)
        rows.append({
            "name": name,
            "page": page,
            "canonical_tile": tile,
            "rough_std": rough_stats["std"],
            "rough_ink": rough_stats["ink"],
            "line_std": line_stats["std"],
            "line_ink": line_stats["ink"],
            "tile_leak": tile in train_tiles,
            "page_leak": page in train_pages,
        })
    return rows


def choose(rows, args):
    candidates = [
        row for row in rows
        if not row["tile_leak"]
        and not row["page_leak"]
        and args.min_line_ink <= row["line_ink"] <= args.max_line_ink
        and row["rough_std"] >= args.min_rough_std
    ]
    by_page = {}
    for row in candidates:
        by_page.setdefault(row["page"], []).append(row)

    selected = []
    for page in sorted(by_page):
        page_rows = sorted(
            by_page[page],
            key=lambda row: (abs(row["line_ink"] - args.target_line_ink), -row["rough_std"], row["name"]),
        )
        selected.extend(page_rows[: args.max_per_page])
    selected = sorted(
        selected,
        key=lambda row: (abs(row["line_ink"] - args.target_line_ink), -row["rough_std"], row["name"]),
    )
    return selected[: args.count]


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "page",
        "canonical_tile",
        "rough_std",
        "rough_ink",
        "line_std",
        "line_ink",
        "tile_leak",
        "page_leak",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_tile(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


def make_montage(rows, path):
    cols = ["rough", "line"]
    header_h = 28
    label_h = 22
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * len(cols), header_h + (IMAGE_SIZE + label_h) * len(rows)),
        (210, 210, 210),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = small_font = ImageFont.load_default()

    for col, title in enumerate(cols):
        width = draw.textlength(title, font=font)
        draw.text((col * IMAGE_SIZE + (IMAGE_SIZE - width) / 2, 6), title, fill=0, font=font)

    for row_idx, row in enumerate(rows):
        name = row["name"]
        y = header_h + row_idx * (IMAGE_SIZE + label_h)
        images = [
            load_tile(DATASET_DIR / "test/rough" / name, autocontrast=True),
            load_tile(DATASET_DIR / "test/line" / name),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        label = f"{Path(name).stem} ink={row['line_ink']:.3f} std={row['rough_std']:.1f}"
        draw.text((6, y + IMAGE_SIZE + 4), label, fill=0, font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--test-list", default=str(DEFAULT_TEST_LIST))
    parser.add_argument("--output-list", default=str(DEFAULT_OUTPUT_LIST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-montage", default=str(DEFAULT_OUTPUT_MONTAGE))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-per-page", type=int, default=4)
    parser.add_argument("--min-line-ink", type=float, default=0.02)
    parser.add_argument("--max-line-ink", type=float, default=0.12)
    parser.add_argument("--target-line-ink", type=float, default=0.055)
    parser.add_argument("--min-rough-std", type=float, default=18.0)
    args = parser.parse_args()

    rows = collect_rows(args)
    selected = choose(rows, args)
    write_csv(rows, Path(args.output_csv))
    Path(args.output_list).write_text("".join(f"{row['name']}\n" for row in selected))
    make_montage(selected, Path(args.output_montage))
    print(f"rows: {len(rows)}")
    print(f"selected: {len(selected)}")
    print(f"output_list: {args.output_list}")
    print(f"output_csv: {args.output_csv}")
    print(f"output_montage: {args.output_montage}")


if __name__ == "__main__":
    main()
