"""Build a rough/model(s)/GT montage for an evaluation sample list."""

import argparse
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SIZE = 480
HEADER_FONT_SIZE = 60
HEADER_MIN_FONT_SIZE = 24
HEADER_H = 88


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def parse_model(value):
    if "=" in value:
        label, directory = value.split("=", 1)
        return label, Path(directory)
    return value, Path("results") / value


def load_image(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


def maybe_load_model_image(directory, base):
    path = directory / f"{base}_out.png"
    if path.exists():
        return load_image(path)
    return None


def make_missing_tile(label):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    text = f"missing\n{label}"
    y = IMAGE_SIZE // 2 - 24
    for line in text.splitlines():
        width = draw.textlength(line, font=font)
        draw.text(((IMAGE_SIZE - width) / 2, y), line, fill=(80, 80, 80), font=font)
        y += 26
    return image


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_font(draw, text, max_width):
    for size in range(HEADER_FONT_SIZE, HEADER_MIN_FONT_SIZE - 1, -2):
        font = load_font(size)
        if draw.textlength(text, font=font) <= max_width:
            return font
    return load_font(HEADER_MIN_FONT_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--model", action="append", required=True, help="LABEL=DIR or model name under results/")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = read_sample_list(args.sample_list)
    models = [parse_model(value) for value in args.model]
    columns = ["rough", *[label for label, _ in models], "line (GT)"]
    header_h = HEADER_H
    label_h = 22
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * len(columns), header_h + (IMAGE_SIZE + label_h) * len(names)),
        (200, 200, 200),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        small_font = ImageFont.load_default()

    for col, title in enumerate(columns):
        clipped = title if len(title) <= 34 else f"{title[:31]}..."
        font = fit_font(draw, clipped, IMAGE_SIZE - 16)
        width = draw.textlength(clipped, font=font)
        y_text = (header_h - font.size) / 2 if hasattr(font, "size") else 8
        draw.text((col * IMAGE_SIZE + (IMAGE_SIZE - width) / 2, y_text), clipped, fill=0, font=font)

    for row, name in enumerate(names):
        base = normalize_name(name)
        y = header_h + row * (IMAGE_SIZE + label_h)
        images = [
            load_image(f"dataset/pairs_480/{args.split}/rough/{base}.jpg", autocontrast=True),
        ]
        for label, directory in models:
            model_image = maybe_load_model_image(directory, base)
            images.append(model_image if model_image is not None else make_missing_tile(label))
        images.append(load_image(f"dataset/pairs_480/{args.split}/line/{base}.jpg"))

        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), base, fill=0, font=small_font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
