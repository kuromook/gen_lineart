"""Build a rough/model/GT montage for a model output directory."""

import argparse
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SIZE = 480


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def load_image(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = read_sample_list(args.sample_list)
    columns = ["rough", args.model_label, "line (GT)"]
    header_h = 28
    label_h = 22
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * len(columns), header_h + (IMAGE_SIZE + label_h) * len(names)),
        (200, 200, 200),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small_font = ImageFont.load_default()

    for col, title in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * IMAGE_SIZE + (IMAGE_SIZE - width) / 2, 6), title, fill=0, font=font)

    for row, name in enumerate(names):
        base = normalize_name(name)
        y = header_h + row * (IMAGE_SIZE + label_h)
        images = [
            load_image(f"dataset/pairs_480/{args.split}/rough/{base}.jpg", autocontrast=True),
            load_image(f"{args.model_dir}/{base}_out.png"),
            load_image(f"dataset/pairs_480/{args.split}/line/{base}.jpg"),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), base, fill=0, font=small_font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
