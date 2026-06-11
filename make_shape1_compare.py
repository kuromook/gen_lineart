"""Build a fixed-sample comparison montage for the shape1 experiment."""

import os

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SIZE = 480
COMPARE_PATH = "results/compare_shape1.png"
SAMPLES = [
    "housei_002_06_15",
    "housei_002_07_12",
    "housei_002_19_12",
    "lineart_004_002",
    "lineart_004_004",
    "lineart_004_006",
    "lineart_004_008",
    "lineart_004_010",
]


def dataset_path(name, kind):
    split = "train" if name.startswith("housei") else "test"
    return f"dataset_480/{split}/{kind}/{name}.jpg"


def load_image(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


def main():
    columns = ["rough", "std15", "warm_regions", "shape1", "line (GT)"]
    header_h = 28
    label_h = 22
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * len(columns), header_h + (IMAGE_SIZE + label_h) * len(SAMPLES)),
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

    for row, name in enumerate(SAMPLES):
        y = header_h + row * (IMAGE_SIZE + label_h)
        images = [
            load_image(dataset_path(name, "rough"), autocontrast=True),
            load_image(f"results/std15/{name}_out.png"),
            load_image(f"results/warm_regions/{name}_out.png"),
            load_image(f"results/shape1/{name}_out.png"),
            load_image(dataset_path(name, "line")),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), name, fill=0, font=small_font)

    os.makedirs(os.path.dirname(COMPARE_PATH), exist_ok=True)
    canvas.save(COMPARE_PATH)
    print(f"saved: {COMPARE_PATH}")


if __name__ == "__main__":
    main()
