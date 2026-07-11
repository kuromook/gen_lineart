"""Build comparison montages for routed inference."""

import os

import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SIZE = 480
FIXED_COMPARE_PATH = "results/compare_routed_vs_shape1.png"
KURIP_COMPARE_PATH = "results/compare_routed_kurip_vs_shape1.png"
FIXED_SAMPLES = [
    "housei_002_06_15",
    "housei_002_07_12",
    "housei_002_19_12",
    "lineart_004_002",
    "lineart_004_004",
    "lineart_004_006",
    "lineart_004_008",
    "lineart_004_010",
]
KURIP_SAMPLES = [
    "kurip_0009_1200_1440",
    "kurip_0020_1200_4320",
    "kurip_0018_2640_1200",
    "kurip_0013_0240_2160",
    "kurip_0030_4080_1440",
    "kurip_0011_2640_0720",
    "kurip_0036_3600_3840",
    "kurip_0011_2880_4560",
]


def load_image(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


def fixed_dataset_path(name, kind):
    split = "train" if name.startswith("housei") else "test"
    return f"dataset_480/{split}/{kind}/{name}.jpg"


def train_dataset_path(name, kind):
    return f"dataset_480/train/{kind}/{name}.jpg"


def draw_montage(samples, image_paths, output_path):
    columns = ["rough", "shape1", "clean540", "routed", "line (GT)"]
    header_h = 28
    label_h = 22
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * len(columns), header_h + (IMAGE_SIZE + label_h) * len(samples)),
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

    for row, name in enumerate(samples):
        y = header_h + row * (IMAGE_SIZE + label_h)
        for col, path in enumerate(image_paths(name)):
            image = load_image(path, autocontrast=(col == 0))
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), name, fill=0, font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"saved: {output_path}")


def main():
    draw_montage(
        FIXED_SAMPLES,
        lambda name: [
            fixed_dataset_path(name, "rough"),
            f"results/shape1/{name}_out.png",
            f"results/kurip_clean540/{name}_out.png",
            f"results/routed/{name}_out.png",
            fixed_dataset_path(name, "line"),
        ],
        FIXED_COMPARE_PATH,
    )
    draw_montage(
        KURIP_SAMPLES,
        lambda name: [
            train_dataset_path(name, "rough"),
            f"results/shape1_kurip_samples/{name}_out.png",
            f"results/kurip_clean540_samples/{name}_out.png",
            f"results/routed_kurip_samples/{name}_out.png",
            train_dataset_path(name, "line"),
        ],
        KURIP_COMPARE_PATH,
    )


if __name__ == "__main__":
    main()
