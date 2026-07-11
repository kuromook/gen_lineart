"""Run fixed-sample warm_regions inference and build a baseline comparison."""

import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps

from unetgenerator import UNetGenerator


IMAGE_SIZE = 480
CHECKPOINT = "checkpoints/warm_regions/best.pth"
OUTPUT_DIR = "results/warm_regions"
COMPARE_PATH = "results/compare_warm_regions.png"
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


def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for name in SAMPLES:
            rough = load_image(dataset_path(name, "rough"), autocontrast=True)
            tensor = TF.to_tensor(rough).unsqueeze(0).to(device)
            output = (1.0 - torch.sigmoid(model(tensor))).clamp(0, 1)
            path = os.path.join(OUTPUT_DIR, f"{name}_out.png")
            TF.to_pil_image(output[0].cpu()).save(path)
            print(f"saved: {path}", flush=True)


def build_comparison():
    columns = [
        ("rough", None),
        ("std15", "results/std15"),
        ("warm2", "results/warm2"),
        ("warm_regions", OUTPUT_DIR),
        ("line (GT)", None),
    ]
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

    for col, (title, _) in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * IMAGE_SIZE + (IMAGE_SIZE - width) / 2, 6), title, fill=0, font=font)

    for row, name in enumerate(SAMPLES):
        y = header_h + row * (IMAGE_SIZE + label_h)
        images = [
            load_image(dataset_path(name, "rough"), autocontrast=True),
            load_image(f"results/std15/{name}_out.png"),
            load_image(f"results/warm2/{name}_out.png"),
            load_image(os.path.join(OUTPUT_DIR, f"{name}_out.png")),
            load_image(dataset_path(name, "line")),
        ]
        for col, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), name, fill=0, font=small_font)

    canvas.save(COMPARE_PATH)
    print(f"saved comparison: {COMPARE_PATH}", flush=True)


if __name__ == "__main__":
    run_inference()
    build_comparison()
