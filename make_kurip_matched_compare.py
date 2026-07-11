"""Run and montage the matched-kurip no-autocontrast fine-tune."""

import csv
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from unetgenerator import UNetGenerator


IMAGE_SIZE = 480
MATCHED_CKPT = "checkpoints/kurip_matched_strict_x4_noac/best.pth"
SHAPE1_CKPT = "checkpoints/shape1/best.pth"
MATCHED_CSV = "results/kurip_matched_tiles_strict_all_noac.csv"
FIXED_COMPARE_PATH = "results/compare_kurip_matched_strict_x4_noac_vs_shape1.png"
MATCHED_COMPARE_PATH = "results/compare_kurip_matched_strict_x4_noac_train_vs_shape1.png"
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


def load_model(checkpoint, device):
    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def infer(model, image_path, output_path, device):
    image = Image.open(image_path).convert("L")
    tensor = TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch.sigmoid(model(tensor))
    out = (1.0 - out).clamp(0, 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(out[0].cpu()).save(output_path)


def load_image(path):
    return TF.resize(Image.open(path).convert("L"), (IMAGE_SIZE, IMAGE_SIZE))


def fixed_dataset_path(name, kind):
    split = "train" if name.startswith("housei") else "test"
    return f"dataset_480/{split}/{kind}/{name}.jpg"


def matched_line_path(name):
    return f"dataset_480/train/line_kurip_matched_clean_t192_cc8/{name}.jpg"


def draw_montage(samples, image_paths, output_path):
    columns = ["rough", "shape1", "matched_x4_noac", "line (GT)"]
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
            canvas.paste(load_image(path).convert("RGB"), (col * IMAGE_SIZE, y))
        draw.text((6, y + IMAGE_SIZE + 3), name, fill=0, font=small_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"saved: {output_path}")


def matched_samples(count=8):
    with open(MATCHED_CSV, newline="") as file:
        rows = list(csv.DictReader(file))
    return [row["name"].replace(".jpg", "") for row in rows[:count]]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape1 = load_model(SHAPE1_CKPT, device)
    matched = load_model(MATCHED_CKPT, device)

    for name in FIXED_SAMPLES:
        rough = fixed_dataset_path(name, "rough")
        infer(shape1, rough, f"results/shape1_noac/{name}_out.png", device)
        infer(matched, rough, f"results/kurip_matched_strict_x4_noac/{name}_out.png", device)

    train_samples = matched_samples()
    for name in train_samples:
        rough = f"dataset_480/train/rough/{name}.jpg"
        infer(shape1, rough, f"results/shape1_kurip_matched_samples/{name}_out.png", device)
        infer(matched, rough, f"results/kurip_matched_strict_x4_noac_samples/{name}_out.png", device)

    draw_montage(
        FIXED_SAMPLES,
        lambda name: [
            fixed_dataset_path(name, "rough"),
            f"results/shape1_noac/{name}_out.png",
            f"results/kurip_matched_strict_x4_noac/{name}_out.png",
            fixed_dataset_path(name, "line"),
        ],
        FIXED_COMPARE_PATH,
    )
    draw_montage(
        train_samples,
        lambda name: [
            f"dataset_480/train/rough/{name}.jpg",
            f"results/shape1_kurip_matched_samples/{name}_out.png",
            f"results/kurip_matched_strict_x4_noac_samples/{name}_out.png",
            matched_line_path(name),
        ],
        MATCHED_COMPARE_PATH,
    )


if __name__ == "__main__":
    main()
