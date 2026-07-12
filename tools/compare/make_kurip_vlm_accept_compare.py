"""Run and montage the Qwen3VL-filtered kurip fine-tune."""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lineart.unetgenerator import UNetGenerator


IMAGE_SIZE = 480
VLM_CKPT = "checkpoints/kurip_vlm_accept_top500_noac/best.pth"
SHAPE1_CKPT = "checkpoints/shape1/best.pth"
VLM_CSV = "results/kurip_vlm_candidates_top500_review_qwen3vl.csv"
DEFAULT_TRAIN_LIST = "dataset/pairs_480/valid_train_kurip_vlm_accept_top500.txt"
DEFAULT_LINE_DIR = "dataset/pairs_480/train/line_kurip_vlm_candidates_top500_clean_t192_cc8"
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
    return f"dataset/pairs_480/{split}/{kind}/{name}.jpg"


def vlm_line_path(name, line_dir):
    return f"{line_dir}/{name}.jpg"


def draw_montage(samples, image_paths, output_path, model_label):
    columns = ["rough", "shape1", model_label, "line (GT)"]
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


def vlm_samples(count=8, train_list=None):
    if train_list:
        with open(train_list) as file:
            return [line.strip().replace(".jpg", "") for line in file if line.strip()][:count]
    with open(VLM_CSV, newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["vlm_decision"] == "accept"]
    return [row["name"].replace(".jpg", "") for row in rows[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=VLM_CKPT)
    parser.add_argument("--tag", default="kurip_vlm_accept_top500_noac")
    parser.add_argument("--label", default="vlm_accept_noac")
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--line-dir", default=DEFAULT_LINE_DIR)
    parser.add_argument("--train-sample-count", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape1 = load_model(SHAPE1_CKPT, device)
    vlm = load_model(args.checkpoint, device)

    for name in FIXED_SAMPLES:
        rough = fixed_dataset_path(name, "rough")
        infer(shape1, rough, f"results/shape1_noac/{name}_out.png", device)
        infer(vlm, rough, f"results/{args.tag}/{name}_out.png", device)

    train_samples = vlm_samples(args.train_sample_count, args.train_list)
    for name in train_samples:
        rough = f"dataset/pairs_480/train/rough/{name}.jpg"
        infer(shape1, rough, f"results/shape1_kurip_vlm_accept_samples/{name}_out.png", device)
        infer(vlm, rough, f"results/{args.tag}_samples/{name}_out.png", device)

    draw_montage(
        FIXED_SAMPLES,
        lambda name: [
            fixed_dataset_path(name, "rough"),
            f"results/shape1_noac/{name}_out.png",
            f"results/{args.tag}/{name}_out.png",
            fixed_dataset_path(name, "line"),
        ],
        f"results/compare_{args.tag}_vs_shape1.png",
        args.label,
    )
    draw_montage(
        train_samples,
        lambda name: [
            f"dataset/pairs_480/train/rough/{name}.jpg",
            f"results/shape1_kurip_vlm_accept_samples/{name}_out.png",
            f"results/{args.tag}_samples/{name}_out.png",
            vlm_line_path(name, args.line_dir),
        ],
        f"results/compare_{args.tag}_train_vs_shape1.png",
        args.label,
    )


if __name__ == "__main__":
    main()
