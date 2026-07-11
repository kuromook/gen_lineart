"""ako5 モデルの推論を実行し、std15 / 正解線画と並べた比較タイル画像を生成する。

列構成: rough(入力, autocontrast後) | std15 | ako5 | line(正解)
行: std実験と同じ8サンプル
出力: results/ako5/<name>_out.png（個別） と results/compare_ako5_vs_std15.png（比較）
"""
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lineart.unetgenerator import UNetGenerator

IMAGE_SIZE = 480
CKPT = "checkpoints/ako5/best.pth"
OUT_DIR = "results/ako5"
STD15_DIR = "results/std15"
COMPARE_PATH = "results/compare_ako5_vs_std15.png"

SAMPLES = [
    "housei_002_06_15", "housei_002_07_12", "housei_002_19_12",
    "lineart_004_002", "lineart_004_004", "lineart_004_006",
    "lineart_004_008", "lineart_004_010",
]


def rough_path(name):
    sub = "train" if name.startswith("housei") else "test"
    return f"dataset/pairs_480/{sub}/rough/{name}.jpg"


def line_path(name):
    sub = "train" if name.startswith("housei") else "test"
    return f"dataset/pairs_480/{sub}/line/{name}.jpg"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    rough_imgs, ako5_imgs, std15_imgs, line_imgs = {}, {}, {}, {}

    for name in SAMPLES:
        img = Image.open(rough_path(name)).convert("L")
        img = ImageOps.autocontrast(img, cutoff=0)
        rough_resized = TF.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        rough_imgs[name] = rough_resized

        x = TF.to_tensor(rough_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(x))
        out = (1.0 - out).clamp(0, 1)
        out_pil = TF.to_pil_image(out[0].cpu())
        out_pil.save(os.path.join(OUT_DIR, f"{name}_out.png"))
        ako5_imgs[name] = out_pil

        std15 = Image.open(os.path.join(STD15_DIR, f"{name}_out.png")).convert("L")
        std15_imgs[name] = TF.resize(std15, (IMAGE_SIZE, IMAGE_SIZE))

        gt = Image.open(line_path(name)).convert("L")
        line_imgs[name] = TF.resize(gt, (IMAGE_SIZE, IMAGE_SIZE))
        print(f"done: {name}")

    # --- モンタージュ生成 ---
    cols = ["rough", "std15", "ako5", "line (GT)"]
    ncol = len(cols)
    cell = IMAGE_SIZE
    header_h = 28
    label_h = 22
    bg = (200, 200, 200)

    grid_w = cell * ncol
    grid_h = header_h + (cell + label_h) * len(SAMPLES)
    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = font_s = ImageFont.load_default()

    for c, title in enumerate(cols):
        w = draw.textlength(title, font=font)
        draw.text((c * cell + (cell - w) / 2, 6), title, fill=(0, 0, 0), font=font)

    for r, name in enumerate(SAMPLES):
        y = header_h + r * (cell + label_h)
        for c, src in enumerate([rough_imgs, std15_imgs, ako5_imgs, line_imgs]):
            canvas.paste(src[name].convert("RGB"), (c * cell, y))
        draw.text((6, y + cell + 3), name, fill=(0, 0, 0), font=font_s)

    canvas.save(COMPARE_PATH)
    print(f"\nsaved compare: {COMPARE_PATH}  ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
