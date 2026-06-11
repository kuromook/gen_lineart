"""warm-start再学習(warm1/warm2)を std15・正解線画と並べた比較タイル画像を生成する。

すべて既存の推論PNGを並べるだけ（推論は再実行しない）。
列構成: rough(入力, autocontrast後) | std15 | warm1 | warm2 | line(正解)
行: std実験と同じ8サンプル
出力: results/compare_warm_vs_std15.png
"""
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageDraw, ImageFont

IMAGE_SIZE = 480
STD15_DIR = "results/std15"
WARM1_DIR = "results/warm1"
WARM2_DIR = "results/warm2"
COMPARE_PATH = "results/compare_warm_vs_std15.png"

SAMPLES = [
    "housei_002_06_15", "housei_002_07_12", "housei_002_19_12",
    "lineart_004_002", "lineart_004_004", "lineart_004_006",
    "lineart_004_008", "lineart_004_010",
]


def rough_path(name):
    sub = "train" if name.startswith("housei") else "test"
    return f"dataset_480/{sub}/rough/{name}.jpg"


def line_path(name):
    sub = "train" if name.startswith("housei") else "test"
    return f"dataset_480/{sub}/line/{name}.jpg"


def load_resized(path, gray=True):
    img = Image.open(path)
    img = img.convert("L") if gray else img
    return TF.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


def main():
    rough_imgs, std15_imgs, warm1_imgs, warm2_imgs, line_imgs = {}, {}, {}, {}, {}

    for name in SAMPLES:
        rough = Image.open(rough_path(name)).convert("L")
        rough = ImageOps.autocontrast(rough, cutoff=0)
        rough_imgs[name] = TF.resize(rough, (IMAGE_SIZE, IMAGE_SIZE))

        std15_imgs[name] = load_resized(os.path.join(STD15_DIR, f"{name}_out.png"))
        warm1_imgs[name] = load_resized(os.path.join(WARM1_DIR, f"{name}_out.png"))
        warm2_imgs[name] = load_resized(os.path.join(WARM2_DIR, f"{name}_out.png"))
        line_imgs[name] = load_resized(line_path(name))
        print(f"done: {name}")

    # --- モンタージュ生成 ---
    cols = ["rough", "std15", "warm1", "warm2", "line (GT)"]
    sources = [rough_imgs, std15_imgs, warm1_imgs, warm2_imgs, line_imgs]
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
        for c, src in enumerate(sources):
            canvas.paste(src[name].convert("RGB"), (c * cell, y))
        draw.text((6, y + cell + 3), name, fill=(0, 0, 0), font=font_s)

    canvas.save(COMPARE_PATH)
    print(f"\nsaved compare: {COMPARE_PATH}  ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
