"""ako5タイルの品質検証: 空白率の統計 + rough/lineペアのモンタージュ生成。"""
import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROUGH_DIR = "dataset/pairs_480/train/rough"
LINE_DIR = "dataset/pairs_480/train/line"
OUT = "results/inspect_ako5_pairs.png"

rough_paths = sorted(glob.glob(os.path.join(ROUGH_DIR, "ako5_*.jpg")))
print(f"ako5 rough tiles: {len(rough_paths)}")

# --- 統計: 空白判定 ---
# blank = 標準偏差が小さい(ほぼ単色) もしくは ほぼ白
rough_std, line_std, line_inkratio = [], [], []
blank_rough = blank_line = 0
for p in rough_paths:
    name = os.path.basename(p)
    r = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    l = np.asarray(Image.open(os.path.join(LINE_DIR, name)).convert("L"), dtype=np.float32)
    rough_std.append(r.std())
    line_std.append(l.std())
    # 線画のインク率(暗いピクセル割合, <128)
    line_inkratio.append((l < 128).mean())
    if r.std() < 5:
        blank_rough += 1
    if l.std() < 5 or (l < 128).mean() < 0.002:
        blank_line += 1

rough_std = np.array(rough_std)
line_std = np.array(line_std)
line_inkratio = np.array(line_inkratio)

print("\n=== 統計 (n={}) ===".format(len(rough_paths)))
print(f"rough std   : min={rough_std.min():.1f} med={np.median(rough_std):.1f} max={rough_std.max():.1f}")
print(f"line  std   : min={line_std.min():.1f} med={np.median(line_std):.1f} max={line_std.max():.1f}")
print(f"line インク率: min={line_inkratio.min()*100:.2f}% med={np.median(line_inkratio)*100:.2f}% max={line_inkratio.max()*100:.2f}%")
print(f"\nほぼ空白 rough (std<5)               : {blank_rough} ({blank_rough/len(rough_paths)*100:.1f}%)")
print(f"ほぼ空白 line  (std<5 or インク<0.2%): {blank_line} ({blank_line/len(rough_paths)*100:.1f}%)")
print(f"線画インク率 <1%  : {(line_inkratio<0.01).sum()} ({(line_inkratio<0.01).mean()*100:.1f}%)")
print(f"線画インク率 <0.5%: {(line_inkratio<0.005).sum()} ({(line_inkratio<0.005).mean()*100:.1f}%)")

# --- モンタージュ: ランダム24ペア ---
random.seed(0)
sample = random.sample(rough_paths, min(24, len(rough_paths)))
THUMB = 200
cols = 6  # 3ペア(rough,line)x... -> 実際は ペアを横並びにして3ペア/行
pairs_per_row = 3
rows = (len(sample) + pairs_per_row - 1) // pairs_per_row
label_h = 16
cell_w = THUMB * 2
cell_h = THUMB + label_h
canvas = Image.new("RGB", (cell_w * pairs_per_row, cell_h * rows), (180, 180, 180))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except OSError:
    font = ImageFont.load_default()

for i, p in enumerate(sample):
    name = os.path.basename(p)
    r = Image.open(p).convert("RGB").resize((THUMB, THUMB))
    l = Image.open(os.path.join(LINE_DIR, name)).convert("RGB").resize((THUMB, THUMB))
    pr, pc = divmod(i, pairs_per_row)
    x = pc * cell_w
    y = pr * cell_h
    canvas.paste(r, (x, y))
    canvas.paste(l, (x + THUMB, y))
    draw.text((x + 2, y + THUMB + 2), f"{name}  (L:rough R:line)", fill=(0, 0, 0), font=font)

canvas.save(OUT)
print(f"\nsaved montage: {OUT}")
