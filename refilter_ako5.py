"""ako5タイルを両側チェックで再フィルタした場合の残存数を試算し、
推奨基準で生き残るペアのモンタージュを生成する。
判定:
  - rough: autocontrast後 std >= ROUGH_STD
  - line : インク率(暗部<128の割合) が [INK_LO, INK_HI] の帯内（空白・ベタ塗りを除外）
"""
import os
import glob
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROUGH_DIR = "dataset_480/train/rough"
LINE_DIR = "dataset_480/train/line"
OUT_MONTAGE = "results/refilter_ako5_pass.png"
OUT_LIST = "dataset_480/valid_train_ako5_clean.txt"

rough_paths = sorted(glob.glob(os.path.join(ROUGH_DIR, "ako5_*.jpg")))
print(f"ako5 tiles: {len(rough_paths)}")

names, rstd, ink = [], [], []
for p in rough_paths:
    name = os.path.basename(p)
    r = Image.open(p).convert("L")
    r_ac = np.asarray(ImageOps.autocontrast(r, cutoff=0), dtype=np.float32)
    l = np.asarray(Image.open(os.path.join(LINE_DIR, name)).convert("L"), dtype=np.float32)
    names.append(name)
    rstd.append(r_ac.std())
    ink.append((l < 128).mean())

rstd = np.array(rstd)
ink = np.array(ink)
names = np.array(names)

print("\n=== 閾値ごとの残存数 (rough autocontrast std >= R, 線画インク率 [LO,HI]) ===")
for R in (10, 15):
    for LO, HI in ((0.01, 0.40), (0.01, 0.25), (0.02, 0.25), (0.02, 0.15), (0.03, 0.15)):
        m = (rstd >= R) & (ink >= LO) & (ink <= HI)
        print(f"  rough_std>={R:2d}  ink[{LO*100:.0f}%,{HI*100:.0f}%] : {m.sum():4d} 枚 ({m.sum()/len(names)*100:.1f}%)")

# --- 推奨基準で確定 ---
R, LO, HI = 15, 0.02, 0.25
mask = (rstd >= R) & (ink >= LO) & (ink <= HI)
keep = names[mask]
print(f"\n採用基準: rough_std>={R}, ink[{LO*100:.0f}%,{HI*100:.0f}%] → {len(keep)} 枚")

with open(OUT_LIST, "w") as f:
    for n in sorted(keep):
        f.write(n + "\n")
print(f"リスト出力: {OUT_LIST}")

# --- モンタージュ: 採用タイルからランダム24ペア ---
random.seed(1)
sample = random.sample(list(keep), min(24, len(keep)))
THUMB = 200
pairs_per_row = 3
rows = (len(sample) + pairs_per_row - 1) // pairs_per_row
label_h = 16
cell_w, cell_h = THUMB * 2, THUMB + label_h
canvas = Image.new("RGB", (cell_w * pairs_per_row, cell_h * rows), (180, 180, 180))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except OSError:
    font = ImageFont.load_default()
for i, name in enumerate(sample):
    r = Image.open(os.path.join(ROUGH_DIR, name)).convert("RGB").resize((THUMB, THUMB))
    l = Image.open(os.path.join(LINE_DIR, name)).convert("RGB").resize((THUMB, THUMB))
    pr, pc = divmod(i, pairs_per_row)
    x, y = pc * cell_w, pr * cell_h
    canvas.paste(r, (x, y))
    canvas.paste(l, (x + THUMB, y))
    draw.text((x + 2, y + THUMB + 2), f"{name} (L:rough R:line)", fill=(0, 0, 0), font=font)
canvas.save(OUT_MONTAGE)
print(f"モンタージュ(採用分): {OUT_MONTAGE}")
