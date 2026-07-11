"""warm-start再学習用の学習リストを2本生成する。
  plan1: std15非ako5(660) + ako5 ink[2%,25%]
  plan2: std15非ako5(660) + ako5 ink[2%,15%]
  regions: std15非ako5(660) + 高信頼ako5局所領域ペア
ako5側は rough autocontrast std>=15 も併せて要求。
"""
import os
import glob
import numpy as np
from PIL import Image, ImageOps

ROUGH_DIR = "dataset/pairs_480/train/rough"
LINE_DIR = "dataset/pairs_480/train/line"
STD15_LIST = "dataset/pairs_480/valid_train_std15.txt"
OUT1 = "dataset/pairs_480/valid_train_warm_plan1.txt"
OUT2 = "dataset/pairs_480/valid_train_warm_plan2.txt"
REGION_LIST = "dataset/pairs_480/valid_train_ako5_regions.txt"
REGION_OUT = "dataset/pairs_480/valid_train_warm_regions.txt"
ROUGH_STD = 15

# std15リストから非ako5(=元のクリーン660)を抽出
with open(STD15_LIST) as f:
    std15 = [ln.strip() for ln in f if ln.strip()]
base = [n for n in std15 if not n.startswith("ako5_")]
print(f"std15非ako5(base): {len(base)} 枚")

# ako5タイルの指標を計算
rough_paths = sorted(glob.glob(os.path.join(ROUGH_DIR, "ako5_*.jpg")))
ako_pass1, ako_pass2 = [], []
for p in rough_paths:
    name = os.path.basename(p)
    r_ac = np.asarray(ImageOps.autocontrast(Image.open(p).convert("L"), cutoff=0), dtype=np.float32)
    if r_ac.std() < ROUGH_STD:
        continue
    l = np.asarray(Image.open(os.path.join(LINE_DIR, name)).convert("L"), dtype=np.float32)
    ink = (l < 128).mean()
    if 0.02 <= ink <= 0.25:
        ako_pass1.append(name)
    if 0.02 <= ink <= 0.15:
        ako_pass2.append(name)

print(f"ako5 [2%,25%]: {len(ako_pass1)} 枚 → plan1 計 {len(base)+len(ako_pass1)}")
print(f"ako5 [2%,15%]: {len(ako_pass2)} 枚 → plan2 計 {len(base)+len(ako_pass2)}")

with open(OUT1, "w") as f:
    f.write("\n".join(sorted(base) + sorted(ako_pass1)) + "\n")
with open(OUT2, "w") as f:
    f.write("\n".join(sorted(base) + sorted(ako_pass2)) + "\n")

with open(REGION_LIST) as f:
    regions = [line.strip() for line in f if line.strip()]
with open(REGION_OUT, "w") as f:
    f.write("\n".join(sorted(base) + sorted(regions)) + "\n")

print(f"ako5 regions: {len(regions)} 枚 → regions 計 {len(base)+len(regions)}")
print(f"出力: {OUT1}\n出力: {OUT2}\n出力: {REGION_OUT}")
