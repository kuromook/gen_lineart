import os
import shutil

SRC_BASE = "/home/sh1/deepl/split"
DST_BASE = "dataset/pairs_480"

pairs = [
    ("train/rough", "train/rough"),
    ("train/line",  "train/line"),
    ("test/rough",  "test/rough"),
    ("test/line",   "test/line"),
]

for src_rel, dst_rel in pairs:
    src = os.path.join(SRC_BASE, src_rel)
    dst = os.path.join(DST_BASE, dst_rel)
    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)
    print(f"{src_rel} → {dst_rel}: {len(files)} ファイル")

    for i, f in enumerate(files, 1):
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        if os.path.exists(dst_file):
            print(f"  スキップ（既存）: {f}")
            continue
        shutil.copy2(src_file, dst_file)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} 完了")

    print(f"  → 完了\n")

print("=== マージ完了 ===")
for split in ("train", "test"):
    for kind in ("rough", "line"):
        path = os.path.join(DST_BASE, split, kind)
        if os.path.exists(path):
            print(f"  {split}/{kind}: {len(os.listdir(path))} 枚")
