"""
ako5データセットの前処理スクリプト
- dataset_ako5.zip を展開
- 各ページ(4961×7016)を480×480タイルに分割
- roughタイルにautocontrastを適用してstd>=15のものを選別
- dataset/pairs_480/train/{rough,line}/ に保存
- valid_train_std15.txt に追記
"""
import io
import os
import re
import zipfile
import numpy as np
from PIL import Image, ImageOps

ZIP_PATH     = os.path.expanduser("~/dataset_ako5.zip")
ROUGH_DIR    = "dataset/pairs_480/train/rough"
LINE_DIR     = "dataset/pairs_480/train/line"
VALID_TXT    = "dataset/pairs_480/valid_train_std15.txt"
TILE_SIZE    = 480
STD_THRESH   = 15


def tile_image(img, tile_size=480):
    """画像を tile_size × tile_size で分割(端数タイルは破棄)"""
    w, h = img.size
    n_cols = w // tile_size
    n_rows = h // tile_size
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tile_size
            y = row * tile_size
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((row, col, tile))
    return tiles


def main():
    os.makedirs(ROUGH_DIR, exist_ok=True)
    os.makedirs(LINE_DIR, exist_ok=True)

    # 既存のstd15リストを読み込む
    existing = set()
    if os.path.exists(VALID_TXT):
        with open(VALID_TXT) as f:
            existing = {l.strip() for l in f if l.strip()}
    print(f"既存 valid_train_std15.txt: {len(existing)} 枚")

    new_entries = []
    stats = {"pages": 0, "tiles_total": 0, "tiles_pass": 0}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        # manifest から sketch/line ペアを取得
        with zf.open("dataset_ako5/manifest.json") as mf:
            import json
            manifest = json.load(mf)

        print(f"ページ数: {len(manifest)}")

        for entry in manifest:
            sketch_name = entry["sketch"]  # e.g. ako5_page0001_sketch.jpg
            line_name   = entry["line"]    # e.g. ako5_page0001_line.jpg

            # ページ番号を抽出: page0001 -> 001
            m = re.search(r"page(\d+)", sketch_name)
            if not m:
                print(f"スキップ(ページ番号不明): {sketch_name}")
                continue
            page_num = int(m.group(1))
            prefix = f"ako5_{page_num:03d}"

            # 既にこのページのタイルが存在するか確認
            existing_for_page = [e for e in existing if e.startswith(prefix)]
            if existing_for_page:
                print(f"  {prefix}: スキップ(既にタイル存在: {len(existing_for_page)}枚)")
                continue

            # 画像をメモリに読み込む
            with zf.open(f"dataset_ako5/{sketch_name}") as sf:
                rough_img = Image.open(io.BytesIO(sf.read())).convert("L")
            with zf.open(f"dataset_ako5/{line_name}") as lf:
                line_img  = Image.open(io.BytesIO(lf.read())).convert("L")

            if rough_img.size != line_img.size:
                print(f"警告: サイズ不一致 {sketch_name}: rough={rough_img.size} line={line_img.size}")
                # 小さい方に合わせる
                w = min(rough_img.width, line_img.width)
                h = min(rough_img.height, line_img.height)
                rough_img = rough_img.crop((0, 0, w, h))
                line_img  = line_img.crop((0, 0, w, h))

            rough_tiles = tile_image(rough_img, TILE_SIZE)
            line_tiles  = tile_image(line_img,  TILE_SIZE)

            assert len(rough_tiles) == len(line_tiles)
            stats["pages"]       += 1
            stats["tiles_total"] += len(rough_tiles)

            page_pass = 0
            for (row, col, rough_tile), (_, _, line_tile) in zip(rough_tiles, line_tiles):
                # autocontrast 後の std でフィルタ
                rough_ac = ImageOps.autocontrast(rough_tile, cutoff=0)
                std = float(np.array(rough_ac).std())
                if std < STD_THRESH:
                    continue

                fname = f"{prefix}_{row:02d}_{col:02d}.jpg"
                rough_tile.save(os.path.join(ROUGH_DIR, fname), quality=95)
                line_tile.save(os.path.join(LINE_DIR,  fname), quality=95)
                new_entries.append(fname)
                page_pass += 1

            stats["tiles_pass"] += page_pass
            print(f"  {prefix}: {len(rough_tiles)} tiles → {page_pass} pass (std>={STD_THRESH})")

    # valid_train_std15.txt に追記
    if new_entries:
        with open(VALID_TXT, "a") as f:
            for fname in sorted(new_entries):
                f.write(fname + "\n")
        print(f"\n追記完了: {len(new_entries)} 枚 → {VALID_TXT}")
    else:
        print("\n新規エントリなし")

    total_now = len(existing) + len(new_entries)
    print(f"\n=== 完了 ===")
    print(f"処理ページ数  : {stats['pages']}")
    print(f"タイル総数    : {stats['tiles_total']}")
    print(f"フィルタ通過  : {stats['tiles_pass']} (std>={STD_THRESH})")
    print(f"valid_train_std15.txt: {len(existing)} → {total_now} 枚")


if __name__ == "__main__":
    main()
