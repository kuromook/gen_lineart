import os
import shutil

# ディレクトリ構成
source_base = "dataset/raw"
dest_base = "dataset/pairs_480"

# コピー元とコピー先のペア
folders = [
    ("rough", "train/rough"),
    ("line", "train/line"),
]

print("=== 480x480データセットをコピー中 ===\n")

# コピー先ディレクトリを作成
for _, dest_folder in folders:
    dest_path = os.path.join(dest_base, dest_folder)
    os.makedirs(dest_path, exist_ok=True)
    print(f"作成: {dest_path}")

print()

# ファイルをコピー
for source_folder, dest_folder in folders:
    source_path = os.path.join(source_base, source_folder)
    dest_path = os.path.join(dest_base, dest_folder)
    
    if not os.path.exists(source_path):
        print(f"⚠️  スキップ: {source_path} が見つかりません")
        continue
    
    files = os.listdir(source_path)
    print(f"コピー中: {source_path} → {dest_path}")
    print(f"  ファイル数: {len(files)}")
    
    for i, filename in enumerate(files, 1):
        src_file = os.path.join(source_path, filename)
        dst_file = os.path.join(dest_path, filename)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            
            if i % 50 == 0:
                print(f"  進捗: {i}/{len(files)} ファイル完了")
    
    print(f"  ✓ 完了: {len(files)} ファイルをコピー\n")

print("=== 完了 ===")
print("\nディレクトリ構成:")
print("  dataset/raw/        ← 元データ（保持）")
print("  dataset/pairs/      ← 旧256x256ペア（保持）")
print("  dataset/pairs_480/  ← 480x480ペア（新規作成）")

print("\n学習コードの変更:")
print('  dataset = SketchDataset("dataset/pairs_480/train/rough", "dataset/pairs_480/train/line", transform)')
