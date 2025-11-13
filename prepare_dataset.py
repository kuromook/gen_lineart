import os
from PIL import Image
from tqdm import tqdm

RAW_DIR = "dataset_raw"
OUT_DIR = "dataset256"
SIZE = (256, 256)

def resize_and_save(in_path, out_path):
    img = Image.open(in_path).convert("RGB")
    img = img.resize(SIZE, Image.BICUBIC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

def get_key(filename):
    """rough_ と line_ の接頭辞を削除して拡張子を除く"""
    name = os.path.splitext(filename)[0]
    name = name.replace("rough_", "").replace("line_", "")
    return name

def main():
    rough_dir = os.path.join(RAW_DIR, "rough")
    line_dir = os.path.join(RAW_DIR, "line")

    rough_files = os.listdir(rough_dir)
    line_files = os.listdir(line_dir)

    rough_dict = {get_key(f): f for f in rough_files}
    line_dict = {get_key(f): f for f in line_files}

    common_keys = sorted(list(set(rough_dict.keys()) & set(line_dict.keys())))
    print(f"共通ペア数: {len(common_keys)}")

    for key in tqdm(common_keys):
        rough_file = rough_dict[key]
        line_file = line_dict[key]
        resize_and_save(
            os.path.join(rough_dir, rough_file),
            os.path.join(OUT_DIR, "train/rough", rough_file)
        )
        resize_and_save(
            os.path.join(line_dir, line_file),
            os.path.join(OUT_DIR, "train/line", line_file)
        )

    print("✅ リサイズ完了")
    print(f"出力先: {OUT_DIR}")

if __name__ == "__main__":
    main()

