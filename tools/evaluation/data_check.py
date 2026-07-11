import os
from PIL import Image
import torch
from torchvision import transforms

# 現在の学習コードと同じDatasetクラス
class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, rough_dir, line_dir, transform=None):
        self.rough_files = sorted(os.listdir(rough_dir))
        self.line_files = sorted(os.listdir(line_dir))
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.transform = transform

    def __getitem__(self, idx):
        rough = Image.open(os.path.join(self.rough_dir, self.rough_files[idx])).convert("L")
        line = Image.open(os.path.join(self.line_dir, self.line_files[idx])).convert("L")

        if self.transform:
            rough = self.transform(rough)
            line = self.transform(line)

        # 反転なし：元データのまま学習
        return rough, line

    def __len__(self):
        return len(self.rough_files)

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SketchDataset("dataset/pairs/train/rough", "dataset/pairs/train/line", transform)

# 1枚目のデータを取得
rough, line = dataset[0]

print("=== 学習に使用されるデータ ===\n")
print(f"Rough (入力):")
print(f"  shape: {rough.shape}")
print(f"  min/max: {rough.min():.3f} / {rough.max():.3f}")
print(f"  mean: {rough.mean():.3f}")

print(f"\nLine (ターゲット):")
print(f"  shape: {line.shape}")
print(f"  min/max: {line.min():.3f} / {line.max():.3f}")
print(f"  mean: {line.mean():.3f}")

# 保存
from torchvision.transforms.functional import to_pil_image
os.makedirs("debug_output", exist_ok=True)

to_pil_image(rough).save("debug_output/training_input.png")
to_pil_image(line).save("debug_output/training_target.png")

print("\n=== 画像を保存しました ===")
print("debug_output/training_target.png を確認してください！")
print("\nこの画像は:")
print("  A. 白背景・黒線 → 正常（モデルはこれを学習すべき）")
print("  B. 黒背景・白線 → 問題（反転が必要）")
print("\nもしBなら、元の line 画像が黒背景・白線になっています")
print("その場合は Dataset の __getitem__ で反転が必要です")