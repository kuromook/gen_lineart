import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os

# trainer.py の U-Net をインポート
from trainer import UNetGenerator

# ========================
# 推論設定
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルロード
generator = UNetGenerator().to(device)
checkpoint_path = "checkpoints/unet_epoch50.pth"  # 最新の学習済みモデルを指定
assert os.path.exists(checkpoint_path), f"{checkpoint_path} が見つかりません"
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

# ========================
# 推論実行
# ========================
img = Image.open("test/rough/sample.jpg").convert("RGB")
img = TF.resize(img, (128, 128))  # trainer と同じサイズに揃える
img = TF.to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = generator(img)

# 出力画像を保存
output_img = TF.to_pil_image((output.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu())
os.makedirs("results", exist_ok=True)
output_img.save("results/result_line.png")

print("✅ 推論完了: results/result_line.png に保存しました")

