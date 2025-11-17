import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os

from unetgenerator import UNetGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetGenerator(in_channels=1, out_channels=1).to(device)
# チェックポイントのロード...
model.load_state_dict(torch.load("checkpoints/unet_1ch_epoch50.pth", map_location=device))
model.eval()

img = Image.open("test/rough/sample.jpg").convert("L")

# ★★★ リサイズ解像度を 256x256 に変更 ★★★
img = TF.resize(img, (256, 256)) 
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

img = TF.to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = torch.sigmoid(model(img))  # 0〜1

out = 1.0 - out 

out = out.clamp(0,1)
out_img = TF.to_pil_image(out[0].cpu())
out_img.save("results/plan2.png")

print("done")