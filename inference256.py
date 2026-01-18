import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# モデル定義（同じ）
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(ch)
    def forward(self, x):
        h = F.relu(self.in1(self.conv1(x)))
        h = self.in2(self.conv2(h))
        return x + h

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, dilation=dilation, padding=dilation)
        self.inorm = nn.InstanceNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.inorm(self.conv(x)))

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.ReLU(True), ResBlock(64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True), ResBlock(128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(DilatedConvBlock(128, 256, dilation=2), ResBlock(256))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DilatedConvBlock(256, 512, dilation=4), ResBlock(512), DilatedConvBlock(512, 512, dilation=4))
        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.InstanceNorm2d(256), nn.ReLU(True), ResBlock(256))
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True), ResBlock(128))
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(DilatedConvBlock(128, 64, dilation=1), ResBlock(64))
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetGenerator(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/unet_sharp_best.pth", map_location=device))
model.eval()

img = Image.open("test/rough/sample.jpg").convert("L")
img = TF.resize(img, (256, 256))
img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = torch.sigmoid(model(img_tensor))

print(f"出力: min={out.min():.3f}, max={out.max():.3f}, mean={out.mean():.3f}, range={out.max()-out.min():.3f}")

# アンチエイリアス処理（線を滑らかに）
apply_antialiasing = True  # ★ True で線が滑らかに ★

if apply_antialiasing:
    # 軽いガウシアンフィルタ（ぼかしではなく、エッジの段差を滑らかに）
    from scipy.ndimage import gaussian_filter
    import numpy as np
    
    out_np = out[0, 0].cpu().numpy()
    
    # 線の部分だけ滑らかにする（背景は保持）
    # しきい値より暗い部分を線と判定
    threshold = 0.7
    line_mask = out_np < threshold
    
    # 線の境界付近だけガウシアンフィルタ
    out_smooth = gaussian_filter(out_np, sigma=0.5)
    
    # 線とその周辺だけ滑らかにする
    from scipy.ndimage import binary_dilation
    edge_mask = binary_dilation(line_mask, iterations=2) & ~line_mask
    
    out_np = np.where(edge_mask, out_smooth, out_np)
    out = torch.from_numpy(out_np).unsqueeze(0).unsqueeze(0).to(device)
    
    print("→ アンチエイリアス適用（線のエッジを滑らかに）")

out_range = out.max() - out.min()

if out_range < 0.3:  # レンジが狭い場合のみ正規化
    print("→ レンジが狭いため正規化適用")
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
else:
    print("→ そのまま出力")

out = out.clamp(0, 1)

import os
os.makedirs("results", exist_ok=True)

out_img = TF.to_pil_image(out[0].cpu())
out_img.save("results/sharp_output.png")

print("完了: results/sharp_output.png")