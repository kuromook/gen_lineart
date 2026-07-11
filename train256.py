import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

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

def sobel_edges(x):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    g_x = F.conv2d(x, sobel_x, padding=1)
    g_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(g_x**2 + g_y**2 + 1e-6)

def edge_loss(pred, target):
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))

def smoothness_loss(x):
    """線の滑らかさを促進する損失（二階微分を小さくする）"""
    # 水平方向の二階微分
    diff_h = x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
    # 垂直方向の二階微分
    diff_v = x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_v))

class SketchDataset(Dataset):
    def __init__(self, rough_dir, line_dir, transform=None):
        self.rough_files = sorted(os.listdir(rough_dir))
        self.line_files = sorted(os.listdir(line_dir))
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.transform = transform
    def __len__(self):
        return len(self.rough_files)
    def __getitem__(self, idx):
        rough = Image.open(os.path.join(self.rough_dir, self.rough_files[idx])).convert("L")
        line = Image.open(os.path.join(self.line_dir, self.line_files[idx])).convert("L")
        if self.transform:
            rough = self.transform(rough)
            line = self.transform(line)
        return rough, line

# データ拡張なし（シャープさ優先）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SketchDataset("dataset/train/rough", "dataset/train/line", transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetGenerator(in_channels=1, out_channels=1).to(device)

# 重要：学習率を下げて、よりシャープな学習を促す
optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

os.makedirs("checkpoints", exist_ok=True)

num_epochs = 100
best_loss = float('inf')

print("=== シャープな線画学習開始 ===\n")

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    model.train()
    
    for rough, line in loader:
        rough, line = rough.to(device), line.to(device)
        
        optimizer.zero_grad()
        pred = model(rough)
        pred_sigmoid = torch.sigmoid(pred)
        
        # ★★★ 改善された損失関数 ★★★
        
        # 1. BCE損失（基本）
        loss_bce = F.binary_cross_entropy(pred_sigmoid, line)
        
        # 2. MSE損失（ピクセル値の一致）
        loss_mse = F.mse_loss(pred_sigmoid, line)
        
        # 3. エッジ損失（線の位置）
        loss_edge = edge_loss(pred_sigmoid, line)
        
        # 4. 滑らかさ損失（線の丸み・曲線美を促進）
        loss_smooth = smoothness_loss(pred_sigmoid)
        
        # 5. 二値化促進損失（0または1に近づける）
        binary_penalty = torch.mean(torch.abs(pred_sigmoid - line) * (1 - torch.abs(pred_sigmoid - 0.5)))
        
        # 6. Focal Loss（線に集中）
        alpha = 0.25
        pt = torch.where(line > 0.5, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = alpha * (1 - pt) ** 2
        loss_focal = torch.mean(focal_weight * F.binary_cross_entropy(pred_sigmoid, line, reduction='none'))
        
        # 総損失（滑らかさを追加）
        loss = (
            0.8 * loss_bce +
            1.0 * loss_mse +
            1.5 * loss_edge +
            0.3 * loss_smooth +     # ★ 滑らかさ損失を追加 ★
            0.2 * binary_penalty +
            0.3 * loss_focal
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch < 10:
        model.eval()
        with torch.no_grad():
            sample_rough, sample_line = next(iter(loader))
            sample_rough = sample_rough.to(device)
            sample_pred = torch.sigmoid(model(sample_rough))
            pred_range = sample_pred.max() - sample_pred.min()
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.6f}")
            print(f"  予測範囲: min={sample_pred.min():.3f}, max={sample_pred.max():.3f}, range={pred_range:.3f}")
        model.train()
    else:
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.6f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/base/unet_sharp_best.pth")
        print(f"  → ベストモデル保存 (loss: {best_loss:.4f})")
    
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"checkpoints/base/unet_sharp_epoch{epoch+1}.pth")
    
    scheduler.step()

print("\n学習完了！checkpoints/base/unet_sharp_best.pth を使用してください")