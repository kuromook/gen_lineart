# train.py
import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

# -----------------------
# Utility blocks
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.norm = nn.InstanceNorm2d(ch, affine=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return F.relu(out + x)

# -----------------------
# Simple Attention Gate (spatial attention)
# -----------------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        # channel-wise average and max
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)  # 2 x H x W
        # reduce to single-channel attention
        att = torch.sigmoid(self.conv(cat))
        return x * att

# -----------------------
# Basic Self-Attention module (non-local)
# -----------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h*w).permute(0,2,1)  # B x N x C'
        proj_key   = self.key_conv(x).view(b, -1, h*w)                    # B x C' x N
        energy = torch.bmm(proj_query, proj_key)                          # B x N x N
        att = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(b, -1, h*w)                  # B x C x N
        out = torch.bmm(proj_value, att.permute(0,2,1))                   # B x C x N
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out

# -----------------------
# U-Net with Residual and Attention
# -----------------------
class AttnResUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, use_self_attn=True):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base_ch), ResidualBlock(base_ch))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(ConvBlock(base_ch, base_ch*2), ResidualBlock(base_ch*2))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(ConvBlock(base_ch*2, base_ch*4), ResidualBlock(base_ch*4))
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(ConvBlock(base_ch*4, base_ch*8), ResidualBlock(base_ch*8))
        self.self_attn = SelfAttention(base_ch*8) if use_self_attn else nn.Identity()

        # decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = nn.Sequential(ConvBlock(base_ch*8, base_ch*4), ResidualBlock(base_ch*4))

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = nn.Sequential(ConvBlock(base_ch*4, base_ch*2), ResidualBlock(base_ch*2))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(ConvBlock(base_ch*2, base_ch), ResidualBlock(base_ch))

        self.final = nn.Conv2d(base_ch, out_ch, 1)
        self.spatial_attn1 = SpatialAttention()
        self.spatial_attn2 = SpatialAttention()
        self.spatial_attn3 = SpatialAttention()

    def forward(self, x):
        e1 = self.enc1(x)           # B,64,H,W
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)          # B,128,H/2,W/2
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)          # B,256,H/4,W/4
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)     # B,512,H/8,W/8
        b = self.self_attn(b)

        u3 = self.up3(b)            # B,256,H/4,W/4
        # attention on skip
        e3_att = self.spatial_attn1(e3)
        d3 = self.dec3(torch.cat([u3, e3_att], dim=1))

        u2 = self.up2(d3)
        e2_att = self.spatial_attn2(e2)
        d2 = self.dec2(torch.cat([u2, e2_att], dim=1))

        u1 = self.up1(d2)
        e1_att = self.spatial_attn3(e1)
        d1 = self.dec1(torch.cat([u1, e1_att], dim=1))

        out = self.final(d1)
        out = torch.sigmoid(out)  # output in [0,1]
        return out

# -----------------------
# Sobel (Edge) loss
# -----------------------
class SobelEdgeLoss(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        # Sobel kernels (horizontal, vertical)
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32) / 4.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32) / 4.0
        self.register_buffer('kx', kx.view(1,1,3,3))
        self.register_buffer('ky', ky.view(1,1,3,3))
        self.p = p

    def forward(self, pred, target):
        # pred/target: Bx1xHxW
        kx = self.kx.to(device=pred.device, dtype=pred.dtype)
        ky = self.ky.to(device=pred.device, dtype=pred.dtype)

        gx_pred = F.conv2d(pred, kx, padding=1)
        gy_pred = F.conv2d(pred, ky, padding=1)
        gx_tgt  = F.conv2d(target, kx, padding=1)
        gy_tgt  = F.conv2d(target, ky, padding=1)

        edge_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)
        edge_tgt  = torch.sqrt(gx_tgt**2 + gy_tgt**2 + 1e-6)

        if self.p == 1:
            return F.l1_loss(edge_pred, edge_tgt)
        else:
            return F.mse_loss(edge_pred, edge_tgt)


# -----------------------
# Small demo dataset (grayscale pairs)
# Replace with your pair dataset: (rough_sketch, line_art)
# -----------------------
class PairImageDataset(Dataset):
    def __init__(self, rough_dir, line_dir, size=(256,256)):
        super().__init__()
        self.rough_files = sorted(glob.glob(os.path.join(rough_dir, '*.jpg')))
        self.line_files  = sorted(glob.glob(os.path.join(line_dir, '*.jpg')))
        assert len(self.rough_files) == len(self.line_files), "pair counts must match"
        self.tr = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.rough_files)
    def __getitem__(self, idx):
        r = Image.open(self.rough_files[idx]).convert('RGB')
        l = Image.open(self.line_files[idx]).convert('RGB')
        return self.tr(r), self.tr(l)

# -----------------------
# Training loop
# -----------------------
def train(
    rough_dir='dataset/train/rough', line_dir='dataset/train/line',
    epochs=50, batch_size=8, lr=1e-4, device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = PairImageDataset(rough_dir, line_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = AttnResUNet(in_ch=1, out_ch=1, base_ch=64, use_self_attn=True).to(device)
    edge_loss = SobelEdgeLoss()
    l1_loss = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(loader))

    scaler = torch.cuda.amp.GradScaler()  # mixed precision
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # loss weights
    w_l1 = 1.0
    w_edge = 5.0   # emphasize edges

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for i, (rough, line) in enumerate(loader):
            rough = rough.to(device)
            line = line.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(rough)
                loss_l1 = l1_loss(out, line)
                loss_edge = edge_loss(out, line)
                loss = w_l1 * loss_l1 + w_edge * loss_edge

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch[{epoch}/{epochs}] Iter[{i+1}/{len(loader)}] Loss: {running_loss / (i+1):.4f}")

        avg = running_loss / len(loader)
        print(f"==> Epoch {epoch} finished. avg loss: {avg:.4f}")

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

    print("Training finished.")

if __name__ == "__main__":
    # Example: point to your directories containing paired PNGs
    train(
        rough_dir='dataset/train/rough',   # 下絵画像フォルダ（png）
        line_dir='dataset/train/line',     # 正解線画フォルダ（png）
        epochs=30,
        batch_size=8,
        lr=2e-4,
        device='cuda'
    )

