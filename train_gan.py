"""
GAN学習スクリプト（ゼロから学習）
Generator: U-Net + Spatial Attention
Discriminator: PatchGAN (条件付き: rough + line を入力)
Loss: LSGAN + 加重L1 + Sobel edge
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

# ─── アーキテクチャ ────────────────────────────────────────────────────────────

class ConvNormReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
    def forward(self, x):
        return F.relu(self.net(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvNormReLU(ch, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm  = nn.InstanceNorm2d(ch, affine=True)
    def forward(self, x):
        return x + self.norm(self.conv2(F.relu(self.conv1(x))))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvNormReLU(1, 64),   ResBlock(64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvNormReLU(64, 128),  ResBlock(128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvNormReLU(128, 256), ResBlock(256))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvNormReLU(256, 512), ResBlock(512))

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(ConvNormReLU(512, 256), ResBlock(256))
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(ConvNormReLU(256, 128), ResBlock(128))
        self.up1  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1 = nn.Sequential(ConvNormReLU(128, 64),  ResBlock(64))

        self.final = nn.Conv2d(64, 1, 1)
        self.spatial_attn1 = SpatialAttention()
        self.spatial_attn2 = SpatialAttention()
        self.spatial_attn3 = SpatialAttention()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  self.spatial_attn3(e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.spatial_attn2(e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.spatial_attn1(e1)], 1))
        return self.final(d1)


class PatchDiscriminator(nn.Module):
    """条件付き PatchGAN: (rough, line) の 2ch を入力"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )
    def forward(self, rough, line):
        return self.net(torch.cat([rough, line], 1))


# ─── 損失関数 ──────────────────────────────────────────────────────────────────

def sobel_edges(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32, device=x.device).view(1,1,3,3)
    return torch.sqrt(F.conv2d(x, kx, padding=1)**2 + F.conv2d(x, ky, padding=1)**2 + 1e-6)


# ─── データセット ──────────────────────────────────────────────────────────────

def paired_augment(rough: Image.Image, line: Image.Image, size: int = 480) -> tuple:
    """rough と line に同一の幾何変換を適用する"""
    # リサイズ
    rough = TF.resize(rough, (size, size))
    line  = TF.resize(line,  (size, size))

    # 水平反転（50%）
    if random.random() > 0.5:
        rough = TF.hflip(rough)
        line  = TF.hflip(line)

    # ランダム回転（±15°）
    angle = random.uniform(-15, 15)
    rough = TF.rotate(rough, angle, fill=255)
    line  = TF.rotate(line,  angle, fill=255)

    # ランダムクロップ → リサイズ（ズームイン効果）
    crop_size = random.randint(int(size * 0.75), size)
    i, j, h, w = transforms.RandomCrop.get_params(rough, (crop_size, crop_size))
    rough = TF.resized_crop(rough, i, j, h, w, (size, size))
    line  = TF.resized_crop(line,  i, j, h, w, (size, size))

    # rough のみ輝度・コントラストを微変動（スキャン条件の差）
    rough = TF.adjust_brightness(rough, 1.0 + random.uniform(-0.2, 0.2))
    rough = TF.adjust_contrast(rough,   1.0 + random.uniform(-0.2, 0.2))

    return TF.to_tensor(rough), TF.to_tensor(line)


class SketchDataset(Dataset):
    def __init__(self, rough_dir, line_dir, file_list: str = None, augment: bool = True):
        if file_list:
            with open(file_list) as f:
                files = [l.strip() for l in f if l.strip()]
        else:
            files = sorted(os.listdir(rough_dir))
        self.files     = files
        self.rough_dir = rough_dir
        self.line_dir  = line_dir
        self.augment   = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rough = Image.open(os.path.join(self.rough_dir, self.files[idx])).convert("L")
        line  = Image.open(os.path.join(self.line_dir,  self.files[idx])).convert("L")
        if self.augment:
            return paired_augment(rough, line)
        rough = TF.to_tensor(TF.resize(rough, (480, 480)))
        line  = TF.to_tensor(TF.resize(line,  (480, 480)))
        return rough, line


# ─── 学習設定 ──────────────────────────────────────────────────────────────────

ROUGH_DIR   = "dataset_480/train/rough"
LINE_DIR    = "dataset_480/train/line"
FILE_LIST   = "dataset_480/valid_train.txt"
CKPT_DIR    = "checkpoints_gan2"
START_EPOCH = 0
NUM_EPOCHS  = 150
SAVE_EVERY  = 10
BATCH_SIZE  = 2
LR_G        = 2e-4
LR_D        = 2e-5   # 1/10 に下げてDの支配を抑制
LAMBDA_L1   = 20.0
LAMBDA_EDGE = 10.0
LINE_WEIGHT = 5.0
N_CRITIC    = 2      # G を N_CRITIC 回更新してから D を 1 回更新

dataset = SketchDataset(ROUGH_DIR, LINE_DIR, file_list=FILE_LIST, augment=True)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}  |  データ: {len(dataset)} 枚")

G = GANGenerator().to(device)
D = PatchDiscriminator().to(device)

opt_G = torch.optim.AdamW(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
opt_D = torch.optim.AdamW(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

os.makedirs(CKPT_DIR, exist_ok=True)

# ─── 学習ループ ────────────────────────────────────────────────────────────────

print(f"\n=== GAN学習開始 (epoch 1 〜 {NUM_EPOCHS}) ===\n")

for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
    G.train(); D.train()
    loss_G_sum = loss_D_sum = 0.0
    step = 0

    for rough, line in loader:
        rough, line = rough.to(device), line.to(device)

        # ── Generator を N_CRITIC 回更新 ──────────────────────────────────────
        for _ in range(N_CRITIC):
            fake = torch.sigmoid(G(rough))
            opt_G.zero_grad()
            fake_pred_G = D(rough, fake)
            loss_adv  = F.mse_loss(fake_pred_G, torch.ones_like(fake_pred_G))
            weight    = torch.where(line < 0.5, LINE_WEIGHT, 1.0)
            loss_l1   = (torch.abs(fake - line) * weight).mean() * LAMBDA_L1
            loss_edge = F.l1_loss(sobel_edges(fake), sobel_edges(line)) * LAMBDA_EDGE
            loss_G = loss_adv + loss_l1 + loss_edge
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

        # ── Discriminator を 1 回更新 ─────────────────────────────────────────
        fake = torch.sigmoid(G(rough)).detach()
        opt_D.zero_grad()
        real_pred = D(rough, line)
        fake_pred = D(rough, fake)
        # LSGAN + ラベルスムージング（real=0.9）
        loss_D = (F.mse_loss(real_pred, torch.full_like(real_pred, 0.9)) +
                  F.mse_loss(fake_pred, torch.zeros_like(fake_pred))) * 0.5
        loss_D.backward()
        opt_D.step()

        loss_G_sum += loss_G.item()
        loss_D_sum += loss_D.item()
        step += 1

    print(f"Epoch {epoch+1:3d}: G={loss_G_sum/step:.4f}  D={loss_D_sum/step:.4f}")

    ep = epoch + 1
    if ep % SAVE_EVERY == 0:
        path = f"{CKPT_DIR}/model_epoch_{ep}.pth"
        torch.save({
            "epoch":   ep,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "optG":    opt_G.state_dict(),
            "optD":    opt_D.state_dict(),
        }, path)
        print(f"  → 保存: {path}")

# 最終保存
final_ep = START_EPOCH + NUM_EPOCHS
torch.save({
    "epoch":   final_ep,
    "G_state": G.state_dict(),
    "D_state": D.state_dict(),
    "optG":    opt_G.state_dict(),
    "optD":    opt_D.state_dict(),
}, f"{CKPT_DIR}/model_epoch_{final_ep}.pth")
print(f"\n学習完了: {CKPT_DIR}/model_epoch_{final_ep}.pth")
