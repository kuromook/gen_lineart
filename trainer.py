import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

# ========================
#  U-Net Generator
# ========================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.down1 = self.conv_block(in_channels, 64)   # 256 -> 128
        self.down2 = self.conv_block(64, 128)           # 128 -> 64
        self.down3 = self.conv_block(128, 256)          # 64 -> 32
        self.down4 = self.conv_block(256, 512)          # 32 -> 16

        # Decoder
        self.up1 = self.up_block(512, 256)              # 16 -> 32
        self.up2 = self.up_block(512, 128)              # 32 -> 64
        self.up3 = self.up_block(256, 64)               # 64 -> 128
        self.up4 = self.up_block(128, 64)               # 128 -> 256 ←★追加

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        out = self.final(u4)

        return torch.tanh(out)



# ========================
#  データセット定義
# ========================
class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, rough_dir, line_dir, transform=None):
        self.rough_files = sorted(os.listdir(rough_dir))
        self.line_files = sorted(os.listdir(line_dir))
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.transform = transform

    def __len__(self):
        return len(self.rough_files)

    def __getitem__(self, idx):
        rough = Image.open(os.path.join(self.rough_dir, self.rough_files[idx])).convert("RGB")
        line = Image.open(os.path.join(self.line_dir, self.line_files[idx])).convert("RGB")
        if self.transform:
            rough = self.transform(rough)
            line = self.transform(line)
        return rough, line

# ========================
#  学習設定
# ========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = SketchDataset(
    "dataset/train/rough",
    "dataset/train/line",
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = UNetGenerator().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# ========================
#  学習ループ
# ========================
os.makedirs("checkpoints", exist_ok=True)

if __name__ == "__main__":
    for epoch in range(50):
        for rough, line in train_loader:
            rough, line = rough.to(device), line.to(device)
            optimizer.zero_grad()
            output = generator(rough)
            loss = criterion(output, line)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        torch.save(generator.state_dict(), f"checkpoints/unet_epoch{epoch+1}.pth")

