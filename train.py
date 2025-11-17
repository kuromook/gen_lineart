import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim

from unetgenerator import UNetGenerator

def sobel_edges(x):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           dtype=torch.float32, device=x.device).view(1,1,3,3)

    g_x = F.conv2d(x, sobel_x, padding=1)
    g_y = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(g_x**2 + g_y**2 + 1e-6)
    
def edge_loss(pred, target):
    return F.l1_loss(sobel_edges(pred), sobel_edges(target))

# Dataset
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
        rough = Image.open(os.path.join(self.rough_dir, self.rough_files[idx])).convert("L")
        line  = Image.open(os.path.join(self.line_dir,  self.line_files[idx])).convert("L")

        if self.transform:
            rough = self.transform(rough)
            line  = self.transform(line)

        return rough, line


# ---------------------------------------------------------------------------
# II. データローディングと設定
# ---------------------------------------------------------------------------
# Dataset クラスはそのまま使用 (paired_transform 引数は不要)

# Transform の修正: Resize を 256x256 に変更
transform = transforms.Compose([
    transforms.Resize((256, 256)), # ★ 128x128 から 256x256 に変更 ★
    transforms.ToTensor(),
])

dataset = SketchDataset("dataset/train/rough", "dataset/train/line", transform)

# DataLoader: 解像度アップに伴い、VRAM節約のためバッチサイズを 2 に下げることを推奨
loader = DataLoader(dataset, batch_size=2, shuffle=True) 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetGenerator(in_channels=1, out_channels=1).to(device)


pos_weight_value = 3.0 
pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float).to(device)

# 2. criterion に Tensor 型の pos_weight を渡す
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# オプティマイザ: 学習率を 0.0001 に下げる (暴走防止)
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

os.makedirs("checkpoints", exist_ok=True)


# ---------------------------------------------------------------------------
# III. トレーニングループ
# ---------------------------------------------------------------------------
for epoch in range(50):
    total_loss = 0.0
    num_batches = 0
    model.train() 
    
    for rough, line in loader:
        rough, line = rough.to(device), line.to(device)

        line = 1.0 - line
        
        optimizer.zero_grad()
        pred = model(rough)
    
        # 1. メイン損失 (BCE with pos_weight)
        loss_main_bce = criterion(pred, line)
    
        # 2. L1損失 (Sigmoid後の出力とターゲットの絶対誤差)
        # L1損失を導入することで、ピクセル値がターゲットに近づくように強制する。
        loss_main_l1 = F.l1_loss(torch.sigmoid(pred), line) 
    
        # 3. メイン損失の組み合わせとエッジ損失
        # BCEとL1をミックス (例: 80% BCE, 20% L1)
        loss_main = 0.8 * loss_main_bce + 0.2 * loss_main_l1 
    
        loss_edge = edge_loss(torch.sigmoid(pred), line) 

        # 総損失 (L1損失が太さを抑制する助けになるため、エッジ重みは 1.0 で維持してもよい)
        loss = loss_main + 1.0 * loss_edge

        loss.backward()
        optimizer.step()
        
        # ログ改善
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/unet_1ch_epoch{epoch+1}.pth")