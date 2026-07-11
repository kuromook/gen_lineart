import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps

from unetgenerator import UNetGenerator
from losses import edge_loss, ink_loss, tolerant_f1_loss

ROUGH_DIR           = "dataset/pairs_480/train/rough"
LINE_DIR            = "dataset/pairs_480/train/line"
FILE_LIST           = "dataset/pairs_480/valid_train.txt"
CHECKPOINT_DIR      = "checkpoints/base"
IMAGE_SIZE          = 480
BATCH_SIZE          = 2
NUM_EPOCHS          = 200
LR                  = 0.0001
POS_WEIGHT          = 5.0
EDGE_WEIGHT         = 1.0
AUTOCONTRAST_ROUGH  = True   # Trueにするとroughのコントラストを自動正規化


class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, rough_dir, line_dir, file_list=None, transform=None,
                 autocontrast_rough=False):
        if file_list:
            with open(file_list) as f:
                files = [l.strip() for l in f if l.strip()]
        else:
            files = sorted(os.listdir(rough_dir))
        self.files              = files
        self.rough_dir          = rough_dir
        self.line_dir           = line_dir
        self.transform          = transform
        self.autocontrast_rough = autocontrast_rough

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rough = Image.open(os.path.join(self.rough_dir, self.files[idx])).convert("L")
        line  = Image.open(os.path.join(self.line_dir,  self.files[idx])).convert("L")
        if self.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        if self.transform:
            rough = self.transform(rough)
            line  = self.transform(line)
        return rough, line


def train(file_list=FILE_LIST, checkpoint_dir=CHECKPOINT_DIR,
          rough_dir=ROUGH_DIR, line_dir=LINE_DIR,
          autocontrast_rough=AUTOCONTRAST_ROUGH, resume_path=None,
          num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, start_epoch_override=None,
          lr=LR, pos_weight_value=POS_WEIGHT, edge_weight=EDGE_WEIGHT,
          shape_weight=0.0, ink_weight=0.0):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = SketchDataset(rough_dir, line_dir, file_list=file_list, transform=transform,
                            autocontrast_rough=autocontrast_rough)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNetGenerator(in_channels=1, out_channels=1).to(device)

    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        if start_epoch_override is not None:
            start_epoch = start_epoch_override
        else:
            stem = os.path.splitext(os.path.basename(resume_path))[0].replace("epoch", "")
            start_epoch = int(stem) if stem.isdigit() else 0
        print(f"Resume from {resume_path} (epoch {start_epoch})")
    else:
        start_epoch = start_epoch_override or 0

    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=num_epochs - start_epoch, eta_min=1e-6)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"file_list: {file_list}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"rough_dir: {rough_dir}")
    print(f"line_dir: {line_dir}")
    print(f"autocontrast_rough: {autocontrast_rough}")
    print(f"lr: {lr}")
    print(f"pos_weight: {pos_weight_value}")
    print(f"edge_weight: {edge_weight}, shape_weight: {shape_weight}, ink_weight: {ink_weight}")
    print(f"データ数: {len(dataset)}, バッチ数: {len(loader)}")
    print(f"デバイス: {device}, 解像度: {IMAGE_SIZE}px")
    print(f"epoch {start_epoch+1} 〜 {num_epochs}\n")

    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0

        for rough, line in loader:
            rough, line = rough.to(device), line.to(device)
            line = 1.0 - line

            optimizer.zero_grad()
            pred     = model(rough)
            pred_sig = torch.sigmoid(pred)

            loss_bce  = criterion(pred, line)
            loss_l1   = F.l1_loss(pred_sig, line)
            loss_main = 0.8 * loss_bce + 0.2 * loss_l1
            loss = loss_main
            if edge_weight:
                loss = loss + edge_weight * edge_loss(pred_sig, line)
            if shape_weight:
                loss = loss + shape_weight * tolerant_f1_loss(pred_sig, line)
            if ink_weight:
                loss = loss + ink_weight * ink_loss(pred_sig, line)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{num_epochs}: loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
              flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best.pth")
            print(f"  → best saved (loss={best_loss:.4f})", flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/epoch{epoch+1:03d}.pth")

    print(f"\n完了。{checkpoint_dir}/best.pth を使用してください。")
    return best_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list",        default=FILE_LIST)
    parser.add_argument("--checkpoint-dir",   default=CHECKPOINT_DIR)
    parser.add_argument("--rough-dir",        default=ROUGH_DIR)
    parser.add_argument("--line-dir",         default=LINE_DIR)
    parser.add_argument("--resume",           default=None)
    parser.add_argument("--autocontrast",     action="store_true", default=AUTOCONTRAST_ROUGH)
    parser.add_argument("--no-autocontrast",  dest="autocontrast", action="store_false")
    parser.add_argument("--epochs",           type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size",       type=int, default=BATCH_SIZE)
    parser.add_argument("--start-epoch",      type=int, default=None)
    parser.add_argument("--lr",               type=float, default=LR)
    parser.add_argument("--pos-weight",       type=float, default=POS_WEIGHT)
    parser.add_argument("--edge-weight",      type=float, default=EDGE_WEIGHT)
    parser.add_argument("--shape-weight",     type=float, default=0.0)
    parser.add_argument("--ink-weight",       type=float, default=0.0)
    args = parser.parse_args()

    train(file_list=args.file_list,
          checkpoint_dir=args.checkpoint_dir,
          rough_dir=args.rough_dir,
          line_dir=args.line_dir,
          autocontrast_rough=args.autocontrast,
          resume_path=args.resume,
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          start_epoch_override=args.start_epoch,
          lr=args.lr,
          pos_weight_value=args.pos_weight,
          edge_weight=args.edge_weight,
          shape_weight=args.shape_weight,
          ink_weight=args.ink_weight)
