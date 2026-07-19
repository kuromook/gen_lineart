import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from lineart.losses import ink_loss, tolerant_f1_loss
from lineart.unetgenerator import UNetGenerator


IMAGE_SIZE = 480


def skeletonize_ink(ink):
    mask = (ink > 0.5).astype(np.uint8) * 255
    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(mask) > 0:
        eroded = cv2.erode(mask, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(mask, opened))
        mask = eroded
    return skel > 0


def line_field_targets(target, aux, max_offset=16.0):
    target_np = target[0].numpy().astype(np.float32)
    aux_np = aux[0].numpy().astype(np.float32)
    skeleton = skeletonize_ink(target_np)
    if not skeleton.any():
        skeleton = target_np > 0.5
    if not skeleton.any():
        skeleton[target_np.shape[0] // 2, target_np.shape[1] // 2] = True

    _, indices = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    yy, xx = np.indices(target_np.shape, dtype=np.float32)
    dy = (indices[0].astype(np.float32) - yy) / max_offset
    dx = (indices[1].astype(np.float32) - xx) / max_offset
    offset = np.stack([np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0)], axis=0)

    aux_ink = 1.0 - aux_np
    offset_mask = ((aux_ink > 0.04) | (target_np > 0.05)).astype(np.float32)[None, :, :]
    center = skeleton.astype(np.float32)[None, :, :]
    return (
        torch.from_numpy(center),
        torch.from_numpy(offset.astype(np.float32)),
        torch.from_numpy(offset_mask),
    )


class LineFieldDataset(Dataset):
    def __init__(self, rough_dir, line_dir, aux_dir, file_list, autocontrast=True):
        with open(file_list) as file:
            self.files = [line.strip() for line in file if line.strip()]
        self.rough_dir = Path(rough_dir)
        self.line_dir = Path(line_dir)
        self.aux_dir = Path(aux_dir)
        self.autocontrast = autocontrast

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        base = Path(name).stem
        rough = Image.open(self.rough_dir / name).convert("L")
        line = Image.open(self.line_dir / name).convert("L")
        aux = Image.open(self.aux_dir / f"{base}_out.png").convert("L")
        if self.autocontrast:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough = TF.resize(rough, (IMAGE_SIZE, IMAGE_SIZE))
        line = TF.resize(line, (IMAGE_SIZE, IMAGE_SIZE))
        aux = TF.resize(aux, (IMAGE_SIZE, IMAGE_SIZE))
        rough_t = TF.to_tensor(rough)
        aux_t = TF.to_tensor(aux)
        target = 1.0 - TF.to_tensor(line)
        center, offset, offset_mask = line_field_targets(target, aux_t)
        x = torch.cat([rough_t, aux_t], dim=0)
        return x, target, center, offset, offset_mask


class LineFieldUNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.net = UNetGenerator(in_channels=in_channels, out_channels=4)

    def forward(self, x):
        raw = self.net(x)
        ink_logits = raw[:, :1]
        center_logits = raw[:, 1:2]
        offset = torch.tanh(raw[:, 2:4])
        return ink_logits, center_logits, offset


def save_checkpoint(path, model, epoch, args):
    torch.save(
        {
            "model_name": "linefield_unet",
            "epoch": epoch,
            "in_channels": 2,
            "G_state": model.state_dict(),
            "args": vars(args),
        },
        path,
    )


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LineFieldDataset(
        args.rough_dir,
        args.line_dir,
        args.aux_dir,
        args.file_list,
        autocontrast=args.autocontrast,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = LineFieldUNet(in_channels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    center_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.center_pos_weight, device=device))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"device={device}")
    print(f"model=linefield_unet aux_dir={args.aux_dir}")
    print(f"file_list={args.file_list} rows={len(dataset)}")
    print(
        f"loss: bce={args.bce_weight} l1={args.l1_weight} "
        f"shape={args.shape_weight} ink={args.ink_weight} "
        f"center={args.center_weight} offset={args.offset_weight}"
    )
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for x, target, center, offset, offset_mask in loader:
            x = x.to(device)
            target = target.to(device)
            center = center.to(device)
            offset = offset.to(device)
            offset_mask = offset_mask.to(device)
            ink_logits, center_logits, pred_offset = model(x)
            pred = torch.sigmoid(ink_logits)
            loss = (
                args.bce_weight * bce(ink_logits, target)
                + args.l1_weight * F.l1_loss(pred, target)
                + args.shape_weight * tolerant_f1_loss(pred, target)
                + args.ink_weight * ink_loss(pred, target)
                + args.center_weight * center_bce(center_logits, center)
                + args.offset_weight * (F.smooth_l1_loss(pred_offset * offset_mask, offset * offset_mask))
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1
        avg = total / max(steps, 1)
        print(f"Epoch {epoch:03d}/{args.epochs}: loss={avg:.4f}", flush=True)
        if avg < best:
            best = avg
            save_checkpoint(Path(args.checkpoint_dir) / "best.pth", model, epoch, args)
        if epoch % args.save_every == 0:
            save_checkpoint(Path(args.checkpoint_dir) / f"epoch{epoch:03d}.pth", model, epoch, args)
    print(f"saved: {args.checkpoint_dir}/best.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", default="dataset/pairs_480/valid_train_milddup800_clean.txt")
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line")
    parser.add_argument("--aux-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--center-pos-weight", type=float, default=12.0)
    parser.add_argument("--bce-weight", type=float, default=0.70)
    parser.add_argument("--l1-weight", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.08)
    parser.add_argument("--ink-weight", type=float, default=0.12)
    parser.add_argument("--center-weight", type=float, default=0.25)
    parser.add_argument("--offset-weight", type=float, default=0.08)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--autocontrast", action="store_true", default=True)
    parser.add_argument("--no-autocontrast", dest="autocontrast", action="store_false")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
