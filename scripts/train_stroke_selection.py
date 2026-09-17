"""Track C step 2: train the minimal pixel-level keep/drop classifier.

Input: the lineart_coarse conditioning image alone (in_channels=1, no raw
rough, per this track's own scope). Output: a per-pixel "keep" logit. Loss
is masked BCE, restricted to the conditioning image's own Canny edge pixels
(`tools/selection/build_keep_labels.py`'s label definition) -- background
and non-edge ink pixels are not a keep/drop decision, so they don't enter
the loss. The AND-with-conditioning-edge constraint that keeps this model
strictly deletion-only (it can only drop edge pixels, never invent new
ink) is applied outside the model, identically at train time (as the loss
mask) and at eval time (as the hard output mask) -- see `to_keep_mask()`.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lineart.model_zoo import build_generator

IMAGE_SIZE = 480


def load_gray01(path):
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


def load_packed_label(path):
    packed = np.asarray(Image.open(path))
    cond_edge = (packed & 1).astype(np.float32)
    keep = ((packed >> 1) & 1).astype(np.float32)
    return cond_edge, keep


class StrokeSelectionDataset(Dataset):
    def __init__(self, rough_dir, label_dir, file_list, limit=0):
        with open(file_list) as f:
            names = [line.strip() for line in f if line.strip()]
        self.rough_dir = Path(rough_dir)
        self.label_dir = Path(label_dir)
        # Only tiles that actually got a label (skips any build_keep_labels.py
        # failures rather than crashing the whole run on a missing file).
        self.files = [n for n in names if (self.label_dir / (Path(n).stem + ".png")).exists()]
        if limit:
            self.files = self.files[:limit]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        cond = load_gray01(self.rough_dir / name)
        cond_edge, keep = load_packed_label(self.label_dir / (Path(name).stem + ".png"))
        cond_t = torch.from_numpy(cond).unsqueeze(0)
        cond_edge_t = torch.from_numpy(cond_edge).unsqueeze(0)
        keep_t = torch.from_numpy(keep).unsqueeze(0)
        return cond_t, cond_edge_t, keep_t


def to_keep_mask(cond_edge, logit, threshold=0.5):
    """The structural deletion-only constraint: the model can only ever
    remove conditioning edge pixels, never add new ones. Used identically
    for the training loss mask and the eval-time hard output mask."""
    return cond_edge * (torch.sigmoid(logit) > threshold).float()


def save_checkpoint(path, model, epoch, args):
    torch.save(
        {
            "model_name": "strokeselect",
            "epoch": epoch,
            "in_channels": 1,
            "G_state": model.state_dict(),
            "args": vars(args),
        },
        path,
    )


def masked_bce(logit, target, mask, pos_weight):
    per_px = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=torch.tensor(pos_weight, device=logit.device), reduction="none"
    )
    return (per_px * mask).sum() / mask.sum().clamp_min(1.0)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = StrokeSelectionDataset(args.rough_dir, args.label_dir, args.file_list, limit=args.limit)
    print(f"dataset: {len(dataset)} tiles", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, drop_last=True)

    model = build_generator("strokeselect", in_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: strokeselect, {n_params} params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for cond, cond_edge, keep in loader:
            cond, cond_edge, keep = cond.to(device), cond_edge.to(device), keep.to(device)
            logit = model(cond)
            loss = masked_bce(logit, keep, cond_edge, args.pos_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()

            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)

        epoch_loss = running / max(len(loader), 1)
        print(f"=== epoch {epoch} done, mean loss {epoch_loss:.4f} ===", flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt_dir / "best.pth", model, epoch, args)
        if epoch % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch{epoch:03d}.pth", model, epoch, args)

    save_checkpoint(ckpt_dir / "final.pth", model, args.epochs, args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C = Path(__file__).resolve().parents[1]
    p.add_argument("--file-list", default=str(C / "data/train_list.txt"))
    p.add_argument("--rough-dir", default=str(C / "data/rough_lineart_coarse"))
    p.add_argument("--label-dir", default=str(C / "results/keep_labels_20260917"))
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--pos-weight", type=float, default=3.97)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="first N tiles only, for smoke tests")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
