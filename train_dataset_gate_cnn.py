"""Train a small CNN dataset/source-style gate for rough images."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset_gate import DEFAULT_LABELS, label_from_name


DEFAULT_ROUGH_DIRS = ["dataset_480/train/rough", "dataset_480/test/rough"]
DEFAULT_CHECKPOINT = "checkpoints_dataset_gate/gate_cnn.pth"
DEFAULT_REPORT = "results/dataset_gate_cnn_report.json"
IMAGE_SIZE = 128


class RoughDataset(Dataset):
    def __init__(self, rows, label_to_index, autocontrast=False):
        self.rows = rows
        self.label_to_index = label_to_index
        self.autocontrast = autocontrast
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        label, path = self.rows[index]
        image = Image.open(path).convert("L")
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC)
        if self.autocontrast:
            image = ImageOps.autocontrast(image, cutoff=0)
        return self.transform(image), self.label_to_index[label]


class GateCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def target_label_from_name(name, target_mode):
    source = label_from_name(name)
    if source is None:
        return None
    if target_mode == "source":
        return source
    if target_mode == "family":
        if source in ("ako5", "kurip"):
            return "aligned_scan"
        if source in ("housei", "lineart"):
            return "general_lineart"
        return None
    if target_mode == "kurip_binary":
        return "kurip" if source == "kurip" else "non_kurip"
    if target_mode == "kurip_ako5_binary":
        if source in ("kurip", "ako5"):
            return source
        return None
    raise ValueError(f"unknown target_mode: {target_mode}")


def labels_for_mode(target_mode, labels):
    if target_mode == "source":
        return list(labels)
    if target_mode == "family":
        return ["aligned_scan", "general_lineart"]
    if target_mode == "kurip_binary":
        return ["kurip", "non_kurip"]
    if target_mode == "kurip_ako5_binary":
        return ["kurip", "ako5"]
    raise ValueError(f"unknown target_mode: {target_mode}")


def collect_rows(rough_dirs, labels, max_per_label, target_mode):
    grouped = {label: [] for label in labels}
    seen = set()
    for rough_dir in rough_dirs:
        for path in sorted(Path(rough_dir).glob("*.jpg")):
            if path.name in seen:
                continue
            seen.add(path.name)
            label = target_label_from_name(path.name, target_mode)
            if label in grouped:
                grouped[label].append(str(path))
    rows = []
    rng = random.Random(1234)
    for label, paths in grouped.items():
        rng.shuffle(paths)
        if max_per_label:
            paths = paths[:max_per_label]
        rows.extend((label, path) for path in paths)
    rng.shuffle(rows)
    return rows, {label: len(paths[:max_per_label] if max_per_label else paths) for label, paths in grouped.items()}


def split_rows(rows, val_ratio):
    by_label = {}
    for label, path in rows:
        by_label.setdefault(label, []).append((label, path))
    train_rows, val_rows = [], []
    for label, label_rows in by_label.items():
        cut = max(1, int(len(label_rows) * val_ratio))
        val_rows.extend(label_rows[:cut])
        train_rows.extend(label_rows[cut:])
    random.Random(5678).shuffle(train_rows)
    random.Random(5678).shuffle(val_rows)
    return train_rows, val_rows


def evaluate(model, loader, labels, device):
    model.eval()
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.long)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            pred = model(images).argmax(dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
            for true, guess in zip(target.cpu(), pred.cpu()):
                confusion[int(true), int(guess)] += 1
    return correct / max(total, 1), confusion.tolist()


def train(args):
    labels = labels_for_mode(args.target_mode, args.labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows, counts = collect_rows(args.rough_dirs, labels, args.max_per_label, args.target_mode)
    train_rows, val_rows = split_rows(rows, args.val_ratio)
    train_data = RoughDataset(train_rows, label_to_index, args.autocontrast)
    val_data = RoughDataset(val_rows, label_to_index, args.autocontrast)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GateCNN(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    best_report = None
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        val_acc, confusion = evaluate(model, val_loader, labels, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1:02d}/{args.epochs}: loss={avg_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_report = {
                "labels": labels,
                "counts": counts,
                "train_count": len(train_rows),
                "val_count": len(val_rows),
                "val_accuracy": val_acc,
                "confusion": confusion,
                "image_size": IMAGE_SIZE,
                "autocontrast": bool(args.autocontrast),
                "target_mode": args.target_mode,
            }
            torch.save({"model": model.state_dict(), **best_report}, args.checkpoint)
            Path(args.report).write_text(json.dumps(best_report, indent=2) + "\n")
            print(f"  saved: {args.checkpoint}", flush=True)

    print(f"best_val_acc={best_acc:.4f}")
    print(f"saved report: {args.report}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rough-dirs", nargs="+", default=DEFAULT_ROUGH_DIRS)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--target-mode",
        choices=["source", "family", "kurip_binary", "kurip_ako5_binary"],
        default="source",
    )
    parser.add_argument("--max-per-label", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
