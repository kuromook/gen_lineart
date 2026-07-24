"""Dataset helpers for variable-aspect rough/line region manifests."""

import csv
import json
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def resolve_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return manifest_path.parent / path


def fit_square(image, size, mode):
    if mode == "resize_stretch":
        return TF.resize(image, (size, size))
    if mode != "square_pad":
        raise ValueError(f"unknown region fit mode: {mode}")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


class RegionManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        image_size,
        fit_mode="square_pad",
        autocontrast_rough=True,
        augment=False,
        rough_key=None,
        line_key=None,
        mask_key=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_manifest(self.manifest_path)
        self.image_size = image_size
        self.fit_mode = fit_mode
        self.autocontrast_rough = autocontrast_rough
        self.augment = augment
        self.rough_key = rough_key
        self.line_key = line_key
        self.mask_key = mask_key

    def __len__(self):
        return len(self.rows)

    def path_for(self, row, explicit_key, candidates):
        keys = [explicit_key] if explicit_key else []
        keys.extend(candidates)
        for key in keys:
            if key and row.get(key):
                return resolve_path(row[key], self.manifest_path)
        raise KeyError(f"missing path key; tried {keys}")

    def __getitem__(self, idx):
        row = self.rows[idx]
        rough_path = self.path_for(
            row,
            self.rough_key,
            ("aligned_rough_path", "v2_rough_path", "final_rough_path", "rough_path"),
        )
        line_path = self.path_for(
            row,
            self.line_key,
            ("aligned_line_path", "v2_line_path", "final_line_path", "line_path"),
        )
        rough = Image.open(rough_path).convert("L")
        line = Image.open(line_path).convert("L")
        mask = None
        if self.mask_key or row.get("valid_mask_path"):
            mask_path = self.path_for(row, self.mask_key, ("valid_mask_path", "mask_path"))
            mask = Image.open(mask_path).convert("L")
        if self.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough = fit_square(rough, self.image_size, self.fit_mode)
        line = fit_square(line, self.image_size, self.fit_mode)
        if mask is not None:
            mask = fit_square(mask, self.image_size, self.fit_mode)
        if self.augment and random.random() < 0.5:
            rough = TF.hflip(rough)
            line = TF.hflip(line)
            if mask is not None:
                mask = TF.hflip(mask)
        if mask is not None:
            return TF.to_tensor(rough), 1.0 - TF.to_tensor(line), TF.to_tensor(mask)
        return TF.to_tensor(rough), 1.0 - TF.to_tensor(line)


def region_collate(batch):
    if len(batch[0]) == 3:
        rough, target, mask = zip(*batch)
        return torch.stack(rough, dim=0), torch.stack(target, dim=0), torch.stack(mask, dim=0)
    rough, target = zip(*batch)
    return torch.stack(rough, dim=0), torch.stack(target, dim=0)
