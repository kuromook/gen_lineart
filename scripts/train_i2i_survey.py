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
from torch.utils.data import DataLoader, Dataset

from lineart.losses import edge_loss, ink_loss, tolerant_f1_loss
from lineart.model_zoo import MultiScalePatchDiscriminator, PatchDiscriminator, build_generator
from lineart.region_dataset import RegionManifestDataset, region_collate


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
    return (skel.astype(np.float32) / 255.0)[None, :, :]


class SketchDataset(Dataset):
    def __init__(
        self,
        rough_dir,
        line_dir,
        file_list,
        autocontrast_rough=True,
        augment=False,
        aux_dir=None,
        aux_dropout=0.0,
        aux_scale_min=1.0,
        use_skeleton=False,
    ):
        with open(file_list) as file:
            self.files = [line.strip() for line in file if line.strip()]
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.autocontrast_rough = autocontrast_rough
        self.augment = augment
        self.aux_dir = aux_dir
        self.aux_dropout = aux_dropout
        self.aux_scale_min = aux_scale_min
        self.use_skeleton = use_skeleton

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        rough = Image.open(os.path.join(self.rough_dir, name)).convert("L")
        line = Image.open(os.path.join(self.line_dir, name)).convert("L")
        if self.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough = TF.resize(rough, (IMAGE_SIZE, IMAGE_SIZE))
        line = TF.resize(line, (IMAGE_SIZE, IMAGE_SIZE))
        tensors = [TF.to_tensor(rough)]
        if self.aux_dir:
            aux_name = f"{Path(name).stem}_out.png"
            aux = Image.open(os.path.join(self.aux_dir, aux_name)).convert("L")
            aux = TF.resize(aux, (IMAGE_SIZE, IMAGE_SIZE))
            aux_tensor = TF.to_tensor(aux)
            if self.aux_dropout > 0.0 and random.random() < self.aux_dropout:
                scale = random.uniform(self.aux_scale_min, 1.0)
                aux_tensor = aux_tensor * scale + (1.0 - scale)
            tensors.append(aux_tensor)
        target = 1.0 - TF.to_tensor(line)
        if self.augment and random.random() < 0.5:
            rough = TF.hflip(rough)
            line = TF.hflip(line)
            tensors = [TF.hflip(tensor) for tensor in tensors]
            target = TF.hflip(target)
        if self.use_skeleton:
            skeleton = torch.from_numpy(skeletonize_ink(target[0].numpy()))
            return torch.cat(tensors, dim=0), target, skeleton
        return torch.cat(tensors, dim=0), target


class UnpairedRoughDataset(Dataset):
    """Rough-only tiles with no paired line-art GT (e.g. dataset/unpaired_rough).
    Yields just the model input tensor (rough, optionally + aux channel) --
    there is no target to return. Meant to be consumed only for the
    adversarial branch of training (see --unpaired-weight), since every loss
    that needs a GT line target is unavailable here."""

    def __init__(self, rough_dir, file_list, aux_dir=None, autocontrast_rough=True):
        with open(file_list) as file:
            self.files = [line.strip() for line in file if line.strip()]
        self.rough_dir = rough_dir
        self.aux_dir = aux_dir
        self.autocontrast_rough = autocontrast_rough

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        rough = Image.open(os.path.join(self.rough_dir, name)).convert("L")
        if self.autocontrast_rough:
            rough = ImageOps.autocontrast(rough, cutoff=0)
        rough = TF.resize(rough, (IMAGE_SIZE, IMAGE_SIZE))
        tensors = [TF.to_tensor(rough)]
        if self.aux_dir:
            aux_name = f"{Path(name).stem}_out.png"
            aux = Image.open(os.path.join(self.aux_dir, aux_name)).convert("L")
            aux = TF.resize(aux, (IMAGE_SIZE, IMAGE_SIZE))
            tensors.append(TF.to_tensor(aux))
        return torch.cat(tensors, dim=0)


def load_generator_weights(model, path, device, strict=True):
    if not path:
        return
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "G_state" in checkpoint:
        state = checkpoint["G_state"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"resume non-strict missing={len(missing)} unexpected={len(unexpected)}")
    print(f"resume_generator={path}")


def save_checkpoint(path, model_name, G, D=None, epoch=0, opt_G=None, opt_D=None, in_channels=1):
    payload = {
        "model_name": model_name,
        "in_channels": in_channels,
        "epoch": epoch,
        "model_state": G.state_dict(),
        "G_state": G.state_dict(),
    }
    if D is not None:
        payload["D_state"] = D.state_dict()
    if opt_G is not None:
        payload["opt_G"] = opt_G.state_dict()
    if opt_D is not None:
        payload["opt_D"] = opt_D.state_dict()
    torch.save(payload, path)


def train_reconstruction(args, device):
    in_channels = 2 if args.aux_dir else 1
    G = build_generator(args.model, in_channels=in_channels).to(device)
    load_generator_weights(G, args.resume_generator, device, strict=args.strict_resume)
    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    return G, None, opt_G, None, bce


def masked_mean(loss, mask=None):
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def binary_confidence_loss(pred):
    """Penalize uncertain mid-gray ink probabilities."""
    return (pred * (1.0 - pred)).mean()


def soft_width_loss(pred):
    """Penalize broad local ink spread while preserving sparse line candidates."""
    local = F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    return (pred * local).mean()


def soft_threshold(pred, threshold, sharpness):
    """Differentiable approximation of hard ink thresholding."""
    return torch.sigmoid((pred - threshold) * sharpness)


def structure_pyramid_loss(pred, target):
    """Compare Sobel/DoG-like structure at two scales."""
    losses = []
    for scale in (1, 2):
        if scale > 1:
            pred_s = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_s = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        else:
            pred_s = pred
            target_s = target
        pred_blur = F.avg_pool2d(pred_s, kernel_size=5, stride=1, padding=2)
        target_blur = F.avg_pool2d(target_s, kernel_size=5, stride=1, padding=2)
        losses.append(F.l1_loss(pred_s - pred_blur, target_s - target_blur))
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=pred.dtype,
            device=pred.device,
        ).unsqueeze(0)
        sobel_y = sobel_x.transpose(-1, -2)
        pred_gx = F.conv2d(pred_s, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_s, sobel_y, padding=1)
        target_gx = F.conv2d(target_s, sobel_x, padding=1)
        target_gy = F.conv2d(target_s, sobel_y, padding=1)
        losses.append(F.l1_loss(pred_gx, target_gx))
        losses.append(F.l1_loss(pred_gy, target_gy))
    return sum(losses) / len(losses)


def background_haze_loss(pred, target, radius):
    """Penalize ink predicted away from any target line support."""
    kernel = radius * 2 + 1
    near_line = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    background = (near_line <= 0.01).to(pred.dtype)
    denom = background.sum().clamp_min(1.0)
    return (pred * background).sum() / denom


def adversarial_mse(scores, target_value):
    if isinstance(scores, (list, tuple)):
        return sum(F.mse_loss(score, torch.full_like(score, target_value)) for score in scores) / len(scores)
    return F.mse_loss(scores, torch.full_like(scores, target_value))


def feature_matching_loss(fake_features, real_features):
    loss = 0.0
    count = 0
    for fake_group, real_group in zip(fake_features, real_features):
        for fake, real in zip(fake_group, real_group):
            loss = loss + F.l1_loss(fake, real.detach())
            count += 1
    return loss / max(count, 1)


def train(args):
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("--require-cuda was set, but torch.cuda.is_available() is False")
    if args.region_manifest:
        if args.aux_dir:
            raise ValueError("--aux-dir is not supported with --region-manifest yet")
        if args.skeleton_weight > 0.0:
            raise ValueError("--skeleton-weight is not supported with --region-manifest yet")
        dataset = RegionManifestDataset(
            args.region_manifest,
            image_size=args.image_size,
            fit_mode=args.region_fit_mode,
            autocontrast_rough=args.autocontrast,
            augment=args.augment,
            rough_key=args.region_rough_key,
            line_key=args.region_line_key,
            mask_key=args.region_mask_key,
        )
        collate_fn = region_collate
    else:
        dataset = SketchDataset(
            args.rough_dir,
            args.line_dir,
            args.file_list,
            autocontrast_rough=args.autocontrast,
            augment=args.augment,
            aux_dir=args.aux_dir,
            aux_dropout=args.aux_dropout,
            aux_scale_min=args.aux_scale_min,
            use_skeleton=args.skeleton_weight > 0.0,
        )
        collate_fn = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    unpaired_loader = None
    if args.unpaired_rough_file_list:
        if not args.gan:
            raise ValueError("--unpaired-rough-file-list requires --gan (the unpaired branch is adversarial-only)")
        unpaired_dataset = UnpairedRoughDataset(
            rough_dir=args.unpaired_rough_dir,
            file_list=args.unpaired_rough_file_list,
            aux_dir=args.unpaired_rough_aux_dir,
            autocontrast_rough=args.autocontrast,
        )
        unpaired_loader = DataLoader(
            unpaired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    G, D, opt_G, opt_D, bce = train_reconstruction(args, device)
    if args.gan:
        if args.multiscale_gan:
            D = MultiScalePatchDiscriminator().to(device)
        else:
            D = PatchDiscriminator().to(device)
        opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    print(f"device={device}")
    print(f"model={args.model} gan={args.gan} multiscale_gan={args.multiscale_gan}")
    print(f"in_channels={2 if args.aux_dir else 1} aux_dir={args.aux_dir or ''}")
    print(f"aux_dropout={args.aux_dropout} aux_scale_min={args.aux_scale_min}")
    if args.region_manifest:
        print(
            f"region_manifest={args.region_manifest} rows={len(dataset)} "
            f"image_size={args.image_size} fit_mode={args.region_fit_mode} "
            f"rough_key={args.region_rough_key or 'auto'} "
            f"line_key={args.region_line_key or 'auto'} "
            f"mask_key={args.region_mask_key or 'auto'}"
        )
    else:
        print(f"file_list={args.file_list} rows={len(dataset)} image_size={args.image_size}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(
        f"loss: bce={args.bce_weight} l1={args.l1_weight} "
        f"tolerant={args.shape_weight} ink={args.ink_weight} "
        f"binary={args.binary_weight} skeleton={args.skeleton_weight} side={args.side_weight} "
        f"width={args.width_weight} "
        f"thresh_shape={args.threshold_shape_weight} "
        f"thresh_ink={args.threshold_ink_weight} "
        f"thresh={args.threshold_value}@{args.threshold_sharpness} "
        f"structure={args.structure_weight} edge={args.edge_weight} "
        f"bg_haze={args.background_haze_weight}@{args.background_haze_radius}px "
        f"fm={args.feature_match_weight} "
        f"adv={args.adv_weight}"
    )
    if unpaired_loader is not None:
        print(
            f"unpaired_rough: file_list={args.unpaired_rough_file_list} "
            f"rows={len(unpaired_dataset)} weight={args.unpaired_weight}"
        )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        G.train()
        if D is not None:
            D.train()
        g_sum = d_sum = 0.0
        steps = 0
        unpaired_iter = iter(unpaired_loader) if unpaired_loader is not None else None
        for batch in loader:
            valid_mask = None
            skeleton = None
            if args.skeleton_weight > 0.0:
                rough, target, skeleton = batch
                skeleton = skeleton.to(device)
            elif len(batch) == 3:
                rough, target, valid_mask = batch
            else:
                rough, target = batch
            rough = rough.to(device)
            target = target.to(device)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device).clamp(0.0, 1.0)

            pred_logits = G(rough)
            aux_skeleton_logits = None
            side_logits_list = None
            if isinstance(pred_logits, tuple):
                pred_logits, aux_output = pred_logits
                if isinstance(aux_output, list):
                    side_logits_list = aux_output
                else:
                    aux_skeleton_logits = aux_output
            pred = torch.sigmoid(pred_logits)
            pred_thresh = soft_threshold(
                pred,
                args.threshold_value,
                args.threshold_sharpness,
            )
            masked_pred = pred if valid_mask is None else pred * valid_mask
            masked_target = target if valid_mask is None else target * valid_mask
            masked_pred_thresh = pred_thresh if valid_mask is None else pred_thresh * valid_mask
            loss_recon = (
                args.bce_weight * masked_mean(bce(pred_logits, target), valid_mask)
                + args.l1_weight * masked_mean((pred - target).abs(), valid_mask)
                + args.shape_weight * tolerant_f1_loss(masked_pred, masked_target)
                + args.ink_weight * ink_loss(masked_pred, masked_target)
                + args.binary_weight * masked_mean(pred * (1.0 - pred), valid_mask)
                + args.width_weight * soft_width_loss(masked_pred)
                + args.threshold_shape_weight * tolerant_f1_loss(masked_pred_thresh, masked_target)
                + args.threshold_ink_weight * ink_loss(masked_pred_thresh, masked_target)
                + args.structure_weight * structure_pyramid_loss(masked_pred, masked_target)
                + args.background_haze_weight
                * background_haze_loss(masked_pred, masked_target, args.background_haze_radius)
            )
            if args.edge_weight > 0.0:
                loss_recon = loss_recon + args.edge_weight * edge_loss(masked_pred, masked_target)
            if skeleton is not None:
                skeleton_pred_logits = aux_skeleton_logits if aux_skeleton_logits is not None else pred_logits
                loss_recon = loss_recon + args.skeleton_weight * masked_mean(bce(skeleton_pred_logits, skeleton), None)
            if side_logits_list is not None and args.side_weight > 0.0:
                side_loss = 0.0
                for side_logits in side_logits_list:
                    side_target = F.adaptive_avg_pool2d(target, side_logits.shape[-2:])
                    side_loss = side_loss + masked_mean(bce(side_logits, side_target), None)
                loss_recon = loss_recon + args.side_weight * (side_loss / len(side_logits_list))
            loss_G = loss_recon
            if D is not None:
                if args.feature_match_weight > 0.0:
                    fake_score, fake_features = D(rough[:, :1], pred, return_features=True)
                    with torch.no_grad():
                        _, real_features = D(rough[:, :1], target, return_features=True)
                    loss_fm = feature_matching_loss(fake_features, real_features)
                else:
                    fake_score = D(rough[:, :1], pred)
                    loss_fm = 0.0
                loss_adv = adversarial_mse(fake_score, 1.0)
                loss_G = loss_G + args.adv_weight * loss_adv
                if args.feature_match_weight > 0.0:
                    loss_G = loss_G + args.feature_match_weight * loss_fm

            if unpaired_iter is not None and args.unpaired_weight > 0.0:
                try:
                    unpaired_rough = next(unpaired_iter)
                except StopIteration:
                    unpaired_iter = iter(unpaired_loader)
                    unpaired_rough = next(unpaired_iter)
                unpaired_rough = unpaired_rough.to(device)
                unpaired_pred_logits = G(unpaired_rough)
                if isinstance(unpaired_pred_logits, tuple):
                    unpaired_pred_logits = unpaired_pred_logits[0]
                unpaired_pred = torch.sigmoid(unpaired_pred_logits)
                unpaired_fake_score = D(unpaired_rough[:, :1], unpaired_pred)
                loss_unpaired_adv = adversarial_mse(unpaired_fake_score, 1.0)
                loss_G = loss_G + args.unpaired_weight * loss_unpaired_adv

            opt_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            if D is not None:
                with torch.no_grad():
                    fake_logits = G(rough)
                    if isinstance(fake_logits, tuple):
                        fake_logits = fake_logits[0]
                    fake = torch.sigmoid(fake_logits)
                opt_D.zero_grad()
                real_score = D(rough[:, :1], target)
                fake_score = D(rough[:, :1], fake.detach())
                loss_D = 0.5 * (
                    adversarial_mse(real_score, 0.9)
                    + adversarial_mse(fake_score, 0.0)
                )
                loss_D.backward()
                opt_D.step()
                d_sum += loss_D.item()

            g_sum += loss_G.item()
            steps += 1

        avg_g = g_sum / max(steps, 1)
        avg_d = d_sum / max(steps, 1) if D is not None else 0.0
        print(f"Epoch {epoch:03d}/{args.epochs}: G={avg_g:.4f} D={avg_d:.4f}", flush=True)
        if avg_g < best_loss:
            best_loss = avg_g
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pth"),
                args.model,
                G,
                D=D,
                epoch=epoch,
                opt_G=opt_G,
                opt_D=opt_D,
                in_channels=2 if args.aux_dir else 1,
            )
        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"epoch{epoch:03d}.pth"),
                args.model,
                G,
                D=D,
                epoch=epoch,
                opt_G=opt_G,
                opt_D=opt_D,
                in_channels=2 if args.aux_dir else 1,
            )
    print(f"saved: {args.checkpoint_dir}/best.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "unet",
            "unet_skip25",
            "unet_skip50",
            "resnet",
            "cleanup",
            "cleanupdark",
            "dualhead",
            "hed",
            "attn",
            "maskcleanup",
            "flowmaskcleanup",
            "flowmaskunet",
            "softflowmaskunet",
        ],
        required=True,
    )
    parser.add_argument("--gan", action="store_true")
    parser.add_argument("--multiscale-gan", action="store_true")
    parser.add_argument("--file-list", default="dataset/pairs_480/valid_train_milddup800_clean.txt")
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line")
    parser.add_argument("--region-manifest", default=None)
    parser.add_argument("--region-fit-mode", choices=["square_pad", "resize_stretch"], default="square_pad")
    parser.add_argument("--region-rough-key", default=None)
    parser.add_argument("--region-line-key", default=None)
    parser.add_argument("--region-mask-key", default=None)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--aux-dir", default=None)
    parser.add_argument("--aux-dropout", type=float, default=0.0)
    parser.add_argument("--aux-scale-min", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume-generator", default=None)
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-d", type=float, default=2e-5)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--bce-weight", type=float, default=0.6)
    parser.add_argument("--l1-weight", type=float, default=0.2)
    parser.add_argument("--shape-weight", type=float, default=0.05)
    parser.add_argument("--ink-weight", type=float, default=0.02)
    parser.add_argument("--binary-weight", type=float, default=0.0)
    parser.add_argument("--skeleton-weight", type=float, default=0.0)
    parser.add_argument("--side-weight", type=float, default=0.0)
    parser.add_argument("--width-weight", type=float, default=0.0)
    parser.add_argument("--threshold-shape-weight", type=float, default=0.0)
    parser.add_argument("--threshold-ink-weight", type=float, default=0.0)
    parser.add_argument("--threshold-value", type=float, default=0.52)
    parser.add_argument("--threshold-sharpness", type=float, default=24.0)
    parser.add_argument("--structure-weight", type=float, default=0.0)
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.0,
        help="Canny-edge L1 loss (lineart.losses.edge_loss), matching the "
        "pre-GAN notebooks/gen_lineart.ipynb recipe",
    )
    parser.add_argument("--background-haze-weight", type=float, default=0.0)
    parser.add_argument("--background-haze-radius", type=int, default=9)
    parser.add_argument("--feature-match-weight", type=float, default=0.0)
    parser.add_argument("--adv-weight", type=float, default=0.02)
    parser.add_argument("--unpaired-rough-file-list", default=None)
    parser.add_argument("--unpaired-rough-dir", default=None)
    parser.add_argument("--unpaired-rough-aux-dir", default=None)
    parser.add_argument("--unpaired-weight", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--autocontrast", action="store_true", default=True)
    parser.add_argument("--no-autocontrast", dest="autocontrast", action="store_false")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
