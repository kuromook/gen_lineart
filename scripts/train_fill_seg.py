"""Stage 2 M0 / M1: predict the solid-fill mask from the rough.

Registered 2026-10-10 in doc/work_log.md. Plain supervised segmentation:
U-Net, BCE + Dice, **no adversarial term**. The falsifier the user was given
stands: if an adversarial term is added, or if the output is ever read as line
art rather than as a mask, prohibition 2 has been violated and that goes in
the headline, not a footnote.

  M0  window only.      input 1ch  = the 480px native-scale rough crop
  M1  window + context. input 4ch  = the same crop, the WHOLE koma downscaled
                        to 480, and two constant planes carrying the crop's
                        normalised centre (y, x)

M1 - M0 is the registered question: does seeing the whole 面 help predict it?
NC4 (M1 run with another panel's context) is what says whether the answer came
from the context stream; it lives in the scorer, not here.

The target is the fill mask of the GT line art, computed on the WHOLE panel
and then cropped -- a distance transform is global, so computing it per window
would change the mask at every window edge.

Training discipline (lesson 7): the holdout is scored at every snapshot and
training loss is never read as evidence of anything.

shortcut: windows are sampled, not enumerated, so an epoch is a fixed number
of sampled windows rather than full coverage; raised if a run shows the
holdout curve still moving when the budget ends.

Usage:
  train_fill_seg.py --arm M0 --smoke
  PYTHONUNBUFFERED=1 train_fill_seg.py --arm M1 --steps 20000 \
      --eval-snapshot-steps 2000
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "evaluation"))
from fill_mask import fill_mask, ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from koma_windows import SIZE, cut, grid, pad_to, stitch  # noqa: E402
from region_metrics import score_panel  # noqa: E402

OUT = Path("results/fill_seg_20261010")
SEED = 20261010
WINDOWS_PER_PANEL = 4
CTX = 480


# --------------------------------------------------------------------------
# model: the project's U-Net, with the input width made an argument

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.in1(self.conv1(x)))
        return x + self.in2(self.conv2(h))


class Dil(nn.Module):
    def __init__(self, i, o, d=2):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 3, dilation=d, padding=d)
        self.n = nn.InstanceNorm2d(o)

    def forward(self, x):
        return F.relu(self.n(self.conv(x)))


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1),
                                  nn.InstanceNorm2d(64), nn.ReLU(True), ResBlock(64))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                  nn.InstanceNorm2d(128), nn.ReLU(True), ResBlock(128))
        self.enc3 = nn.Sequential(Dil(128, 256, 2), ResBlock(256))
        self.bott = nn.Sequential(Dil(256, 512, 4), ResBlock(512), Dil(512, 512, 4))
        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                  nn.InstanceNorm2d(256), nn.ReLU(True), ResBlock(256))
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                  nn.InstanceNorm2d(128), nn.ReLU(True), ResBlock(128))
        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(Dil(128, 64, 1), ResBlock(64))
        self.out = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def in_channels(arm):
    return 1 if arm == "M0" else 4


# --------------------------------------------------------------------------
# data

def panel_tensors(name, root=ROOT):
    """(rough_float, fill_bool, context_float) at panel scale, padded to >=480."""
    rough = cv2.imread(str(Path(root) / "rough" / name), cv2.IMREAD_GRAYSCALE)
    line = cv2.imread(str(Path(root) / "line" / name), cv2.IMREAD_GRAYSCALE)
    if rough is None or line is None:
        return None
    if rough.shape != line.shape:
        return None
    fill = fill_mask(ink_of(line))              # computed on the WHOLE panel
    rough, orig = pad_to(rough, SIZE, value=255)
    fillp, _ = pad_to(fill.astype(np.uint8), SIZE, value=0)
    ctx = cv2.resize(rough, (CTX, CTX), interpolation=cv2.INTER_AREA)
    return rough.astype(np.float32) / 255.0, fillp.astype(bool), \
        ctx.astype(np.float32) / 255.0, orig


def make_input(arm, crop, ctx, y, x, h, w):
    """crop: (480,480) float. Returns (C,480,480) float32."""
    chans = [crop]
    if arm == "M1":
        cy = (y + SIZE / 2) / max(h, 1)
        cx = (x + SIZE / 2) / max(w, 1)
        chans += [ctx,
                  np.full_like(crop, np.float32(cy)),
                  np.full_like(crop, np.float32(cx))]
    return np.stack(chans, 0)


class PanelWindows(Dataset):
    """One item is one panel; it yields WINDOWS_PER_PANEL random windows, so a
    4604px PNG is decoded once per several training samples."""

    def __init__(self, names, arm, per_panel=WINDOWS_PER_PANEL, seed=SEED):
        self.names = names
        self.arm = arm
        self.per = per_panel
        self.seed = seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = random.Random(self.seed + i * 7919 + torch.initial_seed() % 100003)
        for _ in range(4):
            t = panel_tensors(self.names[(i + rng.randrange(97)) % len(self.names)])
            if t is not None:
                break
        else:
            t = None
        if t is None:
            z = np.zeros((in_channels(self.arm), SIZE, SIZE), np.float32)
            return torch.from_numpy(z), torch.zeros(1, SIZE, SIZE)
        rough, fill, ctx, _ = t
        h, w = rough.shape
        xs, ys = [], []
        g = grid(h, w)
        picks = [g[rng.randrange(len(g))] for _ in range(self.per)]
        for (y, x) in picks:
            xs.append(make_input(self.arm, rough[y:y + SIZE, x:x + SIZE], ctx, y, x, h, w))
            ys.append(fill[y:y + SIZE, x:x + SIZE][None].astype(np.float32))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def collate(batch):
    xs = torch.cat([b[0] for b in batch], 0)
    ys = torch.cat([b[1] for b in batch], 0)
    return xs, ys


# --------------------------------------------------------------------------
# loss

def bce_dice(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean(), float(bce), float(dice.mean())


# --------------------------------------------------------------------------
# holdout scoring: panel-level, through the registered stitcher

@torch.no_grad()
def score_holdout(model, names, arm, device, ctx_from=None, limit=40):
    """ctx_from: if given, panel i is scored with panel ctx_from[i]'s context
    -- that is NC4, and it lives here so the same code path produces it."""
    model.eval()
    rows = []
    for i, name in enumerate(names[:limit]):
        t = panel_tensors(name)
        if t is None:
            continue
        rough, fill, ctx, orig = t
        if ctx_from is not None:
            o = panel_tensors(ctx_from[i % len(ctx_from)])
            if o is not None:
                ctx = o[2]
        h, w = rough.shape
        coords = grid(h, w)
        preds = []
        for (y, x) in coords:
            xin = make_input(arm, rough[y:y + SIZE, x:x + SIZE], ctx, y, x, h, w)
            p = torch.sigmoid(model(torch.from_numpy(xin)[None].to(device)))
            preds.append(p[0, 0].cpu().numpy())
        prob = stitch(preds, coords, (h, w), reduce="mean")
        pred = prob >= 0.5
        gt = fill[:h, :w][:prob.shape[0], :prob.shape[1]]
        r = score_panel(pred, gt)
        r["panel"] = name
        rows.append(r)
    model.train()
    if not rows:
        return {}, []
    agg = {"n": len(rows),
           "region_f1": float(np.mean([r["region_f1"] for r in rows])),
           "pixel_iou": float(np.mean([r["iou"] for r in rows])),
           "pred_area_frac": float(np.mean([r["pred_area_frac"] for r in rows])),
           "gt_area_frac": float(np.mean([r["gt_area_frac"] for r in rows]))}
    return agg, rows


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=("M0", "M1"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-panels", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-snapshot-steps", type=int, default=2000)
    ap.add_argument("--eval-panels", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.eval_snapshot_steps, args.eval_panels = 12, 6, 3
        args.workers, args.batch_panels = 0, 1

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    split = load_split()
    train = [n for n, _, _ in split["train"]]
    held = [n for n, _, _ in split["held"]]
    if args.smoke:
        train, held = train[:8], held[:4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.arm}{'_smoke' if args.smoke else ''}"
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"train_{tag}.jsonl", "w")

    print(f"arm {args.arm}  in_channels {in_channels(args.arm)}  device {device}  "
          f"train panels {len(train)}  held panels {len(held)}  steps {args.steps}")
    # self-check with a known value: the input width must match the arm, and a
    # forward pass must return the window shape. Cheap, and it fails loudly.
    model = UNet(in_channels(args.arm)).to(device)
    with torch.no_grad():
        probe = model(torch.zeros(1, in_channels(args.arm), SIZE, SIZE, device=device))
    assert probe.shape == (1, 1, SIZE, SIZE), probe.shape
    assert in_channels("M0") == 1 and in_channels("M1") == 4
    print(f"self-check: forward {tuple(probe.shape)} OK")

    ds = PanelWindows(train, args.arm)
    dl = DataLoader(ds, batch_size=args.batch_panels, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step, t0 = 0, time.time()
    it = iter(dl)
    while step < args.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        x, y = x.to(device), y.to(device)
        loss, bce, dice = bce_dice(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0 or args.smoke:
            print(f"[{tag}] step {step}/{args.steps}  loss {float(loss):.4f} "
                  f"(bce {bce:.4f} dice {dice:.4f})  {step / (time.time() - t0):.2f} it/s",
                  flush=True)
        if step % args.eval_snapshot_steps == 0 or step == args.steps:
            agg, _ = score_holdout(model, held, args.arm, device, limit=args.eval_panels)
            # training loss is NOT evidence; the holdout at a snapshot is.
            print(f"[{tag}] SNAPSHOT step {step}  holdout region-F1 "
                  f"{agg.get('region_f1', float('nan')):.4f}  pixel IoU "
                  f"{agg.get('pixel_iou', float('nan')):.4f}  pred area "
                  f"{agg.get('pred_area_frac', float('nan')):.4f} vs GT "
                  f"{agg.get('gt_area_frac', float('nan')):.4f}", flush=True)
            log.write(json.dumps({"step": step, "arm": args.arm, **agg}) + "\n")
            log.flush()
            torch.save({"step": step, "arm": args.arm,
                        "model": model.state_dict()}, OUT / f"{tag}_step{step}.pt")
    log.close()
    print(f"done: {OUT}/{tag}_step{step}.pt")


if __name__ == "__main__":
    main()
