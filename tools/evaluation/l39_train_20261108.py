# -*- coding: utf-8 -*-
"""L39 腕M: 線の層 -> 塗りの U-Net を**相手の作品だけで**学習し、試験側の頁の確率を書く。

**設定は全部登録 `doc/l39_line_to_fill_registration_20261108.md` §2 で固定した。ここで選ばない。**
**最後の歩の重みを使う。** 監視用の 2 頁の損失は印字するだけで、何も選ばない。

    python tools/evaluation/l39_train_20261108.py <試験の作品 ako6|ako7> <種> [歩数(煙試験だけ)]

出力: `results/l39_runs/test_<work>_seed<種>/{final.pth, train.log, prob/<pid>.png}`
(確率は 0..255 の 1/4 尺 PNG。**重みは追跡しない**。`results/**` は ignore、門は pre-commit)
"""
import glob
import math
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = '/home/sh1/deepl/lineart-fill-region'
CACHE = f'{ROOT}/results/l39_cache'
STEPS, BATCH, CROP, LR, WD = 8000, 8, 384, 1e-3, 1e-4
BASE, DEPTH = 32, 5
N_MONITOR = 2


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                               nn.ReLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    """素の U-Net。5段の縮小、基底 32(登録 §2)。"""

    def __init__(self, base=BASE, depth=DEPTH):
        super().__init__()
        ch = [min(base * 2 ** k, 512) for k in range(depth + 1)]
        self.inc = Block(1, ch[0])
        self.down = nn.ModuleList(Block(ch[k], ch[k + 1]) for k in range(depth))
        self.up = nn.ModuleList(Block(ch[k + 1] + ch[k], ch[k]) for k in reversed(range(depth)))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(Fn.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for u in self.up:
            s = skips.pop()
            y = Fn.interpolate(y, size=s.shape[-2:], mode='bilinear', align_corners=False)
            y = u(torch.cat([y, s], 1))
        return self.out(y)


def load_pages(work_filter):
    out = []
    for p in sorted(glob.glob(f'{CACHE}/*.pkl')):
        d = pickle.load(open(p, 'rb'))
        if work_filter(d['work']):
            out.append(d)
    return out


def crops(pages, rng):
    """コマを一様に選び、その外接矩形(1/4 尺)の中で一様な位置の切り出し(登録 §2)。"""
    pan = [(pi, p['box']) for pi, pg in enumerate(pages) for p in pg['panels']]
    while True:
        xs, ys, ms = [], [], []
        for _ in range(BATCH):
            pi, (y0, y1, x0, x1) = pan[rng.randrange(len(pan))]
            pg = pages[pi]
            H4, W4 = pg['L4'].shape
            a0, a1, b0, b1 = y0 // 4, min(H4, (y1 + 3) // 4), x0 // 4, min(W4, (x1 + 3) // 4)
            cy = rng.randint(a0, max(a0, a1 - 1))
            cx = rng.randint(b0, max(b0, b1 - 1))
            ty, tx = cy - CROP // 2, cx - CROP // 2
            def cut(a):
                o = np.zeros((CROP, CROP), a.dtype)
                sy0, sx0 = max(0, ty), max(0, tx)
                sy1, sx1 = min(a.shape[0], ty + CROP), min(a.shape[1], tx + CROP)
                o[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx] = a[sy0:sy1, sx0:sx1]
                return o
            x, y, m = cut(pg['L4']), cut(pg['F4']), cut(pg['M4'])
            if rng.random() < 0.5:
                x, y, m = x[:, ::-1], y[:, ::-1], m[:, ::-1]
            xs.append(x.astype(np.float32) / 255)
            ys.append(y.astype(np.float32))
            ms.append(m.astype(np.float32))
        yield (torch.from_numpy(np.stack(xs)[:, None].copy()),
               torch.from_numpy(np.stack(ys)[:, None].copy()),
               torch.from_numpy(np.stack(ms)[:, None].copy()))


def loss_fn(logit, y, m):
    bce = Fn.binary_cross_entropy_with_logits(logit, y, reduction='none')
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * m
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict(net, L4, dev):
    H, W = L4.shape
    Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    x = np.zeros((Hp, Wp), np.float32)
    x[:H, :W] = L4.astype(np.float32) / 255
    with torch.autocast('cuda', dtype=torch.float16):
        p = torch.sigmoid(net(torch.from_numpy(x)[None, None].to(dev)))[0, 0, :H, :W]
    return (p.float().cpu().numpy() * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def monitor_loss(net, pages, dev):
    tot, n = 0.0, 0
    for pg in pages:
        p = predict(net, pg['L4'], dev).astype(np.float32) / 255
        m = pg['M4'].astype(bool)
        y = pg['F4'][m].astype(np.float32)
        q = np.clip(p[m], 1e-4, 1 - 1e-4)
        tot += float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).sum())
        n += int(m.sum())
    return tot / max(n, 1)


def main():
    test_work, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    assert test_work in ('ako6', 'ako7')
    out = f'{ROOT}/results/l39_runs/test_{test_work}_seed{seed}' + ('' if steps == STEPS else f'_smoke{steps}')
    os.makedirs(f'{out}/prob', exist_ok=True)
    log = open(f'{out}/train.log', 'a')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train = load_pages(lambda w: w != test_work)
    test = load_pages(lambda w: w == test_work)
    tr_ids, te_ids = {p['pid'] for p in train}, {p['pid'] for p in test}
    say(f'L39 腕M 試験={test_work} 種={seed} 歩={steps}  学習頁 {len(train)} / 試験頁 {len(test)}')
    say(f'NC4 学習頁 ∩ 試験頁 = {len(tr_ids & te_ids)}(0 のはず)')
    assert not (tr_ids & te_ids) and train and test
    mon, fit = train[:N_MONITOR], train[N_MONITOR:]
    say(f'  監視 {[p["pid"] for p in mon]}(**何も選ばない**)/ 当てる {len(fit)} 頁')

    dev = 'cuda'
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.cuda.amp.GradScaler()
    gen = crops(fit, random.Random(seed))
    t0, run = time.time(), 0.0
    for s in range(1, steps + 1):
        x, y, m = (t.to(dev, non_blocking=True) for t in next(gen))
        with torch.autocast('cuda', dtype=torch.float16):
            logit = net(x)
        loss = loss_fn(logit.float(), y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += loss.item()
        if s % 200 == 0:
            say(f'  歩 {s}  学習損失 {run / 200:.4f}  {time.time() - t0:.0f}s')
            run = 0.0
        if s % 2000 == 0 or s == steps:
            net.eval()
            say(f'  歩 {s}  監視 BCE {monitor_loss(net, mon, dev):.4f}(印字だけ)')
            net.train()
    torch.save(net.state_dict(), f'{out}/final.pth')
    net.eval()
    for pg in test:
        cv2.imwrite(f'{out}/prob/{pg["pid"]}.png', predict(net, pg['L4'], dev))
    say(f'書いた: 最後の歩の重みと試験頁 {len(test)} 枚の確率 -> {out}')


if __name__ == '__main__':
    main()
