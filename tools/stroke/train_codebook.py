#!/usr/bin/env python
"""A codebook of clusters: each cluster of strokes gets one word out of 5,000.

C-3 (2026-09-19) found that clusters recur across works but as near relatives,
about 2% of their size apart, not as twins -- a learned codebook, not nearest
neighbour lookup. It is also the proposal half of generate-and-select: the
cloze model can choose, but something has to offer candidates.

Encoder: a set Transformer over the cluster's strokes (normalised frame) -> 5
dims -> FSQ. Decoder: 32 slots (exist, 16 points, width) matched to the true
strokes with the Hungarian algorithm; the point loss is direction-symmetric,
since a raster stroke has no drawing direction.

FSQ is written here rather than imported (user decision 2026-09-19): bound each
dim with tanh, round to its levels, pass the gradient straight through. There is
no table to leave codes behind in, which is why it was chosen after this track's
mean-collapse failure; the collapse check below is still the first bar.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

LEVELS = (8, 5, 5, 5, 5)
SLOTS, P = 32, 16


class FSQ(nn.Module):
    def __init__(self, levels=LEVELS, eps=1e-3):
        super().__init__()
        L = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("L", L)
        self.register_buffer("half_l", (L - 1) * (1 - eps) / 2)
        self.register_buffer("offset", torch.where(L % 2 == 0, 0.5, 0.0))
        self.register_buffer("shift", torch.atanh(self.offset / self.half_l))
        self.register_buffer("half_w", torch.floor(L / 2))
        basis = torch.cumprod(torch.tensor((1,) + tuple(levels[:-1]), dtype=torch.float32), 0)
        self.register_buffer("basis", basis)
        self.n_codes = int(np.prod(levels))

    def forward(self, z):
        b = torch.tanh(z + self.shift) * self.half_l - self.offset
        q = b + (torch.round(b) - b).detach()            # straight-through
        return q / self.half_w

    def index(self, q):
        ints = torch.round(q * self.half_w) + self.half_w
        return (ints * self.basis).sum(-1).long()

    def from_index(self, idx):
        ints = (idx[:, None] // self.basis.long()) % self.L.long()
        return (ints.float() - self.half_w) / self.half_w


def stroke_feats(pts, width):
    """(B,S,16,2),(B,S) normalised -> (B,S,35)"""
    arc = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1, keepdim=True)
    return torch.cat([pts.flatten(2) / 28.0, width[..., None] / 10.0, arc / 56.0,
                      (pts[:, :, -1] - pts[:, :, 0]).norm(dim=-1, keepdim=True) / 56.0], -1)


class Codebook(nn.Module):
    def __init__(self, d=256, layers=4, heads=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(35, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        # LayerNorm + a x2 gain before quantisation: measured 2026-09-19, without
        # it every cluster started in the same FSQ bin (z ~ 0 rounds to 0 in every
        # dim), the decoder only ever saw one input, learned the mean, and after 8
        # epochs only 4-5 codes were in use -- the collapse the first bar guards.
        self.pre_q = nn.LayerNorm(d)
        self.to_z = nn.Linear(d, len(LEVELS))
        nn.init.normal_(self.to_z.weight, std=2.0 / np.sqrt(d))
        self.fsq = FSQ()
        self.from_z = nn.Sequential(nn.Linear(len(LEVELS), d), nn.GELU(), nn.Linear(d, d))
        self.slots = nn.Parameter(torch.randn(1, SLOTS, d) * 0.02)
        dec = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(dec, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1 + P * 2 + 1))

    def encode(self, pts, width, mask):
        x = self.inp(stroke_feats(pts, width))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.enc(x, src_key_padding_mask=m)
        return self.fsq(self.to_z(self.pre_q(h[:, 0])))

    def decode(self, q):
        h = self.slots + self.from_z(q)[:, None, :]
        o = self.head(self.dec(h))
        return o[..., 0], o[..., 1:1 + P * 2].view(len(q), SLOTS, P, 2) * 28.0, o[..., -1] * 10.0

    def forward(self, pts, width, mask):
        q = self.encode(pts, width, mask)
        return (q,) + self.decode(q)


def match_loss(ex, pp, pw, pts, width, mask):
    """Hungarian-matched, direction-symmetric set loss."""
    B = len(pts)
    d_f = (pp[:, :, None] - pts[:, None]).abs().mean((-1, -2))              # (B,S,S)
    d_r = (pp[:, :, None] - pts.flip(2)[:, None]).abs().mean((-1, -2))
    cost = torch.minimum(d_f, d_r) + 0.1 * (pw[:, :, None] - width[:, None]).abs()
    cost_c = cost.detach().float().cpu().numpy()
    mk = mask.cpu().numpy()
    bi, pi, ti = [], [], []
    for b in range(B):
        tj = np.flatnonzero(mk[b])
        r, c = linear_sum_assignment(cost_c[b][:, tj])
        bi += [b] * len(r); pi += r.tolist(); ti += tj[c].tolist()
    bi = torch.tensor(bi, device=pts.device); pi = torch.tensor(pi, device=pts.device); ti = torch.tensor(ti, device=pts.device)
    pts_loss = torch.minimum(d_f[bi, pi, ti], d_r[bi, pi, ti]).mean()
    w_loss = (pw[bi, pi] - width[bi, ti]).abs().mean()
    target = torch.zeros_like(ex); target[bi, pi] = 1.0
    ex_loss = nn.functional.binary_cross_entropy_with_logits(ex, target)
    return pts_loss + 0.1 * w_loss + ex_loss, pts_loss.detach()


def load(root, split, device):
    meta = list(csv.DictReader(open(Path(root) / "meta.csv")))
    sel = np.array([m["split"] == split for m in meta])
    pts = torch.from_numpy(np.load(Path(root) / "pts.npy")[sel]).to(device)
    width = torch.from_numpy(np.load(Path(root) / "width.npy")[sel]).to(device)
    mask = torch.from_numpy(np.load(Path(root) / "mask.npy")[sel]).to(device)
    return pts, width, mask, [m for m, s in zip(meta, sel) if s]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/cluster_set_20260919")
    p.add_argument("--out", default="results/codebook_20260919")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N training clusters (quick checks)")
    a = p.parse_args()
    torch.manual_seed(20260919)
    dev = torch.device("cuda")
    tp, tw, tm, _ = load(a.data, "train", dev)
    vp, vw, vm, _ = load(a.data, "test", dev)
    if a.limit:
        tp, tw, tm = tp[:a.limit], tw[:a.limit], tm[:a.limit]; vp, vw, vm = vp[:4096], vw[:4096], vm[:4096]
    if a.smoke:
        tp, tw, tm = tp[:2048], tw[:2048], tm[:2048]; vp, vw, vm = vp[:512], vw[:512], vm[:512]; a.epochs = 2
    model = Codebook().to(dev)
    print(f"train {len(tp)}  test {len(vp)}  params {sum(x.numel() for x in model.parameters())/1e6:.1f}M  codes {model.fsq.n_codes}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tp) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, steps), pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = torch.randperm(len(tp), device=dev); tot = 0.0; n = 0
        model.train()
        for i in range(0, len(tp) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            pts, w, m = tp[idx].float(), tw[idx].float(), tm[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, ex, pp, pw = model(pts, w, m)
            loss, pl = match_loss(ex.float(), pp.float(), pw.float(), pts, w, m)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(pl); n += 1
        # held-out: point loss, code usage, the collapse check
        model.eval(); codes = []; vl = 0.0; vn = 0; arc_pred = []; arc_true = []; n_pred = []; n_true = []
        with torch.no_grad():
            for i in range(0, len(vp), 1024):
                pts, w, m = vp[i:i + 1024].float(), vw[i:i + 1024].float(), vm[i:i + 1024]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q, ex, pp, pw = model(pts, w, m)
                _l, pl = match_loss(ex.float(), pp.float(), pw.float(), pts, w, m)
                vl += float(pl); vn += 1
                codes.append(model.fsq.index(q.float()).cpu())
                keep = ex.float() > 0
                n_pred.append(keep.sum(1).cpu()); n_true.append(m.sum(1).cpu())
                arc_pred.append((pp.float()[:, :, 1:] - pp.float()[:, :, :-1]).norm(dim=-1).sum(-1)[keep].cpu())
                arc_true.append((pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1)[m].cpu())
        codes = torch.cat(codes)
        cnt = torch.bincount(codes, minlength=model.fsq.n_codes).float()
        pr = cnt / cnt.sum(); ppl = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
        ap = torch.cat(arc_pred); at = torch.cat(arc_true)
        ratio = float(ap.std() / at.std()) if len(ap) > 1 else 0.0
        npr, ntr = torch.cat(n_pred).float(), torch.cat(n_true).float()
        row = {"epoch": ep, "train_pts": tot / max(n, 1), "test_pts": vl / max(vn, 1), "perplexity": ppl,
               "used_codes": int((cnt > 0).sum()), "arc_std_ratio": ratio,
               "n_pred": float(npr.mean()), "n_true": float(ntr.mean()), "sec": time.time() - t0}
        hist.append(row)
        print(f"ep {ep:>3} 学習 {row['train_pts']:.3f} 評価 {row['test_pts']:.3f}  使用符号 {row['used_codes']} "
              f"perplexity {ppl:.0f}  弧長の標準偏差比 {ratio:.2f}  本数 予測{row['n_pred']:.1f}/正解{row['n_true']:.1f}  {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "codebook.pt")


if __name__ == "__main__":
    main()
