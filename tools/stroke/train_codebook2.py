#!/usr/bin/env python
"""Codebook, second design: a word plus detail, and an explicit stroke count.

The first design (train_codebook.py, 2026-09-19) used all 5,000 codes once it
could start (4,536 in use, perplexity 3,342) and still failed the collapse bar:
decoded arc-length std 0.06-0.14 of the truth, 3-6 strokes decoded against 10.4.
Two causes, two changes:

1. Under-counting. Which of the 32 slots matches which true stroke changes from
   step to step, so every slot's "exists" probability averages below one half
   and a 0.0 threshold drops strokes. Here a count head predicts n from the code
   and the n slots with the highest exist logit are kept.
2. Averaging inside a code. One code stands for ~30 diverse clusters, and a
   deterministic decoder returns their mean. Residual FSQ adds detail: stage 1
   is the WORD (5,000 values, the vocabulary this track is after), stages 2-3
   quantise what stage 1 left over, each on a finer grid.

Both are measured: decoding from all three stages (what the representation can
hold) and from the word alone (the per-word prototype a generator would propose).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import LEVELS, P, SLOTS, load, match_loss, stroke_feats  # noqa: E402

STAGES = 3


class ResidualFSQ(nn.Module):
    def __init__(self, levels=LEVELS, stages=STAGES, eps=1e-3):
        super().__init__()
        L = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("L", L)
        self.register_buffer("half_l", (L - 1) * (1 - eps) / 2)
        self.register_buffer("offset", torch.where(L % 2 == 0, 0.5, 0.0))
        self.register_buffer("shift", torch.atanh(self.offset / self.half_l))
        self.register_buffer("half_w", torch.floor(L / 2))
        self.register_buffer("basis", torch.cumprod(torch.tensor((1,) + tuple(levels[:-1]), dtype=torch.float32), 0))
        self.stages = stages

    def rnd(self, b):
        return b + (torch.round(b) - b).detach()

    def forward(self, z, keep_stages=None):
        """-> (q, word_index). q sums the kept stages; stage k lives on a grid
        (2*half_w)^k times finer than stage 1."""
        b = (torch.tanh(z + self.shift) * self.half_l - self.offset) / self.half_w     # in [-1, 1]
        q_total = torch.zeros_like(b); r = b; scale = torch.ones_like(self.half_w)
        word = None
        for k in range(self.stages):
            qk = self.rnd(r * scale * self.half_w) / self.half_w
            if k == 0:
                word = ((torch.round(qk * self.half_w) + self.half_w) * self.basis).sum(-1).long()
            if keep_stages is None or k < keep_stages:
                q_total = q_total + qk / scale
            r = r - qk / scale
            scale = scale * 2 * self.half_w
        return q_total, word


class Codebook2(nn.Module):
    def __init__(self, d=256, layers=4, heads=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(35, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.pre_q = nn.LayerNorm(d)
        self.to_z = nn.Linear(d, len(LEVELS)); nn.init.normal_(self.to_z.weight, std=2.0 / np.sqrt(d))
        self.rfsq = ResidualFSQ()
        self.from_z = nn.Sequential(nn.Linear(len(LEVELS), d), nn.GELU(), nn.Linear(d, d))
        self.slots = nn.Parameter(torch.randn(1, SLOTS, d) * 0.02)
        self.dec = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1 + P * 2 + 1))
        self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, SLOTS + 1))

    def encode(self, pts, width, mask, keep_stages=None):
        x = self.inp(stroke_feats(pts, width))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.enc(x, src_key_padding_mask=m)
        return self.rfsq(self.to_z(self.pre_q(h[:, 0])), keep_stages)

    def decode(self, q):
        c = self.from_z(q)
        o = self.head(self.dec(self.slots + c[:, None, :]))
        return o[..., 0], o[..., 1:1 + P * 2].view(len(q), SLOTS, P, 2) * 28.0, o[..., -1] * 10.0, self.count(c)


def topn_mask(ex, cnt_logits):
    n = cnt_logits.argmax(-1).clamp(min=1)
    order = ex.argsort(-1, descending=True).argsort(-1)
    return order < n[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/cluster_set_20260919")
    p.add_argument("--out", default="results/codebook2_20260919")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stages", type=int, default=STAGES)
    p.add_argument("--count-weight", type=float, default=0.2)
    a = p.parse_args()
    torch.manual_seed(20260919)
    dev = torch.device("cuda")
    tp, tw, tm, _ = load(a.data, "train", dev)
    vp, vw, vm, _ = load(a.data, "test", dev)
    if a.limit:
        tp, tw, tm = tp[:a.limit], tw[:a.limit], tm[:a.limit]; vp, vw, vm = vp[:4096], vw[:4096], vm[:4096]
    model = Codebook2().to(dev)
    model.rfsq.stages = a.stages
    print(f"train {len(tp)}  test {len(vp)}  params {sum(x.numel() for x in model.parameters())/1e6:.1f}M  stages {a.stages} count_w {a.count_weight}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * (len(tp) // a.batch)), pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = torch.randperm(len(tp), device=dev); tot = 0.0; n = 0
        model.train()
        for i in range(0, len(tp) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            pts, w, m = tp[idx].float(), tw[idx].float(), tm[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, _word = model.encode(pts, w, m)
                ex, pp, pw, cl = model.decode(q)
            loss, pl = match_loss(ex.float(), pp.float(), pw.float(), pts, w, m)
            loss = loss + a.count_weight * nn.functional.cross_entropy(cl.float(), m.sum(1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(pl); n += 1
        model.eval(); res = {}
        with torch.no_grad():
            for tag, ks in (("full", None), ("word", 1)):
                arc_p, arc_t, n_p, n_t, words, vl, vn = [], [], [], [], [], 0.0, 0
                for i in range(0, len(vp), 1024):
                    pts, w, m = vp[i:i + 1024].float(), vw[i:i + 1024].float(), vm[i:i + 1024]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        q, word = model.encode(pts, w, m, ks)
                        ex, pp, pw, cl = model.decode(q)
                    _l, pl = match_loss(ex.float(), pp.float(), pw.float(), pts, w, m)
                    vl += float(pl); vn += 1
                    keep = topn_mask(ex.float(), cl.float())
                    pp = pp.float()
                    arc_p.append((pp[:, :, 1:] - pp[:, :, :-1]).norm(dim=-1).sum(-1)[keep].cpu())
                    arc_t.append((pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1)[m].cpu())
                    n_p.append(keep.sum(1).float().cpu()); n_t.append(m.sum(1).float().cpu()); words.append(word.cpu())
                ap, at = torch.cat(arc_p), torch.cat(arc_t)
                wc = torch.bincount(torch.cat(words), minlength=5000).float(); pr = wc / wc.sum()
                res[tag] = {"pts": vl / vn, "arc_std_ratio": float(ap.std() / at.std()),
                            "n_pred": float(torch.cat(n_p).mean()), "n_true": float(torch.cat(n_t).mean()),
                            "words_used": int((wc > 0).sum()), "perplexity": float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))}
        row = {"epoch": ep, "train_pts": tot / max(n, 1), **{f"{t}_{k}": v for t in res for k, v in res[t].items()}, "sec": time.time() - t0}
        hist.append(row)
        f, wd = res["full"], res["word"]
        print(f"ep {ep:>3} 学習 {row['train_pts']:.3f} | 3段: 評価 {f['pts']:.3f} 弧長比 {f['arc_std_ratio']:.2f} 本数 {f['n_pred']:.1f}/{f['n_true']:.1f} "
              f"| 単語のみ: 評価 {wd['pts']:.3f} 弧長比 {wd['arc_std_ratio']:.2f} 本数 {wd['n_pred']:.1f} | 単語 {f['words_used']} ppl {f['perplexity']:.0f} {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "codebook2.pt")


if __name__ == "__main__":
    main()
