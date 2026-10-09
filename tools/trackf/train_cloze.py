#!/usr/bin/env python
"""Set-based cloze model: predict one held-out cluster of a panel from the rest.

Pre-registered 2026-09-20 (doc/work_log.md):
  1. word bits >= 1.0 bits better than unigram (<= 6.35), and below unigram
  2. joint pos (y+x) bits beat independent position marginals
  3. montage of truth vs top-4 predictions (rendered in the panel)
Run with --render after training to draw results/cloze_20260920/cloze_montage.png.
"""
import argparse, csv, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VOCAB = 558
N_WORDS, OFF_Y, OFF_X, OFF_S = 500, 512, 528, 544
BOS, EOS = 556, 557
MAXS = 170          # max clusters per panel (padded)


class ClozeNet(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, dropout=0.1, n_types=0, n_tagf=0):
        super().__init__()
        self.wemb = nn.Embedding(N_WORDS, d)
        self.pY = nn.Embedding(16, d)
        self.pX = nn.Embedding(16, d)
        self.pS = nn.Embedding(12, d)
        self.ft = nn.Embedding(4, d)                    # field-type bias
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.temb = nn.Embedding(n_types, d) if n_types else None
        self.tagin = nn.Linear(n_tagf, d) if n_tagf else None
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True), layers)
        self.pre = nn.LayerNorm(d)
        self.hw = nn.Linear(d, N_WORDS)
        self.hy = nn.Linear(d, 16)
        self.hx = nn.Linear(d, 16)
        self.hs = nn.Linear(d, 12)

    def forward(self, w, py, px, sc, mask, ctype=None, ctag=None):
        # w,py,px,sc: (B,S) token ids (-1 padded); mask: (B,S) True=real cluster
        B, S = w.shape
        x = (self.wemb(w.clamp(min=0)) + self.pY(py.clamp(min=0)) + self.pX(px.clamp(min=0))
             + self.pS(sc.clamp(min=0)) + self.ft.weight.sum(0))
        c = self.cls.expand(B, -1, -1)
        if self.temb is not None:
            c = c + self.temb(ctype).unsqueeze(1)
        if self.tagin is not None:
            c = c + self.tagin(ctag).unsqueeze(1)
        x = torch.cat([c, x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.tr(x, src_key_padding_mask=pad)[:, 0]
        h = self.pre(h)
        return self.hw(h), self.hy(h), self.hx(h), self.hs(h)


def unpack(seq_row):
    """corpus row -> (words, py, px, sc) int arrays of the panel's clusters."""
    t = seq_row[seq_row >= 0]
    cl = t[1:-1].reshape(-1, 4)
    return cl[:, 0], cl[:, 1] - OFF_Y, cl[:, 2] - OFF_X, cl[:, 3] - OFF_S


def build_batch(rows, rng, dev, ctype=None, ctag=None):
    B = len(rows)
    W = np.zeros((B, MAXS), np.int64); PY = np.zeros((B, MAXS), np.int64)
    PX = np.zeros((B, MAXS), np.int64); SC = np.zeros((B, MAXS), np.int64)
    MK = np.zeros((B, MAXS), bool)
    TW = np.zeros(B, np.int64); TY = np.zeros(B, np.int64)
    TX = np.zeros(B, np.int64); TS = np.zeros(B, np.int64); TK = np.zeros(B, np.int64)
    for b, row in enumerate(rows):
        w, py, px, sc = unpack(row)
        n = len(w)
        k = int(rng.integers(n))
        sel = np.arange(n) != k
        m = int(sel.sum())
        W[b, :m] = w[sel]; PY[b, :m] = py[sel]; PX[b, :m] = px[sel]; SC[b, :m] = sc[sel]
        MK[b, :m] = True
        TW[b], TY[b], TX[b], TS[b], TK[b] = w[k], py[k], px[k], sc[k], k
    t = lambda a: torch.from_numpy(a).to(dev)
    out = [t(W), t(PY), t(PX), t(SC), t(MK), t(TW), t(TY), t(TX), t(TS), t(TK)]
    if ctype is not None:
        out.append(t(ctype))
    if ctag is not None:
        out.append(t(ctag).float())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/grammar_corpus_20260920/corpus.npz")
    p.add_argument("--out", default="results/cloze_20260920")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--labels", default="", help="panel type labels .npy (corpus row order) -> type conditioning")
    p.add_argument("--tagfeats", default="", help="tag multi-hot .npy (corpus row order) -> tag conditioning")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(20260920)
    z = np.load(a.data)
    seq, split = z["seq"], z["split"]
    tr, te = seq[split], seq[~split]
    ctype_tr = ctype_te = ctag_tr = ctag_te = None
    n_types = n_tagf = 0
    if a.labels:
        lab = np.load(a.labels)
        ctype = lab.astype(np.int64)
        n_types = int(ctype.max()) + 1
        ctype_tr, ctype_te = ctype[split], ctype[~split]
        print(f"type cond: {n_types} types", flush=True)
    if a.tagfeats:
        tg = np.load(a.tagfeats).astype(np.float32)
        n_tagf = tg.shape[1]
        ctag_tr, ctag_te = tg[split], tg[~split]
        print(f"tag cond: {n_tagf} features", flush=True)
    if a.limit:
        tr, te = tr[:a.limit], te[:512]
        if ctype_tr is not None:
            ctype_tr, ctype_te = ctype_tr[:a.limit], ctype_te[:512]
        if ctag_tr is not None:
            ctag_tr, ctag_te = ctag_tr[:a.limit], ctag_te[:512]
    print(f"train {tr.shape}  test {te.shape}", flush=True)

    # baselines from train split
    wc = np.bincount(tr[:, 1::4][tr[:, 1::4] >= 0], minlength=N_WORDS).astype(np.float64)
    uni = wc / wc.sum()
    H_uni = float(-(uni[uni > 0] * np.log2(uni[uni > 0])).sum())
    yc = np.bincount(tr[:, 2::4][tr[:, 2::4] >= 0] - OFF_Y, minlength=16).astype(np.float64)
    xc = np.bincount(tr[:, 3::4][tr[:, 3::4] >= 0] - OFF_X, minlength=16).astype(np.float64)
    H_pos = float(-(yc / yc.sum() * np.log2(yc / yc.sum())).sum() - (xc / xc.sum() * np.log2(xc / xc.sum())).sum())

    model = ClozeNet(n_types=n_types, n_tagf=n_tagf).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * (len(tr) // a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist, best = [], {"word_bits": 1e9}
    tr_rng = np.random.default_rng(20260920)
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = np.random.permutation(len(tr)); tot = 0.0; nb = 0
        model.train()
        for i in range(0, len(tr) - a.batch + 1, a.batch):
            sl = perm[i:i + a.batch]
            bt = build_batch(tr[sl], tr_rng, dev,
                             ctype_tr[sl] if ctype_tr is not None else None,
                             ctag_tr[sl] if ctag_tr is not None else None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lw, ly, lx, ls = model(*bt[:5], ctype=bt[10] if ctype_tr is not None else None,
                                       ctag=bt[10 + (1 if ctype_tr is not None else 0)] if ctag_tr is not None else None)
            l = (nn.functional.cross_entropy(lw.float(), bt[5])
                 + nn.functional.cross_entropy(ly.float(), bt[6])
                 + nn.functional.cross_entropy(lx.float(), bt[7])
                 + nn.functional.cross_entropy(ls.float(), bt[8]))
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(l); nb += 1
        model.eval(); n = 0; bw = by = bx = 0.0
        with torch.no_grad():
            for i in range(0, len(te), 64):
                bt = build_batch(te[i:i + 64], np.random.default_rng(777), dev,
                                 ctype_te[i:i + 64] if ctype_te is not None else None,
                                 ctag_te[i:i + 64] if ctag_te is not None else None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lw, ly, lx, ls = model(*bt[:5], ctype=bt[10] if ctype_te is not None else None,
                                           ctag=bt[10 + (1 if ctype_te is not None else 0)] if ctag_te is not None else None)
                bw += float(nn.functional.cross_entropy(lw.float(), bt[5], reduction="sum"))
                by += float(nn.functional.cross_entropy(ly.float(), bt[6], reduction="sum"))
                bx += float(nn.functional.cross_entropy(lx.float(), bt[7], reduction="sum"))
                n += len(bt[5])
        row = {"epoch": ep, "train": tot / max(nb, 1), "word_bits": bw / n / math.log(2),
               "pos_bits": (by + bx) / n / math.log(2), "sec": time.time() - t0}
        hist.append(row)
        print(f"ep {ep:>3} 学習 {row['train']:.3f} | test word {row['word_bits']:.3f} bits "
              f"(unigram {H_uni:.3f}, 門 {H_uni - 1.0:.3f}) | pos {row['pos_bits']:.3f} "
              f"(周辺分布 {H_pos:.3f}) | {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "cloze.pt")
        if row["word_bits"] < best["word_bits"]:
            best = dict(word_bits=row["word_bits"], epoch=ep, pos_bits=row["pos_bits"])
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "cloze_best.pt")
    res = {"word_bits": best["word_bits"], "best_epoch": best["epoch"], "pos_bits": best["pos_bits"],
           "unigram_bits": H_uni, "pos_marginal_bits": H_pos,
           "pass": bool(best["word_bits"] <= H_uni - 1.0 and best["pos_bits"] < H_pos)}
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    if a.render:
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "render_cloze.py"),
                        "--ckpt", str(out / "cloze_best.pt"), "--data", a.data, "--out", str(out)])


if __name__ == "__main__":
    main()
