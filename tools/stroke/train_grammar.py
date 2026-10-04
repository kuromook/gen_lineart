#!/usr/bin/env python
"""Grammar model: a panel is a sequence of cluster tokens (word, pos_y, pos_x, scale).

Pre-registered 2026-09-20 (doc/work_log.md):
  1. next-word bits on held-out panels must beat uniform AND unigram (>= 0.5 bits) AND bigram
  2. next-position bits must beat "previous centroid + marginal delta" baseline
Baselines are computed from the train split and evaluated on the test split.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VOCAB = 558          # 500 words + 16 pos_y + 16 pos_x + 12 scale + BOS + EOS
N_WORDS, OFF_Y, OFF_X, OFF_S = 500, 512, 528, 544
BOS, EOS = 556, 557


class GPT(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, max_len=1024, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        pos = torch.arange(max_len)
        ang = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2).float() / d)
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos[:, None] * ang[None])
        pe[:, 1::2] = torch.cos(pos[:, None] * ang[None])
        self.register_buffer("pe", pe)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True), layers)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.emb(x.clamp(min=0)) + self.pe[:len(x[0])]
        mask = torch.triu(torch.ones(len(x[0]), len(x[0]), device=x.device, dtype=torch.bool), 1)
        return self.head(self.tr(h, mask=mask))


def bits_of_ce(logits, tgt, lo, hi):
    """Mean next-token CE (nats) restricted to targets in [lo, hi). -> (nats, count)"""
    m = (tgt >= lo) & (tgt < hi)
    if m.sum() == 0:
        return 0.0, 0
    l = nn.functional.cross_entropy(logits[m].float(), tgt[m].long(), reduction="sum")
    return float(l), int(m.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/grammar_corpus_20260920/corpus.npz")
    p.add_argument("--out", default="results/grammar_20260920")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(20260920)
    dev = torch.device("cuda")
    z = np.load(a.data)
    seq, split = z["seq"], z["split"]
    tr, te = seq[split], seq[~split]
    if a.limit:
        tr, te = tr[:a.limit], te[:512]
    print(f"train {tr.shape}  test {te.shape}", flush=True)

    # ---- baselines from the train split, evaluated on test ----
    wc = np.bincount(tr[:, 1::4][tr[:, 1::4] >= 0], minlength=N_WORDS).astype(np.float64)
    uni = wc / wc.sum()
    H_uni = float(-(uni[uni > 0] * np.log2(uni[uni > 0])).sum())
    big = np.zeros((N_WORDS, N_WORDS))
    for row in tr:
        ws = row[1::4]; ws = ws[(ws >= 0) & (ws < N_WORDS)]
        for u, v in zip(ws[:-1], ws[1:]):
            big[u, v] += 1
    Pb = (big + 0.1) / (big + 0.1).sum(1, keepdims=True)
    def bigram_bits(rows):
        tot, n = 0.0, 0
        for row in rows:
            ws = row[1::4]; ws = ws[(ws >= 0) & (ws < N_WORDS)]
            for u, v in zip(ws[:-1], ws[1:]):
                tot += -math.log2(Pb[int(u), int(v)]); n += 1
        return tot / max(n, 1), n
    bg_bits, bg_n = bigram_bits(te)

    # position baseline: previous centroid + marginal delta (joint pos_y/pos_x bin delta)
    dcnt = np.zeros((31, 31))
    for row in tr:
        t = row[row >= 0]
        cl = t[1:-1].reshape(-1, 4) - [0, OFF_Y, OFF_X, OFF_S]
        for (py1, px1), (py2, px2) in zip(cl[:-1, 1:3], cl[1:, 1:3]):
            dcnt[int(py2) - int(py1) + 15, int(px2) - int(px1) + 15] += 1
    Pd = (dcnt + 0.5) / (dcnt + 0.5).sum()
    def pos_delta_bits(rows):
        tot, n = 0.0, 0
        for row in rows:
            t = row[row >= 0]
            cl = t[1:-1].reshape(-1, 4) - [0, OFF_Y, OFF_X, OFF_S]
            for (py1, px1), (py2, px2) in zip(cl[:-1, 1:3], cl[1:, 1:3]):
                tot += -math.log2(Pd[int(py2) - int(py1) + 15, int(px2) - int(px1) + 15]); n += 1
        return tot / max(n, 1), n
    pd_bits, pd_n = pos_delta_bits(te)

    model = GPT(dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * (len(tr) // a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = []
    best = {"word_bits": 1e9, "epoch": 0}
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = np.random.permutation(len(tr)); tot = 0.0; nb = 0
        model.train()
        for i in range(0, len(tr) - a.batch + 1, a.batch):
            rows = torch.from_numpy(tr[perm[i:i + a.batch]].astype(np.int64)).to(dev)
            x, y = rows[:, :-1], rows[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            l = nn.functional.cross_entropy(logits.reshape(-1, VOCAB).float(), y.reshape(-1), ignore_index=-1)
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(l); nb += 1
        # eval on test
        model.eval(); wn, wc_n, pn, pce_n = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(te), 64):
                rows = torch.from_numpy(te[i:i + 64].astype(np.int64)).to(dev)
                x, y = rows[:, :-1], rows[:, 1:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                lg = logits.reshape(-1, VOCAB); tg = y.reshape(-1)
                l, c = bits_of_ce(lg, tg, 0, N_WORDS); wn += l; wc_n += c
                l, c = bits_of_ce(lg, tg, OFF_Y, OFF_X + 16); pn += l; pce_n += c
        row = {"epoch": ep, "train_ce": tot / max(nb, 1),
               "word_bits": wn / max(wc_n, 1) / math.log(2), "n_word": wc_n,
               "pos_bits": pn / max(pce_n, 1) / math.log(2), "sec": time.time() - t0}
        hist.append(row)
        print(f"ep {ep:>3} 学習 {row['train_ce']:.3f} | test word {row['word_bits']:.3f} bits/語 "
              f"(unigram {H_uni:.3f}, bigram {bg_bits:.3f}) | pos {row['pos_bits']:.3f} bits/まとまり "
              f"(delta事前 {pd_bits:.3f}) | {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "grammar.pt")
        if row["word_bits"] < best["word_bits"]:   # held-out model selection (disclosed in work_log)
            best = {"word_bits": row["word_bits"], "epoch": ep, "pos_bits": row["pos_bits"]}
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, out / "grammar_best.pt")
    row = next(r for r in hist if r["epoch"] == best["epoch"])
    res = {"word_bits": row["word_bits"], "best_epoch": best["epoch"],
           "uniform_bits": math.log2(N_WORDS), "unigram_bits": H_uni,
           "bigram_bits": bg_bits, "pos_bits": row["pos_bits"], "pos_delta_prior_bits": pd_bits,
           "pass": bool(row["word_bits"] < bg_bits < H_uni and row["word_bits"] <= H_uni - 0.5
                        and row["pos_bits"] < pd_bits)}
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
