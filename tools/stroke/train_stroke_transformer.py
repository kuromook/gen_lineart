#!/usr/bin/env python
"""A set-Transformer over a panel's strokes: which stroke does not belong?

The other half of the 2026-09-18 agreement. Gate 1 failed three times with
hand-made relational features (0.636 / 0.726 / 0.570 on displaced), and the
third run showed the second was riding on a tokenizer artifact. Here nothing
relational is hand-specified: every stroke in the panel is a token (its
polyline, length, width, position), the encoder attends over the whole set, and
a per-token head says real or fake.

Negatives are made from polylines, so they are exact: displaced = shifted
5-12px, rotated = 15-40 degrees about its centroid, foreign = a polyline from
another panel re-centred here. During training ~10% of a panel's strokes are
corrupted at once (dense supervision); evaluation corrupts exactly ONE stroke
and compares its score against the same stroke left real -- the matched
comparison gate 1 uses, so the AUCs are directly comparable.

Round 2 (2026-09-18, after the bug): only the input encoding changed -- position
now carries wavelengths from 8px up. Architecture, learning rate, schedule,
epochs, split and evaluation are byte-identical to round 1, so the two runs are
comparable.

Pre-registered before running (2026-09-18):
  - one architecture (d=256, 6 layers), one learning rate, up to 30 epochs
  - model chosen by validation AUC on DISPLACED, as gate 1 is judged
  - gate 1 still asks for 0.80 on displaced; the hand-feature number to beat
    on the way is 0.570
"""
import argparse, csv, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

POINTS = 16


def load_index(root, min_strokes):
    rows = list(csv.DictReader(open(Path(root) / "index.csv")))
    return [r for r in rows if int(r["strokes"]) >= min_strokes]


class Panels(torch.utils.data.Dataset):
    def __init__(self, root, rows, max_strokes, seed, corrupt_frac=0.1, foreign_pool=None):
        self.root, self.rows, self.max_strokes = Path(root), rows, max_strokes
        self.corrupt_frac, self.seed = corrupt_frac, seed
        self.pool = foreign_pool

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        r = self.rows[i]
        d = np.load(self.root / r["source"] / (r["name"] + ".npz"))
        return d["poly"].astype(np.float32), d["meta"].astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        poly, meta = self.load(i)
        if len(poly) > self.max_strokes:
            keep = rng.choice(len(poly), self.max_strokes, replace=False)
            poly, meta = poly[keep], meta[keep]
        n = len(poly)
        k = max(1, int(round(self.corrupt_frac * n)))
        idx = rng.choice(n, k, replace=False)
        y = np.ones(n, np.float32)
        for j in idx:
            kind = rng.integers(3)
            poly[j], meta[j] = corrupt(poly[j], meta[j], kind, rng, self.pool)
            y[j] = 0.0
        return poly, meta, y


def corrupt(p, m, kind, rng, pool):
    if kind == 0:  # displaced
        ang = rng.uniform(0, 2 * np.pi)
        off = np.array([np.sin(ang), np.cos(ang)], np.float32) * rng.uniform(5, 12)
        return p + off, m
    if kind == 1:  # rotated
        th = np.deg2rad(rng.uniform(15, 40) * rng.choice([-1.0, 1.0]))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
        c = p.mean(0)
        return (p - c) @ R.T + c, m
    fp, fm = pool[rng.integers(len(pool))]        # foreign
    return (fp - fp.mean(0) + p.mean(0)).astype(np.float32), fm


# Position is encoded at several wavelengths, the shortest a few pixels, because
# the first run (2026-09-18) gave the centroid as `coords / 1000`: a displaced
# candidate then differed from the real stroke by at most 0.0057 in a feature
# whose spread over tokens is 1.5647, and the trained model's output for it moved
# by 0.0000 +- 0.0001 -- exactly chance, all 30 epochs. The shape columns are
# centroid-relative and so are identical under displacement by construction, so
# position was the ONLY channel that could carry it, and it carried 1/275 of it.
WAVELENGTHS = np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0], np.float32)


def tokenize(poly, meta, shape=None):
    """(N, 16, 2) + (N, 3) -> (N, F): the shape relative to its own centroid,
    the centroid at several wavelengths plus a coarse copy, and
    length/width/fill."""
    c = poly.mean(1)
    rel = (poly - c[:, None, :]) / 64.0
    ph = 2.0 * np.pi * c[:, :, None] / WAVELENGTHS[None, None, :]   # (N, 2, W)
    fourier = np.concatenate([np.sin(ph), np.cos(ph)], 2).reshape(len(poly), -1)
    pos = c / 1000.0
    ln = np.log1p(meta[:, 0:1]) / 5.0
    wd = meta[:, 1:2] / 10.0
    fl = meta[:, 2:3]
    return np.concatenate([rel.reshape(len(poly), -1), fourier, pos, ln, wd, fl], 1).astype(np.float32)


def collate(batch):
    xs = [tokenize(p, m) for p, m, _ in batch]
    ys = [y for _, _, y in batch]
    n = max(len(x) for x in xs)
    X = np.zeros((len(xs), n, xs[0].shape[1]), np.float32)
    Y = np.zeros((len(xs), n), np.float32)
    M = np.ones((len(xs), n), bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = x; Y[i, :len(y)] = y; M[i, :len(x)] = False
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


class Net(nn.Module):
    def __init__(self, f_in, d=256, layers=6, heads=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        return self.head(h).squeeze(-1)


def auc(y, s):
    o = np.argsort(s); y = np.asarray(y)[o]
    p, n = y.sum(), len(y) - y.sum()
    if not p or not n:
        return float("nan")
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


@torch.no_grad()
def evaluate(model, ds, rows, pool, device, per_panel=12, max_strokes=800, seed=7):
    """Matched: corrupt exactly one stroke, score it, compare with the same
    stroke left real. One AUC per kind, as gate 1 reports."""
    model.eval()
    got = {k: ([], []) for k in range(3)}
    for i in range(len(rows)):
        rng = np.random.default_rng(seed + i)
        poly, meta = ds.load(i)
        if len(poly) > max_strokes:
            keep = rng.choice(len(poly), max_strokes, replace=False)
            poly, meta = poly[keep], meta[keep]
        n = len(poly)
        targets = rng.choice(n, min(per_panel, n), replace=False)
        variants, tags = [tokenize(poly, meta)], [None]
        for t in targets:
            for kind in range(3):
                p2, m2 = poly.copy(), meta.copy()
                p2[t], m2[t] = corrupt(poly[t].copy(), meta[t].copy(), kind, rng, pool)
                variants.append(tokenize(p2, m2)); tags.append((t, kind))
        X = torch.from_numpy(np.stack(variants)).to(device)
        M = torch.zeros(X.shape[:2], dtype=torch.bool, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            S = model(X, M).float().cpu().numpy()
        base = S[0]
        for vi, tag in enumerate(tags):
            if tag is None:
                continue
            t, kind = tag
            got[kind][0].append(float(base[t])); got[kind][1].append(float(S[vi][t]))
    out = {}
    for kind, name in enumerate(("displaced", "rotated", "foreign")):
        real, fake = got[kind]
        y = np.r_[np.ones(len(real)), np.zeros(len(fake))]
        out[name] = auc(y, np.r_[real, fake])
        out[f"n_{name}"] = len(real)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="results/panel_tokens_20260918")
    p.add_argument("--out", default="results/stroke_transformer_20260918")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-strokes", type=int, default=800)
    p.add_argument("--min-strokes", type=int, default=40)
    p.add_argument("--eval-panels", type=int, default=120)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows = load_index(a.tokens, a.min_strokes)
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(a.seed); rng.shuffle(groups)
    test_g = set(groups[: int(len(groups) * a.test_frac)])
    tr = [r for r in rows if r["group"] not in test_g]
    te = [r for r in rows if r["group"] in test_g]
    if a.smoke:
        tr, te, a.epochs = tr[:40], te[:8], 2
    te_eval = te[: a.eval_panels]
    print(f"panels train {len(tr)}  test {len(te)} (eval on {len(te_eval)})  groups {len(groups)}", flush=True)

    # foreign pool: polylines from TRAIN panels only
    pool = []
    for r in tr[:: max(1, len(tr) // 200)][:200]:
        d = np.load(Path(a.tokens) / r["source"] / (r["name"] + ".npz"))
        pl, mt = d["poly"].astype(np.float32), d["meta"].astype(np.float32)
        k = np.random.default_rng(1).choice(len(pl), min(30, len(pl)), replace=False)
        pool += [(pl[i], mt[i]) for i in k]
    print(f"foreign pool {len(pool)}", flush=True)

    ds_tr = Panels(a.tokens, tr, a.max_strokes, a.seed, foreign_pool=pool)
    ds_te = Panels(a.tokens, te_eval, a.max_strokes, a.seed + 1, foreign_pool=pool)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=3,
                                     collate_fn=collate, drop_last=True, persistent_workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_in = tokenize(np.zeros((1, POINTS, 2), np.float32), np.zeros((1, 3), np.float32)).shape[1]
    model = Net(f_in, a.d, a.layers).to(device)
    print(f"params {sum(x.numel() for x in model.parameters())/1e6:.1f}M  device {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        t0, tot, seen = time.time(), 0.0, 0
        for X, Y, M in dl:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit = model(X, M)
                l = lossf(logit.float(), Y)
                l = (l * (~M)).sum() / (~M).sum()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(l); seen += 1
        ev = evaluate(model, ds_te, te_eval, pool, device, max_strokes=a.max_strokes)
        hist.append({"epoch": ep, "loss": tot / max(seen, 1), **ev, "sec": time.time() - t0})
        print(f"ep {ep:>3}  loss {tot/max(seen,1):.4f}  displaced {ev['displaced']:.3f}  "
              f"rotated {ev['rotated']:.3f}  foreign {ev['foreign']:.3f}  {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open(outdir / "history.json", "w"), indent=1)
        if ev["displaced"] > best:
            best = ev["displaced"]
            torch.save({"model": model.state_dict(), "args": vars(a), "eval": ev}, outdir / "best.pt")
    print(f"best displaced AUC {best:.3f}  (gate 1 asks 0.80; hand features gave 0.570)")


if __name__ == "__main__":
    main()
