#!/usr/bin/env python
"""Given a drawing with one stroke removed, draw the missing stroke.

This is the track's real question, in the form the user set on 2026-09-18:
line art has no reading order, so the task is not autoregression over an
invented order but MASKED PREDICTION -- hide a stroke, predict it back. The
true stroke exists, so nothing is judged against a synthetic negative.

Input: every other stroke of the same panel, as a set (no order), encoded the
way `train_stroke_transformer.tokenize` encodes them (shape relative to its own
centroid, multi-scale sinusoidal position, length, width, fill share).

Query: the model must be told WHERE to draw, or the task is unidentifiable. The
query carries only the centre of the 64px cell containing the missing stroke's
midpoint -- so the answer is known to +-32px, and how much that alone gives is
measured by the `--context none` baseline, not assumed.

Output: the stroke's 16 points as offsets from the cell centre, plus log arc
length, mean width and fill share.

Loss is direction-symmetric (min over the target and its reverse): a stroke
lifted from a raster has no drawing direction, and penalising the model for
`order_path`'s arbitrary starting end would be scoring our own tool.

Metric: symmetric chamfer distance in native pixels, median over held-out
panels, plus the share within 3px ("recovered") and 8px ("right idea").
Pre-registered (plan 2026-09-18): the model must beat BOTH the query-only
baseline and the gap-closing baseline by 30% on median chamfer.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stroke_transformer import Net, WAVELENGTHS, auc  # noqa: E402

POINTS = 16
CELL = 64.0


def fourier(xy):
    ph = 2.0 * np.pi * xy[..., None] / WAVELENGTHS[None, None, :]
    return np.concatenate([np.sin(ph), np.cos(ph)], -1).reshape(*xy.shape[:-1], -1)


def to16(poly, k):
    """Variable-K polyline (NaN padded) -> exactly POINTS points."""
    p = poly[:k]
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if len(p) < 2 or d[-1] <= 0:
        return np.repeat(p[:1], POINTS, 0)
    t = np.linspace(0, d[-1], POINTS)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1).astype(np.float32)


def encode(pts16, meta):
    """(N,16,2) + (N,5) -> (N, F) context features."""
    c = pts16.mean(1)
    rel = (pts16 - c[:, None, :]) / 64.0
    f = fourier(c)
    ln = np.log1p(meta[:, 3:4]) / 5.0
    wd = meta[:, 1:2] / 10.0
    fl = meta[:, 2:3]
    return np.concatenate([rel.reshape(len(pts16), -1), f, c / 1000.0, ln, wd, fl], 1).astype(np.float32)


def query_feat(cell_centre, f_in):
    q = np.zeros((1, f_in), np.float32)
    ff = fourier(cell_centre[None, :])
    off = POINTS * 2
    q[0, off:off + ff.shape[1]] = ff[0]
    q[0, off + ff.shape[1]:off + ff.shape[1] + 2] = cell_centre / 1000.0
    return q


class Packed(torch.utils.data.Dataset):
    """Same samples as `Panels`, read from the packed memmap.

    The npz path ran at 636s/epoch with the GPU idle; this one slices."""

    def __init__(self, pack, rows, max_strokes, seed, eval_mode=False):
        self.arr = np.load(Path(pack) / "strokes.npy", mmap_mode="r")
        self.rows, self.max_strokes, self.seed, self.eval_mode = rows, max_strokes, seed, eval_mode
        self.f_in = None

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        r = self.rows[i]
        s, n = int(r["start"]), int(r["n"])
        block = np.asarray(self.arr[s:s + n])
        return block[:, :POINTS * 2].reshape(n, POINTS, 2), block[:, POINTS * 2:]

    __getitem__ = None  # set below


class Panels(torch.utils.data.Dataset):
    def __init__(self, root, rows, max_strokes, seed, per_panel=1, eval_mode=False):
        self.root, self.rows = Path(root), rows
        self.max_strokes, self.seed, self.per_panel, self.eval_mode = max_strokes, seed, per_panel, eval_mode

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        r = self.rows[i]
        d = np.load(self.root / r["source"] / (r["name"] + ".npz"))
        poly, kpts, meta = d["poly"], d["kpts"], d["meta"]
        pts = np.stack([to16(poly[j], int(kpts[j])) for j in range(len(poly))]).astype(np.float32)
        return pts, meta.astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed + i) if self.eval_mode else None)
        pts, meta = self.load(i)
        if len(pts) > self.max_strokes:
            k = rng.choice(len(pts), self.max_strokes, replace=False)
            pts, meta = pts[k], meta[k]
        t = int(rng.integers(len(pts)))
        target, tmeta = pts[t], meta[t]
        ctx = np.delete(pts, t, 0); cmeta = np.delete(meta, t, 0)
        mid = target.mean(0)
        cell = (np.floor(mid / CELL) + 0.5) * CELL
        X = encode(ctx, cmeta) if len(ctx) else np.zeros((0, self.f_in), np.float32)
        return X, query_feat(cell.astype(np.float32), X.shape[1] if len(X) else self.f_in), \
            (target - cell).astype(np.float32) / 64.0, \
            np.array([np.log1p(tmeta[3]) / 5.0, tmeta[1] / 10.0, tmeta[2]], np.float32), \
            cell.astype(np.float32), target


Packed.__getitem__ = Panels.__getitem__


def collate(batch):
    f_in = batch[0][1].shape[1]
    n = max(1, max(len(b[0]) for b in batch))
    X = np.zeros((len(batch), n + 1, f_in), np.float32)
    M = np.ones((len(batch), n + 1), bool)
    for i, (ctx, q, *_rest) in enumerate(batch):
        X[i, 0] = q[0]; M[i, 0] = False
        if len(ctx):
            X[i, 1:1 + len(ctx)] = ctx; M[i, 1:1 + len(ctx)] = False
    G = np.stack([b[2] for b in batch]); A = np.stack([b[3] for b in batch])
    C = np.stack([b[4] for b in batch]); T = np.stack([b[5] for b in batch])
    return (torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(G),
            torch.from_numpy(A), torch.from_numpy(C), torch.from_numpy(T))


class Infill(nn.Module):
    def __init__(self, f_in, d=256, layers=6, heads=8):
        super().__init__()
        self.body = Net(f_in, d, layers, heads)
        self.body.head = nn.Identity()
        self.geo = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, POINTS * 2))
        self.attr = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

    def forward(self, x, mask):
        h = self.body.enc(self.body.inp(x), src_key_padding_mask=mask)
        q = h[:, 0]
        return self.geo(q).view(-1, POINTS, 2), self.attr(q)


def sym_huber(pred, target, beta=1.0):
    a = nn.functional.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean((1, 2))
    b = nn.functional.smooth_l1_loss(pred, torch.flip(target, [1]), beta=beta, reduction="none").mean((1, 2))
    return torch.minimum(a, b).mean()


def chamfer(a, b):
    """symmetric chamfer between two (k,2) point sets, in pixels"""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="results/panel_tokens_v2_20260918")
    p.add_argument("--pack", default="", help="packed memmap dir (pack_tokens.py); much faster")
    p.add_argument("--out", default="results/infill_20260918")
    p.add_argument("--context", choices=["full", "none"], default="full")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-strokes", type=int, default=600)
    p.add_argument("--min-strokes", type=int, default=40)
    p.add_argument("--eval-panels", type=int, default=300)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--loader-workers", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.pack:
        rows = [r for r in csv.DictReader(open(Path(a.pack) / "panels.csv"))
                if int(r["n"]) >= a.min_strokes]
    else:
        rows = [r for r in csv.DictReader(open(Path(a.tokens) / "index.csv"))
                if int(r["strokes"]) >= a.min_strokes]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(a.seed); rng.shuffle(groups)
    test_g = set(groups[: int(len(groups) * a.test_frac)])
    tr = [r for r in rows if r["group"] not in test_g]
    te = [r for r in rows if r["group"] in test_g][: a.eval_panels]
    if a.smoke:
        tr, te, a.epochs = tr[:40], te[:8], 2
    print(f"panels train {len(tr)}  eval {len(te)}  groups {len(groups)}", flush=True)
    f_in = POINTS * 2 + len(WAVELENGTHS) * 4 + 2 + 3
    if a.pack:
        ds_tr = Packed(a.pack, tr, a.max_strokes, a.seed)
        ds_te = Packed(a.pack, te, a.max_strokes, a.seed + 1, eval_mode=True)
    else:
        ds_tr = Panels(a.tokens, tr, a.max_strokes, a.seed)
        ds_te = Panels(a.tokens, te, a.max_strokes, a.seed + 1, eval_mode=True)
    ds_tr.f_in = f_in; ds_te.f_in = f_in
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.loader_workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=True)
    dlte = torch.utils.data.DataLoader(ds_te, batch_size=a.batch, shuffle=False, num_workers=max(1, a.loader_workers - 1),
                                       collate_fn=collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Infill(f_in, a.d, a.layers).to(dev)
    print(f"params {sum(x.numel() for x in model.parameters())/1e6:.1f}M  f_in {f_in}  context {a.context}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * len(dl)), pct_start=0.1)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    hist, best = [], 1e9
    for ep in range(1, a.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        model.train()
        for X, M, G, A, C, T in dl:
            X, M, G, A = X.to(dev), M.to(dev), G.to(dev), A.to(dev)
            if a.context == "none":
                M = M.clone(); M[:, 1:] = True
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                g, at = model(X, M)
                loss = sym_huber(g.float(), G) + 0.1 * nn.functional.smooth_l1_loss(at.float(), A)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(loss); n += 1
        model.eval(); ch = []
        with torch.no_grad():
            for X, M, G, A, C, T in dlte:
                X, M = X.to(dev), M.to(dev)
                if a.context == "none":
                    M = M.clone(); M[:, 1:] = True
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    g, _ = model(X, M)
                pred = g.float().cpu().numpy() * 64.0 + C.numpy()[:, None, :]
                for pr, tg in zip(pred, T.numpy()):
                    ch.append(chamfer(pr, tg))
        ch = np.array(ch)
        row = {"epoch": ep, "loss": tot / max(n, 1), "chamfer_med": float(np.median(ch)),
               "within3": float((ch <= 3).mean()), "within8": float((ch <= 8).mean()),
               "sec": time.time() - t0}
        hist.append(row)
        print(f"ep {ep:>3} loss {row['loss']:.4f} chamfer中央値 {row['chamfer_med']:.1f}px "
              f"3px以内 {row['within3']:.1%} 8px以内 {row['within8']:.1%} {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(outdir / f"history_{a.context}.json", "w"), indent=1)
        if row["chamfer_med"] < best:
            best = row["chamfer_med"]
            torch.save({"model": model.state_dict(), "args": vars(a), "eval": row},
                       outdir / f"best_{a.context}.pt")
    print(f"best chamfer中央値 {best:.1f}px  (context={a.context})")


if __name__ == "__main__":
    main()

