#!/usr/bin/env python
"""Cloze test (B'): which of five real strokes is the one missing from this cluster?

Why this shape (2026-09-19). The first infill run regressed a 16-point polyline
and collapsed to the conditional mean -- one small squiggle for every input
(predicted arc std 4.5px against a true 273px). Choosing among candidates cannot
collapse that way and has an exact chance level (1/5).

The review in outbox/KIMIからの意見.md made the negatives the whole question:
  - every distractor is a REAL stroke, length-matched to the truth (arc +-20%),
    and re-centred on the true stroke's centroid, so neither length nor position
    gives it away; a variant also matches width and straightness;
  - three strata, one per question: (1) same panel, other cluster -- the primary
    one, "which group does this stroke belong to" is the grammar; (2) same work,
    other panel; (3) other work. High on (3) only would mean a style detector.

Pre-registered (doc/work_log.md 2026-09-19): on strata (1) and (2), accuracy must
beat the best non-learned baseline by >= 10 points AND the same model without the
cluster by >= 10 points.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import POINTS, encode  # noqa: E402
from train_stroke_transformer import Net  # noqa: E402

K = 5
STRATA = ("same_panel", "same_work", "other_work")


class Corpus:
    def __init__(self, pack):
        pack = Path(pack)
        self.arr = np.load(pack / "strokes.npy", mmap_mode="r")
        self.lab = np.load(pack / "cluster_labels.npy")
        rows = list(csv.DictReader(open(pack / "panels.csv")))
        works = {(r["source"], r["name"]): r["work"] for r in csv.DictReader(open(pack / "panel_works.csv"))}
        self.rows = rows
        n = len(self.lab)
        self.panel_of = np.zeros(n, np.int32)
        wnames = sorted(set(works.values())); widx = {w: i for i, w in enumerate(wnames)}
        self.work_of_panel = np.array([widx[works[(r["source"], r["name"])]] for r in rows], np.int32)
        for i, r in enumerate(rows):
            s, m = int(r["start"]), int(r["n"])
            self.panel_of[s:s + m] = i
        meta = np.asarray(self.arr[:, POINTS * 2:])
        pts = np.asarray(self.arr[:, :POINTS * 2]).reshape(n, POINTS, 2)
        self.arc = meta[:, 3]; self.width = meta[:, 1]
        chord = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1)
        self.straight = chord / np.maximum(self.arc, 1e-6)
        self.work_of = self.work_of_panel[self.panel_of]

    def stroke(self, i):
        b = np.asarray(self.arr[i])
        return b[:POINTS * 2].reshape(POINTS, 2), b[POINTS * 2:]


def split_panels(rows, test_frac=0.25):
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(20260918); rng.shuffle(groups)
    test = set(groups[: int(len(groups) * test_frac)])
    return np.array([r["group"] in test for r in rows])


class Sampler:
    """Draws cloze questions from one side of the split."""

    def __init__(self, C, panel_mask, strict=False):
        self.C, self.strict = C, strict
        ok = panel_mask[C.panel_of] & (C.lab >= 0)
        self.ids = np.flatnonzero(ok)
        # clusters with >= 4 members (target + >= 3 mates)
        key = C.panel_of[self.ids].astype(np.int64) * 100000 + C.lab[self.ids]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        self.targets = self.ids[cnt[inv] >= 4]
        self.pool = np.flatnonzero(panel_mask[C.panel_of])          # distractor pool, same side
        order = np.argsort(C.arc[self.pool]); self.pool_sorted = self.pool[order]
        self.pool_arc = C.arc[self.pool_sorted]

    def _match(self, t, cand):
        C = self.C
        ok = np.abs(C.arc[cand] - C.arc[t]) <= 0.2 * C.arc[t]
        if self.strict:
            ok &= np.abs(C.width[cand] - C.width[t]) <= 0.3 * max(C.width[t], 1.0)
            ok &= np.abs(C.straight[cand] - C.straight[t]) <= 0.1
        return cand[ok]

    def distractors(self, t, stratum, rng):
        C = self.C
        p = C.panel_of[t]
        if stratum == "same_panel":
            s, m = int(C.rows[p]["start"]), int(C.rows[p]["n"])
            cand = np.arange(s, s + m)
            cand = cand[(C.lab[cand] != C.lab[t]) & (C.lab[cand] >= 0)]
        else:
            lo = np.searchsorted(self.pool_arc, 0.8 * C.arc[t]); hi = np.searchsorted(self.pool_arc, 1.2 * C.arc[t])
            cand = self.pool_sorted[lo:hi]
            if len(cand) > 4000:
                cand = cand[rng.choice(len(cand), 4000, replace=False)]
            same_work = C.work_of[cand] == C.work_of[t]
            cand = cand[(C.panel_of[cand] != p) & (same_work if stratum == "same_work" else ~same_work)]
        cand = self._match(t, cand)
        if len(cand) < K - 1:
            return None
        return cand[rng.choice(len(cand), K - 1, replace=False)]

    def question(self, rng, stratum):
        C = self.C
        for _ in range(20):
            t = int(self.targets[rng.integers(len(self.targets))])
            d = self.distractors(t, stratum, rng)
            if d is None:
                continue
            p = C.panel_of[t]; s, m = int(C.rows[p]["start"]), int(C.rows[p]["n"])
            members = np.arange(s, s + m)
            mates = members[(C.lab[members] == C.lab[t]) & (members != t)][:40]
            return t, d, mates
        return None


def build(C, t, d, mates, rng):
    """-> (K, 1+M, F) sequences: candidate token first (flag=1), then the mates."""
    tp, tm = C.stroke(t)
    cen = tp.mean(0)
    cands = [(tp, tm)]
    for j in d:
        pj, mj = C.stroke(j)
        cands.append((pj - pj.mean(0) + cen, mj))         # re-centred on the truth
    order = rng.permutation(K)
    cands = [cands[i] for i in order]
    label = int(np.flatnonzero(order == 0)[0])
    mp = np.stack([C.stroke(i)[0] for i in mates]); mm = np.stack([C.stroke(i)[1] for i in mates])
    mate_tok = np.concatenate([encode(mp, mm), np.zeros((len(mates), 1), np.float32)], 1)
    seqs = []
    for p, m in cands:
        ct = np.concatenate([encode(p[None], m[None]), np.ones((1, 1), np.float32)], 1)
        seqs.append(np.concatenate([ct, mate_tok], 0))
    return np.stack(seqs).astype(np.float32), label, [c[0] for c in cands], mp


class Questions(torch.utils.data.Dataset):
    def __init__(self, C, sampler, n, strata, seed, fixed=False):
        self.C, self.S, self.n, self.strata, self.seed, self.fixed = C, sampler, n, strata, seed, fixed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i if self.fixed else None)
        st = self.strata[i % len(self.strata)]
        q = self.S.question(rng, st)
        while q is None:
            q = self.S.question(rng, st)
        x, y, _c, _m = build(self.C, *q, rng)
        return x, y, STRATA.index(st)


def collate(batch):
    L = max(b[0].shape[1] for b in batch)
    F = batch[0][0].shape[2]
    X = np.zeros((len(batch), K, L, F), np.float32); M = np.ones((len(batch), K, L), bool)
    for i, (x, y, s) in enumerate(batch):
        X[i, :, :x.shape[1]] = x; M[i, :, :x.shape[1]] = False
    return torch.from_numpy(X), torch.from_numpy(M), torch.tensor([b[1] for b in batch]), torch.tensor([b[2] for b in batch])


class Scorer(nn.Module):
    def __init__(self, f_in, d=256, layers=4):
        super().__init__()
        self.body = Net(f_in, d, layers, 8)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, X, M, context=True):
        B, k, L, F = X.shape
        x = X.view(B * k, L, F); m = M.view(B * k, L).clone()
        if not context:
            m[:, 1:] = True
        h = self.body.enc(self.body.inp(x), src_key_padding_mask=m)
        return self.out(h[:, 0]).view(B, k)


# ---- non-learned baselines, scored on the same questions ----
def base_scores(cands, mates):
    ends_m = np.concatenate([mates[:, 0], mates[:, -1]])
    gap, ang = [], []
    cm = mates.mean(1)
    dm = mates[:, -1] - mates[:, 0]
    for c in cands:
        e = np.stack([c[0], c[-1]])
        gap.append(-float(np.linalg.norm(e[:, None] - ends_m[None], axis=-1).min()))
        j = int(np.argmin(np.linalg.norm(cm - c.mean(0), axis=1)))
        v1 = c[-1] - c[0]; v2 = dm[j]
        cos = abs(float(v1 @ v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang.append(cos)
    return np.array(gap), np.array(ang)


def evaluate(model, C, S, n_per, seed, dev, strict_label=""):
    res = {}
    model.eval()
    for si, st in enumerate(STRATA):
        rng = np.random.default_rng(seed + 1000 * si)
        hit = {"model": 0, "no_context": 0, "gap": 0, "angle": 0}; tot = 0
        Xs, Ms, ys, bases = [], [], [], []
        for q_i in range(n_per):
            q = S.question(rng, st)
            if q is None:
                continue
            x, y, cands, mp = build(C, *q, rng)
            g, a = base_scores(cands, mp)
            hit["gap"] += int(np.argmax(g) == y); hit["angle"] += int(np.argmax(a) == y)
            Xs.append(x); ys.append(y); tot += 1
            if len(Xs) == 64 or q_i == n_per - 1:
                X, M, Y, _ = collate([(xx, yy, si) for xx, yy in zip(Xs, ys)])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    s1 = model(X.to(dev), M.to(dev), True).float().cpu()
                    s0 = model(X.to(dev), M.to(dev), False).float().cpu()
                hit["model"] += int((s1.argmax(1) == Y).sum()); hit["no_context"] += int((s0.argmax(1) == Y).sum())
                Xs, ys = [], []
        res[st] = {k: v / max(tot, 1) for k, v in hit.items()}; res[st]["n"] = tot
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--out", default="results/cloze_20260919")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-q", type=int, default=24000)
    p.add_argument("--eval-q", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--strict", action="store_true", help="also match width and straightness")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.manual_seed(20260919)
    C = Corpus(a.pack)
    test = split_panels(C.rows)
    S_tr = Sampler(C, ~test, a.strict); S_te = Sampler(C, test, a.strict)
    print(f"targets train {len(S_tr.targets)} / test {len(S_te.targets)}  strict={a.strict}", flush=True)
    if a.smoke:
        a.epochs, a.train_q, a.eval_q = 1, 256, 60
    dl = torch.utils.data.DataLoader(Questions(C, S_tr, a.train_q, STRATA, 0), batch_size=a.batch,
                                     shuffle=False, num_workers=a.workers, collate_fn=collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_in = POINTS * 2 + 32 + 2 + 3 + 1
    model = Scorer(f_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * len(dl)), pct_start=0.1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "strict" if a.strict else "arc"
    hist = []
    for ep in range(1, a.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        for X, M, Y, _S in dl:
            X, M, Y = X.to(dev), M.to(dev), Y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = nn.functional.cross_entropy(model(X, M, True).float(), Y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(loss); n += 1
        r = evaluate(model, C, S_te, a.eval_q, 777, dev)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), **{f"{s}_{k}": v for s in r for k, v in r[s].items()}})
        line = "  ".join(f"{s}: model {r[s]['model']:.3f} 無文脈 {r[s]['no_context']:.3f} 端点 {r[s]['gap']:.3f} 角度 {r[s]['angle']:.3f}"
                         for s in STRATA)
        print(f"ep {ep:>2} loss {tot/max(n,1):.3f} {time.time()-t0:.0f}s | {line}", flush=True)
        json.dump(hist, open(out / f"history_{tag}.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "args": vars(a)}, out / f"model_{tag}.pt")


if __name__ == "__main__":
    main()
