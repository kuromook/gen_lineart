#!/usr/bin/env python
"""Cloze test, second round: give the model the relation the rule uses, and take
the circularity out of the context.

Round one (cloze.py, 2026-09-19) established that the cluster carries the answer
-- without it the model sits at chance, with it +22 to +31 points -- but a
one-line rule, "pick the candidate whose endpoint touches a mate's endpoint",
beat the model on the hardest stratum (0.476 vs 0.433). Two causes, two fixes:

1. The model could not express contact. A stroke was 16 points relative to its
   own centroid plus sinusoidal position; whether two strokes touch had to be
   inferred through attention. Here every mate token also carries its relation
   to the candidate: log endpoint-to-endpoint gap, log gap from the candidate's
   endpoints to the mate's polyline, tangent continuation at the closest ends,
   |cos| between directions, log centroid distance.

2. The rule was partly circular: clusters were built FROM proximity and endpoint
   continuation, so the true member touched its mates by construction. Training
   uses clusters built without the endpoint edge (cluster_labels_noend.npy).
   EVALUATION goes further: for every held-out question the target is removed
   from its panel, the panel is re-clustered, and the context is the cluster of
   the remaining stroke nearest the target's centroid. The answer's own geometry
   never takes part in choosing its context.

Pre-registered (doc/work_log.md): same bar as round one -- on strata (1) and (2),
beat the best non-learned baseline by >= 10 points AND the no-context model by
>= 10 points, on the re-clustered evaluation set.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloze as c1  # noqa: E402
from cloze import K, STRATA, Corpus, Sampler, Scorer, base_scores, split_panels  # noqa: E402
from stroke_graph import build_edges, cap_degree, communities, load_panel  # noqa: E402
from train_infill import POINTS, encode  # noqa: E402

N_REL = 5


def relations(cand, mates):
    """(M, 5) relation of each mate to the candidate polyline."""
    ce = np.stack([cand[0], cand[-1]])
    ctan = np.stack([cand[1] - cand[0], cand[-2] - cand[-1]])
    ctan /= (np.linalg.norm(ctan, axis=1, keepdims=True) + 1e-6)
    cdir = cand[-1] - cand[0]; cdir = cdir / (np.linalg.norm(cdir) + 1e-6)
    out = np.zeros((len(mates), N_REL), np.float32)
    for k, m in enumerate(mates):
        me = np.stack([m[0], m[-1]])
        mtan = np.stack([m[1] - m[0], m[-2] - m[-1]])
        mtan /= (np.linalg.norm(mtan, axis=1, keepdims=True) + 1e-6)
        de = np.linalg.norm(ce[:, None] - me[None], axis=-1)
        a, b = np.unravel_index(int(de.argmin()), de.shape)
        dpoly = np.linalg.norm(ce[:, None] - m[None], axis=-1).min()
        mdir = m[-1] - m[0]; mdir = mdir / (np.linalg.norm(mdir) + 1e-6)
        out[k] = [np.log1p(de[a, b]) / 5.0, np.log1p(dpoly) / 5.0,
                  float(ctan[a] @ -mtan[b]), abs(float(cdir @ mdir)),
                  np.log1p(np.linalg.norm(cand.mean(0) - m.mean(0))) / 5.0]
    return out


def build(C, t, d, mates, rng, use_rel=True):
    tp, tm = C.stroke(t)
    cen = tp.mean(0)
    cands = [(tp, tm)] + [((lambda pj: pj - pj.mean(0) + cen)(C.stroke(j)[0]), C.stroke(j)[1]) for j in d]
    order = rng.permutation(K)
    cands = [cands[i] for i in order]
    label = int(np.flatnonzero(order == 0)[0])
    mp = np.stack([C.stroke(i)[0] for i in mates]); mm = np.stack([C.stroke(i)[1] for i in mates])
    base = np.concatenate([encode(mp, mm), np.zeros((len(mates), 1), np.float32)], 1)
    seqs = []
    for p, m in cands:
        ct = np.concatenate([encode(p[None], m[None]), np.ones((1, 1), np.float32),
                             np.zeros((1, N_REL), np.float32)], 1)
        rel = relations(p, mp) if use_rel else np.zeros((len(mates), N_REL), np.float32)
        seqs.append(np.concatenate([ct, np.concatenate([base, rel], 1)], 0))
    return np.stack(seqs).astype(np.float32), label, [c[0] for c in cands], mp


class Questions(torch.utils.data.Dataset):
    def __init__(self, C, S, n, use_rel):
        self.C, self.S, self.n, self.use_rel = C, S, n, use_rel

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        st = STRATA[i % len(STRATA)]
        q = self.S.question(rng, st)
        while q is None:
            q = self.S.question(rng, st)
        x, y, _c, _m = build(self.C, *q, rng, self.use_rel)
        return x, y, STRATA.index(st)


# ---- the re-clustered, non-circular evaluation set ----
def _eval_job(t):
    pack, start, n, target, seed = t
    arr = np.load(Path(pack) / "strokes.npy", mmap_mode="r")
    pts, meta = load_panel(arr, start, n)
    keep = np.array([i for i in range(n) if i != target - start])
    lab = communities(cap_degree(build_edges(pts[keep], meta[keep], rule="knn", use_end=False), len(keep)),
                      len(keep), 2.0, 0)
    tc = pts[target - start].mean(0)
    nearest = int(np.argmin(np.linalg.norm(pts[keep].mean(1) - tc, axis=1)))
    mates = keep[lab == lab[nearest]]
    return target, (start + mates[:40]).tolist()


def eval_set(C, S, n_per, seed, pack, workers):
    """Fixed questions; context from the panel re-clustered WITHOUT the target."""
    qs = []
    for si, st in enumerate(STRATA):
        rng = np.random.default_rng(seed + 1000 * si)
        got = 0
        while got < n_per:
            q = S.question(rng, st)
            if q is None:
                continue
            t, d, _m = q
            qs.append((st, int(t), [int(x) for x in d], int(rng.integers(1 << 31))))
            got += 1
    tasks = []
    for st, t, d, s in qs:
        p = C.panel_of[t]
        tasks.append((pack, int(C.rows[p]["start"]), int(C.rows[p]["n"]), t, s))
    t0 = time.time()
    with Pool(workers) as pool:
        mates = dict(pool.imap_unordered(_eval_job, tasks, chunksize=4))
    out = [(st, t, d, mates[t], s) for st, t, d, s in qs if len(mates[t]) >= 3]
    print(f"eval set: {len(out)}/{len(qs)} questions with >= 3 re-clustered mates, {time.time()-t0:.0f}s", flush=True)
    return out


def evaluate(model, C, E, dev, use_rel):
    res = {st: {"model": 0, "no_context": 0, "gap": 0, "angle": 0, "n": 0} for st in STRATA}
    model.eval()
    buf = []

    def flush():
        X, M, Y, S = c1.collate([(x, y, si) for x, y, si in buf])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            s1 = model(X.to(dev), M.to(dev), True).float().cpu()
            s0 = model(X.to(dev), M.to(dev), False).float().cpu()
        for k in range(len(buf)):
            st = STRATA[int(S[k])]
            res[st]["model"] += int(s1[k].argmax() == Y[k]); res[st]["no_context"] += int(s0[k].argmax() == Y[k])
        buf.clear()

    for st, t, d, mates, s in E:
        rng = np.random.default_rng(s)
        x, y, cands, mp = build(C, t, np.array(d), np.array(mates), rng, use_rel)
        g, a = base_scores(cands, mp)
        res[st]["gap"] += int(np.argmax(g) == y); res[st]["angle"] += int(np.argmax(a) == y); res[st]["n"] += 1
        buf.append((x, y, STRATA.index(st)))
        if len(buf) == 64:
            flush()
    if buf:
        flush()
    model.train()
    return {st: {k: (v / max(r["n"], 1) if k != "n" else v) for k, v in r.items()} for st, r in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--out", default="results/cloze2_20260919")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-q", type=int, default=24000)
    p.add_argument("--eval-q", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-rel", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--eval-workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.manual_seed(20260919)
    C = Corpus(a.pack)
    C.lab = np.load(Path(a.pack) / "cluster_labels_noend.npy")
    test = split_panels(C.rows)
    S_tr = Sampler(C, ~test, a.strict); S_te = Sampler(C, test, a.strict)
    if a.smoke:
        a.epochs, a.train_q, a.eval_q = 1, 256, 40
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / f"evalset_{'strict' if a.strict else 'arc'}_{a.eval_q}.json"
    if cache.exists():
        E = [tuple(e) for e in json.load(open(cache))]
    else:
        E = eval_set(C, S_te, a.eval_q, 777, a.pack, a.eval_workers)
        json.dump(E, open(cache, "w"))
    use_rel = not a.no_rel
    tag = ("strict" if a.strict else "arc") + ("_norel" if a.no_rel else "_rel")
    dl = torch.utils.data.DataLoader(Questions(C, S_tr, a.train_q, use_rel), batch_size=a.batch,
                                     num_workers=a.workers, collate_fn=c1.collate)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Scorer(POINTS * 2 + 32 + 2 + 3 + 1 + N_REL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, a.epochs * len(dl)), pct_start=0.1)
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
        r = evaluate(model, C, E, dev, use_rel)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), **{f"{s}_{k}": v for s in r for k, v in r[s].items()}})
        print(f"ep {ep:>2} loss {tot/max(n,1):.3f} {time.time()-t0:.0f}s | " + "  ".join(
            f"{s}: model {r[s]['model']:.3f} 無文脈 {r[s]['no_context']:.3f} 端点 {r[s]['gap']:.3f} 角度 {r[s]['angle']:.3f}"
            for s in STRATA), flush=True)
        json.dump(hist, open(out / f"history_{tag}.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "args": vars(a)}, out / f"model_{tag}.pt")


if __name__ == "__main__":
    main()
