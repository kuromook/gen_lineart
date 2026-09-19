#!/usr/bin/env python
"""Is this vocabulary size a good WORD level? (pre-registered 2026-09-19)

A word must not change when a detail stroke is removed and must change when a
defining one is (user's criterion, after the 5,000-word codebook flipped its
word 48.5% of the time on removing a minor stroke).

  D            P(change | remove the longest stroke) - P(change | remove a 4th-or-lower)
  minor        P(change | remove a 4th-or-lower stroke)            -- must be <= 25%
  works        share of the top-100 words spanning >= 3 works       -- must be 100%
  used         share of the vocabulary in use (train)               -- must be >= 50%
  purity       same-word / different-word chamfer between clusters  -- lower pins shape down
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load, stroke_feats  # noqa: E402
from train_codebook2 import Codebook2  # noqa: E402


def words_of(m, P, W, M, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(P), bs):
            _q, w = m.encode(P[i:i + bs].float(), W[i:i + bs].float(), M[i:i + bs], 1)
            out.append(w)
    return torch.cat(out)


def chamfer(a, b):
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    levels = tuple(ck.get("levels", (8, 5, 5, 5, 5)))
    m = Codebook2(levels=levels).to(dev); m.load_state_dict(ck["model"]); m.eval()
    m.rfsq.stages = ck["args"].get("stages", 3)
    n_words = int(np.prod(levels))
    tp, tw, tm, tmeta = load("results/cluster_set_20260919", "train", dev)
    vp, vw, vm, vmeta = load("results/cluster_set_20260919", "test", dev)
    tw_words = words_of(m, tp, tw, tm).cpu().numpy()
    cnt = np.bincount(tw_words, minlength=n_words)
    used = float((cnt > 0).mean())
    top = np.argsort(-cnt)[:100]
    works = defaultdict(set)
    for wd, mt in zip(tw_words, tmeta):
        works[wd].add(mt["work"])
    top_works = float(np.mean([len(works[w]) >= 3 for w in top]))
    # removal test on held-out clusters
    rng = np.random.default_rng(0)
    idx = rng.choice(len(vp), min(a.n, len(vp)), replace=False)
    ch_long, ch_minor = [], []
    with torch.no_grad():
        for i in idx:
            pp, ww, mk = vp[i].float(), vw[i].float(), vm[i]
            n = int(mk.sum())
            if n < 4:
                continue
            arc = (pp[:n, 1:] - pp[:n, :-1]).norm(dim=-1).sum(-1)
            order = torch.argsort(arc, descending=True).cpu().numpy()
            picks = [order[0]] + list(rng.choice(order[3:], min(2, n - 3), replace=False))
            P = pp[None].repeat(len(picks) + 1, 1, 1, 1); W = ww[None].repeat(len(picks) + 1, 1); M = mk[None].repeat(len(picks) + 1, 1)
            for k, s in enumerate(picks):
                M[k + 1, s] = False
            _q, wds = m.encode(P, W, M, 1)
            ch_long.append(int(wds[1] != wds[0]))
            ch_minor += [int(wds[k + 1] != wds[0]) for k in range(1, len(picks))]
    pl, pm = float(np.mean(ch_long)), float(np.mean(ch_minor))
    # purity: same-word vs different-word shape distance
    by_word = defaultdict(list)
    for j, wd in enumerate(tw_words):
        by_word[wd].append(j)
    v_words = words_of(m, vp[idx], vw[idx], vm[idx]).cpu().numpy()
    same, diff = [], []
    for i, wd in zip(idx[:600], v_words[:600]):
        pool = by_word.get(wd, [])
        if len(pool) < 2:
            continue
        a_pts = vp[i][vm[i]].float().reshape(-1, 2)
        for j in rng.choice(pool, min(3, len(pool)), replace=False):
            same.append(float(chamfer(a_pts, tp[j][tm[j]].float().reshape(-1, 2))))
        for j in rng.choice(len(tp), 3, replace=False):
            diff.append(float(chamfer(a_pts, tp[j][tm[j]].float().reshape(-1, 2))))
    purity = float(np.median(same) / np.median(diff))
    res = {"levels": levels, "n_words": n_words, "D": pl - pm, "p_change_longest": pl, "p_change_minor": pm,
           "used": used, "top100_3works": top_works, "purity": purity,
           "pass": bool(pm <= 0.25 and top_works == 1.0 and used >= 0.5)}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
