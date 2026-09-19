#!/usr/bin/env python
"""Generate-and-select by retrieval (option B, 2026-09-19).

The codebook's encoder carries cluster identity but its decoder cannot draw
(v2 decodes every cluster as a patch of short dashes). B bypasses the decoder:
the proposals are REAL strokes from training clusters that share the query's
word, and the cloze2 model -- the one pre-registered pass in this track --
chooses among them. It tests "propose with vocabulary, choose with grammar"
without the broken part.

Per question: mask one stroke of a held-out cluster; re-normalise the remaining
strokes on THEIR OWN centroid and bbox (the frame never sees the answer); encode
them -> word; retrieve up to 20 training clusters with that word (nearest words
on the FSQ grid if fewer); map every stroke of those clusters into the query
frame -> pool. Choose by the model, the endpoint rule, random; report the pool's
oracle. Caveat: cloze2 was trained with candidates re-centred on the truth; here
candidates carry their own location. Stroke meta is approximated from the
cluster set (width de-normalised, fill share ~ width > 8).
"""
import csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook2 import Codebook2
from cloze import Scorer, base_scores
from cloze2 import N_REL, relations
from train_infill import POINTS, chamfer, encode

DATA = Path("results/cluster_set_20260919")
OUT = Path("results/generate_select_20260919")
SIDE = 56.0
POOL_CLUSTERS = 20
N_Q = 1000
SEED = 20260919


def meta_of(pts, width):
    """(n,16,2) native px, (n,) native width -> (n,5) meta for train_infill.encode"""
    arc = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
    return np.stack([np.zeros(len(pts)), width, (width > 8).astype(np.float32), arc, np.ones(len(pts))], 1).astype(np.float32)


def model_scores(model, dev, cands, cw, mates, mw):
    mm = meta_of(mates, mw)
    base = np.concatenate([encode(mates, mm), np.zeros((len(mates), 1), np.float32)], 1)
    cm = meta_of(cands, cw)
    ct = np.concatenate([encode(cands, cm), np.ones((len(cands), 1), np.float32),
                         np.zeros((len(cands), N_REL), np.float32)], 1)
    seqs = np.stack([np.concatenate([ct[k:k + 1], np.concatenate([base, relations(cands[k], mates)], 1)], 0)
                     for k in range(len(cands))]).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            X = torch.from_numpy(seqs[None, i:i + 256]).to(dev)
            M = torch.zeros(X.shape[:3], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(X, M, True).float().cpu().numpy()[0])
    return np.concatenate(out)


def main():
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    meta = list(csv.DictReader(open(DATA / "meta.csv")))
    pts_all = np.load(DATA / "pts.npy").astype(np.float32)
    w_all = np.load(DATA / "width.npy").astype(np.float32)
    m_all = np.load(DATA / "mask.npy")
    split = np.array([m["split"] for m in meta])
    tr = np.flatnonzero(split == "train"); te = np.flatnonzero(split == "test")

    ck = torch.load("results/codebook2_20260919/codebook2.pt", map_location=dev, weights_only=False)
    cb = Codebook2().to(dev); cb.load_state_dict(ck["model"]); cb.eval()
    fsq = cb.rfsq

    # word of every training cluster (whole cluster)
    words_tr = np.zeros(len(tr), np.int64)
    with torch.no_grad():
        for i in range(0, len(tr), 2048):
            idx = tr[i:i + 2048]
            _q, w = cb.encode(torch.from_numpy(pts_all[idx]).to(dev), torch.from_numpy(w_all[idx]).to(dev),
                              torch.from_numpy(m_all[idx]).to(dev), 1)
            words_tr[i:i + 2048] = w.cpu().numpy()
    by_word = {}
    for k, w in enumerate(words_tr):
        by_word.setdefault(int(w), []).append(int(tr[k]))
    used = np.array(sorted(by_word))
    L = fsq.L.cpu().numpy().astype(int); basis = fsq.basis.cpu().numpy().astype(int)
    grid = lambda w: (np.asarray(w)[..., None] // basis) % L
    used_grid = grid(used)
    print(f"train words {len(used)}", flush=True)

    ck2 = torch.load("results/cloze2_20260919/model_arc_rel.pt", map_location=dev, weights_only=False)
    scorer = Scorer(POINTS * 2 + 32 + 2 + 3 + 1 + N_REL).to(dev)
    scorer.load_state_dict(ck2["model"]); scorer.eval()

    rng = np.random.default_rng(SEED)
    cand_q = te[m_all[te].sum(1) >= 4]
    qs = rng.choice(cand_q, N_Q, replace=False)
    rows, examples = [], []
    t0 = time.time()
    for qi, c in enumerate(qs):
        n = int(m_all[c].sum()); sc = float(meta[c]["scale"])
        cen = np.array([float(meta[c]["cy"]), float(meta[c]["cx"])], np.float32)
        native = pts_all[c, :n] / sc + cen; wn = w_all[c, :n] / sc
        t = int(rng.integers(n))
        truth = native[t]
        keep = np.array([i for i in range(n) if i != t])
        rem, remw = native[keep], wn[keep]
        # frame from the remaining strokes only
        flat = rem.reshape(-1, 2); c2 = flat.mean(0)
        s2 = SIDE / max(float((flat.max(0) - flat.min(0)).max()), 1e-3)
        qp = np.zeros((1, 32, POINTS, 2), np.float32); qw = np.zeros((1, 32), np.float32); qm = np.zeros((1, 32), bool)
        qp[0, :len(keep)] = (rem - c2) * s2; qw[0, :len(keep)] = remw * s2; qm[0, :len(keep)] = True
        with torch.no_grad():
            _q, w = cb.encode(torch.from_numpy(qp).to(dev), torch.from_numpy(qw).to(dev), torch.from_numpy(qm).to(dev), 1)
        word = int(w.item())
        pool_ids = list(by_word.get(word, []))
        if len(pool_ids) < POOL_CLUSTERS:
            d = np.abs(used_grid - grid(word)).sum(1)
            for j in np.argsort(d, kind="stable"):
                if int(used[j]) == word:
                    continue
                pool_ids += by_word[int(used[j])]
                if len(pool_ids) >= POOL_CLUSTERS:
                    break
        pool_ids = [pool_ids[i] for i in rng.permutation(len(pool_ids))[:POOL_CLUSTERS]]
        cands, cw = [], []
        for pc in pool_ids:
            k = int(m_all[pc].sum())
            cands.append(pts_all[pc, :k] / s2 + c2); cw.append(w_all[pc, :k] / s2)
        cands = np.concatenate(cands); cw = np.concatenate(cw)
        ch = np.array([chamfer(x, truth) for x in cands])
        s_model = model_scores(scorer, dev, cands, cw, rem, remw)
        g, _a = base_scores(cands, rem)
        picks = {"model": int(np.argmax(s_model)), "endpoint": int(np.argmax(g)),
                 "random": int(rng.integers(len(cands))), "oracle": int(np.argmin(ch))}
        r = {"q": int(c), "n": n, "pool": len(cands), "word_hit": int(word in by_word),
             "work": meta[c]["work"]}
        for k, p in picks.items():
            r[f"{k}_chamfer"] = float(ch[p])
            r[f"{k}_cent"] = float(np.linalg.norm(cands[p].mean(0) - truth.mean(0)))
        rows.append(r)
        if len(examples) < 16:
            examples.append((rem, truth, cands[picks["model"]], cands[picks["oracle"]], r))
        if (qi + 1) % 100 == 0:
            print(f"{qi+1}/{N_Q} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_question.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summ = {}
    print(f"\n{'選び方':<10}{'chamfer中央値':>14}{'8px以内':>10}{'16px以内':>10}{'重心誤差中央値':>16}")
    for k in ("model", "endpoint", "random", "oracle"):
        ch = np.array([r[f"{k}_chamfer"] for r in rows]); ce = np.array([r[f"{k}_cent"] for r in rows])
        summ[k] = {"chamfer_med": float(np.median(ch)), "w8": float((ch <= 8).mean()), "w16": float((ch <= 16).mean()),
                   "cent_med": float(np.median(ce))}
        print(f"{k:<10}{np.median(ch):>14.1f}{(ch<=8).mean():>10.1%}{(ch<=16).mean():>10.1%}{np.median(ce):>16.1f}")
    pool = np.array([r["pool"] for r in rows])
    summ["pool_median"] = float(np.median(pool))
    summ["word_hit_rate"] = float(np.mean([r["word_hit"] for r in rows]))
    ratio = summ["model"]["chamfer_med"] / summ["random"]["chamfer_med"]
    summ["model_over_random"] = ratio
    print(f"pool median {np.median(pool):.0f} strokes; query word seen in training {summ['word_hit_rate']:.1%}; model/random {ratio:.3f}")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    # montage
    cells = []
    for rem, truth, mp, op, r in examples:
        allp = np.concatenate([rem.reshape(-1, 2), truth, mp, op])
        lo = allp.min(0) - 10; span = float((allp.max(0) - lo).max()) + 10; z = 300.0 / span
        img = np.full((330, 320, 3), 255, np.uint8)
        tf = lambda p: np.round((p - lo) * z + [20, 0])[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        for m in rem:
            cv2.polylines(img, [tf(m)], False, (170, 170, 170), 2)
        cv2.polylines(img, [tf(truth)], False, (40, 170, 40), 3)
        cv2.polylines(img, [tf(op)], False, (220, 120, 0), 1)
        cv2.polylines(img, [tf(mp)], False, (30, 30, 220), 2)
        cv2.putText(img, f"model {r['model_chamfer']:.0f}px  oracle {r['oracle_chamfer']:.0f}px", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cells.append(img)
    grid_img = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, 16, 4)])
    cv2.imwrite(str(OUT / "montage.png"), grid_img)
    print("montage ->", OUT / "montage.png", "(grey=remaining, green=truth, red=model pick, blue=oracle pick)")


if __name__ == "__main__":
    main()
