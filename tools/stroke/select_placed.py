#!/usr/bin/env python
"""A selector that knows WHERE: candidates keep their own location.

Option B (generate_select.py, 2026-09-19) showed retrieval by codebook word
proposes well -- the best candidate in the pool is within 16px of the hole 67.7%
of the time (oracle chamfer 10.1px) -- while cloze2 picked at 60.0px, no better
than the endpoint rule (59.7px). cloze2 was trained with every candidate
re-centred on the truth, so it learned which shape belongs at a given place and
never where the gap is; on the montage its picks run along lines already drawn.

Here the selector is trained on the very kind of pool it will face: mask one
stroke of a TRAIN cluster, encode the remaining strokes to a word, retrieve up to
20 training clusters with that word (excluding the query's own panel and group),
de-normalise their strokes into the query frame. The positive is the pool member
nearest the truth by chamfer (question skipped unless within 16px); the rest are
negatives, plus HARD negatives that draw where ink already is: copies of the
query's own remaining strokes, and pool strokes shifted onto a remaining stroke.
Loss: softmax over the pool (listwise cross-entropy).

Features: cloze2's token encoding and per-mate relations, plus six location
features on the candidate token -- share of its points within 2px / 4px of
existing ink, log min / median distance to existing ink, share inside the hull
of the remaining strokes, centroid distance to the remaining cluster relative to
its radius. `--no-loc` zeroes them (the ablation).

Evaluation re-creates option B's 1,000 questions exactly (same seed, same order
of random draws, same pools) and scores the new selector, the ablation, cloze2,
the endpoint rule, random and the oracle side by side.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook2 import Codebook2
from cloze import Scorer, base_scores
from cloze2 import N_REL, relations
from train_infill import POINTS, chamfer, encode
from generate_select import DATA, POOL_CLUSTERS, N_Q, SEED, SIDE, meta_of

OUT = Path("results/select_placed_20260919")
N_LOC = 6
F_IN = POINTS * 2 + 32 + 2 + 3 + 1 + N_REL + N_LOC


# ---------------------------------------------------------------- corpus
class Corpus:
    def __init__(self, dev):
        self.meta = list(csv.DictReader(open(DATA / "meta.csv")))
        self.pts = np.load(DATA / "pts.npy").astype(np.float32)
        self.w = np.load(DATA / "width.npy").astype(np.float32)
        self.m = np.load(DATA / "mask.npy")
        split = np.array([m["split"] for m in self.meta])
        self.tr = np.flatnonzero(split == "train"); self.te = np.flatnonzero(split == "test")
        self.panel = np.array([int(m["panel"]) for m in self.meta])
        self.group = np.array([m["group"] for m in self.meta])
        ck = torch.load("results/codebook2_20260919/codebook2.pt", map_location=dev, weights_only=False)
        self.cb = Codebook2().to(dev); self.cb.load_state_dict(ck["model"]); self.cb.eval()
        self.dev = dev
        words = np.zeros(len(self.tr), np.int64)
        with torch.no_grad():
            for i in range(0, len(self.tr), 2048):
                idx = self.tr[i:i + 2048]
                _q, w = self.cb.encode(*self._t(self.pts[idx], self.w[idx], self.m[idx]), 1)
                words[i:i + 2048] = w.cpu().numpy()
        self.by_word = {}
        for k, w in enumerate(words):
            self.by_word.setdefault(int(w), []).append(int(self.tr[k]))
        self.used = np.array(sorted(self.by_word))
        L = self.cb.rfsq.L.cpu().numpy().astype(int); basis = self.cb.rfsq.basis.cpu().numpy().astype(int)
        self.grid = lambda w: (np.asarray(w)[..., None] // basis) % L
        self.used_grid = self.grid(self.used)

    def _t(self, p, w, m):
        return (torch.from_numpy(np.ascontiguousarray(p)).to(self.dev), torch.from_numpy(np.ascontiguousarray(w)).to(self.dev),
                torch.from_numpy(np.ascontiguousarray(m)).to(self.dev))

    def native(self, c):
        n = int(self.m[c].sum()); sc = float(self.meta[c]["scale"])
        cen = np.array([float(self.meta[c]["cy"]), float(self.meta[c]["cx"])], np.float32)
        return self.pts[c, :n] / sc + cen, self.w[c, :n] / sc

    def frame(self, rem, remw):
        flat = rem.reshape(-1, 2); c2 = flat.mean(0)
        s2 = SIDE / max(float((flat.max(0) - flat.min(0)).max()), 1e-3)
        qp = np.zeros((32, POINTS, 2), np.float32); qw = np.zeros(32, np.float32); qm = np.zeros(32, bool)
        qp[:len(rem)] = (rem - c2) * s2; qw[:len(rem)] = remw * s2; qm[:len(rem)] = True
        return c2, s2, qp, qw, qm

    def words_of(self, qps, qws, qms):
        out = []
        with torch.no_grad():
            for i in range(0, len(qps), 2048):
                _q, w = self.cb.encode(*self._t(qps[i:i + 2048], qws[i:i + 2048], qms[i:i + 2048]), 1)
                out.append(w.cpu().numpy())
        return np.concatenate(out)

    def pool_ids(self, word, rng, exclude_panel=None, exclude_group=None):
        ids = list(self.by_word.get(word, []))
        if exclude_panel is not None:
            ids = [i for i in ids if self.panel[i] != exclude_panel and self.group[i] != exclude_group]
        if len(ids) < POOL_CLUSTERS:
            d = np.abs(self.used_grid - self.grid(word)).sum(1)
            for j in np.argsort(d, kind="stable"):
                if int(self.used[j]) == word:
                    continue
                more = self.by_word[int(self.used[j])]
                if exclude_panel is not None:
                    more = [i for i in more if self.panel[i] != exclude_panel and self.group[i] != exclude_group]
                ids += more
                if len(ids) >= POOL_CLUSTERS:
                    break
        return [ids[i] for i in rng.permutation(len(ids))[:POOL_CLUSTERS]]

    def pool(self, ids, c2, s2):
        cands, cw = [], []
        for pc in ids:
            k = int(self.m[pc].sum())
            cands.append(self.pts[pc, :k] / s2 + c2); cw.append(self.w[pc, :k] / s2)
        return np.concatenate(cands), np.concatenate(cw)


# ---------------------------------------------------------------- features
def densify(strokes, step=1.0):
    out = []
    for s in strokes:
        seg = np.diff(s, axis=0); L = np.linalg.norm(seg, axis=1)
        for a, v, l in zip(s[:-1], seg, L):
            k = max(1, int(np.ceil(l / step)))
            out.append(a + v * (np.arange(k)[:, None] / k))
        out.append(s[-1:])
    return np.concatenate(out)


def loc_features(cands, rem):
    """(K, 6) location features of every candidate against the remaining ink."""
    tree = cKDTree(densify(rem))
    d, _ = tree.query(cands.reshape(-1, 2))
    d = d.reshape(len(cands), POINTS)
    flat = rem.reshape(-1, 2)
    cen = flat.mean(0); rad = float(np.linalg.norm(flat - cen, axis=1).mean()) + 1.0
    try:
        hull = Delaunay(flat)
        inside = (hull.find_simplex(cands.reshape(-1, 2)) >= 0).reshape(len(cands), POINTS).mean(1)
    except Exception:
        inside = np.zeros(len(cands))
    return np.stack([(d <= 2).mean(1), (d <= 4).mean(1), np.log1p(d.min(1)) / 5.0, np.log1p(np.median(d, 1)) / 5.0,
                     inside, np.linalg.norm(cands.mean(1) - cen, axis=1) / rad], 1).astype(np.float32)


def relations_batch(cands, mates):
    """cloze2.relations for every candidate at once -> (K, M, 5). Same five
    quantities in the same order; vectorised because the per-candidate loop made
    one training question cost 0.12s."""
    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
    ce = np.stack([cands[:, 0], cands[:, -1]], 1)                              # K,2,2
    ctan = unit(np.stack([cands[:, 1] - cands[:, 0], cands[:, -2] - cands[:, -1]], 1))
    cdir = unit(cands[:, -1] - cands[:, 0])
    me = np.stack([mates[:, 0], mates[:, -1]], 1)                              # M,2,2
    mtan = unit(np.stack([mates[:, 1] - mates[:, 0], mates[:, -2] - mates[:, -1]], 1))
    mdir = unit(mates[:, -1] - mates[:, 0])
    de = np.linalg.norm(ce[:, None, :, None, :] - me[None, :, None, :, :], axis=-1)   # K,M,2,2
    flat = de.reshape(len(cands), len(mates), 4)
    j = flat.argmin(-1); a, b = j // 2, j % 2
    dmin = np.take_along_axis(flat, j[..., None], -1)[..., 0]
    dpoly = np.linalg.norm(ce[:, None, :, None, :] - mates[None, :, None, :, :], axis=-1).min((-1, -2))
    ct = ctan[np.arange(len(cands))[:, None], a]                                # K,M,2
    mt = mtan[np.arange(len(mates))[None, :], b]                                # K,M,2
    cont = (ct * -mt).sum(-1)
    cosd = np.abs(cdir @ mdir.T)
    cd = np.linalg.norm(cands.mean(1)[:, None] - mates.mean(1)[None], axis=-1)
    return np.stack([np.log1p(dmin) / 5.0, np.log1p(dpoly) / 5.0, cont, cosd, np.log1p(cd) / 5.0], -1).astype(np.float32)


def sequences(cands, cw, rem, remw, use_loc=True):
    mm = meta_of(rem, remw)
    base = np.concatenate([encode(rem, mm), np.zeros((len(rem), 1), np.float32)], 1)
    loc = loc_features(cands, rem) if use_loc else np.zeros((len(cands), N_LOC), np.float32)
    cm = meta_of(cands, cw)
    ct = np.concatenate([encode(cands, cm), np.ones((len(cands), 1), np.float32),
                         np.zeros((len(cands), N_REL), np.float32), loc], 1)
    zl = np.zeros((len(rem), N_LOC), np.float32)
    rel = relations_batch(cands, rem)                                              # K,M,5
    K, M = len(cands), len(rem)
    mates = np.concatenate([np.broadcast_to(base, (K, M, base.shape[1])), rel, np.broadcast_to(zl, (K, M, N_LOC))], 2)
    return np.concatenate([ct[:, None, :], mates], 1).astype(np.float32)


def overlaps_ink(cand, rem, tol=3.0):
    d, _ = cKDTree(densify(rem)).query(cand)
    return float((d <= tol).mean()) > 0.5


# ---------------------------------------------------------------- training data
class TrainQuestions(torch.utils.data.Dataset):
    def __init__(self, C, qs, use_loc, hard=4, cap=400):
        self.C, self.qs, self.use_loc, self.hard, self.cap = C, qs, use_loc, hard, cap

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, i):
        C = self.C
        c, t, word, c2, s2, seed = self.qs[i]
        rng = np.random.default_rng(seed)
        native, wn = C.native(c)
        truth = native[t]; keep = np.array([k for k in range(len(native)) if k != t])
        rem, remw = native[keep], wn[keep]
        ids = C.pool_ids(int(word), rng, C.panel[c], C.group[c])
        cands, cw = C.pool(ids, c2, s2)
        if len(cands) > self.cap:
            sel = rng.choice(len(cands), self.cap, replace=False); cands, cw = cands[sel], cw[sel]
        ch = np.array([chamfer(x, truth) for x in cands])
        pos = int(np.argmin(ch))
        if ch[pos] > 16.0:
            return None
        # hard negatives: draw where ink already is
        extra, ew = [], []
        for k in rng.choice(len(rem), min(self.hard, len(rem)), replace=False):
            extra.append(rem[k]); ew.append(remw[k])
        for k in rng.choice(len(cands), min(self.hard, len(cands)), replace=False):
            tgt = rem[int(rng.integers(len(rem)))]
            extra.append(cands[k] - cands[k].mean(0) + tgt.mean(0)); ew.append(cw[k])
        cands = np.concatenate([cands, np.stack(extra)]); cw = np.concatenate([cw, np.array(ew)])
        return torch.from_numpy(sequences(cands, cw, rem, remw, self.use_loc)), pos


def build_train_questions(C, n, seed):
    rng = np.random.default_rng(seed)
    cand = C.tr[C.m[C.tr].sum(1) >= 4]
    cs = rng.choice(cand, n, replace=True)
    rows, qps, qws, qms = [], [], [], []
    for c in cs:
        native, wn = C.native(c)
        t = int(rng.integers(len(native)))
        keep = np.array([k for k in range(len(native)) if k != t])
        c2, s2, qp, qw, qm = C.frame(native[keep], wn[keep])
        rows.append([int(c), t, c2, s2, int(rng.integers(1 << 31))]); qps.append(qp); qws.append(qw); qms.append(qm)
    words = C.words_of(np.stack(qps), np.stack(qws), np.stack(qms))
    return [(r[0], r[1], int(w), r[2], r[3], r[4]) for r, w in zip(rows, words)]


def train(C, use_loc, n_q, epochs, lr, workers, dev, tag):
    qs = build_train_questions(C, n_q, SEED + 7)
    ds = TrainQuestions(C, qs, use_loc)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True, num_workers=workers,
                                     collate_fn=lambda x: x, persistent_workers=workers > 0)
    model = Scorer(F_IN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    accum = 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=max(1, epochs * len(ds) // accum + epochs), pct_start=0.1)
    for ep in range(1, epochs + 1):
        t0 = time.time(); tot = 0.0; n = 0; hit = 0; k = 0
        model.train()
        for item in dl:
            if item is None:
                continue
            X, pos = item
            X = X.to(dev)[None]
            M = torch.zeros(X.shape[:3], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(X, M, True).float()
            loss = nn.functional.cross_entropy(s, torch.tensor([pos], device=dev)) / accum
            loss.backward(); k += 1
            tot += float(loss) * accum; n += 1; hit += int(s.argmax(1).item() == pos)
            if k % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        print(f"[{tag}] ep {ep} loss {tot/max(n,1):.3f} train top1 {hit/max(n,1):.3f} questions {n} {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "use_loc": use_loc}, OUT / f"selector_{tag}.pt")
    return model


# ---------------------------------------------------------------- evaluation (option B's questions)
def scores(model, dev, seqs):
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            X = torch.from_numpy(seqs[None, i:i + 256]).to(dev)
            M = torch.zeros(X.shape[:3], dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(X, M, True).float().cpu().numpy()[0])
    return np.concatenate(out)


def evaluate(C, models, dev, n_q=N_Q):
    from generate_select import model_scores
    ck2 = torch.load("results/cloze2_20260919/model_arc_rel.pt", map_location=dev, weights_only=False)
    cloze2 = Scorer(POINTS * 2 + 32 + 2 + 3 + 1 + N_REL).to(dev); cloze2.load_state_dict(ck2["model"]); cloze2.eval()
    rng = np.random.default_rng(SEED)
    cand_q = C.te[C.m[C.te].sum(1) >= 4]
    qs = rng.choice(cand_q, N_Q, replace=False)
    ref = {int(r["q"]): r for r in csv.DictReader(open("results/generate_select_20260919/per_question.csv"))}
    rows, examples, mismatch = [], [], 0
    t0 = time.time()
    for qi, c in enumerate(qs[:n_q]):
        native, wn = C.native(c)
        t = int(rng.integers(len(native)))
        truth = native[t]; keep = np.array([k for k in range(len(native)) if k != t])
        rem, remw = native[keep], wn[keep]
        c2, s2, qp, qw, qm = C.frame(rem, remw)
        word = int(C.words_of(qp[None], qw[None], qm[None])[0])
        ids = C.pool_ids(word, rng)
        cands, cw = C.pool(ids, c2, s2)
        ch = np.array([chamfer(x, truth) for x in cands])
        picks = {}
        for tag, (model, use_loc) in models.items():
            picks[tag] = int(np.argmax(scores(model, dev, sequences(cands, cw, rem, remw, use_loc))))
        picks["cloze2"] = int(np.argmax(model_scores(cloze2, dev, cands, cw, rem, remw)))
        g, _a = base_scores(cands, rem)
        picks["endpoint"] = int(np.argmax(g))
        picks["random"] = int(rng.integers(len(cands)))
        picks["oracle"] = int(np.argmin(ch))
        r0 = ref.get(int(c))
        if r0 is None or int(r0["pool"]) != len(cands) or abs(float(r0["oracle_chamfer"]) - ch[picks["oracle"]]) > 1e-3 \
                or abs(float(r0["random_chamfer"]) - ch[picks["random"]]) > 1e-3:
            mismatch += 1
        r = {"q": int(c), "pool": len(cands)}
        for k, p in picks.items():
            r[f"{k}_chamfer"] = float(ch[p])
            r[f"{k}_overlap"] = int(overlaps_ink(cands[p], rem))
        rows.append(r)
        if len(examples) < 16:
            examples.append((rem, truth, cands[picks.get("loc", picks["oracle"])], cands[picks["oracle"]], r))
        if (qi + 1) % 100 == 0:
            print(f"eval {qi+1}/{N_Q} {time.time()-t0:.0f}s  mismatches vs option B so far {mismatch}", flush=True)
    return rows, examples, mismatch


STRATA = (("short <40px", 0, 40), ("mid 40-120px", 40, 120), ("long >120px", 120, 1e9))
METHODS = ("loc", "noloc", "cloze2", "endpoint", "random", "oracle")


def reconstruct_arcs(C, n_q=N_Q):
    """Arc length (native px) of the masked stroke of each evaluation question.

    Replays evaluate()'s random draws in the same order -- target, pool
    permutation, random pick -- without scoring anything, so a finished run can
    be stratified after the fact (added 2026-09-19 on the user's request: a short
    masked stroke barely changes the cluster's word, so its question is detail
    guessing, and chamfer grows with stroke length, so a pooled median mixes
    regimes)."""
    rng = np.random.default_rng(SEED)
    cand_q = C.te[C.m[C.te].sum(1) >= 4]
    qs = rng.choice(cand_q, N_Q, replace=False)
    out = []
    for c in qs[:n_q]:
        native, wn = C.native(c)
        t = int(rng.integers(len(native)))
        keep = np.array([k for k in range(len(native)) if k != t])
        c2, s2, qp, qw, qm = C.frame(native[keep], wn[keep])
        word = int(C.words_of(qp[None], qw[None], qm[None])[0])
        ids = C.pool_ids(word, rng)
        n_cand = int(sum(int(C.m[i].sum()) for i in ids))
        rng.integers(n_cand)                                   # the random pick
        out.append((int(c), float(np.linalg.norm(np.diff(native[t], axis=0), axis=1).sum())))
    return out


def strata_table(rows, arcs):
    arc = {q: a for q, a in arcs}
    assert all(int(r["q"]) in arc for r in rows), "question ids do not match the replay"
    res = {}
    for name, lo, hi in (("pooled", 0, 1e9),) + STRATA:
        sel = [r for r in rows if lo <= arc[int(r["q"])] < hi]
        res[name] = {"n": len(sel)}
        for k in METHODS:
            ch = np.array([float(r[f"{k}_chamfer"]) for r in sel]); ov = np.array([float(r[f"{k}_overlap"]) for r in sel])
            res[name][k] = {"chamfer_med": float(np.median(ch)), "w8": float((ch <= 8).mean()),
                            "w16": float((ch <= 16).mean()), "overlap": float(ov.mean())}
    return res


def print_strata(res):
    for name, d in res.items():
        print(f"\n[{name}] n={d['n']}")
        print(f"{'':<10}{'chamfer中央値':>14}{'8px以内':>10}{'16px以内':>10}{'既存インクに重なる':>18}")
        for k in METHODS:
            v = d[k]
            print(f"{k:<10}{v['chamfer_med']:>14.1f}{v['w8']:>10.1%}{v['w16']:>10.1%}{v['overlap']:>18.1%}")


def montage(examples, path):
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
        cv2.putText(img, f"pick {r['loc_chamfer']:.0f}px  oracle {r['oracle_chamfer']:.0f}px", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cells.append(img)
    cells += [np.full_like(cells[0], 255)] * (-len(cells) % 4)
    cv2.imwrite(str(path), np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]))


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--train-q", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--strata-only", action="store_true", help="stratify a finished run by masked-stroke length")
    a = p.parse_args()
    if a.strata_only:
        C = Corpus(torch.device("cuda"))
        rows = list(csv.DictReader(open(OUT / "per_question.csv")))
        res = strata_table(rows, reconstruct_arcs(C))
        print_strata(res)
        json.dump(res, open(OUT / "strata.json", "w"), indent=1)
        return
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    if a.smoke:
        a.train_q, a.epochs = 200, 1
        OUT = Path("/tmp/claude-1000/-home-sh1-deepl-lineart-stroke-grammar/9f0272e3-1647-4372-b9f2-b95970862d96/scratchpad/select_smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    C = Corpus(dev)
    m_loc = train(C, True, a.train_q, a.epochs, a.lr, a.workers, dev, "loc")
    m_noloc = train(C, False, a.train_q, a.epochs, a.lr, a.workers, dev, "noloc")
    m_loc.eval(); m_noloc.eval()
    rows, examples, mismatch = evaluate(C, {"loc": (m_loc, True), "noloc": (m_noloc, False)}, dev,
                                        20 if a.smoke else N_Q)
    with open(OUT / "per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {"mismatch_vs_option_b": mismatch}
    print(f"\nquestions {len(rows)}, differing from option B's set: {mismatch}")
    print(f"{'選び方':<10}{'chamfer中央値':>14}{'8px以内':>10}{'16px以内':>10}{'既存インクに重なる':>18}")
    for k in ("loc", "noloc", "cloze2", "endpoint", "random", "oracle"):
        ch = np.array([r[f"{k}_chamfer"] for r in rows]); ov = np.array([r[f"{k}_overlap"] for r in rows])
        summ[k] = {"chamfer_med": float(np.median(ch)), "w8": float((ch <= 8).mean()), "w16": float((ch <= 16).mean()),
                   "overlap": float(ov.mean())}
        print(f"{k:<10}{np.median(ch):>14.1f}{(ch<=8).mean():>10.1%}{(ch<=16).mean():>10.1%}{ov.mean():>18.1%}")
    summ["loc_over_random"] = summ["loc"]["chamfer_med"] / summ["random"]["chamfer_med"]
    summ["loc_over_endpoint"] = summ["loc"]["chamfer_med"] / summ["endpoint"]["chamfer_med"]
    summ["pass"] = bool(summ["loc_over_random"] <= 0.70 and summ["loc_over_endpoint"] <= 0.80)
    print(f"loc/random {summ['loc_over_random']:.3f} (bar <= 0.70)   loc/endpoint {summ['loc_over_endpoint']:.3f} (bar <= 0.80)   pass={summ['pass']}")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    montage(examples, OUT / "montage.png")
    print("montage ->", OUT / "montage.png", "(grey=remaining, green=truth, red=selector pick, blue=oracle pick)")


if __name__ == "__main__":
    main()
