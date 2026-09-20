#!/usr/bin/env python
"""Discover panel shot-types (composition) by k-means on per-panel features.

Features per panel (see work_log 2026-09-21 pre-registration):
  1. log(n_clusters), log(area), aspect ratio
  2. density = n_clusters / area
  3. plane: fraction of clusters whose centroid bin touches the panel border
  4. proportion: 12-bin histogram of cluster scale tokens
  5. word histogram over the top-100 train-frequent words
  6. WD14 tags: top-40 train-frequent tags as binary + tag count
All z-scored with train statistics. k-means for k in {8,12,16}.

Output: results/panel_composition_20260921/
  features.npy (n, d) float32, z-scored, train row order = corpus row order
  labels_k{K}.npy (n,) int32
  report_k{K}.txt  cluster sizes, top tags, top words, centroid feature profile
  montage_k{K}.png  rows = clusters (sorted by size), cols = 8 panels nearest to centroid
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PACK = "results/panel_pack_20260919"
TAGS = "results/panel_tags_20260918"
CORPUS = "results/grammar_corpus_20260920/corpus.npz"
OUT = Path("results/panel_composition_20260921")
TOP_WORDS, TOP_TAGS = 100, 40
KS = (8, 12, 16)
SEED = 0


def load_tags(train_names):
    """panel basename -> tag set; and train tag counts."""
    name2tags = {}
    cnt = Counter()
    with open(f"{TAGS}/tags.csv") as f:
        for r in csv.DictReader(f):
            base = Path(r["name"]).stem
            tags = set(t.strip() for t in r["tags"].split(",") if t.strip())
            name2tags[base] = tags
            if base in train_names:
                cnt.update(tags)
    return name2tags, cnt


def main():
    use_tags = "--no-tags" not in sys.argv
    out = OUT / "notags" if not use_tags else OUT
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(CORPUS)
    seq, split, pids, dims = z["seq"], z["split"], z["panels"], z["dims"]
    n = len(seq)
    panels_csv = list(csv.DictReader(open(f"{PACK}/panels.csv")))
    pid2name = {i: Path(p["name"]).stem for i, p in enumerate(panels_csv)}

    train_names = {pid2name[int(p)] for p, s in zip(pids, split) if s}
    name2tags, tag_cnt = load_tags(train_names)
    top_tags = [t for t, _ in tag_cnt.most_common(TOP_TAGS)] if use_tags else []

    # train word counts for top-word selection
    wc = Counter()
    for i in range(n):
        if not split[i]:
            continue
        w = seq[i, 1::4]
        w = w[(w >= 0) & (w < 500)]
        wc.update(w.tolist())
    top_words = [w for w, _ in wc.most_common(TOP_WORDS)]
    word_rank = {w: i for i, w in enumerate(top_words)}

    n_feat = 3 + 1 + 1 + 12 + TOP_WORDS + (TOP_TAGS + 1 if use_tags else 0)
    F = np.zeros((n, n_feat), np.float32)
    plain = np.zeros((n, 8), np.float32)  # raw values kept for the report
    for i in range(n):
        toks = seq[i][seq[i] >= 0]
        toks = toks[(toks != 556) & (toks != 557)]  # drop BOS/EOS
        fields = toks.reshape(-1, 4)
        w, py, px, sc = fields[:, 0], fields[:, 1] - 512, fields[:, 2] - 528, fields[:, 3] - 544
        H, W = dims[int(pids[i])]
        area = max(float(H) * float(W), 1.0)
        plane = float(np.mean((py == 0) | (py == 15) | (px == 0) | (px == 15))) if len(w) else 0.0
        sh = np.bincount(np.clip(sc, 0, 11), minlength=12) / max(len(w), 1)
        wh = np.zeros(TOP_WORDS, np.float32)
        for ww in w:
            r = word_rank.get(int(ww))
            if r is not None:
                wh[r] += 1
        wh /= max(len(w), 1)
        tg = name2tags.get(pid2name[int(pids[i])], set())
        tb = np.array([t in tg for t in top_tags], np.float32)
        F[i] = np.concatenate(
            [np.log1p([len(w), area]), [W / max(H, 1)],
             [len(w) / area * 1e6], [plane], sh, wh]
            + ([tb, [len(tg)]] if use_tags else []))
        plain[i] = [len(w), area, W / max(H, 1), len(w) / area * 1e6, plane, len(tg), H, W]

    tr = split.astype(bool)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Fz = (F - mu) / sd
    np.save(out / "features.npy", Fz.astype(np.float32))
    # tag cache for conditional experiments (corpus row order)
    TBM = np.zeros((n, len(top_tags)), np.float32)
    if top_tags:
        for i in range(n):
            tg = name2tags.get(pid2name[int(pids[i])], set())
            TBM[i] = [t in tg for t in top_tags]
    np.save(out / "tags_top40.npy", TBM)
    (out / "top_tags.json").write_text(json.dumps(top_tags))

    strokes = np.load(f"{PACK}/strokes.npy", mmap_mode="r")
    pcs = list(csv.DictReader(open(f"{PACK}/panels.csv")))

    for K in KS:
        lab, inertia = kmeans(Fz, K, seed=SEED)
        np.save(out / f"labels_k{K}.npy", lab)
        sizes = np.bincount(lab, minlength=K)
        order = np.argsort(-sizes)
        write_report(out / f"report_k{K}.txt", K, lab, order, sizes, Fz, plain,
                     seq, pids, pid2name, name2tags, top_words, top_tags, F, tag_cnt)
        render_montage(out / f"montage_k{K}.png", K, lab, order, Fz,
                       strokes, pcs, pids, dims)
        print(f"k={K}: sizes {sorted(sizes.tolist(), reverse=True)} "
              f"inertia/1e5 {inertia / 1e5:.1f}")


def kmeans(X, K, seed=0, iters=100, restarts=5):
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    for rs in range(restarts):
        # k-means++ init
        c = [X[rng.integers(len(X))]]
        for _ in range(K - 1):
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(c)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k]
                             for k in range(K)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best[1]:
            best = (lab.copy(), inertia)
    return best


def write_report(path, K, lab, order, sizes, Fz, plain, seq, pids, pid2name,
                 name2tags, top_words, top_tags, Fraw, tag_cnt):
    L = []
    L.append(f"k={K}  (clusters sorted by size)\n")
    for rank, k in enumerate(order):
        m = lab == k
        L.append(f"\n== cluster {k} (rank {rank}) n={m.sum()} ({m.mean() * 100:.1f}%) ==")
        med = np.median(plain[m][:, :5], axis=0)
        L.append(f"median: n_cl={med[0]:.0f} area={med[1]:.0f} aspect={med[2]:.2f} "
                 f"density={med[3]:.2f} plane={med[4]:.2f} "
                 f"n_tags={np.median(plain[m][:, 5]):.0f}")
        # distinctive features: centroid z vs global 0, top abs
        cz = Fz[m].mean(0)
        top = np.argsort(-np.abs(cz))[:12]
        names = (["log_n", "log_area", "aspect", "density", "plane"] +
                 [f"scale{j}" for j in range(12)] +
                 [f"w{top_words[j]}" for j in range(len(top_words))] +
                 [f"tag:{top_tags[j]}" for j in range(len(top_tags))] + ["n_tags"])
        L.append("centroid extremes: " + ", ".join(
            f"{names[j]}={cz[j]:+.2f}" for j in sorted(top, key=lambda j: -abs(cz[j]))))
        # top tags among members (raw prevalence)
        tc = Counter()
        for i in np.where(m)[0]:
            tc.update(name2tags.get(pid2name[int(pids[i])], set()))
        L.append("top member tags: " + ", ".join(
            f"{t}({c / m.sum():.0%})" for t, c in tc.most_common(10)))
    path.write_text("\n".join(L))


def render_panel(strokes, pcs, pid, H, W, cell=200):
    p = pcs[int(pid)]
    rows = np.asarray(strokes[int(p["start"]):int(p["start"]) + int(p["n"])])
    img = Image.new("L", (max(int(W), 2), max(int(H), 2)), 255)
    dr = ImageDraw.Draw(img)
    lw = max(1, int(round(max(H, W) / cell)))
    for r in rows:
        pts = [(float(x), float(y)) for x, y in r[:32].reshape(16, 2)
               if x != 0 or y != 0]
        if len(pts) >= 2:
            dr.line(pts, fill=0, width=lw)
    s = cell / max(img.size)
    return img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))),
                      Image.LANCZOS)


def render_montage(path, K, lab, order, Fz, strokes, pcs, pids, dims, cols=8):
    cell, pad = 200, 4
    rows_out = []
    for k in order:
        m = np.where(lab == k)[0]
        d = ((Fz[m] - Fz[m].mean(0)) ** 2).sum(1)
        rep = m[np.argsort(d)[:cols]]
        tiles = []
        for i in rep:
            H, W = dims[int(pids[i])]
            im = render_panel(strokes, pcs, pids[i], H, W, cell)
            canvas = Image.new("L", (cell, cell), 255)
            canvas.paste(im, ((cell - im.size[0]) // 2, (cell - im.size[1]) // 2))
            tiles.append(canvas)
        while len(tiles) < cols:
            tiles.append(Image.new("L", (cell, cell), 255))
        rows_out.append(tiles)
    grid = Image.new("L", (cols * (cell + pad) + pad, K * (cell + pad) + pad), 200)
    for r, tiles in enumerate(rows_out):
        for c, t in enumerate(tiles):
            grid.paste(t, (pad + c * (cell + pad), pad + r * (cell + pad)))
    grid.save(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
