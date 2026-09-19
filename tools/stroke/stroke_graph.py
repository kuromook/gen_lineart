#!/usr/bin/env python
"""Group a panel's strokes into clusters -- the unit above the stroke.

Why (2026-09-19, user): "one stroke is missing, draw it" has no scope unless the
meaningful unit is known. Asked over a whole panel the question is ill-posed, so
the first attempt supplied an artificial scope -- a 64px cell -- which then set
the metric's floor (19.8px against strokes of median arc 51px). The unit has to
come from the drawing, not from a grid.

Edges, all from the 16-point polylines, computed once per panel over a KD-tree:

  prox  the two strokes come within tau px of each other, where tau follows
        `--tau-rule` (see TAU_RULES below)
  end   an endpoint of one lies within tau of an endpoint of the other AND the
        two terminal tangents continue each other (angle <= 40 deg)
  par   near-parallel (<= 20 deg), projected overlap >= 50%, separation between
        2x and 6x the mean width -- this is hatching and hair
  cross the polylines actually intersect

Weights are fixed here and NOT tuned; degree is capped at the 12 strongest edges
per stroke so that a community is not just a dense patch of ink.

TAU_RULES (2026-09-19): the first rule keyed tau to ink WIDTH, which left 61-66%
of strokes as singletons even though half of those had another stroke within
20px -- elements that group at a distance (hair, hatching, the parts of an eye)
were never joined. Three rules are compared, and only three, because the way to
ruin this is to tune until the clusters "look like eyes":

  width  max(6, 1.5*(w_i+w_j))            -- the original, ink thickness
  arc    max(6, 0.35*min(arc_i, arc_j))   -- the scale of the strokes themselves
  knn    clip(min(t_i, t_j), 6, 120) where t_i is the point-to-point distance
         from stroke i to its 4th nearest OTHER stroke -- local crowding

Candidate pairs now come from a centroid pre-filter plus exact 16x16 point
distances. The earlier version queried a single radius built from the MEDIAN ink
width, which silently dropped pairs involving wide strokes, so `width` numbers
here differ slightly from the 2026-09-19 sweep.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import POINTS  # noqa: E402

W_PROX, W_END, W_PAR, W_CROSS = 1.0, 1.5, 1.5, 0.5
MAX_DEGREE = 12


def load_panel(pack_arr, start, n):
    block = np.asarray(pack_arr[start:start + n])
    pts = block[:, :POINTS * 2].reshape(n, POINTS, 2)
    meta = block[:, POINTS * 2:]
    return pts, meta


def seg_dirs(pts):
    v = pts[:, -1] - pts[:, 0]
    nrm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return v / nrm


TAU_RULES = ("width", "arc", "knn")
TAU_MIN, TAU_CLIP = 6.0, 120.0


def _pair_dmin(pts, pairs, chunk=20000):
    """exact min point-to-point distance for each candidate pair"""
    out = np.empty(len(pairs), np.float32)
    for s in range(0, len(pairs), chunk):
        e = min(s + chunk, len(pairs))
        a = pts[pairs[s:e, 0]]                       # (m, POINTS, 2)
        b = pts[pairs[s:e, 1]]
        d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
        out[s:e] = d.reshape(len(a), -1).min(1)
    return out


def _knn_tau(pts, k=4):
    """distance from each stroke to its k-th nearest OTHER stroke"""
    n = len(pts)
    flat = pts.reshape(-1, 2)
    owner = np.repeat(np.arange(n), POINTS)
    tree = cKDTree(flat)
    tau = np.full(n, TAU_CLIP, np.float32)
    want = min(len(flat), 16 * (k + 6) + 32)
    dd, ii = tree.query(pts.reshape(-1, 2), k=want, workers=1)
    dd = dd.reshape(n, POINTS * want)
    oo = owner[ii].reshape(n, POINTS * want)
    for i in range(n):
        m = oo[i] != i
        if not m.any():
            continue
        o, d = oo[i][m], dd[i][m]
        order = np.argsort(d)
        o, d = o[order], d[order]
        _, first = np.unique(o, return_index=True)
        best = np.sort(d[np.sort(first)])
        if len(best) >= k:
            tau[i] = best[k - 1]
        elif len(best):
            tau[i] = best[-1]
    return np.clip(tau, TAU_MIN, TAU_CLIP)


def stroke_radius(pts, meta, rule, knn_k=4):
    """per-stroke upper bound on tau, so candidate generation cannot miss a pair"""
    w, arc = meta[:, 1], meta[:, 3]
    if rule == "width":
        return np.maximum(TAU_MIN, 1.5 * (w + float(w.max())))
    if rule == "arc":
        return np.clip(np.maximum(TAU_MIN, 0.35 * arc), TAU_MIN, TAU_CLIP)
    if rule == "knn":
        return _knn_tau(pts, knn_k)
    raise ValueError(rule)


def build_edges(pts, meta, rule="width", knn_k=4, max_pairs=400000, use_end=True):
    """-> dict {(i,j): weight} with i<j.

    use_end=False drops the endpoint-continuation edge (2026-09-19): the cloze
    test scores candidates by endpoint contact, and a cluster built FROM that
    contact makes the test partly circular."""
    n = len(pts)
    if n < 2:
        return {}
    w, arc = meta[:, 1], meta[:, 3]
    r_self = stroke_radius(pts, meta, rule, knn_k)
    cent = pts.mean(1)
    rad = np.linalg.norm(pts - cent[:, None, :], axis=2).max(1)
    # candidate pairs: centroids close enough that the polylines could touch
    reach = rad + r_self
    tree_c = cKDTree(cent)
    cand = tree_c.query_pairs(float((reach).max() + rad.max()), output_type="ndarray")
    if len(cand) == 0:
        return {}
    dc = np.linalg.norm(cent[cand[:, 0]] - cent[cand[:, 1]], axis=1)
    slack = rad[cand[:, 0]] + rad[cand[:, 1]] + np.minimum(r_self[cand[:, 0]], r_self[cand[:, 1]])
    cand = cand[dc <= slack]
    if len(cand) == 0:
        return {}
    if len(cand) > max_pairs:                      # safety valve, reported if hit
        keep = np.argsort(np.linalg.norm(cent[cand[:, 0]] - cent[cand[:, 1]], axis=1))[:max_pairs]
        cand = cand[keep]
    dmin = _pair_dmin(pts, cand)
    if rule == "width":
        tau = np.maximum(TAU_MIN, 1.5 * (w[cand[:, 0]] + w[cand[:, 1]]))
    elif rule == "arc":
        tau = np.clip(np.maximum(TAU_MIN, 0.35 * np.minimum(arc[cand[:, 0]], arc[cand[:, 1]])),
                      TAU_MIN, TAU_CLIP)
    else:
        tau = np.clip(np.minimum(r_self[cand[:, 0]], r_self[cand[:, 1]]), TAU_MIN, TAU_CLIP)
    ok = dmin <= tau
    cand, dmin, tau = cand[ok], dmin[ok], tau[ok]
    if len(cand) == 0:
        return {}
    dirs = seg_dirs(pts)
    ends = np.stack([pts[:, 0], pts[:, -1]], 1)
    tans = np.stack([pts[:, 1] - pts[:, 0], pts[:, -2] - pts[:, -1]], 1)
    tans /= (np.linalg.norm(tans, axis=2, keepdims=True) + 1e-6)
    edges = {}
    for (i, j), dm, tu in zip(cand, dmin, tau):
        i, j = int(i), int(j)
        wt = W_PROX * float(np.exp(-dm / tu))
        de = np.linalg.norm(ends[i][:, None, :] - ends[j][None, :, :], axis=-1)
        ei, ej = np.unravel_index(int(de.argmin()), de.shape)
        if use_end and de[ei, ej] <= tu:
            cont = float(np.dot(tans[i, ei], -tans[j, ej]))
            if cont >= np.cos(np.deg2rad(40)):
                wt += W_END
        ang = float(np.arccos(np.clip(abs(np.dot(dirs[i], dirs[j])), 0, 1)))
        if ang <= np.deg2rad(20):
            nrm = np.array([-dirs[i][1], dirs[i][0]])
            sep = abs(float((pts[j].mean(0) - pts[i].mean(0)) @ nrm))
            mw = 0.5 * (w[i] + w[j]) + 1e-6
            proj_i = pts[i] @ dirs[i]; proj_j = pts[j] @ dirs[i]
            ov = min(proj_i.max(), proj_j.max()) - max(proj_i.min(), proj_j.min())
            span = min(proj_i.ptp(), proj_j.ptp()) + 1e-6
            if 2 * mw <= sep <= 6 * mw and ov / span >= 0.5:
                wt += W_PAR
        if dm <= 1.5:
            wt += W_CROSS
        edges[(i, j)] = wt
    return edges


def cap_degree(edges, n, k=MAX_DEGREE):
    per = {}
    for (i, j), w in edges.items():
        per.setdefault(i, []).append((w, j))
        per.setdefault(j, []).append((w, i))
    keep = set()
    for i, lst in per.items():
        for w, j in sorted(lst, reverse=True)[:k]:
            keep.add((min(i, j), max(i, j)))
    return {e: w for e, w in edges.items() if e in keep}


def communities(edges, n, resolution=1.0, seed=0):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        g.add_edge(i, j, w=w)
    parts = nx.community.louvain_communities(g, weight="w", resolution=resolution, seed=seed)
    lab = np.full(n, -1, np.int32)
    for k, c in enumerate(parts):
        for i in c:
            lab[i] = k
    return lab


def ari(a, b):
    """adjusted Rand index, from a contingency table (no sklearn here).

    Caveat that matters here: a partition which leaves most strokes alone gets
    much of its ARI for free -- singletons stay singletons under edge dropout,
    and "every stroke its own cluster" scores a perfect 1.0 against itself. So
    `panel_clusters` reports ARI twice: over all strokes, and over only those
    the reference partition put in a cluster of 2 or more.
    """
    from scipy.special import comb
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    m = np.zeros((len(ua), len(ub)), np.int64)
    np.add.at(m, (ia, ib), 1)
    sij = comb(m, 2).sum()
    sa = comb(m.sum(1), 2).sum(); sb = comb(m.sum(0), 2).sum()
    n = comb(len(a), 2)
    exp = sa * sb / n
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def panel_clusters(pts, meta, resolution=1.0, dropout_runs=0, seed=0, rule="width"):
    n = len(pts)
    edges = cap_degree(build_edges(pts, meta, rule), n)
    lab = communities(edges, n, resolution, seed)
    out = {"n": n, "edges": len(edges), "labels": lab,
           "sizes": np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(0)}
    if dropout_runs and edges:
        rng = np.random.default_rng(seed + 1)
        keys = list(edges)
        sizes = out["sizes"]
        grouped = np.array([lab[i] >= 0 and sizes[lab[i]] >= 2 for i in range(n)])
        scores, scores_g = [], []
        for r in range(dropout_runs):
            sub = {k: edges[k] for k in
                   (keys[i] for i in rng.choice(len(keys), int(len(keys) * 0.9), replace=False))}
            lab2 = communities(sub, n, resolution, seed)
            scores.append(ari(lab, lab2))
            if grouped.sum() >= 2:
                scores_g.append(ari(lab[grouped], lab2[grouped]))
        out["ari"] = float(np.median(scores))
        out["ari_grouped"] = float(np.median(scores_g)) if scores_g else float("nan")
    return out


def work(t):
    pack, row, resolution, dropout, rule = t
    arr = np.load(Path(pack) / "strokes.npy", mmap_mode="r")
    pts, meta = load_panel(arr, int(row["start"]), int(row["n"]))
    t0 = time.time()
    c = panel_clusters(pts, meta, resolution, dropout, rule=rule)
    sizes = c["sizes"]
    return {"source": row["source"], "name": row["name"], "group": row["group"],
            "rule": rule, "strokes": c["n"], "edges": c["edges"], "clusters": int(len(sizes)),
            "size_med": float(np.median(sizes)) if len(sizes) else 0.0,
            "size_p90": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "singletons": float((sizes == 1).mean()) if len(sizes) else 0.0,
            "in_ge3": float(sizes[sizes >= 3].sum() / max(c["n"], 1)) if len(sizes) else 0.0,
            "ari": round(c.get("ari", float("nan")), 4),
            "ari_grouped": round(c.get("ari_grouped", float("nan")), 4),
            "sec": round(time.time() - t0, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--out", default="results/stroke_clusters_20260919")
    p.add_argument("--resolutions", default="0.5,1.0,2.0")
    p.add_argument("--panels", type=int, default=50)
    p.add_argument("--dropout-runs", type=int, default=6)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--tau-rule", default="width", choices=list(TAU_RULES) + ["all"])
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(Path(a.pack) / "panels.csv")) if int(r["n"]) >= 40]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(20260918); rng.shuffle(groups)
    test = set(groups[: int(len(groups) * 0.25)])
    dev = [r for r in rows if r["group"] not in test]
    rng2 = np.random.default_rng(a.seed)
    dev = [dev[i] for i in rng2.choice(len(dev), min(a.panels, len(dev)), replace=False)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"dev panels {len(dev)} (train groups only)", flush=True)
    allrows = []
    rules = list(TAU_RULES) if a.tau_rule == "all" else [a.tau_rule]
    for rule in rules:
        for res in [float(x) for x in a.resolutions.split(",")]:
            t0 = time.time()
            with Pool(a.workers) as pool:
                got = list(pool.imap_unordered(
                    work, [(a.pack, r, res, a.dropout_runs, rule) for r in dev]))
            for g in got:
                g["resolution"] = res
            allrows += got
            med = lambda k: float(np.nanmedian([g[k] for g in got]))
            print(f"{rule:<6} res {res:<5} クラスター/コマ {med('clusters'):>6.0f}  "
                  f"大きさ中央値 {med('size_med'):>4.1f}  p90 {med('size_p90'):>5.1f}  "
                  f"単独の割合 {med('singletons'):>5.1%}  3本以上に属す線 {med('in_ge3'):>5.1%}  "
                  f"ARI(全) {med('ari'):>5.3f}  ARI(塊のみ) {med('ari_grouped'):>5.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    with open(out / "resolution_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print(f"-> {out}/resolution_sweep.csv")


if __name__ == "__main__":
    main()
