#!/usr/bin/env python
"""C-3: do clusters of strokes recur across drawings? (the vocabulary test)

If "cluster = word" -- the user's hypothesis, echoed by an external review
(`outbox/KIMIからの意見.md`) -- similar clusters should appear again in OTHER
drawings, and that recurrence has to beat a null that keeps the strokes but
destroys their arrangement, because arrangement is what the claim is about.

Pre-registered (doc/work_log.md 2026-09-19, written before any result):
  MO-1  median d_NN(real) <= 0.60 x median d_NN(N2 layout-scramble), paired
        bootstrap 95% CI of the ratio below 1.0; AND >= 25% of held-out
        clusters have a cross-group neighbour with d_NN <= tau_close, where
        tau_close = 0.5 x the median within-cluster nearest-stroke gap in the
        same normalised units, measured on TRAIN clusters before any query.
  MO-2  visual: 30 seeds x 5 nearest cross-group neighbours, >= 10 of 30 rows
        nameable as the same thing (judged before the numbers table).
  Sanity: the same numbers with only the same panel excluded (no group
        exclusion); dramatically better means near-duplicate pages drive it.

Descriptor: centroid-translated, bbox long side scaled to 56, strokes drawn 1px
into 64x64, downsampled to 32x32, 64-dim randomized SVD, L2-normalised, kd-tree.
No rotation normalisation: manga panels share an upright frame, and an upright
hatch fan and a sideways one are arguably different words.

Phases (run in order): build -> montage -> stats. `montage` prints no
aggregate number, so the visual judgement can be recorded before `stats`.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

PACK = Path("results/panel_pack_20260919")
OUT = Path("results/motifs_20260919")
P = 16
SIDE = 56.0


# ---------------------------------------------------------------- data
def load_all():
    arr = np.load(PACK / "strokes.npy", mmap_mode="r")
    lab = np.load(PACK / "cluster_labels.npy")
    rows = list(csv.DictReader(open(PACK / "panels.csv")))
    works = {(r["source"], r["name"]): r["work"] for r in csv.DictReader(open(PACK / "panel_works.csv"))}
    return arr, lab, rows, works


def enumerate_clusters(lab, rows, works):
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(20260918); rng.shuffle(groups)
    test_g = set(groups[: int(len(groups) * 0.25)])
    gid = {g: i for i, g in enumerate(sorted({r["group"] for r in rows}))}
    wid = {w: i for i, w in enumerate(sorted(set(works.values())))}
    starts, sizes, panel, grp, wk, test = [], [], [], [], [], []
    members = []
    for pi, r in enumerate(rows):
        s, n = int(r["start"]), int(r["n"])
        l = lab[s:s + n]
        order = np.argsort(l, kind="stable")
        ls = l[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(ls)) + 1, len(ls)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if ls[a] < 0 or not (3 <= b - a <= 40):
                continue
            members.append((s + order[a:b]).astype(np.int64))
            panel.append(pi); grp.append(gid[r["group"]])
            wk.append(wid[works[(r["source"], r["name"])]]); test.append(r["group"] in test_g)
    return (members, np.array(panel), np.array(grp), np.array(wk), np.array(test),
            {v: k for k, v in wid.items()})


def norm_points(pts):
    """(m,16,2) absolute -> (m,16,2) normalised: centroid at 0, bbox long side 56."""
    flat = pts.reshape(-1, 2)
    c = flat.mean(0)
    span = max(float(np.ptp(flat[:, 0])), float(np.ptp(flat[:, 1])), 1e-3)
    return (pts - c) * (SIDE / span)


def render(npts):
    img = np.zeros((64, 64), np.uint8)
    for s in npts:
        xy = np.round(s[:, ::-1] + 32.0).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [xy], False, 255, 1)
    return cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


def cluster_pts(arr, idx):
    return np.asarray(arr[np.sort(idx)][:, :P * 2]).reshape(-1, P, 2)


_ARR = None


def _init():
    global _ARR
    _ARR = np.load(PACK / "strokes.npy", mmap_mode="r")


def _desc(idx_list):
    return np.stack([render(norm_points(cluster_pts(_ARR, idx))) for idx in idx_list])


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (tb.query(a)[0].mean() + ta.query(b)[0].mean())


# ---------------------------------------------------------------- build
def build(a):
    arr, lab, rows, works = load_all()
    members, panel, grp, wk, test, wnames = enumerate_clusters(lab, rows, works)
    print(f"clusters {len(members)}  held-out {int(test.sum())}", flush=True)
    t0 = time.time()
    chunks = [members[i:i + 2000] for i in range(0, len(members), 2000)]
    with Pool(a.workers, initializer=_init) as pool:
        D = np.concatenate(pool.map(_desc, chunks)).astype(np.float32)
    print(f"descriptors {D.shape}  {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    sub = D[rng.choice(len(D), min(50000, len(D)), replace=False)]
    mu = sub.mean(0)
    # randomized SVD (Halko et al.), numpy only
    X = sub - mu
    G = rng.standard_normal((X.shape[1], 64 + 16)).astype(np.float32)
    Y = X @ G
    for _ in range(2):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    V = Vt[:64].T
    Z = (D - mu) @ V
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    # tau_close from TRAIN clusters, before any query
    tr = np.flatnonzero(~test)
    gaps = []
    for ci in rng.choice(tr, min(20000, len(tr)), replace=False):
        npts = norm_points(cluster_pts(arr, members[ci]))
        per = []
        for k in range(len(npts)):
            other = np.concatenate([npts[j] for j in range(len(npts)) if j != k])
            per.append(cKDTree(other).query(npts[k])[0].min())
        gaps.append(np.median(per))
    tau = 0.5 * float(np.median(gaps))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "desc64.npy", Z.astype(np.float32))
    np.savez(OUT / "clusters.npz", panel=panel, grp=grp, wk=wk, test=test,
             lens=np.array([len(m) for m in members]),
             flat=np.concatenate(members), mu=mu, V=V)
    with open(OUT / "works.csv", "w") as f:
        for i in sorted(wnames):
            f.write(f"{i},{wnames[i]}\n")
    open(OUT / "tau_close.txt", "w").write(
        f"{tau}\nwithin-cluster nearest-stroke gap (normalised, median over 20k train clusters): "
        f"{np.median(gaps):.4f}; p25 {np.percentile(gaps,25):.4f}; p75 {np.percentile(gaps,75):.4f}\n")
    print(f"tau_close = {tau:.4f} normalised units (bbox long side = 56)", flush=True)


def load_built():
    z = np.load(OUT / "clusters.npz")
    lens = z["lens"]; offs = np.r_[0, np.cumsum(lens)]
    flat = z["flat"]                     # read once: z["flat"] re-decompresses on every access
    members = [flat[offs[i]:offs[i + 1]] for i in range(len(lens))]
    return (np.load(OUT / "desc64.npy"), members, z["panel"], z["grp"], z["wk"], z["test"],
            z["mu"], z["V"])


# ---------------------------------------------------------------- queries
_S = {}


def _init_q():
    _init()
    Z, members, panel, grp, wk, test, mu, V = load_built()
    _S.update(Z=Z, members=members, panel=panel, grp=grp, wk=wk, mu=mu, V=V, tree=cKDTree(Z))
    lens_arc = np.asarray(_ARR[:, P * 2 + 3])
    order = np.argsort(lens_arc)
    _S.update(arc_sorted=lens_arc[order], arc_order=order)
    pan_of_row = np.empty(len(lens_arc), np.int32)
    for i, m in enumerate(members):
        pan_of_row[m] = panel[i]
    _S["pan_of_row"] = pan_of_row


def embed(npts):
    d = render(npts)
    z = (d - _S["mu"]) @ _S["V"]
    return z / (np.linalg.norm(z) + 1e-9)


def best(qz, qn, q_grp, q_pan, mode, k=50, pool_k=800):
    """mode: 'cross' (exclude group+panel), 'panel' (exclude panel only),
    'within' / 'crosswork' (cross-group AND same / different work as q)."""
    _, idx = _S["tree"].query(qz, k=pool_k)
    grp, pan, wk = _S["grp"][idx], _S["panel"][idx], _S["wk"][idx]
    if mode == "panel":
        ok = pan != q_pan
    else:
        ok = (grp != q_grp) & (pan != q_pan)
        if mode == "within":
            ok &= wk == _S["q_wk"]
        elif mode == "crosswork":
            ok &= wk != _S["q_wk"]
    cand = idx[ok][:k]
    if len(cand) == 0:
        return np.nan, -1
    qflat = qn.reshape(-1, 2)
    ds = [chamfer(qflat, norm_points(cluster_pts(_ARR, _S["members"][c])).reshape(-1, 2)) for c in cand]
    j = int(np.argmin(ds))
    return float(ds[j]), int(cand[j])


def scramble_layout(pts, rng):
    flat = pts.reshape(-1, 2)
    lo, hi = flat.min(0), flat.max(0)
    out = []
    for s in pts:
        c = s.mean(0)
        out.append(s - c + rng.uniform(lo, hi))
    return np.stack(out)


def scramble_shape(idx, pts, rng):
    out = []
    for r, s in zip(np.sort(idx), pts):
        arc = float(_ARR[r, P * 2 + 3])
        lo = np.searchsorted(_S["arc_sorted"], 0.8 * arc); hi = np.searchsorted(_S["arc_sorted"], 1.2 * arc)
        for _ in range(20):
            cand = _S["arc_order"][rng.integers(lo, max(hi, lo + 1))]
            if _S["pan_of_row"][cand] != _S["pan_of_row"][r]:
                break
        cs = np.asarray(_ARR[cand, :P * 2]).reshape(P, 2)
        out.append(cs - cs.mean(0) + s.mean(0))
    return np.stack(out)


def _query(ci):
    rng = np.random.default_rng(1000 + ci)
    idx = _S["members"][ci]
    pts = cluster_pts(_ARR, idx)
    qn = norm_points(pts)
    qg, qp = _S["grp"][ci], _S["panel"][ci]
    _S["q_wk"] = _S["wk"][ci]
    qz = embed(qn)
    real, _ = best(qz, qn, qg, qp, "cross")
    n2p = norm_points(scramble_layout(pts, rng)); n2, _ = best(embed(n2p), n2p, qg, qp, "cross")
    n1p = norm_points(scramble_shape(idx, pts, rng)); n1, _ = best(embed(n1p), n1p, qg, qp, "cross")
    san, _ = best(qz, qn, qg, qp, "panel")
    wi, _ = best(qz, qn, qg, qp, "within")
    cw, _ = best(qz, qn, qg, qp, "crosswork")
    return ci, real, n2, n1, san, wi, cw, len(idx)


# ---------------------------------------------------------------- montage
def montage(a):
    _init_q()
    Z, members, panel, grp, wk, test, _, _ = load_built()
    rows = list(csv.DictReader(open(PACK / "panels.csv")))
    wn = dict(line.strip().split(",", 1) for line in open(OUT / "works.csv"))
    held = np.flatnonzero(test)
    rng = np.random.default_rng(a.seed)
    seeds = rng.choice(held, 30, replace=False)
    C = 104
    out_rows = []
    for ci in seeds:
        idx = members[ci]; pts = cluster_pts(_ARR, idx); qn = norm_points(pts)
        _S["q_wk"] = wk[ci]
        qz = embed(qn)
        _, cand = _S["tree"].query(qz, k=800)
        ok = (grp[cand] != grp[ci]) & (panel[cand] != panel[ci])
        cand = cand[ok][:50]
        ds = [chamfer(qn.reshape(-1, 2), norm_points(cluster_pts(_ARR, members[c])).reshape(-1, 2)) for c in cand]
        top = [cand[j] for j in np.argsort(ds)[:5]]
        cells = []
        for k, c in enumerate([ci] + top):
            npts = norm_points(cluster_pts(_ARR, members[c]))
            img = np.full((C, C, 3), 255, np.uint8)
            for s in npts:
                xy = np.round((s[:, ::-1] + 32.0) * (C / 64.0)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [xy], False, (20, 20, 20) if k else (0, 0, 190), 1, cv2.LINE_AA)
            lab = np.full((26, C, 3), 255, np.uint8)
            r = rows[panel[c]]
            cv2.putText(lab, wn[str(wk[c])][:17], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(lab, r["name"][5:22], (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
            cells.append(np.vstack([img, lab]))
        row = np.hstack([np.hstack([c, np.full((c.shape[0], 4, 3), 150, np.uint8)]) for c in cells])
        out_rows.append(np.vstack([row, np.full((4, row.shape[1], 3), 150, np.uint8)]))
    # two columns of 15 rows so the image stays viewable
    left = np.vstack(out_rows[:15]); right = np.vstack(out_rows[15:])
    img = np.hstack([left, np.full((left.shape[0], 16, 3), 255, np.uint8), right])
    cv2.imwrite(str(OUT / "montage_motifs.png"), img)
    print("->", OUT / "montage_motifs.png", "(seed in red, then 5 nearest cross-group neighbours)")


# ---------------------------------------------------------------- stats
def stats(a):
    Z, members, panel, grp, wk, test, _, _ = load_built()
    held = np.flatnonzero(test)
    rng = np.random.default_rng(a.seed + 1)
    q = rng.choice(held, min(a.queries, len(held)), replace=False)
    t0 = time.time()
    with Pool(a.workers, initializer=_init_q) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_query, q, chunksize=8), 1):
            res.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(q)} {time.time()-t0:.0f}s", flush=True)
    R = np.array([r[1:] for r in res], float)
    cols = ["real", "N2_layout", "N1_shape", "sanity_no_group_excl", "within_work", "cross_work", "size"]
    with open(OUT / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cluster"] + cols)
        for r in res:
            w.writerow(list(r))
    tau = float(open(OUT / "tau_close.txt").readline())
    med = lambda v: float(np.nanmedian(v))
    close = lambda v: float(np.nanmean(v <= tau))
    print(f"\nqueries {len(R)}  tau_close {tau:.4f}  ({time.time()-t0:.0f}s)")
    print(f"{'':<24}{'median d_NN':>13}{'<= tau_close':>14}")
    for j, c in enumerate(cols[:6]):
        print(f"{c:<24}{med(R[:, j]):>13.3f}{close(R[:, j]):>14.1%}")
    ok = ~np.isnan(R[:, 0]) & ~np.isnan(R[:, 1]) & ~np.isnan(R[:, 2])
    real, n2, n1 = R[ok, 0], R[ok, 1], R[ok, 2]
    b = np.random.default_rng(7)
    boot2, boot1 = [], []
    for _ in range(1000):
        s = b.integers(0, len(real), len(real))
        boot2.append(np.median(real[s]) / np.median(n2[s]))
        boot1.append(np.median(real[s]) / np.median(n1[s]))
    r2 = np.median(real) / np.median(n2); r1 = np.median(real) / np.median(n1)
    print(f"\nratio real/N2 {r2:.3f}  95% CI [{np.percentile(boot2,2.5):.3f}, {np.percentile(boot2,97.5):.3f}]")
    print(f"ratio real/N1 {r1:.3f}  95% CI [{np.percentile(boot1,2.5):.3f}, {np.percentile(boot1,97.5):.3f}]")
    mo1 = (r2 <= 0.60) and (np.percentile(boot2, 97.5) < 1.0) and (close(R[:, 0]) >= 0.25)
    print(f"MO-1: {'PASS' if mo1 else 'FAIL'}  (ratio <= 0.60: {r2 <= 0.60}; CI < 1: "
          f"{np.percentile(boot2,97.5) < 1.0}; close >= 25%: {close(R[:, 0]) >= 0.25})")
    # by cluster size
    for lo, hi in [(3, 5), (6, 10), (11, 40)]:
        s = ok & (R[:, 6] >= lo) & (R[:, 6] <= hi)
        if s.sum() > 20:
            print(f"size {lo}-{hi}: n {int(s.sum())}  real {med(R[s,0]):.3f}  N2 {med(R[s,1]):.3f}  "
                  f"ratio {med(R[s,0])/med(R[s,1]):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["build", "montage", "stats"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--queries", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args()
    {"build": build, "montage": montage, "stats": stats}[a.phase](a)


if __name__ == "__main__":
    main()
