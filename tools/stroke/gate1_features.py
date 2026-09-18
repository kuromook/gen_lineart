#!/usr/bin/env python
"""Features for Track F gate 1: does this stroke belong in this drawing?

One row per candidate. A candidate replaces a held-out real token; the context
is every OTHER token of the same drawing. Features are split into two tiers so
they can be ablated:

  a_*  the stroke by itself   (length, straightness, curvature, width, darkness)
  b_*  its relations to the context (endpoint and body distances, the angle it
       meets its nearest neighbour at, how many near-parallel strokes sit beside
       it and how far away, local density)

The claim this track rests on is that b_* carries the signal. If a model with
a_* alone does as well, the feature design is wrong -- that ablation is the
point of the split, so keep the prefixes.

Gate 1 is judged on DISPLACED negatives specifically (rotated and foreign are
visibly easier); per-type AUC is always reported, never a pooled number.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from make_negatives import build_candidates, comps, order_path, place  # noqa: E402
from stroke_churn import dist_to, ink_skeleton, load_gray  # noqa: E402

KINDS = ("true", "displaced", "rotated", "foreign")


def direction(pts):
    """Principal direction as an angle in [0, pi) -- strokes have no head/tail."""
    c = pts - pts.mean(0)
    if len(c) < 2:
        return 0.0
    w, v = np.linalg.eigh(c.T @ c)
    d = v[:, int(w.argmax())]
    return float(np.arctan2(d[0], d[1]) % np.pi)


def ang_diff(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def curvature(path):
    if len(path) < 7:
        return 0.0
    step = max(2, len(path) // 12)
    p = path[::step].astype(float)
    if len(p) < 3:
        return 0.0
    v = np.diff(p, axis=0)
    a = np.arctan2(v[:, 0], v[:, 1])
    return float(np.mean(np.abs(np.diff(np.unwrap(a)))))



def terminal_tangent(path, k=6):
    """Direction the stroke is heading at each end, as (start_dir, end_dir)."""
    if len(path) < 3:
        return np.zeros(2), np.zeros(2)
    k = min(k, len(path) - 1)
    a = path[0] - path[k]
    b = path[-1] - path[-1 - k]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return (a / na if na else a), (b / nb if nb else b)


def unit_angle(u, v):
    """Unsigned angle between two direction vectors, in [0, pi/2] (strokes are undirected)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.pi / 2
    c = float(np.clip(abs(np.dot(u / nu, v / nv)), 0, 1))
    return float(np.arccos(c))


def lattice_residual(offs, spacing_min=2.0):
    """Do the parallel neighbours form a regular ladder, and does 0 sit on a rung?

    `offs` are the neighbours' perpendicular offsets from THIS stroke's line, so
    this stroke sits at 0. The lattice is fitted from the neighbours alone; the
    residual then says whether this stroke lies where the hatching says it
    should. A real hatch line does; one shifted a few px does not.
    Returns (residual in [0, 0.5], spacing cv, spacing).
    """
    if len(offs) < 2:
        return 99.0, 99.0, 99.0
    o = np.sort(np.asarray(offs, float))
    gaps = np.diff(o)
    gaps = gaps[gaps > spacing_min]
    if len(gaps) == 0:
        return 99.0, 99.0, 99.0
    sp = float(np.median(gaps))
    if sp <= spacing_min:
        return 99.0, 99.0, 99.0
    cv = float(np.std(gaps) / sp) if len(gaps) > 1 else 0.0
    phase = float(np.median(o % sp))
    r = (-phase) % sp
    return float(min(r, sp - r) / sp), cv, sp


def features(pts, wmed, wstd, wfill, ctx_dist, ctx_dirs, ctx_cent, ctx_mask, ctx_ends, ctx_tans, ctx_pix, shape=(480, 480)):
    path = order_path(pts)
    span = float(np.linalg.norm(path[0] - path[-1])) if len(path) > 1 else 0.0
    d_body = ctx_dist[pts[:, 0], pts[:, 1]]
    ends = np.array([path[0], path[-1]]) if len(path) > 1 else pts[:1]
    d_ends = np.sort(ctx_dist[ends[:, 0], ends[:, 1]])
    me = direction(pts)
    cen = pts.mean(0)

    f = {
        "a_len": len(pts),
        "a_straight": round(span / len(pts), 4),
        "a_curv": round(curvature(path), 4),
        "a_width": round(wmed, 3),
        "a_width_var": round(wstd, 3),
        "a_fill": round(wfill, 4),
    }
    if len(ctx_cent):
        dc = np.linalg.norm(ctx_cent - cen, axis=1)
        ad = ang_diff(ctx_dirs, me)
        near = dc < 40
        par = near & (ad < np.deg2rad(15))
        f.update({
            "b_end_near": round(float(d_ends[0]), 2),
            "b_end_far": round(float(d_ends[-1]), 2),
            "b_body_med": round(float(np.median(d_body)), 2),
            "b_body_min": round(float(d_body.min()), 2),
            "b_ang_nearest": round(float(ad[int(dc.argmin())]), 4),
            "b_n_parallel": int(par.sum()),
            "b_par_spacing": round(float(np.median(dc[par])), 2) if par.any() else 99.0,
            "b_n_near": int(near.sum()),
        })
    else:
        f.update({k: 99.0 for k in ("b_end_near", "b_end_far", "b_body_med", "b_body_min",
                                    "b_ang_nearest", "b_par_spacing")})
        f.update({"b_n_parallel": 0, "b_n_near": 0})
    # ---- tier C: sharper relations (2026-09-17, after gate 1 failed at 0.636) ----
    t0, t1 = terminal_tangent(path if len(path) > 2 else pts.astype(float))
    my_ends = np.array([path[0], path[-1]]) if len(path) > 1 else pts[:1].astype(float)
    my_tans = [t0, t1]
    if len(ctx_ends):
        gaps, conts, aligns = [], [], []
        for e, t in zip(my_ends, my_tans):
            d = np.linalg.norm(ctx_ends - e, axis=1)
            j = int(d.argmin())
            gaps.append(float(d[j]))
            to = ctx_ends[j] - e
            conts.append(unit_angle(t, to) if np.linalg.norm(to) > 0 else np.pi / 2)
            aligns.append(unit_angle(t, ctx_tans[j]))
        o = np.argsort(gaps)
        f["c_end_gap_near"] = round(gaps[o[0]], 2)
        f["c_end_gap_far"] = round(gaps[o[-1]], 2)
        f["c_end_point_at"] = round(conts[o[0]], 4)
        f["c_end_collinear"] = round(aligns[o[0]], 4)
    else:
        f.update({"c_end_gap_near": 99.0, "c_end_gap_far": 99.0,
                  "c_end_point_at": 99.0, "c_end_collinear": 99.0})

    if len(ctx_cent):
        nrm = np.array([np.cos(me), -np.sin(me)])          # normal to my direction
        dc = np.linalg.norm(ctx_cent - cen, axis=1)
        ad = ang_diff(ctx_dirs, me)
        fam = (dc < 60) & (ad < np.deg2rad(12))
        offs = ((ctx_cent - cen) @ nrm)[fam]
        res, cv, sp = lattice_residual(offs)
        f["c_hatch_n"] = int(fam.sum())
        f["c_hatch_residual"] = round(res, 4)
        f["c_hatch_cv"] = round(cv, 4)
        f["c_hatch_spacing"] = round(sp, 2)
        f["c_d2"] = round(float(np.sort(dc)[1]), 2) if len(dc) > 1 else 99.0
        f["c_d3"] = round(float(np.sort(dc)[2]), 2) if len(dc) > 2 else 99.0
        f["c_ang2"] = round(float(ad[int(np.argsort(dc)[1])]), 4) if len(dc) > 1 else 99.0
    else:
        f.update({"c_hatch_n": 0, "c_hatch_residual": 99.0, "c_hatch_cv": 99.0,
                  "c_hatch_spacing": 99.0, "c_d2": 99.0, "c_d3": 99.0, "c_ang2": 99.0})

    y0, y1 = max(0, int(cen[0]) - 40), min(shape[0], int(cen[0]) + 40)
    x0, x1 = max(0, int(cen[1]) - 40), min(shape[1], int(cen[1]) + 40)
    f["b_density"] = round(float(ctx_mask[y0:y1, x0:x1].mean()), 5)
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = "/home/sh1/deepl/lineart-controlnet-sd15-refine/data"
    p.add_argument("--line-dir", default=f"{data}/line")
    p.add_argument("--list", default=f"{data}/train_list.txt")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--output", default="results/gate1_20260917/candidates.csv")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    names = [l.strip() for l in open(a.list) if l.strip()]
    names = [names[i] for i in rng.choice(len(names), min(a.limit, len(names)), replace=False)]

    tiles = {}
    for n in names:
        g = load_gray(Path(a.line_dir) / n)
        sk = ink_skeleton(g)
        if not sk.any():
            continue
        cs = comps(sk, 8)
        if cs:
            tiles[n] = (g, sk, cs)
    pool_pw = []
    for _n, (_g, _sk, _cs) in tiles.items():
        _d = cv2.distanceTransform((_g < 128).astype(np.uint8), cv2.DIST_L2, 5)
        pool_pw += [(c, _d[c[:, 0], c[:, 1]] * 2.0) for c in _cs if len(c) >= a.min_token_px]
    print(f"tiles {len(tiles)}  pool {len(pool_pw)}", flush=True)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (n, (g, sk, cs)) in enumerate(tiles.items()):
        ink_dist = cv2.distanceTransform((g < 128).astype(np.uint8), cv2.DIST_L2, 5)
        for ti, c in enumerate(c for c in cs if len(c) >= a.min_token_px):
            others = [o for o in cs if o is not c]
            if not others:
                continue
            ctx = np.zeros_like(sk)
            for o in others:
                ctx[o[:, 0], o[:, 1]] = True
            cd = dist_to(ctx)
            dirs = np.array([direction(o) for o in others])
            cent = np.array([o.mean(0) for o in others])
            opaths = [order_path(o).astype(float) for o in others]
            ends, tans = [], []
            for q in opaths:
                if len(q) < 2:
                    continue
                ta, tb = terminal_tangent(q)
                ends += [q[0], q[-1]]
                tans += [ta, tb]
            ends = np.array(ends) if ends else np.zeros((0, 2))
            tans = np.array(tans) if tans else np.zeros((0, 2))
            cands = build_candidates(c, pool_pw, rng, sk.shape, ink_dist)
            # matched sets only: rejecting out-of-frame candidates one at a time
            # would leave the negatives systematically shorter and more central
            # than the positives, which is a leak of its own.
            if any(v is None for v in cands.values()):
                continue
            for kind, cand in cands.items():
                pts, wmed, wstd, wfill = cand
                f = features(pts, wmed, wstd, wfill, cd, dirs, cent, ctx, ends, tans, others)
                f.update({"tile": n, "token": ti, "kind": kind, "label": int(kind == "true"),
                          "on_other": round(float((cd[pts[:, 0], pts[:, 1]] <= 3).mean()), 4)})
                rows.append(f)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tiles)}  rows {len(rows)}", file=sys.stderr, flush=True)

    cols = ["tile", "token", "kind", "label", "on_other"] + [k for k in rows[0] if k[:2] in ("a_", "b_", "c_")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"rows {len(rows)}  -> {out}")
    for k in KINDS:
        print(f"  {k:<11}{sum(1 for r in rows if r['kind']==k):>8}")


if __name__ == "__main__":
    main()
