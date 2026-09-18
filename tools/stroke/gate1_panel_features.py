#!/usr/bin/env python
"""Gate 1 features on the PANEL corpus, with linked strokes as the unit.

Same question and same feature set as gate1_features.py (tier a: the stroke
alone, b: coarse relations, c: sharp relations), but the drawing is a whole
panel at native resolution and a token is a linked stroke rather than a
junction-to-junction span. Context per drawing is ~292 strokes instead of ~27.

`housei` is excluded: its panels are tone and wash, not line art (capture 0.783
against 0.929 elsewhere), and it is a different task by the standing pool rule.

One row per candidate; only matched sets (true + all three negatives) are kept.
Gate 1 is judged on DISPLACED alone.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402
from make_negatives import build_candidates, order_path  # noqa: E402
from gate1_features import KINDS, direction, features, terminal_tangent  # noqa: E402
from stroke_churn import dist_to  # noqa: E402

ROOT = Path("/home/sh1/deepl/lineart")
EXCLUDE_SOURCES = ("regions_housei_koma_panels_20260729",)


def panel_strokes(line_path, skel_path, min_px, fill_max):
    g = cv2.imread(str(line_path), 0)
    sk = cv2.imread(str(skel_path), 0)
    if g is None or sk is None:
        return None
    ink = ink_mask(g)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, _ = strokes_from_skeleton(sk > 0, dist * 2)
    groups = {}
    for i, (r, c) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    out = []
    for k, members in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in members])
        cc = np.concatenate([spans[i][1] for i in members])
        if len(rr) < min_px:
            continue
        w = dist[rr, cc] * 2.0
        if float((w > 8).mean()) > fill_max:
            continue
        out.append(np.stack([rr, cc], 1))
    return g, sk > 0, dist, out


def pool_task(t):
    src, name, line_path, skel_path, min_px, fill_max, take, seed = t
    r = panel_strokes(line_path, skel_path, min_px, fill_max)
    if r is None or len(r[3]) < 2:
        return []
    _g, _sk, dist, cs = r
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cs), min(take, len(cs)), replace=False)
    return [(cs[i], dist[cs[i][:, 0], cs[i][:, 1]] * 2.0) for i in idx]


def panel_task(t):
    src, name, line_path, skel_path, min_px, fill_max, per_panel, seed, pool, group = t
    t0 = time.time()
    r = panel_strokes(line_path, skel_path, min_px, fill_max)
    if r is None or len(r[3]) < 8:
        return []
    g, sk, dist, cs = r
    rng = np.random.default_rng(seed)
    shape = sk.shape
    paths = [order_path(c).astype(float) for c in cs]
    dirs_all = np.array([direction(c) for c in cs])
    cent_all = np.array([c.mean(0) for c in cs])
    ends_all, tans_all, owner = [], [], []
    for i, q in enumerate(paths):
        if len(q) < 2:
            continue
        ta, tb = terminal_tangent(q)
        ends_all += [q[0], q[-1]]; tans_all += [ta, tb]; owner += [i, i]
    ends_all = np.array(ends_all) if ends_all else np.zeros((0, 2))
    tans_all = np.array(tans_all) if tans_all else np.zeros((0, 2))
    owner = np.array(owner, int) if len(owner) else np.zeros(0, int)
    full = np.zeros(shape, bool)
    for c in cs:
        full[c[:, 0], c[:, 1]] = True
    pick = rng.choice(len(cs), min(per_panel, len(cs)), replace=False)
    rows = []
    for ti in pick:
        c = cs[ti]
        ctx = full.copy()
        ctx[c[:, 0], c[:, 1]] = False
        cd = dist_to(ctx)
        sel = np.ones(len(cs), bool); sel[ti] = False
        e_sel = owner != ti if len(owner) else np.zeros(0, bool)
        cands = build_candidates(c, pool, rng, shape, dist)
        if any(v is None for v in cands.values()):
            continue
        for kind, cand in cands.items():
            pts, wmed, wstd, wfill = cand
            f = features(pts, wmed, wstd, wfill, cd, dirs_all[sel], cent_all[sel], ctx,
                         ends_all[e_sel], tans_all[e_sel], None, shape=shape)
            f.update({"tile": f"{src}/{name}", "group": group, "token": int(ti), "kind": kind,
                      "label": int(kind == "true"),
                      "on_other": round(float((cd[pts[:, 0], pts[:, 1]] <= 3).mean()), 4)})
            rows.append(f)
    print(f"  {name[:40]} strokes {len(cs)} rows {len(rows)} {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--panels", type=int, default=800)
    p.add_argument("--per-panel", type=int, default=12)
    p.add_argument("--pool-per-panel", type=int, default=30)
    p.add_argument("--pool-panels", type=int, default=150)
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--fill-max", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--output", default="results/gate1_panels_20260918/candidates.csv")
    a = p.parse_args()

    rows_meta = [r for r in csv.DictReader(open(Path(a.cache) / "panel_strokes.csv"))
                 if r["source"] not in EXCLUDE_SOURCES and int(r["strokes"]) >= 40]
    paths = {}
    for src in {r["source"] for r in rows_meta}:
        for m in csv.DictReader(open(ROOT / "dataset" / src / "manifest.csv")):
            paths[(src, m["name"])] = (str(ROOT / m["native_line_path"]), m.get("content_fingerprint", ""))
    # panels of one page share content and style, so the page fingerprint (not
    # the panel) is the unit the fit must split on. Keep every panel; carry the
    # group id. Panels without a fingerprint group by source+name.
    uniq = rows_meta
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(uniq))
    chosen = [uniq[i] for i in order[: a.panels]]
    pool_src = [uniq[i] for i in order[a.panels: a.panels + a.pool_panels]]
    print(f"panels {len(uniq)} usable, using {len(chosen)} + {len(pool_src)} for the foreign pool", flush=True)

    def task(r, n, extra):
        sp = str(Path(a.cache) / "skeleton" / r["source"] / r["name"])
        return (r["source"], r["name"], paths[(r["source"], r["name"])][0], sp,
                a.min_token_px, a.fill_max) + extra

    t0 = time.time()
    with Pool(a.workers) as pool:
        pool_pw = []
        for got in pool.imap_unordered(pool_task, [task(r, i, (a.pool_per_panel, a.seed + i)) for i, r in enumerate(pool_src)]):
            pool_pw += got
        print(f"foreign pool {len(pool_pw)} strokes  {time.time()-t0:.0f}s", flush=True)
        tasks = [task(r, i, (a.per_panel, a.seed + 1000 + i, pool_pw,
                             paths[(r["source"], r["name"])][1] or f"{r['source']}/{r['name']}"))
                 for i, r in enumerate(chosen)]
        rows = []
        for i, got in enumerate(pool.imap_unordered(panel_task, tasks, chunksize=1), 1):
            rows += got
            if i % 50 == 0:
                print(f"{i}/{len(tasks)} rows {len(rows)} {time.time()-t0:.0f}s", flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tile", "group", "token", "kind", "label", "on_other"] + [k for k in rows[0] if k[:2] in ("a_", "b_", "c_")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    print(f"rows {len(rows)} -> {out}")
    for k in KINDS:
        print(f"  {k:<11}{sum(1 for r in rows if r['kind'] == k):>8}")


if __name__ == "__main__":
    main()
