#!/usr/bin/env python
"""Does a panel survive the trip through stroke tokens and back?

Nothing in this track means anything until it does: a generated token sequence
has to become a drawing, and the measurements of "what the model learned" are
read through the same renderer. Four variants are measured on the SAME panels so
each repair can be credited separately:

  v0_old   16 points per stroke, one median width      (what results/panel_tokens_20260918 holds)
  v1_chain spans chained through junction gaps         (fixes order_path's truncation)
  v2_ppw   + points per 8px of length, width per point
  v3_resid + leftover skeleton (nodes, spurs, bridges, sub-min_len) stamped as discs

Pre-registered criterion (written before the first run, plan 2026-09-18):
  median recall@2px >= 0.95 AND median extra@2px <= 0.02 on 60 random panels,
  and the montage must look like the same drawing.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask, strokes_from_skeleton
from stroke_path import chain, resample, span_paths, widths_at
from render_strokes import agreement, render, render_discs
from make_negatives import order_path

ROOT = Path("/home/sh1/deepl/lineart")
VARIANTS = ("v0_old", "v1_chain", "v2_ppw", "v3_resid")


def panel_paths(cache):
    man = {}
    for m in sorted(ROOT.glob("dataset/regions_*_koma_panels_2026*/manifest.csv")):
        src = m.parent.name
        for r in csv.DictReader(open(m)):
            man[(src, r["name"])] = str(ROOT / r["native_line_path"])
    return man


def build(args):
    source, name, line_path, skel_path = args
    g = cv2.imread(line_path, 0)
    sk = cv2.imread(skel_path, 0)
    if g is None or sk is None:
        return None
    sk = sk > 0
    ink = ink_mask(g)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, info = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i, (rr, cc) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)

    old_p, old_w, ch_p, ch_w, pp_p, pp_w = [], [], [], [], [], []
    kept_old, kept_new, raw_total, n_tok = 0, 0, 0, 0
    assigned = np.zeros_like(sk)
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem])
        cc = np.concatenate([spans[i][1] for i in mem])
        assigned[rr, cc] = True
        pix = np.stack([rr, cc], 1)
        raw_total += len(pix); n_tok += 1
        # v0: order_path over the whole (gapped) pixel set, 16 points, median width
        p_old = order_path(pix).astype(np.float32)
        kept_old += len(p_old)
        w_med = float(np.median(dist[rr, cc] * 2.0))
        pts0 = resample(p_old, per_px=1e9, k_min=16, k_max=16)
        old_p.append(pts0); old_w.append(np.full(len(pts0), w_med, np.float32))
        # v1: chained, still 16 points and one median width
        path, _gaps = chain(span_paths(spans, mem))
        kept_new += len(path)
        pts1 = resample(path, per_px=1e9, k_min=16, k_max=16)
        ch_p.append(pts1); ch_w.append(np.full(len(pts1), w_med, np.float32))
        # v2: chained, adaptive points, per-point width
        pts2 = resample(path, per_px=8.0, k_min=8, k_max=64)
        pp_p.append(pts2); pp_w.append(widths_at(pts2, dist))

    row = {"source": source, "name": name, "tokens": n_tok,
           "kept_old": round(kept_old / max(raw_total, 1), 4),
           "kept_new": round(kept_new / max(raw_total, 1), 4),
           "fill_frac": round(float(((dist * 2)[ink] > 8).mean()), 4),
           "assigned": round(float(assigned.sum()) / max(int(sk.sum()), 1), 4)}
    renders = {}
    renders["v0_old"] = render(old_p, old_w, sk.shape)
    renders["v1_chain"] = render(ch_p, ch_w, sk.shape)
    renders["v2_ppw"] = render(pp_p, pp_w, sk.shape)
    left = sk & ~assigned
    ys, xs = np.nonzero(left)
    renders["v3_resid"] = renders["v2_ppw"] | render_discs(np.stack([ys, xs], 1), dist[ys, xs], sk.shape)
    for v in VARIANTS:
        a = agreement(renders[v], ink)
        row[f"{v}_recall"] = round(a["recall"], 4)
        row[f"{v}_extra"] = round(a["extra"], 4)
        row[f"{v}_f1"] = round(a["f1"], 4)
        row[f"{v}_iou"] = round(a["iou"], 4)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--index", default="results/panel_tokens_20260918/index.csv")
    p.add_argument("--panels", type=int, default=60)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--out", default="results/roundtrip_20260918")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.index)))
    rng = np.random.default_rng(a.seed)
    sel = [rows[i] for i in rng.choice(len(rows), min(a.panels, len(rows)), replace=False)]
    man = panel_paths(a.cache)
    tasks = [(r["source"], r["name"], man[(r["source"], r["name"])],
              str(Path(a.cache) / "skeleton" / r["source"] / r["name"])) for r in sel]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); got = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(build, tasks), 1):
            if r:
                got.append(r)
            if i % 10 == 0:
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(out / "per_panel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(got[0])); w.writeheader(); w.writerows(got)
    med = lambda k: float(np.median([r[k] for r in got]))
    print(f"\n{len(got)} panels -> {out}/per_panel.csv")
    print(f"order_path が残した割合: 旧 {med('kept_old'):.3f} → 連結後 {med('kept_new'):.3f}")
    print(f"骨格のうち線に帰属: {med('assigned'):.3f}   ベタ率 中央値 {med('fill_frac'):.3f}")
    print(f"\n{'変種':<12}{'回収@2px':>12}{'描きすぎ':>12}{'F1@0px':>10}{'IoU@0px':>10}")
    for v in VARIANTS:
        print(f"{v:<12}{med(v+'_recall'):>12.3f}{med(v+'_extra'):>12.3f}{med(v+'_f1'):>10.3f}{med(v+'_iou'):>10.3f}")
    worst = sorted(got, key=lambda r: r["v3_resid_recall"])[:3]
    print("\n最悪の3枚(v3の回収):", ", ".join(f"{r['name'][:30]} {r['v3_resid_recall']:.3f}" for r in worst))


if __name__ == "__main__":
    main()
