#!/usr/bin/env python
"""Real negatives: strokes a trained model drew that the GT does not have.

The synthetic negatives (displaced / rotated / foreign) were a stand-in, used
because the preprocessor's output does not tokenize into strokes. Model OUTPUT
does tokenize -- measured 2026-09-18: the good snapshots give 68-72 strokes per
tile at capture 0.90, against GT's 37 at 0.900 -- so the negatives this project
actually cares about can be taken from real data:

  matched : a stroke of the output that lies on a GT stroke   (label 1)
  extra   : a stroke of the output with no GT ink under it     (label 0)

Both come from the SAME drawing, so there is no provenance confound: the
question is whether, inside one model output, the extra strokes can be told from
the real ones by how they sit among their neighbours.

Features are the same three tiers as gate 1 (a: the stroke alone, b: coarse
relations, c: sharp relations), so the tiers can be ablated exactly as before.
Context for a candidate is every OTHER stroke of the same output, extras
included -- the realistic setting, since at inference nothing is labelled.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402
from gate1_features import direction, features, terminal_tangent  # noqa: E402
from make_negatives import order_path  # noqa: E402
from stroke_churn import dist_to  # noqa: E402

TOL = 3.0


def tile_strokes(path, min_px):
    g = cv2.imread(str(path), 0)
    if g is None:
        return None
    if g.shape != (480, 480):
        g = cv2.resize(g, (480, 480), interpolation=cv2.INTER_AREA)
    ink = ink_mask(g)
    if ink.sum() < 200:
        return None
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, _ = strokes_from_skeleton(skeletonize(ink), dist * 2)
    groups = {}
    for i, (r, c) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    out = []
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        if len(rr) >= min_px:
            out.append(np.stack([rr, cc], 1))
    return g, ink, dist, out


def work(t):
    snap, stem, out_path, gt_path, min_px, per_tile, seed, keep_hi, extra_lo = t
    got = tile_strokes(out_path, min_px)
    gt = cv2.imread(str(gt_path), 0)
    if got is None or gt is None or len(got[3]) < 8:
        return []
    g, ink, dist, cs = got
    gt_ink = gt < 128
    d_gt = dist_to(gt_ink)
    rng = np.random.default_rng(seed)
    shape = ink.shape
    paths = [order_path(c).astype(float) for c in cs]
    dirs = np.array([direction(c) for c in cs])
    cent = np.array([c.mean(0) for c in cs])
    ends, tans, owner = [], [], []
    for i, q in enumerate(paths):
        if len(q) < 2:
            continue
        ta, tb = terminal_tangent(q)
        ends += [q[0], q[-1]]; tans += [ta, tb]; owner += [i, i]
    ends = np.array(ends) if ends else np.zeros((0, 2))
    tans = np.array(tans) if tans else np.zeros((0, 2))
    owner = np.array(owner, int) if len(owner) else np.zeros(0, int)
    full = np.zeros(shape, bool)
    for c in cs:
        full[c[:, 0], c[:, 1]] = True
    order = rng.permutation(len(cs))
    rows = []
    for ti in order:
        c = cs[ti]
        cover = float((d_gt[c[:, 0], c[:, 1]] <= TOL).mean())
        if cover >= keep_hi:
            label, kind = 1, "matched"
        elif cover <= extra_lo:
            label, kind = 0, "extra"
        else:
            continue
        ctx = full.copy(); ctx[c[:, 0], c[:, 1]] = False
        cd = dist_to(ctx)
        sel = np.ones(len(cs), bool); sel[ti] = False
        e_sel = owner != ti if len(owner) else np.zeros(0, bool)
        w = dist[c[:, 0], c[:, 1]] * 2.0
        f = features(c, float(np.median(w)), float(np.std(w)), float((w > 8).mean()),
                     cd, dirs[sel], cent[sel], ctx, ends[e_sel], tans[e_sel], None, shape=shape)
        f.update({"snapshot": snap, "tile": stem, "token": int(ti), "kind": kind,
                  "label": label, "cover": round(cover, 3),
                  "on_other": round(float((cd[c[:, 0], c[:, 1]] <= TOL).mean()), 4)})
        rows.append(f)
        if len(rows) >= per_tile:
            break
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default="/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
    p.add_argument("--roots", nargs="*", default=[
        "/home/sh1/deepl/lineart-pair-signal/results/h34_alignment_probe_20260915/aligned",
        "/home/sh1/deepl/lineart-pair-signal/results/h34_alignment_probe_20260915/control",
        "/home/sh1/deepl/lineart-controlnet-sd15-refine/results/controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914",
    ])
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--per-tile", type=int, default=20)
    p.add_argument("--keep-hi", type=float, default=0.7, help="cover >= this counts as matched")
    p.add_argument("--extra-lo", type=float, default=0.1, help="cover <= this counts as extra")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit-tiles", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--output", default="results/real_negatives_20260918/candidates.csv")
    a = p.parse_args()
    tasks = []
    for root in a.roots:
        tag = Path(root).parent.name.split("_")[0] + "/" + Path(root).name
        for sd in sorted(Path(root).glob("step_*")):
            files = sorted(sd.glob("*.png"))
            if a.limit_tiles:
                files = files[: a.limit_tiles]
            for i, f in enumerate(files):
                stem = f.stem.replace("_out", "")
                gt = Path(a.gt_dir) / f"{stem}.jpg"
                if gt.exists():
                    tasks.append((f"{tag}/{sd.name}", stem, str(f), str(gt), a.min_token_px,
                                  a.per_tile, a.seed + i, a.keep_hi, a.extra_lo))
    print(f"tiles {len(tasks)}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, got in enumerate(pool.imap_unordered(work, tasks, chunksize=4), 1):
            rows += got
            if i % 200 == 0:
                print(f"{i}/{len(tasks)}  rows {len(rows)}  {time.time()-t0:.0f}s", flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["snapshot", "tile", "token", "kind", "label", "cover", "on_other"] + \
           [k for k in rows[0] if k[:2] in ("a_", "b_", "c_")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    n1 = sum(r["label"] for r in rows)
    print(f"rows {len(rows)} -> {out}   matched {n1}  extra {len(rows)-n1}")


if __name__ == "__main__":
    main()
