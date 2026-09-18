#!/usr/bin/env python
"""Re-measure whether the preprocessor's output tokenizes -- with the tools that
did not exist when it was first measured.

2026-09-17 said no: 262 fragments per tile against GT's 27, capture 0.62, and a
token-level selection ceiling of 0.301 against 0.552 for pixel-level selection.
That measurement predates hole filling, spur pruning and junction linking, all
of which turned out to matter a lot on GT, so the number may have been measuring
the tokenizer rather than the preprocessor. Same tiles, same 3px tolerance, same
definitions; only the tokenizer is the current one.

Conditioning images are inverted (bright ink on black): ink is `> INK_HI`.

Reported per tile:
  no_sel      keep all cond ink
  pixel_oracle  keep cond ink within TOL of GT ink   (the pixel-level ceiling)
  token_oracle  keep whole strokes whose pixels are mostly within TOL of GT
  token_ceiling recall of ALL tokenized ink (what any token method could reach)
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402

TOL = 3


def near(mask, tol=TOL):
    """distance transform of the complement: how far each pixel is from `mask`."""
    if not mask.any():
        return np.full(mask.shape, 1e6, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def score(keep, gt_ink, d_gt, gt_n):
    if not keep.any():
        return 0.0, 0.0
    d_keep = near(keep)
    recall = float((d_keep[gt_ink] <= TOL).mean()) if gt_n else 0.0
    prec = float((d_gt[keep] <= TOL).mean())
    return recall, prec


def work(t):
    name, gt_path, cond_path, ink_hi, close_px, hole_max = t
    gt = cv2.imread(str(gt_path), 0)
    cd = cv2.imread(str(cond_path), 0)
    if gt is None or cd is None:
        return None
    if cd.shape != gt.shape:
        cd = cv2.resize(cd, gt.shape[::-1], interpolation=cv2.INTER_AREA)
    gt_ink = gt < 128
    raw = cd > ink_hi
    if close_px:
        raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((close_px, close_px), np.uint8)) > 0
    cond_ink = ink_mask(np.where(raw, 0, 255).astype(np.uint8), hole_max)
    d_gt = near(gt_ink)
    gt_n = int(gt_ink.sum())
    sk = skeletonize(cond_ink)
    dist = cv2.distanceTransform(cond_ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep_s, info = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i, (r, c) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    tok_all = np.zeros_like(sk)
    tok_keep = np.zeros_like(sk)
    n_tok = 0
    for k, mem in groups.items():
        if not keep_s[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        n_tok += 1
        tok_all[rr, cc] = True
        if float((d_gt[rr, cc] <= TOL).mean()) > 0.5:
            tok_keep[rr, cc] = True
    r_no, p_no = score(cond_ink, gt_ink, d_gt, gt_n)
    r_px, p_px = score(cond_ink & (d_gt <= TOL), gt_ink, d_gt, gt_n)
    r_tk, p_tk = score(tok_keep, gt_ink, d_gt, gt_n)
    r_ce, p_ce = score(tok_all, gt_ink, d_gt, gt_n)
    return {"tile": name, "cond_ink": float(cond_ink.mean()), "skel_px": int(sk.sum()),
            "tokens": n_tok, "captured": round(float(tok_all.sum()) / max(int(sk.sum()), 1), 4),
            "junction_frac": round(float(info["nodes"]) / max(n_tok, 1), 3),
            "spans": info["spans"], "links": info["links"], "spurs": info["spurs_removed"],
            "no_sel_recall": round(r_no, 4), "no_sel_prec": round(p_no, 4),
            "pixel_recall": round(r_px, 4), "pixel_prec": round(p_px, 4),
            "token_recall": round(r_tk, 4), "token_prec": round(p_tk, 4),
            "token_ceiling_recall": round(r_ce, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default="/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
    p.add_argument("--cond-dir", default="/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
    p.add_argument("--list", default="/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family.txt")
    p.add_argument("--ink-hi", type=int, default=32)
    p.add_argument("--close-px", type=int, default=3)
    p.add_argument("--hole-max", type=int, default=30)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="results/cond_tokenize_20260918/per_tile.csv")
    a = p.parse_args()
    names = [l.strip() for l in open(a.list) if l.strip()]
    tasks = []
    for n in names:
        stem = Path(n).stem
        gt = Path(a.gt_dir) / f"{stem}.jpg"
        cond = Path(a.cond_dir) / f"{stem}.jpg"
        if gt.exists() and cond.exists():
            tasks.append((stem, gt, cond, a.ink_hi, a.close_px, a.hole_max))
    if a.limit:
        tasks = tasks[: a.limit]
    print(f"tiles {len(tasks)}  ink>{a.ink_hi}, close {a.close_px}, holes<={a.hole_max}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"\ntiles {len(rows)}  -> {out}")
    print(f"{'トークン数/タイル':<22}{med('tokens'):>8.0f}   (GT 27)")
    print(f"{'骨格の捕捉率':<22}{med('captured'):>8.3f}   (GT 0.91)")
    print(f"{'選択なし 回収/精度':<22}{med('no_sel_recall'):>8.3f} / {med('no_sel_prec'):.3f}")
    print(f"{'画素単位オラクル':<22}{med('pixel_recall'):>8.3f} / {med('pixel_prec'):.3f}")
    print(f"{'トークン単位オラクル':<22}{med('token_recall'):>8.3f} / {med('token_prec'):.3f}")
    print(f"{'トークン方式の上限':<22}{med('token_ceiling_recall'):>8.3f}")


if __name__ == "__main__":
    main()
