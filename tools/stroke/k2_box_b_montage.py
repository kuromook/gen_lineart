#!/usr/bin/env python
"""K2 箱 B の目視用モンタージュ(A2 の停止条件の相談材料)。

A2 は「線分上の墨画素数の分布が 0 の山と非 0 の山に分かれなければ襟の取り方が悪いので
止めて相談する」と登録した。実測は 0 に 88 件の山があるが非 0 側は尾を引くだけで山にならない。
→ 判断材料として、墨の水準ごとに実物を並べる。**数は動かさない。**

行(上から):
  墨 0(= 箱 B の分子)/ 墨 1〜2 / 墨 3 以上 / 襟より短くて測れなかった対(隙間 2.0〜4.0px)
各タイルに 2 つの tip を○で、両端を結ぶ線分を点線で描く。
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))

THR, GROW, MAX_BEND = 128, 200, 35.0
TILE, PAD, N_COL, SEED = 160, 34, 8, 20261016


def tile_of(pid, source, name, pairs):
    """1 コマから、指定された対の周りを切り出して返す。"""
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return []
    from skimage.morphology import skeletonize
    from stroke_path import order_path
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(skeletonize(ink), dist * 2.0,
                                                             max_bend=MAX_BEND)
    tips = {}
    for i in range(len(spans)):
        rr, cc = spans[i]
        if len(rr) == 0:
            continue
        p = order_path(np.stack([rr, cc], 1)).astype(float)
        if len(p) >= 2:
            tips[i] = (p[0], p[-1])
    out = []
    for d in pairs:
        i, j, ei, ej = int(d["s0"]), int(d["s1"]), int(d["e0"]), int(d["e1"])
        if i not in tips or j not in tips:
            continue
        a, b = tips[i][ei], tips[j][ej]
        mid = (a + b) / 2.0
        r0 = int(round(mid[0])) - TILE // 2; c0 = int(round(mid[1])) - TILE // 2
        r0 = max(0, min(r0, g.shape[0] - TILE)); c0 = max(0, min(c0, g.shape[1] - TILE))
        crop = cv2.cvtColor(g[r0:r0 + TILE, c0:c0 + TILE], cv2.COLOR_GRAY2BGR)
        if crop.shape[:2] != (TILE, TILE):
            continue
        pa = (int(round(a[1] - c0)), int(round(a[0] - r0)))
        pb = (int(round(b[1] - c0)), int(round(b[0] - r0)))
        for t in np.linspace(0, 1, 14):                       # 隙間を点線で
            if int(t * 14) % 2 == 0:
                q = (int(round(pa[0] + (pb[0] - pa[0]) * t)), int(round(pa[1] + (pb[1] - pa[1]) * t)))
                cv2.circle(crop, q, 1, (0, 180, 255), -1)
        cv2.circle(crop, pa, 6, (0, 0, 220), 1)
        cv2.circle(crop, pb, 6, (220, 60, 0), 1)
        cv2.putText(crop, f"{d['gap']:.0f}px ink{d['ink_on_seg']}", (3, TILE - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 0), 1, cv2.LINE_AA)
        out.append(crop)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pairs", default="results/k2_missed_candidates_20261017/per_pair.csv")
    ap.add_argument("--out", default="results/k2_missed_candidates_20261017/montage_box_b.png")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / a.pairs))
            if r["box"] == "B" and r["same"] == "1" and r["passed"] == "1"]
    for r in rows:
        r["gap"] = float(r["gap"])
    lv = {"ink0": [], "ink1_2": [], "ink3up": [], "short": []}
    for r in rows:
        n = int(r["ink_on_seg"])
        lv["short" if n < 0 else "ink0" if n == 0 else "ink1_2" if n <= 2 else "ink3up"].append(r)
    rng = np.random.default_rng(SEED)
    sel = {k: ([v[i] for i in rng.choice(len(v), min(N_COL, len(v)), replace=False)] if v else [])
           for k, v in lv.items()}
    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    bands = []
    for k in ("ink0", "ink1_2", "ink3up", "short"):
        byp = defaultdict(list)
        for r in sel[k]:
            byp[r["panel"]].append(r)
        imgs = []
        for pid, prs in ((int(k_), v_) for k_, v_ in byp.items()):
            imgs += tile_of(pid, P[pid]["source"], P[pid]["name"], prs)
        if not imgs:
            continue
        while len(imgs) < N_COL:
            imgs.append(np.full((TILE, TILE, 3), 245, np.uint8))
        band = np.concatenate([np.pad(im, ((2, 2), (2, 2), (0, 0)), constant_values=200)
                               for im in imgs[:N_COL]], 1)
        lab = np.full((PAD, band.shape[1], 3), 255, np.uint8)
        cap = {"ink0": f"ink on segment = 0  (box B numerator, n={len(lv['ink0'])})",
               "ink1_2": f"ink 1-2  (n={len(lv['ink1_2'])})",
               "ink3up": f"ink >= 3  (n={len(lv['ink3up'])})",
               "short": f"gap <= 2x collar, untestable  (n={len(lv['short'])})"}[k]
        cv2.putText(lab, cap, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        bands += [lab, band]
    out = ROOT / a.out
    cv2.imwrite(str(out), np.concatenate(bands, 0))
    print(f"書き出した: {out}")
    print({k: len(v) for k, v in lv.items()})


if __name__ == "__main__":
    main()
