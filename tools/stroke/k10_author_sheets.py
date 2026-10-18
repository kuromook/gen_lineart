#!/usr/bin/env python
"""K10 作者への依頼の用紙を作る(2026-10-18 事前登録)。

127 件(箱 B のうち、襟の合計が隙間より長い 55 件を外したもの)から 40 件を抽出し、
反復 8 件と錨 8 件(箱 A)を混ぜて 56 枚のタイルにする。**作者には区別を知らせない。**
答えは番号で返してもらう。対応表は key.json に置き、用紙には出さない。
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K

SEED, N_SAMPLE, N_REPEAT, N_ANCHOR = 20261018, 40, 8, 8
TILE, ZOOM, PER_PAGE, NCOL = 150, 2, 8, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--contig", default="results/k8_contiguity_20261018/per_pair_contig.csv")
    ap.add_argument("--out", default="results/k10_author_request_20261018")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(ROOT / a.contig)) if r["missing"] == "0"]
    boxb = [r for r in rows if r["box"] == "B"]
    answerable = [r for r in boxb if not (r["contiguous"] == "1" and r["comp_same"] == "0")]
    boxa = [r for r in rows if r["box"] == "A"]
    assert len(answerable) == 127, len(answerable)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(answerable))[:N_SAMPLE]
    sample = [dict(answerable[i], kind="main") for i in idx]
    rep = [dict(sample[i], kind="repeat") for i in rng.permutation(N_SAMPLE)[:N_REPEAT]]
    ai = rng.permutation(len(boxa))[:N_ANCHOR]
    anchors = [dict(boxa[i], kind="anchor") for i in ai]
    items = sample + rep + anchors
    for n, it in enumerate(items):
        it["uid"] = n
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    for n, it in enumerate(items):
        it["display_id"] = n + 1

    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    byp = defaultdict(list)
    for it in items:
        byp[int(it["panel"])].append(it)
    tiles = {}
    for pid, its in byp.items():
        g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
        Q = K.pipeline(g)
        for it in its:
            i, j = int(it["s0"]), int(it["s1"])
            ei, ej = int(it["e0"]), int(it["e1"])
            if i not in Q["tips"] or j not in Q["tips"]:
                continue
            pa, pb = Q["tips"][i][ei], Q["tips"][j][ej]
            c = (pa + pb) / 2.0
            r0 = int(np.clip(c[0] - TILE // 2, 0, g.shape[0] - TILE))
            c0 = int(np.clip(c[1] - TILE // 2, 0, g.shape[1] - TILE))
            im = cv2.cvtColor(g[r0:r0 + TILE, c0:c0 + TILE], cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (TILE * ZOOM, TILE * ZOOM), interpolation=cv2.INTER_NEAREST)
            qa = ((pa[1] - c0) * ZOOM, (pa[0] - r0) * ZOOM)
            qb = ((pb[1] - c0) * ZOOM, (pb[0] - r0) * ZOOM)
            for t in np.linspace(0, 1, 26):          # 隙間を点線で
                if int(t * 26) % 2 == 0:
                    z = (int(qa[0] + (qb[0] - qa[0]) * t), int(qa[1] + (qb[1] - qa[1]) * t))
                    cv2.circle(im, z, 1, (0, 165, 255), -1)
            cv2.circle(im, (int(qa[0]), int(qa[1])), 9, (30, 30, 210), 2)
            cv2.circle(im, (int(qb[0]), int(qb[1])), 9, (30, 30, 210), 2)
            head = np.full((34, im.shape[1], 3), 255, np.uint8)
            cv2.putText(head, f"#{it['display_id']}", (6, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2, cv2.LINE_AA)
            tiles[it["display_id"]] = np.pad(np.concatenate([head, im], 0),
                                             ((3, 3), (3, 3), (0, 0)), constant_values=120)
    ids = sorted(tiles)
    pages = []
    for p0 in range(0, len(ids), PER_PAGE):
        chunk = [tiles[k] for k in ids[p0:p0 + PER_PAGE]]
        while len(chunk) % NCOL:
            chunk.append(np.full_like(chunk[0], 250))
        rowsim = [np.concatenate(chunk[k:k + NCOL], 1) for k in range(0, len(chunk), NCOL)]
        pg = np.concatenate(rowsim, 0)
        path = out / f"sheet_{p0 // PER_PAGE + 1:02d}.png"
        cv2.imwrite(str(path), pg)
        pages.append(str(path))
    key = {str(it["display_id"]): dict(kind=it["kind"], uid=it["uid"], panel=it["panel"],
                                       order=it["order"], label=it["label"], box=it["box"],
                                       gap=round(float(it["gap"]), 2),
                                       thread=int(it["contiguous"] == "1" and it["comp_same"] == "1"),
                                       s0=it["s0"], s1=it["s1"])
           for it in items if it["display_id"] in tiles}
    json.dump(dict(registered=dict(seed=SEED, n_sample=N_SAMPLE, n_repeat=N_REPEAT,
                                   n_anchor=N_ANCHOR, pool=len(answerable),
                                   excluded_55="襟の合計が隙間より長い(出力で隙間が見えない)",
                                   note="作者には main/repeat/anchor の区別を知らせない"),
                   pages=pages, key=key), open(out / "key.json", "w"), indent=1, ensure_ascii=False)
    print(f"用紙 {len(pages)} 枚 / タイル {len(tiles)} 枚")
    for p in pages:
        print(" ", p)
    print("対応表(作者には見せない):", out / "key.json")


if __name__ == "__main__":
    main()
