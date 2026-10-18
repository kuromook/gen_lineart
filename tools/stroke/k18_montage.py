#!/usr/bin/env python
"""K18 の目視: tip・伸ばした先・残った墨の隙間を描く(登録 (G) の述語の確認)。

赤○ = 骨格の tip / 緑 = 伸ばした先(墨の端)/ 橙の点線 = 残った墨の隙間。
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-grammar/tools/stroke")
import k2_box_b_gray as K
import k18_gap_cost as K18

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results/k18_gap_cost_20261019"
TILE, ZOOM, NCOL = 120, 3, 6
ANS = {0: "not asked", 1: "INTENDED (tobashi)", 2: "ARTEFACT (kasure)",
       3: "not cut", 4: "cannot tell"}     # cv2.putText は和文を描けない

per = list(csv.DictReader(open(OUT / "per_pair.csv")))
per.sort(key=lambda r: float(r["gap_sk"]))
pick = [per[int(k)] for k in np.linspace(0, len(per) - 1, 24).round()]
byp = defaultdict(list)
for r in pick:
    byp[int(r["panel"])].append(r)
sys.path.insert(0, str(K.TRACK_H / "tools/face"))
import group_server as G
G.load_state(K.TRACK_H / "labels/state.json")
P = G.S["panels"]
tiles = {}
for pid, its in byp.items():
    g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
    Q = K.pipeline(g)
    for r in its:
        i, j, ei, ej = int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"])
        if i not in Q["paths"] or j not in Q["paths"]:
            continue
        pa = Q["paths"][i][0] if ei == 0 else Q["paths"][i][-1]
        pb = Q["paths"][j][0] if ej == 0 else Q["paths"][j][-1]
        ua, ub = K18.outward_tangent(Q, i, ei), K18.outward_tangent(Q, j, ej)
        qa = pa + float(r["ext_a"]) * ua if ua is not None else pa
        qb = pb + float(r["ext_b"]) * ub if ub is not None else pb
        c = (pa + pb) / 2.0
        r0 = int(np.clip(c[0] - TILE // 2, 0, g.shape[0] - TILE))
        c0 = int(np.clip(c[1] - TILE // 2, 0, g.shape[1] - TILE))
        im = cv2.cvtColor(g[r0:r0 + TILE, c0:c0 + TILE], cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (TILE * ZOOM, TILE * ZOOM), interpolation=cv2.INTER_NEAREST)
        px = lambda p: (int((p[1] - c0) * ZOOM), int((p[0] - r0) * ZOOM))
        for t in np.linspace(0, 1, 40):
            if int(t * 40) % 2 == 0:
                z = (int(px(qa)[0] + (px(qb)[0] - px(qa)[0]) * t),
                     int(px(qa)[1] + (px(qb)[1] - px(qa)[1]) * t))
                cv2.circle(im, z, 1, (0, 165, 255), -1)
        for p, q in ((pa, qa), (pb, qb)):
            cv2.line(im, px(p), px(q), (40, 170, 40), 2)
            cv2.circle(im, px(p), 7, (30, 30, 210), 2)
            cv2.circle(im, px(q), 4, (40, 170, 40), -1)
        head = np.full((54, im.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f'gap {float(r["gap_sk"]):.1f} -> ink {float(r["ink_gap"]):.1f}',
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        col = {1: (20, 20, 200), 2: (30, 140, 30)}.get(int(r["answer"]), (130, 130, 130))
        cv2.putText(head, f'{ANS[int(r["answer"])]}  {r["label"]}', (6, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tiles[(float(r["gap_sk"]), pid, i, j)] = np.pad(
            np.concatenate([head, im], 0), ((3, 3), (3, 3), (0, 0)), constant_values=120)
ks = sorted(tiles)
chunk = [tiles[k] for k in ks]
while len(chunk) % NCOL:
    chunk.append(np.full_like(chunk[0], 250))
rowsim = [np.concatenate(chunk[k:k + NCOL], 1) for k in range(0, len(chunk), NCOL)]
cv2.imwrite(str(OUT / "montage_extension.png"), np.concatenate(rowsim, 0))
print(f"{OUT / 'montage_extension.png'}  ({len(ks)} 件、隙間の小さい順に等間隔で抽出)")
