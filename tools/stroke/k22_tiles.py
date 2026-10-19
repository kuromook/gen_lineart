#!/usr/bin/env python
"""K22 K19 の 52 タイルを 1 枚ずつ書き出し、**盲検を保つ manifest** を作る。

manifest に入れるのは **id と画像の名前だけ**。
`kind`(本番/反復/錨/突き合わせ)・`gap`・部位ラベル・前回の答えは**入れない**。
対応表は `key.json` に残り、採点は K 側で行う。
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-grammar/tools/stroke")
import k2_box_b_gray as K
import k18_gap_cost as K18

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "results/k19_author_request_20261019"
OUT = SRC / "tiles"
TILE, ZOOM = 150, 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = json.load(open(SRC / "key.json"))["key"]
    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(K.TRACK_H / "labels/state.json")
    P = G.S["panels"]
    byp = defaultdict(list)
    for did, it in key.items():
        byp[int(it["panel"])].append((int(did), it))
    made = {}
    for pid, its in sorted(byp.items()):
        g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
        Q = K.pipeline(g)
        for did, it in its:
            i, j, ei, ej = int(it["s0"]), int(it["s1"]), int(it["e0"]), int(it["e1"])
            if i not in Q["tips"] or j not in Q["tips"]:
                print(f"  **tip なし** #{did}"); continue
            pa, pb = Q["tips"][i][ei], Q["tips"][j][ej]
            c = (pa + pb) / 2.0
            r0 = int(np.clip(c[0] - TILE // 2, 0, g.shape[0] - TILE))
            c0 = int(np.clip(c[1] - TILE // 2, 0, g.shape[1] - TILE))
            im = cv2.cvtColor(g[r0:r0 + TILE, c0:c0 + TILE], cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (TILE * ZOOM, TILE * ZOOM), interpolation=cv2.INTER_NEAREST)
            px = lambda p: ((p[1] - c0) * ZOOM, (p[0] - r0) * ZOOM)
            qa, qb = px(pa), px(pb)
            mid = (int((qa[0] + qb[0]) / 2), int((qa[1] + qb[1]) / 2))
            cv2.circle(im, mid, 26, (0, 165, 255), 1)
            for p_, sp_, e_ in ((qa, i, ei), (qb, j, ej)):
                u = K18.outward_tangent(Q, sp_, e_)
                if u is None:
                    cv2.circle(im, (int(p_[0]), int(p_[1])), 5, (30, 30, 210), 2); continue
                n_ = np.array([-u[1], u[0]]) * 7.0 * ZOOM / 2.0
                cv2.line(im, (int(p_[0] - n_[1]), int(p_[1] - n_[0])),
                         (int(p_[0] + n_[1]), int(p_[1] + n_[0])), (30, 30, 210), 2, cv2.LINE_AA)
            name = f"t{did:03d}.png"
            cv2.imwrite(str(OUT / name), im, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            made[did] = name
    man = dict(
        id="k19",
        title="線の切れ目の判定(2 回目)",
        intro="各タイルは線画を 3 倍に拡大した切り出しです。"
              "**橙の細い円**が見る場所、その中の**赤い短い棒 2 本**が、"
              "処理が「線の端」と判断した 2 点です(棒は線に直交しています)。"
              "2 本の棒のあいだには何も描いていません。",
        question="その 2 点のあいだは、もともとどうなっていましたか。",
        options=[
            dict(value=1, label="飛ばし線", hint="意図的に切って描いた(処理が繋がなかったのは正しい)"),
            dict(value=2, label="かすれ", hint="描いたが出力で飛んだ(処理が繋ぐべきだった)"),
            dict(value=3, label="切れていない", hint="2 本の棒のあいだが繋がっている"),
            dict(value=4, label="判断できない", hint="迷ったらこれで構いません。捨てずに分母に入れます"),
        ],
        note="判定の対象は**2 本の赤い棒のあいだ**だけです。周りのベタの中の白い点は対象ではありません。",
        tiles=[dict(id=d, src=f"tiles/{n}") for d, n in sorted(made.items())],
    )
    json.dump(man, open(SRC / "manifest.json", "w"), indent=1, ensure_ascii=False)
    tot = sum((OUT / n).stat().st_size for n in made.values())
    print(f"タイル {len(made)} 枚 / 合計 {tot/1e6:.2f}MB  -> {OUT}")
    print(f"manifest: {SRC / 'manifest.json'}(kind/gap/部位ラベルは**入れていない**)")


if __name__ == "__main__":
    main()
