# -*- coding: utf-8 -*-
"""L38 **面の母集団を洗う**: その面は本当に「ベタ」か(2026-11-07)。

作者が無作為30面に答えた中で **8/30(27%)が「分からない」「どれでもない」**で、
一言が理由を言った —— **「これは線」**(面積 1,464)/
**「枠線とキャラクターの線、さらにフキダシの線」**(面積 32,243)。
**問いは単位を「ひとつながりのベタ」と書いたが、母集団にベタでないものが混ざっている。**

**答えは直さない。母集団と問いを直す**(指針 §2-3)。この器具は**面ごとに2つの側から測る**
(`feedback_two_sided_layer_identification`: 片面で済ませない):

- **形の側** —— `distanceTransform` の最大値(= 面に入る最大の円の半径)と、
  **細い画素の割合**(半径 3px 未満)。**線は細長く、ベタは太い**
- **墨の側** —— その面を最も覆っている層の**生の濃淡**。Track N の測定:
  **線は gray 63〜87・純黒 6〜32%、塗りは gray 0・純黒 97〜99%**。
  **面の中の墨だけ**を見るので、層全体の性質ではない

**床は引かない。** 作者の答が在る 33面でこの2つが何を言うかを印字するだけで、
**どちらも「線らしさ」の代用であって、線かどうかは作者が決める**(規律6・規律13)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_face_is_solid_20261107.py [頁数]
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l17_known_cells import arms, jobs, zmap                     # noqa: E402
from l38_merge_cause_20261106 import (drop_wip, layer_masks,     # noqa: E402
                                      panels)

ROOT = '/home/sh1/deepl/lineart-fill-region'
OUT = f'{ROOT}/results/l38_face_is_solid_20261107.json'
THIN = 3          # 「細い」と呼ぶ半径(画素)。**床ではなく、印字する量の定義**


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('L38 **面はベタか**(診断。棒は登録していない)', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'母集団: **現行の素 {len(js)} 頁**  細いと呼ぶ半径 {THIN}px')
    for k, v in cut.most_common():
        print(f'  外した {v}  {k}')
    cut.clear()
    if limit:
        js = js[:limit]
        print(f'  **煙試験: 先頭 {limit} 頁だけ**', flush=True)

    rows = []
    for i, job in enumerate(js, 1):
        lm, shape = layer_masks(job[2], 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁'] += 1
            continue
        paths = {int(lid): p for lid, p in job[2]}
        cache = {}       # **頁ごとに捨てる**(層の生 PNG は1枚 10MB 級)

        def gray(lid):
            if lid not in cache:
                if len(cache) >= 6:
                    cache.pop(next(iter(cache)))
                im = cv2.imread(paths.get(lid, ''), cv2.IMREAD_GRAYSCALE)
                cache[lid] = im if (im is not None and im.shape == shape) else None
            return cache[lid]

        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut):
            cuts = [(lid, m[y0:y1, x0:x1] & P) for lid, m in lm]
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                # ---- 形の側: 面に入る最大の円と、細い画素の割合
                d = cv2.distanceTransform(R.astype(np.uint8), cv2.DIST_L2, 5)[R]
                ov = sorted(((lid, int((s & R).sum())) for lid, s in cuts),
                            key=lambda t: -t[1])
                top = int(ov[0][0]) if ov else -1
                # ---- 墨の側: **面の中の、その層の墨だけ**の生の濃淡
                g, pure, mg = gray(top), None, None
                if g is None:
                    cut['最上位の層の PNG が読めない(墨の側だけ欠ける)'] += 1
                else:
                    v = g[y0:y1, x0:x1][R]
                    v = v[v < 128]
                    if v.size:
                        pure = float((v == 0).mean())
                        mg = float(v.mean())
                    else:
                        cut['面の中にその層の墨が無い(半径4 で広がった分)'] += 1
                rows.append(dict(work=job[1], pid=job[0], comp=int(ci), region=int(gi),
                                 area=ar, top_lid=top,
                                 r_max=float(d.max()), r_med=float(np.median(d)),
                                 thin_frac=float((d < THIN).mean()),
                                 pure_black=pure, mean_gray=mg))
        cache.clear()
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  面 {len(rows)}', flush=True)
    json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'\n面 {len(rows)} -> {OUT}')
    for k, v in cut.most_common():
        print(f'  落とした/欠けた {v}  {k}(規律9)')
    q = sorted(r['r_max'] for r in rows)
    n = len(q)
    print('  面に入る最大の円の半径の分位: ' + ' / '.join(
        f'{p}% {q[int(p / 100 * (n - 1))]:.1f}px' for p in (10, 25, 50, 75, 90)))


if __name__ == '__main__':
    main()
