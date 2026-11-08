# -*- coding: utf-8 -*-
"""L38 **線の半幅を線の層から引く**(腕W の `r`。登録 §6。2026-11-08)。

**役割ごとに引き直す**(規律9)—— `r` は「線の幅のもの」を見分ける定数なので、
**塗りの面から引いてはいけない。線の層から引く**。

やり方: 頁ごとに線の層の墨を取り、**距離変換の尾根**(周り 3x3 の最大と等しい画素)での
**距離の中央**を半幅とする。**尾根で取るのは、墨の全画素で取ると縁の 0.5 に引っ張られるから**。

**印字するもの**: 頁ごとの半幅 / 全体の中央 / 母集団(頁数・墨の画素数)/ 落とした頁の理由別。
"""
import collections
import os
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l17_known_cells import arms, jobs, zmap                        # noqa: E402
from l38_merge_cause_20261106 import drop_wip, layer_masks          # noqa: E402
from l38_neck_crops_20261107 import line_ink                        # noqa: E402


def half_width(ink):
    d = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    mx = cv2.dilate(d, np.ones((3, 3), np.uint8))
    ridge = (d >= mx - 1e-6) & ink
    v = d[ridge]
    return (float(np.median(v)), int(ridge.sum())) if v.size else (None, 0)


def main():
    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'頁 {len(js)}')
    rows, ink_tot = [], 0
    for i, job in enumerate(js, 1):
        pid, work, fz, lz, sel = job
        _, shape = layer_masks(fz, 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁(形が取れない)'] += 1
            continue
        ink = line_ink(lz, shape)
        n = int(ink.sum())
        if n < 10000:
            cut['線の墨が 10,000 画素未満の頁'] += 1
            continue
        h, nr = half_width(ink)
        if h is None:
            cut['尾根が1画素も立たない頁'] += 1
            continue
        rows.append((work, pid, h, n, nr))
        ink_tot += n
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  半幅の在る頁 {len(rows)}', flush=True)

    print(f'\n--- 頁ごとの半幅(**母集団: 半幅の在る頁 {len(rows)} / 墨 {ink_tot:,} 画素**)---')
    for work, pid, h, n, nr in sorted(rows, key=lambda r: r[2]):
        print(f'  {h:5.2f}  墨 {n:>9,}  尾根 {nr:>8,}  {pid}')
    hs = sorted(r[2] for r in rows)
    med = statistics.median(hs)
    print(f'\n  **半幅の中央 {med:.2f} 画素**(1/4 {hs[len(hs) // 4]:.2f} / '
          f'3/4 {hs[3 * len(hs) // 4]:.2f} / 最小 {hs[0]:.2f} / 最大 {hs[-1]:.2f})')
    print(f'  **幅にすると {2 * med:.2f} 画素**')
    print(f'\n  **腕W の `r` = {round(med)}**(= 半幅の中央を丸めた値)。'
          f'**この値は線の層から引いた。塗りの面からは引いていない**(規律9)')
    print('  **作品ごと(畳まない)**:')
    byw = collections.defaultdict(list)
    for work, pid, h, n, nr in rows:
        byw[work].append(h)
    for w, v in sorted(byw.items()):
        print(f'    {w}  頁 {len(v)}  中央 {statistics.median(v):.2f}')
    print('\n落とした頁(理由別。規律9):')
    for k, v in cut.most_common():
        print(f'  {v:>4}  {k}')


if __name__ == '__main__':
    main()
