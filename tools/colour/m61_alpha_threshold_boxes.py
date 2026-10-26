# -*- coding: utf-8 -*-
"""m61: **囲み箱を `alpha > 0` で取っていたことの影響を測る**。

基盤・抽出側の訂正(2026-10-26):
  **`drawn_ratio` は「アルファ26以上の画素の割合」**であって「画素の割合」ではない。
  **`drawn_ratio > 0` は「墨がある」ではない** —— 墨の閾はアルファ128以上。
  実測例: `page0028_L17` は `drawn_ratio` 0.0726 に対して墨は 0.0083。

**こちらの `m56` は囲み箱を `alpha > 0` で取っている。**
**アルファ1の塵が画布に散っていれば、箱は画布ぜんたいに広がり、絵は小さく出る。**
**表示倍率もその箱から決まるので、倍率の主張にも当たる。**

**測る(閾値を3つ並べる。どれかを選ばない)**: `alpha > 0` / `>= 26`(抽出側の閾) /
`>= 128`(墨の閾)の3つで囲み箱を取り、**面積の比**を出す。
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request5')
OUT = Path('results/m61_alpha_box'); OUT.mkdir(parents=True, exist_ok=True)
THRS = [1, 26, 128]


def box(a, t):
    m = a >= t
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return (int(ys.min()), int(xs.min()), int(ys.max() - ys.min() + 1),
            int(xs.max() - xs.min() + 1), int(m.sum()))


def main():
    pages = {p['source']: p for p in
             (json.loads(l) for l in
              open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
              if l.strip())}
    q = json.load(open('results/m56_ui/queue.json'))['queue']
    rows = []
    for it in q:
        p = pages[it['pid']]
        f = [l for l in p['layers'] if l['layer_id'] == it['lid']][0]
        im = cv2.imread(str(DELIV / f['file']), cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] != 4:
            continue
        a = im[..., 3]
        del im
        r = dict(order=it['order'], page=p['page'], layer_id=it['lid'])
        for t in THRS:
            b = box(a, t)
            r[f'box{t}'] = f'{b[2]}x{b[3]}' if b else '-'
            r[f'area{t}'] = b[2] * b[3] if b else 0
            r[f'px{t}'] = b[4] if b else 0
        r['ratio_0_vs_26'] = (round(r['area1'] / r['area26'], 3)
                              if r['area26'] else None)
        r['ratio_0_vs_128'] = (round(r['area1'] / r['area128'], 3)
                               if r['area128'] else None)
        rows.append(r)
        print(f'  問{it["order"]:<4} 箱 a>0 {r["box1"]:>12} / a>=26 {r["box26"]:>12}'
              f' / a>=128 {r["box128"]:>12}  面積比 {r["ratio_0_vs_26"]}'
              f' / {r["ratio_0_vs_128"]}', flush=True)
    with open(OUT / 'alpha_boxes.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    print(f'\n== 測った層 {len(rows)} ==')
    for key, lab in (('ratio_0_vs_26', 'a>0 の箱 / a>=26 の箱'),
                     ('ratio_0_vs_128', 'a>0 の箱 / a>=128 の箱')):
        v = [r[key] for r in rows if r[key]]
        v.sort()
        print(f'  {lab}: 中央 {v[len(v)//2]:.3f} / 最大 {v[-1]:.3f} '
              f'/ **2倍以上 {sum(1 for x in v if x >= 2)} 層** '
              f'/ 10倍以上 {sum(1 for x in v if x >= 10)} 層')
    print('\n== a>0 の箱が a>=26 の箱の2倍以上(= 薄い画素が箱を広げている)上位15 ==')
    bad = sorted((r for r in rows if r['ratio_0_vs_26']),
                 key=lambda r: -r['ratio_0_vs_26'])[:15]
    for r in bad:
        print(f'  問{r["order"]:<4} 比 {r["ratio_0_vs_26"]:>8.1f}  '
              f'a>0 {r["box1"]:>12} -> a>=26 {r["box26"]:>12}  '
              f'{r["page"]} L{r["layer_id"]}')
    print(f'\nwrote {OUT / "alpha_boxes.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
