# -*- coding: utf-8 -*-
"""m59: **断片化しているらしい層を探す**(作者の報告を受けて)。

作者(2026-10-26): **「レイヤーの再構築が失敗していて断片化しているらしいレイヤーが
散見された。この現象は見覚えがある。おおむね答えられたとおもうが修正はしておいてほしい。
既知の現象なので抽出ツールに問い合わせれば伝わるだろう」**

**診断はしない。** 抽出ツールに**名指しで渡せる一覧**を作るのが目的。
**量は3つ出し、どれも印字する**(どれが「断片化」かはこちらでは決めない):

  (a) **アルファの8連結成分の数**(多いほど散っている)
  (b) **囲み箱の中の墨の割合**(低いほど穴だらけ)
  (c) **軸に平行な空の帯**: 囲み箱の中で、**全部透明な行/列が連続する区間の数**
      —— **タイル単位で欠けると、規則的な空の帯が出る**

**(c) が「再構築の失敗」に最も近い形**だが、**閾値は引かない。並べて渡す。**
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request5')
OUT = Path('results/m59_fragmentation'); OUT.mkdir(parents=True, exist_ok=True)
SMALL = 1200            # 測る前にこの長辺まで縮める(成分の数は保たれる向き)


def bands(mask):
    """全部空の行/列が連続する区間の数(囲み箱の内側で数える)。"""
    n = 0
    for v in (mask.any(axis=1), mask.any(axis=0)):
        prev = True
        for x in v:
            if (not x) and prev:
                n += 1
            prev = bool(x)
    return n


def main():
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
             if l.strip()]
    q = {(it['pid'], it['lid']): it for it in
         json.load(open('results/m56_ui/queue.json'))['queue']}
    rows = []
    for p in pages:
        for l in p['layers']:
            it = q.get((p['source'], l['layer_id']))
            if it is None:
                continue                       # 用紙層(外した40枚)
            im = cv2.imread(str(DELIV / l['file']), cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            a = im[..., 3] > 0
            del im
            if not a.any():
                continue
            ys, xs = np.nonzero(a)
            a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = a.shape
            s = min(1.0, SMALL / max(h, w))
            if s < 1.0:
                a = cv2.resize(a.astype(np.uint8), (max(1, round(w * s)),
                                                    max(1, round(h * s))),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            ncomp = cv2.connectedComponents(a.astype(np.uint8),
                                            connectivity=8)[0] - 1
            rows.append(dict(order=it['order'], page=p['page'],
                             source=p['source'], layer_id=l['layer_id'],
                             name=(it['work'].split('「')[-1].split('」')[0]
                                   if '「' in it['work'] else ''),
                             box_h=h, box_w=w,
                             components=int(ncomp),
                             fill=round(float(a.mean()), 4),
                             empty_bands=int(bands(a))))
            print(f'  {len(rows):>4} 問{it["order"]:<4} 成分{ncomp:>5} '
                  f'埋まり{a.mean():>6.1%} 空帯{bands(a):>4}  {p["page"]}'
                  f' L{l["layer_id"]}', flush=True)
    with open(OUT / 'fragmentation.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    print(f'\n== 測った層 {len(rows)} ==')
    for key in ('components', 'empty_bands'):
        v = np.array([r[key] for r in rows])
        print(f'  {key:14s} 中央 {np.median(v):>8.1f} / 平均 {v.mean():>8.1f} '
              f'/ 最大 {v.max():>8d}')
    print('\n== 成分の多い上位20(**閾値は引かない。並べる**)==')
    for r in sorted(rows, key=lambda r: -r['components'])[:20]:
        print(f'  問{r["order"]:<4} 成分{r["components"]:>5} 埋まり{r["fill"]:>7.1%} '
              f'空帯{r["empty_bands"]:>4}  {r["page"]} L{r["layer_id"]} '
              f'「{r["name"][:20]}」')
    print('\n== 空の帯が多い上位20 ==')
    for r in sorted(rows, key=lambda r: -r['empty_bands'])[:20]:
        print(f'  問{r["order"]:<4} 空帯{r["empty_bands"]:>4} 成分{r["components"]:>5} '
              f'埋まり{r["fill"]:>7.1%}  {r["page"]} L{r["layer_id"]} '
              f'「{r["name"][:20]}」')
    print(f'\nwrote {OUT / "fragmentation.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
