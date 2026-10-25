"""M48: **A0 を8頁と7頁の両方で印字する**。そして**頁ごとに最大の面の占有率**を出す。

**なぜ7頁の版が要るか**: `2017\\illustration\\sunshine2\\page0002` は
**クリップ群のベースしか無く、クリップ層のアルファが届いていない**ので
**標準が構成できない頁**である(追補32 §B2)。**測定前の欠陥**であって、結果を見て
外すのではない。**ただし外すのも恣意に見えるので、両方を同じ行に印字する。**

**先に言っておく(基盤)**: **n=8 が n=7 になると検定の力はさらに落ちる。**
**「外したら強くなった」は起きない。**

**安い点検を1つ足す(基盤の案)**: **各頁で、標準の面の面積の和と最大の面の占有率。**
**最大の面が頁の大半を占めていたら、その頁の `region-F1` は1面との IoU に支配されている**
ので、**判定の前に目に入る。**

**再計算ではなく再印字**(保存済みの頁ごとの点から符号検定をやり直すだけ。走行はしない)。

**初版は6頁の結合を落とした**: `m16` の点は**パイロットのマニフェストの `page` 名**
(`000_2018_illust201710_illust201710_page0003`)、`m45` の面積は**納品の `page` 名**
(`000_2018_illust201710_page0003`)で、**名前で結んだら6頁が落ちた。**
**`m24` の初回(255/303 の欠け)と同じ罠で、同じ晩に3度目。**
→ **`source`(= `page_path`)で結ぶ**(T94 の規則どおり)。
"""
import collections
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                         # noqa: E402

SCORES = Path('results/m16_score/scores.csv')
FULL = Path('results/m45_full_canvas/full_canvas_areas.csv')
DROP = 'F:\\manuscripts\\2017\\illustration\\sunshine2\\page0002.clip'
PILOT = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/'
             'colour_request4_known8/colour_layers_pages.jsonl')


def main():
    import json
    rows = list(csv.DictReader(open(SCORES, encoding='utf-8')))
    # **名前ではなく `source` で結ぶ**(T94)
    pilot = {p['page']: p['source'] for p in
             json.load(open(PILOT, encoding='utf-8'))['pages']}
    deliv = {}
    for line in open(DELIV, encoding='utf-8'):
        if line.strip():
            d = json.loads(line)
            deliv[d['page']] = d['source']
    for r in rows:
        r['source'] = pilot.get(r['page'], r['page'])
    print(f'== 各頁の面と、最大の面の占有率(**安い点検**)==')
    areas = collections.defaultdict(list)
    canvas = {}
    for r in csv.DictReader(open(FULL, encoding='utf-8')):
        src = deliv.get(r['page'], r['page'])
        areas[src].append(int(r['area_px']))
        canvas[src] = int(r['canvas_px'])
    print(f'  {"頁":44s} {"面":>4} {"面積の和/画布":>12} {"最大の面の占有":>14}')
    for r in rows:
        pg = r['source']
        a = sorted(areas.get(pg, []), reverse=True)
        cv = canvas.get(pg, 0)
        if not a or not cv:
            print(f'  {pg[14:]:44s} {r["gt_areas"]:>4}  (面積の記録なし)')
            continue
        print(f'  {pg[14:]:44s} {len(a):>4} {sum(a) / cv:>12.3f} {a[0] / cv:>14.3f}'
              + ('  ← **1面が頁の大半**' if a[0] / cv >= 0.5 else ''))

    dom = [r['source'] for r in rows
           if areas.get(r['source']) and canvas.get(r['source'])
           and max(areas[r['source']]) / canvas[r['source']] >= 0.5]
    print(f'\n  → **1面が頁の半分以上を占める頁: {len(dom)} / 8**')
    for d in dom:
        print(f'     {d[14:]}')
    print(f'\n== 各頁の A0 と帰無値(**支配されている頁に印**)==')
    print(f'  {"頁":42s} {"A0":>7} {"N1":>7} {"N2":>7} {"N3":>7}')
    for r in rows:
        mk = ' ← 1面が支配' if r['source'] in dom else ''
        print(f'  {r["source"][14:]:42s} {float(r["a0_f1"]):>7.4f} '
              f'{float(r["n1_f1"]):>7.4f} {float(r["n2_f1"]):>7.4f} '
              f'{float(r["n3_f1"]):>7.4f}{mk}')

    print(f'\n== A0 の符号検定: 3通り ==')
    for label, sel in (('8頁(いまの数)', rows),
                       ('7頁(`sunshine2\\page0002` を外す)',
                        [r for r in rows if r['source'] != DROP]),
                       ('5頁(**1面が支配する3頁を外す**)',
                        [r for r in rows if r['source'] not in dom])):
        print(f'\n  --- {label} / n={len(sel)} ---')
        for key, nm in (('n1_f1', 'N1 blank'), ('n2_f1', 'N2 whole_page'),
                        ('n3_f1', 'N3 grid')):
            win = sum(1 for r in sel
                      if float(r['a0_f1']) > float(r[key]) + 1e-12)
            los = sum(1 for r in sel
                      if float(r['a0_f1']) + 1e-12 < float(r[key]))
            tie = len(sel) - win - los
            p = m12.judge_sign_test(win, los)
            print(f'    {nm:14s} 勝{win} 負{los} 引{tie}  p = {p:.4f}'
                  f'  {"棒 0.05 を満たす" if p <= 0.05 else "**満たさない**"}')
    print(f'\n  **外した理由**: クリップ群のベースしか無く、クリップ層のアルファが'
          f'届いていないので標準が構成できない(追補32 §B2)。**測定前の欠陥**')
    print(f'  **n=8 → n=7 で検定の力はさらに落ちる。「外したら強くなった」は起きない**'
          f'(符号検定の最小 p は n=7 で {1 / 2 ** 6:.4f})')
    print(f'\n  **標準は (1′)**(マスクのアルファを**こちらが足した重なりの規則**'
          f'(小さい成分が勝つ)に通した後の面)。**作者の Q6 の答は「8連結成分」までで、'
          f'重なりの規則は入っていない。**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
