#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補89 —— 遮蔽物が既知セルに在るかを数える(作者の言明の回送を受けて)。**

作者の言明(census track 経由): **3D を判定したいのは遮蔽物だからである。手前に何かが
あると、その後ろの層では描かないで済ます。** 遮蔽物は4種(**3D / 独立したフキダシ層 /
`white` という名の層 / 黒消し**)。

**標準は「ベースの子孫のアルファの8連結成分(床 1e-5)」**なので、**手前の層で隠れる領域は
下の層に無くてよい** —— つまり**「分けすぎ」の一部が正しい分割である可能性が在る**。
**可能性までしか言わない**(母集団をこの変数で分けた測定は無い)。

腕1(**種別**): `has_3d` の列。**帰無値 0**。**これだけに棒を置く。**
腕2(**名前・仮の語彙**): **棒を置かない。記述のみ** —— 語彙は作者が選ぶ前で、
**外せば見落とし、広ければ過検出**。**どちらの向きにも使えない。**

  <venv>/python tools/colour/m69_occluder_census.py
"""
import collections
import csv
import json
import pathlib
import sys

SRC = pathlib.Path('../lineart/inbox/m_request6b_layer_fields_872pages.csv')
PAGES = pathlib.Path('results/alias/occluder_check_pages.json')
PATS = pathlib.Path('results/alias/occluder_name_patterns.json')
OUT = pathlib.Path('results/m69_occluder')


def load(p, what):
    if not p.exists():
        raise SystemExit(f'**{what}が無い: {p}。走らせない**')
    return json.loads(p.read_text(encoding='utf-8'))


def truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'none')


def main():
    if not SRC.exists():
        raise SystemExit(f'**材料が無い: {SRC}**')
    pg = load(PAGES, '頁の道の表')
    cat = load(PATS, '仮の語彙')['categories']
    known = set(pg['known8'])
    p45 = set(pg['p45'])

    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    pages = {r['page_path'] for r in rows}
    print(f'材料 {SRC}  層 {len(rows)}  頁 {len(pages)}')
    print(f'被覆: 既知セル {len(known & pages)}/{len(known)}  '
          f'45頁 {len(p45 & pages)}/{len(p45)}')
    if len(known & pages) != len(known) or len(p45 & pages) != len(p45):
        print('  **被覆が欠けている。結果を読まない**')
        return 1

    nonfolder = [r for r in rows if not truthy(r['folder'])]
    print(f'フォルダーを外した層 {len(nonfolder)}  '
          f'(フォルダー {len(rows) - len(nonfolder)})')

    strata = [('既知セル7頁', known), ('45頁', p45), ('872頁(全体)', pages)]

    # ---- 腕1: 種別 `has_3d`。**ここだけに棒を置く(帰無値 0)** ----
    print('\n== 腕1(**棒あり**): `has_3d` の列。帰無値 0 ==')
    print(f'  {"層別":16s} {"層":>7} {"3D層":>6} {"3Dの頁":>8} {"頁":>5}')
    arm1 = {}
    for lab, ps in strata:
        sub = [r for r in nonfolder if r['page_path'] in ps]
        d3 = [r for r in sub if truthy(r['has_3d'])]
        p3 = {r['page_path'] for r in d3}
        arm1[lab] = (len(sub), len(d3), len(p3), len(ps))
        print(f'  {lab:16s} {len(sub):>7} {len(d3):>6} {len(p3):>8} {len(ps):>5}')

    # 年は畳まない(世代もプール。行ごとに印字する)
    print('\n== 腕1 の年ごと(**畳まない**。全体の 3D 層)==')
    by_year = collections.defaultdict(lambda: [0, 0, set(), set()])
    for r in nonfolder:
        parts = r['page_path'].replace('/', '\\').split('\\')
        y = next((x for x in parts if x.isdigit() and len(x) == 4), '不明')
        b = by_year[y]
        b[0] += 1
        b[3].add(r['page_path'])
        if truthy(r['has_3d']):
            b[1] += 1
            b[2].add(r['page_path'])
    print(f'  {"年":6s} {"層":>7} {"3D層":>6} {"3Dの頁":>8} {"頁":>5}')
    for y in sorted(by_year):
        b = by_year[y]
        print(f'  {y:6s} {b[0]:>7} {b[1]:>6} {len(b[2]):>8} {len(b[3]):>5}')

    # ---- 腕2: 名前。**棒を置かない。記述のみ** ----
    print('\n== 腕2(**棒なし・記述のみ**): 仮の語彙での名前の当たり ==')
    print('  **これは定義ではない。0 は「無い」ではなく「この語彙が見つけなかった」**')
    arm2 = {}
    for lab, ps in strata:
        sub = [r for r in nonfolder if r['page_path'] in ps]
        line = []
        for cname, pats in sorted(cat.items()):
            hit = [r for r in sub
                   if any(p.lower() in (r['name'] or '').lower() for p in pats)]
            line.append((cname, len(hit), len({r['page_path'] for r in hit})))
        arm2[lab] = line
        print(f'  {lab:16s} ' + '  '.join(
            f'{c}: 層 {n} / 頁 {p}' for c, n, p in line))

    # 書き出し。**名前は出さない**(作者の作業名)。層の ID と頁の道だけ
    OUT.mkdir(parents=True, exist_ok=True)
    det = []
    for r in nonfolder:
        hits = [c for c, pats in sorted(cat.items())
                if any(p.lower() in (r['name'] or '').lower() for p in pats)]
        if truthy(r['has_3d']) or hits:
            det.append({'page_path': r['page_path'], 'layer_id': r['layer_id'],
                        'has_3d': truthy(r['has_3d']),
                        'name_categories': '|'.join(hits),
                        'layer_type': r['layer_type'], 'depth': r['depth'],
                        'parent_id': r['parent_id'],
                        'known_cell': r['page_path'] in known,
                        'in45': r['page_path'] in p45})
    with (OUT / 'occluder_candidates.csv').open('w', newline='',
                                                encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(det[0]))
        w.writeheader()
        w.writerows(det)
    print(f'\nwrote {OUT}/occluder_candidates.csv  行 {len(det)}'
          '  (**名前の列は入れない**。層の ID で DB と照らせる)')

    print('\n**読み方(結果を見る前に登録した意味。追補89 §C)**')
    print('  既知セルの `has_3d` = 0 → **3Dに限って**遮蔽されていない。'
          '**他の3種は未判定**')
    print('  既知セルの `has_3d` > 0 → **標準の値 183/823/801 は変わらない**'
          '(標準は定義)。変わるのは**「分けすぎ」の読み**だけ')
    print('  **遮蔽の測定は標準の読みにしか使えない。腕の点数を上げる道具ではない**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
