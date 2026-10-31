#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M99: (b) の標準を作る —— 線が区切った領域を面とする。**

**事前登録**: `doc/prereg_b_standard_20261031.md`(+ **修正1**: `線画` が0枚の頁を外し、
母集団を **`pop_8pages_registered` 8頁 → `pop_6pages_lineart` 6頁**)。
**走る前に commit した**(`612741d` / `9732b8f`)。

  線   = **名前に `線画` を含む可視の<ins>葉</ins>の層**(**作者の裁定**。追補151 §A)
  面   = **墨の補集合の4連結**(前景8連結の双対。**罠 `T-M28`**)
  閉じた面 = **画布の外周に接しない**成分で、**床 1e-5 以上**
  主値 = **P1 = 閉じた面が画布に占める割合(S3、頁ごと)。プールしない**
  棒   = **頁ごとに P1 >= (a) の標準の面が覆う割合。6頁すべてで満たすときだけ通る**
         (**連言は最悪の頁で判定。T-M25**)

**段(同じ走行で全部出す。1つを選んで他を出さない、をやらない)**:

  S0  境界なし(**恒等。閉じた面 0 / 被覆 0%**)
  S1  **線だけ**(裁定3-(i))
  S2  S1 + 遮蔽物 `O_strict`(合成が通常のみ)
  S3  S1 + 遮蔽物 **作者の集合**(乗算でない)—— **主値**
  S4  S1 + 遮蔽物 `O_all`(合成を問わない)
  SA  **可視の全層の墨**(**やりすぎの対照**)

**遮蔽物で「閉じる」= その覆う範囲の<ins>1画素の縁</ins>を境界に足す**(追補151 §C)。
**面を塗り潰すのではない** —— 塗り潰すと覆われた所が面でなくなる。

**フォルダーの `線画` 2枚は線に入れない(走る前に決めて、理由を書く)**:
`3 線画・ディテール` は**子孫0**、`1 <資料> 線画 / 彩色` の**子孫6枚はどれも `線画` でない**
(`レイヤー 1〜5` / `white`)。**入れると塗りの層が線になる。** 除いた数は印字する。

**器具は写さずに import する**: `m12`(床)/ `m16`(標準・木)/ `m77`(非表示)/
`m87`(**`alpha_of` / `subtree_alpha` / `occluder_mask` / `areas_from` / `clip_base_map`**)。

  <venv>/python tools/colour/m99_b_standard.py [--smoke]
"""
import collections
import csv
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402
import m77_standard_visible as m77                                 # noqa: E402
import m87_split_standard_by_occlusion as m87                      # noqa: E402

OUT = pathlib.Path('results/m99_b_standard')
PAPER = m16.PAPER_TYPE
NORMAL, MULTIPLY = '0', '2'
# 修正1 の既知セル。**頁ごとの列まで持つ**(合計だけの門は単位の取り違えを映さない。T-M23)
KC_LINE = {'000_2017_illustration_sunshine2_page0019': 27,
           '000_2017_illustration_sunshine2_page0002': 6,
           '000_2018_illust201710_illust201710_page0003': 5,
           '000_2018_illust201710_illust201710_page0005': 20,
           '000_2018_skima11_skima11_skima12H': 15,
           '000_2018_skima11_skima11_skima12chara': 19}
KC_A_ISLANDS = {'000_2017_illustration_sunshine2_page0019': 77,
                '000_2017_illustration_sunshine2_page0002': 8,
                '000_2018_illust201710_illust201710_page0003': 13,
                '000_2018_illust201710_illust201710_page0005': 81,
                '000_2018_skima11_skima11_skima12H': 220,
                '000_2018_skima11_skima11_skima12chara': 140}
KC_DROPPED = ['022_2018_skima10_skima10_page0064', '022_2018_skima10_skima10_page0073']
STAGES = ('S0', 'S1', 'S2', 'S3', 'S4', 'SA')


def is_line(r):
    """**作者の裁定**: 名前に `線画` を含む層。**葉だけ**(フォルダーは上の但し書き)。"""
    return '線画' in (r['name'] or '') and r['folder'] != 'True'


def inner_edge(m):
    """覆う範囲の**内側1画素の縁**。塗り潰さずに境界だけを足すため。"""
    e = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return m & ~e.astype(bool)


def faces_from(boundary, floor_px):
    """**墨の補集合の4連結**(前景8連結の双対。T-M28)。閉じた面と開いた領域を分けて返す。

    **`connectivity` は必ずキーワードで渡す** —— 第2の位置引数は `labels` で、
    位置で渡すと黙って既定の8連結になる(**T-M28**)。
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        (~boundary).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    closed, open_, under = [], [], 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < floor_px:
            under += 1
            continue
        (open_ if i in border else closed).append(a)
    del lab, st
    return closed, open_, under


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = '--smoke' in sys.argv
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']

    # ---------------------------------------------------- 母集団(修正1)
    keep, dropped = [], []
    for p in pages:
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        n = sum(1 for r in rows_vis if r['layer_type'] != PAPER and is_line(r))
        (keep if n > 0 else dropped).append((p, n))
    print(f'母集団 `pop_6pages_lineart` = {len(keep)} 頁 '
          f'(`pop_8pages_registered` = {len(pages)} 頁)')
    print(f'外した頁 {len(dropped)}(理由: **`線画` の可視の葉が0枚**。規律9)')
    for p, n in dropped:
        print(f'    {p["page"]:44s} 線画 {n}')
    if smoke:
        keep = keep[-2:]

    drops = collections.Counter()
    occ_drop = collections.Counter()
    per_page, t0 = [], time.time()

    for p, n_line in keep:
        rows_all = byp[p['source']]
        rows_vis, d = m77.visible_rows(rows_all)
        drops.update(d)
        _, kids = m16.build_tree(rows_vis)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        px = grid[0] * grid[1]
        floor_px = max(1, round(m12.FLOOR * px))
        cache = {}

        # ---- 線(裁定1・2)。**葉だけ**
        ink = np.zeros(grid, bool)
        n_raster = 0
        for r in rows_vis:
            if r['layer_type'] == PAPER or not is_line(r):
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is None:
                drops['線: ラスタが無い層'] += 1
                continue
            ink[:a.shape[0], :a.shape[1]] |= a
            n_raster += 1

        # ---- 遮蔽物の縁(三組を1回の走査で)。**乗算の扱いだけが違う**
        edges = {k: np.zeros(grid, bool) for k in ('O_strict', 'O_author', 'O_all')}
        n_occ = collections.Counter()
        for r in rows_vis:
            if r['layer_type'] == PAPER:
                continue
            if (r.get('opacity') or '0') == '0':
                occ_drop['不透明度が0'] += 1
                continue
            comp = r.get('composite') or NORMAL
            sets = ['O_all']
            if comp != MULTIPLY:
                sets.append('O_author')          # **乗算は遮蔽物ではない**(作者)
            if comp == NORMAL:
                sets.append('O_strict')
            m = m87.occluder_mask(r, kids, files, grid, cache, cbase,
                                  True, occ_drop)
            if m is None:
                continue
            e = inner_edge(m)
            del m
            for k in sets:
                edges[k] |= e
                n_occ[k] += 1
            del e

        # ---- やりすぎの対照: 可視の全層の墨
        ink_all = np.zeros(grid, bool)
        for r in rows_vis:
            if r['layer_type'] == PAPER or r['folder'] == 'True':
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is not None:
                ink_all[:a.shape[0], :a.shape[1]] |= a

        # ---- (a) の標準(隣に置くだけ。プールしない)
        comps = []
        (_, _), (lab_a, _), _ = m16.measure_standard(p, rows_vis, grid,
                                                     floor_px, out_comps=comps)
        comps = [c for c in comps if c['area'] >= floor_px]
        isl = m87.areas_from(lab_a, comps, floor_px)
        a_area = sum(int(x['area']) for x in isl)
        a_n = len(isl)
        del lab_a, comps, isl
        cache.clear()

        row = {'page': p['page'], 'n_line': n_line, 'n_line_raster': n_raster,
               'canvas_px': px, 'floor_px': floor_px,
               'a_islands': a_n, 'a_cover': a_area / px,
               **{f'n_occ_{k}': n_occ[k] for k in edges}}
        for st_name, bnd in (('S0', np.zeros(grid, bool)),
                             ('S1', ink),
                             ('S2', ink | edges['O_strict']),
                             ('S3', ink | edges['O_author']),
                             ('S4', ink | edges['O_all']),
                             ('SA', ink_all)):
            closed, open_, under = faces_from(bnd, floor_px)
            row[f'{st_name}_closed_n'] = len(closed)
            row[f'{st_name}_cover'] = sum(closed) / px
            row[f'{st_name}_open_n'] = len(open_)
            row[f'{st_name}_open_max'] = (max(open_) / px) if open_ else 0.0
            row[f'{st_name}_under_floor'] = under
            del bnd
        del ink, ink_all, edges
        per_page.append(row)
        print(f'  {p["page"]:44s} 線 {n_raster:>3} '
              f'S1 {row["S1_closed_n"]:>4}面 {row["S1_cover"]*100:5.2f}% | '
              f'**S3 {row["S3_closed_n"]:>4}面 {row["S3_cover"]*100:5.2f}%** | '
              f'S4 {row["S4_closed_n"]:>4}面 {row["S4_cover"]*100:5.2f}% | '
              f'(a) {row["a_islands"]:>3}面 {row["a_cover"]*100:5.2f}%  '
              f'{time.time()-t0:6.1f}s', flush=True)

    # ------------------------------------------------------------------ 門
    print('\n== `standard_visible` が落とす行(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:32s} {v:>6}')
    print('== 遮蔽物で落ちた層(理由別。規律9)==')
    for k, v in occ_drop.most_common():
        print(f'  {k:32s} {v:>6}')

    print('\n== 門(落ちたら以降の行を読まない)==')
    g = {}
    g['IC1 頁'] = ('—' if smoke else 6, len(keep))
    for p, _ in keep:
        want = KC_LINE[p['page']]
        got = [r for r in per_page if r['page'] == p['page']][0]['n_line']
        g[f'IC1 線画 {p["page"][-12:]}'] = (want, got)
        g[f'IC2 (a)島 {p["page"][-12:]}'] = (
            KC_A_ISLANDS[p['page']],
            [r for r in per_page if r['page'] == p['page']][0]['a_islands'])
    g['IC3 S0 の閉じた面(合計)'] = (0, sum(r['S0_closed_n'] for r in per_page))
    g['IC3 S0 の被覆(合計)'] = (0.0, sum(r['S0_cover'] for r in per_page))
    g['IC6 外した頁の数'] = (2 if not smoke else '—', len(dropped))
    g['IC6 外した頁の名'] = (KC_DROPPED if not smoke else '—',
                            sorted(p['page'] for p, _ in dropped))
    ok_all = True
    for k, (want, got) in g.items():
        ok = want == got
        ok_all &= ok or want == '—'
        print(f'  {k:34s} 期待 {str(want)[:28]:>28} / 実測 {str(got)[:28]:>28}  '
              f'{"PASS" if ok else ("—" if want == "—" else "**FAIL**")}')
    if not ok_all:
        print('\n**門が落ちた。以降の行を読まない**(T-M22)')

    # ------------------------------------------------------------------ 判定
    print('\n== C1(主): 頁ごとに P1(S3) >= (a) の被覆。**最悪の頁で判定** ==')
    fail = []
    for r in per_page:
        ok = r['S3_cover'] >= r['a_cover']
        if not ok:
            fail.append(r['page'])
        print(f'  {r["page"]:44s} S3 {r["S3_cover"]*100:6.2f}% / '
              f'(a) {r["a_cover"]*100:6.2f}%  {"PASS" if ok else "**FAIL**"}')
    print(f'  → **C1 {"PASS" if not fail else "FAIL"}**(満たさない頁 {len(fail)}/{len(per_page)})')

    print('\n== C2: 遮蔽物の寄与 P1(S4) − P1(S1) ==')
    pos = 0
    for r in per_page:
        d = r['S4_cover'] - r['S1_cover']
        pos += d > 0
        print(f'  {r["page"]:44s} {d*100:+7.3f} 百分点  '
              f'(S1 {r["S1_cover"]*100:5.2f}% → S4 {r["S4_cover"]*100:5.2f}%)')
    print(f'  → **寄与が正の頁 {pos}/{len(per_page)}**'
          f'{"(0 = 裁定3-(ii) はこの材料では効かない)" if pos == 0 else ""}')

    print('\n== C3: S4 でも最大の開いた領域が画布の過半か ==')
    half = 0
    for r in per_page:
        over = r['S4_open_max'] > 0.5
        half += over
        print(f'  {r["page"]:44s} 最大の開いた領域 {r["S4_open_max"]*100:6.2f}%  '
              f'{"**過半**" if over else "過半でない"}')
    print(f'  → 過半の頁 {half}/{len(per_page)}'
          f'{"  **全頁 = (b) は literal には分割しない。作者へ戻す**" if half == len(per_page) else ""}')

    print('\n== 副値(頁ごと。プールしない)==')
    print(f'{"page":44s} ' + ' '.join(f'{s+"面":>7}' for s in STAGES)
          + ' ' + ' '.join(f'{s+"%":>7}' for s in STAGES))
    for r in per_page:
        print(f'{r["page"]:44s} '
              + ' '.join(f'{r[f"{s}_closed_n"]:>7}' for s in STAGES) + ' '
              + ' '.join(f'{r[f"{s}_cover"]*100:>7.2f}' for s in STAGES))

    with (OUT / 'b_standard.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_page[0].keys()))
        w.writeheader()
        w.writerows(per_page)
    print(f'\nwrote {OUT}/b_standard.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
