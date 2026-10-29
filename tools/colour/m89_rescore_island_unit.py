#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補133 の事前登録を走らせる —— 面を「連結で数えた島」にして A0 を採点し直す。**

**作者が 2026-10-28 に面の数え方を裁定した**(横切られて2つに分かれたラベルを見せて
問うたところ「**二つの面**」)。**追補112 の点数はラベルの数 483 の上の値**なので、
**作者が採った単位(島 692)の上で測り直す。**

**事前登録(追補133。走る前に書いて commit した)**:

  母集団 = **登録集合8頁のまま**(広げない。狭めない)
  腕     = `label_483`(追補112 の再現。既知セル)と **`island_692`(主値)**を同じ走行で
  主値   = **`island_692` の上での A0 の region-F1(頁ごと)**
  副値   = recall / precision / **合った面の数** / A0 の面 / A0の面÷標準の面
  帰無値 = N1 blank / N2 whole_page / N3 grid(**追補7 のまま。緩めない**)
  棒     = A0 が N1・N2・N3 の**すべて**に頁ごとの符号検定で勝つ(両側 α=0.05)。
           **n=8 では 8/8 のときだけ通る**
  床     = **1e-5**(標準側と A0 側で同じ値)。**1e-4 は走らせない**(追補133 §E)
  門     = KC-1〜KC-7(§F)。**落ちたら以降の行を読まない**

**器具は写さずに import する** —— **`m87.areas_from` が 692 の定義**である。書き直さない。

  <venv>/python tools/colour/m89_rescore_island_unit.py
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

OUT = pathlib.Path('results/m89_rescore_island')

# **既知セルは世代ごとに1つ**(規律2 の付帯)。**頁ごとの列まで持つ**(T-M23: 合計だけの
# 門は単位の取り違えを映さない)。並びは manifest の順
KC_LABEL = [67, 5, 13, 66, 82, 0, 143, 107]      # 追補112 §C の `面 all→vis` の vis 側
KC_ISLAND = [77, 8, 13, 81, 153, 0, 220, 140]    # 追補125 の `面(連結)` の列
KC_TOT = {'bases': 152, 'components': 503, 'label': 483, 'island': 692}


def island_labels(lab, comps, floor_px, grid):
    """**島の単位のラベル地図と面積表**。面の定義は `m87.areas_from` から借りる。

    `areas_from` は島ごとに `box` と `mask` を返す。**ここで地図に塗り直すだけ**で、
    **どれが面かの判断は一切していない**(判断を2か所に置かないため)。
    """
    areas = m87.areas_from(lab, comps, floor_px)
    out = np.zeros(grid, np.int32)
    garea = {}
    for i, a in enumerate(areas, start=1):
        y, x, h, w = a['box']
        sub = out[y:y + h, x:x + w]
        sub[a['mask']] = i
        garea[i] = int(a['area'])
    del areas
    return out, garea


def score_unit(gt, garea, a0, s, grid, floor_frac, pw, ph, floor_px):
    """1頁・1単位。A0 と3つの帰無値を同じ標準に当てる。**m81 と同じ順序**。"""
    ic_m6, _ = m12.score_from_small(gt, 1, gt, garea, floor_px)
    n1 = np.zeros((ph, pw), np.int32)
    n2 = np.ones((ph, pw), np.int32)
    n3, gr, gc = m12.grid_null((ph, pw), len(garea))
    r = {}
    for k, pred in (('a0', a0), ('n1', n1), ('n2', n2), ('n3', n3)):
        r[k], _ = m12.score_from_small(pred, s, gt, garea, floor_px)
    del n1, n2, n3
    return {'areas': len(garea), 'ic_m6_f1': round(ic_m6['region_f1'], 6),
            'n3_shape': f'{gr}x{gc}', 'a0_n': r['a0']['n_pred'],
            'matched': r['a0']['matched'],
            'prec': r['a0']['precision'], 'rec': r['a0']['recall'],
            'a0_f1': r['a0']['region_f1'], 'n1_f1': r['n1']['region_f1'],
            'n2_f1': r['n2']['region_f1'], 'n3_f1': r['n3']['region_f1']}


def sign_tests(rows, unit):
    out = {}
    for n in ('n1', 'n2', 'n3'):
        w = sum(1 for r in rows if r[unit]['a0_f1'] > r[unit][f'{n}_f1'])
        l = sum(1 for r in rows if r[unit]['a0_f1'] < r[unit][f'{n}_f1'])
        out[n] = (w, l, m12.judge_sign_test(w, l))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    floor_frac = m12.FLOOR
    print(f'頁 {len(pages)}  ラスタ {m12.D}  床 {floor_frac:g}  '
          f'**主値は島の単位(692)**')

    drops = collections.Counter()
    per_page, t0 = [], time.time()
    for p in pages:
        rows_all = byp[p['source']]
        rows_vis, d = m77.visible_rows(rows_all)
        drops.update(d)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(floor_frac * grid[0] * grid[1]))

        comps = []
        (_, _), (lab_a, garea_l), st = m16.measure_standard(
            p, rows_vis, grid, floor_px, out_comps=comps)
        comps = [c for c in comps if c['area'] >= floor_px]
        lab_i, garea_i = island_labels(lab_a, comps, floor_px, grid)

        im = cv2.imread(str(m12.D / p['preview']['file']), cv2.IMREAD_COLOR)
        a0_lab = m12.equal_value_components(m12.quantise(im, m12.LEVELS))
        a0, _ = m12.filtered_labels(a0_lab, max(1, round(floor_frac * pw * ph)))
        ic_m7 = m12.ic_m7(a0, s, grid, floor_px)
        del im, a0_lab

        row = {'page': p['page'], 'bases': st['bases'],
               'components': st['components'],
               'ic_m7_f1': round(ic_m7['region_f1'], 6),
               'label': score_unit(lab_a, garea_l, a0, s, grid, floor_frac,
                                   pw, ph, floor_px),
               'island': score_unit(lab_i, garea_i, a0, s, grid, floor_frac,
                                    pw, ph, floor_px)}
        del lab_a, lab_i, a0
        per_page.append(row)
        L, I = row['label'], row['island']
        print(f'  {p["page"]:44s} 面 {L["areas"]:>4}->{I["areas"]:>4} '
              f'合った {L["matched"]:>3}->{I["matched"]:>3} '
              f'rec {L["rec"]:.4f}->{I["rec"]:.4f} '
              f'prec {L["prec"]:.4f}->{I["prec"]:.4f} '
              f'F1 {L["a0_f1"]:.4f}->{I["a0_f1"]:.4f} '
              f'{time.time()-t0:6.1f}s', flush=True)

    print('\n== `standard_visible` が落とす層(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:32s} {v:>6}')

    # ---------------------------------------------------------------- 門
    print('\n== 門(追補133 §F。落ちたら以降の行を読まない)==')
    g = {}
    tb = sum(r['bases'] for r in per_page)
    tc = sum(r['components'] for r in per_page)
    tl = sum(r['label']['areas'] for r in per_page)
    ti = sum(r['island']['areas'] for r in per_page)
    for nm, want, got in (('KC-1 ベース', KC_TOT['bases'], tb),
                          ('KC-1 成分', KC_TOT['components'], tc),
                          ('KC-1 面(ラベル)', KC_TOT['label'], tl),
                          ('KC-2 面(島)', KC_TOT['island'], ti)):
        ok = want == got
        g[nm] = ok
        print(f'  {nm:20s} 期待 {want:>5} / 実測 {got:>5}  {"PASS" if ok else "**FAIL**"}')
    for nm, kc, key in (('KC-3 頁ごと(島)', KC_ISLAND, 'island'),
                        ('KC-4 頁ごと(ラベル)', KC_LABEL, 'label')):
        bad = [(r['page'], w, r[key]['areas'])
               for r, w in zip(per_page, kc) if r[key]['areas'] != w]
        g[nm] = not bad
        print(f'  {nm:20s} 合わない頁 {len(bad):>3}  '
              f'{"PASS" if not bad else "**FAIL**"}')
        for pg, w, got in bad:
            print(f'      {pg:44s} 期待 {w} / 実測 {got}')
    for nm, key in (('KC-5 IC-M6(ラベル)', ('label', 'ic_m6_f1')),
                    ('KC-5 IC-M6(島)', ('island', 'ic_m6_f1'))):
        bad = [r['page'] for r in per_page if abs(r[key[0]][key[1]] - 1.0) > 1e-9]
        g[nm] = not bad
        print(f'  {nm:20s} 1.0 でない頁 {len(bad):>3}  '
              f'{"PASS" if not bad else "**FAIL**"}')
    bad = [r['page'] for r in per_page if abs(r['ic_m7_f1'] - 1.0) > 1e-9]
    g['KC-6 IC-M7'] = not bad
    print(f'  {"KC-6 IC-M7":20s} 1.0 でない頁 {len(bad):>3}  '
          f'{"PASS" if not bad else "**FAIL**"}')
    bad = [(r['page'], r['label']['a0_n'], r['island']['a0_n'])
           for r in per_page if r['label']['a0_n'] != r['island']['a0_n']]
    g['KC-7 A0 の面が同一'] = not bad
    print(f'  {"KC-7 A0 の面が同一":20s} 違う頁 {len(bad):>3}  '
          f'{"PASS" if not bad else "**FAIL**"}  '
          f'(**標準は腕の入力ではない**)')
    for pg, a, b in bad:
        print(f'      {pg:44s} ラベル {a} / 島 {b}')

    if all(g.values()):
        print('\n  **門 すべて PASS**')
    else:
        print('\n  **門が落ちた: ' + ', '.join(k for k, v in g.items() if not v)
              + '。以降の行は読まない**')

    # ---------------------------------------------- 主値(頁ごと。プールしない)
    print('\n== 主値(**頁ごと。プールしない**。左=ラベル483 / 右=島692)==')
    hdr = (f'  {"頁":44s} {"面":>11} {"合った":>9} {"recall":>15} '
           f'{"precision":>17} {"A0 の F1":>15} {"N3 格子":>13}')
    print(hdr)
    for r in per_page:
        L, I = r['label'], r['island']
        print(f'  {r["page"]:44s} {L["areas"]:>4}->{I["areas"]:>4} '
              f'  {L["matched"]:>3}->{I["matched"]:>3} '
              f'  {L["rec"]:.4f}->{I["rec"]:.4f} '
              f'  {L["prec"]:.4f}->{I["prec"]:.4f} '
              f'  {L["a0_f1"]:.4f}->{I["a0_f1"]:.4f} '
              f'  {L["n3_shape"]:>5}->{I["n3_shape"]:<5}')

    print('\n== 合計と幅(**幅で書く。平均しない**)==')
    for key, nm in (('label', 'ラベル 483'), ('island', '**島 692(主値)**')):
        a0n = sum(r[key]['a0_n'] for r in per_page)
        mt = sum(r[key]['matched'] for r in per_page)
        ar = sum(r[key]['areas'] for r in per_page)
        rc = [r[key]['rec'] for r in per_page]
        pc = [r[key]['prec'] for r in per_page]
        f1 = [r[key]['a0_f1'] for r in per_page]
        print(f'  {nm:22s} 標準の面 {ar:>4} / A0 の面 {a0n:>6} '
              f'/ **合った面 {mt:>4}** / A0÷標準 {a0n/ar:>6.1f}倍')
        print(f'  {"":22s} recall {min(rc):.4f}〜{max(rc):.4f} '
              f'/ precision {min(pc):.4f}〜{max(pc):.4f} '
              f'/ F1 {min(f1):.4f}〜{max(f1):.4f}')

    print('\n== 棒(**追補133 §E。追補111 §D から写した。1文字も変えていない**)==')
    for key, nm in (('label', 'ラベル 483'), ('island', '**島 692(主値)**')):
        t = sign_tests(per_page, key)
        # **棒は「3つすべてに勝つ」なので、効いているのは最悪の p である。**
        # 最初の版は `min`(最良)で判定を書いていた —— **1つ通れば「満たす」と
        # 印字していた**(島の単位で N3 だけ通ったときに踏んだ)。**最良は参考、
        # 判定は最悪**。追補134 §A
        best = min(t[n][2] for n in t)
        worst = max(t[n][2] for n in t)
        print(f'  {nm}')
        for n, lbl in (('n1', 'N1 blank'), ('n2', 'N2 whole_page'),
                       ('n3', 'N3 grid')):
            w, l, pv = t[n]
            print(f'    {lbl:14s} {w}勝{l}敗  p={pv:.4f}  '
                  f'{"通る" if pv < 0.05 else "通らない"}')
        print(f'    参考: 最良 p = {best:.4f}(**判定には使わない**)')
        verdict = '満たす' if worst < 0.05 else '**満たさない**'
        print(f'    **判定に効く最悪 p = {worst:.4f} / '
              f'棒「3つすべてに勝つ」は {verdict}**')

    with open(OUT / 'rescore_island_unit.csv', 'w', encoding='utf-8',
              newline='') as f:
        cols = ['page', 'bases', 'components', 'ic_m7_f1']
        sub = ['areas', 'matched', 'rec', 'prec', 'a0_f1', 'a0_n',
               'n1_f1', 'n2_f1', 'n3_f1', 'n3_shape', 'ic_m6_f1']
        w = csv.writer(f)
        w.writerow(cols + [f'{u}_{c}' for u in ('label', 'island') for c in sub])
        for r in per_page:
            w.writerow([r[c] for c in cols]
                       + [r[u][c] for u in ('label', 'island') for c in sub])
    print(f'\nwrote {OUT}/rescore_island_unit.csv')


if __name__ == '__main__':
    main()
