#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M96: 原案「塗りの入っていないマスク層だけで面を見つける」を走らせる。**

**事前登録は `doc/prereg_mask_only_division_20261031.md`(修正1 まで)で、
点数を1件も読む前に commit した**(`ff2741d` / `f8de335`)。

  問い   = **D3′ の層の島だけで、作者が既に「面」と裁定したものを落とさずに
           面の分割が取れるか**(「いいえ」で答えられる)
  腕     = **A_mask(D3′ 113層。主腕)** / A_mask_tone(D3 177層。並べる)/
           A_mask∩base(104層。参考)/ A_std(= 標準。恒等)/ N1 / N2 / N3
  主値   = **P1 = 原案が落とした島のうち、作者が「数で答えた」名前のベース由来の島の数**
           (**頁ごと**。プールしない)。**棒 C1 = 0**
  副値   = 原案の島 / 標準の島 / 落とした島(裁定の級ごと)/ 足した島 /
           **参考の region-F1**(**標準と材料が 104/152 重なるので主値にしない**)
  床     = **1e-5**(標準側と腕側で同じ値)

**但し書き(T-M26 で落ちた形。数の隣に必ず置く)**: 標準も**原案の島も**面を束ねている
(作者「いくつかの面は一つのマスクであつかう」追補31)。→ **面の数は下界、
分けすぎの倍率は上界、precision は下界。**

**器具は写さずに import する**: `m12`(帰無値・採点)/ `m16`(標準の作り方と同定)/
`m77`(非表示の規則)/ `m87.areas_from`(**島 = 面の定義**)。**どれも書き直さない。**
腕の層の集合を差し替えるために `m16.tree_bases` を**一時的に置き換える** ——
**標準の作り方を2か所に書かないため**(面の判断は `areas_from` 1か所のまま)。

  <venv>/python tools/colour/m96_mask_only_division.py
"""
import collections
import contextlib
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

OUT = pathlib.Path('results/m96_mask_only')

# 既知セル。**頁ごとの列まで持つ**(T-M23: 合計だけの門は単位の取り違えを映さない)
KC_ISLAND = [77, 8, 13, 81, 153, 0, 220, 140]          # 標準の島(`m89` の列)
KC_IC1 = {'rows': 1005, 'vis': 756, 'bases': 152, 'd3': 177, 'tone': 64,
          'd3n': 113, 'both': 104, 'b_only': 48, 'd3_only': 9}      # 修正1
KC_NC1 = 38       # 非可視の D3′ 層(修正1)
KC_NC3 = 8        # 用紙層は8枚とも D3(修正1)

# **作者の裁定の級**(追補38 §A・追補39 §C)。**名前は裁定を引く鍵で、同定の篩ではない**
RULED_FACE = {'MASK', '線画 hair'}                 # 数で答えた = 面
RULED_NOT = {'Sketch', 'p', 'レイヤー 10'}          # 面でない
PAPER = m16.PAPER_TYPE


def is_d3(r):
    """**塗りの入っていない層**: 画素が色を持たず、色は層の描画色の側にある形。"""
    return str(r.get('draw_color_enable') or '') == '1'


def is_tone(r):
    """**トーン層**。網点は層の画素に無い(T-M1)ので、その画素は面のマスクではない。"""
    return str(r.get('tone_enabled')) == 'True'


def ruling(name):
    return ('面(数で答えた)' if name in RULED_FACE
            else '面でない' if name in RULED_NOT else '未回答')


@contextlib.contextmanager
def bases_replaced(keep):
    """`m16.tree_bases` の返すベースを `keep` の層に差し替える。

    **標準の作り方(重なりの解き方・フォルダーの子孫の和・床)はそのまま使う。**
    差し替えるのは「どの層をベースと呼ぶか」だけである。
    """
    real = m16.tree_bases

    def patched(rows):
        bases, kids = real(rows)
        sel = {int(r['layer_id']): r for r in rows
               if int(r['layer_id']) in keep and r['layer_type'] != PAPER}
        return sel, kids

    m16.tree_bases = patched
    try:
        yield
    finally:
        m16.tree_bases = real


def islands(p, rows_vis, grid, floor_px, keep=None):
    """1頁の島(= 面)を返す。`keep` を渡すと、その層をベースとして作る。"""
    comps = []
    ctx = bases_replaced(keep) if keep is not None else contextlib.nullcontext()
    with ctx:
        (_, _), (lab, _), st = m16.measure_standard(
            p, rows_vis, grid, floor_px, out_comps=comps)
    comps = [c for c in comps if c['area'] >= floor_px]
    isl = m87.areas_from(lab, comps, floor_px)
    out = np.zeros(grid, np.int32)
    garea = {}
    for i, a in enumerate(isl, start=1):
        y, x, h, w = a['box']
        out[y:y + h, x:x + w][a['mask']] = i
        garea[i] = int(a['area'])
    bases = [a['base'] for a in isl]
    del isl, lab
    return out, garea, bases, comps, st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    smoke = len(sys.argv) > 1 and sys.argv[1] == '--smoke'
    if smoke:          # **先に smoke test**(CLAUDE.md §6)。門は印字するが判定しない
        pages = pages[2:3]
    floor_frac = m12.FLOOR
    print(f'頁 {len(pages)}  ラスタ {m12.D}  床 {floor_frac:g}')
    print('**主値 P1 = 原案が落とした島のうち、作者が「数で答えた」ベース由来の島の数**')
    print('**但し書き: 標準も原案の島も面を束ねている → 面の数は下界、precision は下界**\n')

    ic1 = collections.Counter()
    drops = collections.Counter()
    nc = collections.Counter()
    per_page, t0 = [], time.time()

    for p, kc_isl in zip(pages, KC_ISLAND[2:3] if smoke else KC_ISLAND):
        rows_all = byp[p['source']]
        rows_vis, d = m77.visible_rows(rows_all)
        drops.update(d)
        bases_std, _ = m16.tree_bases(rows_vis)
        body = [r for r in rows_vis if r['layer_type'] != PAPER]
        d3 = {int(r['layer_id']) for r in body if is_d3(r)}
        tone = {int(r['layer_id']) for r in body if is_d3(r) and is_tone(r)}
        d3n = d3 - tone
        bid = set(bases_std)

        ic1.update({'rows': len(rows_all), 'vis': len(rows_vis),
                    'bases': len(bid), 'd3': len(d3), 'tone': len(tone),
                    'd3n': len(d3n), 'both': len(d3n & bid),
                    'b_only': len(bid - d3n), 'd3_only': len(d3n - bid)})
        # 負の対照は**実際に数える**(向きではなく数)
        vis_ids = {id(r) for r in rows_vis}
        nc['nc1'] += sum(1 for r in rows_all
                         if r['layer_type'] != PAPER and id(r) not in vis_ids
                         and is_d3(r) and not is_tone(r))
        nc['nc3'] += sum(1 for r in rows_all if r['layer_type'] == PAPER and is_d3(r))
        nc['ic6'] += sum(1 for r in bases_std.values() if is_tone(r))

        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(floor_frac * grid[0] * grid[1]))

        # ---- 標準(A_std)。**島が面**
        gt, garea, gt_bases, _, st = islands(p, rows_vis, grid, floor_px)
        # ---- 腕。**同じ走行で3つ出す。差し替えない**
        arms = {}
        for nm, keep in (('A_mask', d3n), ('A_mask_tone', d3),
                         ('A_mask_cap_base', d3n & bid)):
            lab, ga, ab, _, _ = islands(p, rows_vis, grid, floor_px, keep=keep)
            r, _ = m12.score_from_small(lab, 1, gt, garea, floor_px)
            # **足した島 = 腕の島のうち、ベースが標準のベース集合に無いもの**
            # (= D3′∖ベース の層から出た島)。**落とした島と足して1つにしない。NC6**
            arms[nm] = {'areas': len(ga), 'matched': r['matched'],
                        'rec': r['recall'], 'prec': r['precision'],
                        'f1': r['region_f1'],
                        'added': sum(1 for x in ab if x is not None and x not in bid)}
            del lab, ga
        ic_m6, _ = m12.score_from_small(gt, 1, gt, garea, floor_px)
        n1 = np.zeros((ph, pw), np.int32)
        n2 = np.ones((ph, pw), np.int32)
        n3, gr, gc = m12.grid_null((ph, pw), len(garea))
        nulls = {}
        for nm, pred in (('N1', n1), ('N2', n2), ('N3', n3)):
            r, _ = m12.score_from_small(pred, s, gt, garea, floor_px)
            nulls[nm] = r['region_f1']
        del n1, n2, n3

        # ---- 主値 P1: 落とした島を**裁定の級ごと**に数える
        dropped = collections.Counter()
        for b in gt_bases:
            if b is None or b in d3n:
                continue
            dropped[ruling(bases_std[b]['name'] if b in bases_std else '')] += 1
        added = arms['A_mask']['added']

        row = {'page': p['page'], 'std_areas': len(garea), 'kc_island': kc_isl,
               'ic_m6_f1': round(ic_m6['region_f1'], 6),
               'n3_shape': f'{gr}x{gc}',
               'P1': dropped['面(数で答えた)'],
               'dropped_not_face': dropped['面でない'],
               'dropped_unruled': dropped['未回答'],
               'dropped_total': sum(dropped.values()),
               'added_islands': added,
               **{f'{k}_{m}': v[m] for k, v in arms.items()
                  for m in ('areas', 'matched', 'rec', 'prec', 'f1')},
               **{f'{k}_f1': v for k, v in nulls.items()}}
        per_page.append(row)
        del gt
        print(f'  {p["page"]:44s} 標準 {row["std_areas"]:>4} '
              f'原案 {arms["A_mask"]["areas"]:>4} '
              f'**P1 {row["P1"]:>3}** 落とした島 {row["dropped_total"]:>4} '
              f'(面でない {dropped["面でない"]:>3} / 未回答 {dropped["未回答"]:>4}) '
              f'足した島 {added:>3}  {time.time()-t0:6.1f}s', flush=True)

    # ------------------------------------------------------------------ 門
    print('\n== `standard_visible` が落とす行(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:32s} {v:>6}')

    print('\n== 門(落ちたら以降の行を読まない)==')
    gates = {}
    for k, want in KC_IC1.items():
        gates[f'IC1 {k}'] = ('—' if smoke else want, ic1[k])
    bad = [(r['page'], r['kc_island'], r['std_areas']) for r in per_page
           if r['std_areas'] != r['kc_island']]
    gates['IC2 頁ごとの島'] = (0, len(bad))
    gates['IC2 島の合計'] = ('—' if smoke else sum(KC_ISLAND),
                           sum(r['std_areas'] for r in per_page))
    gates['IC3 IC-M6=1.0 でない頁'] = (
        0, sum(1 for r in per_page if abs(r['ic_m6_f1'] - 1.0) > 1e-9))
    if not smoke:
        gates['IC5 page0073 の標準の面'] = (0, per_page[5]['std_areas'])
    gates['IC6 tone のベース'] = (0, nc['ic6'])
    gates['NC1 非可視の D3′ 層'] = ('—' if smoke else KC_NC1, nc['nc1'])
    gates['NC3 用紙層のうち D3'] = ('—' if smoke else KC_NC3, nc['nc3'])
    ok_all = True
    for k, (want, got) in gates.items():
        ok = want == got
        ok_all &= ok or want == '—'
        print(f'  {k:28s} 期待 {str(want):>5} / 実測 {got:>5}  '
              f'{"PASS" if ok else ("—" if want == "—" else "**FAIL**")}')
    for pg, want, got in bad:
        print(f'      {pg:44s} 期待 {want} / 実測 {got}')
    if not ok_all:
        print('\n**門が落ちた。以降の行を読まない**(T-M22: 器具の誤りか登録の前提の誤りかを'
              '決めてから直す)')

    # ------------------------------------------------------------------ 判定
    print('\n== C1(主): P1 = 0 を頁ごとに判定する ==')
    fail = [r for r in per_page if r['P1'] > 0]
    for r in per_page:
        print(f'  {r["page"]:44s} P1 {r["P1"]:>3}  '
              f'{"PASS" if r["P1"] == 0 else "**FAIL**"}')
    print(f'  合計 P1 = {sum(r["P1"] for r in per_page)}  '
          f'/ FAIL の頁 {len(fail)} / 8  → '
          f'**C1 {"PASS" if not fail else "FAIL"}**')

    print('\n== C3: 原案と標準がほぼ一致していないか(落とした島・足した島が標準の 5% 未満)==')
    for r in per_page:
        tot = r['std_areas']
        if tot == 0:
            print(f'  {r["page"]:44s} 標準 0 —— 判定の外(IC5)')
            continue
        dr, ar = r['dropped_total'] / tot, r['added_islands'] / tot
        print(f'  {r["page"]:44s} 落とした {dr:6.1%} / 足した {ar:6.1%}  '
              f'{"**一致(C3 当たり)**" if dr < 0.05 and ar < 0.05 else "一致せず"}')

    print('\n== 副値(頁ごと。プールしない)。**参考の F1 は材料が 104/152 重なる** ==')
    hdr = (f'{"page":44s} {"標準":>5} {"原案":>5} {"合った":>6} {"rec":>7} '
           f'{"prec":>7} {"F1":>7} {"tone込":>6} {"∩base":>6} '
           f'{"N1":>6} {"N2":>6} {"N3":>6}')
    print(hdr)
    for r in per_page:
        print(f'{r["page"]:44s} {r["std_areas"]:>5} {r["A_mask_areas"]:>5} '
              f'{r["A_mask_matched"]:>6} {r["A_mask_rec"]:>7.4f} '
              f'{r["A_mask_prec"]:>7.4f} {r["A_mask_f1"]:>7.4f} '
              f'{r["A_mask_tone_areas"]:>6} {r["A_mask_cap_base_areas"]:>6} '
              f'{r["N1_f1"]:>6.4f} {r["N2_f1"]:>6.4f} {r["N3_f1"]:>6.4f}')
    print('**precision は下界、原案の面の数も下界**(標準も原案もマスクが面を束ねている)')

    with (OUT / 'mask_only_division.csv').open('w', newline='',
                                               encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_page[0].keys()))
        w.writeheader()
        w.writerows(per_page)
    print(f'\nwrote {OUT}/mask_only_division.csv')


if __name__ == '__main__':
    main()
