#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補111 §D の事前登録を走らせる —— `standard_visible` の上で腕を採点し直す。**

**作者が 2026-10-28 に標準を決めた**(原文「standard visibleを選択」。追補111 §A)。
`standard_all`(801)の上で出した点数は全部この定義の上の値なので、**測り直す**。

**事前登録(追補111 §D。走る前に書いて commit した)**:

  母集団   = **登録集合8頁のまま**(広げない。広げるのも棒を動かすこと)
  腕       = `standard_all` と `standard_visible` の**両方を同じ走行で出す**
  主値     = `standard_visible` の上での **A0 の region-F1(頁ごと)**
  副値     = recall / precision / 面の数
  帰無値   = **N1 blank / N2 whole_page / N3 grid**(追補7 のまま。**緩めない**)
  棒       = A0 が N1・N2・N3 の**すべて**に頁ごとの符号検定で勝つ(両側 α=0.05)。
             **n=8 では 8/8 のときだけ通る**(2*(1/2)^8 = 0.0078。7/8 は 0.070)
  門       = KC-1 `standard_all` が 183/823/801 / KC-2 `standard_visible` が 152/503/483 /
             KC-3 IC-M6 が 8/8 で 1.0 / KC-4 IC-M7 が 8/8 で 1.0。
             **落ちたら以降の行を読まない**
  読み     = 8/8 を取っても**登録集合なので主張は出ない**(追補3 §C8)。
             取らなければ**判定保留のまま**。「標準を変えたら腕が通った」は書かない。
             **precision/recall がどちらに動くかは予言しない**
  床       = 主値 **1e-5**(標準側と A0 側で同じ値)。**1e-4 は副値として別に走らせる**

**器具は写さずに import する**(m16 の標準、m12 の A0・帰無値・符号検定、m77 の非表示の規則)。

  <venv>/python tools/colour/m81_rescore_visible_standard.py              # 主値(床 1e-5)
  <venv>/python tools/colour/m81_rescore_visible_standard.py --floor 1e-4 # 副値
"""
import argparse
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

OUT = pathlib.Path('results/m81_rescore_visible')
OUT.mkdir(parents=True, exist_ok=True)

# **既知セルは世代ごとに1つ。片方に差し替えない**(規律2 の付帯。追補111 §E)
KC = {'standard_all': {'bases': 183, 'components': 823, 'areas': 801},
      'standard_visible': {'bases': 152, 'components': 503, 'areas': 483}}
ARMS = ('standard_all', 'standard_visible')


def score_page(p, rows, floor_frac):
    """1頁を採点する。**m16.main の中身と同じ順序**(写さず import した部品で組む)。"""
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    floor_px = max(1, round(floor_frac * grid[0] * grid[1]))
    im = cv2.imread(str(m12.D / p['preview']['file']), cv2.IMREAD_COLOR)

    # 標準(主値は「面積の小さい成分が勝つ」= 追補7 R3。z 順は印字のためだけ)
    (gt_z, garea_z), (gt, garea), st = m16.measure_standard(p, rows, grid, floor_px)
    del gt_z

    # 器具確認。IC-M6 = 標準を予測として入れる / IC-M7 = ブロック拡大の経路
    ic_m6, _ = m12.score_from_small(gt, 1, gt, garea, floor_px)
    a0_lab = m12.equal_value_components(m12.quantise(im, m12.LEVELS))
    a0, _ = m12.filtered_labels(a0_lab, max(1, round(floor_frac * pw * ph)))
    ic_m7 = m12.ic_m7(a0, s, grid, floor_px)

    n1 = np.zeros((ph, pw), np.int32)
    n2 = np.ones((ph, pw), np.int32)
    n3, gr, gc = m12.grid_null((ph, pw), len(garea))
    r = {}
    for k, pred in (('a0', a0), ('n1', n1), ('n2', n2), ('n3', n3)):
        r[k], _ = m12.score_from_small(pred, s, gt, garea, floor_px)

    out = {'page': p['page'], 'scale': s, 'bases': st['bases'],
           'components': st['components'], 'areas': len(garea),
           'areas_zorder': len(garea_z),
           'ic_m6_f1': round(ic_m6['region_f1'], 6),
           'ic_m7_f1': round(ic_m7['region_f1'], 6),
           'a0_n': r['a0']['n_pred'], 'a0_matched': r['a0']['matched'],
           'a0_prec': round(r['a0']['precision'], 4),
           'a0_rec': round(r['a0']['recall'], 4),
           'a0_f1': round(r['a0']['region_f1'], 4),
           'n1_f1': round(r['n1']['region_f1'], 4),
           'n2_f1': round(r['n2']['region_f1'], 4),
           'n3_f1': round(r['n3']['region_f1'], 4),
           'n3_shape': f'{gr}x{gc}'}
    del gt, a0, a0_lab, im, n1, n2, n3
    return out


def run_arm(arm, pages, byp, floor_frac):
    rows_out, t0 = [], time.time()
    for p in pages:
        rows = byp[p['source']]
        if arm == 'standard_visible':
            rows, _ = m77.visible_rows(rows)
        o = score_page(p, rows, floor_frac)
        rows_out.append(o)
        print(f'  {o["page"]:44s} base{o["bases"]:>4} comp{o["components"]:>4} '
              f'面{o["areas"]:>4} IC6 {o["ic_m6_f1"]:.4f} IC7 {o["ic_m7_f1"]:.4f} '
              f'rec {o["a0_rec"]:.4f} prec {o["a0_prec"]:.4f} '
              f'A0 {o["a0_f1"]:.4f} N1 {o["n1_f1"]:.4f} N2 {o["n2_f1"]:.4f} '
              f'N3 {o["n3_f1"]:.4f} {time.time()-t0:6.1f}s', flush=True)
    return rows_out


def sign_tests(rows):
    """A0 対 各帰無値、頁ごとの符号検定(引き分けは落とす = m12 の定義)。"""
    out = {}
    for n in ('n1', 'n2', 'n3'):
        w = sum(1 for r in rows if r['a0_f1'] > r[f'{n}_f1'])
        l = sum(1 for r in rows if r['a0_f1'] < r[f'{n}_f1'])
        out[n] = (w, l, m12.judge_sign_test(w, l))
    return out


def main(argv=()):
    ap = argparse.ArgumentParser()
    ap.add_argument('--floor', type=float, default=m12.FLOOR)
    a = ap.parse_args(argv)
    primary = abs(a.floor - m12.FLOOR) < 1e-12

    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    print(f'頁 {len(pages)}  床 {a.floor:g}'
          f'{"(**主値**)" if primary else "(**副値。主値は 1e-5 から動かさない**)"}')

    # **落とした層を理由別に印字する**(規律9)。**印字の 64/185 は順に見た理由**
    drops = collections.Counter()
    for p in pages:
        _, d = m77.visible_rows(byp[p['source']])
        drops.update(d)
    print('\n== `standard_visible` が落とす層(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:16s} {v:>5}')
    print(f'  {"合計":16s} {sum(drops.values()):>5}')

    res, tot = {}, {}
    for arm in ARMS:
        print(f'\n== {arm} ==')
        res[arm] = run_arm(arm, pages, byp, a.floor)
        tot[arm] = {k: sum(r[k] for r in res[arm])
                    for k in ('bases', 'components', 'areas')}
        print(f'  合計 ベース {tot[arm]["bases"]} / 成分 {tot[arm]["components"]} '
              f'/ **面 {tot[arm]["areas"]}**')

    print('\n== 門(既知セル。追補111 §E。**両方を持つ**)==')
    ok = True
    for arm in ARMS:
        for k, want in KC[arm].items():
            got = tot[arm][k]
            good = (got == want) if primary else True
            print(f'  {arm:17s} {k:11s} 期待 {want:>4} / 実測 {got:>4}  '
                  f'{"PASS" if got == want else ("**FAIL**" if primary else "(副値なので門にしない)")}')
            ok &= good
    for arm in ARMS:
        for nm, key in (('KC-3 IC-M6', 'ic_m6_f1'), ('KC-4 IC-M7', 'ic_m7_f1')):
            n = sum(1 for r in res[arm] if abs(r[key] - 1.0) < 1e-9)
            print(f'  {arm:17s} {nm} 1.0 の頁 {n}/{len(pages)}  '
                  f'{"PASS" if n == len(pages) else "**FAIL**"}')
            ok &= n == len(pages)
    # **副値の走行では面の数を門にしていない**ので「すべて PASS」と書かない
    # (床を変えると標準の面の数が変わるのは当たり前で、門にすると床で門が動く)
    print(f'\n  **門 {"PASS" if ok else "**FAIL**。以降の行を読まない"}'
          f'{"(主値。面の数も門)" if primary else "(**副値。面の数は門にしていない。床 1e-5 の走行で通っている**)"}**')
    if not ok:
        return 1

    print('\n== 主値: `standard_visible` の上での A0(頁ごと。**プールで平均しない**)==')
    print(f'  {"頁":44s} {"面":>5} {"rec":>7} {"prec":>7} {"A0 F1":>7} '
          f'{"N1":>7} {"N2":>7} {"N3":>7}')
    for r in res['standard_visible']:
        print(f'  {r["page"]:44s} {r["areas"]:>5} {r["a0_rec"]:>7.4f} '
              f'{r["a0_prec"]:>7.4f} {r["a0_f1"]:>7.4f} {r["n1_f1"]:>7.4f} '
              f'{r["n2_f1"]:>7.4f} {r["n3_f1"]:>7.4f}')

    print('\n== 棒: 符号検定(両側 α=0.05。**n=8 は 8/8 のときだけ通る**)==')
    verdict = {}
    for arm in ARMS:
        st = sign_tests(res[arm])
        verdict[arm] = st
        line = ' / '.join(f'{n.upper()} {w}勝{l}敗 p={p:.4f}'
                          f'{"**通る**" if p < 0.05 else "通らない"}'
                          for n, (w, l, p) in st.items())
        print(f'  {arm:17s} {line}')
    pv = verdict['standard_visible']
    passed = all(p < 0.05 for _, _, p in pv.values())
    print(f'\n  **`standard_visible` の棒: '
          f'{"満たした" if passed else "満たしていない"}**  '
          f'最良 p = {min(p for _, _, p in pv.values()):.4f}')
    print('  **登録集合8頁なので、通っても主張は出ない**(追補3 §C8)。'
          '**通らなければ判定保留のまま**')

    print('\n== 副値: 2つの定義の差(**どちらに動くかは予言していない**)==')
    pa = {r['page']: r for r in res['standard_all']}
    print(f'  {"頁":44s} {"面 all→vis":>14} {"rec":>16} {"prec":>16}')
    for r in res['standard_visible']:
        o = pa[r['page']]
        print(f'  {r["page"]:44s} {o["areas"]:>6}→{r["areas"]:<7} '
              f'{o["a0_rec"]:>7.4f}→{r["a0_rec"]:<8.4f} '
              f'{o["a0_prec"]:>7.4f}→{r["a0_prec"]:<8.4f}')
    for k in ('a0_rec', 'a0_prec', 'a0_f1'):
        lo_a = min(r[k] for r in res['standard_all'])
        hi_a = max(r[k] for r in res['standard_all'])
        lo_v = min(r[k] for r in res['standard_visible'])
        hi_v = max(r[k] for r in res['standard_visible'])
        print(f'  {k:8s} standard_all {lo_a:.4f}〜{hi_a:.4f}  ->  '
              f'standard_visible **{lo_v:.4f}〜{hi_v:.4f}**')

    tag = f'{a.floor:g}'.replace('-', 'm').replace('.', 'p')
    for arm in ARMS:
        f = OUT / f'scores_{arm}_floor{tag}.csv'
        with open(f, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(res[arm][0].keys()))
            w.writeheader(); w.writerows(res[arm])
        print(f'wrote {f}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
