#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補105 —— 追補99 §D の事前登録を走らせる: 非表示を落とした標準。**

**事前登録(追補99 §D。結果を見る前に書いた)**:
  主値 = **面の数**(いまの 801)。副値 = ベース(183)・成分(823)
  帰無値 = **動かない**(非表示の層のアルファが、表示の層に完全に覆われている場合)
  動いた場合 = **撤回ではなく名前を2つにする**(`standard_all` / `standard_visible`)
  **どちらを標準と呼ぶかは決めない**(規律6: 作者のもの)

**非表示の読み方**: `visibility` の **1の位**(基盤の T142。**2の位はマスクの有効で別の量**)。
**自身が非表示の層と、祖先のフォルダーが非表示の層の両方を落とす**(基盤の T143)。

  <venv>/python tools/colour/m77_standard_visible.py
"""
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402

OUT = pathlib.Path('results/m77_standard_visible')
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}


def truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'none')


def visible_rows(rows):
    """**表示の層だけ**(自身の1の位が立ち、祖先のフォルダーも全部立つ)。"""
    idx = {r['layer_id']: r for r in rows}

    def vis(r):
        return bool(int(r['visibility'] or 0) & 1)

    keep, drop = [], collections.Counter()
    for r in rows:
        if not vis(r):
            drop['自身が非表示'] += 1
            continue
        pid, seen, bad = r['parent_id'], set(), None
        while pid and pid not in seen:
            seen.add(pid)
            p = idx.get(pid)
            if p is None:
                bad = '親が表に無い'
                break
            if not vis(p):
                bad = '祖先が非表示'
                break
            pid = p['parent_id']
        if bad:
            drop[bad] += 1
            continue
        keep.append(r)
    return keep, drop


def run(pages, byp, filt):
    tot, per = collections.Counter(), []
    for p in pages:
        rows = byp[p['source']]
        if filt:
            rows, _ = visible_rows(rows)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        (_, _), (gt_a, garea_a), st = m16.measure_standard(p, rows, grid, floor_px)
        areas = len(garea_a)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += areas
        per.append({'page': p['page'], 'bases': st['bases'],
                    'components': st['components'], 'areas': areas})
    return tot, per


def main():
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    print(f'頁 {len(pages)}  ラスタ {m12.D}')

    drops = collections.Counter()
    for p in pages:
        _, d = visible_rows(byp[p['source']])
        drops.update(d)
    print('\n== 落とした層(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:16s} {v:>5}')
    print(f'  {"合計":16s} {sum(drops.values()):>5}')

    print('\n== standard_all(いまの標準。`visibility` を見ない)==')
    a_tot, a_per = run(pages, byp, filt=False)
    for r in a_per:
        print(f'  {r["page"]:44s} ベース{r["bases"]:>4} 成分{r["components"]:>4} '
              f'面{r["areas"]:>4}')
    print(f'  合計 ベース {a_tot["bases"]} / 成分 {a_tot["components"]} '
          f'/ **面 {a_tot["areas"]}**')
    ok = all(a_tot[k] == v for k, v in EXPECT.items())
    print(f'  **既知セル(183/823/801)の再現: {"PASS" if ok else "**FAIL**"}**')
    if not ok:
        print('  **再現しない。結果を読まない**')
        return 1

    print('\n== standard_visible(**非表示を落とした**)==')
    v_tot, v_per = run(pages, byp, filt=True)
    for r, a in zip(v_per, a_per):
        d = tuple(r[k] - a[k] for k in ('bases', 'components', 'areas'))
        print(f'  {r["page"]:44s} ベース{r["bases"]:>4} 成分{r["components"]:>4} '
              f'面{r["areas"]:>4}   差 {d}')
    print(f'  合計 ベース {v_tot["bases"]} / 成分 {v_tot["components"]} '
          f'/ **面 {v_tot["areas"]}**')

    print('\n== 主値(面の数)==')
    print(f'  standard_all **{a_tot["areas"]}** / '
          f'standard_visible **{v_tot["areas"]}**  '
          f'差 **{v_tot["areas"] - a_tot["areas"]}**')
    print(f'  ベース {a_tot["bases"]} -> {v_tot["bases"]} / '
          f'成分 {a_tot["components"]} -> {v_tot["components"]}')
    print('  **帰無値は「動かない」**。'
          + ('**動かなかった。非表示の層のアルファは表示の層に覆われていた**'
             if v_tot['areas'] == a_tot['areas'] else
             '**動いた。いまの 801 は紙面に出ていない層を含んだ数である**'))
    print('  **どちらを標準と呼ぶかは決めない(作者のもの)。両方の名前で残す**')

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'standard_all_vs_visible.csv').open('w', newline='',
                                                    encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['page', 'bases_all', 'components_all', 'areas_all',
                    'bases_visible', 'components_visible', 'areas_visible'])
        for a, v in zip(a_per, v_per):
            w.writerow([a['page'], a['bases'], a['components'], a['areas'],
                        v['bases'], v['components'], v['areas']])
    print(f'\nwrote {OUT}/standard_all_vs_visible.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
