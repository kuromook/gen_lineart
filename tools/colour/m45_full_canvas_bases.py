"""M45: **標準の入口に入っている「全面を覆う層」が、標準の面をどれだけ水増ししているか**。

**向きを先に書く(基盤の条件4)**:

> **全面の成分が標準の面として残っているなら、標準の面数は過大、
> A0 の recall は過小に出ていた**はずである。**結果を見てから向きを決めない。**

**条件(基盤。4つとも守る)**:

1. **両方の数を同じ行に印字。** 面数は **801** と **全面の成分を除いた数**を並べる。
   **801 を置き換えない**(撤回ではなく、2つの量)
2. **原因を3つに分けて数える**: **(a) クリップ群のベース**(紙面に出るのはクリップ後の一部。
   **T-M3: アルファは被覆ではない**)/ **(b) 本当に全面に塗られている** /
   **(c) 色調補正層などの別種の混入**。**直し方が3つとも違う**
3. **層ごとに「面を何枚生んだか」と「その面積」の両方**
4. **母集団を数の隣に**: **既知セル8頁は `illust*` が半分**(濃縮3.52倍、しかも上限)。
   **「この8頁での数」であって corpus の数ではない**

**(a)(b)(c) の分け方**:

- **(a)**: そのベースの**上に `clip=1` の兄弟が居る**(構造で判定。名前を使わない)
- **(c)**: 名前に `色相`・`彩度`・`明度`・`トーンカーブ`・`レベル`・`グラデーションマップ` を含む
  **候補**(名前は錨として落ちているので**候補としてのみ**印字。作者に確認する列へ)
- **(b)**: (a) でも (c) でもないもの

**IC-M6 の限界も記録する**: **走行の中の同一性しか見ていない点検は、
両側が共有している汚染を見ない**(`moat` 規則2 / T97 の族)。
**既知セルが 1.0 でも標準が汚れていることはある。**
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402

DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4_known8')
OUT = Path('results/m45_full_canvas'); OUT.mkdir(parents=True, exist_ok=True)
FULL = 0.999
ADJ = re.compile(r'(色相|彩度|明度|トーンカーブ|レベル|グラデーションマップ|明るさ|コントラスト)')
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}


def main():
    rows = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8') if l.strip()]
    m12.D = DELIV
    tot = collections.Counter()
    rec = []
    for p in pages:
        if not p['layers']:
            continue
        prows = byp[p['source']]
        byid = {r['layer_id']: r for r in prows}
        sib = collections.defaultdict(list)
        for r in prows:
            if r['parent_id']:
                sib[r['parent_id']].append(r)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (gt_a, garea_a), st = m16.measure_standard(
            p, prows, grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        # 面(painted)ごとに、元の成分の layer_id を引けるようにする
        lab_of = {}
        for k, c in enumerate(sorted(comps, key=lambda c: -c['area']), start=1):
            lab_of[k] = c
        for lab, area in garea_a.items():
            c = lab_of.get(lab)
            if c is None:
                continue
            r = byid.get(str(c['layer_id'])) or byid.get(c['layer_id'])
            dr = m19.num(r['drawn_ratio'], 0) if r else 0
            isfull = (dr or 0) >= FULL
            kind = ''
            if isfull:
                kids = sib.get(str(c['layer_id']), [])
                has_clip = any(x['clip'] == '1' for x in sib.get(
                    r['parent_id'], []) if x is not r) if r else False
                if has_clip:
                    kind = '(a) クリップ群のベース'
                elif r and ADJ.search(r['name'] or ''):
                    kind = '(c) 色調補正層の候補'
                else:
                    kind = '(b) 本当に全面'
                tot['full_areas'] += 1
                tot['full_px'] += int(area)
                tot[kind] += 1
            rec.append({'page': p['page'], 'layer_id': c['layer_id'],
                        'name': (r or {}).get('name', '?'), 'drawn_ratio': dr,
                        'area_px': int(area), 'full': isfull, 'kind': kind,
                        'canvas_px': grid[0] * grid[1]})
        print(f'  {p["page"]:44s} ベース{st["bases"]:>4} 成分{st["components"]:>4} '
              f'面{len(garea_a):>4}', flush=True)
        del comps, gt_a

    ok = all(tot[k] == EXPECT[k] for k in EXPECT)
    print(f'\n== 既知セル ==')
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if not ok:
        print('**既知セルが合わない。表を書かない。**')
        return 1

    byl = collections.Counter()
    for x in rec:
        if x['full']:
            byl[(x['page'], x['layer_id'], x['name'], x['kind'])] += 1
    print(f'\n== 全面を覆うベースが生んだ面(**この8頁での数**。'
          f'既知セルは `illust*` が半分の層で、corpus の数ではない)==')
    print(f'  **標準の面 801 / そのうち全面のベースから出た面 {tot["full_areas"]}**'
          f' → **全面を除くと {tot["areas"] - tot["full_areas"]}**'
          f'(**801 を置き換えない。2つの量**)')
    print(f'  全面のベースから出た面の画素の合計 {tot["full_px"]:,}')
    print(f'  原因の内訳: ' + ' / '.join(
        f'{k} {tot[k]}' for k in ('(a) クリップ群のベース', '(b) 本当に全面',
                                  '(c) 色調補正層の候補') if tot[k]))
    print(f'\n  層ごと(面を何枚生んだか・面積):')
    for (pg, lid, nm, kind), n in sorted(byl.items(), key=lambda kv: -kv[1])[:20]:
        px = sum(x['area_px'] for x in rec
                 if x['page'] == pg and x['layer_id'] == lid)
        cv = next(x['canvas_px'] for x in rec if x['page'] == pg)
        print(f'    {pg[:36]:36s} L{lid:<4} {nm!r:16s} 面 {n:>3} '
              f'画素 {px:>10,}({px / cv:.3f} 覆い)  {kind}')
    with open(OUT / 'full_canvas_areas.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec[0].keys()))
        w.writeheader()
        w.writerows(rec)
    print(f'\nwrote {OUT}/full_canvas_areas.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
