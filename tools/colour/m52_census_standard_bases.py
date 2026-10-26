# -*- coding: utf-8 -*-
"""m52: 823成分を**ベースの種類で数える**(追補38 §G に事前登録済み)。

目的は「面でない成分が 823 のうちどれだけか」の**大きさ**を知ること。
**どの成分が面かは決めない**(決めるのは作者。規律6)。

登録した条件:
  - 結合は `source` + `layer_id`。**ファイル名では結合しない**(m48 の教訓)
  - 分類はすべて並べる: `folder` / `layer_type` / 名前のバケツ。**掛けた表も出す**
  - 既知セル: bases 183 / components 823 / areas 801。合わなければ他を読まない
  - **プールを跨いで平均しない**: 頁ごとの行も出す
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402
import m27_author_crops as m27                           # noqa: E402

DELIV = m27.DELIV
OUT = Path('results/m52_census'); OUT.mkdir(parents=True, exist_ok=True)
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}
LINE = re.compile(r'(線画|線|line)', re.I)


def bucket(name):
    """名前のバケツ。**篩ではない。仮説を出すための記述**(追補38 §F)。"""
    n = (name or '?').strip()
    if n == 'MASK':
        return 'MASK'
    if n.lower().startswith('sketch'):
        return 'Sketch'
    if LINE.search(n):
        return '線/line を含む'
    return 'その他'


def main():
    rows = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
             if l.strip()]
    m12.D = DELIV
    tot = collections.Counter()
    # 結合は source + layer_id
    per = []          # 成分1件 = 1行
    for p in pages:
        if not p['layers']:
            tot['ベース層が無い頁'] += 1
            continue
        prows = byp[p['source']]
        byid = {str(r['layer_id']): r for r in prows}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (_, garea_a), st = m16.measure_standard(
            p, prows, grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        for c in comps:
            r = byid.get(str(c['layer_id']))
            if r is None:
                tot['**台帳に無いベース**'] += 1
            per.append(dict(page=p['page'], source=p['source'],
                            layer_id=c['layer_id'], area=c['area'],
                            folder=bool(c['folder']),
                            layer_type=(r or {}).get('layer_type', '?'),
                            name=((r or {}).get('name') or '?'),
                            bucket=bucket((r or {}).get('name'))))
        del comps
        print(f'  {p["page"]:44s} 成分{st["components"]:>4} 面{len(garea_a):>4}',
              flush=True)

    print('\n== 既知セル ==')
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if any(tot[k] != EXPECT[k] for k in EXPECT):
        print('**既知セルが合わない。他を読まない。**')
        return 1
    for k in ('ベース層が無い頁', '**台帳に無いベース**'):
        if tot[k]:
            print(f'  {k}: {tot[k]}')

    n = len(per)
    print(f'\n== 成分 {n} 件の分類(**どれかを選ばない。全部並べる**)==')
    for axis in ('folder', 'layer_type', 'bucket'):
        print(f'\n  -- {axis} --')
        c = collections.Counter(str(r[axis]) for r in per)
        for k, v in sorted(c.items(), key=lambda t: -t[1]):
            print(f'    {k:18s} {v:>5} 成分  {v/n:>6.1%}')

    print('\n  -- 掛けた表(bucket x folder x layer_type)--')
    c = collections.Counter(
        (r['bucket'], r['folder'], str(r['layer_type'])) for r in per)
    for (b, f, lt), v in sorted(c.items(), key=lambda t: -t[1]):
        print(f'    {b:14s} folder {str(f):5s} 種別 {lt:2s} {v:>5} 成分 {v/n:>6.1%}')

    print('\n== 頁ごと(**プールを跨いで平均しない**)==')
    pg = collections.defaultdict(collections.Counter)
    for r in per:
        pg[r['page']][r['bucket']] += 1
    ks = ['MASK', 'Sketch', '線/line を含む', 'その他']
    print('  %-44s %s' % ('頁', ' '.join(f'{k:>14s}' for k in ks)))
    for p, c in sorted(pg.items()):
        print('  %-44s %s' % (p, ' '.join(f'{c[k]:>14d}' for k in ks)))

    print('\n== 作者が「面ではない」と言った対象の大きさ(追補38 §E)==')
    nonmask = sum(1 for r in per if r['bucket'] != 'MASK')
    fold = sum(1 for r in per if r['folder'])
    lt0 = sum(1 for r in per if str(r['layer_type']) == '0')
    print(f'  非 MASK 由来        {nonmask:>5} / {n} = **{nonmask/n:.1%}**')
    print(f'  folder のベース由来   {fold:>5} / {n} = **{fold/n:.1%}**')
    print(f'  layer_type 0 由来   {lt0:>5} / {n} = **{lt0/n:.1%}**')
    print('  **3つの軸が同じ集合を指すか**: '
          + ('指す' if nonmask == fold == lt0 else '**指さない(分かれる)**'))

    with open(OUT / 'components_by_base.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader(); w.writerows(per)
    print(f'\nwrote {OUT / "components_by_base.csv"}  ({n} 成分。'
          '**この8頁での数。corpus の数ではない**)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
