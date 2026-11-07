# -*- coding: utf-8 -*-
"""L38 **分離は効いたか / 問題B の率**の標本を選ぶ(2026-11-08)。

登録: `doc/l38_problem_b_registration_20261108.md`(**走る前に固定した**)。

- **層1 割った片** から **30** を**一様に**(種 20261108。素性で選ばない)
- **層2 元のままの面** から **20** を**一様に**
- **既知セル2つ**を名指しで足す(規律2)——
  IC1 面積 2,503,898(「1つのもの」。午前と夜で一致した)/
  IC2 面積 2,232,129 の**最大の片**(親は「髪と衣類」。分離したので片は片方だけ = 1つのもの)
- **11-07 に反転した面(1,293,677「髪とコマ枠線」)は IC に使わない** —— 動いた錨は使い回さない
- **抽出の前と後で面積の分位を印字する**(NC3。寄っていないことを見せる)
- **同じ親から2片以上引いたら、その数を印字する**(独立でない標本)

出力: `data/l38_problemb_20261108/{faces.json(札なし), why.json(札つき)}`
"""
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                      # noqa: E402

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
POOL = ROOT / 'results' / 'l38_faces_separated_20261108.json'
OUT = ROOT / 'data' / 'l38_problemb_20261108'
SEED = 20261108
N_PIECE, N_KEPT = 30, 20
IC1_AREA = 2503898
IC2_PARENT = 2232129


def q(rows):
    a = sorted(r['area'] for r in rows)
    if not a:
        return '—'
    return (f'最小 {a[0]:,} / 1/4 {a[len(a) // 4]:,} / 中央 {statistics.median(a):,.0f}'
            f' / 3/4 {a[3 * len(a) // 4]:,} / 最大 {a[-1]:,}')


def main():
    pool = json.load(open(POOL, encoding='utf-8'))
    piece = [r for r in pool if r['origin'] == 'piece']
    kept = [r for r in pool if r['origin'] == 'kept']
    print(f'分離後の面プール {len(pool):,}(割った片 {len(piece)} / 元のまま {len(kept)})'
          f' 床 {U.FRAGMENT_FLOOR:,}')
    print(f'  層1 割った片   抽出の前: {q(piece)}')
    print(f'  層2 元のまま   抽出の前: {q(kept)}')

    rnd = random.Random(SEED)
    s1 = rnd.sample(piece, min(N_PIECE, len(piece)))
    s2 = rnd.sample(kept, min(N_KEPT, len(kept)))
    print(f'\n一様に抽出(種 {SEED}。**素性で選んでいない**)')
    print(f'  層1 {len(s1)} 片   抽出の後: {q(s1)}')
    print(f'  層2 {len(s2)} 面   抽出の後: {q(s2)}')

    par = collections.Counter(tuple(r['parent']) for r in s1)
    dup = {k: v for k, v in par.items() if v >= 2}
    print(f'\n  **同じ親から2片以上引いた親 {len(dup)}**(標本は独立でない。'
          f'読むときに畳まない):')
    for k, v in sorted(dup.items(), key=lambda t: -t[1]):
        print(f'    親の面積 {k[4]:>9,} から {v} 片')

    rows = [dict(r, group='piece') for r in s1] + [dict(r, group='kept') for r in s2]
    have = {(r['origin'], r['area']) for r in rows}

    ic1 = [r for r in kept if r['area'] == IC1_AREA]
    if not ic1:
        ic1 = [r for r in pool if r['area'] == IC1_AREA]
        print(f'\n  **IC1 {IC1_AREA:,} は「元のまま」に無い**'
              f'(プール全体には {len(ic1)} 行)')
    if ic1 and ('kept', IC1_AREA) not in have:
        rows.append(dict(ic1[0], group='known_one'))
        print(f'  既知セル IC1 を足した 面積 {IC1_AREA:,}(答: 1つのもの)')
    kids = [r for r in piece if r['parent'] and r['parent'][4] == IC2_PARENT]
    if kids:
        big = max(kids, key=lambda r: r['area'])
        if ('piece', big['area']) not in have:
            rows.append(dict(big, group='known_piece'))
            print(f'  既知セル IC2 を足した 親 {IC2_PARENT:,} の最大の片 '
                  f'面積 {big["area"]:,}(親は「髪と衣類」。片は 1つのもの が期待)')
    else:
        print(f'  **IC2 の親 {IC2_PARENT:,} の片がプールに無い**(分離されていない?)')

    rnd.shuffle(rows)                                   # 画面の順は群と無関係に(目隠し)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['fid'] = f'f{i:02d}'
    json.dump(rows, open(OUT / 'why.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump([{k: v for k, v in r.items() if k != 'group'} for r in rows],
              open(OUT / 'faces.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    n = collections.Counter(r['group'] for r in rows)
    print(f'\n**問いの数 {len(rows)}** —— ' + ' / '.join(f'{k} {v}' for k, v in n.items()))
    print(f'書いた: {OUT}/faces.json(札なし) と why.json(札つき。画面に出さない)')


if __name__ == '__main__':
    main()
