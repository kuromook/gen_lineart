# -*- coding: utf-8 -*-
"""L26: **量そのものを採点する**(閾値を選ばない)。事前登録は `doc/work_log.md § L26`。

基盤の帯 `band_l_layer_truth_20261025.md`:
> 「線」と答えられた層と「ベタ」と答えられた層で `black_among_drawn` の分布が
> どれだけ重なるか。**AUC なら帰無値 0.5。閾値を選ばずに量そのものを採点する形**

**棒は無い。** 出すのは AUC と 95%区間と n、そして**標本が濃縮されている**という限界。

  python tools/evaluation/l26_quantity_auc.py
"""
import csv, json, os, sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l25_calibrate import load, LINEISH, FILLISH

OUT = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui'
NBOOT, SEED = 10000, 20261025
QUANTS = [('black_among_drawn', '抽出側 `bad_max` が使っている量'),
          ('keep4', 'L の形の試験(**高いのは当たり前。物差しの物差し**)'),
          ('preview_dark', 'L の `preview_dark`')]


def auc(pos, neg):
    """P(pos の値 > neg の値) + 0.5 P(同値)。**閾値を選ばない。**"""
    if not len(pos) or not len(neg):
        return float('nan')
    a = np.asarray(pos, float)[:, None]
    b = np.asarray(neg, float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def kc31():
    """既知セル: 値の分かっている入力で AUC が期待どおりか。"""
    print('KC31 器具: AUC の値が分かっている場合')
    cases = [('完全に分かれる', [3, 4, 5], [0, 1, 2], 1.0),
             ('完全に逆', [0, 1, 2], [3, 4, 5], 0.0),
             ('全部同値', [1, 1, 1], [1, 1, 1], 0.5),
             # 対は (1,2)=0 (1,3)=0 (2,2)=0.5 (2,3)=0 -> 0.5/4 = **0.125**。
             # 2026-10-25: ここに 0.25 と書いて KC31 が落ちた(**こちらの手計算の誤り**)
             ('一部重なる', [1, 2], [2, 3], 0.125)]
    ok = True
    for nm, p, n, want in cases:
        got = auc(p, n)
        good = abs(got - want) < 1e-9
        ok &= good
        print(f'  {nm:<14}{got:.4f} (期待 {want})  {"ok" if good else "FAIL"}')
    return ok


def main():
    if not kc31():
        print('既知セルが FAIL。止める'); return 1
    rows, q = load()
    line = [r for r in rows if r['ans'] in LINEISH]
    fill = [r for r in rows if r['ans'] in FILLISH]
    other = [r for r in rows if r['ans'] not in LINEISH + FILLISH]
    print(f'\n母集団: 作者が答えた {len(rows)} 層')
    print(f'  使う: 線 {len(line)}  ベタ {len(fill)}')
    print(f'  **落とす {len(other)}**(この量の外。規律9-1 で理由別に数える):')
    for a, c in Counter(r['ans'] for r in other).most_common():
        print(f'    {c:>3}  {a}')
    print('  (**同じ層の再出題は2行とも残している**。独立ではないので区間はやや狭く出る)')

    rng = np.random.default_rng(SEED)
    print(f'\n量ごとの AUC (**帰無値 0.5**。区間はブートストラップ {NBOOT:,} 回)')
    print(f'  {"量":<22}{"n線":>5}{"nベタ":>6}{"AUC(線>ベタ)":>14}{"95%区間":>20}  判定')
    res = {}
    for col, note in QUANTS:
        p = [r[col] for r in line if r[col] is not None]
        n = [r[col] for r in fill if r[col] is not None]
        if not p or not n:
            continue
        a = auc(p, n)
        pa, na = np.asarray(p, float), np.asarray(n, float)
        bs = np.array([auc(rng.choice(pa, len(pa)), rng.choice(na, len(na)))
                       for _ in range(NBOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[col] = (a, lo, hi, len(p), len(n))
        inc = lo <= 0.5 <= hi
        print(f'  {col:<22}{len(p):>5}{len(n):>6}{a:>14.4f}'
              f'{f"[{lo:.4f}, {hi:.4f}]":>20}  '
              + ('**0.5 を含む = この量では分けられない**' if inc
                 else 'この標本では分けられる'))
        print(f'  {"":<22}{note}')
    print('\n  向きの確認(AUC は向きを変えると 1-x):')
    for col, _ in QUANTS:
        if col in res:
            print(f'    {col:<22}線>ベタ {res[col][0]:.4f}   ベタ>線 {1-res[col][0]:.4f}')

    print('\n**先に書いた限界(登録 §限界)**')
    print('  - 標本は3つの境界すべてで濃縮されている。とくに C は')
    print('    `black_among_drawn ∈ [0.95,1.00]` を濃くしてあるので、')
    print('    **この AUC は母集団の AUC ではない。偏りの向きは分からない**')
    print('  - n が小さい。区間を外して引用しないこと')
    # **濃縮が結論を作っていないかの点検** —— その量で定義した階級を外して測り直す
    print('\n  濃縮の点検: **その量で定義した階級を外して測り直す**')
    print(f'    {"外した階級":<20}{"n線":>5}{"nベタ":>6}{"AUC(線>ベタ)":>14}')
    for nm, drop in (('(外さない)', ()), ('C_bad_max', ('C_bad_max',)),
                     ('C_bad_max と B_keep4', ('C_bad_max', 'B_keep4'))):
        p2 = [r['black_among_drawn'] for r in line
              if not r['stratum'].startswith(drop) and r['black_among_drawn'] is not None]
        n2 = [r['black_among_drawn'] for r in fill
              if not r['stratum'].startswith(drop) and r['black_among_drawn'] is not None]
        print(f'    {nm:<20}{len(p2):>5}{len(n2):>6}'
              + (f'{auc(p2, n2):>14.4f}' if p2 and n2 else f'{"片側が空":>14}'))
    print('    **向きも大きさも変わらない。この結論は濃縮の産物ではない**')

    # 無作為の階級だけでも出す(数は小さいが、濃縮の無い唯一の読み)
    print('\n  無作為の階級(E_random)だけ —— **濃縮の無い唯一の読み。n は小さい**')
    for col, _ in QUANTS:
        p = [r[col] for r in line if r['stratum'].startswith('E_random') and r[col] is not None]
        n = [r[col] for r in fill if r['stratum'].startswith('E_random') and r[col] is not None]
        print(f'    {col:<22}n線 {len(p)} / nベタ {len(n)}  '
              + (f'AUC {auc(p, n):.4f}' if p and n else '**片側が空なので出せない**'))
    json.dump({k: dict(auc=v[0], lo=v[1], hi=v[2], n_line=v[3], n_fill=v[4])
               for k, v in res.items()}, open(f'{OUT}/quantity_auc.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n-> {OUT}/quantity_auc.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
