# -*- coding: utf-8 -*-
"""L38 **分離は効いたか / 問題B の率**を読む(2026-11-08)。**答えが1行も来る前に書いた器具**。

登録: `doc/l38_problem_b_registration_20261108.md`。**読み方は登録のとおりで、ここでは選ばない。**
順に: 器具の点検(IC) -> 答えられなかった率(C4) -> 主値(片の率)-> 対照(元のまま)-> 棒。

**一言の欄は面積を付けずに印字する** —— 2026-11-07 に面積から群が読めて主値が漏れた。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_problem_b_join_20261108.py
"""
import collections
import json
import math
import sys
from pathlib import Path

from scipy.stats import fisher_exact

D = Path('/home/sh1/deepl/lineart-fill-region/data/l38_problemb_20261108')
ANS = D / 'l38_problem_b_20261108_answers.jsonl'
IC1_AREA = 2503898          # 「1つのもの」(11-07 の午前と夜で一致した)
IC2_PARENT = 2232129        # 親は「髪と衣類」。**分離したので片は片方だけ** = 1つのもの


def wilson(k, n, z=1.96):
    if not n:
        return float('nan'), float('nan')
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(rows, nm):
    ans = [r for r in rows if r.get('value') in ('one', 'many')]
    m = sum(1 for r in ans if r['value'] == 'many')
    if not ans:
        print(f'  {nm:22} 答えがまだ無い')
        return None
    lo, hi = wilson(m, len(ans))
    print(f'  {nm:22} {m}/{len(ans)} = **{m / len(ans):.1%}**'
          f'(Wilson {lo:.1%}〜{hi:.1%})')
    return m, len(ans), lo, hi


def main():
    kmap = json.load(open(D / 'map.json', encoding='utf-8'))
    if not ANS.exists():
        sys.exit(f'答えがまだ無い: {ANS}')
    last = {}
    for ln in open(ANS, encoding='utf-8'):
        if ln.strip():
            r = json.loads(ln)
            last[r['item_id']] = r              # **最後の行が答え**(直しは新しい行)
    for fid, r in last.items():
        if fid in kmap:
            kmap[fid]['value'] = r['value']
            kmap[fid]['comment'] = r.get('comment', '')
    print(f'答えの在る単位 {len(last)}/{len(kmap)}(**追記だけ。前の行は消えていない**)')

    print('\n--- 器具の点検(IC)---')
    ic = []
    for nm, sel, want in (
            ('IC1 面積 2,503,898(元のまま)', lambda v: v['area'] == IC1_AREA, 'one'),
            ('IC2 「髪と衣類」の最大の片', lambda v: v['origin'] == 'piece' and v['parent']
             and v['parent'][4] == IC2_PARENT and v['area'] == 1317982, 'one')):
        hit = [v for v in kmap.values() if sel(v)]
        if not hit:
            print(f'  {nm}: **標本に無い**')
            continue
        got = hit[0].get('value')
        if got is None:
            print(f'  {nm}: まだ答えが無い')
            continue
        ok = got == want
        ic.append((nm, ok, want, got))
        print(f'  {nm}: 期待 {want} 実際 {got}  {"一致" if ok else "**外れた**"}')
    ic1 = [t for t in ic if t[0].startswith('IC1')]
    if ic1 and not ic1[0][1]:
        print('\n**C5: IC1 が外れた。登録どおり何も読まない。**')
        return
    ic2 = [t for t in ic if t[0].startswith('IC2')]
    if ic2 and not ic2[0][1]:
        print('\n**IC2 が外れた —— これは分離そのものの点検である。**')
        print('**主値より先にこれを報告する**: 片が「2つ以上」なら、分離の規則が間違っている。')

    ansd = [v for v in kmap.values() if v.get('value') in ('one', 'many')]
    held = [v for v in kmap.values() if v.get('value') in ('__unknown__', '__none__')]
    n = len(ansd) + len(held)
    r4 = len(held) / n if n else 0
    print(f'\n--- 答えられなかった率(C4)---')
    print(f'  {len(held)}/{n} = {r4:.1%}(11-07 の実測: 面だけ 17.6% / 分離前の面 3.2%)')
    if r4 >= 0.35:
        print('  **35% 以上 —— 他の棒を読まない**(問いが測っていない)')
        return
    if r4 >= 0.25:
        print('  **25〜35% —— 棒は読むが、問いの作りの欠陥として記録する**')

    print('\n--- 主値と対照 ---')
    P = [v for v in kmap.values() if v['group'].startswith('piece')
         or v['group'] == 'known_piece']
    K = [v for v in kmap.values() if v['group'] in ('kept', 'known_one')]
    p1 = rate(P, '層1 割った片')
    p2 = rate(K, '層2 元のまま(対照)')
    for w in sorted({v['work'] for v in kmap.values()}):   # **作品は畳まない**
        rate([v for v in P if v['work'] == w], f'  層1 / {w}')
        rate([v for v in K if v['work'] == w], f'  層2 / {w}')
    par = collections.Counter(tuple(v['parent']) for v in P
                              if v.get('value') in ('one', 'many') and v['parent'])
    dup = {k: v for k, v in par.items() if v >= 2}
    print(f'  **同じ親から2片以上**({len(dup)} 親)—— 標本は独立でない。畳まずに書く:')
    for k, v in sorted(dup.items(), key=lambda t: -t[1]):
        vals = [x.get('value') for x in P if x['parent'] == list(k)]
        print(f'    親の面積 {k[4]:>9,} から {v} 片: {vals}')

    if p1 and p2:
        m1, n1, lo1, hi1 = p1
        m2, n2, _, _ = p2
        pv = fisher_exact([[m1, n1 - m1], [m2, n2 - m2]])[1]
        d = m1 / n1 - m2 / n2
        print(f'\n  層1 と層2 の差 **{d:+.1%}**(Fisher 両側 p = {pv:.4f})')
        print('\n--- 棒の判定(登録のとおり)---')
        if m1 / n1 <= 0.20 and hi1 < 0.40:
            print(f'  **C1 達成**(片の率 {m1 / n1:.1%} <= 20%、Wilson 上限 {hi1:.1%} < 40%)'
                  f' —— **分離は問題Aを取り除いた。** 次は問題B に進む')
        elif m1 / n1 >= 0.50:
            print(f'  **C2**(片の率 {m1 / n1:.1%} >= 50%)—— **分離は効いていない。**'
                  f' 片の中にまだ2つ以上が居る。**原因を先に切り分ける**'
                  f'(1枚の層の中の結合 / 半径4 の閉じ / 8連結。11-06 の軸1〜3 へ戻る)')
        else:
            print(f'  **判定保留**(片の率 {m1 / n1:.1%} は 20〜50% の帯。'
                  f'Wilson 上限 {hi1:.1%})')
        if d >= 0.20:
            print(f'  **C3 も立つ**(層1 が層2 を {d:+.1%} 上回る)'
                  f' —— 分離の後にも層1 固有の欠陥が残っている')
        print(f'\n  **問題B の率(記述。棒は置いていない)**: 元のままの面で {m2}/{n2} '
              f'= {m2 / n2:.1%} が「2つ以上」')

    cm = [v for v in kmap.values() if v.get('comment')]
    print(f'\n--- 一言の欄 {len(cm)} 件(**面積は出さない**。群が読めてしまうので)---')
    for v in sorted(cm, key=lambda v: v['comment']):
        print(f'    {v.get("value")}: {v["comment"]}')


if __name__ == '__main__':
    main()
