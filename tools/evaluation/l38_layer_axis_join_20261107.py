# -*- coding: utf-8 -*-
"""L38 **層の軸**の答えを読む(2026-11-07)。**答えが1行も来る前に書いた器具である。**

登録: `doc/l38_layer_axis_registration_20261107.md`。**読み方は登録のとおりで、ここでは選ばない。**
順に: 器具の点検(IC) -> 答えられなかった率(C5) -> 主値(29対) -> 並べ替え -> 腕の表 -> 棒の判定。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_layer_axis_join_20261107.py
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import fisher_exact

D = Path('/home/sh1/deepl/lineart-fill-region/data/l38_layeraxis_20261107')
ANS = D / 'l38_layer_axis_20261107_answers.jsonl'
SEED = 20261107
KNOWN = {2232129: ('many', '髪と衣類'), 2503898: ('one', '背景のベタのみ'),
         1293677: ('many', '髪とコマ枠線')}


def wilson(k, n, z=1.96):
    if not n:
        return float('nan'), float('nan')
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def diagnostic(kmap):
    """**器具と問いについての数だけ**を出す(答えられなかった率と一言)。

    主値(群Tと群Cの率・Fisher・腕の表)は**一切出さない** —— IC が外れた走行で
    主値を見てから次の問いを作ると、点を見て棒を動かしたのと同じことになる(条例1)。
    """
    ansd = [v for v in kmap.values() if v.get('value') in ('one', 'many')]
    held = [v for v in kmap.values() if v.get('value') in ('__unknown__', '__none__')]
    n = len(ansd) + len(held)
    print(f'  答えられなかった率: {len(held)}/{n} = {len(held) / n:.1%}'
          f'(第1回: 一様30面 26.7% / 面だけ 17.6% / 小片 38.5%)')
    if held:
        print('    内訳: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(
            v['value'] for v in held).items()))
    cm = [v for v in kmap.values() if v.get('comment')]
    print(f'  一言の欄 {len(cm)} 件(**多ければ問いの作りの欠陥**。指針 §2-3):')
    for v in sorted(cm, key=lambda v: v.get('comment', '')):
        # **面積を出さない** —— 面積は群を言い当てる鍵である(登録の対の表に書いてある)。
        # 2026-11-07 にここから主値が漏れた。診断は主値を含意する鍵を持たない
        print(f'    {v.get("value")}: {v["comment"]}')


def main():
    kmap = json.load(open(D / 'map.json', encoding='utf-8'))
    if not ANS.exists():
        sys.exit(f'答えがまだ無い: {ANS}')
    last = {}
    for ln in open(ANS, encoding='utf-8'):
        if ln.strip():
            r = json.loads(ln)
            last[r['item_id']] = r                      # **最後の行が答え**(直しは新しい行)
    print(f'答えの在る面 {len(last)}/{len(kmap)}'
          f'(**追記だけ。選び直しは新しい行で、前の行は消えていない**)')
    for fid, r in last.items():
        kmap[fid]['value'] = r['value']
        kmap[fid]['comment'] = r.get('comment', '')

    print('\n--- 器具の点検(IC。**外から答が分かっているセル**)---')
    ic_ok, ic_n = 0, 0
    for a, (want, what) in KNOWN.items():
        hit = [v for v in kmap.values() if v['area'] == a]
        if not hit or 'value' not in hit[0]:
            print(f'  {a:,} —— まだ答えが無い({what})')
            continue
        ic_n += 1
        got = hit[0]['value']
        ok = got == want
        ic_ok += ok
        print(f'  {a:,} 期待 {want:<4} 実際 {got:<12} {"一致" if ok else "**外れた**"}({what})')
    if ic_n and ic_ok < ic_n:
        print(f'\n**IC が {ic_ok}/{ic_n}。登録どおり、この走行の主値を読まない。**')
        print('**群Tと群Cの率も、Fisher も、腕の表も印字しない** —— '
              '外れた後に見てから次の問いを作れば、それは点を見て棒を動かしたのと同じになる。')
        print('\n--- 以下は**器具と問いについての診断。棒ではない**(主値とは独立)---')
        diagnostic(kmap)
        return
    print(f'  IC {ic_ok}/{ic_n} 一致' + ('(**まだ全部そろっていない**)' if ic_n < 3 else ''))

    ansd = [v for v in kmap.values() if v.get('value') in ('one', 'many')]
    held = [v for v in kmap.values() if v.get('value') in ('__unknown__', '__none__')]
    n_done = len(ansd) + len(held)
    rate = len(held) / n_done if n_done else 0
    print(f'\n--- 答えられなかった率(C5)---')
    print(f'  {len(held)}/{n_done} = {rate:.1%}'
          f'(第1回の実測: 一様30面 26.7% / 面だけ 17.6% / 小片 38.5%)')
    print(f'  内訳: ' + ', '.join(f'{k} {v}' for k, v in
                                 collections.Counter(v['value'] for v in held).items()))
    if rate >= 0.35:
        print('  **35% 以上 —— 問いが測っていない。登録どおり他の棒を読まない。**')
        return
    if rate >= 0.25:
        print('  **25〜35% —— 棒は読むが、問いの作りの欠陥として記録する**(作り手の負け)')

    print('\n--- 主値: 群T(層 >= 2)と 群C(対照)の「2つ以上」率 ---')
    pairs = collections.defaultdict(dict)
    for v in kmap.values():
        if v['group'] in ('T', 'C') and v['pair'] is not None:
            pairs[v['pair']][v['group']] = v

    def cell(rows):
        m = sum(1 for v in rows if v['value'] == 'many')
        return m, len(rows)

    def tab(sel, name):
        T = [p['T'] for p in pairs.values() if sel(p) and p['T'].get('value') in ('one', 'many')]
        C = [p['C'] for p in pairs.values() if sel(p) and p['C'].get('value') in ('one', 'many')]
        (mt, nt), (mc, nc) = cell(T), cell(C)
        if not nt or not nc:
            print(f'  {name:12} 答えが足りない(群T {nt} / 群C {nc})')
            return None
        p = fisher_exact([[mt, nt - mt], [mc, nc - mc]])[1]
        lo, hi = wilson(mt, nt)
        print(f'  {name:12} 群T {mt}/{nt} = {mt / nt:>6.1%}(Wilson {lo:.1%}〜{hi:.1%})'
              f'  群C {mc}/{nc} = {mc / nc:>6.1%}'
              f'  差 **{mt / nt - mc / nc:+.1%}**  Fisher 両側 p = {p:.4f}')
        return mt, nt, mc, nc, p

    main_res = tab(lambda p: True, '29対ぜんぶ')
    for w in sorted({v['work'] for v in kmap.values()}):      # **作品は畳まない**(条例3)
        tab(lambda p, w=w: p['T']['work'] == w, f'{w} だけ')

    up = [v for v in kmap.values() if v['group'] == 'T_unpaired'
          and v.get('value') in ('one', 'many')]
    if up:
        m, n = cell(up)
        areas = ', '.join(f'{v["area"]:,}' for v in up)
        print(f'  **対に入れなかった群Tの面(2x2 には入れない)**: {m}/{n} が「2つ以上」'
              f' —— 面積 {areas}')

    if main_res:
        mt, nt, mc, nc, p = main_res
        import random
        rnd = random.Random(SEED)
        obs = mt / nt - mc / nc
        pool = [(p_['T'].get('value'), p_['C'].get('value')) for p_ in pairs.values()
                if p_['T'].get('value') in ('one', 'many')
                and p_['C'].get('value') in ('one', 'many')]
        hits = 0
        diffs = []
        for _ in range(2000):
            a = b = 0
            for t, c in pool:
                if rnd.random() < 0.5:
                    t, c = c, t
                a += t == 'many'
                b += c == 'many'
            d = (a - b) / len(pool)
            diffs.append(d)
            hits += abs(d) >= abs(obs) - 1e-12
        print(f'\n  並べ替え 2,000回(**NC3。群の札を対の中で入れ替える**): '
              f'差の中央 {statistics.median(diffs):+.1%}、|差| >= 実測 が {hits}/2000 '
              f'= p {hits / 2000:.4f}')

    print('\n--- 腕の表(**自明な腕を同じ表に置く**。条例2)---')
    rows = ansd
    y = [v['value'] == 'many' for v in rows]
    med = statistics.median(v['area'] for v in rows)
    arms = [('腕0 何も言わない', lambda v: False),
            ('腕1 全部2つ以上', lambda v: True),
            ('腕2 層の軸(層>=2)', lambda v: v['n_layer'] >= 2),
            ('腕3 首の軸(細い首)', lambda v: v['neck'] is not None),
            (f'腕4 面積 >= 中位({med:,.0f})', lambda v: v['area'] >= med)]
    print(f'  母集団 {len(rows)} 面(答えが one/many だったもの)/ うち「2つ以上」 {sum(y)}')
    for nm, f in arms:
        pred = [f(v) for v in rows]
        tp = sum(a and b for a, b in zip(pred, y))
        acc = sum(a == b for a, b in zip(pred, y)) / len(rows)
        prec = tp / sum(pred) if sum(pred) else float('nan')
        rec = tp / sum(y) if sum(y) else float('nan')
        print(f'  {nm:26} 的中 {acc:>6.1%}  言い当て(precision) {prec:>6.1%}'
              f'  取りこぼさず(recall) {rec:>6.1%}  言った数 {sum(pred)}')
    print('  **腕4 が腕2 に並ぶなら、対の中で面積が漏れている**(対は面積を合わせてある)')

    if main_res:
        mt, nt, mc, nc, p = main_res
        d = mt / nt - mc / nc
        print('\n--- 棒の判定(登録のとおり)---')
        if p < 0.05 and d > 0:
            print(f'  **C1 達成**(p = {p:.4f} < 0.05、差 {d:+.1%} > 0)—— '
                  f'**層の軸は過剰結合を当てる。** 次は面を層で割る器具')
        elif p < 0.05 and d < 0:
            print(f'  **C6**(p = {p:.4f} だが 差 {d:+.1%} < 0。逆向きに有意)—— '
                  f'**器具を疑う。** 群の札と対の作りを先に見る。棒の話をしない')
        elif d <= 0.10:
            print(f'  **C3**(差 {d:+.1%} <= +10点)—— **層の軸は過剰結合を当てない。** '
                  f'11-07 の「5/6 対 1/3」は面積の選び方の産物だった。この道は閉じる(条例8)。'
                  f'**過去の数は撤回しない**(選んだ9面での観察として残る)')
        elif d >= 0.20:
            print(f'  **C2 判定保留**(差 {d:+.1%} >= 20点 だが p = {p:.4f})—— '
                  f'40頁に層 >= 2 は32面しか無い。**頁を増やすしか道が無い**')
        else:
            print(f'  **C2 判定保留**(差 {d:+.1%} は +10〜+20点 の帯。登録どおり保留)')
        if nt and nc and mt / nt >= 0.60 and mc / nc >= 0.60:
            print(f'  **C4 も立つ**(両群 60% 以上が「2つ以上」)—— 問いか単位の欠陥。単位に戻る')

    cm = [v for v in kmap.values() if v.get('comment')]
    print(f'\n--- 一言の欄({len(cm)} 件。**多ければ問いの作りの欠陥**)---')
    for v in cm:
        print(f'  面積 {v["area"]:>9,} 群{v["group"]:<11} {v.get("value")}: {v["comment"]}')


if __name__ == '__main__':
    main()
