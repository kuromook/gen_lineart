# -*- coding: utf-8 -*-
"""L38 チェックの答えを読む(2026-11-08)。登録: `doc/l38_decompose_registration_20261108.md`。

**この読み手は答えが来た後に書いた**(前の2走行では答えが0行の時点に書いていた)。
**棒は `7269543` で固定済み**(答えが0行の時点)。**書くときに答えの分布を見ていない** ——
`wc -l` も `Counter` も通していない。**それでも「先に書いた」より弱い。記録する。**
ここにあるのは**登録の文の写し**だけである:

  C1 一様標本の「要る」率 >= 25% / C2 枠線の層 − 一様 >= 30 ポイント / C3 <= 5%
  C4 答えられなかった率 >= 35% で何も読まない / C5 IC1 が外れたら何も読まない
  IC2 と P1 は外れても読む(理由は登録 §4 に先に書いた)
"""
import collections
import json
import math
from pathlib import Path

D = Path('/home/sh1/deepl/lineart-fill-region/data/l38_decompose_20261108')
ANS = D / 'l38_decompose_20261108_answers.jsonl'
IC1_AREA, IC2_AREA, P1_AREA = 1392416, 2503898, 1317982


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(name, rows, pad=26):
    k = sum(1 for r in rows if r['value'] == 'split')
    n = sum(1 for r in rows if r['value'] in ('split', 'whole'))
    p, lo, hi = wilson(k, n)
    print(f'  {name:<{pad}} {k}/{n} = **{p:.1%}**(Wilson {lo:.1%}〜{hi:.1%})')
    return k, n, p, lo, hi


def need(p, width=0.10):
    """Wilson の幅を `width` 以下にするのに要る面の数(**詰めるのに何件要るか**。登録 C)。"""
    for n in range(10, 20001):
        _, lo, hi = wilson(round(p * n), n)
        if hi - lo <= width:
            return n
    return None


def main():
    kmap = json.load(open(D / 'map.json', encoding='utf-8'))
    last = {}
    raw = 0
    for line in open(ANS, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            raw += 1
            last[r['item_id']] = r          # **最後の行が有効。前の行は消さない**(規律7)
    rows = []
    for fid, m in kmap.items():
        a = last.get(fid)
        if a:
            rows.append(dict(m, fid=fid, value=a.get('value'), comment=a.get('comment') or ''))
    print(f'答えの在る単位 {len(rows)}/{len(kmap)}(行 {raw}。**追記だけ**)')

    def grp(g):
        return [r for r in rows if r['group'] == g]

    print('\n--- 器具の点検(IC)と予測セル(P)---')
    verdict = {}
    for nm, area, want, note in (
            ('IC1', IC1_AREA, 'whole', '怪しい幾何が何も無い大きいベタ'),
            ('IC2', IC2_AREA, 'whole', '11-07 午前・夜ともに「1つのもの」。帯の覆い 0.198'),
            ('P1', P1_AREA, 'split', '作者が「右手・左腕・脇の線がまざってる」と書いた片。**予測**')):
        hit = [r for r in rows if r['area'] == area]
        if not hit:
            print(f'  {nm} が答えに無い(**黙って飛ばさない**)')
            continue
        got = hit[0]['value']
        ok = got == want
        verdict[nm] = ok
        print(f'  {nm} 面積 {area:,}: 期待 {want} 実際 {got}  '
              f'{"一致" if ok else "**外れた**"}  —— {note}')
        if hit[0]['comment']:
            print(f'      一言: {hit[0]["comment"]}')
    if verdict.get('IC1') is False:
        print('\n**C5 IC1 が外れた。登録どおり、この走行は何も読まない。**')
        return
    if 'P1' in verdict and not verdict['P1']:
        print('  **P1 が外れた** = 「意味が複数」と「分解が要る」は別物である'
              '(登録 §4 に先に書いた読み)')

    held = [r for r in rows if r['value'] not in ('split', 'whole')]
    r4 = len(held) / len(rows) if rows else 0
    print(f'\n--- 答えられなかった率(C4)---\n  {len(held)}/{len(rows)} = {r4:.1%}')
    if r4 >= 0.35:
        print('  **C4 35% 以上。他の棒を読まない。**')
        return
    if r4 >= 0.25:
        print('  **25〜35%。問いの欠陥として記録する**(棒は読む)')

    print('\n--- 主値(一様標本)と、枠線の層(**畳まない**)---')
    ku, nu, pu, lou, hiu = rate('一様標本', grp('uniform'))
    kb, nb, pb, lob, hib = rate('枠線を飲んだ層', grp('band'))
    print('  **腕0 全部「このままで1つ」= 0%、腕1 全部「分解が要る」= 100%。'
          'どちらの点も率そのもので恒等**(条例2)')

    print('\n  作品ごと(畳まない):')
    for g in ('uniform', 'band'):
        for w in sorted({r['work'] for r in grp(g)}):
            rate(f'{"一様" if g == "uniform" else "枠線"} / {w}',
                 [r for r in grp(g) if r['work'] == w])
    print('\n  一様標本の中の小行:')
    rate('首あり', [r for r in grp('uniform') if r['neck']])
    rate('首なし', [r for r in grp('uniform') if not r['neck']])
    rate('割った片', [r for r in grp('uniform') if r['origin'] == 'piece'])
    rate('元のままの面', [r for r in grp('uniform') if r['origin'] == 'kept'])

    par = collections.Counter(tuple(r['parent']) for r in grp('uniform') if r['parent'])
    dup = {k: v for k, v in par.items() if v >= 2}
    if dup:
        print(f'\n  **同じ親から2片以上 {len(dup)} 親**(標本は独立でない):')
        for k in dup:
            vals = [r['value'] for r in grp('uniform') if r['parent'] and tuple(r['parent']) == k]
            print(f'    親の面積 {k[4]:>9,} から {len(vals)} 片: {vals}')

    print('\n--- 棒の判定(登録のとおり)---')
    if pu >= 0.25:
        print(f'  **C1 達成**(一様 {pu:.1%} >= 25%)—— **B2 は広い。'
              f'面の標準を分解後の版に差し替える**(前後の数を両方印字する。規律10)')
    elif pu <= 0.05:
        print(f'  **C3 達成**(一様 {pu:.1%} <= 5%)—— **B2 は稀。個別の修理で済む。'
              f'プールの版は差し替えない**')
    else:
        n10 = need(pu)
        print(f'  **判定保留**(一様 {pu:.1%} は 5〜25% の間)。'
              f'Wilson の幅 {hiu - lou:.1%}')
        print(f'  **幅を 10 ポイント以下に詰めるのに要る面の数 {n10}**(いま {nu})')
    d = pb - pu
    if nb and nu:
        if d >= 0.30:
            print(f'  **C2 達成**(枠線 {pb:.1%} − 一様 {pu:.1%} = **+{d:.1%}** >= 30 ポイント)'
                  f'—— **作者の仮説の枠線側を支持。帯の覆いが腕2として使える**')
        else:
            print(f'  **C2 未達**(差 {d:+.1%} < 30 ポイント)—— '
                  f'**枠線は B2 の主な出所ではない**(仮説の片側が落ちる)')

    print('\n--- 一言の欄(**面積は出さない**)---')
    for r in rows:
        if r['comment']:
            print(f'    {r["value"]}: {r["comment"]}')

    sp = [r['fid'] for r in rows if r['value'] == 'split']
    print(f'\n**台に乗せる単位(「分解が要る」){len(sp)}**: {" ".join(sorted(sp))}')


if __name__ == '__main__':
    main()
