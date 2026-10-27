# -*- coding: utf-8 -*-
"""L27: 無作為の枠での**食い違いの上限**。事前登録は `doc/work_log.md § L27 事前登録`。

**AUC は出さない**(無作為ではベタが約1/16で、60層では区間が濃縮標本より広い)。
出すのは **件数と、片側95%の上限**。**0件でも「0%」とは書かない。**

母集団 = 掃き出された描画層 2,963 のうち、L25 の `E_random` 15層 + L27 の 60層。
**同じ枠なので合わせて1つの無作為標本として読む。**

  python tools/evaluation/l27_bounds.py
"""
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
import l25_calibrate as C

OUT = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui'
Q2 = f'{OUT}/queue_l27.json'
L2 = '/home/sh1/deepl/lineart-fill-region/labels/region_word_l27.jsonl'
ART = C.LINEISH + C.FILLISH + C.MIXED          # **登録どおりの「完成の絵」**


def upper95(k, n):
    """Clopper-Pearson の片側95%上限。k 件 / n 回。**0件でも 0 とは返らない。**"""
    if n == 0:
        return float('nan')
    lo, hi = 0.0, 1.0
    from math import comb
    def cdf(p):                     # P(X <= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def ci95(k, n):
    """Clopper-Pearson の両側95%区間。**登録した要約(片側上限)が 0件向けだったので足した。**

    登録では「件数と片側95%上限」としたが、それは**0件のときに「0%」と書かないため**の形。
    件数が多い側に出たときは上限が役に立たないので、**区間を併記する**。
    **棒を動かしたのではなく、同じ量の正直な要約を足した**(2026-10-25、結果を見た後)。
    """
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def load_l27():
    q = json.load(open(Q2))
    byo = {r['order']: r for r in q['queue']}
    last = {}
    for ln in open(L2):
        r = json.loads(ln)
        last[r['order']] = r
    import csv
    import l14_known_cells as K
    inv = {(r['page_id'], r['layer_id']): r
           for r in csv.DictReader(open(K.INV, encoding='utf-8-sig'))}
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(K.KEEP4))}
    out = []
    for o, r in sorted(last.items()):
        it = byo[o]
        key = (it['pid'], str(it['lid']))
        iv, kk = inv.get(key, {}), k4.get(key, {})
        ans = r['word'] if r['choice'] == 'word' else (
            r['free'].strip() if r['choice'] == 'free' and r.get('free') else r['choice'])
        out.append(dict(order=o, pid=it['pid'], lid=it['lid'], stratum=it['why'],
                        ans=ans, name=iv.get('name', ''),
                        preview_dark=C.fnum(iv.get('preview_dark')),
                        black_among_drawn=C.fnum(iv.get('black_among_drawn')),
                        keep4=C.fnum(kk.get('keep4'))))
    return out


def main():
    r25, _ = C.load()
    r27 = load_l27()
    # --- 既知セル(L27 の再出題)
    byl = {}
    for r in r27:
        byl.setdefault((r['pid'], r['lid']), []).append(r)
    rep = {k: v for k, v in byl.items() if len(v) > 1}
    same = sum(1 for v in rep.values() if len({x['ans'] for x in v}) == 1)
    print(f'[門] 既知セル(同じ層の再出題) {len(rep)} 枚中 **{same} 枚が同じ答**')
    for k, v in rep.items():
        if len({x['ans'] for x in v}) > 1:
            print(f'    **違う**: {k[0]} L{k[1]}: ' + ' / '.join(x['ans'] for x in v))
    print('  (**5枚なので再現性は数として言わない**)')

    # --- 無作為の標本をまとめる (枠が同じ。層は1つだけ数える)
    rnd = {}
    for r in r25:
        if r['stratum'].startswith('E_random'):
            rnd[(r['pid'], r['lid'])] = r
    for r in r27:
        rnd[(r['pid'], r['lid'])] = r
    rows = list(rnd.values())
    print(f'\n[無作為の標本] **{len(rows)} 層** '
          f'(L25 の E_random + L27 の 60層。同じ枠 2,963 層から)')
    for a, c in Counter(r['ans'] for r in rows).most_common():
        print(f'  {c:>4}  {a}')
    free = [r for r in rows if r['ans'] not in
            list(ART) + ['絵ではない(マスク・紙・下地・トーン)',
                         '下絵・ラフ(ペンでも。完成の絵ではない)', '分からない']]
    if free:
        print('  **登録の語に入らない答(自由入力)。数に入れず、そのまま出す**:')
        for a, c in Counter(r['ans'] for r in free).most_common():
            print(f'    {c:>3}  {a}')

    # --- 登録どおりの食い違い
    print('\n[食い違い] **登録した事象だけ数える。件数と片側95%上限を併記**')
    print('  **言葉の意味は定数ごとに違う**(2026-10-25 作者の指摘で直した):')
    print('    `black_among_drawn` 拒む/受ける = **その1条件の判定**。')
    print('      **ラスタに残るかどうかではない**(門には他の条件もある)。')
    print('      ただし**1つでも満たさなければ通らない**ので、拒まれれば確実に外れる')
    print('    `preview_dark`  外す/入れる = **Track L の母集団に入るかどうか**')
    print('    `keep4`  振る = **形の分類(ベタの形 / 線の形)。何も捨てていない**')
    print('  **登録した要約は片側上限だが、それは0件のための形**なので、')
    print('  件数が多い側に出た行では**両側95%区間**を併記する(結果を見た後に足した)')
    # **「通す/落とす」は定数ごとに意味が違う。`keep4` は何も捨てていない**
    # (2026-10-25 作者の指摘。表だけ見ると「keep4 が57層を捨てた」と読めてしまう)
    jobs = [
        ('preview_dark >= 0.5', 'preview_dark', lambda v: v >= 0.5,
         '**L の母集団から外した**層のうち「完成の絵」', False, ART),
        ('black_among_drawn <= 0.98', 'black_among_drawn', lambda v: v <= 0.98,
         '**この条件が拒んだ**層のうち「線」', False, C.LINEISH),
        ('black_among_drawn <= 0.98', 'black_among_drawn', lambda v: v <= 0.98,
         '**この条件が受けた**層のうち「絵ではない / 下絵」', True,
         C.NOTART + C.ROUGH),
        ('keep4 >= 0.95', 'keep4', lambda v: v >= 0.95,
         '**「ベタの形」に振った**のに作者は「線」', True, C.LINEISH),
        ('keep4 >= 0.95', 'keep4', lambda v: v >= 0.95,
         '**「線の形」に振った**のに作者は「ベタ」', False, C.FILLISH),
    ]
    for label, col, keep, what, side, cats in jobs:
        s = [r for r in rows if r[col] is not None]
        sub = [r for r in s if keep(r[col]) == side]
        k = sum(1 for r in sub if r['ans'] in cats)
        ub = upper95(k, len(sub))
        lo, hi = ci95(k, len(sub))
        print(f'  {label:<28}{what}')
        print(f'      **{k} 件 / {len(sub)} 層 = {k/max(len(sub),1):.4f}**   '
              f'片側95%上限 {ub:.4f}   **両側95% [{lo:.4f}, {hi:.4f}]**'
              + ('   (0件。**「0%」ではない**)' if k == 0 else ''))
        if k:
            for r in sub:
                if r['ans'] in cats:
                    print(f'        {r["pid"]} L{r["lid"]} "{r["name"]}" -> {r["ans"]}'
                          f'  ({col}={r[col]})')
    print('\n**AUC はここでは出さない**(登録どおり)。'
          '無作為でベタは約1/16で、74層では濃縮標本より区間が広い')
    json.dump([{k: r[k] for k in ('pid', 'lid', 'name', 'stratum', 'ans',
                                  'preview_dark', 'keep4', 'black_among_drawn')}
               for r in rows], open(f'{OUT}/random_layers.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'-> {OUT}/random_layers.json  ({len(rows)} 層)')


if __name__ == '__main__':
    main()
