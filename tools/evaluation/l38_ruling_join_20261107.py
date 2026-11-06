# -*- coding: utf-8 -*-
"""作者の裁定(`labels/face_meaning_count.jsonl`)を面の素性に結んで印字する(2026-11-07)。

**これは測定ではない。** 棒も帰無値も登録していないので、出るのは**記述**である
(規律9)。**p の値は試験ではない** —— 床(しきい値)を**答を見た後に選んでいる**ものが
在るので、そう書いた上でしか出さない(条例1)。

結び方: **`(work, pid, comp, region, area)`** —— `comp` は枠ファイルごとに振り直されるので
**鍵だけでは同定にならず**、面積だけでも **1,130面に 55 の重複**が在る。両方で引く
(2026-11-07 に同じ衝突を4回踏んだ後の規則)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_ruling_join_20261107.py
"""
import collections
import json
import math
import os

ROOT = '/home/sh1/deepl/lineart-fill-region'
SAVE = f'{ROOT}/labels/face_meaning_count.jsonl'
CAUSE = f'{ROOT}/results/l38_merge_cause_20261106.json'
AUDIT = f'{ROOT}/results/l38_merge_audit_20261106.json'


def key(r):
    return (r['work'], r['pid'], r['comp'], r['region'], r['area'])


def fold():
    """保存を**ファイルの順に**畳む。`undo` はその時の答を消す(**行は残る**。規律7)。

    **同じ面に新しい行が来たら後の行が生きる。** 2026-11-07、作者は1面目を
    語なしで保存した直後に語を入れて保存し直している —— `undo` は無く、2行とも残る。
    """
    rows = [json.loads(l) for l in open(SAVE, encoding='utf-8') if l.strip()]
    cur, over = {}, 0
    for r in rows:
        k = key(r)
        if r.get('kind') == 'undo':
            cur.pop(k, None)
        else:
            over += k in cur
            cur[k] = r
    return rows, cur, over


def fisher(ans, pred):
    t = collections.Counter((pred(r), r['label'] == '2つ以上') for r in ans)
    a, b = t[(True, True)], t[(True, False)]
    c, d = t[(False, True)], t[(False, False)]
    n, C = a + b + c + d, math.comb
    if min(a + b, c + d) == 0 or min(a + c, b + d) == 0:
        return a, b, c, d, float('nan')
    p0 = C(a + b, a) * C(c + d, c) / C(n, a + c)
    p = 0.0
    for i in range(min(a + b, a + c) + 1):
        j = a + c - i
        if 0 <= j <= c + d:
            q = C(a + b, i) * C(c + d, j) / C(n, a + c)
            if q <= p0 + 1e-12:
                p += q
    return a, b, c, d, p


def main():
    rows, cur, over = fold()
    cause = {key(r): r for r in json.load(open(CAUSE, encoding='utf-8'))}
    audit = {key(r): r for r in json.load(open(AUDIT, encoding='utf-8'))}
    ans, miss = [], collections.Counter()
    for k, r in cur.items():
        if k not in cause or k not in audit:
            miss['素性に当たらない(**鍵と面積の両方で引いている**)'] += 1
            continue
        r = dict(r, **{m: audit[k][m] for m in
                       ('n_layer_1pct', 'top_share', 'split_if_unmerged')})
        r['neck'] = cause[k]['neck']
        ans.append(r)
    print(f'書かれた行 {len(rows)} / 生きている答 {len(cur)} / 素性に結べた {len(ans)}')
    print(f'  **上書きされた答 {over}**(後の行が生きる。前の行は消えていない)')
    for k, v in miss.items():
        print(f'  落とした {v}  {k}')
    print()
    hd = f"{'群':9}{'層':>3}{'首':>5}{'支配率':>8}{'畳まねば':>6}{'面積':>11}  裁定"
    print(hd + '    語')
    for r in sorted(ans, key=lambda r: (r['cell'], -r['area'])):
        print(f"{r['cell']:9}{r['n_layer']:>3}{str(r['neck']):>5}{r['top_share']:>8.4f}"
              f"{r['split_if_unmerged']:>6}{r['area']:>11,}  {r['label']:5} "
              f"{r['word'] or '—'}"
              + ('   **既知セル: ' + r['known'] + '**' if r['known'] else ''))
    print()
    print('**床を答の後に選んだものには印を付ける。印の付いた行の p は試験ではない**')
    for nm, pr, post in [
            ('層 >= 2', lambda r: r['n_layer'] >= 2, False),
            ('首あり', lambda r: r['neck'] is not None, False),
            ('支配率 < 0.9', lambda r: r['top_share'] < 0.9, True),
            ('支配率 < 0.99', lambda r: r['top_share'] < 0.99, True)]:
        a, b, c, d, p = fisher(ans, pr)
        print(f"  {'**後から選んだ床**' if post else '先に在った軸    '} {nm:14}"
              f' 真 {a}/{a + b} が「2つ以上」 / 偽 {c}/{c + d}   Fisher 両側 p={p:.3f}')
    print()
    w = [r['word'] for r in ans if r['label'] == '2つ以上']
    print(f'「2つ以上」の語 {len(w)} 件 —— **{sum(1 for x in w if "髪" in x)} 件に「髪」が在る**')
    for x in w:
        print(f'  {x}')


if __name__ == '__main__':
    main()
