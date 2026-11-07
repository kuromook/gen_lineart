# -*- coding: utf-8 -*-
"""L38: **小片の床のあたりを、これまでの裁定の統計から出す**(2026-11-07)。

作者の裁定: **「面とも線ともつかない単位をつくるべきだろう　小さすぎるものは」**、
続けて **「まずこれまでの裁定の統計からあたりをつけて」**。

集めるのは **作者が答えた面**と、**その面で答えられたかどうか**:

- `labels/fill_words.jsonl`(L19、面ごとの語 628件)—— **`choice=skip` が「保留」**
- 2026-11-07 の聞き取り2回(33 + 24)—— **`どれでもない` / `分からない`**

**結びは `(work, pid, comp, region)`。`comp` は枠ファイルごとに振り直されるので曖昧になる** ——
**曖昧な鍵は落として数える**(規律9)。落とした中身は「現行40頁に無い面」も含む。

出すのは3つ: **帯ごとの「答えられなかった」率** / **Youden の J で最良の床**(並べ替え検定つき)/
**ブートストラップでの床の揺れ**。**床はこの器具では決めない** —— 決めるのは作者(規律6)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_fragment_floor_20261107.py
"""
import collections
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CAUSE = ROOT / 'results' / 'l38_merge_cause_20261106.json'
FW = ROOT / 'labels' / 'fill_words.jsonl'
IVS = (('data/l38_meaning_20261107', 'l38_face_meaning_20261107_answers.jsonl'),
       ('data/l38_solidcheck_20261107', 'l38_face_is_solid_20261107_answers.jsonl'))
BANDS = ((0, 1000), (1000, 2000), (2000, 5000), (5000, 10000),
         (10000, 20000), (20000, 100000), (100000, 10 ** 12))


def youden(A, U):
    out = []
    for T in sorted(set(A.tolist())):
        p = A < T
        tp, fn = int((p & U).sum()), int(((~p) & U).sum())
        fp, tn = int((p & ~U).sum()), int(((~p) & ~U).sum())
        if tp + fn == 0 or fp + tn == 0:
            continue
        out.append((tp / (tp + fn) - fp / (fp + tn), int(T), tp, fn, fp, tn))
    out.sort(reverse=True)
    return out


def main():
    idx = collections.defaultdict(list)
    for r in json.load(open(CAUSE, encoding='utf-8')):
        idx[(r['work'], r['pid'], r['comp'], r['region'])].append(r)
    data, cut = {}, collections.Counter()
    for ln in open(FW, encoding='utf-8'):
        if not ln.strip():
            continue
        r = json.loads(ln)
        h = idx.get((r['work'], r['pid'], r['comp'], r['region']), [])
        if len(h) != 1:
            cut['鍵が曖昧(同じ `comp` の別のコマ)か、現行40頁に無い面'] += 1
            continue
        f = h[0]
        data[(f['work'], f['pid'], f['comp'], f['region'], f['area'])] = (
            f['area'], r['choice'] == 'skip', 'L19 面の語')
    for d, af in IVS:
        km = json.loads((ROOT / d / 'map.json').read_text(encoding='utf-8'))
        for ln in open(ROOT / d / af, encoding='utf-8'):
            if not ln.strip():
                continue
            r = json.loads(ln)
            m = km[r['item_id']]
            data[(m['work'], m['pid'], m['comp'], m['region'], m['area'])] = (
                m['area'], r['value'] in ('__unknown__', '__none__'), '11-07 の聞き取り')
    rows = list(data.values())
    n = len(rows)
    u = sum(1 for r in rows if r[1])
    print(f'裁定の在る面 **{n}**(同じ面は後の回。**行は消していない**)')
    for k, v in cut.most_common():
        print(f'  落とした {v}  {k}')
    print(f'  うち**答えられなかった**(保留 / 分からない / どれでもない)**{u} = {u / n:.1%}**')
    print('  出所: ' + ' / '.join(f'{k} {v}' for k, v in
                                collections.Counter(r[2] for r in rows).most_common()))
    print('\n--- 帯ごとの「答えられなかった」率 ---')
    for lo, hi in BANDS:
        g = [r for r in rows if lo <= r[0] < hi]
        if not g:
            continue
        k = sum(1 for r in g if r[1])
        print(f'  {lo:>8,} 〜 {hi if hi < 10 ** 12 else "":>9}  面 {len(g):>4}'
              f'  答えられず {k:>3}({k / len(g):>5.1%})')

    A = np.array([r[0] for r in rows])
    U = np.array([r[1] for r in rows])
    best = youden(A, U)
    print('\n--- 「答えられない」を当てる床(Youden の J)---')
    for J, T, tp, fn, fp, tn in best[:3]:
        print(f'  床 {T:>8,}  J={J:.3f}  拾った {tp}/{tp + fn}'
              f' / 巻き込み {fp}/{fp + tn}')
    rnd = random.Random(7)
    obs, hitn = best[0][0], 0
    for _ in range(2000):
        s = U.copy()
        rnd.shuffle(s)
        b = youden(A, s)
        hitn += bool(b) and b[0][0] >= obs
    print(f'  **並べ替え検定**: 無作為に並べ替えて同じ以上 {hitn}/2000 = **p={hitn / 2000:.3f}**'
          f'(= **大きさは当てずっぽうではない**)')
    bs = []
    for _ in range(400):
        ix = [rnd.randrange(n) for _ in range(n)]
        b = youden(A[ix], U[ix])
        if b:
            bs.append(b[0][1])
    bs.sort()
    print(f'  **ブートストラップ {len(bs)} 回の床**: 中央 **{bs[len(bs) // 2]:,}** / '
          f'5% {bs[int(.05 * len(bs))]:,} / 95% {bs[int(.95 * len(bs))]:,}'
          f' —— **床の位置は定まらない**')
    hi = [r for r in rows if r[0] >= best[0][1]]
    print(f'\n  **床の上でも「答えられなかった」は消えない**: '
          f'{sum(1 for r in hi if r[1])}/{len(hi)} = '
          f'{sum(1 for r in hi if r[1]) / len(hi):.1%} —— '
          f'**小片 ≠ 分からない**。床は「分からない」を消す道具ではない')


if __name__ == '__main__':
    main()
