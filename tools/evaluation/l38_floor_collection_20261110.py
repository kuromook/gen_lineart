# -*- coding: utf-8 -*-
"""床の引き金(条件 1)の段階 (iii): **床の器具が使う集まり**で数える(2026-11-10、ユーザー「進めて」)。

数え方の正本は `l38_units_20261107.py` 冒頭。11-07 の器具 `l38_fragment_floor_20261107.py` は
**床 6,000 の出所なので書き換えない**。この器具は同じ形の集まり `(面積, 答えられなかったか, 出所)` を、
出所を広げて作る:

- L19 の面の語 `labels/fill_words.jsonl`(11-07 の器具と同じ結び方。**曖昧な鍵は落として数える**)
- 面を問う聞き取り **7 本**(段階 (ii) の `l38_ruling_count_20261110.load()` をそのまま使う。写しを持たない)

**答えられなかった単位も数える** —— 床の統計は「答えられなかった」を面積で分けるので、それが材料である
(段階 (ii) で答えの在る単位だけを数えたのとは違う。**前後を両方印字する**)。
1 単位に複数の答えが在れば、**どれか1つでも答えていれば「答えられた」**。

**床の推定値(Youden・ブートストラップ)はここでは出さない** —— 引き金(500)が引けるまで出すと、
結果を見て床を動かす道が開く(規律10)。出すのは数だけ。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_floor_collection_20261110.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_ruling_count_20261110 as C                               # noqa: E402

ROOT = C.ROOT
FW = f'{ROOT}/labels/fill_words.jsonl'
TRIGGER = 500


def collect():
    """単位の鍵 -> dict(area, answered(bool), src(set))。落とした数も返す。"""
    pre_rows = json.load(open(C.PRE, encoding='utf-8'))
    idx = collections.defaultdict(list)
    for r in pre_rows:
        idx[(r['work'], r['pid'], r['comp'], r['region'])].append(r)
    units, cut = {}, collections.Counter()

    def add(k, area, answered, src):
        u = units.setdefault(k, dict(area=area, answered=False, src=set()))
        u['answered'] |= answered
        u['src'].add(src)

    for ln in open(FW, encoding='utf-8'):
        if not ln.strip():
            continue
        r = json.loads(ln)
        h = idx.get((r['work'], r['pid'], r['comp'], r['region']), [])
        if len(h) != 1:
            cut['L19: 鍵が曖昧(同じ `comp` の別のコマ)か、現行40頁に無い面'] += 1
            continue
        f = h[0]
        add(('face', f['work'], f['pid'], f['comp'], f['region'], f['area']), f['area'],
            r['choice'] != 'skip', 'L19 面の語')
    rs, skip = C.load()
    for k, v in skip.items():
        if v:
            cut[k] += v
    for r in rs:
        add(r['key'], r['key'][-1], r['value'] not in C.NO_ANSWER, r['iv'])
    return units, cut


def main():
    units, cut = collect()
    n = len(units)
    u = sum(1 for x in units.values() if not x['answered'])
    print(f'段階 (iii) 床の器具の集まり: **単位 {n}**(引き金 {TRIGGER})'
          f' / うち答えられなかった {u} ({u / n:.1%})')
    for k, v in cut.most_common():
        print(f'  落とした {v}  {k}')
    only_l19 = sum(1 for x in units.values() if x['src'] == {'L19 面の語'})
    both = sum(1 for x in units.values() if 'L19 面の語' in x['src'] and len(x['src']) > 1)
    print(f'  出所: L19 の語だけ {only_l19} / 聞き取りだけ {n - only_l19 - both} / 両方 {both}')
    big = sum(1 for x in units.values() if x['area'] >= 6000)
    print(f'  床の上 {big} / 床の下 {n - big}')
    print('\n=== 前後(名前を付けて。どれも撤回しない)===')
    print('  11-07 床の器具の集まり(L19 + 聞き取り2本)            302(床 6,000 の出所)')
    print('  11-08 聞き取り6本の数え直し                             257')
    print('  11-10 段階 (ii) 聞き取り7本・答えの在る単位だけ         258')
    print(f'  11-10 段階 (iii) L19 + 面を問う聞き取り全部({len(C.INTERVIEWS)}本)・答えの無い単位も含む  **{n}**')
    print(f'\n**引き金: {n} {"≥" if n >= TRIGGER else "<"} {TRIGGER} -> '
          f'{"引き直す(床の推定は別の手で、登録どおりに)" if n >= TRIGGER else "引けない。床 6,000 は動かない"}**')


if __name__ == '__main__':
    main()
