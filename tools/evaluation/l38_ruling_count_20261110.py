# -*- coding: utf-8 -*-
"""床の引き金(条件 1)を数える —— 数え方は `l38_units_20261107.py` 冒頭(2026-11-10)。

段階 (i): `--trace 10` で、一様に引いた 10 件を**答え -> 鍵 -> 単位**まで1件ずつ印字する。
段階 (ii): 引数なしで全件を数える(**ユーザーが (i) を確かめてから**)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_ruling_count_20261110.py --trace 10
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ROOT = '/home/sh1/deepl/lineart-fill-region'
PRE = f'{ROOT}/results/l38_merge_cause_20261106.json'        # 分ける前(1,130 単位、面積 >= 501)
SEP = f'{ROOT}/results/l38_faces_separated_20261108.json'    # 分けた後の面プール 656
# 面を問う聞き取り(L39 はコマを問うので入れない)
INTERVIEWS = ['l38_solidcheck_20261107', 'l38_meaning_20261107', 'l38_layeraxis_20261107',
              'l38_problemb_20261108', 'l38_decompose_20261108', 'l38_bleed_20261108',
              'l38_overlap_20261109']
NO_ANSWER = {'__unknown__', '__none__', '__hold__'}
SEED = 20261110


def unit_key(m):
    """聞き取りの map の1項目 -> 単位の鍵。重ね塗りの問いは**親の面**を問うている。"""
    if isinstance(m, list):                                   # overlap: 片のリスト。問うたのは親
        p = m[0]
        return ('face', p['work'], p['pid'], p['comp'], p['region'], p['parent_area'])
    if m.get('origin') == 'piece':
        return ('piece', tuple(m['parent']), m['piece_lid'], m['area'])
    return ('face', m['work'], m['pid'], m['comp'], m['region'], m['area'])


def load():
    out, skip = [], collections.Counter()
    for iv in INTERVIEWS:
        d = f'{ROOT}/data/{iv}'
        m = json.load(open(f'{d}/map.json', encoding='utf-8'))
        rows = [json.loads(l) for f in glob.glob(f'{d}/*_answers.jsonl') for l in open(f)]
        last = {}
        for r in rows:
            last[r['item_id']] = r                             # 問いごとに最新
        skip[f'{iv}: 選び直しで古くなった行'] += len(rows) - len(last)
        for iid, r in sorted(last.items()):
            if iid not in m:
                skip[f'{iv}: map に無い問い'] += 1
                continue
            out.append(dict(iv=iv, item=iid, value=r['value'], label=r.get('label'),
                            key=unit_key(m[iid]), m=m[iid]))
    return out, skip


def pools():
    pre = {('face', r['work'], r['pid'], r['comp'], r['region'], r['area']): r
           for r in json.load(open(PRE, encoding='utf-8'))}
    sep = json.load(open(SEP, encoding='utf-8'))
    kept = {('face', r['work'], r['pid'], r['comp'], r['region'], r['area']) for r in sep
            if r['origin'] == 'kept'}
    piece = {('piece', tuple(r['parent']), r['piece_lid'], r['area']) for r in sep
             if r['origin'] == 'piece'}
    parents = {('face',) + tuple(r['parent']) for r in sep if r['origin'] == 'piece'}
    return pre, kept, piece, parents


def where(k, P):
    pre, kept, piece, parents = P
    if k[0] == 'piece':
        return '面プール656の片' if k in piece else '**片が656に無い**'
    s = []
    if k in kept:
        s.append('面プール656(元のまま)')
    if k in parents:
        s.append('656では片に割られた親')
    if k in pre:
        s.append('分ける前の1,130')
    elif k[5] < 501:
        s.append('分ける前の表の外(面積<501)')
    else:
        s.append('**分ける前の1,130に無い**')
    return ' / '.join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', type=int, default=0)
    a = ap.parse_args()
    rs, skip = load()
    P = pools()
    print(f'面を問う聞き取り {len(INTERVIEWS)} 本 / 問い(最新の答え) {len(rs)}')
    for k, v in skip.items():
        if v:
            print(f'  落とした {v}  {k}')
    if a.trace:
        rng = np.random.default_rng(SEED)
        ix = sorted(rng.choice(len(rs), a.trace, replace=False))
        print(f'\n=== 段階 (i): 一様に {a.trace} 件(default_rng({SEED}))===')
        for n, i in enumerate(ix, 1):
            r = rs[i]
            k = r['key']
            same = [x for x in rs if x['key'] == k]
            print(f'{n:2d}. {r["iv"]} {r["item"]}  答え「{r["label"]}」({r["value"]})')
            print(f'    単位 {k[0]}  {k[1:]}')
            print(f'    在りか: {where(k, P)}  / 同じ単位への問い {len(same)} '
                  f'{[(x["iv"].split("_")[1], x["item"], x["value"]) for x in same]}')
            crop = glob.glob(f'{ROOT}/data/{r["iv"]}/crops/{r["item"]}*')
            print(f'    切り出し {crop[0] if crop else "(無い)"}')
        print('\n**全件の数は段階 (ii)。ユーザーの確認の後に出す**')
        return
    by = collections.defaultdict(list)
    for r in rs:
        by[r['key']].append(r)
    ans = {k for k, v in by.items() if any(x['value'] not in NO_ANSWER for x in v)}
    print(f'単位 {len(by)} / **答えの在る単位 {len(ans)}** / 答えの無い単位 {len(by) - len(ans)}')
    w = collections.Counter(where(k, P).split(' / ')[0] for k in ans)
    for k, v in w.most_common():
        print(f'  {v:>4}  {k}')


if __name__ == '__main__':
    main()
