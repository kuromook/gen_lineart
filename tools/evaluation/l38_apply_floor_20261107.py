# -*- coding: utf-8 -*-
"""床 6,000 を当てたとき、**これまでの数がどう動くか**を前後で印字する(2026-11-07)。

規律10 —— **動く錨は、前後の数を両方、名前を付けて印字する。過去の結果は撤回しない。**
**この器具は数を書き換えない。** 同じ量を**2つの単位**で並べるだけである。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_apply_floor_20261107.py
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l38_units_20261107 as U                                   # noqa: E402

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CAUSE = ROOT / 'results' / 'l38_merge_cause_20261106.json'
SOLID = ROOT / 'results' / 'l38_face_is_solid_20261107.json'
FRAME = ROOT / 'results' / 'l38_frame_axis_20261107.json'
IV = ROOT / 'data' / 'l38_meaning_20261107'


def wilson(k, n, z=1.96):
    if not n:
        return (float('nan'), float('nan'))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    print(f'床 **{U.FRAGMENT_FLOOR:,} 画素**(登録 {U.FLOOR_REGISTERED}、作者が選んだ)。'
          f'**小片は捨てない。別プール**')
    cause = json.load(open(CAUSE, encoding='utf-8'))
    face, frag = U.split(cause)
    ta = sum(r['area'] for r in cause)
    print(f'\n--- 母集団(現行40頁)---')
    print(f'  面の数:       **前 {len(cause):,}** -> **面 {len(face):,}** / 小片 {len(frag):,}'
          f'({len(frag) / len(cause):.1%})')
    print(f'  塗られた面積: **前 {ta:,}** -> 面 {sum(r["area"] for r in face):,}'
          f'({sum(r["area"] for r in face) / ta:.2%}) / '
          f'小片 {sum(r["area"] for r in frag):,}({sum(r["area"] for r in frag) / ta:.2%})')

    print('\n--- 軸の数(**同じ量を2つの単位で**)---')
    def row(nm, pred, rows=cause):
        f, g = U.split([r for r in rows if pred(r)])
        a, b = U.split(rows)
        print(f'  {nm:26} 前 {len(f) + len(g):>4}/{len(rows):<5}'
              f'({(len(f) + len(g)) / len(rows):>5.1%})   '
              f'面だけ {len(f):>4}/{len(a):<5}({len(f) / len(a):>5.1%})')
    row('首あり(細い首)', lambda r: r['neck'] is not None)
    row('層 >= 2', lambda r: r['n_layer'] >= 2)
    sol = json.load(open(SOLID, encoding='utf-8'))
    row('形(細い割合 >= 0.55)', lambda r: r['thin_frac'] >= 0.55, sol)
    row('墨(純黒 < 0.80)',
        lambda r: r['pure_black'] is not None and r['pure_black'] < 0.80, sol)
    fr = json.load(open(FRAME, encoding='utf-8'))
    row('コマ枠線を飲み込んだ面', lambda r: r['band_cov'] >= 0.5, fr)

    print('\n--- 帰無値(無作為30面。**選び直していない。床で分けただけ**)---')
    kmap = json.loads((IV / 'map.json').read_text(encoding='utf-8'))
    ans = {}
    for ln in open(IV / 'l38_face_meaning_20261107_answers.jsonl', encoding='utf-8'):
        if ln.strip():
            r = json.loads(ln)
            ans[r['item_id']] = r
    rd = [(kmap[i]['area'], ans[i]['value']) for i in kmap
          if kmap[i]['why'].startswith('無作為')]
    for nm, g in (('前(床なし)', rd),
                  ('面だけ', [r for r in rd if r[0] >= U.FRAGMENT_FLOOR]),
                  ('小片だけ', [r for r in rd if r[0] < U.FRAGMENT_FLOOR])):
        if not g:
            continue
        m = sum(1 for a, v in g if v == 'many')
        u = sum(1 for a, v in g if v in ('__unknown__', '__none__'))
        lo, hi = wilson(m, len(g))
        print(f'  {nm:12} 面 {len(g):>3}  「2つ以上」{m}({m / len(g):>5.1%}'
              f'、Wilson {lo:.1%}〜{hi:.1%})  答えられず {u}({u / len(g):.0%})')
    print('\n  **過去の数は撤回しない。** `面 1,130` も `0/30` も、'
          'その時の単位での数である(規律1)')


if __name__ == '__main__':
    main()
