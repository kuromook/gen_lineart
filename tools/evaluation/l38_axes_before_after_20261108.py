# -*- coding: utf-8 -*-
"""軸の数を**分離の前と後で**並べる(2026-11-08。規律10 —— 両方、名前を付けて印字する)。

**この器具は数を書き換えない。** 同じ軸を2つの母集団で数えるだけである。
分離後の軸は `l38_separate_problem_a_20261108.py` が**同じ走りで測り直した値**
(旧い軸の数は旧い面の上の値なので、分離後のプールには当てられない)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_axes_before_after_20261108.py
"""
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                      # noqa: E402

R = Path('/home/sh1/deepl/lineart-fill-region/results')
BAND, THIN, PURE = 0.5, 0.55, 0.80


def main():
    old = json.load(open(R / 'l38_merge_cause_20261106.json', encoding='utf-8'))
    frame = {(r['work'], r['pid'], r['comp'], r['region'], r['area']): r['band_cov']
             for r in json.load(open(R / 'l38_frame_axis_20261107.json', encoding='utf-8'))}
    sol = {(r['work'], r['pid'], r['comp'], r['region'], r['area']): r
           for r in json.load(open(R / 'l38_face_is_solid_20261107.json', encoding='utf-8'))}
    bf, _ = U.split(old)
    for r in bf:
        k = (r['work'], r['pid'], r['comp'], r['region'], r['area'])
        r['band_cov'] = frame.get(k)
        s = sol.get(k) or {}
        r['thin_frac'], r['pure_black'] = s.get('thin_frac'), s.get('pure_black')
    new = json.load(open(R / 'l38_faces_separated_20261108.json', encoding='utf-8'))
    print(f'前(11-07 の面) {len(bf):,} / 後(分離した面) {len(new):,}'
          f'(割った片 {sum(1 for r in new if r["origin"] == "piece")} / '
          f'元のまま {sum(1 for r in new if r["origin"] == "kept")})')
    print('**軸は分離後のプールの上で測り直した値である**(旧い値を当て直していない)\n')

    def row(nm, pred):
        a = [r for r in bf if pred(r)]
        b = [r for r in new if pred(r)]
        p = [r for r in b if r['origin'] == 'piece']
        print(f'  {nm:28} 前 {len(a):>4}/{len(bf):<4}({len(a) / len(bf):>5.1%})'
              f'   後 {len(b):>4}/{len(new):<4}({len(b) / len(new):>5.1%})'
              f'   うち割った片 {len(p):>3}')
    row('首あり(細い首)', lambda r: r['neck'] is not None)
    row(f'枠線を飲んだ(帯の覆い>={BAND})',
        lambda r: r.get('band_cov') is not None and r['band_cov'] >= BAND)
    row(f'形(細い割合>={THIN})',
        lambda r: r.get('thin_frac') is not None and r['thin_frac'] >= THIN)
    row(f'墨(純黒<{PURE})',
        lambda r: r.get('pure_black') is not None and r['pure_black'] < PURE)
    for nm, f in (('band_cov', 'band_cov'), ('thin_frac', 'thin_frac'),
                  ('pure_black', 'pure_black')):
        na = sum(1 for r in bf if r.get(f) is None)
        nb = sum(1 for r in new if r.get(f) is None)
        if na or nb:
            print(f'  **測れなかった行**({nm}): 前 {na} / 後 {nb}(分母から外している)')
    print('\n**分離は軸を作らない。** 片は親より小さいので、'
          '形と墨の軸は値そのものが動く —— **前後を並べたのはそのため**')


if __name__ == '__main__':
    main()
