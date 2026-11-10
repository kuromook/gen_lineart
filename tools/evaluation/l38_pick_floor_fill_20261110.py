# -*- coding: utf-8 -*-
"""床の材料を足す聞き取りの標本(登録 `doc/l38_floor_fill_registration_20261110.md`)。

枠 = いまの単位(分ける前の表から片に割られた親を除く + 片)− 床の集まりに既に在る単位。
`default_rng(20261110)` で枠を並べ替え、**前から順に**取る。切り出しの器具が落とした分は
この順で補う(`--want` 件そろうまで。器具が落とした数は器具が印字する)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_pick_floor_fill_20261110.py --take 40
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_floor_collection_20261110 as F                           # noqa: E402
import l38_ruling_count_20261110 as C                               # noqa: E402

OUT = f'{C.ROOT}/data/l38_floorfill_20261110/order.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--take', type=int, default=40, help='並べ替えた順の先頭から何件書くか(補い用に余分を含む)')
    a = ap.parse_args()
    units, _ = F.collect()
    pre = json.load(open(C.PRE, encoding='utf-8'))
    sep = json.load(open(C.SEP, encoding='utf-8'))
    parents = {('face',) + tuple(r['parent']) for r in sep if r['origin'] == 'piece'}
    frame = []
    for r in pre:
        k = ('face', r['work'], r['pid'], r['comp'], r['region'], r['area'])
        if k not in parents and k not in units:
            frame.append(dict(work=r['work'], pid=r['pid'], comp=r['comp'], region=r['region'],
                              area=r['area'], origin='kept', parent=None, piece_lid=None))
    for r in sep:
        if r['origin'] != 'piece':
            continue
        k = ('piece', tuple(r['parent']), r['piece_lid'], r['area'])
        if k not in units:
            frame.append(dict(work=r['work'], pid=r['pid'], comp=r['comp'], region=r['region'],
                              area=r['area'], origin='piece', parent=r['parent'],
                              piece_lid=r['piece_lid']))
    rng = np.random.default_rng(20261110)
    order = [frame[i] for i in rng.permutation(len(frame))[:a.take]]
    for n, r in enumerate(order, 1):
        r['fid'] = f'u{n:03d}'
        r['rank'] = n
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(order, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'枠 {len(frame)}(床の上 {sum(r["area"] >= 6000 for r in frame)})-> 並べ替えの先頭 {len(order)} -> {OUT}')


if __name__ == '__main__':
    main()
