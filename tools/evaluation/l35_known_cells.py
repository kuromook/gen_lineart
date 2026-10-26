# -*- coding: utf-8 -*-
"""L35 の既知セル。**本体の前に回す。1つでも落ちたら回さない**(規律2)。

登録は `doc/work_log.md` の 2026-10-26 15:07 の節。

**KC41 は登録した形では走れない。差し替える(数を1つも見る前に)。**

登録にはこう書いた:

> **KC41(データ経路)**: L19 で判定済みの551面の名前が、新しい標準の名前に
> 全部在るか。**欠け0でなければ本体を読まない**

**これは走れない。** 面の名前は `f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'` で、
`gi` は**そのコマの中の成分の番号**である。標準を変えれば成分が変わるので
`gi` が変わり、**名前は原理的に一致しない。** 「欠け0」を要求すると
**必ず落ちる点検**になる —— T102(走れない点検)の鏡像で、
**通る道が無い点検**である。登録のときに考え切れていなかった。

**差し替える形**(棒ではないので動かしてよいが、**理由と時点を書く**。規律3):

> **KC41'(データ経路。包含で見る)**: 生の墨の和集合は、開いた和集合の**上位集合**
> である(`fill_mask` の出力は常に `ink` の部分集合だから)。だから
> **古い標準の面の画素は、新しい標準の面にほぼ全部入る**はずで、入らない分は
> `MINREG` とコマ面積の帯が落とした分だけである。
> **被覆 >= 0.95 を要求する。** これなら向きと粒度が名前に書けていて、
> 配線を間違えれば(別の層集合を渡す等)落ちる。

KC42 / KC43 は登録のまま。
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_arms import acc_add
from l14_known_cells import A, MINREG, comps, gated_layers
from l17_known_cells import (MAX_FRAC, MIN_FRAC, beta_fill_layers, jobs,
                             panels_and_regions, zmap)

DEL = '/home/sh1/deepl/lineart/data/incoming_manuscript'


def kc42():
    """**標準が実際に変わったことの確認。** 幅6px の橋で繋がった2塊を合成し、
    **既定('solid')では2面に割れ、L35('cover')では1面のまま**であること。

    **両方で同じなら、opening を外した効果が器具に入っていない**(T102)。
    """
    im = np.zeros((300, 600), bool)
    im[80:220, 40:260] = True          # 塊1
    im[80:220, 340:560] = True         # 塊2
    im[147:153, 260:340] = True        # 幅6px の橋
    out = {}
    for which in ('solid', 'cover'):
        acc = np.zeros(im.shape, bool)
        acc_add(acc, None, im, which)
        _, gids, _ = comps(acc, 8)
        out[which] = len(gids)
    ok = out['solid'] == 2 and out['cover'] == 1
    print(f"KC42 幅6px の橋: 既定(solid) {out['solid']} 面 (==2) / "
          f"L35(cover) {out['cover']} 面 (==1)  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print('     **両方が同じなら、標準の切り替えが器具に入っていない**')
    return ok


def kc43(limit=0):
    """**選別が空でないことの確認。** 納品 beta が1層も無い頁・コマの数を印字する。
    **0 件なら「この点検は空」と印字する**(登録 §7)。"""
    # **【印 2026-10-26 T131】この既知セルは、本体と違う母集団で回っている。**
    # `gated_layers()` の腕は **1,681** だが、本体(`--raw-beta-standard`)が
    # 使うのは **1,392**(両側に居る289層を外した後)。
    # **既知セルが本体の母集団を錨にしていない** —— 規律2 は
    # 「既知セルが再現しなければ本体を読まない」だが、**別の母集団の上で再現しても錨にならない。**
    # **移すと出力が変わるので、登録どおり移さずに印を付ける**
    # (「再現しない呼び手は移さず印を付ける」)。**直すのは別の走行。**
    zz = zmap()
    _, line = gated_layers()
    fill = beta_fill_layers()
    js = jobs(zz, fill, line)
    if limit:
        js = js[:limit]
    allp = set()
    for work in ('ako6', 'ako7'):
        d = json.load(open(f'{DEL}/{work}/koma/koma_manifest.json'))
        for p in d['pages']:
            allp.add((p['page'], work))
    have = {(pid, work) for pid, work, _, _, _ in js}
    miss = allp - have
    print(f'KC43 コマ一覧の頁 {len(allp)}  納品 beta から標準が作れる頁 {len(have)}  '
          f'**作れない頁 {len(miss)}**')
    if not miss:
        print('     **この点検は空**(作れない頁が0件。'
              '母集団の縮退はこの軸では起きていない)')
    else:
        for pid, work in sorted(miss)[:8]:
            print(f'       {work} {pid}')
    return True            # 印字が目的の点検。落とさない


def kc44():
    """**KC44(漏れの点検。2026-10-26 に KC41' が落ちて足した)**:
    **標準を作る層が、腕の入力にも居てはならない。**

    居れば、腕は**面を定義しているその層から面を予測する**ことになる。
    こちらの分類では **289層**が「作者の納品 beta なのに `line` 側」に居たので、
    標準を納品 beta に替えると**その289層が両側に居る**。
    **登録にこの点検は無かった。** 無いまま回していたら、
    汚染された `cov_honest` を読んでいた。
    """
    # **【T131】ここは `arms('beta_standard')` にしてはいけない。**
    # **この点検の目的は、外される前の重なりを数えることである。**
    # `beta_standard` は**その289層を外した後**を返すので、**0 になって点検が死ぬ。**
    # **生の腕が要る点検が在る** —— だから `arms('gated')` と使い分ける意味が在る。
    _, line = gated_layers()
    bfill = beta_fill_layers()
    b = {(k, lid) for k, v in bfill.items() for lid, _ in v}
    l = {(k, lid) for k, v in line.items() for lid, _ in v}
    both = b & l
    print(f'KC44 両側に居る層: **{len(both)}** '
          f'(標準 {len(b)} / 腕の入力 {len(l)})')
    print('     -> 本体は `--raw-beta-standard` でこの層を**腕から外す**。'
          'その結果 **腕の母集団が変わる** ので、'
          '**L35 の腕は L18 の腕と同じものではない**')
    return True            # 印字と、本体側で外すことが目的


def kc41(npages=2):
    """**KC41''(2回目の差し替え)**: **層集合を固定して、opening だけを外す。**

    1回目の差し替え(包含 >= 0.95)は **0.8465 で落ちた**。理由は配線ではなく、
    **比べたものが2つ同時に変わっていたから**である:

      変更A **層集合**  こちらの分類 300層 -> 作者の納品 496層
        (両方に在る 205 / **こちらだけ 95** / **作者だけ 291、うち289は `line` 側に居た**)
      変更B **墨の作り方**  半径4の opening -> 生の墨

    **「生の墨は開いた墨の上位集合」が成り立つのは、同じ層集合の上だけ**である。
    1回目はそこに変更Aを混ぜていた。**T110 と同じ形を、既知セルの中で踏んだ。**

    **だから層集合を納品 beta に固定して、変更Bだけを見る。被覆 >= 0.95 を要求する。**
    **変更Aの大きさは棒ではなく報告に出す**(消さない。規律7)。
    """
    # **【印 2026-10-26 T131】この既知セルは、本体と違う母集団で回っている。**
    # `gated_layers()` の腕は **1,681** だが、本体(`--raw-beta-standard`)が
    # 使うのは **1,392**(両側に居る289層を外した後)。
    # **既知セルが本体の母集団を錨にしていない** —— 規律2 は
    # 「既知セルが再現しなければ本体を読まない」だが、**別の母集団の上で再現しても錨にならない。**
    # **移すと出力が変わるので、登録どおり移さずに印を付ける**
    # (「再現しない呼び手は移さず印を付ける」)。**直すのは別の走行。**
    zz = zmap()
    _, line = gated_layers()
    bfill = beta_fill_layers()
    keys = [k for k in sorted(bfill)][:npages]
    if not keys:
        print('KC41\' 回せない: 両方の標準が作れる頁が無い')
        return False
    tot_old = tot_in = 0
    for pid, work in keys:
        jo = jobs(zz, bfill, line)
        jn = jobs(zz, bfill, line)
        jo = [j for j in jo if (j[0], j[1]) == (pid, work)]
        jn = [j for j in jn if (j[0], j[1]) == (pid, work)]
        if not jo or not jn:
            continue
        # 新しい標準の面を、コマ番号ごとに画素の集合として持つ
        new = {}
        for p, w, ci, P, glab, gids, gstat, _ in panels_and_regions(jn[0], 'cover'):
            m = np.zeros(glab.shape, bool)
            for gi in gids:
                if gstat[gi, A] >= MINREG:
                    m |= (glab == gi)
            new[(ci, glab.shape)] = m
        for p, w, ci, P, glab, gids, gstat, _ in panels_and_regions(jo[0], 'solid'):
            old = np.zeros(glab.shape, bool)
            for gi in gids:
                if gstat[gi, A] >= MINREG:
                    old |= (glab == gi)
            nm = new.get((ci, glab.shape))
            if nm is None:
                tot_old += int(old.sum())
                continue
            tot_old += int(old.sum())
            tot_in += int((old & nm).sum())
    cov = tot_in / tot_old if tot_old else 0.0
    ok = cov >= 0.95
    print(f"KC41'' 開いた標準の面の画素 {tot_old:,} のうち生の標準の面に入る "
          f"{tot_in:,} = **{cov:.4f}** (>=0.95)  {'PASS' if ok else 'FAIL'}  "
          f"({len(keys)} 頁)")
    print('      (**層集合は納品 beta に固定。変更Bだけを見ている。**'
          ' 登録の KC41 は走れない形で、1回目の差し替えは変更A と B を混ぜていた)')
    return ok


def main():
    print('L35 既知セル (登録 2026-10-26 15:07。**1つでも落ちたら本体を回さない**)')
    ok = True
    ok &= kc42()
    ok &= kc41()
    kc43()
    kc44()
    print()
    print('**既知セル ' + ('すべて PASS。本体を回してよい**' if ok
                           else 'が落ちた。本体を回さない**'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
