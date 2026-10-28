#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m82: **基盤の T146 をこちらで引き受ける** —— 「下地が非表示」を
**Track M の標準が歩く母集団**で引き直す(追補113)。

**基盤の指摘(2026-10-28 `reply_m_two_general_forms_20261028.md`)**:
> 2,485 は撤回しない(全非フォルダークリップ層では正しい)。**そちらの 13.41%(84倍)も
> 同じ扱い —— そちらの器具が通る母集団で引き直すと数が変わる可能性があります。
> 確かめるのはそちらの判断です(こちらの数で上書きしないでください)**

**測るもの**: `standard_visible`(作者が 2026-10-28 に選んだ標準)が歩く層のうち、
**クリップ層は残るのに下地が落ちる**組。**これだけが標準の面を変えうる。**
**下地の同定は `m16.tree_bases` と同じ規則**(同じ親の兄弟・`order_in_parent` 昇順)。

**帰無値(走る前に書いた)**: **祖先の経路は 0 になる。**
下地とクリップ層は**同じ親の兄弟なので祖先を共有する** ——
祖先が非表示なら**クリップ層自身も落ちている**ので、標準はその下地を一度も見ない。
**0 でなければこの推論が誤っている。**

**棒は置かない。診断である**(追補100 と同じ)。
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m77_standard_visible as m77                     # noqa: E402

CSV = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
PAPER = '1584'


def main():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    vis = lambda r: bool(int(r['visibility'] or 0) & 1)   # noqa: E731

    t = collections.Counter()
    for rs in byp.values():
        keep, _ = m77.visible_rows(rs)
        kept = {r['layer_id'] for r in keep}
        sib = collections.defaultdict(list)
        for r in rs:
            if r['layer_type'] == PAPER or not r['parent_id']:
                continue
            sib[r['parent_id']].append(r)
        for g in sib.values():
            g.sort(key=lambda r: int(r['order_in_parent']))
            base = None
            for r in g:
                if r['clip'] == '0':
                    base = r
                    continue
                if base is None:
                    t['下地が無い'] += 1
                    continue
                t['クリップ層(全部)'] += 1
                if r['layer_id'] not in kept:
                    t['クリップ層自身が標準から落ちる'] += 1
                elif base['layer_id'] not in kept:
                    t['**標準に残るのに下地が落ちる**'] += 1
                    t['  下地自身が非表示' if not vis(base)
                      else '  下地の祖先が非表示'] += 1
                else:
                    t['両方残る'] += 1

    print(f'== 母集団: {CSV.name}(872頁。**Track M の標準が歩く表**)==')
    for k, v in t.most_common():
        print(f'  {k:36s} {v:>7}')
    n, h = t['クリップ層(全部)'], t['**標準に残るのに下地が落ちる**']
    print(f'\n  **標準に効く割合 {h}/{n} = {100 * h / n:.4f}%**')
    anc = t['  下地の祖先が非表示']
    print(f'  **帰無値「祖先の経路は 0」: 実測 {anc} -> '
          f'{"守られた" if anc == 0 else "**破れた。推論が誤り**"}**')
    print('\n  追補100 の 13.41%(1,578/11,763)は**撤回しない** —— '
          'あれは**非フォルダー clip=1 の全体**の上の数である。')
    print('  **標準が歩く母集団では、効くのは「下地自身が非表示」の経路だけ**。')
    return 0 if anc == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
