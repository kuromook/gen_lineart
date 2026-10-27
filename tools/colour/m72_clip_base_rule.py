#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補97 の材料 —— 基盤のクリップ下地規則を、色原稿872頁で独立に走らせる。**

**測るだけ。腕も棒も無い**(基盤の規則の再現と、こちらの m8 の欠陥の大きさ)。
規則(基盤 §1): 同じ親の中で `order_in_parent` 昇順(0 = 下)、
**その層より下で最も近い `clip=0` の兄弟**を下地とする。**フォルダーも兄弟に数える。**
"""
import collections
import csv
import pathlib
import sys

SRC = pathlib.Path('inbox/m_colour_page_layers_872.csv')


def truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'none')


def main():
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    print(f'材料 {SRC}  行 {len(rows)}  頁 {len({r["page_path"] for r in rows})}')
    folder = sum(1 for r in rows if truthy(r['folder']))
    print(f'  フォルダー {folder} / 非フォルダー {len(rows) - folder}')

    sib = collections.defaultdict(list)
    for r in rows:
        sib[(r['page_path'], r['parent_id'])].append(r)
    empty = [r for r in rows if not str(r['order_in_parent']).strip()]
    print(f'  `order_in_parent` が空 {len(empty)} '
          f'(うちフォルダー {sum(1 for r in empty if truthy(r["folder"]))}、'
          f'`clip=1` {sum(1 for r in empty if int(r["clip"] or 0) == 1)}) '
          f'—— **頁の根のフォルダー1枚ずつ。-1 に置く**')
    for v in sib.values():
        v.sort(key=lambda r: int(r['order_in_parent'] or -1))

    def base_of(group, i, skip_folders):
        for j in range(i - 1, -1, -1):
            b = group[j]
            if skip_folders and truthy(b['folder']):
                continue
            if int(b['clip'] or 0) == 0:
                return b
        return None

    cat = collections.Counter()
    moved = collections.Counter()
    consec = 0
    kind = collections.Counter()
    for group in sib.values():
        for i, r in enumerate(group):
            if truthy(r['folder']) or int(r['clip'] or 0) != 1:
                continue
            if i > 0 and int(group[i - 1]['clip'] or 0) == 1:
                consec += 1
            b = base_of(group, i, skip_folders=False)
            if b is not None:
                kind[('フォルダー ' if truthy(b['folder']) else '')
                     + b['layer_type']] += 1
            if b is None:
                cat['下地が無い'] += 1
            elif truthy(b['folder']):
                cat['下地がフォルダー'] += 1
            elif not (int(b['visibility'] or 0) & 1):
                cat['下地が非表示'] += 1
            else:
                cat['下地が在る(可視・非フォルダー)'] += 1
            # m8 の形: フォルダーを兄弟から落とすと下地はどう動くか
            b2 = base_of(group, i, skip_folders=True)
            if (b is None) != (b2 is None):
                moved['在無が変わる'] += 1
            elif b is not None and b['layer_id'] != b2['layer_id']:
                moved['別の層に繰り上がる'] += 1
            else:
                moved['同じ'] += 1

    n = sum(cat.values())
    print(f'\n== 非フォルダーの `clip=1` {n} 層を、基盤の規則で分ける(理由別・残り 0)==')
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f'  {k:30s} {v:>7}  {v / n:.3%}')
    print(f'  {"合計":30s} {n:>7}   **残り {n - sum(cat.values())}**')

    print(f'  連続クリップ(すぐ下の兄弟も `clip=1`) {consec}  {consec / n:.3%}'
          '  —— **上の4行と重なる。別の軸**')
    print('\n== 下地の `layer_type`(行ごと)==')
    for k, v in kind.most_common():
        print(f'  {k:30s} {v:>7}')

    print('\n== フォルダーを兄弟から落とすと(= m8 の母集団の形)==')
    for k, v in sorted(moved.items(), key=lambda kv: -kv[1]):
        print(f'  {k:30s} {v:>7}  {v / n:.3%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
