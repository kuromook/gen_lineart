#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補99 —— 祖先フォルダーの表示を、こちらの計算が見ていないことの大きさ。**

基盤の T143(抽出側の器具の決定3): **非表示フォルダーの中身は描かれないので、祖先を見るのが
正しい。** こちらの `m8`(クリップの実効被覆)も `m72`(下地の分布)も**表示を層自身の
1の位しか見ていない**(標準 `m11`/`m16` は `visibility` を一度も読んでいない)。

**測るだけ。腕も棒も無い。** 画素なし、構造の表だけ。

  <venv>/python tools/colour/m73_hidden_ancestor_effect.py
"""
import collections
import csv
import json
import pathlib
import sys

SRC = pathlib.Path('inbox/m_colour_page_layers_872.csv')
PAGES = pathlib.Path('results/alias/occluder_check_pages.json')   # **道は追跡外**


def truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'none')


def main():
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    print(f'材料 {SRC}  行 {len(rows)}  頁 {len({r["page_path"] for r in rows})}')

    by_page = collections.defaultdict(dict)
    for r in rows:
        by_page[r['page_path']][r['layer_id']] = r

    def hidden_ancestor(page, r):
        """祖先のフォルダーに1つでも非表示(1の位が立たない)が在るか。"""
        seen = set()
        pid = r['parent_id']
        while pid and pid not in seen:
            seen.add(pid)
            p = by_page[page].get(pid)
            if p is None:
                return None                      # 親が表に無い(数える)
            if not (int(p['visibility'] or 0) & 1):
                return True
            pid = p['parent_id']
        return False

    cat = collections.Counter()
    clip_cat = collections.Counter()
    for page, layers in by_page.items():
        for r in layers.values():
            if truthy(r['folder']):
                continue
            self_vis = bool(int(r['visibility'] or 0) & 1)
            ha = hidden_ancestor(page, r)
            if ha is None:
                k = '親が表に無い'
            elif not self_vis:
                k = '自身が非表示'
            elif ha:
                k = '**自身は表示・祖先が非表示**'
            else:
                k = '表示(祖先まで)'
            cat[k] += 1
            if int(r['clip'] or 0) == 1:
                clip_cat[k] += 1

    n = sum(cat.values())
    print(f'\n== 非フォルダー {n} 層(理由別・残り 0)==')
    for k, v in cat.most_common():
        print(f'  {k:28s} {v:>7}  {v / n:.3%}')
    print(f'  {"合計":28s} {n:>7}   **残り {n - sum(cat.values())}**')

    m = sum(clip_cat.values())
    print(f'\n== そのうち `clip=1` {m} 層 ==')
    for k, v in clip_cat.most_common():
        print(f'  {k:28s} {v:>7}  {v / m:.3%}')

    # 既知セル8頁(標準 183/823/801 の母集団)
    if PAGES.exists():
        known = json.loads(PAGES.read_text(encoding='utf-8'))['known8']
        kc = collections.Counter()
        for page in known:
            for r in by_page.get(page, {}).values():
                if truthy(r['folder']):
                    continue
                self_vis = bool(int(r['visibility'] or 0) & 1)
                ha = hidden_ancestor(page, r)
                kc['自身が非表示' if not self_vis else
                   ('**祖先が非表示**' if ha else '表示')] += 1
        print(f'\n== 既知セル {len(known)} 頁(標準 183/823/801 の母集団)==')
        for k, v in kc.most_common():
            print(f'  {k:28s} {v:>7}')
        print('  **標準は `visibility` を一度も読んでいない**(`m11`/`m16`)')
    else:
        print(f'\n(既知セルの頁の表が無い: {PAGES}。既知セルは数えない)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
