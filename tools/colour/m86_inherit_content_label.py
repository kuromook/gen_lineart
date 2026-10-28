#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m86: **バイト同一の組では「置き換えられた」は成立しない。代表が内容の札を継ぐ**(追補123)。

**作者の裁定(2026-10-28)**: 34組の一覧を見た上で **「案1: 代表が内容の札を継ぐ」**。

**理由(登録に書いてから当てる)**:

> **34組は全部バイト同一である。**
> 「**置き換えられた**」「**改訂の古い側**」という札は、
> **どちらかが他方を置き換えた**という関係を主張している。
> **しかしバイト列が完全に同一なら、版の差は存在しない** —— 同じファイルが2つの
> フォルダに在るだけで、**置き換える対象が無い。**
> → **成立しない関係を主張していた札を取り下げ、残った唯一の内容判断を代表に引き継ぐ。**
> **札を甘くするのではない。** 内容判断は Excel で作者が記入した側から来ており、
> **絵は同一なので代表がどちらでもその判断は有効である。**

**27組のうち19組は「b 付きが新しい」からこちらが推論して付けた札**で、
**作者はその推論を明示的に否定している**(「**2019b が 2019 の新しいバージョンではない**」/
「**b,c はバージョンではなく時期を表している。2024c は終わり、2024b は中盤**」)。
**残り15組も、バイト同一という同じ理由で成立しない**(改訂の差も無い)。

**操作**: 代表(= **年代の古い方**。作者の裁定。`m84`)の札が**置き換え系**なら、
**同じ組の最も新しい複製の札を継ぐ**。**元の札は捨てずに列に残す。**

**継ぐ先の決め方(最初の走行で直した。経緯を残す)**:
  **最初は「組の最も新しい複製の札を継ぐ」と書いて走らせ、既知セルが落ちた** ——
  **継ぐ先が食い違う組が 7**。中を見ると**7組すべて3枚組**で、
  候補は「置き換え系の札」と「内容の札」が**1つずつ**だった。
  → **曖昧ではなく、規則の書き方が雑だった。**
  **正しい形は「置き換え系でない唯一の札から継ぐ」**(= 内容の札。Excel 由来)。
  **「最も新しい」は順序の話で、内容の話ではない** —— **作者が否定した軸を、
  継ぐ先の決め方にもう一度持ち込んでいた**(T-M13 の形)。

**既知セル**:
  KC-1 畳んだ札の表は **800行**(**行数は変わらない。札が変わるだけ**)
  KC-2 **内容の札の候補が1つに決まらない組は 0**(決まらなければ継げない。止める)
  **A/B は事前登録した値ではない。** 親の走行が出した **32行 / A 652 / B 732** は
  **1組に2枚しか無い場合を前提にした近似**で、**3枚組 7つを落としていた。**
  **近似の値を既知セルにしない**(それは結果を見てから棒を置く形になる)。
  **A/B は結果として印字し、近似との差も印字する。**

  <venv>/python tools/colour/m86_inherit_content_label.py
"""
import collections
import csv
import pathlib
import re
import sys

ALL = pathlib.Path('/home/sh1/deepl/lineart/inbox/'
                   'request9_file_sha256_all_20261026.csv')
ROLES = pathlib.Path('results/m78_workbook_applied/version_roles_current.csv')
OUT = pathlib.Path('results/m86_inherit_content_label')
LAB = re.compile(r'[\\/](\d{4}[a-z]?)[\\/]')
# **置き換え系の札** —— バイト同一の組では成立しない(追補123)
SUPERSEDED = ('採点しない(置き換えられた',
              '採点しない(改訂の古い側',
              '採点しない(同じ改訂が別の道')
EXPECT = {'rows': 800}
APPROX = {'changed': 32, 'a': 652, 'b': 732}   # **親の近似。既知セルではない**


def label(p):
    m = LAB.search(p)
    return m.group(1) if m else '?'


def key(y):
    return (int(y[:4]), y[4:]) if y != '?' else (9999, '')


def cnt(uses):
    us = list(uses)                      # **生成子を2回回さない**(T-M17)
    a = sum(1 for u in us if u.startswith('採点する'))
    return a, a + sum(1 for u in us if u.startswith('別行で採点'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ALL, encoding='utf-8-sig'))
            if r['f_sha256'] and not r['f_read_error']]
    roles = {r['page_path']: dict(r) for r in
             csv.DictReader(open(ROLES, encoding='utf-8-sig'))}
    by = collections.defaultdict(list)
    for r in rows:
        by[r['f_sha256']].append(r)

    folded, changed, why = {}, [], collections.Counter()
    conflict = 0
    for v in by.values():
        srt = sorted(v, key=lambda r: (key(label(r['page_path'])),
                                       r['page_path']))
        rep = srt[0]['page_path']
        if rep not in roles:
            continue                     # 札の表の外(規律9: 下で理由別に数える)
        row = dict(roles[rep])
        row['継承前の札'] = ''
        row['継承の出所'] = ''
        if len(v) > 1 and row['use'].startswith(SUPERSEDED):
            # **継ぐ先は「置き換え系でない唯一の札」= 内容の札**(Excel 由来)。
            # **「最も新しい」では決めない** —— 順序は内容の話ではなく、
            # **作者が否定した軸である**(追補117)
            src = [r for r in srt[1:] if r['page_path'] in roles
                   and not roles[r['page_path']]['use'].startswith(SUPERSEDED)]
            cand = {roles[s['page_path']]['use'] for s in src}
            if len(cand) == 1:
                row['継承前の札'] = row['use']
                row['継承の出所'] = src[0]['page_path']
                row['use'] = cand.pop()
                row['decided_by'] = (
                    f'作者の裁定(案1。バイト同一に置き換えは無い)'
                    f' / 元: {roles[rep]["decided_by"]}')
                changed.append(row)
                why[(row['継承前の札'], row['use'])] += 1
            elif len(cand) > 1:
                conflict += 1
                why[('**内容の札が複数。継がない**', row['use'])] += 1
            else:
                why[('**内容の札が無い。継がない**', row['use'])] += 1
        folded[rep] = row

    print(f'== 材料 ==\n  母集団 {len(rows)} / 札の表 {len(roles)} 頁 '
          f'/ 畳んだ札の表 {len(folded)} 行')
    a0, b0 = cnt(r['use'] for r in
                 (dict(roles[p]) for p in folded))          # 継承前
    a1, b1 = cnt(r['use'] for r in folded.values())         # 継承後

    print('\n== 既知セル ==')
    ok = len(folded) == EXPECT['rows']
    print(f'  KC-1 畳んだ札の表の行    {len(folded):>5} / 期待 '
          f'{EXPECT["rows"]:>5}  {"PASS" if ok else "**FAIL**"}')
    print(f'  KC-2 内容の札が1つに決まらない組 {conflict:>5} / 期待     0  '
          f'{"PASS" if conflict == 0 else "**FAIL**"}')
    ok &= conflict == 0
    print(f'\n  **既知セル {"すべて PASS" if ok else "**FAIL**。書き出さない"}**')
    if not ok:
        return 1

    print('\n== 札が変わった行(理由別。規律9)==')
    for (a_, b_), n in why.most_common():
        print(f'  {n:>3}  「{a_[:34]}」 -> 「{b_[:34]}」')
    print(f'  合計 {sum(why.values())}')

    print('\n== A / B(**結果。事前登録した値ではない**)==')
    print(f'  継承前(`m84` の畳んだ表)   A {a0} / B {b0}')
    print(f'  **継承後**                 **A {a1} / B {b1}**  '
          f'(A {a1-a0:+d} / B {b1-b0:+d})')
    print(f'  親の近似(2枚組の前提)     A {APPROX["a"]} / B {APPROX["b"]} '
          f'/ 札が変わる行 {APPROX["changed"]}  '
          f'-> **差 A {a1-APPROX["a"]:+d} / B {b1-APPROX["b"]:+d} '
          f'/ 行 {len(changed)-APPROX["changed"]:+d}**')
    print('  **872頁の表の行は1行も増減していない。札だけが変わった。**')

    f = OUT / 'version_roles_folded_inherited.csv'
    with open(f, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(folded.values()))))
        w.writeheader()
        w.writerows(sorted(folded.values(), key=lambda r: r['page_path']))
    f2 = OUT / 'inherited_rows.csv'
    with open(f2, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(changed[0]))
        w.writeheader()
        w.writerows(sorted(changed, key=lambda r: r['page_path']))
    print(f'\nwrote {f}\nwrote {f2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
