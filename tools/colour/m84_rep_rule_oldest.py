#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m84: **代表は「年代の古い方」。作者の裁定で向きが逆になった**(追補121)。

**作者の裁定(2026-10-28 夜)**:

> **ページが同バイト数なら年代の古いものを代表とするのはきめた**(`census` 経由で届いた言明)

**こちらが問い直して確認した**: 「バイト同一の複製で代表として残すのはどちら側か」
→ **「古い方(`census` の記録どおり)」**。

→ **追補110 §A の「年代が新しい方を残す」を撤回する。**
**`m80_apply_rep_rule.py` は撤回した器具として残す**(消さない。規律3: 撤回はパス)。

**先に測ってある数(走る前に書く。追補121 §B)**:
  年代を跨ぐ 1,489組のうち **両方が872頁の表の外 1,447 / 両方が中 42 / 片方だけ 0**
  → **どちら向きでも 872頁の表から落ちる頁は 0。** 代表の向きは**どのファイルを測るかを変える**

**門(落ちたら書き出さない)**:
  KC-1 畳んだ後の母集団 **7,418**(**畳み方は変えていない**)
  KC-2 **代表の札ごとの数が、`m83` の「時期の札」と全札で一致**
       —— **代表が最も古い札になるので、2つの列は同じ量になるはず**
       (`census` の「裁定で代表の札と時期の札が同じ量になりました」の独立確認)
  KC-3 **872頁の表から落ちる頁 0**
  KC-4 代表のパスが `m80`(新しい方)と違う組が **1,489**(向きが本当に逆になった証拠)

**A/B は確定させない(追補121 §D)**: 代表が古い方になると、**34組で生き残る行の `use` が
入れ替わる**(最大の内訳は「古い側 = 採点しない(置き換えられた)」/「新しい側 = 採点する」が
**27組**)。**その「置き換えられた」の札は作者の審査中である** ——
**27組のうち 19組は「b 付きが新しい」からこちらが推論して付けた札**で、
**作者はその推論を否定している**(「2019b が 2019 の新しいバージョンではない」)。
→ **両方の数を印字し、「審査中の札に乗っている」と明記する。**

  <venv>/python tools/colour/m84_rep_rule_oldest.py
"""
import collections
import csv
import pathlib
import re
import sys

ALL = pathlib.Path('/home/sh1/deepl/lineart/inbox/'
                   'request9_file_sha256_all_20261026.csv')
ROLES = pathlib.Path('results/m78_workbook_applied/version_roles_current.csv')
M80 = pathlib.Path('results/m80_rep_rule_newest/folded_population_newest.csv')
M83 = pathlib.Path('results/m83_period_label/folded_population_period.csv')
OUT = pathlib.Path('results/m84_rep_rule_oldest')
LAB = re.compile(r'[\\/](\d{4}[a-z]?)[\\/]')
A_PREFIX = '採点する'
B_EXTRA = '別行で採点'


def label(p):
    m = LAB.search(p)
    return m.group(1) if m else '?'


def key(y):
    """**時期の前後**(作者の裁定 `2024` < `2024b` < `2024c`)。**版の新旧ではない**。"""
    return (int(y[:4]), y[4:]) if y != '?' else (9999, '')


def cnt(uses):
    # **生成子を渡されても2回回せるように実体化する。**
    # 直前の走行は生成子を2回回し、**2回目が空になって B が A と同じ数で出た**
    # (**例外ではなく「追加が 0」として静かに出る**形。T-M17)
    us = list(uses)
    a = sum(1 for u in us if u.startswith(A_PREFIX))
    b = a + sum(1 for u in us if u.startswith(B_EXTRA))
    return a, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ALL, encoding='utf-8-sig'))
            if r['f_sha256'] and not r['f_read_error']]
    roles = {r['page_path']: r for r in
             csv.DictReader(open(ROLES, encoding='utf-8-sig'))}
    by = collections.defaultdict(list)
    for r in rows:
        by[r['f_sha256']].append(r)
    print(f'== 材料 ==\n  母集団 {len(rows)} / 札の表 {len(roles)} 頁')

    keep, drop, flipped, lost = [], [], 0, 0
    label_n = collections.Counter()
    m80keep = {r['page_path'] for r in
               csv.DictReader(open(M80, encoding='utf-8-sig'))}
    for v in by.values():
        # **代表 = 年代の古い方。同順は道の辞書順の先頭**(機械的な決め方)
        srt = sorted(v, key=lambda r: (key(label(r['page_path'])),
                                       r['page_path']))
        rep = srt[0]
        keep.append(rep)
        label_n[label(rep['page_path'])] += 1
        for r in srt[1:]:
            drop.append(dict(page_path=r['page_path'], 代表=rep['page_path'],
                             年代=label(r['page_path']),
                             代表の年代=label(rep['page_path']),
                             組の大きさ=len(v)))
        if len(v) > 1 and rep['page_path'] not in m80keep:
            flipped += 1
        # **872頁の表から落ちる頁**: 表に在る複製が、代表でないために落ちる
        in_tbl = [r for r in v if r['page_path'] in roles]
        if in_tbl and rep['page_path'] not in roles:
            lost += len(in_tbl)

    print('\n== 門 ==')
    ok = True
    print(f'  KC-1 畳んだ後の母集団 {len(keep):>5} / 期待 7418  '
          f'{"PASS" if len(keep) == 7418 else "**FAIL**"}')
    ok &= len(keep) == 7418

    per = {r['時期の札']: 0 for r in csv.DictReader(open(M83, encoding='utf-8-sig'))}
    per = collections.Counter(r['時期の札'] for r in
                              csv.DictReader(open(M83, encoding='utf-8-sig')))
    same = all(label_n[k] == per[k] for k in set(label_n) | set(per))
    print(f'  KC-2 代表の札ごとの数 == `m83` の時期の札  '
          f'{"PASS(2つの列が同じ量になった)" if same else "**FAIL**"}')
    if not same:
        for k in sorted(set(label_n) | set(per), key=key):
            if label_n[k] != per[k]:
                print(f'       {k}: 代表 {label_n[k]} / 時期 {per[k]}')
    ok &= same
    print(f'  KC-3 872頁の表から落ちる頁 {lost:>5}  '
          f'{"PASS" if lost == 0 else "**FAIL**"}')
    ok &= lost == 0
    print(f'  KC-4 代表が `m80`(新しい方)と違う組 {flipped:>5} / 期待 1489  '
          f'{"PASS(向きが逆になった)" if flipped == 1489 else "**FAIL**"}')
    ok &= flipped == 1489
    print(f'\n  **門 {"すべて PASS" if ok else "**FAIL**。書き出さない"}**')
    if not ok:
        return 1

    kept = {r['page_path'] for r in keep}
    folded_roles = [r for p, r in roles.items() if p in kept]
    a_old, b_old = cnt(r['use'] for r in roles.values())
    a_new, b_new = cnt(r['use'] for r in folded_roles)
    m80_roles = [r for p, r in roles.items() if p in m80keep]
    a_m80, b_m80 = cnt(r['use'] for r in m80_roles)

    print('\n== 札の表(**A/B は確定させない。§D の理由**)==')
    print(f'  {"":34s} {"行":>6} {"A":>6} {"B":>6}')
    print(f'  {"畳む前":34s} {len(roles):>6} {a_old:>6} {b_old:>6}')
    print(f'  {"畳んだ後(撤回した規則=新しい方)":34s} '
          f'{len(m80_roles):>6} {a_m80:>6} {b_m80:>6}')
    print(f'  {"**畳んだ後(作者の裁定=古い方)**":34s} '
          f'{len(folded_roles):>6} **{a_new}** **{b_new}**')
    print(f'\n  **A の差 {a_new - a_m80:+d} / B の差 {b_new - b_m80:+d}**')
    print('  **この差は「採点しない(置き換えられた)」の札に乗っている。**')
    print('  **その札は作者の審査中である**(27組。うち19組は「b 付きが新しい」からの推論で、')
    print('  **作者はその推論を否定している**)。**(b) の答が出るまで A/B を確定させない。**')

    with open(OUT / 'folded_population_oldest.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=['page_path', 'format', 'f_sha256',
                                           'f_bytes', '年代'])
        w.writeheader()
        for r in sorted(keep, key=lambda r: r['page_path']):
            w.writerow({'page_path': r['page_path'], 'format': r['format'],
                        'f_sha256': r['f_sha256'], 'f_bytes': r['f_bytes'],
                        '年代': label(r['page_path'])})
    with open(OUT / 'dropped_oldest.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(drop[0]))
        w.writeheader(); w.writerows(sorted(drop, key=lambda r: r['page_path']))
    with open(OUT / 'version_roles_folded_oldest.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(folded_roles[0]))
        w.writeheader()
        w.writerows(sorted(folded_roles, key=lambda r: r['page_path']))
    print(f'\nwrote {OUT}/ (3 ファイル)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
