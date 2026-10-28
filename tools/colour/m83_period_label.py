#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m83: **時期の札を、代表のパスと別の列にする**(追補117)。

**作者の説明(2026-10-28)**:

> **2019 は 2019年の前半頃のフォルダであり 2019b は後半のほう。
> 2019b が 2019 の「新しいバージョンではない」。
> 新しい・古いという言葉で聞いているが解釈を間違わないよう**

> **sketch というフォルダはかなり長い期間にわたって継続してコピーして使われている。
> 中のページ単位では世代として新しい・古いが混在する。
> ページファイル単位でみたとき、バイト一致なら最初につくられた時期のものとみなすのが筋か**

**年代札は制作の時期を表し、版の関係は表していない。**
m80 は1つの欄で両方を兼ねていた。**2つの列に分ける**:

| 列 | 決め方 | 出所 |
|---|---|---|
| **代表のパス**(どのファイルを残すか) | **年代が新しい方** | 作者 2026-10-28。**変えない** |
| **時期の札**(どの期間に数えるか) | **最も早い札** | 作者 2026-10-28(今回) |

**ファイル側に制作時刻は無い**(追補117 §A で測った): `f_mtime` は 9,653 のうち
**7,426 が 2022年**、2020年より前が **0**(フォルダは 2012 まで遡る)、
**73.0% が札の年と不一致**、**バイト同一 1,569組のうち 956組は組の中で mtime が複数**。
→ **mtime はコピーの時刻。札が唯一の証拠。**

**既に見ている数**(走る前に書く。診断は追補117 §B で先に走らせた):
  畳んだ組 **7,418** / 札が1つに決まらない組 **1,489**
  早い札にすると `2015` 333→626 / `2014` 319→590 / `2024c` 550→237 など

**門(落ちたら書き出さない)**:
  KC-1 母集団は **7,418** のまま(**畳み方は変えていない**)
  KC-2 **代表のパスは m80 の出力と 1行も違わない**(**作者の前の裁定を動かしていない証拠**)
  KC-3 時期の札が代表の札と違う行が **0 でない**(0 なら列を足した意味が無い = 器具を疑う)

**限界(全ての行に付ける)**: **札はどのページについても「それ以降ではない」という上限**で、
**早い札に付け替えられるのは、同じバイト列が早いフォルダにも在るページだけ。**
**1回しか現れないページは、使い回されていても補正されない** ——
**補正は不均等で、不均等自身が傾向を作り得る。**
→ **この列の名前は「制作時期」ではなく「最も早く現れた札」である。**

  <venv>/python tools/colour/m83_period_label.py
"""
import collections
import csv
import pathlib
import re
import sys

ALL = pathlib.Path('/home/sh1/deepl/lineart/inbox/'
                   'request9_file_sha256_all_20261026.csv')
M80 = pathlib.Path('results/m80_rep_rule_newest/folded_population_newest.csv')
OUT = pathlib.Path('results/m83_period_label')
EXPECT = {'folded': 7418, 'split_groups': 1489}
LAB = re.compile(r'[\\/](\d{4}[a-z]?)[\\/]')


def label(p):
    m = LAB.search(p)
    return m.group(1) if m else '?'


def key(y):
    """**時期の前後**(作者の説明: `2019` が前半、`2019b` が後半)。**版の新旧ではない**。

    **接尾辞の綴り順は作者の裁定である**(2026-10-28、`census` 経由で確認。2段階):
      (a) 「**2024より2024c のが新しい 2022より2022b のが新しい 2019より2019b のが新しい**」
      (b) `2024b` —— `census` が「**綴りの規則を延ばして 2024 < 2024b < 2024c と置いた数です**」
          と断った上で作者が「**それでOK**」。**綴り順そのものが承認された**
    → **ここは「Claude が延ばした読み」ではない。** 追補120。
    **正本は `census` の列**(原稿の情報は国勢調査の領分。ユーザーの指示 2026-10-28)。
    """
    return (int(y[:4]), y[4:]) if y != '?' else (9999, '')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ALL, encoding='utf-8-sig'))
            if r['f_sha256'] and not r['f_read_error']]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['f_sha256']].append(r)
    print(f'== 材料 ==\n  {ALL}\n  母集団 {len(rows)} / バイト同一の組 '
          f'{sum(1 for v in by.values() if len(v) > 1)}')

    out, split = [], 0
    late_n, early_n = collections.Counter(), collections.Counter()
    for h, v in by.items():
        labs = sorted({label(r['page_path']) for r in v}, key=key)
        # **代表 = 年代が新しい方。同順は道の辞書順**(m80 と同じ。変えない)
        rep = sorted(v, key=lambda r: ([-x for x in key(label(r['page_path']))[:1]],
                                       [chr(255 - ord(c)) for c in
                                        key(label(r['page_path']))[1] or ' '],
                                       r['page_path']))[0]
        period = labs[0]                      # **時期の札 = 最も早い札**
        late_n[labs[-1]] += 1
        early_n[period] += 1
        if labs[0] != labs[-1]:
            split += 1
        out.append(dict(page_path=rep['page_path'], format=rep['format'],
                        f_sha256=h, f_bytes=rep['f_bytes'],
                        代表の札=label(rep['page_path']),
                        時期の札=period,
                        組の大きさ=len(v),
                        札の数=len(labs),
                        札=' '.join(labs)))

    print('\n== 門(既知セル)==')
    ok = True
    print(f'  KC-1 畳んだ後の母集団   {len(out):>5} / 期待 {EXPECT["folded"]}  '
          f'{"PASS" if len(out) == EXPECT["folded"] else "**FAIL**"}')
    ok &= len(out) == EXPECT['folded']
    m80 = {r['page_path'] for r in
           csv.DictReader(open(M80, encoding='utf-8-sig'))}
    mine = {r['page_path'] for r in out}
    diff = (m80 ^ mine)
    print(f'  KC-2 代表のパスが m80 と違う行  {len(diff):>5}  '
          f'{"PASS" if not diff else "**FAIL**(作者の前の裁定を動かしている)"}')
    if diff:
        for d in sorted(diff)[:4]:
            print(f'       {d}')
    ok &= not diff
    nd = sum(1 for r in out if r['代表の札'] != r['時期の札'])
    print(f'  KC-3 時期の札が代表の札と違う行 {nd:>5}  '
          f'{"PASS" if nd else "**FAIL**(列を足した意味が無い)"}')
    ok &= bool(nd)
    print(f'  KC-4 札が1つに決まらない組     {split:>5} / 期待 '
          f'{EXPECT["split_groups"]}  '
          f'{"PASS" if split == EXPECT["split_groups"] else "**FAIL**"}')
    ok &= split == EXPECT['split_groups']
    print(f'\n  **門 {"すべて PASS" if ok else "**FAIL**。書き出さない"}**')
    if not ok:
        return 1

    print('\n== 札ごとの数(**2つの列は違う数を出す。どちらも印字する**)==')
    print(f'  {"札":8s} {"代表の札で数える":>18} {"**時期の札で数える**":>22} {"差":>8}')
    for k in sorted(set(late_n) | set(early_n), key=key):
        print(f'  {k:8s} {late_n[k]:>16} {early_n[k]:>20} {early_n[k]-late_n[k]:>+8}')
    print(f'  {"合計":8s} {sum(late_n.values()):>16} {sum(early_n.values()):>20}')

    print('\n  **限界(この表を引用するときは必ず付ける)**: '
          '**札はどのページについても「それ以降ではない」という上限**であり、'
          '\n  **早い札に付け替えられるのは同じバイト列が早いフォルダにも在るページだけ**。'
          '\n  **1回しか現れないページは使い回されていても補正されない —— '
          '補正は不均等で、不均等自身が傾向を作り得る。**'
          '\n  **この列は「制作時期」ではなく「最も早く現れた札」である。**')

    f = OUT / 'folded_population_period.csv'
    with open(f, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: r['page_path']))
    print(f'\nwrote {f}')
    f2 = OUT / 'label_counts_two_ways.csv'
    with open(f2, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['札', '代表の札で数える', '時期の札で数える', '差'])
        for k in sorted(set(late_n) | set(early_n), key=key):
            w.writerow([k, late_n[k], early_n[k], early_n[k] - late_n[k]])
    print(f'wrote {f2}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
