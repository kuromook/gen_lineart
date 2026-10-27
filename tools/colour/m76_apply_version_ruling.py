#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補104 —— 作者の裁定1・2 を版の札に当てる(追補103 §E の事前登録どおり)。**

**裁定3(同年代)は当てない。** 作者の記入待ちで、**Excel に在る 6頁は札を言い直すだけ**
(「sha256 待ちの暫定」→「同年代。作者に問い合わせ中」。**sha256 は届いたので待っていない**)。

**代表の選び方(裁定1)**: **バイト列が同じなら中身は同じ**なので、どちらを代表にしても
画素は変わらない。**道の辞書順で先に来る方を代表にする**(機械的に決める。
**内容の判断ではない**ことをここに書いておく)。

**`identical_metadata` は `identical_signature` に言い直す**(追補50 §C の項目1)——
**メタデータが同じでもバイト列は同じではない。** 名前が量を誤って指していた。

  <venv>/python tools/colour/m76_apply_version_ruling.py
"""
import collections
import csv
import pathlib
import re
import sys

H = pathlib.Path('../lineart/inbox')
DUP = H / 'sha256_dup_clusters_20261026.csv'
ALLH = H / 'request9_file_sha256_all_20261026.csv'
SRC = pathlib.Path('results/m43_extracted/version_roles_current.csv')
OUT = pathlib.Path('results/m76_version_ruling')
PROV = '採点する(sha256 待ちの暫定)'
YEAR = re.compile(r'^(\d{4})([a-z]?)$')


def rel(p):
    i = p.lower().find('manuscripts\\')
    return p[i + len('manuscripts\\'):] if i >= 0 else p


def year_of(p):
    m = YEAR.match(rel(p).split('\\')[0])
    return int(m.group(1)) if m else None


def main():
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    dup = list(csv.DictReader(DUP.open(encoding='utf-8')))
    allh = {r['page_path']: r for r in csv.DictReader(ALLH.open(encoding='utf-8'))}
    bycl = collections.defaultdict(list)
    for r in dup:
        bycl[r['dup_cluster_id']].append(r)
    print(f'材料 {SRC} {len(rows)}行 / 重複の表 {len(dup)}頁 {len(bycl)}塊 '
          f'/ 全数ハッシュ {len(allh)}')

    before = collections.Counter(r['use'] for r in rows)
    prov = [r for r in rows if r['use'] == PROV]
    print(f'\n暫定の札 {len(prov)}頁(**帰無値: この数が動かない**)')

    # ---- `identical_metadata` -> `identical_signature`(名前の訂正。母集団は動かない)
    tagged = [c for c, m in bycl.items()
              if any(r['tag_identical_metadata'] == '1' for r in m)]
    bad = [c for c in tagged if len({r['sha256'] for r in bycl[c]}) > 1]
    print('\n== 追補50 §C 項目1・2: `identical_metadata` -> `identical_signature` ==')
    print(f'  署名が同じと印の付いた塊 **{len(tagged)}**'
          f' / **うちバイト列が一致しない塊 {len(bad)}**: '
          + ' '.join(sorted(bad, key=int)))
    print('  **台帳の「53塊のうち9塊」は古い数**(当時の表の数)。'
          f'**いまの表では {len(tagged)}塊のうち {len(bad)}塊**。**上書きせず両方を書く**')
    ident = [c for c, m in bycl.items() if len(m) > 1
             and len({r['sha256'] for r in m}) == 1]
    print(f'  バイト同一の塊 {len(ident)} / 印が付いていてバイト同一 {len(tagged) - len(bad)}'
          f'  **同じ集合か: {set(ident) == set(tagged) - set(bad)}**')

    # ---- 裁定を当てる
    moved = collections.Counter()
    detail = []
    for r in prov:
        p = r['page_path']
        cid = r['dup_cluster_id']
        mem = bycl.get(cid, [])
        me = allh.get(p, {}).get('f_sha256', '')
        twins = [m for m in mem if m['page_path'] != p]
        same = [m for m in twins if m['sha256'] == me]
        if same:
            grp = sorted([p] + [m['page_path'] for m in same])
            if p == grp[0]:
                use, why = ('採点する(バイト同一の代表。裁定1)',
                            f'塊{cid} のバイト同一 {len(grp)}頁のうち、道の辞書順で先頭')
                moved['裁定1: 代表になった'] += 1
            else:
                use, why = ('採点しない(バイト同一の複製。代表は別。裁定1)',
                            f'塊{cid} の代表は {rel(grp[0])}')
                moved['裁定1: 複製として落ちた'] += 1
        elif twins and all(year_of(m['page_path']) != year_of(p) for m in twins):
            ys = {year_of(m['page_path']) for m in twins} | {year_of(p)}
            if year_of(p) == max(ys):
                use, why = ('採点する(年代の新しい側。裁定2)',
                            f'塊{cid} の年代 {sorted(ys)} のうち最新')
                moved['裁定2: 新しい側として残った'] += 1
            else:
                use, why = ('採点しない(年代の古い側。裁定2)',
                            f'塊{cid} の年代 {sorted(ys)} のうち古い側')
                moved['裁定2: 古い側として落ちた'] += 1
        else:
            use, why = ('採点する(同年代。作者に問い合わせ中。裁定3)',
                        f'塊{cid}。Excel の「問い(同年代)」に在る')
            moved['裁定3: 作者の記入待ち(言い直しonly)'] += 1
        r['use'], r['decided_by'] = use, why
        detail.append({'page_path': rel(p), 'dup_cluster_id': cid,
                       'use': use, 'decided_by': why})

    print('\n== 20頁の行き先(理由別・残り 0)==')
    n = sum(moved.values())
    for k, v in moved.most_common():
        print(f'  {k:38s} {v:>3}')
    print(f'  {"合計":38s} {n:>3}   **残り {len(prov) - n}**')

    after = collections.Counter(r['use'] for r in rows)
    sc = lambda c: sum(v for k, v in c.items() if k.startswith('採点する'))
    print('\n== 主値: `採点する` と数える頁(事前登録の主値)==')
    print(f'  前 **{sc(before)}** / 後 **{sc(after)}**  差 **{sc(after) - sc(before)}**')
    print(f'  (帰無値は「動かない」。**動いた = 裁定1 で複製が畳まれた**)')
    print('\n== 札の差分(理由別)==')
    for k in sorted(set(before) | set(after)):
        if before[k] != after[k]:
            print(f'  {k:52s} {before[k]:>4} -> {after[k]:>4}')

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'version_roles_current.csv').open('w', newline='',
                                                  encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (OUT / 'ruling_applied_20pages.csv').open('w', newline='',
                                                   encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(detail)
    with (OUT / 'identical_signature_mismatch.csv').open('w', newline='',
                                                         encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['dup_cluster_id', 'n_pages', 'n_sha256',
                    'tag_identical_signature', '相対パス(;; 区切り)'])
        for c in sorted(bad, key=int):
            m = bycl[c]
            w.writerow([c, len(m), len({r['sha256'] for r in m}), 1,
                        ' ;; '.join(rel(r['page_path']) for r in m)])
    print(f'\nwrote {OUT}/version_roles_current.csv'
          f', ruling_applied_20pages.csv, identical_signature_mismatch.csv')
    print('  **m43 の表は上書きしていない**(凍結物。差分を読めるようにする)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
