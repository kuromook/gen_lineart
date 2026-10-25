"""M43: 作者の回答 O。**12番目の関係 `extracted_for_delivery`**。

作者の回答(逐語。`decided_by` にこのまま):

> O　イラストのみを作成して載せていたcmc管理ファイルからそれぞれの目的(納品のために)
> １ファイルとして取り出した

対象: **塊25・26**(`toramusume1` ↔ `toramusume2\2`)と **塊3**
(`plecia\work.clip` ↔ `skima2\task\page0023` ↔ `2019b\skima2\task\page0023`)。

**どちらが「集めた側」かは、こちらの列では分からなかった**(下に測定を印字)。
**名前で決めない**(今夜7回目の警告)。**内容は一致しているので、代表をどちらに取っても
画素は変わらない** → **代表は `page_path` の辞書順で先(決定論)。**
**ただし「作品ごとの頁数」は、この塊については当てにならない** ——
**由来の側が分からないまま、片方の作品に1頁を数えることになる。**

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-O1** | 3塊・**7頁**に札が付く(25: 2 / 26: 2 / 3: 3) |
| **KC-O2** | **主値は 701 のまま**(どの塊も代表1頁で、前から1頁ずつ数えていた) |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m41_all_done/version_roles_current.csv')
OUT = Path('results/m43_extracted'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): O　イラストのみを作成して載せていたcmc管理ファイルから'
            'それぞれの目的(納品のために)１ファイルとして取り出した')
CLUSTERS = ['25', '26', '3']
EXPECT = {'pages': 7, 'done': 701}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    bp = collections.defaultdict(list)
    for r in lay:
        bp[r['page_path']].append(r)
    before_done = [q for q, v in new.items() if v['use'].startswith('採点する')]

    print('== どちらが「集めた側」か、列で分けられるか ==')
    for ci in CLUSTERS:
        for r in byc[ci]:
            p = r['page_path']
            v = bp[p]
            d = [x for x in v if x['folder'] != 'True'
                 and (m19.num(x['drawn_ratio'], 0) or 0) > 0]
            print(f'  塊{ci:>3} 層{len(v):>4} 描画{len(d):>4} '
                  f'フォルダー{sum(1 for x in v if x["folder"] == "True"):>3} '
                  f'最上位{sum(1 for x in v if x["parent_id"] == ""):>2} '
                  f'canvas {v[0]["canvas"]:>14}  {p[14:]}')
    print('  → **全列が一致。こちらの列では「集めた側」と「取り出した側」を分けられない。**')
    print('  **名前(`work.clip` / `illust20172.clip` / `SD.clip`)で決めない。**')

    n = 0
    for ci in CLUSTERS:
        g = sorted(r['page_path'] for r in byc[ci])
        rep = g[0]
        for p in g:
            new[p]['version_role'] = 'extracted_for_delivery'
            new[p]['decided_by'] = (VERBATIM + ' / **由来の側は不明(列で分けられない)。'
                                    '代表は辞書順で先(決定論)。内容が同一なので画素は変わらない**')
            new[p]['superseded_by'] = '' if p == rep else rep
            new[p]['use'] = ('採点する(取り出し元の組の代表)' if p == rep
                             else '採点しない(同じ内容の代表が別に在る)')
            n += 1
        print(f'  塊{ci:>3} 代表 {rep[14:]}')

    done = [q for q, v in new.items() if v['use'].startswith('採点する')]
    sc = m25.scoring_population(new)
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  主値 {len(before_done)} -> {len(done)}  '
          f'理由: **代表の数は変わらない**(前も1塊1頁)')
    print(f'  採点 {len(sc)} / 頁 {pages}')
    print(f'  **注意: この3塊については「作品ごとの頁数」が当てにならない** ——'
          f'由来の側が分からないまま片方の作品に1頁を数える')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-O1 札が付いた頁', n, EXPECT['pages']),
                         ('KC-O2 主値', len(done), EXPECT['done'])):
        good = got == want
        ok &= good
        print(f'  {k:22s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    with open(OUT / 'version_roles_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'\nwrote {OUT}/version_roles_current.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
