"""M29: 作者の問い A の回答を当てる。**`comics4\\underworld` ↔ `underworld` は内容同一**。

作者の回答(逐語。`decided_by` にこのまま):

> A は内容は同じだな

**`done*` とは札が違う**(基盤、2026-10-26):

| | 作者の言葉 | 札 |
|---|---|---|
| `done*`(ako6/ako7) | 背景や仕上げが抜けている | `final` + `wip`(**欠けている**ので別プール) |
| **A の9塊** | **内容は同じ** | **`duplicate_content`(欠けていない)** |

→ **A に `wip` を使わない。** どちらが現行かは作者が述べていないので
**`superseded` とも書かない**(T94: 内容の同一性を頁の置き換えに化けさせない)。

**代表の選び方(決定論。印字する)**: **`page_path` の辞書順で先の方**。
R-0 が同じ年のときに使っていた規則と同じなので、台帳が揺れない。
**内容が同一なので、どちらを代表にしても画素は変わらない**(基盤)。

**期待(走らせる前に書く)**:

| | 内容 | 出所 |
|---|---|---|
| **KC-U1** | 当たる塊は **9**(96・97・98・99・100・101・108・109・110) | 追補18 §E(10-26 00:18 に印字した広い走査) |
| **KC-U2** | 頁の母集団 **807 → 816**(+9。`superseded` が戻る) | 追補20 §G の数 |
| **KC-U3** | 採点の母集団は **794 のまま**(代表1つは前からそうだった) | 同上 |
"""
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m25_author_version_answers as m25                # noqa: E402

SRC = Path('results/m28_done_wip/version_roles_done_wip.csv')
OUT = Path('results/m29_underworld'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = '作者の回答(2026-10-26): A は内容は同じだな'
RULE = '代表は page_path の辞書順で先の方(決定論。内容が同一なので画素は変わらない)'
BUNDLE = re.compile(r'^F:\\manuscripts\\2018\\comics4\\underworld\\underworld\\',
                    re.I)
PLAINU = re.compile(r'^F:\\manuscripts\\2018\\underworld\\underworld\\', re.I)
PRIOR = {'96', '97', '98', '99', '100', '101', '108', '109', '110'}
EXPECT = {'pages': 816, 'scoring': 794}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    hit = {ci for ci, g in byc.items()
           if any(BUNDLE.match(r['page_path']) for r in g)
           and any(PLAINU.match(r['page_path']) for r in g)}
    print(f'== 当たる塊 {len(hit)}  {sorted(hit, key=int)} ==')
    before_pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    before_scoring = len(m25.scoring_population(new))
    for ci in hit:
        for r in byc[ci]:
            q = r['page_path']
            new[q]['version_role'] = 'duplicate_content'
            new[q]['decided_by'] = f'{VERBATIM} / {RULE}'
            new[q]['superseded_by'] = ''
        rep = sorted(r['page_path'] for r in byc[ci])[0]
        print(f'  塊{ci:>4} 代表 {rep[14:]}')
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    scoring = m25.scoring_population(new)
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-U1 塊の数', len(hit), len(PRIOR)),
                         ('KC-U1 塊の顔ぶれ', int(hit == PRIOR), 1),
                         ('KC-U2 頁の母集団', pages, EXPECT['pages']),
                         ('KC-U3 採点の母集団', len(scoring), EXPECT['scoring'])):
        good = got == want
        ok &= good
        print(f'  {k:22s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    print(f'  (前) 頁 {before_pages} / 採点 {before_scoring}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    role = collections.Counter(v['version_role'] for v in new.values())
    wip = sum(1 for p in scoring if new[p]['version_role'] == 'wip')
    print(f'\n== 役割 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')
    print(f'\n  頁の母集団 **{pages}** / 採点 **{len(scoring)}**'
          f'(うち `wip` {wip} → **仕上がった頁 {len(scoring) - wip}**)')
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
