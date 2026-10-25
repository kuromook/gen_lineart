"""M30: 作者の問い B の回答を当てる。**別経路のブランチ。どちらが新しいはない**。

作者の回答(逐語。`decided_by` にこのまま):

> B　一部内容が被ってそうだが別経路のブランチ　どちらが新しいはない

**役割の4つめ `branch`**(基盤、2026-10-26):

| | 作者の言葉 | 札 | 母集団 |
|---|---|---|---|
| `done*` | 背景や仕上げが抜けている | `final` + `wip` | 主値は `final`、`wip` は別行 |
| A | 内容は同じ | `duplicate_content` | 代表1頁 |
| **B** | **別経路のブランチ。どちらが新しいはない** | **`branch`** | **両方が現行の別の頁。両方入れる** |

**`superseded` も `wip` も `duplicate_content` も使わない。** 「どちらが新しいか」という
問いがこの塊では成り立っていないので、**順序を付けない札**にする。

**塊の単位そのものへの疑い**(基盤): 塊は *「1つの頁の、複数の複写」* を前提に組まれている。
**「一部内容が被る別経路」はその前提を満たさない。** → `cluster_assumption_void` を
`decided_by` に明記する。**T95 の次の段: 塊は「1頁の複写の集合」でもない。**

**測る(基盤の指示。新規の発注はしない)**: 4塊それぞれの**重なりの量**を、
`m19.sig`(**塊を組んだのと同じ署名**。画素ではない)で数える。
**包含があっても順序ではない**(作者が明言。`superset` が 1/3 で落ちたのと同じ形)。

**期待(走らせる前に書く)**:

| | 内容 | 出所 |
|---|---|---|
| **KC-S1** | 当たる塊は **4**(17・21・22・23) | 00:58 に印字した skima10 の塊 |
| **KC-S2** | 頁の母集団 **816 → 818**(塊17・22 の `superseded` が戻る) | 追補20 §G と `m29` |
| **KC-S3** | 採点 **794 → 796** / 仕上がった頁 **761 → 763** | 同上 |
| **KC-S4** | **重なりは4塊とも Jaccard >= 0.5**(塊を組んだ基準なので当然。**これは独立な証拠ではない**) | `m19` の組み方 |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m29_underworld/version_roles_current.csv')
CSV872 = m19.CSV872
OUT = Path('results/m30_branches'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): B　一部内容が被ってそうだが別経路のブランチ　'
            'どちらが新しいはない / cluster_assumption_void'
            '(この塊は「1頁の複写の集合」ではない)')
PRIOR = {'17', '21', '22', '23'}
EXPECT = {'pages': 818, 'scoring': 796, 'done': 763}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    hit = {ci for ci, g in byc.items()
           if all('\\skima10\\' in r['page_path'] for r in g)}
    print(f'== 当たる塊 {len(hit)}  {sorted(hit, key=int)} ==')

    lay = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    pset = collections.defaultdict(set)
    for r in lay:
        if r['folder'] == 'True' or (m19.num(r['drawn_ratio'], 0) or 0) <= 0:
            continue
        pset[r['page_path']].add(m19.sig(r))

    print(f'\n== 重なりの量(`m19.sig` = **塊を組んだのと同じ署名**。画素ではない)==')
    print(f'  {"塊":>4} {"浅い側":>6} {"深い側":>6} {"共通":>5} {"Jaccard":>8}  向き')
    ok4 = True
    for ci in sorted(hit, key=int):
        g = byc[ci]
        sh = min((r['page_path'] for r in g), key=lambda q: q.count('\\'))
        dp = [r['page_path'] for r in g if r['page_path'] != sh][0]
        a, b = pset[sh], pset[dp]
        c = len(a & b)
        j = c / max(1, len(a | b))
        ok4 &= j >= 0.5
        d = ('深い側が浅い側を包む' if a < b else
             '浅い側が深い側を包む' if b < a else 'どちらも包まない')
        print(f'  {ci:>4} {len(a):>6} {len(b):>6} {c:>5} {j:>8.3f}  {d}')
    print('  **包含があっても順序ではない**(作者が明言)')

    before_p = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    before_s = m25.scoring_population(new)
    cause = collections.Counter()
    for ci in hit:
        for r in byc[ci]:
            q = r['page_path']
            cause[new[q]['version_role']] += 1
            new[q]['version_role'] = 'branch'
            new[q]['decided_by'] = VERBATIM
            new[q]['superseded_by'] = ''
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    sc = m25.scoring_population(new)
    wip = sum(1 for p in sc if new[p]['version_role'] == 'wip')

    print(f'\n== 母集団が動いた理由(規律9)==')
    print('  もとの札 -> branch: ' + ' / '.join(
        f'{k} {v}' for k, v in cause.most_common()))
    print(f'  **`superseded` だった {cause["superseded"]}頁が、現行の別の頁として戻った**'
          f' = 頁 {before_p} -> {pages}(+{pages - before_p})')
    print(f'  `undecided` だった {cause["undecided"]}頁は、もとから母集団に居る'
          f'(札だけが「決まらない」から「順序が無い」に変わった)')
    print(f'  採点 {len(before_s)} -> {len(sc)}  / 仕上がった頁 '
          f'{len(before_s) - sum(1 for p in before_s if new[p]["version_role"] == "wip")}'
          f' -> {len(sc) - wip}')

    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-S1 塊の数', len(hit), len(PRIOR)),
                         ('KC-S1 塊の顔ぶれ', int(hit == PRIOR), 1),
                         ('KC-S2 頁の母集団', pages, EXPECT['pages']),
                         ('KC-S3 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-S3 仕上がった頁', len(sc) - wip, EXPECT['done']),
                         ('KC-S4 重なり Jaccard>=0.5', int(ok4), 1)):
        good = got == want
        ok &= good
        print(f'  {k:24s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    role = collections.Counter(v['version_role'] for v in new.values())
    print(f'\n== 役割 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')
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
