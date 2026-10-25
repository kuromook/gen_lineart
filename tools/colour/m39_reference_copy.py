"""M39: 作者の問い J。**資料コピー(10番目の関係)**と、**塊24 は作者にも不明**。

作者の回答(逐語。`decided_by` にこのまま):

> J　内容同じ　aljanne2 は PhantomThief の続編のため 資料参照のために
> 新しい作業フォルダにコピーしたようだ　塊24については不明

- **塊29**: `2016\\aljanne2\\aljanne2\\page0029` は **PhantomThief の頁**で、
  **続編の作業フォルダーに資料として置かれているだけ**。
  → **`reference_copy`(10番目)。代表は元の `phantom\\PhantomThief\\page0026`。**
  **フォルダーの側を代表にすると、PhantomThief の頁を aljanne2 の頁として数える。**
- **塊24**(`2019\\road\\page0076` ↔ `2018\\skima11\\skima12`): **作者にも不明**
  → **`undecided`**(D・E と同じ。sha256 の列へ)

**基盤の裁定§4 は、3塊中2塊まで覆った** —— 58 は作者が否定(マージ前)、
29 も作者が否定(資料コピー)、**24 だけが未決**。
**29 は「両方残す」ではなく代表1頁**になる(行為も変わる)。

**測る(基盤の指示。作者の時間は使わない)**:
**作品を跨いで署名が一致する頁対を、65塊の外も含めて全数で**。
**署名の一致は内容の一致ではない(T106)** ので、
**「作品を跨ぐ候補」であって「資料コピー」ではない**と書く。

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-J1** | 塊29: `aljanne2` 側が `reference_copy`、`phantom` 側が代表 |
| **KC-J2** | 採点 **797 → 798** / 主値 **758 → 759**(塊24 の2頁が両方数えられる) |
| **KC-J3** | 頁の母集団は **823** のまま |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m37_done_b/version_roles_current.csv')
OUT = Path('results/m39_reference'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): J　内容同じ　aljanne2 は PhantomThief の続編のため '
            '資料参照のために新しい作業フォルダにコピーしたようだ　塊24については不明')
COPY = r'F:\manuscripts\2016\aljanne2\aljanne2\page0029.clip'
ORIG = r'F:\manuscripts\2016\phantom\PhantomThief\page0026.clip'
EXPECT = {'scoring': 798, 'done': 759, 'pages': 823}


def work(p):
    q = p.replace('\\', '/').split('/')
    return q[3] if len(q) > 3 else ''


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    before_sc = m25.scoring_population(new)
    before_done = [q for q, v in new.items() if v['use'].startswith('採点する')]

    new[COPY]['version_role'] = 'reference_copy'
    new[COPY]['decided_by'] = VERBATIM
    new[COPY]['use'] = '採点しない(続編の作業フォルダーへの資料コピー。元の頁が代表)'
    new[COPY]['superseded_by'] = ORIG
    new[ORIG]['version_role'] = 'duplicate_content'
    new[ORIG]['decided_by'] = VERBATIM
    new[ORIG]['use'] = '採点する(内容が同じ組の代表。元の作品の頁)'
    new[ORIG]['superseded_by'] = ''
    for r in byc['24']:
        q = r['page_path']
        new[q]['version_role'] = 'undecided'
        new[q]['decided_by'] = VERBATIM + ' / **塊24 は作者にも不明。sha256 待ち**'
        new[q]['use'] = '採点する(sha256 待ちの暫定)'
        new[q]['superseded_by'] = ''
    print('== J の適用 ==')
    for q in (COPY, ORIG, *(r['page_path'] for r in byc['24'])):
        print(f'  {new[q]["version_role"]:>16}  {q[14:]}')

    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    pset = collections.defaultdict(set)
    for r in lay:
        if r['folder'] == 'True' or (m19.num(r['drawn_ratio'], 0) or 0) <= 0:
            continue
        pset[r['page_path']].add(m19.sig(r))
    print(f'\n== 作品を跨いで署名が一致する頁対(**全数。65塊の外も含む**)==')
    print('  **署名の一致は内容の一致ではない(T106)。'
          'これは「作品を跨ぐ候補」であって「資料コピー」ではない。**')
    out = []
    for ci, g in sorted(byc.items(), key=lambda kv: int(kv[0])):
        ws = {work(r['page_path']) for r in g}
        if len(ws) <= 1:
            continue
        ps = [r['page_path'] for r in g]
        j = None
        if len(ps) == 2:
            a, b = pset[ps[0]], pset[ps[1]]
            j = len(a & b) / max(1, len(a | b))
        out.append((ci, ps, j))
        roles = '/'.join(new[p]['version_role'] for p in ps)
        print(f'  塊{ci:>4} J={j if j is None else round(j, 3)!s:>5} '
              f'{roles:<34} ' + ' ↔ '.join(p[14:] for p in ps))
    print(f'  → **{len(out)} 塊**'
          + ('(規律7: 無ければ無いと印字)' if not out else ''))

    sc = m25.scoring_population(new)
    done = [q for q, v in new.items() if v['use'].startswith('採点する')]
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  採点 {len(before_sc)} -> {len(sc)}  理由: 塊24 の2頁が '
          f'`duplicate_content`(代表1頁)から `undecided`(両方)に **+1**')
    print(f'  主値 {len(before_done)} -> {len(done)}  同上')
    print(f'  頁 823 -> {pages}  **変わらない**(塊29 の資料コピーも台帳に残す)')
    print(f'  **曖昧な `koa\\koakuma2.clip` を含めると主値は {len(done) + 1}**'
          f'(作者は「koa下の二つ」、こちらは3つに `rough` を当てた)')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-J1 資料コピー',
                          int(new[COPY]['version_role'] == 'reference_copy'), 1),
                         ('KC-J2 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-J2 主値', len(done), EXPECT['done']),
                         ('KC-J3 頁の母集団', pages, EXPECT['pages'])):
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
    with open(OUT / 'cross_work_candidates.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['dup_cluster_id', 'Jaccard', 'page_path', 'version_role',
                    '作品フォルダー'])
        for ci, ps, j in out:
            for p in ps:
                w.writerow([ci, '' if j is None else f'{j:.3f}', p,
                            new[p]['version_role'], work(p)])
    print(f'\nwrote {OUT}/version_roles_current.csv / cross_work_candidates.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
