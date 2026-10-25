"""M28: 作者の `done*` の回答を当てる。**`2018\\ako6` と `2019\\ako7` 限定**。

作者の回答(逐語。これを `decided_by` に入れる):

> 作業途中の退避だ　バラバラに別のcmc 管理化に作成されたものが最終的にako6, ako7 に
> マージされている　この2018/ako6 と 2019/ako7下の done?/*.clip は背景や仕上げがぬけている
> 他のフォルダではdone が別の意味で使われているかもしれないがここでは別の意味となっている

当てる規則:

  D1 `2018\\ako6\\done*` と `2019\\ako7\\done*` の頁 → 役割 **`wip`**(作業途中の退避)
  D2 同じ塊に素の `ako6\\ako6\\pageNNNN` / `ako7\\ako7\\pageNNNN` が**ちょうど1つ**あれば、
     それが `final`。`wip` の `superseded_by` はその頁

**広げない(作者の明言)**: 他のフォルダーの `done`、`_bk`、`backup`、`Done`、`rev2` には
当てない。**`2019b\\ako7\\done2`(年の b 側)も当てない** —— 道が違うので、
**当たるかどうかは問いに回す**(規律9-2: 借りた規則は別の母集団で材料を黙って消す)。

**役割に `wip` を足す**: `done*` は**背景や仕上げが抜けている**(作者)。
**捨てない。** 欠けの向きが既知のプールなので材料になりうるが、**1行に混ぜない**(裁定§2-2)。

**期待(走らせる前に書く。出所は 00:34 に印字した ako6/ako7 の塊の一覧 = この裁定より前)**:

| | 内容 |
|---|---|
| **KC-D1** | `wip` になる塊は **14**(1・2・8・9・56・57・58・59・60・62・95・111・112・113) |
| **KC-D2** | 素の頁が1つある塊は **6**(8・57・58・59・60・62。塊1 は素が3つなので別扱い) |
| **KC-D3** | 役割が変わる頁は **`done*`(D1)か 素の `pageNNNN`(D2)だけ**。他は1つも動かない |

**KC-D2 は最初 7 で登録して落ちた(実測6)。** 登録した**顔ぶれは 6 つしか挙げていない**
のに数を 7 と書いた —— **自分の登録の中の足し算の誤り**で、顔ぶれは一致している。
**KC-D3 も最初は範囲を `2018\\ako6`/`2019\\ako7` の2つだけで書いて落ちた(6頁)。**
動いた6頁は**マージ先の素の頁**(`2018\\ako7\\ako7\\pageNNNN`)で、作者の
「最終的に ako6, ako7 にマージされている」が**年を限っていない**。
**点検の範囲の書き方の誤りで、動いた頁の素性は期待どおり**だった。
| **KC-D4** | `m25` が作者の回答で `final` にした頁は**全部 `final` のまま** |

落ちたら表を書かない。
"""
import collections
import csv
import re
import sys
from pathlib import Path

SRC = Path('results/m25_author_versions/version_roles_author.csv')
OUT = Path('results/m28_done_wip'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): 作業途中の退避だ　バラバラに別のcmc 管理化に'
            '作成されたものが最終的にako6, ako7 にマージされている　'
            'この2018/ako6 と 2019/ako7下の done?/*.clip は背景や仕上げがぬけている　'
            '他のフォルダではdone が別の意味で使われているかもしれないが'
            'ここでは別の意味となっている')
WIP = re.compile(r'^F:\\manuscripts\\(2018\\ako6|2019\\ako7)\\done[^\\]*\\', re.I)
PLAIN = re.compile(r'^F:\\manuscripts\\20\d\d\\(ako6|ako7)\\\1\\page\d+\.clip$',
                   re.I)
EXPECT = {'clusters': 14, 'with_plain': 6}
PRIOR = {'1', '2', '8', '9', '56', '57', '58', '59', '60', '62',
         '95', '111', '112', '113'}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)

    wip = [r['page_path'] for r in rows if WIP.match(r['page_path'])]
    clusters = {new[p]['dup_cluster_id'] for p in wip if new[p]['dup_cluster_id']}
    plain_of = {}
    for ci in clusters:
        pl = [r['page_path'] for r in byc[ci] if PLAIN.match(r['page_path'])]
        if len(pl) == 1:
            plain_of[ci] = pl[0]
    print(f'== 当たる範囲 ==')
    print(f'  `wip` になる頁 **{len(wip)}** / 塊 **{len(clusters)}**  '
          f'{sorted(clusters, key=int)}')
    print(f'  素の頁がちょうど1つある塊 **{len(plain_of)}**  '
          f'{sorted(plain_of, key=int)}')
    allwip = sorted({ci for ci in clusters
                     if all(WIP.match(r['page_path']) for r in byc[ci])}, key=int)
    print(f'  構成員が全員 `wip` の塊(現行がこのプールに無い) {allwip}')

    before = {p: v['version_role'] for p, v in new.items()}
    for p in wip:
        ci = new[p]['dup_cluster_id']
        new[p]['version_role'] = 'wip'
        new[p]['decided_by'] = VERBATIM
        new[p]['superseded_by'] = plain_of.get(ci, '')
    for ci, p in plain_of.items():
        new[p]['version_role'] = 'final'
        new[p]['decided_by'] = VERBATIM
        new[p]['superseded_by'] = ''

    # 動いてよいのは D1(`done*`)と D2(マージ先の素の頁)だけ
    moved = [p for p, v in new.items() if v['version_role'] != before[p]]
    outside = [p for p in moved if not WIP.match(p) and not PLAIN.match(p)]
    promoted = [p for p in moved if PLAIN.match(p)]
    m25final = [r['page_path'] for r in rows
                if r['version_role'] == 'final' and '作者の回答' in r['decided_by']]
    still = [p for p in m25final if new[p]['version_role'] == 'final']

    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-D1 塊の数', len(clusters), EXPECT['clusters']),
                         ('KC-D1 塊の顔ぶれ', int(clusters == PRIOR), 1),
                         ('KC-D2 素が1つの塊', len(plain_of),
                          EXPECT['with_plain']),
                         ('KC-D3 範囲外で変わった頁', len(outside), 0),
                         ('KC-D4 m25 の final が残る', len(still), len(m25final))):
        good = got == want
        ok &= good
        print(f'  {k:28s} {got:>4} / 期待 {want:>4}  '
              f'{"PASS" if good else "**FAIL**"}')
    if clusters != PRIOR:
        print(f'    事前の顔ぶれに無い塊: {sorted(clusters - PRIOR, key=int)}')
        print(f'    事前にあって当たらない塊: {sorted(PRIOR - clusters, key=int)}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1

    print(f'\n== D2 で現行になったマージ先の素の頁 {len(promoted)} ==')
    for q2 in sorted(promoted):
        print(f'    {q2[14:]}  {before[q2]} -> {new[q2]["version_role"]}')
    # 塊24・29: **行為は保守的に継続、根拠は失効**(基盤、2026-10-26)。
    # §4 の理由は「どちらも自分の作品の実在する頁」だったが、作者の
    # 「フォルダーの親は作品を意味しない」で、その前提が塊58 で実際に壊れた。
    # **作者が 24・29 に触れていないことは、前提が生きている証拠ではない。**
    stale = 0
    for ci in ('24', '29'):
        for r in byc.get(ci, []):
            q3 = r['page_path']
            if new[q3]['version_role'] == 'duplicate_content':
                new[q3]['decided_by'] = ('基盤の裁定(根拠は失効、行為のみ保守的に継続。'
                                        '作者の問い J で確認中)')
                stale += 1
    print(f'\n== 根拠を失効と印した頁(行為は継続) {stale}(塊24・29) ==')

    role = collections.Counter(v['version_role'] for v in new.values())
    print(f'\n== 新しい役割の分布 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')

    q = sorted(p for p in new
               if re.match(r'^F:\\manuscripts\\20\d\d[a-z]\\ako7\\done', p, re.I))
    print(f'\n== 問いに回す: 年の b 側の `done*` {len(q)} 頁(規則を広げない) ==')
    for p in q:
        print(f'    {p[14:]}  いまの役割 {new[p]["version_role"]}')

    with open(OUT / 'version_roles_done_wip.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'\nwrote {OUT}/version_roles_done_wip.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
