"""M41: 作者の回答 L(**`done*` は全部同じ運用・共通して背景がない**)と
M'(**`toho\\koa` の5ファイルは全部下絵**)。

作者の回答(逐語。`decided_by` にこのまま):

> L 全部同じ運用だ　共通して背景がない
> M' 全部下絵

1. **`done*` はフォルダーを問わず `wip`** —— ako6/ako7 系の外の **56頁**も。
   **`done*` の全95頁が `wip`**
2. **`toho\\koa` の5ファイルは全部 `rough`** —— こちらは**保守的に3つ**に当てていた。
   **作者は5つ**。**こちらが立てた二択(3つ=759 / 4つ=760)の外に答が在った。**

**記録(基盤)**: **曖昧を二択で立てると、真の答がその外に在りうる。**
**範囲ではなく「作者に決めてもらう」が常に正しい。**

**篩の初めての検証**: 否定形の篩は `koa` の5ファイルを **5/5** で拾っていた
(0.001 / 0.020 / 0.018 / 0.018 / 0.003、棒 0.020)。
**篩が作者と一致し、保守的に付けた札の方が外れていた。**
**ただし5件は5件** —— **棒はこの5件の最大値で引かれている**ので、
**「棒を引いた当の集合で一致した」以上のことは言えない。**

**同名ファイルの点検(T94 の族)**: `koakuma.clip` は2か所に在る ——
`2017\\toho\\koa\\koakuma.clip`(層21 / 描画6、**`rough`**)と
`2017\\toho\\koakuma.clip`(署名92、**`variant`**)。
**この木の定数はすべてフルパスで書いてある**(`m33` の `VARIANT`、`m36` の `ROUGH`)。

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-L1** | `done*` の `wip` が **95頁**(ako6/ako7 系 39 + 外 56) |
| **KC-M1** | `koa` の `rough` が **5頁** |
| **KC-L2** | 主値は **759 → 757(M' で −2)→ さらに L の分だけ減る**。理由別に印字 |
"""
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m40_revision/version_roles_current.csv')
OUT = Path('results/m41_all_done'); OUT.mkdir(parents=True, exist_ok=True)
VL = '作者の回答(2026-10-26): L 全部同じ運用だ　共通して背景がない'
VM = '作者の回答(2026-10-26): M\' 全部下絵'
DONE = re.compile(r'\\done[^\\]*\\', re.I)
KOA = re.compile(r'^F:\\manuscripts\\2017\\toho\\koa\\', re.I)
EXPECT = {'done': 95, 'koa': 5}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    before_done = [q for q, v in new.items() if v['use'].startswith('採点する')]
    before_wip = [q for q, v in new.items() if v['use'].startswith('別行')]
    cause = collections.Counter()

    done = [q for q in new if DONE.search(q)]
    for q in done:
        if new[q]['version_role'] != 'wip':
            cause[f'{new[q]["version_role"]}->wip'] += 1
        new[q]['version_role'] = 'wip'
        new[q]['decided_by'] = VL
        new[q]['superseded_by'] = ''
        new[q]['use'] = '別行で採点(マージ前のバラバラページ。共通して背景がない)'
    koa = [q for q in new if KOA.match(q)]
    for q in koa:
        if new[q]['version_role'] != 'rough':
            cause[f'{new[q]["version_role"]}->rough'] += 1
        new[q]['version_role'] = 'rough'
        new[q]['decided_by'] = VM
        new[q]['use'] = '採点しない(下絵のみ。**作者が5ファイルとも下絵と述べた**)'
    print(f'== `done*` を `wip` に: {len(done)} 頁 / '
          f'`koa` を `rough` に: {len(koa)} 頁 ==')
    print('  札の移動: ' + ' / '.join(f'{k} {v}' for k, v in cause.most_common()))

    sc = m25.scoring_population(new)
    cur = [q for q, v in new.items() if v['use'].startswith('採点する')]
    wip = [q for q, v in new.items() if v['use'].startswith('別行')]
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  主値 {len(before_done)} -> **{len(cur)}**')
    for k, v in cause.most_common():
        print(f'    {k}: {v} 頁')
    print(f'  `wip` 別行 {len(before_wip)} -> **{len(wip)}**'
          f'(`done*` の全95頁)')
    print(f'  採点(主値 + 別行) {len(sc)}  / 頁の母集団 {pages}')

    print(f'\n== 二択の外に答が在った(記録)==')
    print(f'  こちらが立てた曖昧: **3つ(759)か 4つ(760)か**')
    print(f'  作者の答: **5つ** → 主値は **{len(cur)}**'
          f'(`koakuma.clip` 層21/描画6 と `koakuma_check.clip` 28/12 も下絵)')
    print(f'  **曖昧を二択で立てると、真の答がその外に在りうる。**')

    print(f'\n== 否定形の篩の初めての検証 ==')
    print(f'  篩(棒 0.020)は `koa` の5ファイルを **5/5** で拾っていた'
          f'(0.001 / 0.020 / 0.018 / 0.018 / 0.003)')
    print(f'  M の札(保守的に3つ)は **3/5**。**篩が作者と一致し、札の方が外れた**')
    print(f'  **ただし棒はこの5件の最大値で引かれている** ——'
          f'**「棒を引いた当の集合で一致した」以上のことは言えない**')

    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-L1 done* の wip', len(done), EXPECT['done']),
                         ('KC-M1 koa の rough', len(koa), EXPECT['koa'])):
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
