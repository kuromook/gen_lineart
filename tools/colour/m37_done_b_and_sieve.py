"""M37: 作者の問い I(`2019b\\ako7\\done*` も `wip`)と、**`rough` の篩が効かないこと**。

作者の回答(逐語。`decided_by` にこのまま):

> I 2018/ako6 2019/ako7 同様に done? はマージ前のバラバラページ

→ **`2019b\\ako7\\done*` の6頁も `wip`。** 射程は
**`2018\\ako6` / `2019\\ako7` / `2019b\\ako7` の `done*` まで**で、
**それ以外の作品の `done*` は未決**(作者は「他のフォルダでは別の意味かもしれない」と明言)。
**`2019b` の `done` 以外**(素の `2019b\\ako7\\ako7\\pageNNNN`)は **b の規則のまま `final`**。

**同時に3つ印字する**:

1. **ako6/ako7 系の外に在る `done*` の列挙**(作者への次の便のため。0なら0と印字)
2. **`rough` の篩** —— **効かない**ことを数で示す
3. **母集団の差分を理由別に**(規律9。**定常の規則にする**)

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-I1** | `2019b\\ako7\\done*` の **6頁**が `wip` になる |
| **KC-I2** | 主値 **764 → 758**(6頁とも `wip` になる) |

**KC-I2 は最初 760 で登録して落ちた(実測758)。** 「`final` 4頁が抜けて `sibling` 2頁は
残る」と書いたが、**`wip` は前の札に関係なく主値から出る**(別行で採点する)。
**規則(6頁を `wip` にする)は登録どおりで、その帰結の算術を間違えた。**
| **KC-I3** | 頁の母集団は **823** のまま |
"""
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m36_rough_variant/version_roles_current.csv')
OUT = Path('results/m37_done_b'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): I 2018/ako6 2019/ako7 同様に done? は'
            'マージ前のバラバラページ')
TARGET = re.compile(r'^F:\\manuscripts\\2019b\\ako7\\done[^\\]*\\', re.I)
OTHER = re.compile(r'^F:\\manuscripts\\(?!2018\\ako6|2019\\ako7|2019b\\ako7)'
                   r'[^\\]+\\[^\\]+\\done[^\\]*\\', re.I)
ANYDONE = re.compile(r'\\done[^\\]*\\', re.I)
EXPECT = {'done': 758, 'pages': 823, 'hits': 6}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    bp = collections.defaultdict(list)
    for r in lay:
        bp[r['page_path']].append(r)
    before_sc = m25.scoring_population(new)
    before_done = [q for q, v in new.items() if v['use'].startswith('採点する')]

    hits = [q for q in new if TARGET.match(q)]
    cause = collections.Counter()
    for q in hits:
        cause[new[q]['version_role']] += 1
        new[q]['version_role'] = 'wip'
        new[q]['decided_by'] = VERBATIM
        new[q]['superseded_by'] = ''
        new[q]['use'] = '別行で採点(マージ前のバラバラページ)'
    print(f'== I の適用: {len(hits)} 頁 ==')
    for q in sorted(hits):
        print(f'  {q[14:]}')
    print(f'  もとの札: ' + ' / '.join(f'{k} {v}' for k, v in cause.most_common()))

    sc = m25.scoring_population(new)
    done = [q for q, v in new.items() if v['use'].startswith('採点する')]
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  主値 {len(before_done)} -> {len(done)}  '
          f'理由: `final` {cause["final"]}頁が `wip` に(マージ前)。'
          f'`sibling` {cause["sibling"]}頁も `wip` に')
    print(f'  採点 {len(before_sc)} -> {len(sc)}  **変わらない**'
          f'(`wip` は採点に残り、別行で出す)')
    print(f'  頁 823 -> {pages}  **変わらない**')

    print(f'\n== ako6/ako7 系の外に在る `done*`(次の便のため)==')
    other = sorted(q for q in new if OTHER.match(q))
    print(f'  **{len(other)} 頁**' + ('(規律7: 無ければ無いと印字)' if not other else ''))
    byfolder = collections.Counter(
        '\\'.join(q.replace('/', '\\').split('\\')[2:5]) for q in other)
    for k, v in byfolder.most_common():
        print(f'    {k:40s} {v:>3} 頁')
    print(f'  (参考)`done*` を含む頁の全数 '
          f'{sum(1 for q in new if ANYDONE.search(q))}')

    print(f'\n== `rough` の篩は効かない ==')
    def f5(q):
        return sum(1 for r in bp[q] if r['formats'] in ("['5']", '5'))
    pool = [q for q, v in new.items() if v['use'].startswith('採点する')]
    nm = re.compile(r'(下絵|下書|下描|ラフ|rough|あたり|アタリ|ネーム|sketch)', re.I)
    name_hits = [q for q in new if any(nm.search(r['name'] or '')
                                      for r in bp[q])]
    print(f'  篩1「カラー層(形式5)が2枚以下」: 主値の母集団 {len(pool)} 頁のうち '
          f'**{sum(1 for q in pool if f5(q) <= 2)} 頁**'
          f' —— **多すぎる。中身はモノクロの原稿で、下絵ではない**')
    print(f'  篩2「ラフ系の名前の層を持つ」: **{len(name_hits)} / {len(new)} 頁**'
          f' —— **`下絵` は作者のテンプレートの標準の層で、823頁中800頁に在る。篩にならない**')
    print(f'  → **機械的な篩で `rough` を洗い出せない。'
          f'0件ではなく、篩が効かないという結果。**')
    koa = sorted(q for q in bp if '\\toho\\koa\\' in q)
    print(f'\n  そして `koa` の下は **{len(koa)} ファイル**'
          f'(作者は「koa下の二つ」と述べた):')
    for q in koa:
        d = [r for r in bp[q] if r['folder'] != 'True'
             and (m19.num(r['drawn_ratio'], 0) or 0) > 0]
        print(f'    形式5 {f5(q):>2} 層 {len(bp[q]):>3} 描画層 {len(d):>3} '
              f'役割 {new.get(q, {}).get("version_role", "-"):>10}  '
              f'{q.split(chr(92))[-1]}')
    print('  **同じ性質なのは `rough` を当てた3つ(カラー層1枚・層33〜39・描画層16〜21)。**')
    print('  **`koakuma.clip`(層21・描画6)と `koakuma_check.clip`(カラー層3)は別の形で、')
    print('  いまは `single` のまま。「二つ」がどれを指すかは作者にしか決まらない。**')

    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-I1 当たった頁', len(hits), EXPECT['hits']),
                         ('KC-I2 主値', len(done), EXPECT['done']),
                         ('KC-I3 頁の母集団', pages, EXPECT['pages'])):
        good = got == want
        ok &= good
        print(f'  {k:20s} {got:>5} / 期待 {want:>5}  '
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
