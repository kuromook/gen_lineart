"""M33: 作者の問い F。**「backupのほうは使わない」= 使用の軸での決着**。
あわせて**作者が置いていった仮説を検定する**(当たっても外れても印字する)。

作者の回答(逐語。`decided_by` にこのまま):

> F backup のほうにはゴミレイヤがのっている なぜこうなってるのか不明だが
> (この余分のレイヤを消したら一致するかもしれない)、backupのほうは使わない

**札の6つめ `not_used`。`superseded` ではない** —— 作者は「古い」と言っておらず
**「使わない」**と言った。**使用の軸での決着**である。

**作者の仮説**: *この余分のレイヤを消したら一致するかもしれない*
→ **`_backup` の署名集合から余分な層を除くと `ako4` の署名集合と一致するか**を測る。
**一致すれば作者の読みが署名の水準で裏付けられる**(画素の水準は sha256)。
**一致しなければそれも結果**で、**差分の層を両側から**列挙する(片側だけ見るのは T101)。

**そして列を2軸に分ける**(基盤、2026-10-26): 今夜の5問で、作者が答えたのは
**「どちらを使うか」と「どういう関係か」**で、**「どちらが新しいか」には5問中5問
答えていない。** → **`use` 列を足す。** `version_role` は関係の軸(既存のまま、
抽出側の DB と揃える)、**`use` が測定にとっての正本**。

| `version_role` | `use` |
|---|---|
| `final` / `single` / `sibling` | 採点する |
| `branch` | 採点する(sha256 待ちの暫定) |
| `undecided` | 採点する(両方。sha256 待ちの暫定) |
| `wip` | **別行で採点**(背景や仕上げが抜けている) |
| `duplicate_content` | 代表のみ採点する / それ以外は採点しない |
| `superseded` | 採点しない(置き換えられた) |
| **`not_used`** | **採点しない(作者が使わないと述べた)** |

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-F1** | 塊35 は `ako4.clip` = `final` / `ako4_backup.clip` = `not_used` |
| **KC-F2** | 採点 **801 → 800** / 主値 **768 → 767**(頁の母集団は **823** のまま) |
| **KC-F3** | **`use` 列から数えた採点の母集団が、`scoring_population()` と一致する** |
| **KC-F4** | sha256 待ちの暫定は **7塊** → 一致なら主値 **767 − 7 = 760** |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m32_charadone/version_roles_current.csv')
OUT = Path('results/m33_use_axis'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): F backup のほうにはゴミレイヤがのっている '
            'なぜこうなってるのか不明だが(この余分のレイヤを消したら一致するかも'
            'しれない)、backupのほうは使わない / **使用の軸での決着(順序ではない)**')
A = r'F:\manuscripts\2017\ako4\color\ako4.clip'
B = r'F:\manuscripts\2017\ako4\color\ako4_backup.clip'
PROVISIONAL = {'17', '22', '30', '103', '104', '105', '18'}
EXPECT = {'scoring': 800, 'done': 767, 'pages': 823}
USE = {'final': '採点する', 'single': '採点する', 'sibling': '採点する',
       'branch': '採点する(sha256 待ちの暫定)',
       'undecided': '採点する(sha256 待ちの暫定)',
       'wip': '別行で採点(背景や仕上げが抜けている)',
       'superseded': '採点しない(置き換えられた)',
       'not_used': '採点しない(作者が使わないと述べた)'}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}

    # --- 作者の仮説の検定 ---
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    sigs = collections.defaultdict(list)
    for r in lay:
        if r['page_path'] in (A, B) and r['folder'] != 'True' \
                and (m19.num(r['drawn_ratio'], 0) or 0) > 0:
            sigs[r['page_path']].append(r)
    sa = {m19.sig(r) for r in sigs[A]}
    sb = {m19.sig(r) for r in sigs[B]}
    print('== 作者の仮説「この余分のレイヤを消したら一致するかもしれない」==')
    print(f'  `ako4` の署名 {len(sa)} / `ako4_backup` の署名 {len(sb)}'
          f'  共通 {len(sa & sb)}')
    only_b = sb - sa
    only_a = sa - sb
    print(f'  `_backup` にだけ在る署名 {len(only_b)} / '
          f'`ako4` にだけ在る署名 {len(only_a)}')
    hit = len(only_a) == 0 and len(only_b) >= 1
    print(f'  → **{"作者の仮説は署名の水準で当たっている" if hit else "一致しない"}**'
          f'({"`_backup` から余分な層を除くと `ako4` の署名集合を包含する" if hit else "両側に差分がある"})')
    for lab, ss, src in (('`_backup` にだけ', only_b, B), ('`ako4` にだけ', only_a, A)):
        for r in sigs[src]:
            if m19.sig(r) in ss:
                print(f'    {lab}: 名前 {r["name"]!r} / 種別 {r["layer_type"]} / '
                      f'描画比 {r["drawn_ratio"]} / 彩度平均 {r["sat_mean"]} / '
                      f'最多色の占有 {r["top_colour_share"]} / 色数 {r["colours_q16"]}'
                      f' / クリップ {r["clip"]}')

    # --- F の適用 ---
    new[A]['version_role'], new[A]['decided_by'] = 'final', VERBATIM
    new[B]['version_role'], new[B]['decided_by'] = 'not_used', VERBATIM
    new[A]['superseded_by'] = new[B]['superseded_by'] = ''
    print(f'\n== 塊35 ==')
    print(f'  {A[14:]} -> final(使う)')
    print(f'  {B[14:]} -> not_used(使わない)')

    # --- use 列 ---
    hasfinal = {v['dup_cluster_id'] for v in new.values()
                if v['version_role'] == 'final' and v['dup_cluster_id']}
    seen = set()
    for q, v in new.items():
        role, ci = v['version_role'], v['dup_cluster_id']
        u = USE.get(role, '')
        if role == 'duplicate_content':
            if ci in hasfinal or ci in seen:
                u = '採点しない(内容が同じ代表が別に在る)'
            else:
                seen.add(ci)
                u = '採点する(内容が同じ組の代表)'
        v['use'] = u
    sc = m25.scoring_population(new)
    by_use = [q for q, v in new.items() if v['use'].startswith('採点する')]
    wip_rows = [q for q, v in new.items() if v['use'].startswith('別行')]
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== `use` 列の分布 ==')
    for k, c in collections.Counter(v['use'] for v in new.values()).most_common():
        print(f'  {k:40s} {c:>4}')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-F1 ako4 = final', int(new[A]['version_role'] == 'final'), 1),
                         ('KC-F1 backup = not_used',
                          int(new[B]['version_role'] == 'not_used'), 1),
                         ('KC-F2 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-F2 主値(仕上がった頁)', len(by_use), EXPECT['done']),
                         ('KC-F2 頁の母集団', pages, EXPECT['pages']),
                         ('KC-F3 use 列 = scoring()',
                          len(by_use) + len(wip_rows), len(sc))):
        good = got == want
        ok &= good
        print(f'  {k:28s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    print(f'\n**主値 {len(by_use)} / `wip` 別行 {len(wip_rows)}。'
          f'sha256 で暫定の7塊が一致すれば主値 {len(by_use) - 7}**')
    cols = list(rows[0].keys()) + ['use']
    with open(OUT / 'version_roles_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'wrote {OUT}/version_roles_current.csv(`use` 列つき)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
