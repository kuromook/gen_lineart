"""M36: 作者の問い H。**1つの塊に「下絵」と「仕様違いの成果物」が同居していた**。

作者の回答(逐語。`decided_by` にこのまま):

> H　koa下の二つは下絵のみ　koakuma.clip と koakuma2.clipは中身はほぼ同じだが、
> koakuma2 のほうはテーブルがある　おそらく製作する際の依頼者により２種類おねがいされたもの

**札の8つめ `rough`(下絵のみ)と9つめ `variant`(仕様違い。両方が成果物)。**

- **`rough` は `wip` より深い欠け** —— `wip` は*背景や仕上げが抜けている*、
  **`rough` は下絵しかない = 面の標準にとって採点する中身が無い**。
  **主値から外し、理由つきで台帳に残す**
- **`variant` は `branch` と違う** —— **差が名指しされている**(テーブルの有無)、
  **外因がある**(依頼者が2種類頼んだ)、**両方が納品された成果物**。順序は無い

**作者の言葉との食い違いを1つ記録する**: 作者は「**koa下の二つ**」と述べたが、
**こちらの母集団では `koa` の下に3ファイル在る**(`koakuma2b..clip` /
`koakuma2..clip` / `koakuma2.clip`)。**3つとも形式5(カラー)が1枚だけ**で
同じ性質なので同じ札を当てるが、**この1件は作者の確認が要る**(推測で埋めない)。

**あわせて測る(作者の時間を使わない)**:

1. **`rough` の候補を機械的に洗い出す** —— 下絵しかないファイルが母集団に他にも在れば、
   **仕上がった頁として採点されている**。篩は **形式5(カラー)の層が 2枚以下**
   (名指しされた3ファイルは1枚)。**候補が0なら0と印字**(規律7)。
   **注意**: `nuru`/`NY`/`poolsideIdle` は**カラー層が何十枚もある**ので篩に掛からない
   —— 「平坦な面が見当たらない」と「下絵しかない」は別の形(基盤)
2. **`variant` の差が署名に見えるか** —— 作者が名指しした差(テーブルの有無)を
   `m19.sig` の差分で探す。**見えれば署名の感度の一例、見えなければ T106 の追加の証拠。
   どちらでも印字**

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-H1** | `koa` の3ファイル → `rough`、`toho\\koakuma(2).clip` 2つ → `variant` |
| **KC-H2** | 主値は **767 − 3(rough)= 764**。頁の母集団は **823 のまま**(落とさない) |
| **KC-H3** | 採点の母集団は **800 − 3 = 797** |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m34_calibration/version_roles_current.csv')
OUT = Path('results/m36_rough_variant'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): H　koa下の二つは下絵のみ　koakuma.clip と '
            'koakuma2.clipは中身はほぼ同じだが、koakuma2 のほうはテーブルがある　'
            'おそらく製作する際の依頼者により２種類おねがいされたもの')
NOTE3 = ('(作者は「koa下の二つ」と述べたが母集団には3ファイル在る。'
         '3つとも形式5が1枚で同じ性質。**作者の確認が要る**)')
ROUGH = [r'F:\manuscripts\2017\toho\koa\koakuma2b..clip',
         r'F:\manuscripts\2017\toho\koa\koakuma2..clip',
         r'F:\manuscripts\2017\toho\koa\koakuma2.clip']
VARIANT = [r'F:\manuscripts\2017\toho\koakuma.clip',
           r'F:\manuscripts\2017\toho\koakuma2.clip']
EXPECT = {'done': 764, 'scoring': 797, 'pages': 823}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    bp = collections.defaultdict(list)
    for r in lay:
        bp[r['page_path']].append(r)
    before_sc = m25.scoring_population(new)
    before_done = len(before_sc) - sum(
        1 for q in before_sc if new[q]['version_role'] == 'wip')

    for q in ROUGH:
        new[q]['version_role'] = 'rough'
        new[q]['decided_by'] = VERBATIM + ' ' + NOTE3
        new[q]['superseded_by'] = ''
        new[q]['use'] = '採点しない(下絵のみ。面の標準にとって中身が無い)'
    for q in VARIANT:
        new[q]['version_role'] = 'variant'
        new[q]['decided_by'] = VERBATIM
        new[q]['superseded_by'] = ''
        new[q]['use'] = '採点する(仕様違い。両方が成果物)'

    # --- 1. rough の候補を洗い出す ---
    def f5(p):
        return sum(1 for r in bp[p] if r['formats'] in ("['5']", '5'))

    def drawn(p):
        return [r for r in bp[p] if r['folder'] != 'True'
                and (m19.num(r['drawn_ratio'], 0) or 0) > 0]
    pool = [q for q, v in new.items()
            if v['use'].startswith('採点する') or v['use'].startswith('別行')]
    cand = sorted((q for q in pool if f5(q) <= 2), key=lambda q: (f5(q), q))
    print(f'== `rough` の候補(形式5が2枚以下。採点に入っている頁のみ)==')
    print(f'  母集団 {len(pool)} 頁のうち **{len(cand)} 頁**')
    for q in cand[:40]:
        print(f'    形式5 {f5(q):>2} 描画層 {len(drawn(q)):>4} '
              f'クリップ {sum(1 for r in bp[q] if r["clip"] == "1"):>3}  '
              f'{q[14:]}')
    if len(cand) > 40:
        print(f'    … 他 {len(cand) - 40} 頁')
    if not cand:
        print('    **0頁**(規律7: 無ければ無いと印字)')

    # --- 2. variant の差が署名に見えるか ---
    a, b = VARIANT
    sa = {m19.sig(r): r for r in drawn(a)}
    sb = {m19.sig(r): r for r in drawn(b)}
    print(f'\n== `variant` の差(作者が名指しした差 = テーブルの有無)==')
    print(f'  `koakuma.clip` の署名 {len(sa)} / `koakuma2.clip` {len(sb)}  '
          f'共通 {len(set(sa) & set(sb))}  Jaccard '
          f'{len(set(sa) & set(sb)) / len(set(sa) | set(sb)):.3f}')
    for lab, only, src in (('koakuma にだけ', set(sa) - set(sb), sa),
                           ('koakuma2 にだけ', set(sb) - set(sa), sb)):
        print(f'  {lab} {len(only)} 層:')
        for k in sorted(only, key=lambda k: -(m19.num(src[k]['drawn_ratio'], 0) or 0))[:12]:
            r = src[k]
            print(f'    名前 {r["name"]!r:28s} 形式 {r["formats"]:>6} '
                  f'描画比 {r["drawn_ratio"]:>10} 彩度 {r["sat_mean"]:>7} '
                  f'クリップ {r["clip"]}')

    sc = m25.scoring_population(new)
    by_use = [q for q, v in new.items() if v['use'].startswith('採点する')]
    pages = sum(1 for v in new.values()
                if v['version_role'] not in ('superseded',))
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  主値 {before_done} -> {len(by_use)}  '
          f'理由: `rough` 3頁を外した(下絵のみ)')
    print(f'  採点 {len(before_sc)} -> {len(sc)}  同上')
    print(f'  頁 823 -> {pages}  **変わらない**(`rough` も台帳には残す)')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-H2 主値', len(by_use), EXPECT['done']),
                         ('KC-H3 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-H2 頁の母集団', pages, EXPECT['pages'])):
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
    with open(OUT / 'rough_candidates.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', '形式5の層', '描画層', 'クリップ層',
                    'いまの version_role'])
        for q in cand:
            w.writerow([q, f5(q), len(drawn(q)),
                        sum(1 for r in bp[q] if r['clip'] == '1'),
                        new[q]['version_role']])
    print(f'\nwrote {OUT}/version_roles_current.csv / rough_candidates.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
