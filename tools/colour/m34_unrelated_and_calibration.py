"""M34: 作者の問い G。**「まったくべつもの」= 塊の、測られた最初の偽陽性**。
あわせて**作者が札を付けた20塊 × Jaccard** を印字する(**率は計算しない**)。

作者の回答(逐語):

> G　まったくべつもの

(`2019b\\hitoduma\\CD.clip` ↔ `CDrev2.clip`、塊27)

**版の関係ではなく「関係が無い」という答。** → **札の7つめ `unrelated`。
塊を解いて両方を独立した頁として扱う**(代表も順序も無い)。

**濃縮標本であることを明記する(基盤)**: 作者に出した20塊は
**こちらが「決まらない難しい塊」として選んだ濃縮標本**である。
**精度も偽陽性率も出せない。** 出せるのは
**「作者の札ごとに Jaccard がどう分布するか」**と**「偽陽性が少なくとも1つ在る」**こと。

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-G1** | 塊27 の2頁が `unrelated`。**採点の母集団は 800 のまま**(どちらも採点する頁だった) |
| **KC-G2** | 主値 **767 のまま**(塊27 は `undecided` で両方採点していた) |
| **KC-G3** | 作者が札を付けた塊は **20**(A 9 / B 4 / D 4 / E 1 / F 1 / G 1) |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402
import m33_ako4_not_used as m33                          # noqa: E402

SRC = Path('results/m33_use_axis/version_roles_current.csv')
OUT = Path('results/m34_calibration'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): G　まったくべつもの'
            ' / **版の関係ではなく「関係が無い」。塊の偽陽性**')
LABEL = {'内容は同じ(A)': ['96', '97', '98', '99', '100', '101', '108', '109', '110'],
         '別経路(B)': ['17', '21', '22', '23'],
         'わからない(D)': ['30', '103', '104', '105'],
         'おそらくバックアップ(E)': ['18'],
         '使わない(F)': ['35'],
         'まったくべつもの(G)': ['27']}
EXPECT = {'scoring': 800, 'done': 767, 'labelled': 20}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)

    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    pset = collections.defaultdict(set)
    for r in lay:
        if r['folder'] == 'True' or (m19.num(r['drawn_ratio'], 0) or 0) <= 0:
            continue
        pset[r['page_path']].add(m19.sig(r))

    def jac(ci):
        g = [r['page_path'] for r in byc[ci]]
        if len(g) != 2:
            return None, None, None
        a, b = pset[g[0]], pset[g[1]]
        return len(a & b), len(a | b), len(a & b) / max(1, len(a | b))

    c, u, j = jac('27')
    print(f'== G(塊27)の重なり —— **閾値の較正そのもの** ==')
    for r in byc['27']:
        print(f'  {r["page_path"][14:]}  層の署名 {len(pset[r["page_path"]])}')
    print(f'  共通 **{c}** / 和 {u} / **Jaccard {j:.3f}**'
          f'  (塊に入れた条件は Jaccard >= 0.5)')
    print(f'  → **{"閾値 0.5 のすぐ上。境界ぎわの偽陽性" if j < 0.6 else "閾値から離れた高い値。署名6列が別物を同じに見せている(T106 がより深い)"}**')

    for r in byc['27']:
        q = r['page_path']
        new[q]['version_role'] = 'unrelated'
        new[q]['decided_by'] = VERBATIM
        new[q]['superseded_by'] = ''
        new[q]['use'] = '採点する(塊を解いた。独立した頁)'

    sc = m25.scoring_population(new)
    by_use = [q for q, v in new.items() if v['use'].startswith('採点する')]
    wipr = [q for q, v in new.items() if v['use'].startswith('別行')]

    print(f'\n== 作者の札 × Jaccard(**濃縮標本。率は出さない**)==')
    print(f'  {"作者の札":26s} {"塊":>3}  Jaccard')
    nlab = 0
    for lab, cis in LABEL.items():
        vals = []
        for ci in cis:
            _, _, jj = jac(ci)
            if jj is not None:
                vals.append(jj)
        nlab += len(cis)
        if vals:
            print(f'  {lab:26s} {len(cis):>3}  '
                  f'最小 {min(vals):.3f} / 中央 {sorted(vals)[len(vals)//2]:.3f} / '
                  f'最大 {max(vals):.3f}'
                  + ('  ' + ' '.join(f'{v:.3f}' for v in sorted(vals))
                     if len(vals) <= 4 else ''))
        else:
            print(f'  {lab:26s} {len(cis):>3}  (構成員が2つでない塊)')
    print('  **この20塊は「決まらない難しい塊」として選んだ濃縮標本である。**')
    print('  **精度も偽陽性率も出せない。** 言えるのは'
          '**偽陽性が少なくとも1つ在る**ことと、札ごとの分布だけ。')

    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-G1 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-G2 主値', len(by_use), EXPECT['done']),
                         ('KC-G3 作者が札を付けた塊', nlab, EXPECT['labelled'])):
        good = got == want
        ok &= good
        print(f'  {k:24s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    print(f'\n== 役割 ==')
    for k, cc in collections.Counter(
            v['version_role'] for v in new.values()).most_common():
        print(f'  {k:20s} {cc:>4}')
    with open(OUT / 'version_roles_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'\nwrote {OUT}/version_roles_current.csv')
    print(f'**主値 {len(by_use)} / `wip` 別行 {len(wipr)}。'
          f'sha256 で暫定7塊が一致すれば {len(by_use) - 7}**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
