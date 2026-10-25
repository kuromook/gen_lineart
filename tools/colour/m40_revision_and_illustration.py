"""M40: 作者の問い K。**`revision`(初めての順序。ただし作者の推定)**と
**`illustration`(1枚イラストは原稿の頁ではない)**。

作者の回答(逐語。`decided_by` にこのまま):

> K　単行本化のために原稿を束ねたフォルダ　改訂があるはず　color_comics4 はこの
> 単行本用のカラーカット(１枚イラスト)

1. **`comics4` = 単行本化のために原稿を束ねたフォルダ** ——
   「フォルダー ≠ 作品」の**3つめの機序: 刊行の単位**
2. **`china_v2` / `mars_v2` = 改訂**(塊5・6・13・36・37)。**7問目にして初めての順序。**
   **ただし作者は「改訂があるはず」= 推定。** → 札 **`revision`**、向きは **v2 が新しい**、
   `decided_by` に**「作者の推定(『はず』)」**と明記。**断定として流通させない**
   → **sha256 で検定できる**: **一致 = 単行本化で手を入れなかった頁**(それ自体が制作の事実)/
   **不一致 = 改訂された頁**。**作者の「はず」を頁の数に変えられる**
3. **`color_comics4` の19頁 = カラーカット(1枚イラスト)。原稿の頁ではない**
   → 札 **`illustration`**。**落とさないが、別の層(stratum)として印字する**

**そして登録集合そのものを点検する**(基盤の指摘から):
**8ページのうち 4ページが `illustration`/`illust*` のフォルダー、2ページが単独ファイル。**
**A0 と標準は、ほぼイラストの集合で較正されている。**

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-K1** | `revision` が **5塊 13頁**(china の3塊は**3者** —— `2016\china`・`2018\china_v2`・`2018\comics4\china_v2`)。向きは v2 が新しい |

**KC-K1 を 10頁で登録して落ちた(実測13)。** china の3塊が2者だと思い込んでいた
(**作品跨ぎの走査で3者だと自分で印字していた**)。**規則は正しく、数え方が違った。**
**そして3者だと v2 側が2つある** —— `comics4` の下と外に同じ改訂が在るので、
**v2 側は代表1つにする**(A と同じ。内容が同じものを2回数えない)。代表は
**`page_path` の辞書順で先**(= `comics4` の外。決定論)。
| **KC-K2** | `illustration` が **19頁**(`color_comics4`) |
| **KC-K3** | **主値は 759 のまま**(代表の規則が効くので)。**頁の母集団は 823 → 831**
             —— `revision` の古い側5頁と v2 の重複3頁が、`superseded` から戻るため。
             **理由別に印字する** |
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m39_reference/version_roles_current.csv')
OUT = Path('results/m40_revision'); OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
VERBATIM = ('作者の回答(2026-10-26): K　単行本化のために原稿を束ねたフォルダ　'
            '改訂があるはず　color_comics4 はこの単行本用のカラーカット(１枚イラスト)')
REV = ['5', '6', '13', '36', '37']
V2 = re.compile(r'\\(china_v2|mars_v2)\\', re.I)
ILLDIR = re.compile(r'\\color_comics4\\', re.I)
ILLCAND = re.compile(r'(illust|illustration|color_)', re.I)
NOPAGE = re.compile(r'page\d+\.clip$', re.I)
EXPECT = {'rev': 13, 'ill': 19, 'done': 759, 'pages': 831}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    before_sc = m25.scoring_population(new)
    before_done = [q for q, v in new.items() if v['use'].startswith('採点する')]

    nrev = 0
    print('== `revision`(v2 が新しい。**作者の推定**)==')
    for ci in REV:
        g = byc[ci]
        v2s = sorted(r['page_path'] for r in g if V2.search(r['page_path']))
        rep = v2s[0] if v2s else None
        for r in g:
            q = r['page_path']
            isv2 = bool(V2.search(q))
            new[q]['version_role'] = 'revision'
            new[q]['decided_by'] = (VERBATIM + ' / **作者の推定(「はず」)。'
                                    'sha256 で検定する(一致=手を入れていない頁)**')
            new[q]['superseded_by'] = ''
            if not isv2:
                new[q]['use'] = '採点しない(改訂の古い側。作者の推定)'
                tag = '旧       '
            elif q == rep:
                new[q]['use'] = '採点する(改訂の新しい側の代表。作者の推定)'
                tag = 'v2(代表) '
            else:
                new[q]['use'] = '採点しない(同じ改訂が別の道にも在る。代表は別)'
                tag = 'v2(重複) '
            nrev += 1
            print(f'  塊{ci:>3} {tag}  {q[14:]}')

    ill = [q for q in new if ILLDIR.search(q)]
    for q in ill:
        # **層の札であって、採点の判断ではない。** 初版は `use` を
        # 「採点する」で上書きして、**内容の重複で外してあった2頁を呼び戻した**
        # (主値が 759 → 761 になって既知セルが捕まえた)。
        # → **`use` は変えず、層だけ足す。**
        new[q]['version_role'] = 'illustration'
        new[q]['decided_by'] = VERBATIM
        new[q]['use'] = (new[q]['use']
                         + ' / 層: 1枚イラスト(原稿の頁ではない)')
    print(f'\n== `illustration`(作者が名指し): **{len(ill)} 頁**'
          f'(`color_comics4`)==')

    print(f'\n== 1枚イラストの候補(**名前では決められない。候補として作者に出す**)==')
    cand_dir = sorted(q for q in new if ILLCAND.search(q) and q not in ill)
    cand_single = sorted(q for q in new if not NOPAGE.search(q))
    print(f'  `illust`/`color_` を道に含む(名指しの19頁を除く): **{len(cand_dir)} 頁**')
    for w, c in collections.Counter(
            q.replace('\\', '/').split('/')[3] for q in cand_dir).most_common():
        print(f'    {w:24s} {c:>3} 頁')
    print(f'  `pageNNNN` でない単独ファイル: **{len(cand_single)} 頁**'
          f'  ← 1枚絵でありうるが、コマのある原稿の可能性もある')

    print(f'\n== **登録集合8ページの素性**(基盤の指摘)==')
    reg = [p['source'] for p in json.load(open(MANIFEST, encoding='utf-8'))['pages']]
    nd = nill = 0
    for p in reg:
        d = bool(ILLCAND.search(p))
        s = not NOPAGE.search(p)
        nill += d
        nd += s
        print(f'  {"illust系" if d else "       ":8s} '
              f'{"単独ファイル" if s else "pageNNNN":12s} {p[14:]}')
    print(f'  → **8ページ中 {nill} ページが `illust*` の道、{nd} ページが単独ファイル。**')
    print(f'  **A0 と標準は、ほぼイラストの集合で較正されている。**')
    print(f'  母集団の側は `illust`/`color_` が {len(cand_dir) + len(ill)} / 872'
          f'、単独ファイルが {len(cand_single)} / 872 ——'
          f' **構成が違う。登録集合の数を母集団に当てるときは層を明記する。**')

    sc = m25.scoring_population(new)
    done = [q for q, v in new.items() if v['use'].startswith('採点する')]
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  採点 {len(before_sc)} -> {len(sc)}  / 主値 {len(before_done)} -> '
          f'{len(done)}  理由: **札を付けただけ。落としていない**'
          f'(`revision` の古い側5頁は元から `superseded` 相当で主値の外)')
    back = [q for q, v in new.items() if v['version_role'] == 'revision']
    print(f'  頁 823 -> {pages}  理由: `revision` の {len(back)} 頁のうち、'
          f'`superseded` だった分が台帳に戻った(改訂の古い側も作者の持つ頁)')
    print(f'  **主値の内訳(層)**: 作者が名指しした1枚イラスト {len(ill)} 頁 / '
          f'候補 {len(cand_dir)} 頁 / 残りが原稿の頁')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-K1 revision の頁', nrev, EXPECT['rev']),
                         ('KC-K2 illustration の頁', len(ill), EXPECT['ill']),
                         ('KC-K3 主値', len(done), EXPECT['done']),
                         ('KC-K3 頁の母集団', pages, EXPECT['pages'])):
        good = got == want
        ok &= good
        print(f'  {k:24s} {got:>5} / 期待 {want:>5}  '
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
    with open(OUT / 'illustration_candidates.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', '根拠', 'いまの version_role'])
        for q in ill:
            w.writerow([q, '作者が名指し(color_comics4 = カラーカット)',
                        new[q]['version_role']])
        for q in cand_dir:
            w.writerow([q, '道に illust/color_ を含む(候補)',
                        new[q]['version_role']])
        for q in cand_single:
            if q not in ill and q not in cand_dir:
                w.writerow([q, 'pageNNNN でない単独ファイル(候補)',
                            new[q]['version_role']])
    print(f'\nwrote {OUT}/version_roles_current.csv / illustration_candidates.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
