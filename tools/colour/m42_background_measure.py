"""M42: 作者の言明「**共通して背景がない**」を**量で**測る。

**定義を先に書く(走らせる前)**:

> **背景の量 = 用紙でもクリップでもない描画層のうち、`drawn_ratio` が最大のもの。**
> (= **紙面をいちばん広く覆っている1枚**。背景が在ればそれが大きく、無ければ小さい)

**初版の定義2つは外れた。記録する**:

1. **「z順で下から3枚の最大」** —— 中央 0.057 にしかならず、背景を捉えていなかった。
   漫画の頁では紙の白地が**描画層ではない**(用紙層)ため、下の方に居るのは小さい部品だった
2. **比較の母集団を「作品フォルダーの `pageNNNN`」に限った** ——
   **`done*` のファイルは `ako6-29.clip` / `road03.clip` のような名前で、`pageNNNN` ではない。**
   **結果、`done*` が0頁になって比較が成立しなかった**(交絡を除くつもりで対象を全部消した)
   → **1枚絵(`illust*`/`color_`)だけを除く**形に直した

**クリップ層を外す理由**: **T-M3**(クリップ層のアルファは全キャンバスを申告する)。
入れると「背景」が全頁に在ることになる。

**なぜこの量か**: 背景は**紙面の大半を覆い**、**層の重なりの一番下に居る**。
**名前で `背景`/`BG` を探さない** —— `下絵` が 823頁中800頁に在った(名前は6つめの落ちた錨)
のと同じ失敗になるため(基盤の指示)。

**交絡を先に除く(基盤の指摘)**: **1枚絵は背景が無いのが普通**なので、
**`illustration` と `illust*`/`color_` 系、`pageNNNN` でない単独ファイルを除いた
「作品フォルダーの `pageNNNN`」だけで比べる。**
さもないと**イラストを全部「マージ前」として拾う。**

**2つの使い道**:

| | 内容 |
|---|---|
| **(a) 作者の言明の検定** | `done*` の95頁に背景が無いことを確かめる。**外れた頁が在れば、それは問い** |
| **(b) 候補生成器** | **背景が無い頁は `done*` に居なくてもマージ前かもしれない**(名前でなく性質で引ける初めての道) |

**どちらでも印字する。** 棒はまだ引かない —— **分布を両方見てから**(`rough` の篩で
300倍の隔たりが見えたのと同じ手順)。
"""
import collections
import csv
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402

SRC = Path('results/m41_all_done/version_roles_current.csv')
OUT = Path('results/m42_background'); OUT.mkdir(parents=True, exist_ok=True)
PAPER = '1584'
DONE = re.compile(r'\\done[^\\]*\\', re.I)
ILL = re.compile(r'(illust|illustration|color_)', re.I)
NOPAGE = re.compile(r'page\d+\.clip$', re.I)
N_BOTTOM = 3


def main():
    roles = {r['page_path']: r for r in csv.DictReader(open(SRC, encoding='utf-8'))}
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    bp = collections.defaultdict(list)
    for r in lay:
        bp[r['page_path']].append(r)

    def bg(p):
        rows = bp[p]
        roots, kids = m16.build_tree(rows)
        z = m16.z_order(roots, kids)
        drawn = [r for r in rows
                 if r['folder'] != 'True' and r['layer_type'] != PAPER
                 and (m19.num(r['drawn_ratio'], 0) or 0) > 0]
        if not drawn:
            return None
        nonclip = [r for r in drawn if r['clip'] != '1']
        if not nonclip:
            return None
        return max((m19.num(r['drawn_ratio'], 0) or 0) for r in nonclip)

    vals = {}
    for p in bp:
        v = bg(p)
        if v is not None:
            vals[p] = v

    manga = [p for p in vals
             if not ILL.search(p)
             and roles[p]['version_role'] not in ('superseded',)]
    d_done = sorted(vals[p] for p in manga if DONE.search(p))
    d_fin = sorted(vals[p] for p in manga
                   if not DONE.search(p)
                   and roles[p]['use'].startswith('採点する'))

    def desc(name, xs):
        if not xs:
            print(f'  {name:34s} 0 頁')
            return
        q = statistics.quantiles(xs, n=4) if len(xs) > 3 else [xs[0]] * 3
        print(f'  {name:34s} {len(xs):>4} 頁  '
              f'最小 {xs[0]:.3f} / 下位25% {q[0]:.3f} / 中央 '
              f'{statistics.median(xs):.3f} / 上位25% {q[2]:.3f} / 最大 {xs[-1]:.3f}')

    print('== 背景の量(z順で下から3枚の描画層の drawn_ratio の最大値)==')
    print('  **1枚絵(`illust*`/`color_`)を除いた頁だけ**(交絡。'
          '`done*` のファイル名は `pageNNNN` ではないので名前では絞らない)')
    desc('`done*` の頁(作者=背景がない)', d_done)
    desc('それ以外の採点対象の頁', d_fin)
    if d_done and d_fin:
        print(f'\n  **中央の比: {statistics.median(d_fin) / max(1e-9, statistics.median(d_done)):.1f}倍**'
              f'(それ以外 / `done*`)')
        over = [p for p in manga if DONE.search(p)
                and vals[p] >= statistics.median(d_fin)]
        print(f'  **`done*` なのに「それ以外」の中央以上に背景が在る頁: {len(over)}**'
              f' —— **外れた頁は問い**')
        for p in sorted(over, key=lambda p: -vals[p])[:10]:
            print(f'    {vals[p]:.3f}  {p[14:]}')
        low = sorted((vals[p], p) for p in manga
                     if not DONE.search(p)
                     and roles[p]['use'].startswith('採点する')
                     and vals[p] <= (max(d_done) if d_done else 0))
        print(f'\n  **(b) `done*` の外で、背景の量が `done*` の最大値以下の頁: '
              f'{len(low)}** —— **候補(名前でなく性質で引いた初めての集合)**')
        for v, p in low[:12]:
            print(f'    {v:.3f}  {roles[p]["version_role"]:>9}  {p[14:]}')
    with open(OUT / 'background_measure.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', '背景の量', 'done*か', 'version_role', 'use'])
        for p in sorted(manga, key=lambda p: vals[p]):
            w.writerow([p, f'{vals[p]:.6f}', 'yes' if DONE.search(p) else '',
                        roles[p]['version_role'], roles[p]['use']])
    print(f'\nwrote {OUT}/background_measure.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
