"""M20: 基盤の裁定(`ruling_m_three_bands_and_year_20261025.md`)の条件を測る。

条件(基盤。走らせる前にここへ写した):
  (a)② **直すのは4ページではなく分割の規則。** 重複に関わる頁が 248/872 = 28.4% なら
      ページ単位で切る限り再発する。**切る単位は重複の塊で、塊ごと片側に寄せる**
  (a)③ **主値と込みを両方印字し、どちらが主値かを先に書く**
  (c)  **566/872 を使う前に、306 が何で区別されるかを印字する**(作品・年・様式別)。
      標準はクリップの連鎖がある所にだけ在る = **選別**。無作為でなければ、
      566 の上の標準は「**クリップの連鎖を持つページの標準**」であって
      カラー原稿の標準ではない

**期待(走らせる前に書く)**:
  F-a 塊ごとに寄せると、登録集合側に寄る頁は 8 より多い(登録集合の8頁のうち
      4頁が塊に入っている)。検定集合の主値はそのぶん減る
  F-c **306 は無作為ではない**。クリップの連鎖は「部位ごとにシルエットを置く」
      描き方の痕跡なので、**単品イラスト系(`illust` / `skima` / `illustration`)に
      連鎖があり、漫画の頁に無い**側に偏るはず(P21 と同じ向き)

**この器具は腕を採点しない。** 母集団の名前と分母だけを決める。
T95(基盤): **プールを2つ以上と呼ぶたびに、跨いでいる塊の数と頁数を同じ段落に書く。**
"""
import collections
import csv
import hashlib
import re
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
ROLES = Path('results/m19_versions/version_roles.csv')
OUT = Path('results/m20_ruling'); OUT.mkdir(parents=True, exist_ok=True)
PAPER = '1584'
REG8 = 8
ILLUST = re.compile(r'illust|skima|illustration|circlecut|color_', re.I)


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    roles = {r['page_path']: r for r in csv.DictReader(open(ROLES, encoding='utf-8'))}
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    reg = set(pages[:REG8])
    bypage = collections.defaultdict(list)
    for r in rows:
        bypage[r['page_path']].append(r)

    def part(p, i):
        return p.replace('\\', '/').split('/')[i]

    def has_chain(v):
        sib = collections.defaultdict(list)
        for r in v:
            if r['layer_type'] == PAPER or r['parent_id'] == '':
                continue
            sib[r['parent_id']].append(r)
        for g in sib.values():
            g.sort(key=lambda r: int(r['order_in_parent']))
            base = None
            for r in g:
                if r['clip'] == '0':
                    base = r
                elif base is not None:
                    return True
        return False

    chain = {p: has_chain(v) for p, v in bypage.items()}
    no_chain = [p for p in pages if not chain[p]]
    print(f'== (c) クリップの連鎖が無い {len(no_chain)} / {len(pages)} '
          f'= {len(no_chain)/len(pages):.3f} の内訳 ==')

    print('\n  年フォルダ別(連鎖なし / 全体):')
    for y, n in sorted(collections.Counter(part(p, 2) for p in pages).items()):
        k = sum(1 for p in no_chain if part(p, 2) == y)
        print(f'    {y:>6} {k:>4} / {n:>4} = {k/n:.3f}')

    print('\n  様式別(パスに illust/skima/illustration/circlecut/color_ を含むか):')
    for lab, sel in (('単品イラスト系', True), ('それ以外(漫画の頁ほか)', False)):
        tot = [p for p in pages if bool(ILLUST.search(p)) == sel]
        k = [p for p in tot if not chain[p]]
        print(f'    {lab:<22} {len(k):>4} / {len(tot):>4} = '
              f'{len(k)/len(tot) if tot else 0:.3f}')

    print('\n  作品単位でまとまっているか(= 頁の性質か作品の性質か):')
    byw = collections.defaultdict(list)
    for p in pages:
        byw[(part(p, 2), part(p, 3))].append(p)
    allno = [w for w, ps in byw.items() if all(not chain[p] for p in ps)]
    allyes = [w for w, ps in byw.items() if all(chain[p] for p in ps)]
    mixed = [w for w, ps in byw.items() if w not in allno and w not in allyes]
    print(f'    作品 {len(byw)}: 全頁が連鎖なし **{len(allno)}** / '
          f'全頁が連鎖あり {len(allyes)} / 混在 {len(mixed)}')
    print(f'    連鎖なし306頁のうち、「全頁が連鎖なし」の作品に属する頁: '
          f'**{sum(len(byw[w]) for w in allno)}**')
    print(f'    混在作品の例(頁数>=6): ' + ', '.join(
        f'{w[0]}/{w[1]}({sum(1 for p in byw[w] if not chain[p])}/{len(byw[w])})'
        for w in mixed if len(byw[w]) >= 6)[:300])

    print('\n  層数と色の量(連鎖なし 対 連鎖あり。中央値):')
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs)//2] if xs else 0
    for lab, sel in (('連鎖なし', False), ('連鎖あり', True)):
        ps = [p for p in pages if chain[p] == sel]
        print(f'    {lab:<8} 頁 {len(ps):>4}  層数 {med([len(bypage[p]) for p in ps]):>4}  '
              f'描画層 {med([sum(1 for r in bypage[p] if r["folder"] != "True" and (num(r["drawn_ratio"],0) or 0) > 0) for p in ps]):>4}  '
              f'形式5の層 {med([sum(1 for r in bypage[p] if r["formats"] == "5") for p in ps]):>4}')

    print(f'\n== (a)② 塊ごとに寄せた分割 ==')
    # 指紋の塊(m19 と同じ規則)
    def sig(r):
        return hashlib.md5('|'.join((
            r['name'], r['layer_type'], r['drawn_ratio'], r['sat_mean'],
            r['top_colour_share'], r['colours_q16'])).encode()).hexdigest()[:16]
    cid = {p: roles[p]['dup_cluster_id'] for p in pages}
    reg_clusters = {cid[p] for p in reg if cid[p]}
    pulled = [p for p in pages if cid[p] and cid[p] in reg_clusters and p not in reg]
    print(f'  登録集合8頁のうち塊に入っている頁: '
          f'{sum(1 for p in reg if cid[p])}  '
          f'その塊: {sorted(reg_clusters)}')
    print(f'  **塊ごと登録側に寄せると、検定集合から外れる頁 {len(pulled)}**')
    for p in pulled:
        print(f'    塊{cid[p]:>4} {p.replace(chr(92), "/")[14:]} '
              f'({roles[p]["version_role"]})')
    sup = [p for p in pages if roles[p]['version_role'] in
           ('superseded', 'duplicate_content')]
    test_all = [p for p in pages if p not in reg]
    test_primary = [p for p in test_all if p not in set(pulled)
                    and p not in set(sup)]
    test_incl = [p for p in test_all if p not in set(sup)]
    print(f'\n  **主値(先に書く): 検定集合 = {len(test_primary)} 頁**'
          f'(872 − 登録8 − 置き換え/重複内容 {len(sup)} − 塊で寄せた {len(pulled)}'
          f' の重なりを除いた数)')
    print(f'  込み(併記): 塊で寄せずに4組の双子を残す場合 = {len(test_incl)} 頁')
    print(f'  うちクリップの連鎖があり標準が作れる頁: '
          f'主値 {sum(1 for p in test_primary if chain[p])} / '
          f'込み {sum(1 for p in test_incl if chain[p])}')

    # T95: プールを2つと呼ぶので、跨いでいる塊と頁数を同じ段落に書く
    straddle = [c for c in {cid[p] for p in pages if cid[p]}
                if len({(p in reg) for p in pages if cid[p] == c}) > 1]
    print(f'\n  T95: この2つ(登録集合 / 検定集合)を跨ぐ塊 **{len(straddle)}**、'
          f'その頁 **{sum(1 for p in pages if cid[p] in set(straddle))}**')

    with open(OUT / 'split.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', 'in_registration', 'dup_cluster_id',
                    'version_role', 'has_clip_chain', 'in_test_primary'])
        for p in pages:
            w.writerow([p, p in reg, cid[p], roles[p]['version_role'],
                        chain[p], p in set(test_primary)])
    print(f'\nwrote {OUT/"split.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
