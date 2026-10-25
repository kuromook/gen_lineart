"""M22: **全頁がクリップの連鎖を持たない作品の一覧**を出す(基盤の裁定 §3 の追加条件)。

基盤: 「**全頁連鎖なしの42作品(306頁中115頁)の一覧を標準の隣に**。
作品まるごと欠ける選別は後から効きます。」

標準は**クリップの連鎖がある頁にだけ在る**(追補15 §N2)。
連鎖の有無がページの性質であれば作品の中で混ざるが、**作品まるごと欠けるなら、
その作品はこの標準からは一度も見えない。** その一覧を標準の隣に置く。

**この器具は腕を採点しない。** 一覧と、その作品が何であるか(頁数・層・色の量)を出す。
"""
import collections
import csv
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
OUT = Path('results/m22_chainless'); OUT.mkdir(parents=True, exist_ok=True)
PAPER = '1584'


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
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
    byw = collections.defaultdict(list)
    for p in bypage:
        byw[(part(p, 2), part(p, 3))].append(p)
    allno = {w: ps for w, ps in byw.items() if all(not chain[p] for p in ps)}
    allyes = {w: ps for w, ps in byw.items() if all(chain[p] for p in ps)}
    mixed = {w: ps for w, ps in byw.items()
             if w not in allno and w not in allyes}

    def stats(ps):
        lay = [len(bypage[p]) for p in ps]
        f5 = [sum(1 for r in bypage[p] if r['formats'] == '5') for p in ps]
        dr = [sum(num(r['drawn_ratio'], 0) or 0 for r in bypage[p]
                  if r['folder'] != 'True') for p in ps]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
        return med(lay), med(f5), med(dr)

    print(f'== 全頁がクリップの連鎖を持たない作品: **{len(allno)}** '
          f'(頁 {sum(len(ps) for ps in allno.values())}) ==')
    print(f'  対照: 全頁が連鎖あり {len(allyes)} 作品 '
          f'({sum(len(ps) for ps in allyes.values())} 頁) / '
          f'混在 {len(mixed)} 作品 ({sum(len(ps) for ps in mixed.values())} 頁)')
    print(f'\n{"年":>6} {"作品":<18} {"頁":>4} {"層数(中央)":>10} '
          f'{"形式5の層":>9} {"描画比の和":>10}')
    for w, ps in sorted(allno.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ml, mf, md = stats(ps)
        print(f'{w[0]:>6} {w[1][:18]:<18} {len(ps):>4} {ml:>10} {mf:>9} {md:>10.2f}')
    ml, mf, md = stats([p for ps in allno.values() for p in ps])
    print(f'{"":>6} {"— 中央(42作品)":<18} '
          f'{sum(len(ps) for ps in allno.values()):>4} {ml:>10} {mf:>9} {md:>10.2f}')
    ml, mf, md = stats([p for ps in allyes.values() for p in ps])
    print(f'{"":>6} {"— 中央(連鎖あり作品)":<18} '
          f'{sum(len(ps) for ps in allyes.values()):>4} {ml:>10} {mf:>9} {md:>10.2f}')

    with open(OUT / 'chainless_works.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w2 = csv.writer(fh)
        w2.writerow(['年', '作品', '頁数', '層数の中央', '形式5の層の中央',
                     '描画比の和の中央', 'page_path の例'])
        for w, ps in sorted(allno.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            ml, mf, md = stats(ps)
            w2.writerow([w[0], w[1], len(ps), ml, mf, round(md, 3),
                         sorted(ps)[0]])
    print(f'\nwrote {OUT/"chainless_works.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
