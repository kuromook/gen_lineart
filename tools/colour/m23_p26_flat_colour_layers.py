"""M23: **P26 を採点する**(追補16 §O6 で、測る前に登録した)。

P26: 作者の3択のうち **(1) 塗りの層そのものが面** が当たっているなら、
**連鎖なし42作品にも「平坦なカラー層」が同じくらいある**。
棒: 形式5・`clip=0` の層のうち `top_colour_share >= 0.99` の割合が **>= 0.25**
(登録集合8ページでは 42/114 = **0.368**)。

**S2 の切り口は追補5 §E8 で登録済みのものをそのまま使う**(形式5 / `clip=0` / 0.99)。
**この器具は腕を採点しない。作者の3択も縮めない。材料であって判定ではない。**

既知セル: **登録集合8ページで 42/114 = 0.368 を再現すること。**
再現しなければ以降を読まない(切り口が同じものであることの確認)。
"""
import collections
import csv
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
PAPER = '1584'
FLAT = 0.99
REG8 = 8
KC = (42, 114)


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
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    reg = set(pages[:REG8])

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

    def s2_pool(ps):
        """形式5・clip=0・描画>0 の層(= 汚染されていないカラー層)。"""
        out = []
        for p in ps:
            for r in bypage[p]:
                if (r['formats'] == '5' and r['clip'] == '0'
                        and (num(r['drawn_ratio'], 0) or 0) > 0):
                    out.append(r)
        return out

    def rate(pool):
        flat = [r for r in pool
                if (num(r['top_colour_share'], 0) or 0) >= FLAT]
        return len(flat), len(pool), (len(flat) / len(pool) if pool else 0.0)

    k = rate(s2_pool(reg))
    ok = (k[0], k[1]) == KC
    print(f'既知セル(登録集合8ページ): {k[0]}/{k[1]} = {k[2]:.3f} '
          f'(期待 {KC[0]}/{KC[1]} = {KC[0]/KC[1]:.3f}) -> '
          f'{"PASS" if ok else "FAIL"}')
    if not ok:
        print('切り口が同じものでない。以降は読まない。')
        return 1

    byw = collections.defaultdict(list)
    for p in pages:
        byw[(part(p, 2), part(p, 3))].append(p)
    allno = {w: ps for w, ps in byw.items() if all(not chain[p] for p in ps)}
    allyes = {w: ps for w, ps in byw.items() if all(chain[p] for p in ps)}

    print(f'\n== P26: 平坦なカラー層(形式5・clip=0・top_colour_share >= {FLAT}) ==')
    rows_out = []
    for lab, works in (('**全頁 連鎖なし(42作品)**', allno),
                       ('全頁 連鎖あり(37作品)', allyes)):
        ps = [p for w, l in works.items() for p in l]
        f, n, rt = rate(s2_pool(ps))
        rows_out.append((lab, len(works), len(ps), f, n, rt))
        print(f'  {lab:<26} 作品{len(works):>3} 頁{len(ps):>4}  '
              f'平坦 {f:>5} / カラー層 {n:>5} = **{rt:.3f}**')
    rt = rows_out[0][5]
    print(f'\n  棒 >= 0.25 -> **{"PASS" if rt >= 0.25 else "FAIL"}**'
          f'(登録集合 {KC[0]/KC[1]:.3f})')

    print('\n== 例外として挙げた作品ごと(§O2 の表) ==')
    print(f'  {"年":>6} {"作品":<14} {"頁":>3} {"カラー層":>8} {"平坦":>6} {"割合":>7}')
    for w in (('2017', 'nuru'), ('2017', 'NY'), ('2017', 'mars'),
              ('2021', 'nanami'), ('2016', 'poolsideIdle'),
              ('2018', 'ako7')):
        if w not in byw:
            continue
        f, n, rt2 = rate(s2_pool(byw[w]))
        print(f'  {w[0]:>6} {w[1]:<14} {len(byw[w]):>3} {n:>8} {f:>6} '
              f'{rt2:>7.3f}')

    print('\n== 対照: 平坦でないカラー層は何なのか(連鎖なし42作品) ==')
    pool = s2_pool([p for l in allno.values() for p in l])
    notflat = [r for r in pool if (num(r['top_colour_share'], 0) or 0) < FLAT]
    g = sum(1 for r in notflat if r['has_gradation_fill'] == 'True')
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    print(f'  平坦でない {len(notflat)}: `has_gradation_fill` 真 {g} '
          f'({g/len(notflat) if notflat else 0:.3f})  '
          f'`colours_q16` の中央 {med([num(r["colours_q16"], 0) for r in notflat]):.0f}  '
          f'`top_colour_share` の中央 '
          f'{med([num(r["top_colour_share"], 0) for r in notflat]):.4f}')
    print('\n作者の3択は縮めない。これは材料であって判定ではない。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
