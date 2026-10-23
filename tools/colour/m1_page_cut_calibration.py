"""M1: ページ単位選別の器具確認 —— 色画素率の分布と 2% の棒の感度。

読むもの: inbox/work_color_mode_pages.csv (1,348ページ) と inbox/work_color_mode.csv (289作品)。
出すもの: 既知セルの再現、色画素率の分布、棒を動かしたときに側を変えるページ数。

較正できるのはページ側の棒 (2%) だけである。画素側の棒 (彩度 >= 24) はこの在庫からは
較正できない —— ページあたりの彩度ヒストグラムが無い。それは新しい測定の依頼になる。
"""
import csv
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    pages = load('inbox/work_color_mode_pages.csv')
    works = load('inbox/work_color_mode.csv')

    # --- 既知セル: 抽出側の判定を手元から再現する ---------------------------
    by_work = defaultdict(list)
    for r in pages:
        by_work[(r['年'], r['作品'])].append(float(r['色画素率']))
    mism = 0
    for w in works:
        v = by_work[(w['年'], w['作品'])]
        n = sum(1 for x in v if x >= 0.02)
        j = '全カラー' if n == len(v) else ('全モノクロ' if n == 0 else '混在')
        if (len(v), n, j) != (int(w['測定ページ']), int(w['カラーページ']), w['判定']):
            mism += 1
    print(f'known cell: 289作品の判定を色画素率>=0.02から再現 -> mismatch {mism}')
    print('  ' + ' / '.join(f'{k} {v}' for k, v in
                            sorted(Counter(w['判定'] for w in works).items())))
    if mism:
        print('再現しないので他を読まない (規律2)')
        return 1

    rates = sorted(float(r['色画素率']) for r in pages)
    n = len(rates)
    print(f'\npages {n}')

    # --- 分布: 対数側に寄るので十進の帯で印字する ---------------------------
    edges = [0.0, 1e-9, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2,
             1e-1, 2e-1, 5e-1, 1.01]
    print('\n色画素率の帯 (彩度>=24 の画素の割合):')
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for x in rates if lo <= x < hi)
        lab = 'ちょうど0' if hi == 1e-9 else f'[{lo:g}, {hi:g})'
        print(f'  {lab:>18}  {c:5d}  {100*c/n:5.1f}%')

    # --- 棒の感度: 棒を動かすと何ページが側を変えるか -----------------------
    print('\n棒を動かしたときのカラー側のページ数:')
    for bar in [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20]:
        c = sum(1 for x in rates if x >= bar)
        mark = '  <- 抽出側の棒' if abs(bar - 0.02) < 1e-12 else ''
        print(f'  >= {bar:<6g}  {c:5d}  ({100*c/n:5.1f}%){mark}')

    # --- 2% の近傍に何があるか (谷か、それとも体の中か) ---------------------
    near = [x for x in rates if 0.005 <= x <= 0.08]
    print(f'\n2% の近傍 [0.005, 0.08] に {len(near)} ページ:')
    print('  ' + ', '.join(f'{x:.4f}' for x in near))
    return 0


if __name__ == '__main__':
    sys.exit(main())
