"""追補232 §B: W(τ=6.0)の標本外の頁を引く。census の材料表から、トーンを除いた材料 ≥ 3 の頁を seed 0 で 8頁。

落としたものは理由別に印字する(規律9)。恒久除外は census 側で外れているが、ここでも名簿で門を当てる。
"""
import csv
from pathlib import Path

import numpy as np

CEN = Path('inbox/mask_material_pages_20261107.csv')
EXCL = Path('../lineart-census/data/census_full/author_exclusions.csv')
FOLD = Path('results/m79_folded_population/folded_population.csv')
OUT = Path('results/m138_holdout_pages')
IN_SAMPLE = ('sunshine2\\page0002.clip', 'gakusai3\\新規キャンバス.clip')
N, SEED, MIN_MAT = 8, 0, 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CEN, encoding='utf-8-sig')))
    excl = {r['page_path'] for r in csv.DictReader(open(EXCL, encoding='utf-8-sig'))}
    fold = {r['page_path'] for r in csv.DictReader(open(FOLD, encoding='utf-8-sig'))}
    if not excl:
        raise SystemExit('**除外の名簿が空。止まる**')
    print(f'census の表 {len(rows)} 頁 / 除外の名簿 {len(excl)} / M の畳んだ母集団 {len(fold)}')
    drop, pool = {}, []
    for r in rows:
        nm = int(r['n_material']) - int(r['n_material_tone'])
        why = ('恒久除外' if r['page_path'] in excl else
               '標本内の2頁' if r['page_path'].endswith(IN_SAMPLE) else
               'トーンを除いた材料 < 3' if nm < MIN_MAT else
               '畳んだ母集団に無い(版の重複)' if r['page_path'] not in fold else None)
        if why:
            drop[why] = drop.get(why, 0) + 1
        else:
            pool.append(r)
    print('落とした(最初に当たった理由1つ): ' + ' / '.join(f'{k} {v}' for k, v in drop.items()))
    print(f'母集団 {len(pool)} 頁')
    pool.sort(key=lambda r: r['page_path'])
    pick = sorted(np.random.default_rng(SEED).permutation(len(pool))[:N].tolist())
    with open(OUT / 'drawn_pages.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['no', 'page_path', 'year_label', 'n_material', 'n_material_tone', 'n_line_layers', 'mask_raster_status'])
        for k, i in enumerate(pick, 1):
            r = pool[i]
            w.writerow([k, r['page_path'], r['year_label'], r['n_material'], r['n_material_tone'], r['n_line_layers'], r['mask_raster_status']])
            print(k, r['year_label'], r['n_material'], r['n_material_tone'], r['n_line_layers'], r['mask_raster_status'], r['page_path'])


if __name__ == '__main__':
    main()
