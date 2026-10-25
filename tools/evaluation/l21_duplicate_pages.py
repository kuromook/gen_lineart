# -*- coding: utf-8 -*-
"""L21: 納品の中に**同じ頁が二状態**で入っていないかを数える (2026-10-25)。

層の PNG を md5 で突き合わせる。**画素を読まない** —— 一致はバイト一致なので、
閾値も器具の偏りも無い。頁の近さは層の md5 集合の Jaccard。

  python tools/evaluation/l21_duplicate_pages.py [--thr 0.5]
"""
import argparse, collections, hashlib, itertools, json, os, re, sys

BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l21_dirtiness'


def hashes():
    pg = collections.defaultdict(dict)
    for work in ('ako6', 'ako7'):
        d = f'{BASE}/{work}/all_layers'
        for fn in sorted(os.listdir(d)):
            m = re.match(r'(.+)_L(\d+)\.png$', fn)
            if not m:
                continue
            h = hashlib.md5(open(f'{d}/{fn}', 'rb').read()).hexdigest()
            pg[m.group(1)][int(m.group(2))] = h
    return pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=float, default=0.5)
    a = ap.parse_args()
    pg = hashes()
    print(f'頁 {len(pg)}  層 {sum(len(v) for v in pg.values())}')
    sets = {k: set(v.values()) for k, v in pg.items()}
    pairs = []
    for x, y in itertools.combinations(sorted(sets), 2):
        i = len(sets[x] & sets[y])
        if not i:
            continue
        j = i / len(sets[x] | sets[y])
        pairs.append((j, i, len(sets[x]), len(sets[y]), x, y))
    pairs.sort(reverse=True)
    hi = [p for p in pairs if p[0] >= a.thr]
    inv = {p for _, _, _, _, x, y in hi for p in (x, y)}
    print(f'層が1枚でも byte 一致する頁の組 {len(pairs)}')
    print(f'**Jaccard >= {a.thr} の組 {len(hi)}  関わる頁 {len(inv)} / {len(pg)}**'
          f'  (実質の別頁 およそ {len(pg) - len(hi)})')
    for j, i, nx, ny, x, y in hi:
        print(f'  {j:.3f}  共通 {i:>3} / {nx}+{ny}   {x}  <->  {y}')
    json.dump(dict(thr=a.thr, n_pages=len(pg),
                   pairs=[dict(jaccard=round(j, 4), common=i, a=x, b=y) for
                          j, i, _, _, x, y in hi],
                   pages=sorted(inv)),
              open(f'{OUT}/duplicate_pages.json', 'w'), ensure_ascii=False, indent=1)
    print(f'-> {OUT}/duplicate_pages.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
