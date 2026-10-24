# -*- coding: utf-8 -*-
"""L13 本体: 作者のベタ領域の輪郭は、どれだけ線に裏打ちされているか。

登録は doc/work_log.md「L13 事前登録」(commit 216188f)、P19' の修正も同節 §7(回す前)。
既知セル l13_known_cells.py が PASS しなければ起動しない。

  barrier_line := ラスタの墨 ∧ ¬truth4      ベタの塊だけ引く。細い印(隙間埋め)は残す
  support(R)   := |∂R のうち barrier_line から 2px 以内| / |∂R|
  open_runs(R) := support されない輪郭画素の連結成分の数
  bite(R)      := |R ∧ barrier_line| / |∂R|   (工程3の食い込み厚 px 相当)
  P19' 対照    := R を同じコマ内でランダム平行移動して同じ器具で測る
"""
import csv, os, subprocess, sys, collections, statistics as st
import numpy as np, cv2

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import ink_of
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l13_known_cells import boundary, support_of, dist_to, MINREG, SUP_PX

TRUTH = os.environ['L13_TRUTH']
KOMA  = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12   = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT   = '/home/sh1/deepl/lineart-fill-region/results/l13_boundary_support'
SEED  = 20261024


def gate():
    r = subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'l13_known_cells.py')],
                       env={**os.environ}, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print('既知セルが FAIL。本体を起動しない(登録 §4)。'); sys.exit(1)


def main():
    gate()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    l12 = list(csv.DictReader(open(L12)))
    rows = []
    cache = {}
    for i, pr in enumerate(l12, 1):
        m = man[pr['name']]
        pid = m['pid'].replace('__', '_'); w = pr['work']
        if pid not in cache:
            cache.clear()
            cache[pid] = cv2.imread(os.path.join(TRUTH, f'{w}_{pid}_t4.png'),
                                    cv2.IMREAD_GRAYSCALE) > 127
        t4p = cache[pid]
        x0, y0, x1, y1 = (int(m[k]) for k in ('x0', 'y0', 'x1', 'y1'))
        g = cv2.imread(f'{KOMA}/line/{pr["name"]}', cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        t4 = t4p[y0:y1, x0:x1]
        if ink.shape != t4.shape:
            continue
        barrier = ink & ~t4
        dist = dist_to(barrier)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(t4.astype(np.uint8), 8)
        H, W = t4.shape
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < MINREG:
                continue
            bx, by = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            cy0, cy1 = max(0, by - 1), min(H, by + bh + 1)
            cx0, cx1 = max(0, bx - 1), min(W, bx + bw + 1)
            Rc = (lab[cy0:cy1, cx0:cx1] == k)
            s = support_of(Rc, barrier[cy0:cy1, cx0:cx1],
                           dist=(None if dist is None else dist[cy0:cy1, cx0:cx1]))
            if s is None:
                continue
            # P19' 平行移動の対照: bbox をコマ内のランダムな位置へ(同じ bbox 寸法で切って測る)
            sh = None
            if bw + 2 < W and bh + 2 < H:
                patch = (lab[by:by + bh, bx:bx + bw] == k)
                for _ in range(8):
                    nx = int(rng.integers(1, W - bw - 1)); ny = int(rng.integers(1, H - bh - 1))
                    if abs(nx - bx) + abs(ny - by) < 0.2 * (W + H):
                        continue
                    Rs = np.zeros((bh + 2, bw + 2), bool); Rs[1:-1, 1:-1] = patch
                    sh = support_of(Rs, barrier[ny - 1:ny + bh + 1, nx - 1:nx + bw + 1],
                                    dist=(None if dist is None else
                                          dist[ny - 1:ny + bh + 1, nx - 1:nx + bw + 1]))
                    break
            rows.append(dict(name=pr['name'], work=w, pid=pid, region=k,
                             area=int(stats[k, cv2.CC_STAT_AREA]),
                             panel_area=H * W, leak_all=float(pr['leak_all']),
                             l7_stratum=pr['l7_stratum'],
                             bnd_px=s['bnd_px'], support=round(s['support'], 6),
                             open_runs=s['open_runs'], bite=round(s['bite'], 4),
                             shuf_support=(round(sh['support'], 6) if sh else ''),
                             shuf_open_runs=(sh['open_runs'] if sh else '')))
        if i % 25 == 0:
            print(f'  {i}/{len(l12)} コマ  領域 {len(rows)}', flush=True)
    with open(f'{OUT}/per_region.csv', 'w', newline='', encoding='utf-8') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f'-> {OUT}/per_region.csv  領域 {len(rows)}')
    report(rows)


def band(v):
    return '<0.10' if v < 0.10 else ('0.10-0.50' if v < 0.50 else '>=0.50')


def report(rows):
    print('\n' + '=' * 76)
    print('P16 support の中央 (棒 >= 0.80) / P17 open_runs 中央 <=8 かつ open_frac 中央 <=0.20')
    print(f'{"作品":5} {"漏れ":10} {"領域":>5} {"support中央":>11} {"open_runs中央":>13} '
          f'{"open_frac中央":>13} {"bite中央":>9}')
    for w in ('ako6', 'ako7'):
        for b in ('<0.10', '0.10-0.50', '>=0.50'):
            s = [r for r in rows if r['work'] == w and band(r['leak_all']) == b]
            if not s:
                continue
            sup = st.median([r['support'] for r in s])
            print(f'{w:5} {b:10} {len(s):5d} {sup:11.4f} '
                  f'{st.median([r["open_runs"] for r in s]):13.1f} '
                  f'{1-sup:13.4f} {st.median([r["bite"] for r in s]):9.2f}')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w]
        sup = st.median([r['support'] for r in s])
        p16 = sup >= 0.80
        orn = st.median([r['open_runs'] for r in s])
        p17 = orn <= 8 and (1 - sup) <= 0.20
        print(f'  {w} 全体: n={len(s)} support中央 {sup:.4f} -> P16 {"PASS" if p16 else "FAIL"}'
              f' / open_runs中央 {orn:.1f} open_frac中央 {1-sup:.4f}'
              f' -> P17 {"PASS" if p17 else "FAIL"}')
        print(f'      P18 bite 中央 {st.median([r["bite"] for r in s]):.2f} px相当 (棒 1.0-3.0)'
              f'  -> {"PASS" if 1.0 <= st.median([r["bite"] for r in s]) <= 3.0 else "FAIL"}')
    print('\nP19\' 平行移動の対照 (棒: 本来 - 平行移動 >= 0.20)')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w and r['shuf_support'] != '']
        if not s:
            continue
        a = st.median([r['support'] for r in s]); b = st.median([r['shuf_support'] for r in s])
        print(f'  {w}: n={len(s)}  本来 {a:.4f} / 平行移動 {b:.4f} / 差 {a-b:.4f}'
              f'  -> {"PASS" if a-b >= 0.20 else "FAIL"}')
    print('\nsupport の分布 (0.0->1.0 の10分位):')
    for w in ('ako6', 'ako7'):
        h = [0] * 10
        for r in rows:
            if r['work'] == w:
                h[min(9, int(r['support'] * 10))] += 1
        print(f'  {w}: {h}')
    print('\nsupport が低い領域の上位 (規律6: 原寸で見る対象):')
    for r in sorted(rows, key=lambda r: r['support'])[:8]:
        print(f'  support {r["support"]:.3f} open_runs {r["open_runs"]:4d} '
              f'面積 {r["area"]:8,} bite {r["bite"]:5.2f}  {r["name"][5:44]} #{r["region"]}')


if __name__ == '__main__':
    main()
