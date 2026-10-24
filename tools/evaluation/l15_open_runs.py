# -*- coding: utf-8 -*-
"""L15 本体: 開いた区間は「線の切れ目」か「線が最初から無い」のか。
登録は doc/work_log.md「L15 事前登録」(commit 8602482)。門が PASS しなければ起動しない。"""
import csv, collections, os, subprocess, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A
from l14_arms import acc_add
from l15_known_cells import runs_of, endpoints, SUP_PX, END_PX

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
DIAG = '/home/sh1/deepl/lineart-fill-region/results/l14_arms/diagnosis.csv'
ATTR = '/home/sh1/deepl/lineart-fill-region/results/l14_arms/attribution.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_open_runs'
SEED = 20261026


def dmaps(barrier, shape):
    d = (cv2.distanceTransform((~barrier).astype(np.uint8), cv2.DIST_L2, 3)
         if barrier.any() else np.full(shape, 1e9, np.float32))
    e = endpoints(barrier)
    de = (cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
          if e.any() else np.full(shape, 1e9, np.float32))
    return d, de, int(e.sum())


def one_page(job):
    pid, work, fl, ll, panels = job
    F = G = B = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    rng = np.random.default_rng(SEED + abs(hash(pid)) % 100000)
    rows = []
    for nm, x0, y0, x1, y1 in panels:
        f, g, b = F[y0:y1, x0:x1], G[y0:y1, x0:x1], B[y0:y1, x0:x1]
        glab, gids, gstat = comps(f, 8)
        if not gids:
            continue
        H, W = f.shape
        maps = {'line': dmaps(b, f.shape), 'line+gap': dmaps(b | g, f.shape)}
        for gi in gids:
            yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            a0, a1 = max(0, yy - 6), min(H, yy + hh + 6)
            c0, c1 = max(0, xx - 6), min(W, xx + ww + 6)
            R = glab[a0:a1, c0:c1] == gi
            for var, (d, de, ne) in maps.items():
                bar = (b | g) if var == 'line+gap' else b
                for u in runs_of(R, bar[a0:a1, c0:c1],
                                 d=d[a0:a1, c0:c1], de=de[a0:a1, c0:c1]):
                    rows.append(dict(name=nm, work=work, region=gi, variant=var,
                                     arm='real', area=int(gstat[gi, A]), **u))
            # P27 対照: R をコマ内のランダムな位置へ。障壁は本物のまま
            if ww + 2 < W and hh + 2 < H:
                patch = (glab[yy:yy + hh, xx:xx + ww] == gi)
                for _ in range(16):
                    nx = int(rng.integers(1, W - ww - 1)); ny = int(rng.integers(1, H - hh - 1))
                    if abs(nx - xx) + abs(ny - yy) < 0.10 * (W + H):
                        continue
                    Rs = np.zeros((hh + 2, ww + 2), bool); Rs[1:-1, 1:-1] = patch
                    d, de, _ = maps['line']
                    for u in runs_of(Rs, b[ny - 1:ny + hh + 1, nx - 1:nx + ww + 1],
                                     d=d[ny - 1:ny + hh + 1, nx - 1:nx + ww + 1],
                                     de=de[ny - 1:ny + hh + 1, nx - 1:nx + ww + 1]):
                        rows.append(dict(name=nm, work=work, region=gi, variant='line',
                                         arm='shuffled', area=int(gstat[gi, A]), **u))
                    break
    return rows


def gate(rows=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'l15_known_cells.py')],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip()[-1500:])
    if r.returncode != 0:
        print('既知セル(合成)が FAIL。本体を起動しない(登録 §5)。'); sys.exit(1)


def main():
    gate()
    os.makedirs(OUT, exist_ok=True)
    fill, line = gated_layers()
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    panels = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        panels[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = [(pid, w, sorted(fl), sorted(line.get((pid, w), [])), panels[pid])
            for (pid, w), fl in sorted(fill.items()) if pid in panels]
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  区間 {len(rows)}', flush=True)
    # ---- KC15 データ経路。腕の数値を1つも印字する前に見る
    got = sum(1 for r in rows if r['arm'] == 'real' and r['variant'] == 'line+gap')
    want = sum(int(r['open_runs']) for r in csv.DictReader(open(DIAG)))
    ok = got == want
    print(f'\nKC15 データ経路: line+gap の区間 {got} / L14 diagnosis.csv の open_runs 合計 '
          f'{want}  {"PASS" if ok else "FAIL"}')
    with open(f'{OUT}/per_run.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/per_run.csv  {len(rows)} 区間')
    if not ok:
        print('KC15 FAIL。分類の数値は読まない(登録 §5)。'); sys.exit(1)
    report(rows)


def tab(rows, label):
    n = len(rows)
    if not n:
        print(f'  {label}: 区間なし'); return None
    c = collections.Counter(r['kind'] for r in rows)
    den = c['E'] + c['H'] + c['F']
    e = c['E'] / den if den else 0.0
    f = c['F'] / den if den else 0.0
    wds = [r['width'] for r in rows if r['kind'] != 'O']
    print(f'  {label:26} 区間 {n:6d}  E {c["E"]:6d} ({e:.4f})  H {c["H"]:6d}  '
          f'F {c["F"]:6d} ({f:.4f})  O {c["O"]:5d} ({c["O"]/n:.3f})  '
          f'幅中央 {st.median(wds) if wds else 0:6.1f}px  長さ中央 '
          f'{st.median([r["length"] for r in rows]):6.1f}px')
    return dict(E=e, F=f, O=c['O'] / n, width=(st.median(wds) if wds else 0.0), n=n)


def report(rows):
    attr = {(r['name'], r['region']): r['layer_name']
            for r in csv.DictReader(open(ATTR))}
    for r in rows:
        r['layer_name'] = attr.get((r['name'], str(r['region'])), '')
    print('\n' + '=' * 92)
    print('L15. 一次 = 障壁は線だけ(line)。プールを跨いで平均しない')
    res = {}
    for w in ('ako6', 'ako7'):
        print(f'\n{w}')
        for var in ('line', 'line+gap'):
            s = [r for r in rows if r['work'] == w and r['arm'] == 'real'
                 and r['variant'] == var]
            res[(w, var)] = tab(s, f'本来 ({var})')
        sh = [r for r in rows if r['work'] == w and r['arm'] == 'shuffled']
        res[(w, 'shuf')] = tab(sh, '対照 (平行移動, line)')
    print('\n--- 棒 ---')
    for w in ('ako6', 'ako7'):
        a, c = res[(w, 'line')], res[(w, 'shuf')]
        if not a:
            continue
        print(f'  {w}  P26 E {a["E"]:.4f} >= 0.50 -> {"PASS" if a["E"]>=0.50 else "FAIL"}'
              f'   / 逆の読み F {a["F"]:.4f} >= 0.50 -> '
              f'{"線が表現していない側" if a["F"]>=0.50 else "どちらでもない=判定保留"}')
        if c:
            print(f'      P27 対照 E {c["E"]:.4f} / 差 {a["E"]-c["E"]:+.4f} >= 0.20 -> '
                  f'{"PASS" if a["E"]-c["E"]>=0.20 else "FAIL"}  (対照の O 率 {c["O"]:.3f})')
        print(f'      P28 幅中央 {a["width"]:.1f}px <= 30 -> '
              f'{"PASS" if a["width"]<=30 else "FAIL"}')
    print('\n--- P29 開示 (棒なし) ---')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w and r['arm'] == 'real'
             and r['variant'] == 'line']
        per = collections.Counter((r['name'], r['region']) for r in s)
        print(f'  {w}: 面 {len(per)}  1面あたりの区間 中央 '
              f'{st.median(list(per.values())):.1f} / 平均 {sum(per.values())/len(per):.1f}')
        for lab, lo, hi in (('<10px', 0, 10), ('10-30px', 10, 30),
                            ('30-100px', 30, 100), ('>=100px', 100, 10 ** 9)):
            g = [r for r in s if r['kind'] != 'O' and lo <= r['width'] < hi]
            if not g:
                continue
            c = collections.Counter(r['kind'] for r in g)
            print(f'    幅 {lab:9} 区間 {len(g):6d}  E {c["E"]/len(g):.4f}  '
                  f'F {c["F"]/len(g):.4f}')
    print('\n--- 二次: 層の名前ごと (結論には使わない。L14 の帰属を見た後の切り口) ---')
    for w in ('ako6', 'ako7'):
        for nm in ('black', '線画', 'effect', '描き文字　　白フチ', 'bg base'):
            s = [r for r in rows if r['work'] == w and r['arm'] == 'real'
                 and r['variant'] == 'line' and r['layer_name'] == nm]
            if len(s) >= 30:
                tab(s, f'{w} {nm}')


if __name__ == '__main__':
    main()
