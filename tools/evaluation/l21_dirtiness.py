# -*- coding: utf-8 -*-
"""L21: 線画データの「汚さ」の起案のための測定 (Track L、2026-10-25)。

基盤の依頼 `inbox/notice_dirtiness_case_l_and_m_20261025.md` に答えるための器具。
**「汚い」を測らない。** 名前の付いた欠陥を1つずつ、分母つきで数える。

器具:
  ridge_width(ink)   線の太さ。距離変換の尾根 (dt が 3x3 の最大と等しい画素) で 2*dt。
                     **尾根に限るのは、墨画素全部で取ると縁の薄い所に引っ張られるから**
  bbox_fill(ink)     墨の外接矩形の占有率。貼り込んだ画像は 1.0 に寄る
  cut_ink(P, ink)    コマの縁 1px に乗っている線墨 / コマの縁の長さ

**帰無値も棒もここには無い。** KC25〜KC27 が PASS しなければ本体は回さない。

  python tools/evaluation/l21_dirtiness.py --kc        既知セルだけ
  python tools/evaluation/l21_dirtiness.py --limit 4   頁を絞って本体
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l17_known_cells import (DEL, DS, D1, MIN_FRAC, MAX_FRAC, adj_sets,
                             assign_top, gated_layers, jobs,
                             panels_and_regions, zmap)
from l15_known_cells import boundary
from l14_known_cells import comps, A
from l14_arms import acc_add
from fill_mask import fill_mask

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l21_dirtiness'
TRAIN_SIDE = 1024          # 学習側が長辺をここまで落とす (前処理器の既定)
WMAX = 200                 # 太さの柱状図の上限 (px)。これを超える分は最後の柱に入れる
TMAX, TSTEP = 20.0, 0.25   # 学習解像度に落とした太さの柱状図
WCUT = (1, 3, 5, 9, 17, 33)        # 累積で報告する閾値 (太さ px)
TCUT = (0.5, 1.0, 1.5, 2.0, 3.0)   # 同じく学習解像度の太さ


# ---------------------------------------------------------------- 器具

def ridge_width(ink):
    """線の太さの分布。戻り値は尾根画素ごとの太さ (float32 配列)。

    dt = 墨の内側の距離変換 (cv2 の規約: 背景に接する墨は 1、背景は 0)。
    尾根 = dt が 3x3 近傍の最大と等しい墨画素。**太さ = 2*dt - 1。**
    幅 w (奇数) の帯なら中心画素の dt = (w+1)/2 なので w が返る。
    **分解能は奇数** —— 幅 2 は 1 と、幅 4 は 3 と読む (KC25 の許容 ±1 はこれを含む)。
    尾根に限るのは、墨画素すべてで取ると縁の薄い所に引っ張られて過小になるから。
    """
    u = ink.astype(np.uint8)
    if not u.any():
        return np.zeros(0, np.float32)
    dt = cv2.distanceTransform(u, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mx = cv2.dilate(dt, np.ones((3, 3), np.uint8))
    r = ink & (dt >= mx - 1e-6) & (dt > 0)
    return (2.0 * dt[r] - 1.0).astype(np.float32)


def bbox_fill(ink):
    """墨の外接矩形の占有率と矩形の大きさ。貼り込み画像は 1.0 に寄る。

    **占有率だけでは貼り込みを名指せない** —— まっすぐな帯も 1.0 になる
    (KC26 がこれを掴んだ)。貼り込みの条件は **占有率が高く、かつ縦横とも大きい**。
    """
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        return 0.0, 0, 0
    h = int(ys.max() - ys.min() + 1); w = int(xs.max() - xs.min() + 1)
    return float(ys.size) / (h * w), h, w


def cut_ink(P, ink):
    """コマの縁 1px に乗っている線墨 / コマの縁の長さ。断ち切りの指標。"""
    u = P.astype(np.uint8)
    er = cv2.erode(u, np.ones((3, 3), np.uint8))
    edge = P & (er == 0)
    n = int(edge.sum())
    return (int((edge & ink).sum()), n)


def acc_w(w, H=None):
    """太さを 2本の柱状図に足す。

    **単位を2つ出すのが肝** (基盤の L18 の裁定と同じ原則: 納品の単位に合わせる):
      len = 尾根画素ごとに 1       → **線の長さ**の分布 (細い線が何本あるか)
      ink = 尾根画素ごとに 太さ    → **墨の面積**の分布 (絵のどれだけが細いか)
    """
    if H is None:
        H = dict(len=np.zeros(WMAX + 1, np.int64), ink=np.zeros(WMAX + 1, np.float64))
    if w.size:
        i = np.clip(np.rint(w), 0, WMAX).astype(np.int32)
        H['len'] += np.bincount(i, minlength=WMAX + 1)
        H['ink'] += np.bincount(i, weights=w.astype(np.float64), minlength=WMAX + 1)
    return H


def acc_t(w, scale, H=None):
    """学習解像度 (長辺 TRAIN_SIDE) に落としたときの太さ。単位は acc_w と同じ。"""
    n = int(TMAX / TSTEP) + 1
    if H is None:
        H = dict(len=np.zeros(n, np.int64), ink=np.zeros(n, np.float64))
    if w.size:
        t = w * scale
        i = np.clip(np.rint(t / TSTEP), 0, n - 1).astype(np.int32)
        H['len'] += np.bincount(i, minlength=n)
        H['ink'] += np.bincount(i, weights=w.astype(np.float64), minlength=n)
    return H


def add_H(a, b):
    for k in ('len', 'ink'):
        a[k] += b[k]
    return a


def cum(H, key, cuts, step=1.0):
    """P(太さ <= t) を cuts ごとに返す。"""
    tot = H[key].sum()
    if tot <= 0:
        return [float('nan')] * len(cuts)
    c = np.cumsum(H[key])
    out = []
    for t in cuts:
        i = min(len(c) - 1, int(round(t / step)))
        out.append(float(c[i] / tot))
    return out


def quant(H, key, qs=(0.25, 0.5, 0.75), step=1.0):
    tot = H[key].sum()
    if tot <= 0:
        return [float('nan')] * len(qs)
    c = np.cumsum(H[key]) / tot
    return [float(np.searchsorted(c, q) * step) for q in qs]


# ---------------------------------------------------------------- 既知セル

def band(kind, r, n=500):
    """1画素の芯線を半径 r の真円で太らせた帯。**幅は 2r+1 と定義できる。**

    `cv2.line(thickness=w)` は幅が w にならないので試験には使えない
    (2026-10-25: KC25 がこれを掴んだ)。
    """
    core = np.zeros((n, n), np.uint8)
    if kind == '直線':
        core[n // 2, 40:n - 40] = 1
    elif kind == '斜め':
        for i in range(40, n - 40):
            core[i, i] = 1
    else:
        cv2.circle(core, (n // 2, n // 2), n // 3, 1, 1)
    if r == 0:
        return core.astype(bool)
    dt = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2,
                               cv2.DIST_MASK_PRECISE)
    return dt <= r


def kc25():
    """器具: 既知の太さの帯を入れて、その太さが返るか。**偏りの表を必ず印字する。**

    許容は ±2 で、**その中に系統的な偏りがある** —— 直線は厳密、斜めと円弧は
    最大 1.5px 低く出る(離散の距離変換が斜めの縁で近い零画素を取るため)。
    **偏りの向きは「細く見える」側**なので、細さを主張する数には必ずこの表を添える。
    """
    print('KC25 器具: 幅の分かっている帯 (芯線を半径 r で太らせ、幅 = 2r+1)')
    print(f'  {"幅":>4}{"直線":>9}{"斜め":>9}{"円弧":>9}   偏り(最小 - 幅)')
    ok = True
    for r in (0, 1, 2, 4, 8, 16):
        w = 2 * r + 1
        got = {}
        for kind in ('直線', '斜め', '円弧'):
            g = ridge_width(band(kind, r))
            got[kind] = float(np.median(g)) if g.size else -1
        bias = min(got.values()) - w
        good = all(abs(v - w) <= 2.0 for v in got.values())
        ok &= good
        print(f'  {w:>4}{got["直線"]:>9.2f}{got["斜め"]:>9.2f}{got["円弧"]:>9.2f}'
              f'   {bias:+.2f}  {"ok" if good else "FAIL"}')
    print('  直線は厳密。斜め・円弧は最大 1.5px 低く出る(= 細く見える側の偏り)')
    return ok


def kc26():
    """器具: bbox_fill と cut_ink。"""
    print('KC26 器具: bbox_fill(貼り込みの検出) と cut_ink(断ち切り)')
    ok = True
    solid = np.zeros((100, 200), bool); solid[10:90, 20:180] = True
    f, h, w = bbox_fill(solid)
    good = abs(f - 1.0) < 1e-9 and (h, w) == (80, 160)
    ok &= good
    print(f'  べた矩形 占有 {f:.4f} 大きさ {h}x{w}  (期待 1.0 / 80x160)'
          f'  {"ok" if good else "FAIL"}')
    straight = np.zeros((100, 200), bool); straight[49:52, 20:180] = True
    f2, h2, w2 = bbox_fill(straight)
    good = abs(f2 - 1.0) < 1e-9 and min(h2, w2) == 3
    ok &= good
    print(f'  まっすぐな幅3の帯 占有 {f2:.4f} 短辺 {min(h2, w2)}'
          f'  (期待 1.0 / 3 —— **占有率だけでは貼り込みと区別できない**)'
          f'  {"ok" if good else "FAIL"}')
    diag = band('斜め', 1, n=200)
    f3, h3, w3 = bbox_fill(diag)
    good = f3 < 0.10
    ok &= good
    print(f'  斜めの幅3の帯 占有 {f3:.4f} 短辺 {min(h3, w3)}  (期待 < 0.10)'
          f'  {"ok" if good else "FAIL"}')
    P = np.zeros((100, 100), bool); P[10:90, 10:90] = True
    ink = np.zeros((100, 100), bool); ink[50, 10:90] = True      # 縁に2画素だけ乗る
    hit, nb = cut_ink(P, ink)
    good = hit == 2 and nb == 2 * 80 + 2 * 78
    ok &= good
    print(f'  縁に乗る墨 {hit} / 縁 {nb}  (期待 2 / {2 * 80 + 2 * 78})'
          f'  {"ok" if good else "FAIL"}')
    return ok


def kc27():
    """データ経路: 線墨の総和が L17 の P36 / L18 の KC24 の分母と一致するか。"""
    print('KC27 データ経路: 線墨の総和が L17/L18 と一致するか')
    want = {'ako6': 43279719, 'ako7': 45598813}
    zz = zmap(); fill, line = gated_layers()
    got = Counter()
    for job in jobs(zz, fill, line):
        for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
            lab, keys = assign_top(sub, P.shape)
            got[work] += int(((lab > 0) & P).sum())
    ok = True
    for w in ('ako6', 'ako7'):
        good = got[w] == want[w]
        ok &= good
        print(f'  {w}: 今回 {got[w]:,}  L17/L18 {want[w]:,}  {"ok" if good else "FAIL"}')
    return ok


# ---------------------------------------------------------------- 本体

def pages_with_window(job):
    """`panels_and_regions` と**同じ母集団**を、コマの窓 (y0, x0) つきで返す。

    窓が要るのは、面を「どの FILL 層の墨か」に戻すため。共有器具の戻り値を
    増やすと L17/L18 の呼び出しが壊れるので、ここに写した。
    **写しが元と一致していることは KC28 (面の数) で点検する。**
    """
    pid, work, fz, lz, sel = job
    F, fl = None, []
    for lid, path in fz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        if F is None:
            F = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid')
        fl.append((lid, ink))
    if F is None:
        return
    H, W = F.shape
    lines = []
    for lid, path, z in lz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (H, W):
            continue
        lines.append((lid, im < 128, z))
    for fr in sel:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            glab, gids, gstat = comps(f, 8)
            if not gids:
                continue
            sub = [(lid, ink[y0:y1, x0:x1], z) for lid, ink, z in lines]
            yield (pid, work, ci, P, glab, gids, gstat, sub,
                   (y0, x0, y1, x1), fl, lines)


def one_page(job):
    """1頁 → 太さの階級 / 断ち切り / 面の出所 / 層の矩形占有。"""
    wh = acc_w(np.zeros(0, np.float32))
    wt = acc_t(np.zeros(0, np.float32), 1.0)
    wh_lay = {}
    cut_hit = cut_n = 0
    regions, panels, inkpx, ridgepx = [], [], 0, 0
    fl_cache, fm_cache = None, {}
    pid = work = None
    k = 0        # **L19 と同じ通し番号**。面の鍵 (pid, comp, region) は一意でないので、
                 # 突き合わせは L19 の切り出し名で行う (KC28)
    ln_cache = []
    for (pid, work, ci, P, glab, gids, gstat,
         sub, (y0, x0, y1, x1), fl, lines) in pages_with_window(job):
        fl_cache = fl
        ln_cache = [(lid, ink) for lid, ink, z in lines]
        lab, keys = assign_top(sub, P.shape)
        inkP = (lab > 0) & P
        inkpx += int(inkP.sum())
        sc = TRAIN_SIDE / max(y1 - y0, x1 - x0)
        panels.append(dict(pid=pid, work=work, comp=ci, area=int(P.sum()),
                           h=int(y1 - y0), w=int(x1 - x0), scale=round(sc, 5)))
        w = ridge_width(inkP)
        ridgepx += int(w.size)
        acc_w(w, wh)
        acc_t(w, sc, wt)
        for i, ky in enumerate(keys):     # **k を潰さないこと** (通し番号に使っている)
            m = inkP & (lab == i + 1)
            if m.any():
                nm = NAMES.get((pid, str(ky)), f'id={ky}')
                wh_lay.setdefault(nm, acc_w(np.zeros(0, np.float32)))
                acc_w(ridge_width(m), wh_lay[nm])
        h, n = cut_ink(P, inkP)
        cut_hit += h
        cut_n += n
        # ---- 面の出所: その面の墨を持っている FILL 層 (層ごとの fill_mask で引く)
        for lid, ink in fl:
            if lid in fm_cache:
                continue
            fm = np.zeros(ink.shape, bool)
            acc_add(fm, None, ink, 'solid')
            fm_cache[lid] = fm
        for gi in gids:
            R = glab == gi
            b = boundary(R)
            vals, cnt = adj_sets(R, lab, P, ds=(D1,), bnd=b)[D1]
            if cnt.sum() == 0:       # **L19 と同じ条件で飛ばす**(飛ばすと k は増えない)
                continue
            k += 1
            npx = int(R.sum())
            src = []
            for lid, ink in fl:
                c = int((fm_cache[lid][y0:y1, x0:x1] & R).sum())
                if c:
                    src.append((lid, c))
            src.sort(key=lambda t: -t[1])
            regions.append(dict(
                pid=pid, work=work, comp=ci, region=gi, area=npx,
                crop=f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png',
                src=[dict(lid=lid, px=c, share=round(c / max(npx, 1), 4),
                          name=NAMES.get((pid, str(lid)), f'id={lid}')) for lid, c in src],
                n_src=len(src)))
    lay_rect = []
    for kind, src in (('fill', fl_cache or []), ('line', ln_cache)):
        for lid, ink in src:
            f, h, w = bbox_fill(ink)
            lay_rect.append(dict(pid=pid, work=work, lid=lid, kind=kind,
                                 name=NAMES.get((pid, str(lid)), f'id={lid}'),
                                 ink=int(ink.sum()), fillrate=round(f, 4), h=h, w=w))
    return dict(work=work, pid=pid, wh=wh, wt=wt, wh_lay=wh_lay,
                cut=(cut_hit, cut_n), panels=panels, regions=regions,
                inkpx=inkpx, ridgepx=ridgepx, lay_rect=lay_rect)


def kc28(res):
    """データ経路: 写した母集団が L19 の待ち行列と**同じ面**か。

    面の鍵 (pid, comp, region) は一意でない(コマ番号が枠ファイルごとに振り直される)。
    L19 の切り出し名は頁内の通し番号を持つので一意で、**名前の集合が一致すれば
    面の数も順序も同じ**。写し (`pages_with_window`) が元から離れていないことの点検。
    """
    print('KC28 データ経路: 面の集合が L19 の待ち行列と一致するか (切り出し名で照合)')
    q = json.load(open('/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'))
    want = {r['crop'] for r in q['queue']}          # 行列に出た 551 面
    got = {x['crop'] for r in res for x in r['regions']}
    got_reg = sum(len(r['regions']) for r in res)
    got_pan = sum(len(r['panels']) for r in res)
    o1 = got_reg == q['n_all']
    o2 = got_pan == 246
    o3 = want <= got
    print(f'  面 今回 {got_reg}  L19 {q["n_all"]}  {"ok" if o1 else "FAIL"}')
    print(f'  コマ 今回 {got_pan}  前の走り 246  {"ok" if o2 else "FAIL"}')
    print(f'  行列の 551 面が今回の名前に全部あるか: {"ok" if o3 else "FAIL"}'
          f'  (欠け {len(want - got)})')
    return o1 and o2 and o3


# ---------------------------------------------------------------- 報告

def report(res):
    os.makedirs(OUT, exist_ok=True)
    pools = ('ako6', 'ako7')
    print('\n' + '=' * 72)
    print('[1] 線の太さ (母集団 = コマの中の線墨。KC25 の偏り表を必ず添えて読む)')
    print('    単位が2つある —— **長さ** (尾根画素ごとに1) と **墨** (尾根画素を太さで重み)。')
    print('    「細い線が何本あるか」と「絵のどれだけが細いか」は別の問いで、答も違う。')
    wh_pool, wt_pool = {}, {}
    for w in pools:
        H = acc_w(np.zeros(0, np.float32)); T = acc_t(np.zeros(0, np.float32), 1.0)
        ink = ridge = 0
        for r in res:
            if r['work'] == w:
                add_H(H, r['wh']); add_H(T, r['wt'])
                ink += r['inkpx']; ridge += r['ridgepx']
        if ridge == 0:
            continue
        wh_pool[w] = (H, ink, ridge); wt_pool[w] = T
        print(f'\n  {w}  線墨 {ink:,}px  尾根 {ridge:,}px')
        for key, nm in (('len', '長さ'), ('ink', '墨')):
            q = quant(H, key)
            cc = cum(H, key, WCUT)
            print(f'    {nm}で重み  四分位 {q[0]:.0f} / **中央 {q[1]:.0f}** / {q[2]:.0f} px'
                  + '   ' + '  '.join(f'<={t}px {v:.4f}' for t, v in zip(WCUT, cc)))

    print('\n[2] 学習解像度に落としたときの太さ')
    print(f'    コマの長辺を {TRAIN_SIDE}px にする。**1px を切った線は消えるか灰になる**')
    for w in pools:
        ps = [p for r in res if r['work'] == w for p in r['panels']]
        if not ps:
            continue
        ls = sorted(max(p['h'], p['w']) for p in ps)
        sc = sorted(p['scale'] for p in ps)
        print(f'\n  {w}  コマ {len(ps)}  長辺 中央 {ls[len(ls) // 2]:,}px'
              f'  縮小率 中央 {sc[len(sc) // 2]:.4f} (= 1/{1 / sc[len(sc) // 2]:.2f})')
        T = wt_pool[w]
        for key, nm in (('len', '長さ'), ('ink', '墨')):
            q = quant(T, key, step=TSTEP)
            cc = cum(T, key, TCUT, step=TSTEP)
            print(f'    {nm}で重み  中央 {q[1]:.2f}px'
                  + '   ' + '  '.join(f'<{t}px {v:.4f}' for t, v in zip(TCUT, cc)))

    print('\n[3] コマの縁に乗っている線墨 (断ち切りの指標)')
    for w in pools:
        hit = sum(r['cut'][0] for r in res if r['work'] == w)
        n = sum(r['cut'][1] for r in res if r['work'] == w)
        print(f'  {w:<7}縁の画素 {n:>10,}  うち線墨 {hit:>9,}  {hit / max(n, 1):.4f}')

    print('\n[4] FILL 層の矩形占有 (貼り込んだ画像は 1.0 に寄る)')
    rect = [x for r in res for x in r['lay_rect'] if x['ink'] >= 10000]
    rect.sort(key=lambda x: -x['fillrate'])
    # 貼り込みの条件は **占有率が高く、かつ縦横とも大きい** (KC26)
    big2 = [x for x in rect if min(x['h'], x['w']) >= 200]
    print(f'  墨 1万px 以上の FILL 層 {len(rect)}  (縦横とも 200px 以上 {len(big2)})')
    print(f'  そのうち 占有 >=0.90 が {sum(1 for x in big2 if x["fillrate"] >= 0.90)}  '
          f'>=0.98 が {sum(1 for x in big2 if x["fillrate"] >= 0.98)}'
          f'  = **貼り込んだ画像の候補**')
    rect = big2 or rect
    for x in rect[:12]:
        print(f'   {x["fillrate"]:.4f}  {x["ink"]:>9,}px  {x["h"]:>5}x{x["w"]:<5}'
              f'  {x["work"]} {x["pid"]} L{x["lid"]}  {x["name"]}')

    print('\n[5] 面の出所: 作者の語ごとに、その面の墨を持つ FILL 層')
    q = json.load(open('/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'))
    crop_of = {r['order']: r['crop'] for r in q['queue']}
    lab = {}
    pl = '/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'
    for ln in open(pl):
        r = json.loads(ln)
        if r['choice'] == 'free' and r.get('free'):
            r['choice'], r['word'] = 'word', r['free'].strip()
        lab[crop_of[r['order']]] = r          # **order ごとの最後の記録。鍵は切り出し名**
    regs = {x['crop']: x for r in res for x in r['regions']}
    rows = []
    for k, r in lab.items():
        x = regs.get(k)
        if x is None:
            continue
        ans = r['word'] if r['choice'] == 'word' else r['choice']
        rows.append((ans, x, r))
    print(f'  突き合わせた面 {len(rows)} / 作者の答 {len(lab)}')
    by = defaultdict(list)
    for ans, x, r in rows:
        by[ans].append(x)
    print(f'  {"作者の答":<14}{"面":>5}{"出所層数 中央":>13}  最も多い出所層名(上位3)')
    for ans, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ns = sorted(x['n_src'] for x in xs)
        nm = Counter(x['src'][0]['name'] for x in xs if x['src'])
        print(f'  {ans:<14}{len(xs):>5}{ns[len(ns) // 2]:>13}  '
              + '  '.join(f'{a}:{b}' for a, b in nm.most_common(3)))
    with open(f'{OUT}/region_source.csv', 'w') as fh:
        fh.write('work,pid,comp,region,area,answer,n_src,top_src_name,top_src_share\n')
        for ans, x, r in sorted(rows, key=lambda t: (t[1]['work'], t[1]['pid'])):
            s = x['src'][0] if x['src'] else dict(name='', share='')
            fh.write(f'{x["work"]},{x["pid"]},{x["comp"]},{x["region"]},{x["area"]},'
                     f'{ans},{x["n_src"]},"{s["name"]}",{s["share"]}\n')
    print(f'  -> {OUT}/region_source.csv')

    print('\n[6] 層名ごとの太さ (尾根 10万画素以上の層名)')
    agg = {}
    for r in res:
        for k, H in r['wh_lay'].items():
            agg.setdefault(k, acc_w(np.zeros(0, np.float32)))
            add_H(agg[k], H)
    big = sorted(agg.items(), key=lambda kv: -kv[1]['len'].sum())
    print(f'  {"層名":<32}{"尾根":>11}{"長さ中央":>9}{"墨中央":>8}  ' +
          '  '.join(f'墨<={t}px' for t in (1, 3, 9)))
    for k, H in big:
        t = int(H['len'].sum())
        if t < 100000:
            continue
        ql = quant(H, 'len')[1]; qi = quant(H, 'ink')[1]
        cc = cum(H, 'ink', (1, 3, 9))
        print(f'  {k[:31]:<32}{t:>11,}{ql:>9.0f}{qi:>8.0f}  '
              + '  '.join(f'{v:>8.4f}' for v in cc))
    json.dump({k: {'len': H['len'].tolist(), 'ink': H['ink'].tolist()}
               for k, H in agg.items()},
              open(f'{OUT}/width_by_layer.json', 'w'), ensure_ascii=False)
    print(f'  -> {OUT}/width_by_layer.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kc', action='store_true', help='既知セルだけ')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    ok = kc25() and kc26()
    if not ok:
        print('器具の既知セルが FAIL。止める'); return 1
    if a.kc:
        print('KC27 (データ経路) は本体と同じ読み込みなので、本体の走りで出す')
        return 0
    zz = zmap(); fill, line = gated_layers()
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'\n頁 {len(js)}')
    res = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, js), 1):
            res.append(r)
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁', flush=True)
    print()
    got = Counter()
    for r in res:
        got[r['work']] += r['inkpx']
    want = {'ako6': 43279719, 'ako7': 45598813}
    print('KC27 データ経路: 線墨の総和が L17/L18 の分母と一致するか')
    o = True
    for w in ('ako6', 'ako7'):
        g = got[w] == want[w] if not a.limit else None
        o &= bool(g) or a.limit > 0
        print(f'  {w}: 今回 {got[w]:,}  L17/L18 {want[w]:,}  '
              f'{"(頁を絞ったので比較しない)" if a.limit else ("ok" if g else "FAIL")}')
    if not a.limit:
        o &= kc28(res)
        if not o:
            print('データ経路の既知セルが FAIL。報告を出さない'); return 1
    report(res)
    return 0


NAMES = {(r['page_id'], r['layer_id']): r['name']
         for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}

if __name__ == '__main__':
    sys.exit(main())
