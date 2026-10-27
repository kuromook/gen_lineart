# -*- coding: utf-8 -*-
"""L18 の器具と既知セル KC21〜KC24。PASS しなければ本体は起動しない(登録 §5)。

器具:
  seed_of     ∂R から d 以内の**門が通した層**の墨 (L17 の adj) と、その argmax 層
  prop_full   seed に触れる8連結成分の和 (g=∞。登録の一次)
  geo_spread  測地距離で打ち切った伝播 (感度。uint16 のラベル画像を max フィルタで広げる)
  comp_pur    |C ∧ L| / |C|

**帰無値も棒もここには無い。** 器具が壊れていないかだけを見る。
"""
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l15_known_cells import boundary
from l17_known_cells import (arms, place, D1, adj_sets, assign_top, jobs,
                             panels_and_regions, purity, ring, zmap)

K8 = np.ones((3, 3), np.uint8)
GS = (32, 128, 512)          # 測地距離の打ち切り (感度。判定に使わない)
# L17 の P36 の分母。KC24 はこれと突き合わせる
INK_TOTAL = {'ako6': 43279719, 'ako7': 45598813}


def seed_of(R, lab, P, d=D1, bnd=None):
    """(seed の墨, argmax のラベル番号, adj の画素数)。L17 の adj と同じもの。"""
    b = boundary(R) if bnd is None else bnd
    if not b.any():
        return np.zeros(R.shape, bool), 0, 0
    dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                               cv2.DIST_MASK_PRECISE)
    m = (dt <= d) & P & (lab > 0)
    v = lab[m]
    if v.size == 0:
        return m, 0, 0
    vals, cnt = np.unique(v, return_counts=True)
    return m, int(vals[int(np.argmax(cnt))]), int(cnt.sum())


def ink_components(lab, P):
    ink = (lab > 0) & P
    n, cl, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    return ink, cl, st, n


def prop_full(seed, cl):
    """seed に触れる8連結成分の和集合 (g=∞)。"""
    ids = np.unique(cl[seed])
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros(cl.shape, bool), ids
    return np.isin(cl, ids), ids


def geo_spread(seedlab, ink, steps):
    """ラベル画像を墨の上で steps 回だけ広げる。max フィルタなので全ラベル同時。

    同着は**ラベル番号の大きい方**が勝つ(決定的だが任意。記録しておく)。
    """
    cur = seedlab.astype(np.uint16).copy()
    out = {}
    snaps = sorted(s for s in steps)
    for i in range(1, max(snaps) + 1):
        d = cv2.dilate(cur, K8, 1)
        new = (cur == 0) & ink & (d > 0)
        cur[new] = d[new]
        if i in snaps:
            out[i] = cur.copy()
    return out


def comp_pur(C, lab, L):
    n = int(C.sum())
    if n == 0:
        return float('nan')
    return float(int((C & (lab == L)).sum())) / n


# ---------------------------------------------------------------- 既知セル

def two_arcs(H=200, r=40):
    """交差して1成分になる2本の弧。別々の「層」。"""
    yy, xx = np.mgrid[0:H, 0:H]
    c = H // 2
    dist = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    rg = (dist >= r + 1.5) & (dist <= r + 3.5)
    a1 = rg & (ang < 0.7)
    a2 = rg & (ang >= 0.65)        # 0.65-0.70 で重なる -> 1成分
    return (dist <= r), a1, a2


def kc21():
    """合成: 2本の弧が1成分になるとき comp_pur が画素比に一致する。"""
    R, a1, a2 = two_arcs()
    P = np.ones(R.shape, bool)
    lab, keys = assign_top([(1, a1, 0), (2, a2, 1)], R.shape)
    ink, cl, st, n = ink_components(lab, P)
    seed, L, adj = seed_of(R, lab, P)
    C, ids = prop_full(seed, cl)
    ok = True
    print('KC21 器具: 1成分の中で comp_pur が画素比に一致するか (許容 ±0.02)')
    print(f'  成分 {n-1} 個 (期待 1)  C の画素 {int(C.sum())} = 墨 {int(ink.sum())} ?')
    ok &= (n - 1 == 1) and int(C.sum()) == int(ink.sum())
    for Lx in (1, 2):
        got = comp_pur(C, lab, Lx)
        exp = float((lab == Lx).sum()) / int(ink.sum())
        bad = abs(got - exp) > 0.02
        ok &= not bad
        print(f'  層 {Lx}: comp_pur {got:.4f}  期待(画素比) {exp:.4f}  '
              f'{"NG" if bad else "ok"}')
    # 1本だけなら 1.0
    lab2, _ = assign_top([(1, a1 | a2, 0)], R.shape)
    ink2, cl2, st2, n2 = ink_components(lab2, P)
    seed2, L2, _ = seed_of(R, lab2, P)
    C2, _ = prop_full(seed2, cl2)
    bad = abs(comp_pur(C2, lab2, 1) - 1.0) > 1e-9
    ok &= not bad
    print(f'  1層のみ: comp_pur {comp_pur(C2, lab2, 1):.4f} (期待 1.0)  '
          f'{"NG" if bad else "ok"}')
    return ok


def kc22():
    """合成: prop は seed に触れる成分ちょうど。触れていない成分を含まない。"""
    R, a1, a2 = two_arcs()
    P = np.ones(R.shape, bool)
    far = np.zeros(R.shape, bool)
    far[5:15, 5:60] = True                      # 離れた別の成分
    lab, _ = assign_top([(1, a1, 0), (2, a2, 1), (3, far, 2)], R.shape)
    ink, cl, st, n = ink_components(lab, P)
    seed, L, _ = seed_of(R, lab, P)
    C, ids = prop_full(seed, cl)
    inc_far = int((C & far).sum())
    covers = int((C & (a1 | a2)).sum()) == int((a1 | a2).sum())
    sup = bool((C | ~seed).all())               # prop ⊇ seed
    bad = (inc_far != 0) or (not covers) or (not sup) or (n - 1 != 2)
    print('KC22 器具: prop は seed に触れる成分ちょうどか')
    print(f'  成分 {n-1} 個 (期待 2)  離れた成分を含む画素 {inc_far} (期待 0)  '
          f'弧を全部含む {covers}  prop ⊇ seed {sup}  {"NG" if bad else "ok"}')
    # 測地距離の打ち切りが単調か
    sl = np.zeros(R.shape, np.uint16); sl[seed] = 1
    sn = geo_spread(sl, ink, GS)
    sz = [int((sn[g] > 0).sum()) for g in sorted(GS)]
    mono = all(sz[i] <= sz[i + 1] for i in range(len(sz) - 1)) and sz[-1] <= int(C.sum())
    print(f'  打ち切り g={sorted(GS)} の画素 {sz}  g=∞ {int(C.sum())}  '
          f'単調 {mono}  {"NG" if not mono else "ok"}')
    return (not bad) and mono


def kc23(limit=4):
    """データ経路: prop ⊇ seed が全面で成立し、cov を別経路で数えて一致する。"""
    zz = zmap(); fill, line = arms('gated')
    js = jobs(zz, fill, line)[:limit]
    print(f'KC23 データ経路: prop ⊇ seed と cov の二重計算 (頁 {len(js)})')
    nreg = 0; bad = 0; badcov = 0
    for job in js:
        for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
            if not sub:
                continue
            lab, keys = assign_top(sub, P.shape)
            ink, cl, st, n = ink_components(lab, P)
            acc = np.zeros(P.shape, bool); ids_all = set()
            for gi in gids:
                R = glab == gi
                seed, L, adj = seed_of(R, lab, P)
                if adj == 0:
                    continue
                C, ids = prop_full(seed, cl)
                if int((seed & ~C).sum()) != 0:
                    bad += 1
                acc |= C; ids_all |= set(int(x) for x in ids)
                nreg += 1
            # 別経路: 成分 id の集合から面積を足す
            want = sum(int(st[i, A]) for i in ids_all)
            if want != int(acc.sum()):
                badcov += 1
    print(f'  面 {nreg}  prop ⊂ seed の破れ {bad}  cov の不一致 {badcov}  '
          f'{"NG" if (bad or badcov) else "ok"}')
    return nreg > 0 and bad == 0 and badcov == 0


def kc24():
    """データ経路: コマの**門が通した層の墨**の総和が L17 の P36 の分母と一致する。"""
    zz = zmap(); fill, line = arms('gated')
    tot = {}
    for job in jobs(zz, fill, line):
        for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
            if not sub:
                continue
            lab, _ = assign_top(sub, P.shape)
            tot[work] = tot.get(work, 0) + int(((lab > 0) & P).sum())
    print('KC24 データ経路: **門が通した層の墨**の総和が L17 の P36 の分母と一致するか')
    ok = True
    for w in sorted(INK_TOTAL):
        bad = tot.get(w, 0) != INK_TOTAL[w]
        ok &= not bad
        print(f'  {w}: 今回 {tot.get(w,0):,}  L17 {INK_TOTAL[w]:,}  '
              f'{"NG" if bad else "ok"}')
    return ok


def kc32():
    """器具: **対照の置き場が、走行をまたいで同じか。**

    2026-10-25: `place()` の乱数の種に `abs(hash(pid))` を使っていた。
    **`hash(str)` は過程ごとに変わる**(PYTHONHASHSEED)ので、
    **帰無値も、置けた面の集合も、走行ごとに別物になっていた。**
    L18 の対照は 2回の走行で cov_honest 中央 0.0945 と 0.1178、
    `r0` は 0.0213 と 0.0655 —— **3倍違った。**

    ここは**記録した定数**と突き合わせる。過程に依る量が種に入れば即座に落ちる。
    """
    print('KC32 器具: 対照の置き場が走行をまたいで同じか (**記録した定数と照合**)')
    import cv2 as _cv
    g = np.zeros((400, 400), np.int32); g[100:160, 100:200] = 1
    P = np.ones((400, 400), bool)
    n, lab, st, _ = _cv.connectedComponentsWithStats((g > 0).astype(np.uint8), 8)
    # **実測して記録した定数**(2026-10-25、crc32 に直した後)。手で書かない
    want = {20261026: (2, 155), 20261027: (181, 40), 20261028: (67, 50),
            20261029: (255, 175), 20261030: (98, 258)}
    ok = True
    for sd, exp in want.items():
        R = place(lab, 1, st, P, sd, '018_2018_ako6_ako6_page0028', 1)
        if R is None:
            print(f'  seed {sd}: 置けなかった  FAIL'); ok = False; continue
        ys, xs = np.nonzero(R)
        got = (int(ys.min()), int(xs.min()))
        good = got == exp
        ok &= good
        print(f'  seed {sd}: 置いた位置 {got}  記録 {exp}  {"ok" if good else "FAIL"}')
    if not ok:
        print('  **記録と違う。種に過程依存の量が入っていないか見ること**')
    return ok


def main():
    res = [('KC21', kc21()), ('KC22', kc22()), ('KC23', kc23()), ('KC24', kc24()),
           ('KC32', kc32())]
    print('=' * 72)
    for k, v in res:
        print(f'{k}: {"PASS" if v else "FAIL"}')
    if not all(v for _, v in res):
        print('既知セルが破れた。本体は起動しない(登録 §5)')
        sys.exit(1)
    print('既知セル全 PASS。本体を起動してよい')


if __name__ == '__main__':
    main()
