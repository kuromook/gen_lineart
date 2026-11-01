"""原案の腕 第2段: **マスク層の8連結成分を面として申告し、(b) の標準の面と照合する**。

**事前登録は 追補179**(走行前に commit `907bd07`)。**棒は置いていない** ——
規律9「棒の前に帰無値を出す」。**この走行の点数を「棒に届いた/届かない」と読まない。**

正解源 = **(b) の標準 `T4` の閉じた面**(追補149 の裁定)。母集団
**`pop_5pages_no_page0005`**(作者の裁定。追補175)。

面の提案 = **選ばれた層のアルファの8連結成分**(判定の単位。作者 2026-10-28)。
**層をまたいだ合併はしない。重なりを許す。床は標準と同じ `1e-5`。**

腕 = 鍵6本(K2〜K7)+ 帰無値6本(N0〜N5)。**全部同じ表に並べる。**
主値 = **`F1@0.5`**(面どうしの IoU ≥ 0.5 の貪欲な1対1対応)。**頁ごと。プールしない。**
**IoU 0.3 / 0.7 も印字するが棒には使わない**(走る前に決めた)。
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m77_standard_visible as m77                                   # noqa: E402
import m87_split_standard_by_occlusion as m87                        # noqa: E402
import m99_b_standard as m99                                         # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402
import m113_mask_key_agreement as m113                               # noqa: E402

OUT = Path('results/m114_proposal_arm')
V3 = Path('results/m104_b_standard_v3/b_standard_v3.csv')
EXCLUDED = 'page0005'
KEYS = ('K2', 'K3', 'K4', 'K5', 'K6', 'K7')
NULLS = ('N0', 'N1', 'N2', 'N3', 'N4', 'N5')
THR = (0.3, 0.5, 0.7)
MAIN = 0.5
IC4 = {'K2': 75, 'K3': 58, 'K4': 139, 'K5': 149, 'K6': 44, 'K7': 61}


def comps_of(mask, floor_px, store, tag):
    """1枚の真偽図の**8連結成分**を提案として積む。床で落ちた数も数える。"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                                     connectivity=8)
    under = 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < floor_px:
            under += 1
            continue
        x0 = int(st[i, cv2.CC_STAT_LEFT])
        y0 = int(st[i, cv2.CC_STAT_TOP])
        w = int(st[i, cv2.CC_STAT_WIDTH])
        h = int(st[i, cv2.CC_STAT_HEIGHT])
        store.append((tag, a, y0, y0 + h, x0, x0 + w,
                      lab[y0:y0 + h, x0:x0 + w] == i))
    del lab
    return under


def score(props, std_lab, std_area, flip_w=None):
    """提案と標準の面を IoU で貪欲に1対1対応。閾値ごとの P/R/F1 を返す。"""
    cand = []
    for _, a, y0, y1, x0, x1, m in props:
        if flip_w is not None:                 # NC3: 左右反転(誤り)
            xx0, xx1 = flip_w - x1, flip_w - x0
            m = m[:, ::-1]
        else:
            xx0, xx1 = x0, x1
        sub = std_lab[y0:y1, xx0:xx1]
        v = sub[m]
        if v.size == 0:
            continue
        ids, cnt = np.unique(v[v > 0], return_counts=True)
        for sid, c in zip(ids.tolist(), cnt.tolist()):
            iou = c / (a + std_area[sid] - c)
            cand.append((iou, sid, len(cand)))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    out = {}
    for t in THR:
        used_s, used_p, hit = set(), set(), 0
        for iou, sid, pid in cand:
            if iou < t:
                break
            if sid in used_s or pid in used_p:
                continue
            used_s.add(sid)
            used_p.add(pid)
            hit += 1
        np_, ns = len(props), len(std_area)
        pr = hit / np_ if np_ else float('nan')
        rc = hit / ns if ns else float('nan')
        f1 = 0.0 if (not np_ or not ns or hit == 0) else 2 * pr * rc / (pr + rc)
        out[t] = (pr, rc, f1, hit)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    known = {}
    for r in csv.DictReader(open(V3, encoding='utf-8-sig')):
        known[r['page']] = dict(closed=int(r['T4_closed_n']),
                                med=round(float(r['T4_ratio_med']), 4),
                                lt05=int(r['T4_ratio_lt05_n']),
                                under=int(r['T4_under_floor']))
    pages = [p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
             if p['page'] in known and EXCLUDED not in p['page']]
    print(f'母集団 `pop_5pages_no_page0005` = {len(pages)} 頁 / '
          f'既知セル {len(pages)*4} 件 / 主値 F1@{MAIN}\n')

    rows_out, keytot, nc_rows = [], collections.Counter(), []
    for p in pages:
        ink, bnd, floor_px, n_line, n_edge = m106.build(p, byp, kinds)
        srows, under = m106.faces(bnd, ink, floor_px)
        rs = sorted(r['ratio'] for r in srows)
        m_ = rs[len(rs)//2] if len(rs) % 2 else (rs[len(rs)//2-1]+rs[len(rs)//2])/2
        got = dict(closed=len(srows), med=round(m_, 4),
                   lt05=sum(1 for r in srows if r['ratio'] < 0.5), under=under)
        k = known[p['page']]
        ok = all(abs(got[x] - k[x]) < 1e-9 for x in k)
        print(f'{p["page"]:44s} 標準の面 {len(srows):>4} / 床 {floor_px:>6} px  '
              + ('IC1 PASS' if ok else '**IC1 FAIL** ' + str(got)))
        if not ok:
            print('  **門が落ちた。この頁は読まない**')
            del ink, bnd
            continue

        H, W = bnd.shape
        n, std_lab, st, _ = cv2.connectedComponentsWithStats(
            (~bnd).astype(np.uint8), connectivity=4)
        keep = {r['face'] for r in srows}
        std_area = {r['face']: r['area'] for r in srows}
        std_lab = np.where(np.isin(std_lab, np.fromiter(keep, int, len(keep))),
                           std_lab, 0)
        del bnd

        K, drop, body = m113.keyset(byp[p['source']], kinds, p['source'])
        for kk in KEYS:
            keytot[kk] += len(K[kk])
        files = {l['layer_id']: l['file'] for l in p['layers']}
        grid = (H, W)
        cache = {}
        alphas = {}
        for r in body:
            lid = str(r['layer_id'])
            if any(lid in K[kk] for kk in KEYS) or True:
                a = m87.alpha_of(r['layer_id'], files, grid, cache)
                if a is not None:
                    full = np.zeros(grid, bool)
                    full[:a.shape[0], :a.shape[1]] = a
                    alphas[lid] = full
        cache.clear()

        arms = {}
        for kk in KEYS:
            props, u = [], 0
            for lid in sorted(K[kk], key=int):
                if lid in alphas:
                    u += comps_of(alphas[lid], floor_px, props, lid)
            arms[kk] = (props, u)
        # ---- 帰無値
        arms['N0'] = ([], 0)
        whole = np.ones(grid, bool)
        pr = []
        comps_of(whole, floor_px, pr, 'all')
        arms['N1'] = (pr, 0)
        del whole
        cells = max(1, len(srows))
        g = max(1, int(round(cells ** 0.5)))
        gl = np.zeros(grid, np.int32)
        for iy in range(g):
            for ix in range(g):
                gl[iy*H//g:(iy+1)*H//g, ix*W//g:(ix+1)*W//g] = iy*g+ix+1
        pr, u = [], 0
        for i in range(1, g*g+1):
            m2 = gl == i
            ys, xs = np.nonzero(m2)
            if len(ys) == 0:
                continue
            a = int(m2.sum())
            if a < floor_px:
                u += 1
                continue
            pr.append(('grid', a, ys.min(), ys.max()+1, xs.min(), xs.max()+1,
                       m2[ys.min():ys.max()+1, xs.min():xs.max()+1]))
        arms['N2'] = (pr, u)
        del gl
        pr, u = [], 0
        for lid in sorted(K['K1'], key=int):
            if lid in alphas:
                u += comps_of(alphas[lid], floor_px, pr, lid)
        arms['N3'] = (pr, u)
        pr = []
        for r in srows:
            m2 = std_lab[r['y0']:r['y1'], r['x0']:r['x1']] == r['face']
            pr.append(('std', r['area'], r['y0'], r['y1'], r['x0'], r['x1'], m2))
        arms['N4'] = (pr, 0)
        pr = []
        u = comps_of(ink, floor_px, pr, 'ink')
        arms['N5'] = (pr, u)
        del ink, alphas

        for nm_ in KEYS + NULLS:
            props, u = arms[nm_]
            s = score(props, std_lab, std_area)
            row = dict(page=p['page'], arm=nm_, n_prop=len(props),
                       dropped_floor=u, n_std=len(srows))
            for t in THR:
                prc, rcl, f1, hit = s[t]
                row[f'P@{t}'] = prc
                row[f'R@{t}'] = rcl
                row[f'F1@{t}'] = f1
                row[f'hit@{t}'] = hit
            rows_out.append(row)
        # ---- NC3(左右反転)と NC1(閾値0)は K7 と N3 で回す
        for nm_ in ('K7', 'N3'):
            props, _ = arms[nm_]
            sf = score(props, std_lab, std_area, flip_w=W)
            nc_rows.append((p['page'], nm_, s and None, sf[MAIN][2]))
        del std_lab, arms
    print()

    ic4 = all(keytot[k] == v for k, v in IC4.items())
    print('IC4 鍵ごとの層の枚数(5頁の合計。追補178 §C と一致するか): '
          + ' / '.join(f'{k} {v}/{keytot[k]}' for k, v in IC4.items())
          + ('  PASS' if ic4 else '  **FAIL**'))
    by = {(r['page'], r['arm']): r for r in rows_out}
    ic2 = all(abs(by[(p['page'], 'N4')][f'F1@{t}'] - 1.0) < 1e-9
              for p in pages for t in THR if (p['page'], 'N4') in by)
    print(f'IC2 N4(標準そのもの)が全閾値で F1 = 1.000'
          + ('  PASS' if ic2 else '  **FAIL**')
          + '  (**恒等。算術である。棒ではない**)')
    ic3 = all(by[(p['page'], 'N0')][f'R@{MAIN}'] == 0.0
              for p in pages if (p['page'], 'N0') in by)
    print(f'IC3 N0 の recall = 0.000' + ('  PASS' if ic3 else '  **FAIL**'))
    if not (ic4 and ic2 and ic3):
        print('**門が落ちた。以降の行は読まない**')
        return 1

    with open(OUT / 'arm_scores.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    print(f'\n=== A. `F1@{MAIN}`(**頁ごと。プールしない**。条例2) ===')
    print(f'{"腕":<5}' + ''.join(f'{p["page"][-11:]:>13}' for p in pages)
          + f'{"最悪":>8}')
    worst = {}
    for nm_ in KEYS + NULLS:
        v = [by[(p['page'], nm_)][f'F1@{MAIN}'] for p in pages]
        worst[nm_] = min(v)
        print(f'{nm_:<5}' + ''.join(f'{x:>13.3f}' for x in v)
              + f'{worst[nm_]:>8.3f}')
    print('  **連言は最悪で判定する**(条例1)')

    print(f'\n=== B. 精度と再現率(`@{MAIN}`。**F1 だけでは向きが読めない**) ===')
    print(f'{"腕":<5}{"提案":>8}{"床で落ち":>9}{"標準":>7}'
          + ''.join(f'{p["page"][-9:]:>22}' for p in pages))
    for nm_ in KEYS + NULLS:
        tot = sum(by[(p['page'], nm_)]['n_prop'] for p in pages)
        du = sum(by[(p['page'], nm_)]['dropped_floor'] for p in pages)
        ns = sum(by[(p['page'], nm_)]['n_std'] for p in pages)
        cells = ''
        for p in pages:
            r = by[(p['page'], nm_)]
            cells += f'  P{r[f"P@{MAIN}"]:.2f}/R{r[f"R@{MAIN}"]:.2f}/当{r[f"hit@{MAIN}"]:>3}'
        print(f'{nm_:<5}{tot:>8}{du:>9}{ns:>7}{cells}')

    print('\n=== C. 閾値を変えた形(**棒には使わない**。走る前に決めた) ===')
    print(f'{"腕":<5}' + ''.join(f'{"F1@"+str(t):>10}' for t in THR)
          + '   (最悪の頁)')
    for nm_ in KEYS + NULLS:
        print(f'{nm_:<5}' + ''.join(
            f'{min(by[(p["page"], nm_)][f"F1@{t}"] for p in pages):>10.3f}'
            for t in THR))

    print('\n=== D. 負の対照 ===')
    print('  NC3 提案を左右反転する(誤り。位置が合わなくなる)')
    for pg, nm_, _, f1 in nc_rows:
        base = by[(pg, nm_)][f'F1@{MAIN}']
        # **正位置が既に 0 の頁では、この対照は定義できない。**
        # 最初の版は「点数が位置を見ていない」と印字した —— **誤りである**
        verdict = ('  **正位置が 0.000 なので、この対照は定義できない(判定保留)**'
                   if base == 0.0 else
                   '  **下がった。PASS**' if f1 < base else
                   '  **下がらない。点数が位置を見ていない。FAIL**')
        print(f'    {pg[-11:]:<12} {nm_:<4} 正位置 {base:.3f} → 反転 {f1:.3f}'
              + verdict)
    print('  NC1 閾値を 0.3 に緩める = §C の左列。**どの腕でも recall が上がるはず**')
    for nm_ in KEYS + NULLS:
        a = min(by[(p['page'], nm_)][f'R@{THR[0]}'] for p in pages)
        b = min(by[(p['page'], nm_)][f'R@{MAIN}'] for p in pages)
        print(f'    {nm_:<4} R@0.3 {a:.3f} ≥ R@0.5 {b:.3f}'
              + ('  PASS' if a >= b - 1e-12 else '  **FAIL**'))

    print('\n=== E. 判定(追補179 の C1〜C4。**棒ではない**) ===')
    ref = ('N1', 'N2', 'N3', 'N5')
    win = [k for k in KEYS
           if all(by[(p['page'], k)][f'F1@{MAIN}']
                  > by[(p['page'], n)][f'F1@{MAIN}']
                  for p in pages for n in ref)]
    print(f'  C1 全5頁で N1・N2・N3・N5 のすべてに勝つ鍵: '
          f'**{" ".join(win) if win else "無し"}**')
    # **C1 の形の欠点を印字する**: 腕と帰無値が両方 0.000 の頁では
    # 「厳密に勝つ」が成り立たないので、C1 はその頁で必ず落ちる
    flat = [p['page'] for p in pages
            if max(by[(p['page'], a)][f'F1@{MAIN}'] for a in KEYS + ref) == 0.0]
    if flat:
        print(f'    **ただし C1 の形には欠点が在る** —— 腕も帰無値も全部 0.000 の頁が '
              f'{len(flat)}/{len(pages)} 在り、そこでは「厳密に勝つ」が成り立たない。'
              f'**C1 はその頁で必ず落ちる**: ' + ' / '.join(x[-11:] for x in flat))
        sub = [p for p in pages if p['page'] not in flat]
        w2 = [k for k in KEYS
              if all(by[(p['page'], k)][f'F1@{MAIN}']
                     > by[(p['page'], n)][f'F1@{MAIN}']
                     for p in sub for n in ref)]
        print(f'    点の付く {len(sub)} 頁だけで読むと、勝つ鍵: '
              f'**{" ".join(w2) if w2 else "無し"}**'
              f'(**これは登録した C1 ではない。隣に置く値である**)')
    perpage = {}
    for p in pages:
        b = max(KEYS, key=lambda k: by[(p['page'], k)][f'F1@{MAIN}'])
        perpage[p['page']] = (b, by[(p['page'], b)][f'F1@{MAIN}'])
    print('  C3 頁ごとに最も高い鍵: ' + ' / '.join(
        f'{pg[-11:]} {b}({f:.3f})' for pg, (b, f) in perpage.items()))
    lose = [k for k in KEYS
            if any(by[(p['page'], k)][f'F1@{MAIN}']
                   < by[(p['page'], 'N3')][f'F1@{MAIN}'] for p in pages)]
    print(f'  C4 どこかの頁で N3(全部の可視の葉)に負ける鍵: '
          f'**{" ".join(lose) if lose else "無し"}**')
    print('\n**棒はこの走行の後に、帰無値を見てから登録する**(追補179)。'
          '**この点数を「棒に届いた/届かない」と読まない**')
    print('**precision は下界である** —— 標準は作者の面を束ねている(追補31)')
    print(f'\n表: {OUT / "arm_scores.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
