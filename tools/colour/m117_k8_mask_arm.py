"""原案の腕 K8: **`has_mask` で引いた層を、マスクの見える側で形を取って面として申告する**。

**事前登録は 追補196**(走行前に commit `10c7e8d`)。枠は 追補179(第2段 `m114`)のまま、
**採点は `m114.score` を import する。書き直さない。** 母集団 **`pop_4pages_colour`**。

K8 = 追補183 §B のとおり / K8r = 追補188〜190 の材料(全面を隠す・全面を通すを除き、
作者の裁定 線の層 追補189・ハイライト 追補190 を重ねた形。追補196 の訂正)。
層の形 = **マスクの見える側 ∧ 層のアルファ**(ラスタの無い塗りつぶし層はアルファ全面)。
K2〜K7・N0〜N5 は第2段と同じ(アルファだけ)で、**第2段の頁別の行を再現することを門にする**。
**NC2(提案の床を外す)を入れる** —— 第2段で登録したのに測っていなかったもの。
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
import m87_split_standard_by_occlusion as m87                        # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402
import m113_mask_key_agreement as m113                               # noqa: E402
import m114_proposal_arm_vs_nulls as m114                            # noqa: E402

OUT = Path('results/m117_k8_mask_arm')
V3 = Path('results/m104_b_standard_v3/b_standard_v3.csv')
M114 = Path('results/m114_proposal_arm/arm_scores.csv')
MD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/dataset_colour_pilot_masks_20261101')
MM = MD / 'layer_masks_manifest.json'
POP = ('page0002', 'page0003', 'skima12H', 'skima12chara')
OLD = ('K2', 'K3', 'K4', 'K5', 'K6', 'K7')
NEW = ('K8', 'K8r')
NULLS = m114.NULLS
THR, MAIN = m114.THR, m114.MAIN
LINE = ('線画',)                                                     # `m116` と同じ語
HIGHLIGHT = ('high', 'ハイライト', 'ﾊｲﾗｲﾄ', '発光', '光', 'glow', 'shine', '艶')
FILTER_TYPE = '4098'
# 門の期待値(**こちらの外、または過去の登録から来た数**。追補196)
IC5_K8 = {'page0002': 14, 'page0003': 0, 'skima12H': 1, 'skima12chara': 3}
IC5_DROP4098 = 2
IC6_K8R = {'page0002': 12, 'page0003': 0, 'skima12H': 0, 'skima12chara': 0}


def ok(b):
    return 'PASS' if b else '**FAIL**'


def short(page):
    return next(k for k in POP if page.endswith(k))


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
    prev = {(r['page'], r['arm']): r
            for r in csv.DictReader(open(M114, encoding='utf-8'))}
    pages = [p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
             if any(p['page'].endswith(k) for k in POP)]
    mm = json.load(open(MM, encoding='utf-8'))['layers']
    masks = {(r['page_path'], int(r['layer_id'])): r for r in mm}
    print(f'母集団 `pop_4pages_colour` = {len(pages)} 頁 / 主値 F1@{MAIN}\n')

    # ---- IC7 マスクの向き(追補188 §B)
    inv = sum(1 for r in mm if int((m114_read(MD / r['mask_file']) <= 127).sum())
              == r['mask_px_ge128'])
    print(f'IC7 マスクの向き: 反転が一致 {inv}/{len(mm)}  {ok(inv == len(mm) == 111)}')
    if inv != len(mm):
        print('**向きが決まらない。止める**')
        return 1

    rows_out, fails, nc2, nc3, bundle = [], 0, [], [], []
    k8n, k8drop, k8rn = {}, 0, {}
    for p in pages:
        pg = short(p['page'])
        ink, bnd, floor_px, _, _ = m106.build(p, byp, kinds)
        srows, under = m106.faces(bnd, ink, floor_px)
        rs = sorted(r['ratio'] for r in srows)
        m_ = rs[len(rs)//2] if len(rs) % 2 else (rs[len(rs)//2-1]+rs[len(rs)//2])/2
        got = dict(closed=len(srows), med=round(m_, 4),
                   lt05=sum(1 for r in srows if r['ratio'] < 0.5), under=under)
        ic1 = all(abs(got[x] - known[p['page']][x]) < 1e-9 for x in got)
        print(f'{pg:<14} 標準の面 {len(srows):>4} / 床 {floor_px:>6} px  IC1 {ok(ic1)}')
        fails += not ic1
        H, W = bnd.shape
        _, std_lab, _, _ = cv2.connectedComponentsWithStats(
            (~bnd).astype(np.uint8), connectivity=4)
        keep = np.fromiter((r['face'] for r in srows), int, len(srows))
        std_area = {r['face']: r['area'] for r in srows}
        std_lab = np.where(np.isin(std_lab, keep), std_lab, 0)
        del bnd

        K, _, body = m113.keyset(byp[p['source']], kinds, p['source'])
        # ---- K8 / K8r(構造で引き、K8r は裁定を名前で重ねる。`m116` と同じ)
        k8, k8r, k8r_drop = [], [], collections.Counter()
        for r in body:
            if r['has_mask'] not in ('True', 'true', '1'):
                continue
            if r['layer_type'] == FILTER_TYPE:
                k8drop += 1
                continue
            if r['layer_type'] not in ('2', '3'):
                continue
            k8.append(r)
            nm = r['name'] or ''
            mk = masks.get((p['source'], int(r['layer_id'])))
            ge = mk['mask_px_ge128'] if mk else None   # **見える側の画素数。PNG の生の値ではない**
            if ge == 0:
                k8r_drop['全面隠す'] += 1
            elif ge is not None and ge >= mk['canvas'][0] * mk['canvas'][1]:
                k8r_drop['全面通す'] += 1
            elif any(t in nm for t in LINE):
                k8r_drop['線'] += 1
            elif any(t in nm.lower() or t in nm for t in HIGHLIGHT):
                k8r_drop['ハイライト'] += 1
            else:
                k8r.append(r)
        k8n[pg] = len(k8)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        grid = (H, W)
        cache, alphas = {}, {}
        for r in body:
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is not None:
                full = np.zeros(grid, bool)
                full[:a.shape[0], :a.shape[1]] = a
                alphas[str(r['layer_id'])] = full
        cache.clear()

        def shape(r):
            mk = masks.get((p['source'], int(r['layer_id'])))
            if mk is None:
                return None, 'マスクが納品に無い'
            g = m114_read(MD / mk['mask_file'])[:H, :W] <= 127
            a = alphas.get(str(r['layer_id']))
            return (g if a is None else (g & a)), ('アルファ無し' if a is None else
                                                   ('アルファ全面' if a.all() else 'アルファ部分'))

        arms, why = {}, collections.Counter()
        for kk in OLD:
            props, u = [], 0
            for lid in sorted(K[kk], key=int):
                if lid in alphas:
                    u += m114.comps_of(alphas[lid], floor_px, props, lid)
            arms[kk] = (props, u)
        same_as_mask = []
        for nm_, sel in (('K8', k8), ('K8r', k8r)):
            props, u = [], 0
            for r in sorted(sel, key=lambda r: int(r['layer_id'])):
                s, w = shape(r)
                why[(nm_, w)] += 1
                if s is None:
                    continue
                if nm_ == 'K8r':
                    g = m114_read(MD / masks[(p['source'], int(r['layer_id']))]['mask_file'])[:H, :W] <= 127
                    same_as_mask.append(bool((g == s).all()))
                u += m114.comps_of(s, floor_px, props, str(r['layer_id']))
            arms[nm_] = (props, u)
        k8rn[pg] = len(arms['K8r'][0])
        print(f'{"":<14} K8 {len(k8)} 層 / K8r {len(k8r)} 層(K8 から除いた: '
              + (', '.join(f'{a} {b}' for a, b in k8r_drop.items()) or '0') + ')')
        print(f'{"":<14} K8r → 面 {len(arms["K8r"][0])}'
              f'  形の取り方: ' + ', '.join(f'{a}:{b} {c}' for (a, b), c in sorted(why.items()))
              + (f'  / K8r でアルファ∧マスク = マスク {sum(same_as_mask)}/{len(same_as_mask)}'
                 if same_as_mask else ''))
        if pg == 'page0002':
            fails += not (same_as_mask and all(same_as_mask))

        # ---- 帰無値(`m114` と同じ作り方)
        arms['N0'] = ([], 0)
        pr = []
        m114.comps_of(np.ones(grid, bool), floor_px, pr, 'all')
        arms['N1'] = (pr, 0)
        g_ = max(1, int(round(max(1, len(srows)) ** 0.5)))
        pr, u = [], 0
        for iy in range(g_):
            for ix in range(g_):
                y0, y1, x0, x1 = iy*H//g_, (iy+1)*H//g_, ix*W//g_, (ix+1)*W//g_
                a = (y1 - y0) * (x1 - x0)
                if a == 0:
                    continue
                if a < floor_px:
                    u += 1
                    continue
                pr.append(('grid', a, y0, y1, x0, x1, np.ones((y1-y0, x1-x0), bool)))
        arms['N2'] = (pr, u)
        pr, u = [], 0
        for lid in sorted(K['K1'], key=int):
            if lid in alphas:
                u += m114.comps_of(alphas[lid], floor_px, pr, lid)
        arms['N3'] = (pr, u)
        arms['N4'] = ([('std', r['area'], r['y0'], r['y1'], r['x0'], r['x1'],
                        std_lab[r['y0']:r['y1'], r['x0']:r['x1']] == r['face'])
                       for r in srows], 0)
        pr = []
        u = m114.comps_of(ink, floor_px, pr, 'ink')
        arms['N5'] = (pr, u)

        for nm_ in OLD + NEW + NULLS:
            props, u = arms[nm_]
            s = m114.score(props, std_lab, std_area)
            row = dict(page=p['page'], arm=nm_, n_prop=len(props),
                       dropped_floor=u, n_std=len(srows))
            for t in THR:
                prc, rcl, f1, hit = s[t]
                row.update({f'P@{t}': prc, f'R@{t}': rcl, f'F1@{t}': f1, f'hit@{t}': hit})
            rows_out.append(row)
            # ---- NC3 左右反転
            if nm_ in ('K7', 'K8', 'K8r', 'N3'):
                nc3.append((pg, nm_, row[f'F1@{MAIN}'],
                            m114.score(props, std_lab, std_area, flip_w=W)[MAIN][2]))
        # ---- NC2 床を外す(誤り): 床 1 px で作り直す
        for nm_ in OLD + NEW + ('N3', 'N5'):
            pr0 = []
            if nm_ in OLD or nm_ == 'N3':
                ids = K['K1'] if nm_ == 'N3' else K[nm_]
                for lid in sorted(ids, key=int):
                    if lid in alphas:
                        m114.comps_of(alphas[lid], 1, pr0, lid)
            elif nm_ == 'N5':
                m114.comps_of(ink, 1, pr0, 'ink')
            else:
                for r in sorted(k8 if nm_ == 'K8' else k8r, key=lambda r: int(r['layer_id'])):
                    s, _ = shape(r)
                    if s is not None:
                        m114.comps_of(s, 1, pr0, str(r['layer_id']))
            props, u = arms[nm_]
            s0 = m114.score(pr0, std_lab, std_area)[MAIN]
            s1 = m114.score(props, std_lab, std_area)[MAIN]
            nc2.append((pg, nm_, len(props), u, len(pr0), s1[0], s0[0]))
            del pr0
        # ---- 束ね(記述): K8r の各面に 50% 以上入る標準の面の数
        for _, a, y0, y1, x0, x1, m in arms['K8r'][0]:
            sub = std_lab[y0:y1, x0:x1][m]
            ids, cnt = np.unique(sub[sub > 0], return_counts=True)
            inside = [int(i) for i, c in zip(ids, cnt) if c >= 0.5 * std_area[int(i)]]
            bundle.append((pg, a / (H * W), len(inside)))
        del std_lab, arms, alphas, ink

    # ---------------------------------------------------------------- 門
    print()
    by = {(r['page'], r['arm']): r for r in rows_out}
    keys_ = ('n_prop', 'dropped_floor', 'n_std') + tuple(f'hit@{t}' for t in THR)
    bad = [(pg, a) for (pg, a), r in by.items() if a in OLD + NULLS
           and any(int(float(prev[(pg, a)][k])) != r[k] for k in keys_)]
    print(f'IC4 第2段(`m114`)の頁別の行を再現(K2〜K7・N0〜N5 × 4頁、提案・床・当たり)'
          f'  不一致 {len(bad)}  {ok(not bad)}' + (f'  {bad}' if bad else ''))
    fails += bool(bad)
    print(f'IC5 K8 の層 期待 {IC5_K8} 実測 {k8n} / 除いた 4098 {k8drop}(期待 {IC5_DROP4098})'
          f'  {ok(k8n == IC5_K8 and k8drop == IC5_DROP4098)}')
    fails += not (k8n == IC5_K8 and k8drop == IC5_DROP4098)
    print(f'IC6 K8r の面 期待 {IC6_K8R} 実測 {k8rn}  {ok(k8rn == IC6_K8R)}'
          f'(かつ page0002 でアルファ∧マスク = マスク。上の行)')
    fails += k8rn != IC6_K8R
    ic2 = all(abs(by[(p['page'], 'N4')][f'F1@{t}'] - 1) < 1e-9 for p in pages for t in THR)
    ic3 = all(by[(p['page'], 'N0')][f'R@{MAIN}'] == 0 for p in pages)
    print(f'IC2 N4 が全閾値で F1 = 1.000  {ok(ic2)}(**恒等。算術である**) / IC3 N0 の recall = 0  {ok(ic3)}')
    fails += (not ic2) + (not ic3)
    if fails:
        print(f'\n**門が {fails} 本落ちた。以降の行は読まない**')
        return 1

    with open(OUT / 'arm_scores.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    cols = [short(p['page']) for p in pages]
    print(f'\n=== A. `F1@{MAIN}`(**頁ごと。プールしない**)/ 当たり / 提案 ===')
    print(f'{"腕":<5}' + ''.join(f'{c:>22}' for c in cols) + f'{"最悪":>8}')
    for nm_ in NEW + OLD + NULLS:
        v = [by[(p['page'], nm_)] for p in pages]
        print(f'{nm_:<5}' + ''.join(
            f'{r[f"F1@{MAIN}"]:>9.3f} {r[f"hit@{MAIN}"]:>4}/{r["n_prop"]:<7}' for r in v)
            + f'{min(r[f"F1@{MAIN}"] for r in v):>8.3f}')
    print('  **K8・K8r は「マスク ∧ アルファ」、他はアルファだけ**(追補196)。連言は最悪で読む(条例1)')

    print(f'\n=== B. 精度と再現率 `@{MAIN}` と閾値の形(**棒に使わない**)===')
    for nm_ in NEW:
        for p in pages:
            r = by[(p['page'], nm_)]
            if not r['n_prop']:
                continue
            print(f'  {nm_:<4} {short(p["page"]):<13} 提案 {r["n_prop"]:>3} / 標準 {r["n_std"]:>3}  '
                  + '  '.join(f'@{t}: P{r[f"P@{t}"]:.3f} R{r[f"R@{t}"]:.3f} F1 {r[f"F1@{t}"]:.3f} 当{r[f"hit@{t}"]}'
                              for t in THR))
    print('  **但し書き**: precision は下界(標準は作者の面を束ねる。追補31)/ '
          '`page0002` の標準は 3面なので当たりは最大 3、P@0.5 は最大 3/12 = 0.25(追補196)')
    n_all = sum(by[(p['page'], 'K8r')]['n_std'] for p in pages)
    h_all = sum(by[(p['page'], 'K8r')][f'hit@{MAIN}'] for p in pages)
    print(f'  4頁の recall(K8r): 当たり {h_all} / 標準 {n_all} = {h_all / n_all:.4f}'
          f'(**上界 12/525 = 0.0229 の隣に置く**。プールした値は棒に使わない)')

    print('\n=== C. 負の対照 ===')
    print('  NC2 床を外す(誤り): 提案(床なし) = 提案 + 床で落ちた / P@0.5 が上がらない')
    for pg, nm_, n1, u, n0, p1, p0 in nc2:
        eq = n0 == n1 + u
        pdn = (n1 == 0 and n0 == 0) or (not np.isnan(p0) and (np.isnan(p1) or p0 <= p1 + 1e-12))
        print(f'    {pg:<13} {nm_:<4} {n1:>6} + {u:>6} = {n1+u:>6} 対 {n0:>6}  {ok(eq)}'
              f'   P {p1:.3f} → {p0:.3f}  {ok(pdn)}')
    print('  NC3 左右反転(誤り)')
    for pg, nm_, b, f in nc3:
        v = ('**正位置が 0.000。定義できない(判定保留)**' if b == 0 else
             'PASS' if f < b else '**下がらない。FAIL**')
        print(f'    {pg:<13} {nm_:<4} {b:.3f} → {f:.3f}  {v}')
    print('  NC1 閾値 0.3 で recall が下がらない')
    for nm_ in NEW + OLD + NULLS:
        okk = all(by[(p['page'], nm_)][f'R@0.3'] >= by[(p['page'], nm_)][f'R@{MAIN}'] - 1e-12
                  for p in pages)
        print(f'    {nm_:<4} {ok(okk)}', end='')
    print()

    print('\n=== D. 判定(追補179 の C1〜C4 + 追補196 の読み。**棒ではない**)===')
    ref = ('N1', 'N2', 'N3', 'N5')
    for nm_ in NEW:
        win = all(by[(p['page'], nm_)][f'F1@{MAIN}'] > by[(p['page'], n)][f'F1@{MAIN}']
                  for p in pages for n in ref)
        lose = [short(p['page']) for p in pages
                if by[(p['page'], nm_)][f'F1@{MAIN}'] < by[(p['page'], 'N3')][f'F1@{MAIN}']]
        r2 = by[(pages[0]['page'], nm_)] if short(pages[0]['page']) == 'page0002' else None
        print(f'  {nm_}: C1 全頁で勝つ = {"はい" if win else "いいえ"}'
              f'(**K8r は3頁で0面なので構造上落ちる**)/ C4 N3 に負ける頁 = {lose or "無し"}')
        if r2:
            print(f'     page0002: 当たり {r2[f"hit@{MAIN}"]} / 提案 {r2["n_prop"]} / 標準 {r2["n_std"]}'
                  f'  → ' + ('**当たり ≥ 1: マスクの面が線の閉じた面と重なる例が在る(1頁)**'
                             if r2[f'hit@{MAIN}'] else
                             '**当たり 0: マスクの面は線の閉じた面と1対1にならない**'))
    print('\n=== E. 束ね(記述。棒に使わない): K8r の各面に 50% 以上入る標準の面の数 ===')
    for pg, fr, n in bundle:
        print(f'    {pg:<13} 面 {fr*100:>6.2f}%  標準の面 {n}')
    print('\n**棒は置いていない**(追補196)。**採用4頁(gakusai3 の30面)は標準が無いので入っていない**')
    print(f'表: {OUT / "arm_scores.csv"}')
    return 0


def m114_read(path):
    g = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return g[..., 0] if g.ndim == 3 else g


if __name__ == '__main__':
    sys.exit(main())
