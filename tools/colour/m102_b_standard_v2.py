#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M102: (b) の標準 第2版 —— 作者の裁定 R1〜R5 を入れて走る。**

**事前登録** `doc/prereg_b_standard_v2_20261031.md`(**走る前に commit した** `aaf5aab`)。
**第1版**(`m99`、追補152・153)**は撤回しない。名前を付けて並べる**(規律10)。

  線   = **名前に `線画` を含む可視の<ins>葉</ins>**(作者の裁定。`m99.is_line` を**写さず import**)
  面   = **墨の補集合の4連結**(前景8連結の双対。**`T-M28`**。`m99.faces_from`)
  主値 = **P2 = 面の境目に占める<ins>線</ins>の割合(線率)の、頁ごとの中央値(T3)**
  棒   = **C1: 頁ごとに P2(T3) > P2(T0)。6頁すべてで満たすときだけ通る**(**連言は最悪で判定。T-M25**)

**段**(登録 §Arms。**同じ走行で全部出す**):

  Z    境界なし(**恒等**。閉じた面 0 / 被覆 0%)
  L    **線だけ**(閉じる力の下限。第1版の S1)。**線率は定義上 1.0 なので C1 の判定に入れない**
  T0   **第1版の S3 の再現**(R1〜R4 を入れない)—— **既知セル**
  T1   **R1 のみ**(縁は<ins>外周に通じている領域</ins>にだけ足す)
  T2   **R1 + R2**(クリップ層・作者が名指しした層は縁を出さない)
  T3   **R1 + R2 + R3 + R4** —— **主値**
  T3c  **T3 + 非クリップの `color` も落とす**(**下の「登録の読み方」(2)**)

**登録の読み方(走る前に書く。登録の文を広げも狭めもしないため)**:

1. **登録 §R2 の表は (ii)「資料」を §R3 に送り、§Arms は R3 を T3 に置いている。**
   → **資料は T3 で落ちる。T2 は (i) クリップ と (iii) 名指し だけ。**
2. **(iii) の定義節は「作者が名指しで『境界に使わない』と言った層」**で、
   **作者が名指ししたのは `参考` / `p` / `high light` / `over lay` / `deep` の5つ**
   (追補153 §F2 の問いの一覧に答えた。**`color` は問いに入っていない**)。
   → **(iii) はこの5つ(とフォルダーならその子孫)で実装する。**
   **登録 §R2 の括弧書きは「`p` / `コマ 3` / 残りの `color`」と書いており、`color` を含む** ——
   **作者が名指ししていない層なので主の段には入れず、`T3c` として<ins>隣に</ins>出す。**
   **C1 の判定は登録どおり T3 で行う。**
3. **`参考` は (iii) で落ちる**(作者が名指しした)。**R3 は、作者が名指ししていない資料
   (`<資料>` など)を構造で拾う役である。**

**R1 の実装**(登録 §R1): **線だけで面を作り、<ins>外周に通じている領域</ins>の中にだけ
遮蔽物の縁を足して作り直す。1回で止める**(反復しない)。

**器具は写さずに import する**: `m12`(床)/ `m16`(木・層の表)/ `m77`(非表示)/
`m87`(アルファ・遮蔽範囲)/ `m99`(線の同定・縁・面)/ `m101`(作者に見せた窓)。

  <venv>/python tools/colour/m102_b_standard_v2.py [--smoke]
"""
import collections
import csv
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402
import m77_standard_visible as m77                                 # noqa: E402
import m87_split_standard_by_occlusion as m87                      # noqa: E402
import m99_b_standard as m99                                       # noqa: E402
import m101_why_split as m101                                      # noqa: E402

OUT = pathlib.Path('results/m102_b_standard_v2')
KIND_CSV = pathlib.Path('../lineart/inbox/f_request3_layer_kind_rendered_20261026.csv')
PAPER = m16.PAPER_TYPE
MULTIPLY = '2'
STAGES = ('Z', 'L', 'T0', 'T1', 'T2', 'T3', 'T3c')
JUDGED = ('T0', 'T1', 'T2', 'T3', 'T3c')      # 線率を読む段(Z は面が無い)
NAMED = ('参考', 'p', 'high light', 'over lay', 'deep')   # **作者が名指しした5つ**
SKETCH = ('下絵', 'sketch set', '下描')                    # R4
TONE_CORRECTION = 4096                                     # `layer_type & 4096` = 色調補正
RECT_BAR = 0.95                                            # R3(登録 §R3。分布は追補155 §A)
FULL_CANVAS = 0.99                                         # 外接箱が画布を覆う割合

# ---------------------------------------------------------------- 既知セル(登録 §IC)
IC1_FACES = {'000_2017_illustration_sunshine2_page0019': 536,
             '000_2017_illustration_sunshine2_page0002': 226,
             '000_2018_illust201710_illust201710_page0003': 26,
             '000_2018_illust201710_illust201710_page0005': 524,
             '000_2018_skima11_skima11_skima12H': 793,
             '000_2018_skima11_skima11_skima12chara': 279}
IC1_COVER = {'000_2017_illustration_sunshine2_page0019': 38.30,
             '000_2017_illustration_sunshine2_page0002': 78.73,
             '000_2018_illust201710_illust201710_page0003': 6.12,
             '000_2018_illust201710_illust201710_page0005': 18.78,
             '000_2018_skima11_skima11_skima12H': 38.42,
             '000_2018_skima11_skima11_skima12chara': 35.34}
IC4_L_FACES = {'000_2017_illustration_sunshine2_page0019': 49,
               '000_2017_illustration_sunshine2_page0002': 3,
               '000_2018_illust201710_illust201710_page0003': 15,
               '000_2018_illust201710_illust201710_page0005': 60,
               '000_2018_skima11_skima11_skima12H': 56,
               '000_2018_skima11_skima11_skima12chara': 134}
IC2_A_ISLANDS = m99.KC_A_ISLANDS
IC5_EXPECT = {'線画': (92, 0), '描画色(MASK)': (156, 0)}     # (枚数, うちクリップ)
IC6_EXPECT = {'参考': 6, '<資料>': 5}                        # R3 が拾う数。取りこぼし 1
IC7_PAGE = '000_2017_illustration_sunshine2_page0002'        # `線画` 6枚を除外しない
# C3 の予言(登録 §Pass criteria)。窓は `m101.ASK` と同一
C3 = {'000_2017_illustration_sunshine2_page0019': ('T2', ('p', 'コマ 3')),
      '000_2018_skima11_skima11_skima12H': ('T3', ('参考', 'gob 2', 'gob_chara 2',
                                                   'レイヤー 8 2')),
      '000_2017_illustration_sunshine2_page0002': ('T2', ('color', 'high light',
                                                          'over lay'))}
C4_FACTOR = 2.0    # **C4 の読み: T3 の被覆が L の 2倍未満なら「L と同じ水準」**(走る前に決めた)


def anc_parts(r):
    return [s for s in (r['ancestors'] or '').split('/') if s]


def is_named(r):
    """**R2(iii)**: 作者が名指しした層。**フォルダーなら子孫も**(祖先の名で見る)。"""
    nm = (r['name'] or '').strip()
    if nm in NAMED:
        return True
    return any(a.strip() in NAMED for a in anc_parts(r))


def is_sketch(r):
    """**R4**: `下絵` / `sketch set` / `下描` の子孫。**ただし `線画` は除外しない**(案イ)。"""
    if m99.is_line(r):
        return False
    nm = (r['name'] or '')
    parts = anc_parts(r) + [nm]
    return any(t in a for a in parts for t in SKETCH)


def is_tone_correction(r):
    try:
        return bool(int(r['layer_type'] or 0) & TONE_CORRECTION)
    except ValueError:
        return False


def rect_of(a, px):
    """**矩形度 = 覆いの面積 ÷ 外接箱の面積**、と**外接箱が画布を覆う割合**。"""
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return 0.0, 0.0
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    return len(ys) / (w * h), (w * h) / px


def load_kinds():
    """納品の層種別(`v_layer_kind`)。**結合は `page_path` + `layer_id`**(名前で結合しない)。"""
    out = {}
    with open(KIND_CSV, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            out[(r['page_path'], str(r['layer_id']))] = r['v_layer_kind']
    return out


def faces_and_ratios(boundary, ink, floor_px):
    """閉じた面・開いた領域・床で落ちた数・**各面の線率**。

    **線率** = その面の境目(8近傍で接する境界の画素)のうち**線の墨**に当たる割合。
    **面ごとに外接箱の中だけで数える**(頁ごとの全画布で膨張させない)。
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        (~boundary).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    closed, open_, under, ratios = [], [], 0, []
    k3 = np.ones((3, 3), np.uint8)
    H, W = boundary.shape
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < floor_px:
            under += 1
            continue
        if i in border:
            open_.append(a)
            continue
        closed.append(a)
        x0 = max(0, int(st[i, cv2.CC_STAT_LEFT]) - 1)
        y0 = max(0, int(st[i, cv2.CC_STAT_TOP]) - 1)
        x1 = min(W, x0 + int(st[i, cv2.CC_STAT_WIDTH]) + 3)
        y1 = min(H, y0 + int(st[i, cv2.CC_STAT_HEIGHT]) + 3)
        m = (lab[y0:y1, x0:x1] == i)
        ring = cv2.dilate(m.astype(np.uint8), k3).astype(bool) & ~m \
            & boundary[y0:y1, x0:x1]
        nr = int(ring.sum())
        ratios.append((int((ring & ink[y0:y1, x0:x1]).sum()) / nr) if nr else 0.0)
        del m, ring
    del lab, st
    return closed, open_, under, ratios


def stat(v):
    if not v:
        return (0.0, 0.0, 0.0, 0, 0.0)
    a = np.asarray(v, float)
    return (float(a.min()), float(np.median(a)), float(a.max()),
            int((a < 0.5).sum()), float((a < 0.5).mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = '--smoke' in sys.argv
    kinds = load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']

    # ------------------------------------------------- 母集団(登録どおり。動かさない)
    keep, dropped = [], []
    for p in pages:
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        n = sum(1 for r in rows_vis if r['layer_type'] != PAPER and m99.is_line(r))
        (keep if n > 0 else dropped).append((p, n))
    print(f'母集団 `pop_6pages_lineart` = {len(keep)} 頁'
          f'(`pop_8pages_registered` = {len(pages)} 頁。外した2頁は `線画` が0枚)')
    for p, n in dropped:
        print(f'    外した頁 {p["page"]:44s} 線画 {n}')
    if smoke:
        keep = keep[-2:]
        print('**--smoke: 最後の2頁だけ。門の頁数は判定しない**')

    drops = collections.Counter()
    occ_drop = collections.Counter()
    ic5 = collections.Counter()
    r3_hits, r4_rows, nc4_rows, hole_rows = [], [], [], []
    per_page, c3_rows, t0 = [], [], time.time()

    for p, n_line in keep:
        rows_vis, d = m77.visible_rows(byp[p['source']])
        drops.update(d)
        _, kids = m16.build_tree(rows_vis)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        px = grid[0] * grid[1]
        floor_px = max(1, round(m12.FLOOR * px))
        cache = {}
        body = [r for r in rows_vis if r['layer_type'] != PAPER]

        # ---- IC5 / 穴(条例6: 納品の層種別の表に当たらない行)
        for r in body:
            if m99.is_line(r):
                ic5['線画'] += 1
                ic5['線画 clip'] += (r['clip'] == '1')
            if r['draw_color_enable'] == '1' and r['tone_enabled'] != '1':
                ic5['描画色(MASK)'] += 1
                ic5['描画色(MASK) clip'] += (r['clip'] == '1')
            k = kinds.get((p['source'], str(r['layer_id'])))
            if k is None:
                hole_rows.append((p['page'], r['layer_id'], r['name']))
            ic5['層種別 ' + (k or '**表に無い**')] += 1

        # ---- 線(R4 は `線画` を除外しない = 案イ。IC7)
        ink = np.zeros(grid, bool)
        n_raster = 0
        for r in body:
            if not m99.is_line(r):
                continue
            if is_sketch(r):                       # **起きない設計。起きたら印字**
                drops['**線が R4 で落ちた(案イ では起きないはず)**'] += 1
                continue
            if any(t in a for a in anc_parts(r) for t in SKETCH):
                r4_rows.append((p['page'], r['layer_id'], r['name'],
                                '**線画なので残した**'))
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is None:
                drops['線: ラスタが無い層'] += 1
                continue
            ink[:a.shape[0], :a.shape[1]] |= a
            n_raster += 1

        # ---- 遮蔽物の縁を4つの集合に分ける(1回の走査)
        E = {k: np.zeros(grid, bool) for k in ('all', 'r2', 'r3r4', 'r3r4c')}
        nE = collections.Counter()
        reason = collections.Counter()
        for r in body:
            if (r.get('opacity') or '0') == '0':
                occ_drop['不透明度が0'] += 1
                continue
            if (r.get('composite') or '0') == MULTIPLY:
                occ_drop['乗算(作者の集合に入らない)'] += 1
                continue
            m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True, occ_drop)
            if m is None:
                continue
            leaf = r['folder'] != 'True'
            # **矩形度は<ins>素のアルファ</ins>で測る** —— 登録 §R3 の分布(追補155 §A)が
            # 素のアルファの上の数だから。**クリップと交わらせた後の覆いで測ると
            # 全画布の 98 枚が消えて、NC4 の期待値と比べられなくなる**(T-M3)。
            rect, boxr = rect_of(m87.alpha_of(r['layer_id'], files, grid, cache), px) \
                if leaf else (0.0, 0.0)
            kind = kinds.get((p['source'], str(r['layer_id'])))
            mat = leaf and (kind == 'image_material'
                            or (not is_tone_correction(r) and rect >= RECT_BAR
                                and boxr < FULL_CANVAS))
            if leaf and not is_tone_correction(r) and rect >= RECT_BAR:
                nc4_rows.append((p['page'], r['layer_id'], r['name'], rect, boxr,
                                 boxr >= FULL_CANVAS))
            clip = r['clip'] == '1'
            named = is_named(r)
            sk = is_sketch(r)
            colr = 'color' in (r['name'] or '').lower() and not clip
            sets = ['all']
            if not clip and not named:
                sets.append('r2')
                if not mat and not sk:
                    sets.append('r3r4')
                    if not colr:
                        sets.append('r3r4c')
            if clip:
                reason['R2(i) クリップ'] += 1
            elif named:
                reason['R2(iii) 作者が名指し'] += 1
            elif mat:
                reason['R3 資料'] += 1
            elif sk:
                reason['R4 下絵の子孫'] += 1
            elif colr:
                reason['(T3c のみ) 非クリップの `color`'] += 1
            if mat:
                r3_hits.append((p['page'], r['layer_id'], r['name'], kind, rect, boxr,
                                '/'.join(anc_parts(r))))
                if m99.is_line(r):
                    r3_hits.append((p['page'], r['layer_id'], r['name'],
                                    '**線画が資料と判定された**', rect, boxr, ''))
            if sk:
                r4_rows.append((p['page'], r['layer_id'], r['name'], '外した'))
            e = m99.inner_edge(m)
            del m
            for k in sets:
                E[k] |= e
                nE[k] += 1
            # C3: 名指しの層の縁が窓の中に残るか
            if p['page'] in C3:
                x0, y0, _ = m101.ASK[p['page']]
                nm = (r['name'] or '').strip()
                if any(t == nm or t in nm for t in C3[p['page']][1]):
                    sub = e[y0:y0 + m101.WIN, x0:x0 + m101.WIN]
                    if sub.any():
                        c3_rows.append({'page': p['page'], 'layer_id': r['layer_id'],
                                        'name': nm, 'px': int(sub.sum()),
                                        **{f'in_{k}': (k in sets) for k in E}})
            del e
        cache.clear()

        # ---- R1: 線だけの面で「外周に通じている領域」を取る
        nl, labl, stl, _ = cv2.connectedComponentsWithStats(
            (~ink).astype(np.uint8), connectivity=4)
        bl = set(np.unique(np.r_[labl[0], labl[-1], labl[:, 0], labl[:, -1]])) - {0}
        open_mask = np.isin(labl, np.fromiter(bl, int, len(bl)))
        del labl, stl
        print(f'  {p["page"]:44s} 線 {n_raster:>3} / 外周に通じた領域 '
              f'{open_mask.mean()*100:5.2f}% / 縁を出した層 '
              + ' '.join(f'{k}={nE[k]}' for k in ('all', 'r2', 'r3r4', 'r3r4c')),
              flush=True)

        # ---- (a) の標準(隣に置くだけ。プールしない)
        comps = []
        (_, _), (lab_a, _), _ = m16.measure_standard(p, rows_vis, grid,
                                                     floor_px, out_comps=comps)
        comps = [c for c in comps if c['area'] >= floor_px]
        isl = m87.areas_from(lab_a, comps, floor_px)
        row = {'page': p['page'], 'n_line': n_line, 'n_line_raster': n_raster,
               'canvas_px': px, 'floor_px': floor_px, 'a_islands': len(isl),
               'a_cover': sum(int(x['area']) for x in isl) / px,
               'open_after_lines': float(open_mask.mean()),
               **{f'n_edge_{k}': nE[k] for k in E},
               **{f'drop_{k}': v for k, v in reason.items()}}
        del lab_a, comps, isl

        for name, bnd in (('Z', np.zeros(grid, bool)),
                          ('L', ink),
                          ('T0', ink | E['all']),
                          ('T1', ink | (E['all'] & open_mask)),
                          ('T2', ink | (E['r2'] & open_mask)),
                          ('T3', ink | (E['r3r4'] & open_mask)),
                          ('T3c', ink | (E['r3r4c'] & open_mask))):
            closed, open_, under, ratios = faces_and_ratios(bnd, ink, floor_px)
            lo, med, hi, nlow, flow = stat(ratios)
            row.update({f'{name}_closed_n': len(closed),
                        f'{name}_cover': sum(closed) / px,
                        f'{name}_open_n': len(open_),
                        f'{name}_open_max': (max(open_) / px) if open_ else 0.0,
                        f'{name}_under_floor': under,
                        f'{name}_ratio_min': lo, f'{name}_ratio_med': med,
                        f'{name}_ratio_max': hi,
                        f'{name}_ratio_lt05_n': nlow, f'{name}_ratio_lt05_frac': flow})
            del bnd
            print(f'      {name:3s} 面 {len(closed):>5} 被覆 {sum(closed)/px*100:6.2f}% '
                  f'線率 中央 {med*100:6.2f}% (最小 {lo*100:6.2f} 最大 {hi*100:6.2f}) '
                  f'<0.5 が {nlow:>4} ({flow*100:5.1f}%) 床落ち {under:>5} '
                  f'{time.time()-t0:6.1f}s', flush=True)

        # ---- 負の対照(頁ごと。安いものだけ)
        n8, _, st8, _ = cv2.connectedComponentsWithStats(
            (~ink).astype(np.uint8), connectivity=8)
        row['NC1_L_faces_conn8'] = sum(
            1 for i in range(1, n8) if int(st8[i, cv2.CC_STAT_AREA]) >= floor_px)
        del st8
        row['NC6_L_faces_no_floor'] = row['L_closed_n'] + row['L_under_floor']
        nc3 = ink | (E['all'] & ~E['r2'])      # NC3: クリップ層だけが縁を出す(逆)
        cl3, _, _, _ = faces_and_ratios(nc3, ink, floor_px)
        row['NC3_closed_n'] = len(cl3)
        row['NC3_cover'] = sum(cl3) / px
        del nc3, cl3, ink, open_mask
        for k in list(E):
            del E[k]
        per_page.append(row)

    # =================================================================== 門
    print('\n== `standard_visible` が落とす行(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== 遮蔽物で落ちた層(理由別。規律9)==')
    for k, v in occ_drop.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== R2〜R4 で縁を落とした層(理由別・全頁。規律9)==')
    agg = collections.Counter()
    for r in per_page:
        for k, v in r.items():
            if k.startswith('drop_'):
                agg[k[5:]] += v
    for k, v in agg.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== 納品の層種別(条例6: 「一覧に無い」は「存在しない」ではない)==')
    for k, v in sorted(ic5.items()):
        if k.startswith('層種別 '):
            print(f'  {k:46s} {v:>6}')
    print(f'  **表に無い行 {len(hole_rows)}**(宿題。内訳は下の CSV)')

    print('\n== 門(落ちたら以降の行を読まない。T-M22)==')
    g = {}
    g['IC1 頁数'] = ('—' if smoke else 6, len(keep))
    for r in per_page:
        t = r['page'][-12:]
        g[f'IC1 T0 面 {t}'] = (IC1_FACES[r['page']], r['T0_closed_n'])
        g[f'IC1 T0 被覆 {t}'] = (IC1_COVER[r['page']], round(r['T0_cover'] * 100, 2))
        g[f'IC2 (a) 島 {t}'] = (IC2_A_ISLANDS[r['page']], r['a_islands'])
        g[f'IC4 L 面 {t}'] = (IC4_L_FACES[r['page']], r['L_closed_n'])
        g[f'IC3 Z 面 {t}'] = (0, r['Z_closed_n'])
        g[f'IC3 Z 被覆 {t}'] = (0.0, round(r['Z_cover'] * 100, 2))
    for k, (n, c) in IC5_EXPECT.items():
        g[f'IC5 {k}'] = (('—', '—') if smoke else (n, c), (ic5[k], ic5[k + ' clip']))
    got6 = collections.Counter()
    for pg, lid, nm, kind, rect, boxr, anc in r3_hits:
        for key in IC6_EXPECT:
            if key in anc:
                got6[key] += 1
    for k, v in IC6_EXPECT.items():
        g[f'IC6 R3 が `{k}` で拾う'] = ('—' if smoke else v, got6[k])
    ic7 = [x for x in r4_rows if x[0] == IC7_PAGE and '残した' in x[3]]
    g['IC7 `page0002` の `線画` を残す'] = ('—' if smoke else 6, len(ic7))
    ok_all = True
    for k, (want, got) in g.items():
        ok = want == got
        ok_all &= ok or '—' in str(want)
        print(f'  {k:34s} 期待 {str(want)[:26]:>26} / 実測 {str(got)[:26]:>26}  '
              f'{"PASS" if ok else ("—" if "—" in str(want) else "**FAIL**")}')
    print('\n**門は全部 PASS**' if ok_all else '\n**門が落ちた。以降の行を読まない**(T-M22)')

    # =================================================================== 判定
    print('\n== C1(主): 頁ごとに **P2(T3) > P2(T0)**。**最悪の頁で判定**(T-M25)==')
    fail = []
    for r in per_page:
        ok = r['T3_ratio_med'] > r['T0_ratio_med']
        fail += [] if ok else [r['page']]
        print(f'  {r["page"]:44s} T0 {r["T0_ratio_med"]*100:6.2f}% → '
              f'T3 {r["T3_ratio_med"]*100:6.2f}%  '
              f'({(r["T3_ratio_med"]-r["T0_ratio_med"])*100:+6.2f} 点)  '
              f'{"PASS" if ok else "**FAIL**"}')
    print(f'  → **C1 {"PASS" if not fail else "FAIL"}**'
          f'(満たさない頁 {len(fail)}/{len(per_page)})')
    print('  **L は線率が定義上 1.0 なので C1 の判定に入れない**(登録の注)')

    print('\n== C2: 線率 < 0.5 の面の割合が T0 より T3 で減る頁 ==')
    dec = 0
    for r in per_page:
        d = r['T3_ratio_lt05_frac'] - r['T0_ratio_lt05_frac']
        dec += d < 0
        print(f'  {r["page"]:44s} T0 {r["T0_ratio_lt05_frac"]*100:6.2f}% '
              f'({r["T0_ratio_lt05_n"]:>4}面) → T3 {r["T3_ratio_lt05_frac"]*100:6.2f}% '
              f'({r["T3_ratio_lt05_n"]:>4}面)  {d*100:+6.2f} 点')
    print(f'  → **減った頁 {dec}/{len(per_page)}**'
          f'{"  **0 = R1〜R4 は却下された姿を減らしていない**" if dec == 0 else ""}')

    print('\n== C3: 却下された3窓で、該当の層の縁が消えるか(**頁ごとの予言を先に書いた**)==')
    for pg, (st_want, culprits) in C3.items():
        rows = [c for c in c3_rows if c['page'] == pg]
        if not rows:
            print(f'  {pg[-12:]:12s} 予言 {st_want}: **窓の中に該当の層の縁が無い**')
            continue
        print(f'  {pg[-12:]:12s} 予言 **{st_want}** / 該当 {culprits}')
        for c in rows:
            where = [k for k in ('r2', 'r3r4', 'r3r4c') if c[f'in_{k}']]
            gone = {'T2': 'r2' not in where, 'T3': 'r3r4' not in where}
            print(f'      L{c["layer_id"]:>4} {c["name"][:20]:20s} 窓の中 {c["px"]:>7}px  '
                  f'T2 で {"消えた" if gone["T2"] else "**残る**"} / '
                  f'T3 で {"消えた" if gone["T3"] else "**残る**"}')
        bad = [c for c in rows if c[f'in_{"r2" if st_want == "T2" else "r3r4"}']]
        print(f'      → **{st_want} で残る層 {len(bad)}**'
              f'{" = この裁定の実装が足りていない" if bad else ""}')

    print('\n== C4(この工程が丸ごと誤りである出方): T3 の被覆が L と同じ水準か ==')
    same = 0
    for r in per_page:
        hit = r['T3_cover'] < C4_FACTOR * max(r['L_cover'], 1e-12)
        same += hit
        print(f'  {r["page"]:44s} L {r["L_cover"]*100:6.2f}% / '
              f'T3 {r["T3_cover"]*100:6.2f}% / T0 {r["T0_cover"]*100:6.2f}%  '
              f'{"**L と同水準**" if hit else "上回る"}')
    print(f'  → **同水準の頁 {same}/{len(per_page)}**'
          + ('  **全頁 = 閉じ方そのものを作者に問い直す**' if same == len(per_page) else ''))

    # =================================================================== 副値
    print('\n== 副値: 段ごとの面の数 / 被覆 / 線率の中央(頁ごと。プールしない)==')
    print(f'{"page":44s} ' + ' '.join(f'{s+"面":>8}' for s in STAGES))
    for r in per_page:
        print(f'{r["page"]:44s} ' + ' '.join(f'{r[s+"_closed_n"]:>8}' for s in STAGES))
    print(f'{"":44s} ' + ' '.join(f'{s+"%":>8}' for s in STAGES))
    for r in per_page:
        print(f'{r["page"]:44s} ' + ' '.join(f'{r[s+"_cover"]*100:>8.2f}' for s in STAGES))
    print(f'{"":44s} ' + ' '.join(f'{s+"線率":>8}' for s in JUDGED))
    for r in per_page:
        print(f'{r["page"]:44s} ' + ' '.join(f'{r[s+"_ratio_med"]*100:>8.2f}'
                                            for s in JUDGED))

    print('\n== 負の対照 ==')
    print('  NC1 背景を8連結で数える(T-M28): 面が減るはず '
          '(**48→38 は `m98` の母集団の値。この走行の期待値ではない**)')
    for r in per_page:
        print(f'    {r["page"]:44s} L 4連結 {r["L_closed_n"]:>5} → '
              f'8連結 {r["NC1_L_faces_conn8"]:>5}')
    print('  NC2 R1 を外す(全部の縁を足す)= T0 そのもの。**T1 と T0 の差が R1 の効き**')
    for r in per_page:
        print(f'    {r["page"]:44s} T0 面 {r["T0_closed_n"]:>5} / T1 面 '
              f'{r["T1_closed_n"]:>5} / 被覆 {r["T0_cover"]*100:6.2f}% → '
              f'{r["T1_cover"]*100:6.2f}%')
    print('  NC3 R2(i) を逆にする(クリップ層だけが縁を出す)')
    print('    **登録の期待「`線画` が全部落ちて面がほぼ0」は機構が成り立たない** ——')
    print('    **線は墨(ink)で入るので、遮蔽物の集合を変えても落ちない。登録の誤りを申告する**')
    for r in per_page:
        print(f'    {r["page"]:44s} 面 {r["NC3_closed_n"]:>5} 被覆 '
              f'{r["NC3_cover"]*100:6.2f}%(T3 は {r["T3_closed_n"]:>5} / '
              f'{r["T3_cover"]*100:6.2f}%)')
    print(f'  NC4 R3 の「全画布でない」を外す: 矩形度>={RECT_BAR} かつ色調補正でない層 '
          f'**{len(nc4_rows)}**(うち全画布 '
          f'**{sum(1 for x in nc4_rows if x[5])}**)→ **外すと資料の候補がこの数に跳ねる**')
    print(f'  NC5 R4 で `線画` も除外する(案ア): 外れる `線画` は '
          f'**{len([x for x in r4_rows if "残した" in x[3]])}** 枚'
          f'(`page0002` の 6枚)。**案イ では残している**')
    print('  NC6 床を当てない')
    for r in per_page:
        print(f'    {r["page"]:44s} L 面 {r["L_closed_n"]:>5} → 床なし '
              f'{r["NC6_L_faces_no_floor"]:>6}(床で落ちた {r["L_under_floor"]:>6})')

    print(f'\n== R3 が資料と判定した層 {len(r3_hits)} ==')
    for pg, lid, nm, kind, rect, boxr, anc in r3_hits:
        print(f'  {pg[-12:]:12s} L{lid:>4} {str(nm)[:24]:24s} 種別 {str(kind)[:16]:16s} '
              f'矩形度 {rect:5.3f} 画布比 {boxr*100:6.2f}%  /{anc}')
    print(f'== R4 の対象 {len(r4_rows)} ==')
    for pg, lid, nm, how in r4_rows:
        print(f'  {pg[-12:]:12s} L{lid:>4} {str(nm)[:24]:24s} {how}')

    with (OUT / 'b_standard_v2.csv').open('w', newline='', encoding='utf-8') as fh:
        keys = sorted({k for r in per_page for k in r})
        w = csv.DictWriter(fh, fieldnames=['page'] + [k for k in keys if k != 'page'])
        w.writeheader()
        w.writerows(per_page)
    with (OUT / 'holes_not_in_kind_table.csv').open('w', newline='',
                                                    encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page', 'layer_id', 'name'])
        w.writerows(hole_rows)
    print(f'\nwrote {OUT}/b_standard_v2.csv  ({len(hole_rows)} 行の穴も CSV に)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
