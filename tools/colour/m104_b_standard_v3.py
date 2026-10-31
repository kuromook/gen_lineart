#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M104: (b) の標準 第3版 —— 作者の裁定 R6〜R8 を入れて走る。**

**事前登録** `doc/prereg_b_standard_v3_20261031.md`(**走る前に commit した** `3052d2c`)。
**第1版**(`m99`、追補152・153)**・第2版**(`m102`、追補158)**は撤回しない**(規律10)。

**作者の裁定3件**(逐語は登録 §作者の裁定):

  R6   **セリフのテキスト層は境界に使わない。描き文字の層は使う**           → **新しい規則**
  R7   **3D から起こした背景・小物は境界に使う**                           → **第2版どおり。変更なし**
  R7b  **`rendering_lineart` は定義上「線」**                             → **`T5` に<ins>隔離</ins>**
  R7c  **機械生成の線画は取り組む意味が薄い**                              → **頁に札。母集団は動かさない**
  R8   **非クリップの `color` / `MASK` は境界に使う**(それ自体が面)        → **第2版どおり。`T3c` は却下**
  R8b  **残骸かどうかは数値から判断する**                                  → **候補を4つの量で印字**

**段**(登録 §Arms。**同じ走行で全部出す**):

  Z    境界なし(**恒等**)
  L    **線だけ**(閉じる力の下限)。**線率は定義上 1.0 なので C1 の判定に入れない**
  T0   **第1版 S3 の再現** —— **既知セル IC1**
  T1   R1 のみ / T2  R1+R2 / T3  R1+R2+R3+R4 —— **第2版の主値。この版では<ins>既知セル IC8</ins>**
  T4   **T3 + R6** —— **第3版の主値**
  T5   **T4 + R7b**(`rendering_lineart` を墨に入れる)—— **隣。線率の上がりは<ins>定義による</ins>**
  T3c  **作者が却下した段**(非クリップ `color` を落とす)。**撤回せず印字する**

**R7b を隔離する理由**(登録 §R7b。**走る前に書いた**): 墨を動かすと **IC1 と IC8 が崩れ**、
**落ちていた頁の線率が<ins>定義だけで</ins>上がる**(= 「答を渡された腕」と同じ形)。
→ **`T0`〜`T4` の墨は第1版・第2版と同一に保ち、`T5` だけ別の墨で測る。**

**R6 の実装**: **`v_layer_kind == 'text'` の層は縁を出さない。フォルダーの範囲からも
`text` の子孫を除いて作り直す**(`subtree_alpha_ex`)。**名前は使わない**(条例4)。
**除いて空になったフォルダーは縁を出さない**(理由別に印字)。

**器具は写さずに import する**: `m12` / `m16` / `m77` / `m87` / `m99` / `m101` / **`m102`**
(既知セル・`is_named` / `is_sketch` / `rect_of` / `faces_and_ratios` / `stat` …)。

  <venv>/python tools/colour/m104_b_standard_v3.py [--smoke]
"""
import collections
import csv
import hashlib
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
import m102_b_standard_v2 as m102                                  # noqa: E402

OUT = pathlib.Path('results/m104_b_standard_v3')
PAPER = m16.PAPER_TYPE
MULTIPLY = '2'
TEXT_KIND = 'text'                      # **R6。納品の `v_layer_kind`。名前は使わない**
RENDER_LINE = 'rendering_lineart'       # **R7b。名前でしか引けない(登録 §R7 の申告)**
RENDER_ANY = 'rendering'                # **NC9 と 3D の札。同じ申告が当たる**
RESIDUE_NAMES = ('color', 'MASK')       # **R8b。作者の言う「単体の color mask」(厳密一致)**
STAGES = ('Z', 'L', 'T0', 'T1', 'T2', 'T3', 'T3c', 'T4', 'T5')
PAGES6 = tuple(m102.IC1_FACES)          # 登録の頁順(既知セルの順と同じ)

# ----------------------------------------------------------- 既知セル(登録 §IC。走る前の値)
IC4_L_COVER = dict(zip(PAGES6, (2.32, 0.02, 2.44, 1.37, 2.60, 11.44)))
IC8_FACES = dict(zip(PAGES6, (148, 3, 20, 482, 363, 139)))
IC8_COVER = dict(zip(PAGES6, (2.56, 0.02, 6.13, 18.81, 20.88, 12.49)))
IC8_RATIO = dict(zip(PAGES6, (0.00, 100.00, 100.00, 0.00, 42.09, 100.00)))
IC9_R6_DROP = dict(zip(PAGES6, (1, 0, 0, 0, 0, 0)))          # R6 が落とす層(kind=='text')
IC10_RENDER = 3                                               # `rendering` を名前に含む可視層
IC10_RENDER_PAGE = '000_2018_illust201710_illust201710_page0005'
IC11_RESIDUE = dict(zip(PAGES6, (0, 1, 6, 31, 26, 28)))       # 厳密一致・非クリップの葉 = 92
IC11_SUBSTR = 7                                               # 部分一致 `color`(第2版 T3c の引き方)


def fit(a, grid):
    """**ラスタが画布より小さいときに形を揃える**(`m87` は揃えずに `|` する。実務では
    全部が画布大だが、揃えないと黙って例外になるので揃えてから使う)。"""
    if a is None or a.shape == grid:
        return a
    out = np.zeros(grid, bool)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def subtree_alpha_ex(lid, kids, files, grid, cache, skip):
    """**R6**: フォルダーの範囲から `skip` の子孫を除いて作り直す。

    `m87.subtree_alpha` を**書き換えない**(第2版が再現できなくなる)。**写しではなく
    `skip` を足した別の関数**である。
    """
    m = None
    for i in [int(lid)] + list(m16.descendants(int(lid), kids)):
        if i in skip:
            continue
        a = fit(m87.alpha_of(i, files, grid, cache), grid)
        if a is not None:
            m = a.copy() if m is None else (m | a)
    return m


def occluder_mask_r6(r, kids, files, grid, cache, cbase, skip, drop):
    """**R6 を当てた遮蔽範囲。** 葉が `text` なら縁を出さない。フォルダーは作り直す。"""
    lid = int(r['layer_id'])
    if r['folder'] != 'True':
        if lid in skip:
            drop['R6 セリフのテキスト層'] += 1
            return None
        return m87.occluder_mask(r, kids, files, grid, cache, cbase, True, drop)
    m = subtree_alpha_ex(lid, kids, files, grid, cache, skip)
    if m is None or not m.any():
        drop['R6 文字を除いて空になったフォルダー'] += 1
        return None
    if r['clip'] == '1':                       # クリップは下地と交わらせる(T-M3)
        b = cbase.get(r['layer_id'])
        if b is None:
            drop['クリップだが下地が無い'] += 1
            return None
        bm = (m87.subtree_alpha(b['layer_id'], kids, files, grid, cache)
              if b['folder'] == 'True'
              else m87.alpha_of(b['layer_id'], files, grid, cache))
        if bm is None:
            drop['下地のラスタが無い'] += 1
            return None
        m = m & bm
        if not m.any():
            drop['下地と交わらない'] += 1
            return None
    return m


def open_from(ink):
    """**R1**: 墨だけで面を作り、**外周に通じている領域**の真偽図を返す。**1回で止める**。"""
    n, lab, _, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8),
                                                    connectivity=4)
    bl = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    out = np.isin(lab, np.fromiter(bl, int, len(bl)))
    del lab
    return out


def residue_rows(p, body, kids, files, grid, px, cache, kinds, z):
    """**R8b**: 非クリップの `color` / `MASK` の葉について4つの量を出す。

    **(1)** アルファの画素数と `drawn_ratio` / **(2)** **上に在る不透明・非クリップ・非乗算の
    葉に隠れている割合** / **(3)** アルファが空か / **(4)** 同じ頁の他の層とアルファが完全一致か。

    **落とす規則ではない**(登録 §R8b)。**作者「それは数値から判断しよう」の材料である。**
    """
    leaves = [r for r in body if r['folder'] != 'True']
    dig = collections.defaultdict(list)
    for r in leaves:                                  # (4) 完全一致(複製の残骸)
        a = fit(m87.alpha_of(r['layer_id'], files, grid, cache), grid)
        if a is None or not a.any():
            continue
        dig[hashlib.blake2b(np.ascontiguousarray(a).tobytes(),
                            digest_size=16).hexdigest()].append(r['layer_id'])
    targets = {r['layer_id']: r for r in leaves
               if r['clip'] != '1' and (r['name'] or '').strip() in RESIDUE_NAMES}
    above = np.zeros(grid, bool)                      # (2) 上から下へ降りて union を積む
    out = []
    for r in sorted(leaves, key=lambda r: -z[int(r['layer_id'])]):
        a = fit(m87.alpha_of(r['layer_id'], files, grid, cache), grid)
        if r['layer_id'] in targets:
            n = int(a.sum()) if a is not None else 0
            hid = (int((a & above).sum()) / n) if n else 0.0
            d = [k for k, v in dig.items() if r['layer_id'] in v]
            twin = [i for i in (dig[d[0]] if d else []) if i != r['layer_id']]
            out.append({'page': p['page'], 'layer_id': r['layer_id'],
                        'name': (r['name'] or '').strip(),
                        'kind': kinds.get((p['source'], str(r['layer_id']))) or '**表に無い**',
                        'ancestors': r['ancestors'], 'drawn_ratio': r['drawn_ratio'],
                        'draw_color_enable': r['draw_color_enable'],
                        'alpha_px': n, 'alpha_frac': n / px,
                        'hidden_above_frac': hid, 'is_empty': n == 0,
                        'exact_twin_ids': ' '.join(twin)})
        if (a is not None and r['clip'] != '1'
                and (r.get('opacity') or '0') == '256'
                and (r.get('composite') or '0') != MULTIPLY):
            above |= a
    del above
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = '--smoke' in sys.argv
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']

    # --------------------------------------------- 母集団(登録どおり。3D の札でも動かさない)
    keep, dropped = [], []
    for p in pages:
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        n = sum(1 for r in rows_vis if r['layer_type'] != PAPER and m99.is_line(r))
        (keep if n > 0 else dropped).append((p, n))
    print(f'母集団 `pop_6pages_lineart` = {len(keep)} 頁'
          f'(`pop_8pages_registered` = {len(pages)} 頁。外した2頁は `線画` が0枚)')
    for p, n in dropped:
        print(f'    外した頁 {p["page"]:44s} 線画 {n}')
    print('**3D の札は頁を外さない**(登録 §R7c。母集団も棒の一部。規律9)')
    if smoke:
        keep = keep[-2:]
        print('**--smoke: 最後の2頁だけ。門の頁数は判定しない**')

    drops = collections.Counter()
    occ_drop = collections.Counter()
    r6_drop = collections.Counter()
    ic5 = collections.Counter()
    nc8 = collections.Counter()
    per_page, res_rows, hole_rows, r6_rows, render_rows = [], [], [], [], []
    t0 = time.time()

    for p, n_line in keep:
        rows_vis, d = m77.visible_rows(byp[p['source']])
        drops.update(d)
        roots, kids = m16.build_tree(rows_vis)
        z = m16.z_order(roots, kids)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        px = grid[0] * grid[1]
        floor_px = max(1, int(round(px * m12.FLOOR)))
        cache = {}
        body = [r for r in rows_vis if r['layer_type'] != PAPER]

        # ---- R6 の対象(**納品の kind。名前は使わない**)と NC8(名前で引いた形)
        skip = set()
        for r in body:
            k = kinds.get((p['source'], str(r['layer_id'])))
            if k is None:
                hole_rows.append((p['page'], r['layer_id'], r['name']))
            ic5['層種別 ' + (k or '**表に無い**')] += 1
            if k == TEXT_KIND:
                skip.add(int(r['layer_id']))
                r6_rows.append((p['page'], r['layer_id'], r['name'], r['folder'],
                                r['layer_type'], r['ancestors']))
            if TEXT_KIND in (r['name'] or '').lower():
                nc8[p['page']] += 1                 # **NC8: 名前で引いた数(条例4)**
            if m99.is_line(r):
                ic5['線画'] += 1
                ic5['線画 clip'] += (r['clip'] == '1')
            if r['draw_color_enable'] == '1' and r['tone_enabled'] != '1':
                ic5['描画色(MASK)'] += 1
                ic5['描画色(MASK) clip'] += (r['clip'] == '1')
            nm = (r['name'] or '').strip()
            if r['folder'] != 'True' and r['clip'] != '1':
                if nm in RESIDUE_NAMES:
                    ic5['IC11 厳密一致'] += 1
                    ic5['IC11 ' + p['page']] += 1
                if 'color' in nm.lower():
                    ic5['IC11 部分一致 color'] += 1
            if RENDER_ANY in nm.lower():
                render_rows.append((p['page'], r['layer_id'], nm, r['folder'],
                                    kinds.get((p['source'], str(r['layer_id']))),
                                    r['draw_color_enable'], r['ancestors']))

        # ---- 墨(**T0〜T4。第1版・第2版と同一に保つ**)と T5 の墨(**R7b**)
        ink = np.zeros(grid, bool)
        ink5 = np.zeros(grid, bool)
        ink9 = np.zeros(grid, bool)                 # NC9: `rendering_*` を全部墨に入れる(誤り)
        n_raster = n_r7b = n_nc9 = 0
        for r in body:
            nm = (r['name'] or '').lower()
            line = m99.is_line(r)
            if not line and RENDER_ANY not in nm:
                continue
            if line and m102.is_sketch(r):
                drops['**線が R4 で落ちた(案イ では起きないはず)**'] += 1
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is None:
                drops['線: ラスタが無い層'] += 1
                continue
            if line:
                ink[:a.shape[0], :a.shape[1]] |= a
                n_raster += 1
            if r['folder'] == 'True':
                continue                            # フォルダーは墨に入れない(R5 と同じ扱い)
            if line or RENDER_LINE in nm:
                ink5[:a.shape[0], :a.shape[1]] |= a
                n_r7b += (RENDER_LINE in nm and not line)
            if line or RENDER_ANY in nm:
                ink9[:a.shape[0], :a.shape[1]] |= a
                n_nc9 += (not line)
        is3d = n_r7b > 0                            # **3D の札(名前で引いた推測)**

        # ---- 遮蔽物の縁。**4つは第2版と同一**、`t4` だけ R6 を当てる
        E = {k: np.zeros(grid, bool) for k in ('all', 'r2', 'r3r4', 'r3r4c', 't4',
                                               'text_only')}
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
            rect, boxr = m102.rect_of(m87.alpha_of(r['layer_id'], files, grid, cache),
                                      px) if leaf else (0.0, 0.0)
            kind = kinds.get((p['source'], str(r['layer_id'])))
            mat = leaf and (kind == 'image_material'
                            or (not m102.is_tone_correction(r) and rect >= m102.RECT_BAR
                                and boxr < m102.FULL_CANVAS))
            clip = r['clip'] == '1'
            named = m102.is_named(r)
            sk = m102.is_sketch(r)
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
            e = m99.inner_edge(m)
            del m
            for k in sets:
                E[k] |= e
                nE[k] += 1
            # ---- R6: `t4` は `r3r4` と同じ条件、ただし覆いを作り直す
            if 'r3r4' in sets:
                has_text = (int(r['layer_id']) in skip
                            or any(i in skip for i in m16.descendants(int(r['layer_id']),
                                                                      kids)))
                if not has_text:
                    E['t4'] |= e
                    nE['t4'] += 1
                else:
                    m6 = occluder_mask_r6(r, kids, files, grid, cache, cbase, skip,
                                          r6_drop)
                    if m6 is not None:
                        E['t4'] |= m99.inner_edge(m6)
                        nE['t4'] += 1
                        reason['R6 文字を除いて作り直したフォルダー'] += 1
                        del m6
            if int(r['layer_id']) in skip and leaf:   # NC7: 文字だけが縁を出す
                E['text_only'] |= e
                nE['text_only'] += 1
            del e
        cache.clear()

        om = open_from(ink)
        om5 = open_from(ink5)
        om9 = open_from(ink9)
        print(f'  {p["page"]:44s} 線 {n_raster:>3} / 外周 {om.mean()*100:5.2f}% / '
              f'縁 ' + ' '.join(f'{k}={nE[k]}' for k in
                                ('all', 'r2', 'r3r4', 'r3r4c', 't4'))
              + f' / R7b で墨に入れた層 {n_r7b} / 3D札 {is3d}', flush=True)

        # ---- (a) の標準(隣に置くだけ。プールしない)
        comps = []
        (_, _), (lab_a, _), _ = m16.measure_standard(p, rows_vis, grid,
                                                     floor_px, out_comps=comps)
        comps = [c for c in comps if c['area'] >= floor_px]
        isl = m87.areas_from(lab_a, comps, floor_px)
        row = {'page': p['page'], 'n_line': n_line, 'n_line_raster': n_raster,
               'canvas_px': px, 'floor_px': floor_px, 'a_islands': len(isl),
               'a_cover': sum(int(x['area']) for x in isl) / px,
               'open_after_lines': float(om.mean()),
               'r7b_ink_layers': n_r7b, 'is_3d_guess_by_name': is3d,
               **{f'n_edge_{k}': nE[k] for k in E},
               **{f'drop_{k}': v for k, v in reason.items()}}
        del lab_a, comps, isl

        for name, bnd, ref in (('Z', np.zeros(grid, bool), ink),
                               ('L', ink, ink),
                               ('T0', ink | E['all'], ink),
                               ('T1', ink | (E['all'] & om), ink),
                               ('T2', ink | (E['r2'] & om), ink),
                               ('T3', ink | (E['r3r4'] & om), ink),
                               ('T3c', ink | (E['r3r4c'] & om), ink),
                               ('T4', ink | (E['t4'] & om), ink),
                               ('T5', ink5 | (E['t4'] & om5), ink5)):
            closed, open_, under, ratios = m102.faces_and_ratios(bnd, ref, floor_px)
            lo, med, hi, nlow, flow = m102.stat(ratios)
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
        n8, _, st8, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8),
                                                         connectivity=8)
        row['NC1_L_faces_conn8'] = sum(
            1 for i in range(1, n8) if int(st8[i, cv2.CC_STAT_AREA]) >= floor_px)
        del st8
        row['NC6_L_faces_no_floor'] = row['L_closed_n'] + row['L_under_floor']
        c7, _, _, _ = m102.faces_and_ratios(ink | (E['text_only'] & om), ink, floor_px)
        row['NC7_closed_n'] = len(c7)                 # 文字だけが縁を出す
        row['NC7_cover'] = sum(c7) / px
        c9, _, _, r9 = m102.faces_and_ratios(ink9 | (E['t4'] & om9), ink9, floor_px)
        row['NC9_closed_n'] = len(c9)
        row['NC9_ratio_med'] = m102.stat(r9)[1]       # 全部の `rendering_*` を墨に(誤り)
        del c7, c9, r9, om, om5, om9
        for k in list(E):
            del E[k]
        del ink5, ink9

        # ---- R8b(残骸の候補。**落とす規則ではない**)
        res = residue_rows(p, body, kids, files, grid, px, cache, kinds, z)
        res_rows.extend(res)
        cache.clear()
        del ink
        row['R8b_targets'] = len(res)
        row['R8b_empty'] = sum(1 for r in res if r['is_empty'])
        row['R8b_hidden_ge99'] = sum(1 for r in res if r['hidden_above_frac'] >= 0.99)
        row['R8b_twin'] = sum(1 for r in res if r['exact_twin_ids'])
        per_page.append(row)
        print(f'      R8b 対象 {len(res):>3} / 空 {row["R8b_empty"]} / '
              f'99%以上隠れ {row["R8b_hidden_ge99"]} / 完全一致の双子 {row["R8b_twin"]}',
              flush=True)

    # ================================================================ 規律9 の印字
    print('\n== `standard_visible` が落とす行(理由別。規律9)==')
    for k, v in drops.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== 遮蔽物で落ちた層(理由別。規律9)==')
    for k, v in occ_drop.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== R2〜R6 で縁を落とした/作り直した層(理由別・全頁。規律9)==')
    agg = collections.Counter()
    for r in per_page:
        for k, v in r.items():
            if k.startswith('drop_'):
                agg[k[5:]] += v
    for k, v in agg.most_common():
        print(f'  {k:46s} {v:>6}')
    for k, v in r6_drop.most_common():
        print(f'  {k:46s} {v:>6}')
    print('== 納品の層種別(条例6: 「一覧に無い」は「存在しない」ではない)==')
    for k, v in sorted(ic5.items()):
        if k.startswith('層種別 '):
            print(f'  {k:46s} {v:>6}')
    print(f'  **表に無い行 {len(hole_rows)}**(宿題。内訳は下の CSV)')

    # ==================================================================== 門
    print('\n== 門(落ちたら以降の行を読まない。T-M22)==')
    g = {}

    def gate(nm, exp, act, tol=None):
        ok = '—' if exp in (None, '—') else (
            'PASS' if (abs(float(act) - float(exp)) <= tol if tol is not None
                       else act == exp) else '**FAIL**')
        g[nm] = ok
        print(f'  {nm:40s} 期待 {str(exp):>26s} / 実測 {str(act):>26s}  {ok}')

    gate('IC1 頁数', 6 if not smoke else '—', len(per_page))
    for r in per_page:
        k, sh = r['page'], r['page'][-12:]
        gate(f'IC1 T0 面 {sh}', m102.IC1_FACES.get(k, '—'), r['T0_closed_n'])
        gate(f'IC1 T0 被覆 {sh}', m102.IC1_COVER.get(k, '—'),
             round(r['T0_cover'] * 100, 2), 0.01)
        gate(f'IC2 (a) 島 {sh}', m102.IC2_A_ISLANDS.get(k, '—'), r['a_islands'])
        gate(f'IC3 Z 面 {sh}', 0, r['Z_closed_n'])
        gate(f'IC3 Z 被覆 {sh}', 0.0, round(r['Z_cover'] * 100, 2), 0.001)
        gate(f'IC4 L 面 {sh}', m102.IC4_L_FACES.get(k, '—'), r['L_closed_n'])
        gate(f'IC4 L 被覆 {sh}', IC4_L_COVER.get(k, '—'),
             round(r['L_cover'] * 100, 2), 0.01)
        gate(f'IC8 T3 面 {sh}', IC8_FACES.get(k, '—'), r['T3_closed_n'])
        gate(f'IC8 T3 被覆 {sh}', IC8_COVER.get(k, '—'),
             round(r['T3_cover'] * 100, 2), 0.01)
        gate(f'IC8 T3 線率 {sh}', IC8_RATIO.get(k, '—'),
             round(r['T3_ratio_med'] * 100, 2), 0.01)
    for nm, (n, c) in m102.IC5_EXPECT.items():
        gate(f'IC5 {nm}', (n, c) if not smoke else '—', (ic5[nm], ic5[nm + ' clip']))
    gate('IC7 `page0002` の `線画` を除外しない', 0,
         sum(1 for r in r6_rows if False) + agg.get(
             '**線が R4 で落ちた(案イ では起きないはず)**', 0))
    # IC9: R6 が落とす層(頁ごと)
    r6_by = collections.Counter(r[0] for r in r6_rows)
    for r in per_page:
        gate(f'IC9 R6 が落とす層 {r["page"][-12:]}',
             IC9_R6_DROP.get(r['page'], '—'), r6_by[r['page']])
    gate('IC10 `rendering` を含む可視層', IC10_RENDER if not smoke else '—',
         len(render_rows))
    gate('IC10 全部 page0005 か', True if not smoke else '—',
         all(r[0] == IC10_RENDER_PAGE for r in render_rows) if render_rows else '—')
    gate('IC10 kind は raster か', True if not smoke else '—',
         all(r[4] == 'raster' for r in render_rows) if render_rows else '—')
    gate('IC10 kind `3d` の層', 0 if not smoke else '—', ic5['層種別 3d'])
    gate('IC11 厳密一致 `color`/`MASK`', 92 if not smoke else '—',
         ic5['IC11 厳密一致'])
    for r in per_page:
        gate(f'IC11 {r["page"][-12:]}', IC11_RESIDUE.get(r['page'], '—'),
             ic5['IC11 ' + r['page']])
    gate('IC11 部分一致 `color`', IC11_SUBSTR if not smoke else '—',
         ic5['IC11 部分一致 color'])

    bad = [k for k, v in g.items() if v == '**FAIL**']
    print(f'\n**門は全部 PASS**' if not bad else
          f'\n**門が {len(bad)} 件 FAIL: ' + ' / '.join(bad) + '** —— 以降は読まない(T-M22)')

    # ==================================================================== 棒
    print('\n== C1(主): 頁ごとに **P2(T4) > P2(T0)**。**最悪の頁で判定**(T-M25)==')
    n_fail = 0
    for r in per_page:
        a, b = r['T0_ratio_med'] * 100, r['T4_ratio_med'] * 100
        ok = 'PASS' if b > a else '**FAIL**'
        n_fail += (b <= a)
        print(f'  {r["page"]:44s} T0 {a:6.2f}% → T4 {b:6.2f}%  ({b-a:+6.2f} 点)  {ok}'
              + ('   **3D札**' if r['is_3d_guess_by_name'] else ''))
    print(f'  → **C1 {"PASS" if n_fail == 0 else "FAIL"}**(満たさない頁 {n_fail}/{len(per_page)})')
    print('  **L は線率が定義上 1.0 なので C1 の判定に入れない**(登録の注)')

    print('\n== C5: 3D の札の無い頁だけで C1 を読む(登録 §R7c・§C5。**主は上の6頁**)==')
    sub = [r for r in per_page if not r['is_3d_guess_by_name']]
    nf5 = sum(1 for r in sub if r['T4_ratio_med'] <= r['T0_ratio_med'])
    print(f'  札の無い頁 {len(sub)} / 満たさない頁 {nf5} → '
          f'**{"PASS" if nf5 == 0 else "FAIL"}**(**隣に置く値である。棒の緩めではない**)')

    print('\n== C2: 線率 < 0.5 の面の割合が T0 より T4 で減る頁 ==')
    n_down = 0
    for r in per_page:
        a, b = r['T0_ratio_lt05_frac'] * 100, r['T4_ratio_lt05_frac'] * 100
        n_down += (b < a)
        print(f'  {r["page"]:44s} T0 {a:6.2f}% ({r["T0_ratio_lt05_n"]:>4}面) → '
              f'T4 {b:6.2f}% ({r["T4_ratio_lt05_n"]:>4}面)  {b-a:+7.2f} 点')
    print(f'  → **減った頁 {n_down}/{len(per_page)}**')

    print('\n== C3: `page0019` の予言3つ(**走る前に書いた**)==')
    pg = '000_2017_illustration_sunshine2_page0019'
    r19 = next((r for r in per_page if r['page'] == pg), None)
    if r19 is None:
        print('  **この走行に `page0019` が無い**(smoke)。判定しない')
    else:
        print(f'  (a) 閉じた面が 148 から減る      : T3 {r19["T3_closed_n"]} → '
              f'T4 {r19["T4_closed_n"]}  '
              f'{"PASS" if r19["T4_closed_n"] < 148 else "**FAIL**"}')
        print(f'  (b) 線率<0.5 の面が 97 → 0      : T3 {r19["T3_ratio_lt05_n"]} → '
              f'T4 {r19["T4_ratio_lt05_n"]}  '
              f'{"PASS" if r19["T4_ratio_lt05_n"] == 0 else "**FAIL**"}')
        print(f'  (c) 線率の中央が 0.00% から上がる: T3 {r19["T3_ratio_med"]*100:.2f}% → '
              f'T4 {r19["T4_ratio_med"]*100:.2f}%  '
              f'{"PASS" if r19["T4_ratio_med"] > 0 else "**FAIL**"}')

    print('\n== C4(この工程が丸ごと誤りである出方): T4 の被覆が L と同じ水準か ==')
    n_same = 0
    for r in per_page:
        same = r['T4_cover'] < r['L_cover'] * m102.C4_FACTOR
        n_same += same
        print(f'  {r["page"]:44s} L {r["L_cover"]*100:6.2f}% / T4 '
              f'{r["T4_cover"]*100:6.2f}% / T0 {r["T0_cover"]*100:6.2f}%  '
              f'{"**L と同水準**" if same else "上回る"}')
    print(f'  → **同水準の頁 {n_same}/{len(per_page)}**')

    # ================================================================== 副値
    print('\n== 副値: 段ごとの面の数 / 被覆 / 線率の中央(頁ごと。プールしない)==')
    for lab, key, sc in (('面', 'closed_n', 1), ('%', 'cover', 100),
                         ('線率', 'ratio_med', 100)):
        st = [s for s in STAGES if not (lab == '線率' and s == 'Z')]
        print(f'{"page":50s}' + ''.join(f'{s+lab:>9s}' for s in st))
        for r in per_page:
            print(f'{r["page"]:50s}' + ''.join(
                (f'{r[s+"_"+key]*sc:9.0f}' if sc == 1 else f'{r[s+"_"+key]*sc:9.2f}')
                for s in st))

    print('\n== T5(**R7b。線率の上がりは<ins>定義による</ins>。棒の証拠にしない**)==')
    for r in per_page:
        print(f'  {r["page"]:44s} 墨に入れた `rendering_lineart` {r["r7b_ink_layers"]} / '
              f'T4 線率 {r["T4_ratio_med"]*100:6.2f}% → T5 {r["T5_ratio_med"]*100:6.2f}% / '
              f'T5 面 {r["T5_closed_n"]:>5} 被覆 {r["T5_cover"]*100:6.2f}%')

    print('\n== 負の対照 ==')
    print('  NC1 背景を8連結で数える(T-M28): 面が減るはず')
    for r in per_page:
        print(f'    {r["page"]:44s} L 4連結 {r["L_closed_n"]:>5} → '
              f'8連結 {r["NC1_L_faces_conn8"]:>5}')
    print('  NC2 R1 を外す = T0。**T1 と T0 の差が R1 の効き**')
    for r in per_page:
        print(f'    {r["page"]:44s} T0 面 {r["T0_closed_n"]:>5} / T1 面 '
              f'{r["T1_closed_n"]:>5} / 被覆 {r["T0_cover"]*100:6.2f}% → '
              f'{r["T1_cover"]*100:6.2f}%')
    print('  **NC3 は出さない** —— 第2版で機構が成り立たないことを申告した(追補158 §G、T-M30)')
    print('  NC6 床を当てない')
    for r in per_page:
        print(f'    {r["page"]:44s} L 面 {r["L_closed_n"]:>5} → 床なし '
              f'{r["NC6_L_faces_no_floor"]:>5}(床で落ちた {r["L_under_floor"]:>5})')
    print('  NC7 R6 を逆にする(**文字<ins>だけ</ins>が縁を出す**)')
    for r in per_page:
        print(f'    {r["page"]:44s} 面 {r["NC7_closed_n"]:>5} 被覆 '
              f'{r["NC7_cover"]*100:6.2f}%(T4 は {r["T4_closed_n"]:>5} / '
              f'{r["T4_cover"]*100:6.2f}%)')
    print('  NC8 R6 を**名前**で引く(条例4 が禁じた形)')
    for r in per_page:
        print(f'    {r["page"]:44s} 名前に `text` {nc8[r["page"]]:>4} / '
              f'kind で引いた {r6_by[r["page"]]:>4}  '
              f'{"**食い違う**" if nc8[r["page"]] != r6_by[r["page"]] else "同じ"}')
    print('  NC9 `rendering_*` を**全部**墨に入れる(誤り。**定義で上がることの対照**)')
    for r in per_page:
        print(f'    {r["page"]:44s} T5 線率 {r["T5_ratio_med"]*100:6.2f}% → '
              f'NC9 {r["NC9_ratio_med"]*100:6.2f}% / 面 {r["NC9_closed_n"]:>5}')

    print(f'\n== R6 が落とした層 {len(r6_rows)}(**納品の kind が `text`**)==')
    for a in r6_rows:
        print(f'  {a[0][-12:]:13s} L{str(a[1]):>4s} {(a[2] or "")[:30]:30s} '
              f'folder={a[3]:5s} layer_type={a[4]:>6s} /{a[5]}')
    print(f'== `rendering` を名前に含む可視層 {len(render_rows)}'
          f'(**名前で引いた推測。census の3D精査待ち**)==')
    for a in render_rows:
        print(f'  {a[0][-12:]:13s} L{str(a[1]):>4s} {a[2][:30]:30s} folder={a[3]:5s} '
              f'kind={str(a[4]):12s} 描画色={a[5]} /{a[6]}')

    print(f'\n== R8b 残骸の候補(**落とす規則ではない**。全 {len(res_rows)} 枚は CSV)==')
    print('  **空のアルファ / 99%以上隠れている / 完全一致の双子が在る** ものだけ印字する')
    for a in res_rows:
        if a['is_empty'] or a['hidden_above_frac'] >= 0.99 or a['exact_twin_ids']:
            print(f'  {a["page"][-12:]:13s} L{str(a["layer_id"]):>4s} {a["name"]:5s} '
                  f'画素 {a["alpha_px"]:>9} ({a["alpha_frac"]*100:6.3f}%) 隠れ '
                  f'{a["hidden_above_frac"]*100:6.2f}% 空={a["is_empty"]} '
                  f'双子={a["exact_twin_ids"] or "—"} /{a["ancestors"]}')
    for r in per_page:
        print(f'  {r["page"]:44s} 対象 {r["R8b_targets"]:>3} / 空 {r["R8b_empty"]} / '
              f'99%以上隠れ {r["R8b_hidden_ge99"]} / 双子 {r["R8b_twin"]}')

    # ==================================================================== CSV
    for nm, rows in (('b_standard_v3.csv', per_page),
                     ('r8b_residue_candidates.csv', res_rows)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(OUT / nm, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {OUT / nm}  ({len(rows)} 行)')
    with open(OUT / 'holes_not_in_kind_table.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page', 'layer_id', 'name'])
        w.writerows(hole_rows)
    print(f'wrote {OUT / "holes_not_in_kind_table.csv"}  ({len(hole_rows)} 行)')


if __name__ == '__main__':
    main()
