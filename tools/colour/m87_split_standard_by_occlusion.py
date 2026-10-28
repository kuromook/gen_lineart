#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m87: **追補122 の第1段を走らせる —— 標準の面を遮蔽で割り直す**。

**登録は追補122**(走る前に書いて commit `7f5763e`。**走行の値は1つも見ずに書いた**)。
**主値 `N`** = 標準の面を、**遮蔽された画素を外して8連結成分に割り直した破片の数**の総和。
**比 `N / 483`**。床は **1e-5**(標準側と同じ。動かさない)。**頁ごとに印字・プールしない。**

**遮蔽物の定義(作者 2026-10-28)と三組**(基盤 `occlusion_definition.md`):

  `O_strict` = 表示 ∧ 不透明度>0 ∧ **合成が通常(0)のみ**
  `O_wide`   = 表示 ∧ 不透明度>0 ∧ **合成が乗算(2)以外すべて** = **作者の集合**
  `O_all`    = 表示 ∧ 不透明度>0 ∧ **合成を見ない**
  **`O_strict` ⊆ 作者 ⊆ `O_all` は 2=乗算 の対応表に依存しない**(帯の端は仮説に乗らない側で取る)。
  **2 = 乗算 は抽出側が PSD と同名の 33組で 6/6 一致で同定**(1 対 3,174 の外挿)。

**クリップ層は下地と合わせて遮蔽範囲を決める**(**T-M3**: export はクリップを適用しないので、
クリップ層のアルファそのものは被覆ではない。**下地のアルファと交わらせる**)。
**非表示は含めない**(自身・祖先。`visibility` の1の位。基盤 T142/T143)。
**z は m16 の登録のまま**(木の深さ優先・`order_in_parent` 昇順 = 下から。
フォルダーの z はその子孫のうち最も上)。

**既知セルと負の対照(追補122 §F を、追補124 §C〜§E で2つ書き換えた)**:
  IC-1 `standard_visible` が **152/503**(面は連結で数えて **692**。作者の裁定)
  IC-3 **遮蔽を適用しなければ `N` = 面の数ちょうど**(恒等)
  **IC-2'** **最上の面を覆っている層に、標準のベース層が1枚も無い**
       —— 旧 IC-2「最上の面は割れない」は**前提が誤り**。面はシルエット層から来るが、
       **線画はすべてのシルエット層より上に在る**(追補124 §C)
  **IC-4'** **遮蔽画素数と「完全に隠れた面の数」が三組で単調**
       —— 旧 IC-4「`N` が単調」は**存在しない性質**。遮蔽物を増やすと `N` は
       **分断で増え、完全な消滅で減る**(追補124 §D。**`N` に単調性を要求しない**)
  **IC-5/6/7(負)** **遮蔽画素が増える**(旧「`N` が増える」は §D と同じ誤り。T-M21)
  IC-8 **z 順を逆に読むと `N` が変わる**(順序が効いていることの確認)

**修理(追補124 §B、T-M20)**: クリップ層の除外を「保留して後で戻す」で書いたところ
**カーソルを保留した枝でも進めていたので、保留が永久の破棄になっていた** ——
**クリップ層は一度も遮蔽物になっていなかった**(IC-6 の差がちょうど +0 が唯一の証拠)。
→ **除外は層の集合からではなく面ごとに引く。計数で持ち、面の矩形の中だけで引く。**

**落としたものは理由別に印字する**(規律9。`continue` にも当てる)。

  <venv>/python tools/colour/m87_split_standard_by_occlusion.py
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

OUT = pathlib.Path('results/m87_split_by_occlusion')
MULTIPLY = '2'            # **抽出側が 6/6 で同定**(対応表に依存する箇所はここだけ)
NORMAL = '0'
# **面は作者の裁定で「連結な島」= 692**(483 はラベルの数。消さず名前で読み分け)
EXPECT_STD = {'bases': 152, 'components': 503}
EXPECT_AREAS_LABEL = 483     # ラベルで数えた旧い値(残す)
BAR = 1.20                # 追補122 §E の M-D1。**追補124 でも動かしていない**
# **修理前の走行で見てしまった値**(追補124 §A の申告。クリップ層が遮蔽物に
# 入っていなかった状態の値。**比較のために印字する。棒ではない**)
PRE_FIX = {'N': 856, 'areas': 692, 'ratio': 1.2370}
HOLD = 0.0445             # 向きの分からない残差(マスクのオフセット 8,457層 = 4.45%)

# 腕と対照。(名前, 合成の条件, 非表示を入れるか, クリップを下地と交わらせるか, z を逆に読むか)
VARIANTS = [
    ('O_strict',       lambda c: c == NORMAL,  False, True,  False),
    ('O_wide(作者)',   lambda c: c != MULTIPLY, False, True,  False),
    ('O_all',          lambda c: True,          False, True,  False),
    ('IC-3 遮蔽なし',   None,                    False, True,  False),
    ('IC-6 クリップ非交差', lambda c: c != MULTIPLY, False, False, False),
    ('IC-7 非表示込み',  lambda c: c != MULTIPLY, True,  True,  False),
    ('IC-8 z 逆読み',   lambda c: c != MULTIPLY, False, True,  True),
]


def vis(r):
    return bool(int(r['visibility'] or 0) & 1)


def clip_base_map(rows):
    """クリップ層 -> その下地の行(m16.tree_bases と同じ規則: 同じ親の兄弟・順昇順)。"""
    sib = collections.defaultdict(list)
    for r in rows:
        if r['layer_type'] == m16.PAPER_TYPE or not r['parent_id']:
            continue
        sib[r['parent_id']].append(r)
    out = {}
    for g in sib.values():
        g.sort(key=lambda r: int(r['order_in_parent']))
        base = None
        for r in g:
            if r['clip'] == '0':
                base = r
            elif base is not None:
                out[r['layer_id']] = base
    return out


def alpha_of(lid, files, grid, cache):
    """1層の描画アルファ(原寸・bool)。ラスタが無ければ None。

    **`layer_id` の型は manifest が `int`、CSV が `str`。** 最初の走行は
    `files.get('2')` が常に None で、**遮蔽物のラスタを1枚も読んでいなかった**
    (全部「ラスタが無い層」として静かに落ちた。**T-M19**)。**int で引く。**
    """
    lid = int(lid)
    if lid in cache:
        return cache[lid]
    f = files.get(lid)
    a = None
    if f is not None:
        im = cv2.imread(str(m12.D / f), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            a = (im[..., 3] >= m12.ALPHA_DRAWN)[:grid[0], :grid[1]]
            a = np.ascontiguousarray(a)
        del im
    cache[lid] = a
    return a


def subtree_alpha(lid, kids, files, grid, cache):
    """フォルダーの範囲 = 子孫の描画アルファの和。"""
    ids = [i for i in m16.descendants(int(lid), kids)]
    m = None
    for i in ids:
        a = alpha_of(i, files, grid, cache)
        if a is None:
            continue
        m = a.copy() if m is None else (m | a)
    own = alpha_of(lid, files, grid, cache)
    if own is not None:
        m = own.copy() if m is None else (m | own)
    return m


def occluder_mask(r, kids, files, grid, cache, cbase, intersect_clip, drop):
    """1層の遮蔽範囲。クリップ層は下地と交わらせる(T-M3)。"""
    lid = r['layer_id']
    m = (subtree_alpha(lid, kids, files, grid, cache)
         if r['folder'] == 'True' else alpha_of(lid, files, grid, cache))
    if m is None:
        drop['ラスタが無い層'] += 1
        return None
    if not m.any():
        drop['描画が0の層'] += 1
        return None
    if intersect_clip and r['clip'] == '1':
        b = cbase.get(lid)
        if b is None:
            drop['クリップだが下地が無い'] += 1
            return None
        bm = (subtree_alpha(b['layer_id'], kids, files, grid, cache)
              if b['folder'] == 'True'
              else alpha_of(b['layer_id'], files, grid, cache))
        if bm is None:
            drop['下地のラスタが無い'] += 1
            return None
        m = m & bm
        if not m.any():
            drop['下地と交わらない'] += 1
            return None
    return m


def areas_from(lab, comps, floor_px):
    """**面 = 重なりを解いた後のラベルを、さらに8連結な島に割ったもの**。

    **作者の裁定(2026-10-28)**: 1つのラベルが別の部位に横切られて2つの島に
    分かれている原寸の切り出しを見せて問うたところ、**「二つの面」**。
    → **分断された島は別の面として数える。** 面は **483 ではなく 692**
    (ベース152 / 成分503 は不変)。**483 は消さず3つ目の名前として並べる**
    (`standard_all` 801 / `standard_visible` 483 / **連結で数えた 692**)。

    **各面の z は、その面を取った成分の z** を付ける。

    **最初の走行は成分(503)を割っていた。面は 483** —— `m16` の paint が
    「**面積の小さい成分が勝つ**」で塗り直した後の単位である(追補7 R3)。
    **paint を写さずに対応を取る**: 面の画素を1つ取り、**その画素を含む成分のうち
    面積が最小のもの**が paint の勝者である(優先度が面積そのものだから)。
    """
    out = []
    nlab = int(lab.max())
    for k in range(1, nlab + 1):
        mk = (lab == k)
        if not mk.any():
            continue
        n, cl, st, _ = cv2.connectedComponentsWithStats(
            mk.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            if int(st[j, cv2.CC_STAT_AREA]) < floor_px:
                continue
            x0 = int(st[j, cv2.CC_STAT_LEFT]); y0 = int(st[j, cv2.CC_STAT_TOP])
            w = int(st[j, cv2.CC_STAT_WIDTH]); h = int(st[j, cv2.CC_STAT_HEIGHT])
            isl = np.ascontiguousarray(cl[y0:y0 + h, x0:x0 + w] == j)
            ys, xs = np.nonzero(isl)
            py, px = int(ys[0]) + y0, int(xs[0]) + x0
            win = None
            for c in comps:
                by, bx, bh, bw = c['box']
                if by <= py < by + bh and bx <= px < bx + bw \
                        and c['mask'][py - by, px - bx]:
                    if win is None or c['area'] < win['area']:
                        win = c
            out.append({'z': win['z'] if win else 0,
                        'base': win['layer_id'] if win else None,
                        'area': int(st[j, cv2.CC_STAT_AREA]),
                        'box': (y0, x0, h, w), 'mask': isl})
        del mk, cl
    return out


def ancestors(lid, parent):
    out, cur = [], parent.get(str(lid))
    while cur:
        out.append(str(cur))
        cur = parent.get(str(cur))
    return out


def run_variant(p, rows_vis, rows_all, areas, grid, floor_px, spec, cache,
                base_ids):
    """1頁・1腕。破片の数・遮蔽画素数・完全に隠れた面を返す。

    **追補124 §B で直した**: 作者の裁定「**クリップ層は自分の下地を遮蔽しない**」を
    **層の集合から引く**(保留して後で戻す)形で書いたところ、**カーソルを保留した枝でも
    進めていたので保留が永久の破棄になり、クリップ層が一度も遮蔽物にならなかった**
    (**T-M20**。負の対照 IC-6 の差がちょうど +0 だったのが唯一の証拠)。

    → **除外は面ごとに引く。** 全候補を z の順に積み、
    **面 `c` を評価するときだけ、`c` の下地(とその祖先のフォルダー)にクリップする層の
    寄与を外す**。**重なりを正しく外すために和ではなく整数の計数で持ち**、
    **面の外接矩形の中だけで `計数 − 除外する層` を引いて `> 0` を取る**
    (**和の引き算では、除外する層と他の遮蔽物が重なった画素を誤って開けてしまう**)。
    """
    name, comp_ok, with_hidden, intersect_clip, reverse_z = spec
    files = {l['layer_id']: l['file'] for l in p['layers'] if l.get('file')}
    src = rows_all if with_hidden else rows_vis
    roots, kids = m16.build_tree(src)
    z = m16.z_order(roots, kids)
    cbase = clip_base_map(src)
    parent = {str(r['layer_id']): (str(r['parent_id']) if r['parent_id']
                                   else None) for r in src}
    drop = collections.Counter()

    cand = []
    if comp_ok is not None:
        for r in src:
            if r['layer_type'] == m16.PAPER_TYPE:
                continue
            if r['folder'] == 'True':
                # **フォルダーは遮蔽物にしない。** 範囲が子孫の和なので
                # **その面を含むフォルダーは必ず 100% 覆う**(祖先が全部遮蔽物になる)。
                # 画素は子にあり、子は別途候補に入っているので**二重計上**でもある
                drop['フォルダー(入れ物)'] += 1
                continue
            if int(r['opacity'] or 0) <= 0:
                drop['不透明度が0'] += 1
                continue
            if not comp_ok(r['composite']):
                drop['合成が条件外'] += 1
                continue
            zz = z.get(int(r['layer_id']))
            if zz is None:
                drop['z が無い'] += 1
                continue
            cand.append((zz, r))
    cand.sort(key=lambda t: t[0], reverse=not reverse_z)

    cnt = np.zeros(grid, np.uint16)          # **和ではなく計数**(§B)
    clips_by_base = collections.defaultdict(list)   # 下地 -> 積んだクリップ層
    cmask = {}                                      # クリップ層の遮蔽範囲(除外用)
    added = []                                      # 積んだ層(IC-2' の診断用)
    order = sorted(areas, key=lambda c: c['z'], reverse=not reverse_z)
    i, frag_tot, covered, full = 0, 0, [], 0
    hist = collections.Counter()
    top_z = max((a['z'] for a in areas), default=None)
    top_cov, top_frag, top_occ = 0.0, 0, []
    for c in order:
        zc = c['z']
        while i < len(cand) and ((cand[i][0] > zc) if not reverse_z
                                 else (cand[i][0] < zc)):
            r = cand[i][1]
            i += 1                           # **カーソルは必ず進める。枝で止めない**
            m = occluder_mask(r, kids, files, grid, cache, cbase,
                              intersect_clip, drop)
            if m is None:
                continue
            cnt += m                         # bool -> 0/1 を足す
            added.append(r)
            bb = cbase.get(r['layer_id'])
            if bb is not None:
                clips_by_base[str(bb['layer_id'])].append(str(r['layer_id']))
                cmask[str(r['layer_id'])] = m
        y, x, h, w = c['box']
        # **自分の下地にクリップする層の寄与を、この面の矩形の中だけで外す**
        sub = cnt[y:y + h, x:x + w].astype(np.int32)
        if c['base'] is not None:
            keys = [str(c['base'])] + ancestors(c['base'], parent)
            for b in keys:
                for lid in clips_by_base.get(b, ()):
                    mm = cmask.get(lid)
                    if mm is not None:
                        sub -= mm[y:y + h, x:x + w]
        occ = sub > 0
        rem = c['mask'] & ~occ
        a0 = int(c['mask'].sum())
        cov = 1.0 - (int(rem.sum()) / a0 if a0 else 0.0)
        covered.append(cov)
        if rem.any():
            n, _, st, _ = cv2.connectedComponentsWithStats(
                rem.astype(np.uint8), connectivity=8)
            k = sum(1 for j in range(1, n)
                    if int(st[j, cv2.CC_STAT_AREA]) >= floor_px)
        else:
            k = 0
        if c['z'] == top_z and not top_occ and not reverse_z:
            top_cov, top_frag = cov, k
            # **IC-2'(追補124 §C)**: 最上の面を実際に覆っている層を印字し、
            # **その中に標準のベース層が1枚も無いことを要求する**。
            # **ベースが現れたら z を誤って読んでいる**(線画・効果は正常)
            d2 = collections.Counter()
            for r in added:
                mm = occluder_mask(r, kids, files, grid, cache, cbase,
                                   intersect_clip, d2)
                if mm is None:
                    continue
                ov = int((c['mask'] & mm[y:y + h, x:x + w]).sum())
                if ov:
                    top_occ.append({'layer_id': str(r['layer_id']),
                                    'ベース': str(r['layer_id']) in base_ids,
                                    'composite': r['composite'],
                                    '被覆': ov / a0 if a0 else 0.0})
            top_occ.sort(key=lambda t: -t['被覆'])
        if k == 0:
            full += 1
            hist['0(完全に隠れた/床で消えた)'] += 1
        else:
            frag_tot += k
            hist['1(割れない)' if k == 1 else ('2' if k == 2 else '3以上')] += 1
    occ_pix = int((cnt > 0).sum())       # **恒等的に単調な量**(追補124 §D)
    del cnt, cmask
    return dict(name=name, N=frag_tot, areas=len(areas), full=full,
                occ_pix=occ_pix, cand=len(cand),
                top_cov=top_cov, top_frag=top_frag, top_occ=top_occ,
                cov_med=float(np.median(covered)) if covered else 0.0,
                cov_max=float(np.max(covered)) if covered else 0.0,
                drop=drop, hist=hist)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    print(f'頁 {len(pages)}  ラスタ {m12.D}  床 {m12.FLOOR:g}')

    tot = collections.Counter()
    agg = collections.Counter()
    topdiag = []
    per_page, t0 = [], time.time()
    for p in pages:
        rows_all = byp[p['source']]
        rows_vis, _ = m77.visible_rows(rows_all)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (lab_a, garea), st = m16.measure_standard(
            p, rows_vis, grid, floor_px, out_comps=comps)
        # **単位は「面」**(重なりを解いた後。483)。**成分(503)ではない**
        comps = [c for c in comps if c['area'] >= floor_px]
        areas = areas_from(lab_a, comps, floor_px)
        del lab_a
        tot['bases'] += st['bases']; tot['components'] += st['components']
        tot['areas_label'] += len(garea)
        tot['areas'] += len(areas)
        cache = {}
        res = {}
        base_ids = {str(c['layer_id']) for c in comps}
        for spec in VARIANTS:
            r = run_variant(p, rows_vis, rows_all, areas, grid, floor_px,
                            spec, cache, base_ids)
            res[spec[0]] = r
            for k in ('N', 'occ_pix', 'full'):
                agg[(spec[0], k)] += r[k]
            for k, v in r['drop'].items():
                tot[f'落: {k}'] += v
        cache.clear()
        del areas
        top = res['O_wide(作者)']['top_occ']
        top_base = [t for t in top if t['ベース']]
        topdiag.append((p['page'], top, top_base))
        # **単位**: 主値の分母は**連結で数えた面**(合計 692。作者の裁定)。
        # **ラベルで数えた数(合計 483)は別の列**に、別の名前で置く ——
        # 最初の走行は**頁ごとの列にラベルの数を入れ、合計だけ連結の数で割っていた**
        # (`IC-3 遮蔽なし` が頁ごとに合わないのが証拠だった。**T-M23**)
        row = {'page': p['page'],
               '面(連結)': res['O_wide(作者)']['areas'],
               '面(ラベル)': len(garea),
               **{k: v['N'] for k, v in res.items()},
               '被覆の中央(作者)': round(res['O_wide(作者)']['cov_med'], 4),
               '最上の面の被覆': round(res['O_wide(作者)']['top_cov'], 6),
               '最上の面の破片': res['O_wide(作者)']['top_frag'],
               '最上の面の遮蔽物': len(top),
               '最上の面を覆うベース層': len(top_base),
               '遮蔽画素(作者)': res['O_wide(作者)']['occ_pix'],
               '完全に隠れた面(作者)': res['O_wide(作者)']['full']}
        per_page.append(row)
        print(f'  {p["page"]:44s} 面{row["面(連結)"]:>4} '
              + ' '.join(f'{k.split("(")[0]}{res[k]["N"]:>4}' for k in res)
              + f'  {time.time()-t0:6.1f}s', flush=True)
        del comps

    print('\n== IC-1 標準の再現 ==')
    ok = all(tot[k] == v for k, v in EXPECT_STD.items())
    for k, v in EXPECT_STD.items():
        print(f'  {k:11s} 期待 {v:>4} / 実測 {tot[k]:>4}  '
              f'{"PASS" if tot[k] == v else "**FAIL**"}')
    print(f'  **IC-1 {"PASS" if ok else "**FAIL**。先を読まない"}**')
    if not ok:
        return 1

    A = tot['areas']
    names = [v[0] for v in VARIANTS]
    N = {k: agg[(k, 'N')] for k in names}
    PIX = {k: agg[(k, 'occ_pix')] for k in names}
    FULL = {k: agg[(k, 'full')] for k in names}
    print(f'  面(連結で数えた)  {A}  / ラベルで数えた {tot["areas_label"]}')

    print('\n== IC-3 遮蔽を適用しなければ `N` = 面の数(**頁ごとにも当てる**)==')
    # **合計だけで当てていたので単位の取り違えが映らなかった**(T-M23)。
    # **恒等なので頁ごとに当てられる。棒ではないので厳しくしても主値は動かない**
    bad3 = [r for r in per_page if r['IC-3 遮蔽なし'] != r['面(連結)']]
    ic3 = N['IC-3 遮蔽なし'] == A and not bad3
    print(f'  N(遮蔽なし) {N["IC-3 遮蔽なし"]} / 面 {A}  '
          f'/ 頁ごとに合わない頁 {len(bad3)}  '
          f'{"PASS" if ic3 else "**FAIL**。器具の誤り。先を読まない"}')
    for r in bad3:
        print(f'       {r["page"][-34:]:36s} IC-3 {r["IC-3 遮蔽なし"]} '
              f'/ 面(連結) {r["面(連結)"]} / 面(ラベル) {r["面(ラベル)"]}')
    if not ic3:
        return 1

    print("\n== IC-2' 最上の面を覆っているのは何か(追補124 §C)==")
    print('  **要求: 標準のベース層が1枚も現れない。** '
          '線画・効果が現れるのは正常')
    ic2 = True
    for page, top, top_base in topdiag:
        print(f'  {page[-34:]:36s} 遮蔽物 {len(top):>3} / '
              f'うちベース層 {len(top_base):>2}  '
              f'{"PASS" if not top_base else "**FAIL**"}')
        for t in top[:3]:
            print(f'      layer {t["layer_id"]:>6} 合成 {t["composite"]:>2} '
                  f'ベース {str(t["ベース"]):>5} 被覆 {t["被覆"]:.4f}')
        ic2 &= not top_base
    print(f'  **IC-2\' {"PASS" if ic2 else "**FAIL**。先を読まない"}**')
    if not ic2:
        return 1

    print("\n== IC-4' 恒等的に単調な2つの量で単調(追補124 §D)==")
    print('  **`N` に単調性は要求しない** —— 遮蔽物を増やすと'
          '分断で増え、完全な消滅で減る(T-M21)')
    tri = ['O_strict', 'O_wide(作者)', 'O_all']
    p1 = PIX[tri[0]] <= PIX[tri[1]] <= PIX[tri[2]]
    p2 = FULL[tri[0]] <= FULL[tri[1]] <= FULL[tri[2]]
    print(f'  (1) 遮蔽画素       {PIX[tri[0]]:>9} <= {PIX[tri[1]]:>9} '
          f'<= {PIX[tri[2]]:>9}  {"PASS" if p1 else "**FAIL**"}')
    print(f'  (2) 完全に隠れた面 {FULL[tri[0]]:>9} <= {FULL[tri[1]]:>9} '
          f'<= {FULL[tri[2]]:>9}  {"PASS" if p2 else "**FAIL**"}')
    print(f'      (参考)`N`     {N[tri[0]]:>9}    {N[tri[1]]:>9}    '
          f'{N[tri[2]]:>9}  **門にしない**')
    print(f'  **IC-4\' {"PASS" if p1 and p2 else "**FAIL**。先を読まない"}**')
    if not (p1 and p2):
        return 1

    print('\n== 負の対照(**判定は遮蔽画素。追補124 §E**)==')
    base = N['O_wide(作者)']
    bpix = PIX['O_wide(作者)']
    ctl = True
    for k, lab in (('O_all', 'IC-5 乗算込み'),
                   ('IC-6 クリップ非交差', 'IC-6 クリップ非交差'),
                   ('IC-7 非表示込み', 'IC-7 非表示込み')):
        d = PIX[k] - bpix
        print(f'  {lab:22s} 遮蔽画素 {PIX[k]:>9} 差 {d:>+9}  '
              f'{"発火(増えた)" if d > 0 else "**発火せず**"}'
              f'   / `N` {N[k]:>5} 差 {N[k]-base:>+5}(門にしない)')
        ctl &= d > 0
    dz = N['IC-8 z 逆読み'] - base
    print(f'  {"IC-8 z 逆読み":22s} `N` {N["IC-8 z 逆読み"]:>5} 差 {dz:>+5}  '
          f'{"PASS(順序が効いている)" if dz else "**FAIL**"}')
    ctl &= dz != 0
    print(f'  **負の対照 {"すべて発火" if ctl else "**発火せず**。先を読まない"}**')
    if not ctl:
        return 1

    print('\n== 主値(**頁ごと。プールしない**)==')
    print(f'  {"頁":44s} {"面(連結)":>8} {"N(作者)":>8} {"N/面":>7} {"被覆中央":>9} {"全隠れ":>6}')
    for r in per_page:
        n, a = r['O_wide(作者)'], r['面(連結)']
        print(f'  {r["page"]:44s} {a:>8} {n:>8} '
              f'{(n/a if a else 0):>7.3f} {r["被覆の中央(作者)"]:>9.4f} '
              f'{r["完全に隠れた面(作者)"]:>6}')
    ratio = base / A
    print(f'\n  **合計 N(作者) {base} / 面 {A} -> N/面 = {ratio:.4f}**')
    print(f'  O_strict {N["O_strict"]/A:.4f} / O_all {N["O_all"]/A:.4f}')
    print(f'  **完全に隠れた面(副値。`N` から落ちている)** '
          f'{FULL["O_wide(作者)"]} / 面 {A}')
    print(f'\n  **修理前の走行(追補124 §A で申告した、見てしまった値)** '
          f'N {PRE_FIX["N"]} / 面 {PRE_FIX["areas"]} -> {PRE_FIX["ratio"]:.4f}')
    print(f'  **差 N {base - PRE_FIX["N"]:+d} / 比 {ratio - PRE_FIX["ratio"]:+.4f}**'
          f'  (修理前はクリップ層を1枚も遮蔽物にしていなかった。T-M20)')

    print('\n== 棒 M-D1(追補122 §E。走る前に置いた。追補124 でも動かしていない)==')
    monop = all(r['O_wide(作者)'] >= r['面(連結)'] for r in per_page)
    print(f'  N/面 >= {BAR}           {ratio:.4f}  '
          f'{"満たす" if ratio >= BAR else "**満たさない**"}')
    print(f'  8頁すべてで N >= 面     {"PASS" if monop else "**破れた**"}')
    print(f'  判定保留の帯 |N/面-1| < {HOLD}  {ratio-1:.4f}  '
          f'{"**判定保留**" if abs(ratio-1) < HOLD else "帯の外"}')
    lo = min(N[k] for k in tri) / A
    hi = max(N[k] for k in tri) / A
    print(f'  三組の帯(min〜max。単調でなくてよい) {lo:.4f}〜{hi:.4f}  '
          f'{"**棒を跨ぐ -> 判定保留**" if lo < BAR <= hi else "跨がない"}')

    print('\n== 落としたもの(理由別。規律9)==')
    for k, v in sorted(tot.items()):
        if k.startswith('落: '):
            print(f'  {k[3:]:26s} {v:>6}')

    f = OUT / 'split_by_occlusion.csv'
    with open(f, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_page[0]))
        w.writeheader(); w.writerows(per_page)
    print(f'\nwrote {f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
