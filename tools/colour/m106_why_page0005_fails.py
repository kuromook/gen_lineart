"""`page0005` の面が、なぜ手描きの `線画` で囲まれていないのかを面ごとに見る。

**器具は写さない。** 段の組み立ては `m104` の関数をそのまま呼ぶ。
`m104.main()` は頁を選べないので、**この器具は `m104` の頁ごとの前段を同じ順序で
呼び直し、`T4` の境界が登録済みの値を再現することを門にする**(既知セル)。

**既知セル(第3版の走行、`results/m104_b_standard_v3/b_standard_v3.csv` の `page0005` 行)**:
  閉じた面 **482** / 線率 中央 **0.0000** / 線率<0.5 の面 **344** / 床下 **5075**
**再現しなければ、この器具の出す面の位置は読まない。**

出すもの(**面ごと**。頁ごとの合計で論じない):
  - 面の重心・面積・線率
  - **その面の境目から最も近い `線画` の墨までの距離**(原寸の画素)
  - **その面が `線画` の外接箱の中に在るか**
  - 切り出しの窓(**規律6**。モンタージュで論じない)
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

OUT = Path('results/m106_why_page0005')
PAGE = 'page0005'
IC = {'closed': 482, 'med': 0.0000, 'lt05': 344, 'under': 5075}


def build(p, byp, kinds):
    """`m104` の前段を同じ順序で呼ぶ。返すのは墨と `T4` の境界。"""
    rows_vis, _ = m77.visible_rows(byp[p['source']])
    roots, kids = m16.build_tree(rows_vis)
    cbase = m87.clip_base_map(rows_vis)
    files = {l['layer_id']: l['file'] for l in p['layers']}
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    px = grid[0] * grid[1]
    floor_px = max(1, int(round(px * m12.FLOOR)))
    cache = {}
    body = [r for r in rows_vis if r['layer_type'] != m104.PAPER]

    skip = set()
    for r in body:
        if kinds.get((p['source'], str(r['layer_id']))) == m104.TEXT_KIND:
            skip.add(int(r['layer_id']))

    ink = np.zeros(grid, bool)
    n_line = 0
    for r in body:
        if not m99.is_line(r) or m102.is_sketch(r):
            continue
        a = m87.alpha_of(r['layer_id'], files, grid, cache)
        if a is None:
            continue
        ink[:a.shape[0], :a.shape[1]] |= a
        n_line += 1

    E = np.zeros(grid, bool)
    drop = collections.Counter()
    n_edge = 0
    for r in body:
        if (r.get('opacity') or '0') == '0':
            continue
        if (r.get('composite') or '0') == m104.MULTIPLY:
            continue
        m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True, drop)
        if m is None:
            continue
        leaf = r['folder'] != 'True'
        rect, boxr = m102.rect_of(m87.alpha_of(r['layer_id'], files, grid, cache),
                                  px) if leaf else (0.0, 0.0)
        kind = kinds.get((p['source'], str(r['layer_id'])))
        mat = leaf and (kind == 'image_material'
                        or (not m102.is_tone_correction(r) and rect >= m102.RECT_BAR
                            and boxr < m102.FULL_CANVAS))
        if r['clip'] == '1' or m102.is_named(r) or mat or m102.is_sketch(r):
            del m
            continue                                  # R2(i)(iii) / R3 / R4
        has_text = (int(r['layer_id']) in skip
                    or any(i in skip for i in m16.descendants(int(r['layer_id']), kids)))
        if has_text:
            m6 = m104.occluder_mask_r6(r, kids, files, grid, cache, cbase, skip,
                                       collections.Counter())
            if m6 is None:
                del m
                continue
            E |= m99.inner_edge(m6)
            del m6
        else:
            E |= m99.inner_edge(m)
        n_edge += 1
        del m
    cache.clear()
    om = m104.open_from(ink)
    return ink, (ink | (E & om)), floor_px, n_line, n_edge


def faces(boundary, ink, floor_px):
    """`m102.faces_and_ratios` と同じ組み立てで、**面ごとの位置も返す**。"""
    n, lab, st, cen = cv2.connectedComponentsWithStats((~boundary).astype(np.uint8),
                                                       connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    H, W = boundary.shape
    k3 = np.ones((3, 3), np.uint8)
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    rows, under = [], 0
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < floor_px:
            under += 1
            continue
        if i in border:
            continue
        x0 = max(0, int(st[i, cv2.CC_STAT_LEFT]) - 1)
        y0 = max(0, int(st[i, cv2.CC_STAT_TOP]) - 1)
        x1 = min(W, x0 + int(st[i, cv2.CC_STAT_WIDTH]) + 3)
        y1 = min(H, y0 + int(st[i, cv2.CC_STAT_HEIGHT]) + 3)
        m = (lab[y0:y1, x0:x1] == i)
        ring = cv2.dilate(m.astype(np.uint8), k3).astype(bool) & ~m \
            & boundary[y0:y1, x0:x1]
        nr = int(ring.sum())
        ratio = (int((ring & ink[y0:y1, x0:x1]).sum()) / nr) if nr else 0.0
        d = float(dist[y0:y1, x0:x1][ring].min()) if nr else -1.0
        rows.append(dict(face=i, area=a, ratio=ratio, ring=nr, d_ink=d,
                         cx=float(cen[i][0]), cy=float(cen[i][1]),
                         x0=x0, y0=y0, x1=x1, y1=y1))
        del m, ring
    return rows, under


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if PAGE in q['page']][0]
    print(f'頁 {p["page"]}')

    ink, bnd, floor_px, n_line, n_edge = build(p, byp, kinds)
    print(f'  線 {n_line} 枚 / 縁を出した層 {n_edge} / 床 {floor_px} px')
    rows, under = faces(bnd, ink, floor_px)
    rs = sorted(r['ratio'] for r in rows)
    med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs)//2-1]+rs[len(rs)//2])/2
    lt05 = sum(1 for r in rows if r['ratio'] < 0.5)
    got = {'closed': len(rows), 'med': round(med, 4), 'lt05': lt05, 'under': under}
    print('  門(既知セル = 第3版の `page0005` 行):')
    ok = True
    for k, v in IC.items():
        good = abs(got[k] - v) < 1e-9
        ok &= good
        print(f'    {k:<8} 期待 {v:<10} 実測 {got[k]:<10} {"PASS" if good else "**FAIL**"}')
    if not ok:
        print('  **門が落ちた。面の位置は読まない**')
        return 1

    ys, xs = np.nonzero(ink)
    bb = (xs.min(), ys.min(), xs.max(), ys.max())
    print(f'  `線画` の外接箱 x {bb[0]}–{bb[2]} / y {bb[1]}–{bb[3]}'
          f'(画布 {ink.shape[1]}×{ink.shape[0]})')

    def inbox(r):
        return bb[0] <= r['cx'] <= bb[2] and bb[1] <= r['cy'] <= bb[3]

    for r in rows:
        r['in_ink_box'] = int(inbox(r))
    with open(OUT / 'faces_page0005.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\n  **線画の外接箱の中 / 外**(面ごと。合計で論じない)')
    for tag, sel in (('箱の中', [r for r in rows if r['in_ink_box']]),
                     ('箱の外', [r for r in rows if not r['in_ink_box']])):
        if not sel:
            print(f'    {tag}: 0 面')
            continue
        v = sorted(r['ratio'] for r in sel)
        z = sum(1 for r in sel if r['ratio'] == 0.0)
        dd = sorted(r['d_ink'] for r in sel)
        print(f'    {tag}: {len(sel):>4} 面 / 線率 中央 {v[len(v)//2]:.4f} / '
              f'線率ちょうど0 {z} ({z/len(sel):.1%}) / '
              f'墨までの距離 中央 {dd[len(dd)//2]:.0f}px 最大 {dd[-1]:.0f}px')

    print('\n  **墨までの距離で束ねる**(その面の境目から最も近い `線画` の墨まで)')
    bins = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 10 ** 9)]
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r['d_ink'] < hi]
        if not sel:
            continue
        v = sorted(r['ratio'] for r in sel)
        z = sum(1 for r in sel if r['ratio'] == 0.0)
        print(f'    {lo:>4}–{hi if hi < 10**9 else "∞":<6} {len(sel):>4} 面 / '
              f'線率 中央 {v[len(v)//2]:.4f} / ちょうど0 {z}')
    print(f'\n  表: {OUT / "faces_page0005.csv"}')
    return 0




# --------------------------------------------------------------------------- 切り出し
# **窓の規則は先に決める**(規律6。`m105` で窓を外した反省 —— 構造の規則で決め、
# 絵を見てから動かさない):
#   群 = 「墨までの距離 < 1px」と「墨までの距離 >= 256px」の2つ
#   面 = **その群の面積の中央値に当たる面**(最大を選ばない。代表であって見栄えではない)
#   窓 = その面の重心を中心とする原寸 1200×1200(画布の端で寄せる)
# 出すもの: `線画` の墨(原寸)/ `T4` の境界(原寸)/ その面(原寸)/ プレビュー(**1/4。
#   原寸の合成は抽出側がまだ持っていない —— 拡大はしない**)
WIN = 1200


def crops(p, ink, bnd, rows):
    import m12_score_a0_nulls as _m12
    prev = cv2.imread(str(_m12.D / f'{p["page"]}_preview.png'), cv2.IMREAD_UNCHANGED)
    s = int(round(p['preview']['縮小率']))
    H, W = ink.shape
    groups = (('near', [r for r in rows if r['d_ink'] < 1]),
              ('far', [r for r in rows if r['d_ink'] >= 256]))
    out = []
    for tag, sel in groups:
        if not sel:
            continue
        sel = sorted(sel, key=lambda r: r['area'])
        f = sel[len(sel) // 2]
        cx, cy = int(f['cx']), int(f['cy'])
        x0 = min(max(0, cx - WIN // 2), max(0, W - WIN))
        y0 = min(max(0, cy - WIN // 2), max(0, H - WIN))
        x1, y1 = min(W, x0 + WIN), min(H, y0 + WIN)
        print(f'  {tag}: 面 {f["face"]} 面積 {f["area"]:,}px 線率 {f["ratio"]:.4f} '
              f'墨まで {f["d_ink"]:.0f}px / 窓 x {x0}–{x1} y {y0}–{y1}(原寸)')
        for name, img in (('ink', ink), ('boundary', bnd)):
            a = (~img[y0:y1, x0:x1]).astype(np.uint8) * 255   # 墨を黒で
            q = OUT / f'{tag}_{name}_native.png'
            cv2.imwrite(str(q), a)
            out.append(q)
        if prev is not None:
            pc = prev[y0 // s:y1 // s, x0 // s:x1 // s]
            q = OUT / f'{tag}_preview_quarter.png'
            cv2.imwrite(str(q), pc)
            out.append(q)
    return out


if __name__ == '__main__' and '--crops' not in sys.argv:
    sys.exit(main())

if '--crops' in sys.argv:
    OUT.mkdir(parents=True, exist_ok=True)
    _kinds = m102.load_kinds()
    _byp = m16.page_rows()
    _p = [q for q in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
          if PAGE in q['page']][0]
    _ink, _bnd, _fl, _nl, _ne = build(_p, _byp, _kinds)
    _rows, _ = faces(_bnd, _ink, _fl)
    for _q in crops(_p, _ink, _bnd, _rows):
        print('   ', _q)
