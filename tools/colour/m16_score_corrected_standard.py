"""M16: 追補10 で直した同定で標準を作り直し、**A0 と N1/N2/N3 を採点し直す**。

登録(追補10 §I3。走らせる前に commit した `a0af044`):
  - シルエット層の同定は**納品の木の全行**(フォルダー・描画0の層を兄弟に入れる)。
    用紙(`layer_type` 1584)は兄弟から外す。規則自体は `m11.clip_bases` と同一
  - **フォルダーがベースのときの面は、そのフォルダーの子孫の描画層のアルファの和**
  - **重なりは z 順で上の成分が勝つ**(木の深さ優先・`order_in_parent` 昇順 = 下から。
    合成が見せているものに合わせる)。**面積の小さい成分が勝つ版も印字して差を示す**

**2026-10-24 17:38 の訂正(走らせた結果、z 順が標準を壊した)**: z 順で塗ると
823成分が **355面**に落ち、`skima12H` では **148成分 → 1面**、`page0002` は 9 → 1 に
なった。原因は **罠 T-M3** —— **export されたアルファはクリップ後の姿ではない**ので、
**全面を申告するクリップベースが上にあると、可視は数画素でもページ全部を取る**。
「合成が見せているもの」は **export には存在しない**。
→ **主値は「面積の小さい成分が勝つ」(追補7 R3)に戻し、z 順の数は印字して
T-M3 の効きを示す。** これは**腕の点数ではなく標準自身の形**(148成分 → 1面)で
決めた判断である。棒・A0 の構成・床・帰無値・符号検定は動いていない。
  - 棒・A0 の構成(8段)・床(1e-5)・帰無値・符号検定は**追補7 のまま**。
    **動いたのは標準の同定だけである**

フォルダーの z 位置は**その子孫のうち最も上の位置**とする(フォルダーは親の中の
1つの枠を占め、その枠の上端まで覆うため)。**これも登録のうちで、ここに書いてから
走らせた。**

既知セル:
  - 同定: ベースの集合が `m15` の **183**(フォルダー16・描画0が1)と一致すること
  - IC-M6: 標準を予測として入れて region-F1 = 1.0(全8ページ)
  - IC-M7: ブロック拡大の経路(拡大した自分自身を標準にして 1.0、全8ページ)

**登録集合8ページなので主張は出ない。** 追補7 §G5 の表は消さず、並べて読む。
"""
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                       # noqa: E402

CSV = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_known8.csv')
OUT = Path('results/m16_score'); OUT.mkdir(parents=True, exist_ok=True)
PAPER_TYPE = '1584'
EXPECT_BASES = {'total': 183, 'folder': 16, 'zero_drawn': 1}


def page_rows():
    with open(CSV, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    return byp


def build_tree(rows):
    kids = collections.defaultdict(list)
    roots = []
    for r in rows:
        if r['parent_id'] == '':
            roots.append(r)
        else:
            kids[int(r['parent_id'])].append(r)
    for v in kids.values():
        v.sort(key=lambda r: int(r['order_in_parent']))
    return roots, kids


def z_order(roots, kids):
    """木の深さ優先で下から数えた通し番号。フォルダーは枠の中で展開する。"""
    z, idx = {}, [0]

    def walk(node):
        lid = int(node['layer_id'])
        for c in kids.get(lid, []):
            walk(c)
        idx[0] += 1
        z[lid] = idx[0]
    for r in roots:
        walk(r)
    return z


def descendants(lid, kids):
    out = []
    for c in kids.get(lid, []):
        out.append(int(c['layer_id']))
        out.extend(descendants(int(c['layer_id']), kids))
    return out


def tree_bases(rows):
    """追補10 §I3 の同定。{layer_id: row} とフォルダー判定。"""
    _, kids = build_tree(rows)
    sib = collections.defaultdict(list)
    for r in rows:
        if r['layer_type'] == PAPER_TYPE or r['parent_id'] == '':
            continue
        sib[int(r['parent_id'])].append(r)
    bases = {}
    for group in sib.values():
        group.sort(key=lambda r: int(r['order_in_parent']))
        base = None
        for r in group:
            if r['clip'] == '0':
                base = r
            elif base is not None:
                bases[int(base['layer_id'])] = base
    return bases, kids


def measure_standard(page, rows, grid, floor_px):
    """183ベースの原寸ラベル地図を2通り作る(z順 / 小さい成分が勝つ)。"""
    gh, gw = grid
    files = {l['layer_id']: l['file'] for l in page['layers']}
    bases, kids = tree_bases(rows)
    roots, _ = build_tree(rows)
    z = z_order(roots, kids)
    comps, st = [], collections.Counter()
    for lid, r in sorted(bases.items()):
        is_folder = r['folder'] == 'True'
        if is_folder:
            ids = [i for i in descendants(lid, kids) if i in files]
            zz = max([z[i] for i in descendants(lid, kids) if i in z] or [z[lid]])
            st['folder'] += 1
        else:
            ids = [lid] if lid in files else []
            zz = z[lid]
            if lid not in files:
                st['zero_drawn'] += 1
        m = None
        for i in ids:
            im = cv2.imread(str(m12.D / files[i]), cv2.IMREAD_UNCHANGED)
            a = (im[..., 3] >= m12.ALPHA_DRAWN)[:gh, :gw]
            m = a if m is None else (m | a)
            del im, a
        if m is None:
            st['no_raster'] += 1
            continue
        n, cl, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < floor_px:
                continue
            y, x, h, w = (stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT],
                          stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH])
            comps.append({'z': zz, 'area': a, 'layer_id': lid,
                          'folder': is_folder, 'box': (y, x, h, w),
                          'mask': np.ascontiguousarray(cl[y:y + h, x:x + w] == i)})
        del m, cl
    st['bases'] = len(bases)
    st['components'] = len(comps)

    def paint(order_key, prio_key):
        lab = np.zeros(grid, np.int32)
        own = np.zeros(grid, np.int32)
        for k, c in enumerate(sorted(comps, key=order_key), start=1):
            y, x, h, w = c['box']
            o, lb = own[y:y + h, x:x + w], lab[y:y + h, x:x + w]
            pr = prio_key(c)
            take = c['mask'] & ((o == 0) | (o > pr))
            o[take], lb[take] = pr, k
        return m12.filtered_labels(lab, floor_px)

    # z 順(登録): 下から塗って上が勝つ。優先度は z の逆順(小さいほど強い=上)。
    zmax = max([c['z'] for c in comps], default=1) + 1
    lab_z, area_z = paint(lambda c: c['z'], lambda c: zmax - c['z'])
    # 比較(印字のみ): 面積の小さい成分が勝つ(追補7 R3)
    lab_a, area_a = paint(lambda c: -c['area'], lambda c: c['area'])
    for c in comps:
        c.pop('mask')
    return (lab_z, area_z), (lab_a, area_a), st


def main():
    byp = page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    rows, t0 = [], time.time()
    tot = collections.Counter()
    for p in pages:
        pr = byp[p['source']]
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        im = cv2.imread(str(m12.D / p['preview']['file']), cv2.IMREAD_COLOR)
        (gt, garea), (gt_a, garea_a), st = measure_standard(p, pr, grid, floor_px)
        tot.update(st)

        known, _ = m12.score_from_small(gt_a, 1, gt_a, garea_a, floor_px)
        a0_lab = m12.equal_value_components(m12.quantise(im, m12.LEVELS))
        a0, _ = m12.filtered_labels(a0_lab, max(1, round(m12.FLOOR * pw * ph)))
        self_up = m12.ic_m7(a0, s, grid, floor_px)
        _ = gt  # z 順の標準は印字のためだけに保持する
        n1 = np.zeros((ph, pw), np.int32)
        n2 = np.ones((ph, pw), np.int32)
        n3, gr, gc = m12.grid_null((ph, pw), len(garea_a))
        # 主値 = 面積の小さい成分が勝つ標準(追補7 R3。上の訂正を参照)
        r_a0, _ = m12.score_from_small(a0, s, gt_a, garea_a, floor_px)
        r_n1, _ = m12.score_from_small(n1, s, gt_a, garea_a, floor_px)
        r_n2, _ = m12.score_from_small(n2, s, gt_a, garea_a, floor_px)
        r_n3, _ = m12.score_from_small(n3, s, gt_a, garea_a, floor_px)
        # 印字のみ: z 順の標準(T-M3 で壊れることを示すため)
        r_a0_alt, _ = m12.score_from_small(a0, s, gt, garea, floor_px)

        rows.append({'page': p['page'], 'scale': s, 'bases': st['bases'],
                     'folder_bases': st['folder'], 'components': st['components'],
                     'gt_areas': len(garea_a), 'gt_areas_zorder': len(garea),
                     'known_cell_f1': round(known['region_f1'], 6),
                     'ic_m7_f1': round(self_up['region_f1'], 6),
                     'a0_n': r_a0['n_pred'], 'a0_matched': r_a0['matched'],
                     'a0_prec': round(r_a0['precision'], 4),
                     'a0_rec': round(r_a0['recall'], 4),
                     'a0_f1': round(r_a0['region_f1'], 4),
                     'a0_f1_zorder': round(r_a0_alt['region_f1'], 4),
                     'n1_f1': round(r_n1['region_f1'], 4),
                     'n2_f1': round(r_n2['region_f1'], 4),
                     'n3_f1': round(r_n3['region_f1'], 4),
                     'n3_shape': f'{gr}x{gc}'})
        print(f'{p["page"]:44s} base {st["bases"]:>3}(fold {st["folder"]:>2}) '
              f'gt {len(garea_a):>4}(z {len(garea):>4}) '
              f'known {known["region_f1"]:.4f} '
              f'up {self_up["region_f1"]:.4f} rec {r_a0["recall"]:.4f} '
              f'A0 {r_a0["region_f1"]:.4f} N1 {r_n1["region_f1"]:.4f} '
              f'N2 {r_n2["region_f1"]:.4f} N3 {r_n3["region_f1"]:.4f} '
              f'{time.time()-t0:6.1f}s', flush=True)
        del gt, gt_a, a0, a0_lab

    with open(OUT / 'scores.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print('\n== 同定の既知セル(m15 と一致すること) ==')
    for k, v in EXPECT_BASES.items():
        got = tot['bases'] if k == 'total' else tot[k]
        print(f'  {k:11s} 期待 {v:>4} / 実測 {got:>4}  '
              f'{"OK" if got == v else "**DIFFERS**"}')
    print('\n== 既知セル ==')
    b6 = [r for r in rows if abs(r['known_cell_f1'] - 1.0) > 1e-9]
    b7 = [r for r in rows if abs(r['ic_m7_f1'] - 1.0) > 1e-9]
    print(f'  IC-M6 標準を予測に: {len(rows)-len(b6)}/{len(rows)} '
          f'-> {"PASS" if not b6 else "FAIL"}')
    print(f'  IC-M7 拡大の経路  : {len(rows)-len(b7)}/{len(rows)} '
          f'-> {"PASS" if not b7 else "FAIL"}')
    if b6 or b7:
        print('既知セルが再現しない。他の行は読まない。'); return 1

    print('\n== ページごと(プールしない。登録集合。追補7 §G5 と並べて読む) ==')
    print(f'{"page":30s} {"base":>4} {"fold":>4} {"成分":>5} {"gt":>4} '
          f'{"A0n":>5} {"照合":>4} {"prec":>6} {"rec":>6} {"A0":>7} '
          f'{"A0(z)":>7} {"N1":>6} {"N2":>6} {"N3":>6}')
    for r in rows:
        print(f'{r["page"][-28:]:30s} {r["bases"]:>4} {r["folder_bases"]:>4} '
              f'{r["components"]:>5} {r["gt_areas"]:>4} {r["a0_n"]:>5} '
              f'{r["a0_matched"]:>4} {r["a0_prec"]:>6.3f} {r["a0_rec"]:>6.3f} '
              f'{r["a0_f1"]:>7.4f} {r["a0_f1_zorder"]:>7.4f} '
              f'{r["n1_f1"]:>6.3f} {r["n2_f1"]:>6.3f} {r["n3_f1"]:>6.3f}')

    print('\n== judge: ページごとの符号検定(両側。n=8 では 8/8 以外は通らない) ==')
    for k, nm in (('n1_f1', 'N1 blank'), ('n2_f1', 'N2 whole_page'),
                  ('n3_f1', 'N3 grid')):
        win = sum(1 for r in rows if r['a0_f1'] > r[k])
        loss = sum(1 for r in rows if r['a0_f1'] < r[k])
        tie = len(rows) - win - loss
        pv = m12.judge_sign_test(win, loss)
        print(f'  A0 対 {nm:14s} 勝 {win} 負 {loss} 引 {tie}  p={pv:.4f} -> '
              f'{"棒を満たす" if (pv < 0.05 and win > loss) else "満たさない"}')

    print(f'\n== 標準の形 ==')
    print(f'  ベース {tot["bases"]}(フォルダー {tot["folder"]} / '
          f'描画0 {tot["zero_drawn"]} / ラスタ無し {tot["no_raster"]}) '
          f'成分 {tot["components"]} '
          f'主値(小面積優先)の面 {sum(r["gt_areas"] for r in rows)} '
          f'z順の面 {sum(r["gt_areas_zorder"] for r in rows)}')
    print('  A0(z) は z 順の標準に対する A0。印字のみで判定に使わない。')
    print('  z 順は T-M3 で壊れる(全面を申告するクリップベースが下を消す)。')
    print('登録集合なので主張は出ない。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
