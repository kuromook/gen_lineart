"""**マスクの面を正解源にする第1走行**。点数は出さない —— 面が線で何個に分かれるか / 縁が線に沿う割合 / 隙間。

**事前登録は 追補200**(走行前に commit `2654a66`)。針路は追補199(作者 2026-11-02)。
面 = 材料の層のマスクの見える側 − 同じ親で `order_in_parent` が大きい材料の層のマスクの見える側。
材料は追補196 訂正の K8r と同じ語(`m116` の語を import する。書き直さない)。
帰無値 N1 左右反転した線 / N2 右へ 32 px ずらした線。**門が1本でも落ちたら表を読まない。**
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
import m116_mask_material_adopted4 as m116                           # noqa: E402

OUT = Path('results/m119_mask_faces_and_lines')
MD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/dataset_colour_pilot_masks_20261101')
FLOOR = 1e-5
DISTS = (1, 3, 6)
GAP = 3
SHIFT = 32
PAD = 48
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

# 門の期待値(追補200 §E。**こちらのパイプラインの外から来た数**)
G2 = {'page0002': (6, 12), 'gakusai3': (11, 30)}
G3 = {'HAIR': 464900, 'UNDERWARE': 1493042, 'TEETH': 58304, 'EYE': 37986, 'CLOTH': 1057}
N_LINE_GAKUSAI3 = 5


def ok(b):
    return 'PASS' if b else '**FAIL**'


def truthy(v):
    return str(v) in ('True', 'true', '1')


def vis_side(path, H, W):
    g = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    g = g[..., 0] if g.ndim == 3 else g
    return np.ascontiguousarray(g[:H, :W] <= 127)                    # **反転。<=127 が見える側**(T-M46)


def classify(rows, mm, area, kinds, pp):
    """`m116` と同じ語で材料を選ぶ。返り値: 材料の行(manifest)、材料外で has_mask の可視の葉、落とした数。"""
    vis, _ = m77.visible_rows(rows)
    visid = {int(r['layer_id']) for r in vis}
    kids = {int(r['parent_id']) for r in rows if r['parent_id']}
    idx = {int(r['layer_id']): r for r in rows}
    mat, other, drop = [], [], collections.Counter()
    for r in mm:
        lid = int(r['layer_id'])
        if r['folder'] or lid in kids:
            drop['フォルダー'] += 1
            continue
        if lid not in visid:
            drop['非表示'] += 1
            continue
        row = idx[lid]
        name = row['name'] or ''
        ge = r['mask_px_ge128']
        why = None
        if int(r['layer_type']) not in (2, 3):
            why = 'layer_type'
        elif ge == 0:
            why = '全面隠す'
        elif ge >= area:
            why = '全面通す'
        elif any(t in name for t in m116.LINE):
            why = '線'
        elif any(t in name.lower() or t in name for t in m116.HIGHLIGHT):
            why = 'ハイライト'
        elif kinds.get((pp, str(lid))) == 'filter':
            why = 'filter'
        if why:
            drop[why] += 1
            other.append((r, row))
        else:
            mat.append((r, row))
    return mat, other, drop


def components(m, conn, floor_px):
    n, lb, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=conn)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= floor_px]
    return lb, st, big, (n - 1 - len(big))


def measure(F, inks, dists, floor_px, H, W):
    """面 F(全画布の bool)に (i)〜(iii) を当てる。`inks` は {名前: (墨, 距離)}。"""
    ys, xs = np.nonzero(F)
    y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
    f = F[y0:y1, x0:x1]
    edge = f & ~cv2.erode(f.astype(np.uint8), K4, borderValue=1).astype(bool)
    on_border = int(F[0].sum() + F[-1].sum() + F[:, 0].sum() + F[:, -1].sum())
    ne = int(edge.sum())
    out = dict(area=int(f.sum()), edge=ne, edge_on_canvas_border=on_border)
    parts = {}
    for nm, (ink, dist) in inks.items():
        k = ink[y0:y1, x0:x1]
        d = dist[y0:y1, x0:x1]
        lb, _, big, under = components(f & ~k, 4, floor_px)
        out[f'{nm}_n_split'] = len(big)
        out[f'{nm}_split_under_floor'] = under
        out[f'{nm}_ink_in_face'] = float((f & k).sum() / f.sum())
        for t in DISTS:
            out[f'{nm}_along_le{t}'] = float((edge & (d <= t)).sum() / ne) if ne else float('nan')
        g = edge & (d > GAP)
        n, _, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
        L = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0, int)
        out[f'{nm}_gap_runs'] = int(n - 1)
        out[f'{nm}_gap_px'] = int(L.sum())
        out[f'{nm}_gap_max'] = int(L.max()) if len(L) else 0
        if nm == 'line':
            parts = dict(lb=lb, big=big, gap=g, box=(y0, y1, x0, x1))
    return out, parts


def load_page0002(kinds):
    byp = m16.page_rows()
    p = next(p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
             if p['page'].endswith('page0002'))
    ink, bnd, floor_px, _, _ = m106.build(p, byp, kinds)
    del bnd
    H, W = ink.shape
    rows = byp[p['source']]
    mm = [r for r in json.load(open(MD / 'layer_masks_manifest.json', encoding='utf-8'))['layers']
          if r['page_path'] == p['source']]
    # G6: 2経路(`m106.build` の墨 と `線画` の層のアルファを直接読んだ和)
    vis, _ = m77.visible_rows(rows)
    files = {l['layer_id']: l['file'] for l in p['layers']}
    direct = np.zeros((H, W), bool)
    n_line, cache = 0, {}
    for r in vis:
        if r['layer_type'] == m104.PAPER or not m99.is_line(r):
            continue
        a = m87.alpha_of(r['layer_id'], files, (H, W), cache)
        if a is None:
            continue
        direct[:a.shape[0], :a.shape[1]] |= a
        n_line += 1
    g6 = bool((direct == ink).all())
    print(f'G6 page0002 の線: m106.build の墨 {int(ink.sum())} px / 直接読んだ和 {int(direct.sum())} px '
          f'(層 {n_line})  {ok(g6)}')
    return dict(key='page0002', pp=p['source'], rows=rows, mm=mm, mdir=MD, ink=ink, H=H, W=W,
                floor_px=max(1, int(round(H * W * FLOOR)))), g6


def load_gakusai3(kinds):
    pages = json.load(open(m116.CM, encoding='utf-8'))['pages']
    p = next(p for p in pages if '新規キャンバス' in p['source'] and 'gakusai3' in p['source'])
    rows = [r for r in csv.DictReader(open(m116.V2, encoding='utf-8-sig'))
            if r['page_path'] == p['source']]
    mm = [r for r in json.load(open(m116.MM, encoding='utf-8'))['layers']
          if r['page_path'] == p['source']]
    W, H = p['canvas']
    files = {int(l['layer_id']): l['file'] for l in p['layers'] if l.get('file')}
    vis, _ = m77.visible_rows(rows)
    ink = np.zeros((H, W), bool)
    names = []
    for r in vis:
        if not m99.is_line(r):
            continue
        f = files.get(int(r['layer_id']))
        if f is None:
            print(f'  線の層 L{r["layer_id"]} にラスタが無い(飛ばした)')
            continue
        im = cv2.imread(str(m116.D / f), cv2.IMREAD_UNCHANGED)
        a = im[..., 3] >= m12.ALPHA_DRAWN
        ink[:a.shape[0], :a.shape[1]] |= a[:H, :W]
        names.append(f'L{r["layer_id"]} {r["name"]}')
        del im, a
    g6 = len(names) == N_LINE_GAKUSAI3
    print(f'G6 gakusai3 の線: 読んだ層 {len(names)}(期待 {N_LINE_GAKUSAI3})墨 {int(ink.sum())} px  {ok(g6)}')
    print('    ' + ' / '.join(names))
    return dict(key='gakusai3', pp=p['source'], rows=rows, mm=mm, mdir=m116.D, ink=ink, H=H, W=W,
                floor_px=max(1, int(round(H * W * FLOOR)))), g6


def run_page(pg, kinds, fails, faces_out, crops):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    area = H * W
    print(f'\n======== {key}  {pg["pp"]}  {W}x{H}  床 {floor_px} px')
    mat, other, drop = classify(pg['rows'], pg['mm'], area, kinds, pg['pp'])
    print(f'  材料外(理由別。規律9): ' + ' / '.join(f'{k} {v}' for k, v in sorted(drop.items())))

    # G1 向き
    vis = {int(r['layer_id']): vis_side(pg['mdir'] / r['mask_file'], H, W) for r, _ in mat}
    inv = sum(1 for r, _ in mat if int(vis[int(r['layer_id'])].sum()) == r['mask_px_ge128'])
    print(f'G1 向き: 反転が mask_px_ge128 と一致 {inv} / {len(mat)}  {ok(inv == len(mat))}')
    fails += inv != len(mat)

    info = {int(r['layer_id']): row for r, row in mat}
    parents = {row['parent_id'] for row in info.values()}
    print(f'G5 材料の層の親 {sorted(parents)}  {ok(len(parents) == 1)}')
    fails += len(parents) != 1
    order = sorted(info, key=lambda l: int(info[l]['order_in_parent']))

    n0 = sum(len(components(vis[l], 8, floor_px)[2]) for l in order)
    print(f'G2 引く前の材料 {len(order)} 層・{n0} 面(期待 {G2[key]})  {ok((len(order), n0) == G2[key])}')
    fails += (len(order), n0) != G2[key]

    if key == 'page0002':
        sk = next(l for l in order if info[l]['name'] == 'SKIN')
        got = {info[l]['name']: int((vis[sk] & vis[l]).sum()) for l in order if l != sk}
        print(f'G3 SKIN との重なり {got}  {ok(got == G3)}')
        fails += got != G3

    # 面を作る(上の材料の和を引く)
    face_mask, g4 = {}, True
    above = np.zeros((H, W), bool)
    sub_px = {}
    for l in reversed(order):                                       # 上から
        m = vis[l] & ~above
        sub_px[l] = int((vis[l] & above).sum())
        g4 &= int(m.sum()) + sub_px[l] == int(vis[l].sum())
        face_mask[l] = m
        above |= vis[l]
    g4 &= sub_px[order[-1]] == 0
    print(f'G4 恒等(引いた後 + 引かれた = 引く前、最上層は 0)  {ok(g4)}')
    fails += not g4

    # 材料外を引いた場合に追加で消える画素(差の内訳。条例6)
    par = next(iter(parents))
    for r, row in other:
        if row['parent_id'] != par:
            continue
        o = vis_side(pg['mdir'] / r['mask_file'], H, W)
        below = [l for l in order if int(info[l]['order_in_parent']) < int(row['order_in_parent'])]
        extra = sum(int((face_mask[l] & o).sum()) for l in below)
        print(f'  材料外 L{row["layer_id"]} {row["name"]}(順 {row["order_in_parent"]})を引けば追加で消える {extra} px')

    if key == 'page0002':
        nm = {info[l]['name']: l for l in order}
        author = vis[nm['SKIN']] & ~vis[nm['HAIR']] & ~vis[nm['CLOTH']] & ~vis[nm['UNDERWARE']]
        rule = face_mask[nm['SKIN']]
        print(f'  作者の式(肌−髪−CLOTH−UNDERWARE)と規則の差: 規則にだけ {int((rule & ~author).sum())} px'
              f' / 作者の式にだけ {int((author & ~rule).sum())} px')
    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return fails

    ink = pg['ink']
    n1 = np.ascontiguousarray(ink[:, ::-1])
    n2 = np.zeros_like(ink)
    n2[:, SHIFT:] = ink[:, :-SHIFT]
    inks = {}
    for nm_, k in (('line', ink), ('N1', n1), ('N2', n2)):
        inks[nm_] = (k, cv2.distanceTransform((~k).astype(np.uint8), cv2.DIST_L2, 5))

    print(f'\n  {"層":<14}{"順":>4}{"引かれた":>11}{"面":>4}{"床未満":>6}')
    for l in order:
        lb, st, big, under = components(face_mask[l], 8, floor_px)
        print(f'  L{l:<4}{info[l]["name"]:<9}{info[l]["order_in_parent"]:>4}{sub_px[l]:>11}{len(big):>4}{under:>6}')
        for j, i in enumerate(sorted(big, key=lambda i: -st[i, cv2.CC_STAT_AREA])):
            F = lb == i
            row, parts = measure(F, inks, DISTS, floor_px, H, W)
            row = dict(page=key, layer_id=l, name=info[l]['name'], order=info[l]['order_in_parent'],
                       face=j, area_pct=row['area'] / area * 100, **row)
            faces_out.append(row)
            crops.setdefault((key, l), []).append((None, parts))
            del F
        del lb
    return fails


def print_table(faces):
    print('\n=== 面ごと(**頁ごと。プールしない**。記述、棒に使わない)===')
    h = (f'{"頁":<9}{"層":<14}{"面":>3}{"面積%":>8} | {"分割 線/N1/N2":>14} | '
         + ' | '.join(f'沿う≤{t} 線/N1/N2' for t in DISTS) + f' | {"隙間 本/px/最長(線)":>20}')
    print(h)
    for r in faces:
        s = (f'{r["page"]:<9}L{r["layer_id"]:<4}{r["name"]:<9}{r["face"]:>3}{r["area_pct"]:>8.3f} | '
             f'{r["line_n_split"]:>4}/{r["N1_n_split"]:>3}/{r["N2_n_split"]:>3}    | ')
        s += ' | '.join(f'{r[f"line_along_le{t}"]:.2f}/{r[f"N1_along_le{t}"]:.2f}/{r[f"N2_along_le{t}"]:.2f}'
                        for t in DISTS)
        s += f' | {r["line_gap_runs"]:>5}/{r["line_gap_px"]:>7}/{r["line_gap_max"]:>6}'
        print(s)
    print('\n=== 頁ごとの要約(縁の画素で重み付け。**2頁をプールしない**)===')
    for pg in sorted({r['page'] for r in faces}):
        rs = [r for r in faces if r['page'] == pg]
        ne = sum(r['edge'] for r in rs)
        s = f'{pg:<9} 面 {len(rs)} / 縁 {ne} px(画布外周の画素 {sum(r["edge_on_canvas_border"] for r in rs)} は縁に数えない)'
        print(s)
        for t in DISTS:
            v = {n: sum(r[f'{n}_along_le{t}'] * r['edge'] for r in rs) / ne for n in ('line', 'N1', 'N2')}
            print(f'    沿う ≤{t}px: 線 {v["line"]:.3f} / N1 {v["N1"]:.3f} / N2 {v["N2"]:.3f}  '
                  f'線が両方を上回る面 {sum(1 for r in rs if r[f"line_along_le{t}"] > max(r[f"N1_along_le{t}"], r[f"N2_along_le{t}"]))}/{len(rs)}')
        gp = sum(r['line_gap_px'] for r in rs)
        print(f'    隙間(線から >{GAP}px): {sum(r["line_gap_runs"] for r in rs)} 本 / {gp} px = 縁の {gp / ne:.3f}')
        print(f'    分割 ≥2 の面: 線 {sum(1 for r in rs if r["line_n_split"] >= 2)} / N1 {sum(1 for r in rs if r["N1_n_split"] >= 2)}'
              f' / N2 {sum(1 for r in rs if r["N2_n_split"] >= 2)}(面 {len(rs)})')


def write_crops(pg, crops):
    rng = np.random.default_rng(0)
    ink = pg['ink']
    for (key, l), lst in crops.items():
        if key != pg['key']:
            continue
        ys0, ys1, xs0, xs1 = pg['H'], 0, pg['W'], 0
        for _, pa in lst:
            y0, y1, x0, x1 = pa['box']
            ys0, ys1, xs0, xs1 = min(ys0, y0), max(ys1, y1), min(xs0, x0), max(xs1, x1)
        Y0, Y1 = max(0, ys0 - PAD), min(pg['H'], ys1 + PAD)
        X0, X1 = max(0, xs0 - PAD), min(pg['W'], xs1 + PAD)
        im = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
        for _, pa in lst:
            y0, y1, x0, x1 = pa['box']
            sub = im[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
            base = rng.integers(60, 200, 3)
            for k, i in enumerate(pa['big']):
                m = pa['lb'] == i
                c = np.clip(base + (k % 4) * 30 - 45, 0, 255)
                sub[m] = (sub[m] * 0.35 + c * 0.65).astype(np.uint8)
        im[ink[Y0:Y1, X0:X1]] = (0, 0, 0)
        for _, pa in lst:
            y0, y1, x0, x1 = pa['box']
            g = cv2.dilate(pa['gap'].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            im[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0][g] = (0, 0, 230)
        name = next(r for r in pg['rows'] if int(r['layer_id']) == l)['name']
        out = OUT / f'{key}_L{l}_{name}_x{X0}_y{Y0}.png'
        cv2.imwrite(str(out), im)
        print(f'  切り出し {out}  {X1 - X0}x{Y1 - Y0}(原寸)')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    faces, fails = [], 0
    for loader in (load_page0002, load_gakusai3):
        pg, g6 = loader(kinds)
        fails += not g6
        crops = {}
        fails = run_page(pg, kinds, fails, faces, crops)
        if fails:
            return
        write_crops(pg, crops)
        del pg
    print_table(faces)
    with open(OUT / 'faces.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(faces[0]))
        w.writeheader()
        w.writerows(faces)
    print(f'\n表: {OUT / "faces.csv"}')
    print('**棒は置いていない(追補200)。他の6頁は材料 0 なので面が無い。**')


if __name__ == '__main__':
    main()
