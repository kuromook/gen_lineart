"""**漏れの経路を数える**(記述の走行、棒なし)。

**事前登録は 追補216**(走行前に commit `8ca63e9`)。土台は B のラベル図(`m126`)。入口 = 同じ域外の提案の中でマスクの面の縁を
線から 4 px より離れて横切る所。型は6つ(長い開口 / 薄い線 / 墨なしの短い口 / 手前で止まる / 向かい合う端 / ずれた端)。
塞ぐのは正解のマスクの縁を使う診断で、腕ではない。**門 E1・E2 が落ちたら表を読まない。**
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m77_standard_visible as m77                                      # noqa: E402
import m99_b_standard as m99                                # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402
import m116_mask_material_adopted4 as m116                                        # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m126_line_ends as m126                                        # noqa: E402

SRC = m126.OUT
OUT = Path('results/m129_leak_entrances')
K = m122.MAIN_K
NEAR_INK = 4.0
END_R = 12.0
LONG = 40.0
TOP = 10
NCROP = 4
PAD = 100
B_F1 = {'page0002': 0.303, 'gakusai3': 0.132}
LEAK_U = {'page0002': 0.856, 'gakusai3': 0.515}
TYPES = ['長い開口', '薄い線', '墨なしの短い口', '手前で止まる', '向かい合う端', 'ずれた端']
K3 = np.ones((3, 3), np.uint8)


def ok(b):
    return 'PASS' if b else '**FAIL**'


def raw_line_alpha(pg):
    """`線画` 層のアルファの最大(0〜255)。墨の閾値未満の薄い線を見るため。"""
    H, W = pg['H'], pg['W']
    if pg['key'] == 'page0002':
        p = next(p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages'] if p['page'].endswith('page0002'))
        files, root = {int(l['layer_id']): l['file'] for l in p['layers']}, m12.D
    else:
        p = next(p for p in json.load(open(m116.CM, encoding='utf-8'))['pages']
                 if '新規キャンバス' in p['source'] and 'gakusai3' in p['source'])
        files, root = {int(l['layer_id']): l['file'] for l in p['layers'] if l.get('file')}, m116.D
    vis, _ = m77.visible_rows(pg['rows'])
    amax = np.zeros((H, W), np.uint8)
    n = 0
    for r in vis:
        if r['layer_type'] == m104.PAPER or not m99.is_line(r):
            continue
        f = files.get(int(r['layer_id']))
        if f is None:
            continue
        im = cv2.imread(str(root / f), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        a = im[:H, :W, 3]
        np.maximum(amax[:a.shape[0], :a.shape[1]], a, out=amax[:a.shape[0], :a.shape[1]])
        n += 1
        del im
    return amax, n


def leak_and_f1(lab, Fl, U):
    keep, _ = m122.in_domain(lab, U)
    inside = np.zeros(int(lab.max()) + 1, bool)
    inside[keep] = True
    outside = (lab > 0) & ~inside[lab]
    f1 = m122.score(lab, Fl, keep)[m122.MAIN]['F1']
    return float((outside & U).sum()) / float(U.sum()), f1, outside, inside


def run_page(pg, kinds):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, _, faces, _ = m122.build_truth(pg, kinds)
    U = Fl > 0
    ink = pg['ink']
    arms = json.load(open(SRC / f'{key}_arms.json'))
    inkB = ink.astype(np.uint8).copy()
    for (ay, ax), (by, bx) in arms['bridges']:
        cv2.line(inkB, (int(ax), int(ay)), (int(bx), int(by)), 1, 1, cv2.LINE_8)
    inkB = inkB.astype(bool)
    lab = np.load(SRC / f'{key}_labB.npy')
    labR, _, _ = m122.regions(inkB, K, floor_px)
    e0 = bool((labR == lab).all())
    leak0, f0, outside, inside = leak_and_f1(lab, Fl, U)
    e1 = round(f0, 3) == B_F1[key] and e0
    e2 = round(leak0, 3) == LEAK_U[key]
    print(f'E1 B の図(読んだ図 = 線と橋から作り直した図 {e0})F1@0.5 {f0:.3f}(追補213 {B_F1[key]})  {ok(e1)}')
    print(f'E2 漏れ(U) {leak0:.3f}(追補213 §D {LEAK_U[key]})  {ok(e2)}')
    del labR
    if not (e1 and e2):
        print('**門が落ちた。表を読まない**')
        return None

    # §B 横切り画素 → 入口画素 → 入口
    cross = np.zeros((H, W), bool)
    for (a, b, fa, fb, sa, sb) in (
            (lab[:, :-1], lab[:, 1:], Fl[:, :-1], Fl[:, 1:], np.s_[:, :-1], np.s_[:, 1:]),
            (lab[:-1, :], lab[1:, :], Fl[:-1, :], Fl[1:, :], np.s_[:-1, :], np.s_[1:, :])):
        m = (a == b) & (a > 0) & ~inside[a] & (fa != fb)
        cross[sa] |= m
        cross[sb] |= m
    dist = cv2.distanceTransform((~inkB).astype(np.uint8), cv2.DIST_L2, 5)
    ent = cross & (dist > NEAR_INK)
    print(f'横切り画素 {int(cross.sum())} → 入口画素 {int(ent.sum())}(線から ≤ {NEAR_INK:g} px で落とした {int((cross & ~ent).sum())})')
    n, cl = cv2.connectedComponents(cv2.dilate(ent.astype(np.uint8), K3, iterations=3), connectivity=8)
    ys, xs = np.nonzero(ent)
    cid = cl[ys, xs]
    nC = n - 1
    length = np.bincount(cid, minlength=n)[1:] / 2.0
    paper = np.bincount(cid, weights=(Fl[ys, xs] == 0), minlength=n)[1:] > 0

    pts, dirs, _, _, _ = m126.endpoints(ink)
    tree = cKDTree(pts)
    ends = [set() for _ in range(n)]
    for k, lst in enumerate(tree.query_ball_point(np.stack([ys, xs], 1).astype(np.float64), END_R)):
        ends[cid[k]].update(lst)
    amax, nl = raw_line_alpha(pg)
    faint_px = (amax[ys, xs] > 0) & (amax[ys, xs] < m12.ALPHA_DRAWN)
    faint = np.bincount(cid, weights=faint_px, minlength=n)[1:] > 0
    print(f'入口 {nC}(薄い線を見た `線画` 層 {nl})')
    del amax

    typ = np.zeros(nC, np.int32)
    for c in range(nC):
        E = sorted(ends[c + 1])
        if length[c] >= LONG:
            typ[c] = 0
        elif not E:
            typ[c] = 1 if faint[c] else 2
        elif len(E) == 1:
            typ[c] = 3
        else:
            fac = any(m126.facing(pts[i], dirs[i], pts[j], dirs[j], m126.P_MAIN)
                      or m126.facing(pts[j], dirs[j], pts[i], dirs[i], m126.P_MAIN)
                      for x, i in enumerate(E) for j in E[x + 1:])
            typ[c] = 4 if fac else 5

    def close(sel):
        """sel: 入口の番号(0..nC-1)の bool。その入口画素を線に足して作り直す。"""
        m = np.zeros((H, W), np.uint8)
        pick = sel[cid - 1]
        m[ys[pick], xs[pick]] = 1
        inkC = inkB | cv2.dilate(m, K3).astype(bool)
        labC, _, _ = m122.regions(inkC, K, floor_px)
        lk, f1, _, _ = leak_and_f1(labC, Fl, U)
        del labC, inkC
        return lk, f1

    print(f'\n  {"型":<12}{"入口":>6}{"長さの和":>10}{"紙に接する":>10} | 型だけ塞ぐ: 漏れ(U) / F1@0.5')
    rows = []
    for t, nm in enumerate(TYPES):
        sel = typ == t
        lk, f1 = close(sel) if sel.any() else (leak0, f0)
        print(f'  {nm:<12}{int(sel.sum()):>6}{length[sel].sum():>10.0f}{int(paper[sel].sum()):>10} | {lk:.3f} / {f1:.3f}')
        rows.append(dict(page=key, kind=nm, n=int(sel.sum()), length=float(length[sel].sum()), leak_u=lk, f1=f1))
    lkA, f1A = close(np.ones(nC, bool))
    print(f'  {"全部":<12}{nC:>6}{length.sum():>10.0f}{int(paper.sum()):>10} | {lkA:.3f} / {f1A:.3f}')
    print(f'  塞ぐ前                                         | {leak0:.3f} / {f0:.3f}')
    if lkA > 0.05:
        print(f'  **全部塞いでも漏れ(U) {lkA:.3f} > 0.05 —— 入口の定義が経路を取りこぼしている(登録 §D (i))**')

    print(f'\n  長さの上位 {TOP} を1つずつ単独で塞ぐ')
    order = np.argsort(-length, kind='stable')[:TOP]
    for r_, c in enumerate(order.tolist()):
        sel = np.zeros(nC, bool)
        sel[c] = True
        lk, f1 = close(sel)
        yy, xx = ys[cid == c + 1], xs[cid == c + 1]
        print(f'  {r_ + 1:>2}. 入口 {c + 1:>5} {TYPES[typ[c]]:<10} 長さ {length[c]:>6.0f}  紙 {"はい" if paper[c] else "いいえ"}'
              f'  中心 x {xx.mean():.0f} y {yy.mean():.0f} | 漏れ(U) {leak0:.3f} → {lk:.3f} / F1 {f1:.3f}')

    # 代表例
    rng = np.random.default_rng(0)
    edge = np.zeros((H, W), bool)
    edge[:, :-1] |= Fl[:, :-1] != Fl[:, 1:]
    edge[:-1, :] |= Fl[:-1, :] != Fl[1:, :]
    for t, nm in enumerate(TYPES):
        cs = np.nonzero(typ == t)[0]
        pick = cs if len(cs) <= NCROP else rng.choice(cs, NCROP, replace=False)
        for c in sorted(pick.tolist()):
            yy, xx = ys[cid == c + 1], xs[cid == c + 1]
            Y0, Y1 = max(0, yy.min() - PAD), min(H, yy.max() + 1 + PAD)
            X0, X1 = max(0, xx.min() - PAD), min(W, xx.max() + 1 + PAD)
            im = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
            im[outside[Y0:Y1, X0:X1]] = (240, 200, 215)
            im[edge[Y0:Y1, X0:X1]] = (230, 80, 0)
            e = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            e[yy - Y0, xx - X0] = 1
            im[cv2.dilate(e, K3).astype(bool)] = (0, 0, 230)
            im[inkB[Y0:Y1, X0:X1]] = 0
            cv2.imwrite(str(OUT / f'{key}_t{t + 1}_{nm}_c{c + 1}_len{int(length[c])}_x{X0}_y{Y0}.png'), im)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    allrows = []
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, g6 = loader(kinds)
        if not g6:
            print('**線の読み込みの門が落ちた**')
            return
        r = run_page(pg, kinds)
        if r is None:
            return
        allrows += r
        del pg
    with open(OUT / 'by_type.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(f'\n表: {OUT / "by_type.csv"}  切り出し: {OUT}/')


if __name__ == '__main__':
    main()
