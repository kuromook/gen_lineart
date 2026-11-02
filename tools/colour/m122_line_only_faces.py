"""**線だけから面を見つける第1走行**。正解はマスクの面を線で分けた部分(`T_fine`)。

**事前登録は 追補204**(走行前に commit `daea678`)。腕は線の墨だけを見る(L0 / Lk = 隙間を k で閉じる、主は L4)。
帰無値 N1 反転した線で L4 / N2 格子 / N0 神託。評価域 = マスクの面の和に 50% 以上入る提案。
採点は IoU の貪欲 1対1 で、**提案の番号を提案の側から振る**(`m114.score` の T-M49 を直した)。
**門が1本でも落ちたら表を読まない。**
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402

OUT = Path('results/m122_line_only_faces')
M119_CSV = Path('results/m119_mask_faces_and_lines/faces.csv')
KS = (0, 1, 2, 4, 8, 16)
MAIN_K = 4
THR = (0.3, 0.5, 0.7)
MAIN = 0.5
C2_BAR = 0.50
FLAGGED = {('gakusai3', 29): '乳輪', ('gakusai3', 26): '瞳', ('page0002', 63): '瞳'}


def ok(b):
    return 'PASS' if b else '**FAIL**'


def compact(lab, n, st, floor_px):
    """床以上の成分だけを 1.. に振り直す。返り値: 新しい図・数・床で落ちた画素。"""
    keep = st[1:n, cv2.CC_STAT_AREA] >= floor_px
    remap = np.zeros(n, np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    dropped = int(st[1:n, cv2.CC_STAT_AREA][~keep].sum())
    return remap[lab], int(keep.sum()), dropped


def regions(ink, k, floor_px):
    """L 腕: 線を半径 k で太らせた補集合の 4連結成分を種に、`¬線` へ k 回育てる。"""
    inkk = ink if k == 0 else cv2.dilate(
        ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats((~inkk).astype(np.uint8), connectivity=4)
    lab, m, dropped = compact(lab, n, st, floor_px)
    free = ~ink
    for _ in range(k):
        d = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        grow = (lab == 0) & free & (d > 0)
        lab[grow] = d[grow]
    return lab, m, dropped


def grid(H, W, s, floor_px):
    yy, xx = np.indices((H, W), dtype=np.int32)
    lab = (yy // s) * ((W + s - 1) // s) + xx // s + 1
    del yy, xx
    n = int(lab.max()) + 1
    st = np.zeros((n, 5), np.int64)
    st[:, cv2.CC_STAT_AREA] = np.bincount(lab.ravel(), minlength=n)
    return compact(lab, n, st, floor_px)[:2]


def score(P, T, keep, thr=THR):
    """IoU 貪欲 1対1。`keep` は数える提案の番号(評価域)。**提案の番号は P のラベルそのもの**(T-M49)。"""
    aP = np.bincount(P.ravel())
    aT = np.bincount(T.ravel())
    both = (P > 0) & (T > 0)
    key = P[both].astype(np.int64) * (len(aT)) + T[both]
    u, c = np.unique(key, return_counts=True)
    pid, sid = u // len(aT), u % len(aT)
    sel = np.isin(pid, keep)
    pid, sid, c = pid[sel], sid[sel], c[sel]
    iou = c / (aP[pid] + aT[sid] - c)
    order = np.lexsort((sid, pid, -iou))
    nT = int((aT[1:] > 0).sum())
    out = {}
    for t in thr:
        used_s, used_p, hits, pids = set(), set(), [], []
        for j in order:
            if iou[j] < t:
                break
            if sid[j] in used_s or pid[j] in used_p:
                continue
            used_s.add(int(sid[j]))
            used_p.add(int(pid[j]))
            hits.append(int(sid[j]))
            pids.append(int(pid[j]))
        h, npr = len(hits), len(keep)
        pr = h / npr if npr else float('nan')
        rc = h / nT if nT else float('nan')
        f1 = 0.0 if (h == 0 or not npr or not nT) else 2 * pr * rc / (pr + rc)
        out[t] = dict(P=pr, R=rc, F1=f1, hit=h, n_prop=npr, n_true=nT, hits=set(hits), pids=set(pids))
    return out


def in_domain(P, U):
    aP = np.bincount(P.ravel())
    aU = np.bincount(P[U].ravel(), minlength=len(aP))
    ids = np.arange(len(aP))
    ok_ = (ids > 0) & (aP > 0) & (aU >= 0.5 * aP)
    return ids[ok_], int(((ids > 0) & (aP > 0) & ~ok_).sum())


def unit_test():
    """IG5: 2つの正解に半分ずつ重なる1つの提案は、閾値 0.3 で当たり 1 つだけ。"""
    T = np.zeros((4, 8), np.int32)
    T[:, :4], T[:, 4:] = 1, 2
    P = np.zeros_like(T)
    P[:, 2:6] = 1
    r = score(P, T, np.array([1]), thr=(0.3,))
    return r[0.3]['hit'] == 1


def build_truth(pg, kinds):
    """追補200 §B の規則でマスクの面を作り、`T_fine`(F ∧ ¬線 の 4連結・床以上)に分ける。"""
    H, W, floor_px = pg['H'], pg['W'], pg['floor_px']
    mat, _, _ = m119.classify(pg['rows'], pg['mm'], H * W, kinds, pg['pp'])
    info = {int(r['layer_id']): row for r, row in mat}
    vis = {int(r['layer_id']): m119.vis_side(pg['mdir'] / r['mask_file'], H, W) for r, _ in mat}
    order = sorted(info, key=lambda l: int(info[l]['order_in_parent']))
    Fl = np.zeros((H, W), np.int32)
    Tl = np.zeros((H, W), np.int32)
    face_of_fine, faces = {}, []
    above = np.zeros((H, W), bool)
    for l in reversed(order):
        fm = vis[l] & ~above
        above |= vis[l]
        lb, st, big, _ = m119.components(fm, 8, floor_px)
        for i in big:
            F = lb == i
            fid = len(faces) + 1
            Fl[F] = fid
            faces.append(dict(fid=fid, layer=l, name=info[l]['name'],
                              flag=FLAGGED.get((pg['key'], l), '')))
            lb2, _, big2, _ = m119.components(F & ~pg['ink'], 4, floor_px)
            for i2 in big2:
                tid = int(Tl.max()) + 1
                Tl[lb2 == i2] = tid
                face_of_fine[tid] = fid
            del F, lb2
        del lb
    return Fl, Tl, faces, face_of_fine


def group_bound(P, Fl, nF):
    """神託のまとめ: 各提案を面積の 50% 以上を含む面に割り当て、和と面の IoU。"""
    aP = np.bincount(P.ravel())
    aF = np.bincount(Fl.ravel(), minlength=nF + 1)
    both = (P > 0) & (Fl > 0)
    key = P[both].astype(np.int64) * (nF + 1) + Fl[both]
    u, c = np.unique(key, return_counts=True)
    pid, fid = u // (nF + 1), u % (nF + 1)
    sel = c >= 0.5 * aP[pid]
    uni = np.bincount(fid[sel], weights=aP[pid[sel]], minlength=nF + 1)
    inter = np.bincount(fid[sel], weights=c[sel], minlength=nF + 1)
    iou = np.where(aF > 0, inter / np.maximum(aF + uni - inter, 1), 0)
    return iou[1:]


def run_page(pg, kinds, rows_out):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, Tl, faces, face_of_fine = build_truth(pg, kinds)
    nF, nT = len(faces), int(Tl.max())
    U = Fl > 0
    fails = 0
    exp = sum(int(r['line_n_split']) for r in csv.DictReader(open(M119_CSV, encoding='utf-8'))
              if r['page'] == key)
    print(f'IG1 T_fine {nT} / 追補201 の line_n_split の和 {exp}(マスクの面 {nF})  {ok(nT == exp)}')
    fails += nT != exp
    r0 = score(Tl, Tl, np.arange(1, nT + 1))
    g2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in THR)
    print(f'IG2 N0 神託 F1 ' + ' / '.join(f'{r0[t]["F1"]:.3f}' for t in THR) + f'  {ok(g2)}')
    fails += not g2

    ink = pg['ink']
    arms = {}
    for k in KS:
        lab, m, dropped = regions(ink, k, floor_px)
        if k == 0:
            g3 = int((lab > 0).sum()) + dropped + int(ink.sum()) == H * W
            print(f'IG3 L0 恒等(提案 + 床で落ちた + 線 = 画布)  {ok(g3)}')
            fails += not g3
        arms[f'L{k}'] = (lab, m)
    # IG4: ラベル図なので画素は1つの提案にしか属さない —— 図の作り方から自明だが、育てた画素が線に入っていないかを当てる
    g4 = all(int(((lab > 0) & ink).sum()) == 0 for lab, _ in arms.values())
    print(f'IG4 L 腕の提案は互いに素、かつ線の上に無い  {ok(g4)}')
    fails += not g4
    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return fails

    labN1, mN1, _ = regions(np.ascontiguousarray(ink[:, ::-1]), MAIN_K, floor_px)
    arms['N1'] = (labN1, mN1)
    s = int(round(np.sqrt(np.median(np.bincount(Tl.ravel())[1:]))))
    labN2, mN2 = grid(H, W, s, floor_px)
    arms['N2'] = (labN2, mN2)
    print(f'  N2 格子の一辺 s = {s} px(T_fine の面積の中央値の平方根。**正解の尺度を渡している**)')

    print(f'\n  {"腕":<5}{"提案":>8}{"域外":>8}{"域内":>7} | ' + ' | '.join(f'@{t} P/R/F1(当たり)' for t in THR))
    res = {}
    for nm, (lab, m) in arms.items():
        keep, outside = in_domain(lab, U)
        r = score(lab, Tl, keep)
        res[nm] = r
        print(f'  {nm:<5}{m:>8}{outside:>8}{len(keep):>7} | ' + ' | '.join(
            f'{r[t]["P"]:.3f}/{r[t]["R"]:.3f}/{r[t]["F1"]:.3f}({r[t]["hit"]})' for t in THR))
        rows_out.append(dict(page=key, arm=nm, n_prop=m, outside=outside, in_domain=len(keep), n_true=nT,
                             **{f'{q}@{t}': r[t][q] for t in THR for q in ('P', 'R', 'F1', 'hit')}))

    lab4 = arms[f'L{MAIN_K}'][0]
    gb = group_bound(lab4, Fl, nF)
    print(f'\n  まとめの上界(L{MAIN_K}、神託のまとめ): 面 {nF} のうち IoU ≥0.5 {int((gb >= 0.5).sum())} / ≥0.8 {int((gb >= 0.8).sum())}'
          f'  中央値 {np.median(gb):.3f}')
    keep, _ = in_domain(lab4, U)
    rf = score(lab4, Fl, keep)
    print(f'  粗の段 T_face に対する L{MAIN_K} の F1@0.5 {rf[MAIN]["F1"]:.3f}({rf[MAIN]["hit"]}/{nF})')
    hits = res[f'L{MAIN_K}'][MAIN]['hits']
    print(f'\n  {"面":>4} {"層":<14}{"印":<5}{"細の数":>6}{"L4 当たり":>9}{"まとめIoU":>10}')
    for f in faces:
        fine = [t for t, fid in face_of_fine.items() if fid == f['fid']]
        h = sum(1 for t in fine if t in hits)
        print(f'  {f["fid"]:>4} L{f["layer"]:<4}{f["name"]:<9}{f["flag"]:<5}{len(fine):>6}{h:>9}{gb[f["fid"] - 1]:>10.3f}')
    return 0, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g5 = unit_test()
    print(f'IG5 採点器の単体試験(T-M49): 閾値 0.3 で当たり 1  {ok(g5)}')
    if not g5:
        return
    kinds = m102.load_kinds()
    rows, allres = [], {}
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, g6 = loader(kinds)
        if not g6:
            print('**線の読み込みの門が落ちた**')
            return
        out = run_page(pg, kinds, rows)
        if isinstance(out, int):
            return
        allres[pg['key']] = out[1]
        del pg
    with open(OUT / 'arm_scores.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\n=== 棒(追補204 §E。**主の腕 L4、F1@0.5、頁ごと・最悪の頁で**)===')
    main_arm = f'L{MAIN_K}'
    c1 = {pg: r[main_arm][MAIN]['F1'] > max(r['N1'][MAIN]['F1'], r['N2'][MAIN]['F1']) for pg, r in allres.items()}
    c2 = {pg: r[main_arm][MAIN]['F1'] >= C2_BAR for pg, r in allres.items()}
    for pg, r in allres.items():
        print(f'  {pg:<9} L4 {r[main_arm][MAIN]["F1"]:.3f} / N1 {r["N1"][MAIN]["F1"]:.3f} / N2 {r["N2"][MAIN]["F1"]:.3f}'
              f'  C1 {"はい" if c1[pg] else "いいえ"} / C2(≥{C2_BAR}) {"はい" if c2[pg] else "いいえ"}')
    worst = min(r[main_arm][MAIN]['F1'] for r in allres.values())
    print(f'  **C1 {"通る" if all(c1.values()) else "落ちる"} / C2 {"通る" if all(c2.values()) else "落ちる"}**'
          f'(最悪の頁の L4 = {worst:.3f})')
    print(f'\n表: {OUT / "arm_scores.csv"}')
    print('**第2の棒(作者の判定)はこの後。C1・C2 は動かさない(規律10)。**')


if __name__ == '__main__':
    main()
