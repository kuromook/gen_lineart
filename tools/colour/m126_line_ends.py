"""**線の端で分ける・つなぐ**(B = 向かい合う端点を橋で塞ぐ / BJ = B の上で自由端の境目をつなぐ)。

**事前登録は 追補212**(走行前に commit `476fa16`)。正解 `T_face`、採点 `m122.score`、評価域 `m122.in_domain`。
帰無値 NB 無作為の橋 / NJ 無作為のつなぎ / N1 反転した線 / N0 神託。**門 J1〜J5 が1本でも落ちたら表を読まない**(J6 は境目の正解だけを止める)。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m124_group_faces as m124                                      # noqa: E402

OUT = Path('results/m126_line_ends')
MAIN = m122.MAIN
K = m122.MAIN_K
TRACE = 10
P_MAIN = dict(D=40.0, ang=30.0, lat=4.0, R=8.0)
SENS = [('D', 20.0), ('D', 80.0), ('ang', 15.0), ('ang', 45.0), ('lat', 2.0), ('lat', 8.0), ('R', 4.0), ('R', 16.0)]
D_MAX = 80.0
N_RAND = 20
L4_FACE = m124.L4_FACE
# 追補209 の作者の判定(m125 の出力順)
AUTHOR_H2 = {'page0002': [False, True, False], 'gakusai3': [False, False, True, True]}
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def ok(b):
    return 'PASS' if b else '**FAIL**'


# ---------------------------------------------------------------- §A 端点
def endpoints(ink):
    sk = skeletonize(ink)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    ys, xs = np.nonzero(sk & (nb == 1))
    H, W = ink.shape
    pts, dirs = [], []
    drop = {'行き止まり(短い線)': 0, '分岐に届く(棘)': 0}
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        path = [(y0, x0)]
        seen = {(y0, x0)}
        why = None
        while len(path) <= TRACE:
            cy, cx = path[-1]
            if len(path) > 1 and nb[cy, cx] >= 3:
                why = '分岐に届く(棘)'
                break
            prev = path[-2] if len(path) > 1 else None
            cand = []
            for dy, dx in N8:
                y, x = cy + dy, cx + dx
                if 0 <= y < H and 0 <= x < W and sk[y, x] and (y, x) not in seen:
                    if prev is not None and abs(y - prev[0]) <= 1 and abs(x - prev[1]) <= 1:
                        continue
                    cand.append((y, x))
            if len(cand) != 1:
                why = '行き止まり(短い線)' if not cand else '分岐に届く(棘)'
                break
            path.append(cand[0])
            seen.add(cand[0])
        if len(path) <= TRACE:
            drop[why] += 1
            continue
        v = np.array([y0 - path[TRACE][0], x0 - path[TRACE][1]], np.float64)
        pts.append((y0, x0))
        dirs.append(v / np.linalg.norm(v))
    return np.array(pts, np.float64).reshape(-1, 2), np.array(dirs).reshape(-1, 2), len(ys), drop, sk


def facing(a, da, b, db, p):
    v = b - a
    d = float(np.hypot(*v))
    if d == 0 or d > p['D'] or v @ da <= 0 or -v @ db <= 0:
        return False
    if np.degrees(np.arccos(np.clip(da @ -db, -1, 1))) > p['ang']:
        return False
    lat = max(abs(da[0] * v[1] - da[1] * v[0]), abs(db[0] * v[1] - db[1] * v[0]))
    return lat <= p['lat']


def bridges(pts, dirs, pairs, p):
    """pairs: 距離 ≤ D_MAX の端点の組 (i, j, d)、d 昇順。"""
    used, out = set(), []
    for i, j, d in pairs:
        if d > p['D'] or i in used or j in used:
            continue
        if facing(pts[i], dirs[i], pts[j], dirs[j], p):
            used |= {i, j}
            out.append((i, j))
    return out


def random_bridges(pairs, D, n, seed):
    cand = [(i, j) for i, j, d in pairs if d <= D]
    used, out = set(), []
    for t in np.random.default_rng(seed).permutation(len(cand)):
        if len(out) >= n:
            break
        i, j = cand[t]
        if i in used or j in used:
            continue
        used |= {i, j}
        out.append((i, j))
    return out


def draw(ink, pts, br):
    im = ink.astype(np.uint8)
    for i, j in br:
        a, b = pts[i].astype(int), pts[j].astype(int)
        cv2.line(im, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, 1, cv2.LINE_8)
    return im.astype(bool)


def all_pairs(pts):
    if len(pts) < 2:
        return []
    tree = cKDTree(pts)
    pr = np.array(sorted(tree.query_pairs(D_MAX)), np.int64).reshape(-1, 2)
    d = np.hypot(*(pts[pr[:, 0]] - pts[pr[:, 1]]).T)
    o = np.argsort(d, kind='stable')
    return [(int(pr[k, 0]), int(pr[k, 1]), float(d[k])) for k in o]


# ---------------------------------------------------------------- §B BJ
def seams(lab):
    """墨を挟まず接する組ごとの境目の画素。返り値 {(p, q): array[(y, x)]}。"""
    ys, xs, ps, qs = [], [], [], []
    for a, b, oy, ox in ((lab[:, :-1], lab[:, 1:], 0, 0), (lab[:-1, :], lab[1:, :], 0, 0)):
        m = (a > 0) & (b > 0) & (a != b)
        yy, xx = np.nonzero(m)
        ys.append(yy); xs.append(xx)
        ps.append(np.minimum(a[m], b[m])); qs.append(np.maximum(a[m], b[m]))
    ys, xs, ps, qs = (np.concatenate(z) for z in (ys, xs, ps, qs))
    key = ps.astype(np.int64) * (int(lab.max()) + 1) + qs
    o = np.argsort(key, kind='stable')
    key, ys, xs = key[o], ys[o], xs[o]
    u, st = np.unique(key, return_index=True)
    out = {}
    for t, k in enumerate(u.tolist()):
        e = st[t + 1] if t + 1 < len(st) else len(key)
        out[(k // (int(lab.max()) + 1), k % (int(lab.max()) + 1))] = np.stack([ys[st[t]:e], xs[st[t]:e]], 1)
    return out


def classify_seams(sm, pts, dirs, p):
    tree = cKDTree(pts) if len(pts) else None
    kind = {}
    for pq, s in sm.items():
        E = set()
        if tree is not None:
            for lst in tree.query_ball_point(s.astype(np.float64), p['R']):
                E.update(lst)
        E = sorted(E)
        if not E:
            kind[pq] = 3
            continue
        face = any(facing(pts[i], dirs[i], pts[j], dirs[j], p) or facing(pts[j], dirs[j], pts[i], dirs[i], p)
                   for x, i in enumerate(E) for j in E[x + 1:])
        kind[pq] = 1 if face else 2
    return kind


def join(lab, chosen):
    n = int(lab.max()) + 1
    uf = m124.UF(n)
    for p, q in chosen:
        uf.union(p, q)
    return m124.relabel(lab, uf, n)


def majority(lab, Fl, nF):
    n = int(lab.max()) + 1
    c = np.bincount(lab.ravel().astype(np.int64) * (nF + 1) + Fl.ravel(), minlength=n * (nF + 1)).reshape(n, nF + 1)
    a = c.sum(1)
    j = c.argmax(1)
    return np.where(c[np.arange(n), j] >= 0.5 * np.maximum(a, 1), j, -1)


def accuracy(pairs, chosen, maj):
    ch = set(chosen)
    dec = [(pq in ch, maj[pq[0]] == maj[pq[1]]) for pq in pairs if maj[pq[0]] >= 0 and maj[pq[1]] >= 0]
    if not dec:
        return float('nan'), 0
    return sum(a == b for a, b in dec) / len(dec), len(dec)


# ---------------------------------------------------------------- 採点
def evaluate(lab, Fl, U, HW):
    r, keep, _ = m124.face_score(lab, Fl, U)
    aP = np.bincount(lab.ravel())
    inside = np.zeros(len(aP), bool)
    inside[keep] = True
    leak = float(aP[1:][~inside[1:]].sum()) / HW
    return r[MAIN]['F1'], r[MAIN]['hit'], len(keep), leak


def unit_test():
    ink = np.zeros((80, 160), bool)
    res = {}
    for name, segs in (('一直線 20px の切れ目', [((40, 10), (40, 70)), ((40, 90), (40, 150))]),
                       ('直交', [((40, 10), (40, 70)), ((5, 90), (75, 90))]),
                       ('横に 10px ずれ', [((35, 10), (35, 70)), ((45, 90), (45, 150))])):
        im = np.zeros_like(ink, np.uint8)
        for (y0, x0), (y1, x1) in segs:
            cv2.line(im, (x0, y0), (x1, y1), 1, 3)
        pts, dirs, _, _, _ = endpoints(im.astype(bool))
        res[name] = len(bridges(pts, dirs, all_pairs(pts), P_MAIN))
    want = {'一直線 20px の切れ目': 1, '直交': 0, '横に 10px ずれ': 0}
    return res == want, res


# ---------------------------------------------------------------- 頁
def b_and_bj(ink, floor_px, pts, dirs, pairs, p):
    br = bridges(pts, dirs, pairs, p)
    inkB = draw(ink, pts, br)
    labB, _, _ = m122.regions(inkB, K, floor_px)
    sm = seams(labB)
    kind = classify_seams(sm, pts, dirs, p)
    chosen = [pq for pq, k in kind.items() if k == 2]
    labJ, _ = join(labB, chosen)
    return br, inkB, labB, sm, kind, chosen, labJ


def width_stats(ink, sk, Fl):
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    edge = np.zeros(Fl.shape, bool)
    edge[:, :-1] |= Fl[:, :-1] != Fl[:, 1:]
    edge[:, 1:] |= Fl[:, :-1] != Fl[:, 1:]
    edge[:-1, :] |= Fl[:-1, :] != Fl[1:, :]
    edge[1:, :] |= Fl[:-1, :] != Fl[1:, :]
    dE = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    s = sk & ink
    w = 2 * dt[s]
    outl = dE[s] <= w / 2 + 1
    out = {}
    for nm, sel in (('輪郭', outl), ('描き込み', ~outl)):
        v = w[sel]
        out[nm] = (int(v.size), *(np.percentile(v, [25, 50, 75]).tolist() if v.size else [float('nan')] * 3))
    return out


def run_page(pg, kinds, log):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    HW = H * W
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, _, faces, _ = m122.build_truth(pg, kinds)
    nF = len(faces)
    U = Fl > 0
    ink = pg['ink']
    fails = 0

    L4, _, _ = m122.regions(ink, K, floor_px)
    f0, h0, n0, leak0 = evaluate(L4, Fl, U, HW)
    j1 = round(f0, 3) == L4_FACE[key]
    print(f'J1 原図の L4 の T_face F1@0.5 {f0:.3f}(追補205 {L4_FACE[key]})  {ok(j1)}')
    fails += not j1
    r0 = m122.score(Fl, Fl, np.arange(1, nF + 1))
    j2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in m122.THR)
    print(f'J2 神託 F1 = 1  {ok(j2)}')
    fails += not j2

    # J6 境目の正解 ⇄ 作者の7組
    maj0 = majority(L4, Fl, nF)
    ga = [pq for pq, c in m124.contacts(L4).items() if c >= m124.MAIN_M]
    truth = [bool(maj0[p] == maj0[q]) if maj0[p] >= 0 and maj0[q] >= 0 else None for p, q in ga]
    j6 = sum(t == a for t, a in zip(truth, AUTHOR_H2[key]))
    print(f'J6 境目の正解 ⇄ 作者(追補209): ' + ' '.join(
        f'{i + 1}:{"つなぐ" if t else ("分ける" if t is not None else "不定")}/{"○" if a else "✗"}'
        for i, (t, a) in enumerate(zip(truth, AUTHOR_H2[key]))) + f'  一致 {j6}/{len(ga)}')

    pts, dirs, n_end, drop, sk = endpoints(ink)
    print(f'端点: 骨格の端 {n_end} → 使う {len(pts)} / 捨てた ' + ' / '.join(f'{k} {v}' for k, v in drop.items()))
    pairs = all_pairs(pts)
    print(f'距離 ≤ {D_MAX:.0f} px の端点の組 {len(pairs)}')

    p = P_MAIN
    br, inkB, labB, sm, kind, chosen, labJ = b_and_bj(ink, floor_px, pts, dirs, pairs, p)
    j4 = bool((~ink | inkB).all())
    print(f'J4 B の線 ⊇ 元の線(橋 {len(br)} 本、足した画素 {int(inkB.sum() - ink.sum())})  {ok(j4)}')
    fails += not j4
    fB, hB, nB, leakB = evaluate(labB, Fl, U, HW)
    fJ, hJ, nJ, leakJ = evaluate(labJ, Fl, U, HW)
    cnt = {k: sum(v == k for v in kind.values()) for k in (1, 2, 3)}
    print(f'BJ の境目: 接する組 {len(sm)} = ①向かい合う {cnt[1]} + ②自由端(つなぐ){cnt[2]} + ③端点なし {cnt[3]}')
    majB = majority(labB, Fl, nF)
    accJ, nacc = accuracy(list(sm), chosen, majB)

    # 帰無値
    nb_f, nb_l, j5 = [], [], True
    for s in range(N_RAND):
        rb = random_bridges(pairs, p['D'], len(br), s)
        j5 &= len(rb) == len(br)
        lab, _, _ = m122.regions(draw(ink, pts, rb), K, floor_px)
        f, _, _, lk = evaluate(lab, Fl, U, HW)
        nb_f.append(f); nb_l.append(lk)
        del lab
    nj_f, nj_a = [], []
    plist = list(sm)
    for s in range(N_RAND):
        idx = np.random.default_rng(s).permutation(len(plist))[:len(chosen)]
        ch = [plist[i] for i in idx]
        j5 &= len(ch) == len(chosen)
        lab, _ = join(labB, ch)
        nj_f.append(evaluate(lab, Fl, U, HW)[0])
        nj_a.append(accuracy(plist, ch, majB)[0])
        del lab
    print(f'J5 帰無値の数 = 腕(橋 {len(br)}・つなぎ {len(chosen)}、20回すべて)  {ok(j5)}')
    fails += not j5

    inkM = np.ascontiguousarray(ink[:, ::-1])
    ptsM, dirsM, _, _, _ = endpoints(inkM)
    _, _, labBM, _, _, chM, labJM = b_and_bj(inkM, floor_px, ptsM, dirsM, all_pairs(ptsM), p)
    fBM = evaluate(labBM, Fl, U, HW)[0]
    fJM = evaluate(labJM, Fl, U, HW)[0]
    del labBM, labJM, inkM

    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return None

    print(f'\n  {"腕":<14}{"F1@0.5":>8}{"当たり":>7}{"域内":>6}{"漏れ":>8}')
    for nm, (f, h, n, lk) in (('原図の L4', (f0, h0, n0, leak0)), ('B', (fB, hB, nB, leakB)), ('BJ', (fJ, hJ, nJ, leakJ))):
        print(f'  {nm:<14}{f:>8.3f}{h:>7}{n:>6}{lk:>8.4f}')
    print(f'  NB 無作為の橋  F1 最大 {max(nb_f):.3f} / 中央値 {np.median(nb_f):.3f} / 最小 {min(nb_f):.3f}'
          f'  漏れ 中央値 {np.median(nb_l):.4f}')
    print(f'  NJ 無作為のつなぎ F1 最大 {max(nj_f):.3f} / 中央値 {np.median(nj_f):.3f}'
          f'  正答率 最大 {np.nanmax(nj_a):.3f} / 中央値 {np.nanmedian(nj_a):.3f}')
    print(f'  N1 反転した線  B {fBM:.3f} / BJ {fJM:.3f}')
    print(f'  BJ の境目の正答率 {accJ:.3f}(不定を除く組 {nacc} / 接する組 {len(sm)})')

    print('\n  感度(棒は読まない): 1つずつ動かす')
    for name, val in SENS:
        q = dict(P_MAIN, **{name: val})
        b2, _, lb2, sm2, kd2, ch2, lj2 = b_and_bj(ink, floor_px, pts, dirs, pairs, q)
        eb, ej = evaluate(lb2, Fl, U, HW), evaluate(lj2, Fl, U, HW)
        print(f'  {name}={val:<5g} 橋 {len(b2):>4}  B F1 {eb[0]:.3f} 漏れ {eb[3]:.4f} | 接する組 {len(sm2):>4} つなぐ {len(ch2):>4}'
              f'  BJ F1 {ej[0]:.3f}  正答率 {accuracy(list(sm2), ch2, majority(lb2, Fl, nF))[0]:.3f}')
        del lb2, lj2

    ws = width_stats(ink, sk, Fl)
    print('\n  太さ(記述、§F): 骨格画素 / 四分位 25・50・75(px)')
    for nm, (n, a, b, c) in ws.items():
        print(f'  {nm:<6}{n:>9}  {a:.2f} / {b:.2f} / {c:.2f}')

    json.dump(dict(bridges=[[pts[i].tolist(), pts[j].tolist()] for i, j in br],
                   joins=[[int(a), int(b)] for a, b in chosen],
                   join_truth=[bool(majB[a] == majB[b]) if majB[a] >= 0 and majB[b] >= 0 else None for a, b in chosen]),
              open(OUT / f'{key}_arms.json', 'w'), ensure_ascii=False)
    np.save(OUT / f'{key}_labB.npy', labB)
    return dict(f0=f0, leak0=leak0, fB=fB, leakB=leakB, fJ=fJ, nb=max(nb_f), n1B=fBM, n1J=fJM,
                accJ=accJ, nj=float(np.nanmax(nj_a)), j6=(j6, len(ga)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good, res = unit_test()
    print(f'J3 単体試験 {res}  {ok(good)}')
    if not good:
        print('**門が落ちた。走らない**')
        return
    kinds = m102.load_kinds()
    allr = {}
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, g6 = loader(kinds)
        if not g6:
            print('**線の読み込みの門が落ちた**')
            return
        r = run_page(pg, kinds, None)
        if r is None:
            return
        allr[pg['key']] = r
        del pg
    j6 = sum(r['j6'][0] for r in allr.values())
    j6n = sum(r['j6'][1] for r in allr.values())
    j6ok = j6 >= 6
    print(f'\nJ6 境目の正解 ⇄ 作者 {j6}/{j6n}(6 以上)  {ok(j6ok)}' + ('' if j6ok else ' —— D3 の正答率は判定できない'))
    print('\n=== 棒(追補212 §E、最悪の頁で読む)===')
    d1 = {k: r['leakB'] < r['leak0'] and r['fB'] >= L4_FACE[k] for k, r in allr.items()}
    d2 = {k: r['fB'] > max(r['nb'], r['n1B']) for k, r in allr.items()}
    d3a = {k: r['accJ'] > r['nj'] for k, r in allr.items()}
    d3b = {k: r['fJ'] > r['fB'] for k, r in allr.items()}
    for k, r in allr.items():
        print(f'  {k:<9} 漏れ L4 {r["leak0"]:.4f} → B {r["leakB"]:.4f} / F1 L4 {r["f0"]:.3f} → B {r["fB"]:.3f} → BJ {r["fJ"]:.3f}'
              f' / NB 最大 {r["nb"]:.3f} / N1 B {r["n1B"]:.3f} BJ {r["n1J"]:.3f} / 正答率 BJ {r["accJ"]:.3f} NJ 最大 {r["nj"]:.3f}')
        print(f'            D1 {"はい" if d1[k] else "いいえ"} / D2 {"はい" if d2[k] else "いいえ"}'
              f' / D3 正答率 {"はい" if d3a[k] else "いいえ"} F1 {"はい" if d3b[k] else "いいえ"}')
    a = lambda d: '通る' if all(d.values()) else '落ちる'  # noqa: E731
    d3 = ('判定できない(J6)' + f'、F1 の部分は {a(d3b)}') if not j6ok else ('通る' if all(d3a.values()) and all(d3b.values()) else '落ちる')
    print(f'  **D1 {a(d1)} / D2 {a(d2)} / D3 {d3}**')
    print('**H3(作者の判定)はこの後。棒は動かさない(規律10)。**')


if __name__ == '__main__':
    main()
