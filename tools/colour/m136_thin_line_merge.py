"""**細い線で隔てられた面を結合する(W)**: 土台 B / B+P / B+P+T のそれぞれに掛ける。

**事前登録は 追補227**(走行前に commit `53fcda7`)。帰無値 NW 無作為な結合 / N1 反転した線 / N0 神託。
太さは追補214(m126.width_stats)と同じ測り方: 骨格の画素で 2×距離変換(L2, mask 5)。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m126_line_ends as m126                                        # noqa: E402
import m129_leak_entrances as m129                                   # noqa: E402
import m130_t_junctions as m130                                      # noqa: E402
import m133_offset_pairs as m133                                     # noqa: E402

OUT = Path('results/m136_thin_line_merge')
K = m122.MAIN_K
TAU, MIN_PX = 4.0, 5
SENS = (('τ=3.0', 3.0, 5), ('τ=5.0', 5.0, 5), ('最小画素 10', 4.0, 10))
N_RAND = 20
ANG = np.linspace(0, 2 * np.pi, 16, endpoint=False)
W1 = {'B': {'page0002': 0.303, 'gakusai3': 0.132}, 'B+P': {'page0002': 0.286, 'gakusai3': 0.143},
      'B+P+T': {'page0002': 0.250, 'gakusai3': 0.145}}
ok = m130.ok


def seam_widths(ink, lab):
    """返り値 {(a, b): array[太さ]}(a < b)。"""
    sk = skeletonize(ink) & ink
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    H, W = ink.shape
    ys, xs = np.nonzero(sk)
    r = np.ceil(dt[ys, xs]) + 2
    yy = np.clip(np.rint(ys[:, None] + r[:, None] * np.sin(ANG)[None]).astype(int), 0, H - 1)
    xx = np.clip(np.rint(xs[:, None] + r[:, None] * np.cos(ANG)[None]).astype(int), 0, W - 1)
    L = lab[yy, xx]
    w = 2 * dt[ys, xs]
    acc = {}
    for i in range(len(ys)):
        u = np.unique(L[i])
        u = u[u > 0]
        if len(u) < 2:
            continue
        for a_ in range(len(u)):
            for b_ in range(a_ + 1, len(u)):
                acc.setdefault((int(u[a_]), int(u[b_])), []).append(w[i])
    return {k: np.array(v) for k, v in acc.items()}


def border_faces(lab):
    b = np.zeros(int(lab.max()) + 1, bool)
    b[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    b[0] = False
    return b


def candidates(sw, bord, min_px):
    return [k for k, v in sw.items() if len(v) >= min_px and not bord[k[0]] and not bord[k[1]]]


def merge(lab, pairs):
    n = int(lab.max()) + 1
    par = np.arange(n)

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in pairs:
        ra, rb = f(a), f(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    root = np.array([f(x) for x in range(n)])
    return root[lab]


def w_pairs(sw, cand, tau):
    return [k for k in cand if np.median(sw[k]) < tau]


def nfaces(lab):
    return len(np.unique(lab[lab > 0]))


def unit_test():
    res = {}
    for name, wsplit, want in (('細い線で2分', 1, (3, 2)), ('太い線で2分', 9, (3, 3))):
        im = np.zeros((220, 220), np.uint8)
        cv2.rectangle(im, (30, 30), (190, 190), 1, 1)       # 細い枠: 内側の面と縁に触れる外側を隔てる
        cv2.line(im, (110, 30), (110, 190), 1, wsplit)
        ink = im.astype(bool)
        lab, _, _ = m122.regions(ink, K, 50)
        sw = seam_widths(ink, lab)
        mp = w_pairs(sw, candidates(sw, border_faces(lab), MIN_PX), TAU)
        res[name] = (nfaces(lab), nfaces(merge(lab, mp)))
        res[name + '?'] = res[name] == want
    return all(v for k, v in res.items() if k.endswith('?')), {k: v for k, v in res.items() if not k.endswith('?')}


def evaluate(lab, Fl, U):
    lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
    keep, _ = m122.in_domain(lab, U)
    r = m122.score(lab, Fl, keep)[m122.MAIN]
    return f, lk, r['hit'], r['n_prop']


def bases(pg):
    ink = pg['ink']
    inkB, pts, dirs, br, free = m130.build_b(ink)
    jp, _, _ = m133.join(pts, dirs, m133.pairs_free(pts, free, m133.P_MAIN['D']), m133.P_MAIN)
    inkP = m126.draw(inkB, pts, jp)
    return inkB, inkP


def run_page(pg, kinds):
    key, floor_px = pg['key'], pg['floor_px']
    print(f'\n======== {key}  床 {floor_px} px')
    Fl, _, faces, _ = m122.build_truth(pg, kinds)
    U = Fl > 0
    fails = 0
    r0 = m122.score(Fl, Fl, np.arange(1, len(faces) + 1))
    w2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in m122.THR)
    print(f'W2 神託 F1 = 1  {ok(w2)}')
    fails += not w2
    inkB, inkP = bases(pg)
    segsT = json.load(open(m130.OUT / f'{key}_segs.json'))['segs']
    inkPT = m130.draw_segs(inkP, [tuple(map(tuple, s)) for s in segsT])
    # 反転した線(N1): 同じ手順を左右反転した線に
    pgM = dict(pg, ink=np.ascontiguousarray(pg['ink'][:, ::-1]))
    iBM, iPM = bases(pgM)
    sM = []  # T は反転した線では求め直す
    pM_pts = m130.build_b(pgM['ink'])
    sM, _ = m130.extend(iBM, pM_pts[1], pM_pts[2], pM_pts[4], m130.D_MAIN)
    iPTM = m130.draw_segs(iPM, sM)
    out = {}
    for name, ink_, inkM_ in (('B', inkB, iBM), ('B+P', inkP, iPM), ('B+P+T', inkPT, iPTM)):
        lab, _, _ = m122.regions(ink_, K, floor_px)
        f0, lk0, h0, n0 = evaluate(lab, Fl, U)
        w1 = round(f0, 3) == W1[name][key]
        print(f'\n--- 土台 {name}\nW1 土台の再現 F1 {f0:.3f}(登録 {W1[name][key]})  {ok(w1)}')
        fails += not w1
        sw = seam_widths(ink_, lab)
        bord = border_faces(lab)
        cand = candidates(sw, bord, MIN_PX)
        mp = w_pairs(sw, cand, TAU)
        labW = merge(lab, mp)
        w4 = nfaces(labW) <= nfaces(lab)
        print(f'  隣接の組 {len(sw)} → 境目 {MIN_PX} 画素以上かつ縁に触れない {len(cand)}(落とした: 画素不足 '
              f'{sum(len(v) < MIN_PX for v in sw.values())} / 縁 {sum(len(v) >= MIN_PX for v in sw.values()) - len(cand)})→ 結合 {len(mp)} 組')
        print(f'W4 面の数 {nfaces(lab)} → {nfaces(labW)}  {ok(w4)}')
        fails += not w4
        fW, lkW, hW, nW = evaluate(labW, Fl, U)
        nf, w5 = [], True
        rng = np.random.default_rng(0)
        for s in range(N_RAND):
            pick = [cand[i] for i in rng.permutation(len(cand))[:len(mp)]]
            w5 &= len(pick) == len(mp)
            nf.append(evaluate(merge(lab, pick), Fl, U)[0])
        print(f'W5 NW の組の数 = {len(mp)}(20 回すべて)  {ok(w5)}')
        fails += not w5
        labM, _, _ = m122.regions(inkM_, K, floor_px)
        swM = seam_widths(inkM_, labM)
        f1M = evaluate(merge(labM, w_pairs(swM, candidates(swM, border_faces(labM), MIN_PX), TAU)), Fl, U)[0]
        del labM
        print(f'  {"":<12}{"F1@0.5":>8}{"漏れ(U)":>9}{"当たり":>6}{"提案":>6}')
        print(f'  {name:<12}{f0:>8.3f}{lk0:>9.3f}{h0:>6}{n0:>6}')
        print(f'  {name + "+W":<12}{fW:>8.3f}{lkW:>9.3f}{hW:>6}{nW:>6}')
        print(f'  NW 無作為な結合 F1 最大 {max(nf):.3f} / 中央値 {np.median(nf):.3f} / 最小 {min(nf):.3f}   N1 反転 {f1M:.3f}')
        for sn, tau, mpx in SENS:
            m2 = w_pairs(sw, candidates(sw, bord, mpx), tau)
            f2, lk2, h2, n2 = evaluate(merge(lab, m2), Fl, U)
            print(f'  感度 {sn:<10} 結合 {len(m2):>4} 組  F1 {f2:.3f}  漏れ(U) {lk2:.3f}  当たり {h2} / 提案 {n2}')
        out[name] = dict(f0=f0, fW=fW, nw=max(nf), n1=f1M)
        del lab, labW
    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return None
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good, res = unit_test()
    print(f'W3 単体試験 {res}  {ok(good)}')
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
        r = run_page(pg, kinds)
        if r is None:
            return
        allr[pg['key']] = r
        del pg
    print('\n=== 棒(追補227 §D、土台ごと、最悪の頁で読む)===')
    for b in ('B', 'B+P', 'B+P+T'):
        wb1 = {k: r[b]['fW'] > r[b]['f0'] for k, r in allr.items()}
        wb2 = {k: r[b]['fW'] > max(r[b]['nw'], r[b]['n1']) for k, r in allr.items()}
        for k, r in allr.items():
            x = r[b]
            print(f'  {b:<6} {k:<9} F1 {x["f0"]:.3f} → +W {x["fW"]:.3f} / NW 最大 {x["nw"]:.3f} / N1 {x["n1"]:.3f}'
                  f'  WB1 {"はい" if wb1[k] else "いいえ"} / WB2 {"はい" if wb2[k] else "いいえ"}')
        a = lambda d: '通る' if all(d.values()) else '落ちる'  # noqa: E731
        print(f'  **{b}: WB1 {a(wb1)} / WB2 {a(wb2)}**')


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def describe():
    """記述(正解を使う。棒ではない): 隣接する面の組を「同じマスクの面 / 違う面」に分け、境目の太さの分布を比べる。
    同じ面 = 2つの提案の多数派のマスクの面が同じ(どちらも 50% 以上が U)。"""
    kinds = m102.load_kinds()
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, floor_px = pg['key'], pg['floor_px']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        inkB, inkP = bases(pg)
        segsT = json.load(open(m130.OUT / f'{key}_segs.json'))['segs']
        for name, ink_ in (('B+P', inkP), ('B+P+T', m130.draw_segs(inkP, [tuple(map(tuple, s)) for s in segsT]))):
            lab, _, _ = m122.regions(ink_, K, floor_px)
            n = int(lab.max()) + 1
            keep, _ = m122.in_domain(lab, U)
            inside = np.zeros(n, bool); inside[keep] = True
            maj = np.zeros(n, int)
            for p in keep:
                t, c = np.unique(Fl[lab == p], return_counts=True)
                c, t = c[t > 0], t[t > 0]
                maj[p] = t[np.argmax(c)]
            sw = seam_widths(ink_, lab)
            bord = border_faces(lab)
            g = {'同じ面': [], '違う面': []}
            nb = {'同じ面': 0, '違う面': 0}
            for (a, b), v in sw.items():
                if len(v) < MIN_PX or not (inside[a] and inside[b]):
                    continue
                k = '同じ面' if maj[a] == maj[b] else '違う面'
                if bord[a] or bord[b]:
                    nb[k] += 1
                    continue
                g[k].append(float(np.median(v)))
            print(f'{key} {name}(評価域の内どうしの隣接、境目 {MIN_PX} 画素以上)')
            for k, v in g.items():
                v = np.array(v)
                q = np.percentile(v, [25, 50, 75]).round(1).tolist() if len(v) else []
                print(f'  {k}: {len(v)} 組(縁で落とした {nb[k]})  太さの中央値の四分位 {q}  < 4.0 は {int((v < 4).sum())}  < 5.0 は {int((v < 5).sum())}')
            s, d = np.array(g['同じ面']), np.array(g['違う面'])
            if len(s) and len(d):
                auc = float(((s[:, None] < d[None]).sum() + 0.5 * (s[:, None] == d[None]).sum()) / (len(s) * len(d)))
                print(f'  同じ面の境目の方が細い確率(AUC) {auc:.2f}(0.5 = 太さで分からない)')
            del lab
        del pg, Fl, U


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'describe':
    describe()
