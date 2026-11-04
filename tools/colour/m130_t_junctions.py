"""**T字路で閉じる(T)**: B の上で、自由な端点をその向きに伸ばし、40 px 以内で墨に当たれば結ぶ。

**事前登録は 追補219**(走行前に commit `336636e`)。帰無値 NT 無作為な向き / N1 反転した線 / N0 神託。
**門 T1〜T5 が1本でも落ちたら表を読まない。**
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m126_line_ends as m126                                        # noqa: E402
import m129_leak_entrances as m129                                   # noqa: E402

OUT = Path('results/m130_t_junctions')
K = m122.MAIN_K
D_MAIN = 40.0
D_SENS = (20.0, 80.0)
FAR = 200.0
STEP = 0.5
N_RAND = 20
B_F1 = m129.B_F1
LEAK_U = m129.LEAK_U


def ok(b):
    return 'PASS' if b else '**FAIL**'


def ray(e, d, ink, D):
    """返り値 (理由, 当たった点 or None, 距離)。理由: 'hit' / '出られない' / '40 px 超' / '当たらない'。"""
    H, W = ink.shape
    exited = False
    t = STEP
    while t <= FAR:
        y, x = int(round(e[0] + t * d[0])), int(round(e[1] + t * d[1]))
        if not (0 <= y < H and 0 <= x < W):
            break
        if not exited:
            if not ink[y, x]:
                exited = True
            elif t > D:
                return '出られない', None, t
        elif ink[y, x]:
            return ('hit', (y, x), t) if t <= D else ('40 px 超', None, t)
        t += STEP
    return ('当たらない' if exited else '出られない'), None, t


def build_b(ink):
    pts, dirs, _, _, _ = m126.endpoints(ink)
    br = m126.bridges(pts, dirs, m126.all_pairs(pts), m126.P_MAIN)
    used = {i for p in br for i in p}
    free = [i for i in range(len(pts)) if i not in used]
    return m126.draw(ink, pts, br), pts, dirs, br, free


def extend(inkB, pts, dirs, free, D):
    segs, why = [], {}
    for i in free:
        r, hit, _ = ray(pts[i], dirs[i], inkB, D)
        why[r] = why.get(r, 0) + 1
        if r == 'hit':
            segs.append((tuple(pts[i].astype(int).tolist()), hit))
    return segs, why


def draw_segs(ink, segs):
    im = ink.astype(np.uint8).copy()
    for (ay, ax), (by, bx) in segs:
        cv2.line(im, (int(ax), int(ay)), (int(bx), int(by)), 1, 1, cv2.LINE_8)
    return im.astype(bool)


def random_ext(inkB, pts, free, n, D, seed):
    rng = np.random.default_rng(seed)
    segs, tries = [], 0
    while len(segs) < n and tries < 400 * max(n, 1):
        tries += 1
        i = free[int(rng.integers(len(free)))]
        a = rng.uniform(0, 2 * np.pi)
        r, hit, _ = ray(pts[i], np.array([np.sin(a), np.cos(a)]), inkB, D)
        if r == 'hit':
            segs.append((tuple(pts[i].astype(int).tolist()), hit))
    return segs


def unit_test():
    res = {}
    cases = (('15 px 手前', (140, 120), [((10, 60), (55, 60)), ((70, 10), (70, 110))]),
             ('反対を向く', (140, 120), [((70, 60), (130, 60)), ((70, 10), (70, 110))]),
             ('60 px 手前', (200, 120), [((10, 60), (70, 60)), ((130, 10), (130, 110))]))
    for name, shape, segs in cases:
        im = np.zeros(shape, np.uint8)
        for (y0, x0), (y1, x1) in segs:
            cv2.line(im, (x0, y0), (x1, y1), 1, 3)
        inkB, pts, dirs, _, free = build_b(im.astype(bool))
        res[name] = len(extend(inkB, pts, dirs, free, D_MAIN)[0])
    want = {'15 px 手前': 1, '反対を向く': 0, '60 px 手前': 0}
    return res == want, res


def run_page(pg, kinds):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, _, faces, _ = m122.build_truth(pg, kinds)
    nF = len(faces)
    U = Fl > 0
    ink = pg['ink']
    fails = 0
    inkB, pts, dirs, br, free = build_b(ink)
    labB, _, _ = m122.regions(inkB, K, floor_px)
    lkB, fB, _, _ = m129.leak_and_f1(labB, Fl, U)
    t1 = round(fB, 3) == B_F1[key] and round(lkB, 3) == LEAK_U[key]
    print(f'T1 B の再現 F1 {fB:.3f}(追補213 {B_F1[key]})/ 漏れ(U) {lkB:.3f}(追補217 {LEAK_U[key]})  {ok(t1)}')
    fails += not t1
    r0 = m122.score(Fl, Fl, np.arange(1, nF + 1))
    t2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in m122.THR)
    print(f'T2 神託 F1 = 1  {ok(t2)}')
    fails += not t2
    del labB

    segs, why = extend(inkB, pts, dirs, free, D_MAIN)
    print(f'端点 {len(pts)} − 橋 {2 * len(br)} = 使える {len(free)}: ' + ' / '.join(f'{k} {v}' for k, v in sorted(why.items())))
    inkT = draw_segs(inkB, segs)
    t4 = bool((~inkB | inkT).all())
    print(f'T4 B+T の線 ⊇ B の線(延長 {len(segs)} 本、足した画素 {int(inkT.sum() - inkB.sum())})  {ok(t4)}')
    fails += not t4
    labT, _, _ = m122.regions(inkT, K, floor_px)
    lkT, fT, _, _ = m129.leak_and_f1(labT, Fl, U)
    np.save(OUT / f'{key}_labT.npy', labT)
    del labT

    nt_f, nt_l, t5 = [], [], True
    for s in range(N_RAND):
        rs = random_ext(inkB, pts, free, len(segs), D_MAIN, s)
        t5 &= len(rs) == len(segs)
        lab, _, _ = m122.regions(draw_segs(inkB, rs), K, floor_px)
        lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
        nt_f.append(f); nt_l.append(lk)
        del lab
    print(f'T5 NT の本数 = T の {len(segs)}(20 回すべて)  {ok(t5)}')
    fails += not t5

    inkM = np.ascontiguousarray(ink[:, ::-1])
    iBM, pM, dM, _, fM = build_b(inkM)
    sM, _ = extend(iBM, pM, dM, fM, D_MAIN)
    labM, _, _ = m122.regions(draw_segs(iBM, sM), K, floor_px)
    _, f1M, _, _ = m129.leak_and_f1(labM, Fl, U)
    del labM, inkM, iBM

    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return None
    print(f'\n  {"腕":<10}{"F1@0.5":>8}{"漏れ(U)":>9}')
    print(f'  {"B":<10}{fB:>8.3f}{lkB:>9.3f}')
    print(f'  {"B+T":<10}{fT:>8.3f}{lkT:>9.3f}')
    print(f'  NT 無作為な向き F1 最大 {max(nt_f):.3f} / 中央値 {np.median(nt_f):.3f} / 最小 {min(nt_f):.3f}  漏れ(U) 中央値 {np.median(nt_l):.3f} / 最小 {min(nt_l):.3f}')
    print(f'  N1 反転した線 B+T F1 {f1M:.3f}')
    print('\n  感度(棒は読まない)')
    for D in D_SENS:
        s2, _ = extend(inkB, pts, dirs, free, D)
        lab, _, _ = m122.regions(draw_segs(inkB, s2), K, floor_px)
        lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
        print(f'  上限 {D:g} px  延長 {len(s2):>4}  F1 {f:.3f}  漏れ(U) {lk:.3f}')
        del lab
    json.dump(dict(segs=[[list(a), list(b)] for a, b in segs]), open(OUT / f'{key}_segs.json', 'w'))
    return dict(fB=fB, lkB=lkB, fT=fT, lkT=lkT, nt=max(nt_f), n1=f1M)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good, res = unit_test()
    print(f'T3 単体試験 {res}  {ok(good)}')
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
    print('\n=== 棒(追補219 §D、最悪の頁で読む)===')
    tb1 = {k: r['lkT'] < r['lkB'] and r['fT'] >= r['fB'] for k, r in allr.items()}
    tb2 = {k: r['fT'] > max(r['nt'], r['n1']) for k, r in allr.items()}
    for k, r in allr.items():
        print(f'  {k:<9} 漏れ(U) B {r["lkB"]:.3f} → B+T {r["lkT"]:.3f} / F1 B {r["fB"]:.3f} → B+T {r["fT"]:.3f}'
              f' / NT 最大 {r["nt"]:.3f} / N1 {r["n1"]:.3f}  TB1 {"はい" if tb1[k] else "いいえ"} / TB2 {"はい" if tb2[k] else "いいえ"}')
    a = lambda d: '通る' if all(d.values()) else '落ちる'  # noqa: E731
    print(f'  **TB1 {a(tb1)} / TB2 {a(tb2)}**')
    print('**H4(作者の判定)はこの後。棒は動かさない(規律10)。**')


if __name__ == '__main__':
    main()
