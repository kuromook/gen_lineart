"""**ズレた線の組を繋ぐ(P)**: B の後の自由な端点で、向きが反対で平行、横に 4〜20 px ずれた組を結ぶ。

**事前登録は 追補223**(走行前に commit `772e316`)。帰無値 NP 無作為な組 / N1 反転した線 / N0 神託。
**門 P1〜P5 が1本でも落ちたら表を読まない。**
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
import m130_t_junctions as m130                                      # noqa: E402

OUT = Path('results/m133_offset_pairs')
K = m122.MAIN_K
P_MAIN = dict(D=40.0, ang=30.0, lat_lo=4.0, lat_hi=20.0, over=20.0)
SENS = (('lat ≤ 10', dict(P_MAIN, lat_hi=10.0)), ('lat ≤ 40', dict(P_MAIN, lat_hi=40.0)), ('重なり 0', dict(P_MAIN, over=0.0)))
N_RAND = 20
ok = m130.ok


def offset(a, da, b, db, p):
    """返り値 (合うか, 理由, lat, 縦)。"""
    v = b - a
    d = float(np.hypot(*v))
    if d == 0 or d > p['D']:
        return False, '距離', 0.0, 0.0
    if np.degrees(np.arccos(np.clip(da @ -db, -1, 1))) > p['ang']:
        return False, '向き', 0.0, 0.0
    u = da - db
    u = u / np.linalg.norm(u)
    lat = abs(u[0] * v[1] - u[1] * v[0])
    along = float(v @ u)
    if lat <= p['lat_lo']:
        return False, '横ずれ ≤ 4(B の領分)', lat, along
    if lat > p['lat_hi']:
        return False, '横ずれが大きい', lat, along
    if along < -p['over']:
        return False, '行き過ぎ', lat, along
    return True, 'ok', lat, along


def pairs_free(pts, free, D):
    f = np.array(free, int)
    out = []
    for a_ in range(len(f)):
        d = np.hypot(*(pts[f[a_ + 1:]] - pts[f[a_]]).T)
        for k in np.nonzero(d <= D)[0]:
            out.append((int(f[a_]), int(f[a_ + 1 + k]), float(d[k])))
    return sorted(out, key=lambda t: t[2])


def join(pts, dirs, cand, p):
    used, out, why, geo = set(), [], {}, []
    for i, j, _ in cand:
        if i in used or j in used:
            continue
        good, r, lat, al = offset(pts[i], dirs[i], pts[j], dirs[j], p)
        why[r] = why.get(r, 0) + 1
        if good:
            used |= {i, j}
            out.append((i, j))
            geo.append((lat, al))
    return out, why, geo


def unit_test():
    cases = (('横 10 ずれ', [((50, 10), (50, 60)), ((60, 75), (60, 130))], 1),
             ('横 10・15 行き過ぎ', [((50, 10), (50, 80)), ((60, 65), (60, 140))], 1),
             ('一直線', [((50, 10), (50, 60)), ((50, 75), (50, 130))], 0),
             ('直交', [((50, 10), (50, 60)), ((60, 75), (120, 75))], 0),
             ('横 30', [((50, 10), (50, 60)), ((80, 75), (80, 130))], 0))
    res = {}
    for name, segs, _ in cases:
        im = np.zeros((140, 160), np.uint8)
        for (y0, x0), (y1, x1) in segs:
            cv2.line(im, (x0, y0), (x1, y1), 1, 3)
        inkB, pts, dirs, _, free = m130.build_b(im.astype(bool))
        res[name] = len(join(pts, dirs, pairs_free(pts, free, P_MAIN['D']), P_MAIN)[0])
    return all(res[n] == w for n, _, w in cases), res


def run_page(pg, kinds):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, _, faces, _ = m122.build_truth(pg, kinds)
    nF = len(faces)
    U = Fl > 0
    ink = pg['ink']
    fails = 0
    inkB, pts, dirs, br, free = m130.build_b(ink)
    labB, _, _ = m122.regions(inkB, K, floor_px)
    lkB, fB, _, _ = m129.leak_and_f1(labB, Fl, U)
    del labB
    p1 = round(fB, 3) == m129.B_F1[key] and round(lkB, 3) == m129.LEAK_U[key]
    print(f'P1 B の再現 F1 {fB:.3f}(追補213 {m129.B_F1[key]})/ 漏れ(U) {lkB:.3f}(追補217 {m129.LEAK_U[key]})  {ok(p1)}')
    fails += not p1
    r0 = m122.score(Fl, Fl, np.arange(1, nF + 1))
    p2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in m122.THR)
    print(f'P2 神託 F1 = 1  {ok(p2)}')
    fails += not p2

    cand = pairs_free(pts, free, P_MAIN['D'])
    jp, why, geo = join(pts, dirs, cand, P_MAIN)
    print(f'端点 {len(pts)} − 橋 {2 * len(br)} = 自由 {len(free)}; 距離 ≤ 40 の組 {len(cand)}: '
          + ' / '.join(f'{k} {v}' for k, v in sorted(why.items())) + '(使用済みで飛ばした組は数に入らない)')
    inkP = m126.draw(inkB, pts, jp)
    p4 = bool((~inkB | inkP).all())
    print(f'P4 B+P の線 ⊇ B の線(P {len(jp)} 本、足した画素 {int(inkP.sum() - inkB.sum())})  {ok(p4)}')
    fails += not p4
    if geo:
        g = np.array(geo)
        print(f'  横ずれ 四分位 {np.percentile(g[:, 0], [25, 50, 75]).round(1).tolist()} px / '
              f'縦 四分位 {np.percentile(g[:, 1], [25, 50, 75]).round(1).tolist()} px(負 = 行き過ぎ、{int((g[:, 1] < 0).sum())} 本)')
    labP, _, _ = m122.regions(inkP, K, floor_px)
    lkP, fP, _, _ = m129.leak_and_f1(labP, Fl, U)
    np.save(OUT / f'{key}_labP.npy', labP)
    del labP

    nf, nl, p5 = [], [], True
    for s in range(N_RAND):
        rb = m126.random_bridges(cand, P_MAIN['D'], len(jp), s)
        p5 &= len(rb) == len(jp)
        lab, _, _ = m122.regions(m126.draw(inkB, pts, rb), K, floor_px)
        lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
        nf.append(f); nl.append(lk)
        del lab
    print(f'P5 NP の本数 = P の {len(jp)}(20 回すべて)  {ok(p5)}')
    fails += not p5

    inkM = np.ascontiguousarray(ink[:, ::-1])
    iBM, pM, dM, _, fM = m130.build_b(inkM)
    jM, _, _ = join(pM, dM, pairs_free(pM, fM, P_MAIN['D']), P_MAIN)
    labM, _, _ = m122.regions(m126.draw(iBM, pM, jM), K, floor_px)
    _, f1M, _, _ = m129.leak_and_f1(labM, Fl, U)
    del labM, inkM, iBM

    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return None
    print(f'\n  {"腕":<10}{"F1@0.5":>8}{"漏れ(U)":>9}')
    print(f'  {"B":<10}{fB:>8.3f}{lkB:>9.3f}')
    print(f'  {"B+P":<10}{fP:>8.3f}{lkP:>9.3f}')
    print(f'  NP 無作為な組 F1 最大 {max(nf):.3f} / 中央値 {np.median(nf):.3f} / 最小 {min(nf):.3f}  漏れ(U) 中央値 {np.median(nl):.3f} / 最小 {min(nl):.3f}')
    print(f'  N1 反転した線 B+P F1 {f1M:.3f}')
    print('\n  記述(棒は読まない)')
    segsT = json.load(open(m130.OUT / f'{key}_segs.json'))['segs']
    inkTP = m130.draw_segs(inkP, [tuple(map(tuple, s)) for s in segsT])
    lab, _, _ = m122.regions(inkTP, K, floor_px)
    lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
    print(f'  B+T+P    F1 {f:.3f}  漏れ(U) {lk:.3f}(T の延長 {len(segsT)} 本を重ねた)')
    del lab, inkTP
    for name, p in SENS:
        j2, _, _ = join(pts, dirs, cand if p['D'] == P_MAIN['D'] else pairs_free(pts, free, p['D']), p)
        lab, _, _ = m122.regions(m126.draw(inkB, pts, j2), K, floor_px)
        lk, f, _, _ = m129.leak_and_f1(lab, Fl, U)
        print(f'  {name:<9} P {len(j2):>4}  F1 {f:.3f}  漏れ(U) {lk:.3f}')
        del lab
    json.dump(dict(segs=[[pts[i].astype(int).tolist(), pts[j].astype(int).tolist()] for i, j in jp],
                   geo=[list(map(float, g)) for g in geo]), open(OUT / f'{key}_segs.json', 'w'))
    return dict(fB=fB, lkB=lkB, fP=fP, lkP=lkP, np=max(nf), n1=f1M)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good, res = unit_test()
    print(f'P3 単体試験 {res}  {ok(good)}')
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
    print('\n=== 棒(追補223 §D、最悪の頁で読む)===')
    pb1 = {k: r['lkP'] < r['lkB'] and r['fP'] >= r['fB'] for k, r in allr.items()}
    pb2 = {k: r['fP'] > max(r['np'], r['n1']) for k, r in allr.items()}
    for k, r in allr.items():
        print(f'  {k:<9} 漏れ(U) B {r["lkB"]:.3f} → B+P {r["lkP"]:.3f} / F1 B {r["fB"]:.3f} → B+P {r["fP"]:.3f}'
              f' / NP 最大 {r["np"]:.3f} / N1 {r["n1"]:.3f}  PB1 {"はい" if pb1[k] else "いいえ"} / PB2 {"はい" if pb2[k] else "いいえ"}')
    a = lambda d: '通る' if all(d.values()) else '落ちる'  # noqa: E731
    print(f'  **PB1 {a(pb1)} / PB2 {a(pb2)}**')
    print('**H5(作者の判定)はこの後。棒は動かさない(規律10)。**')


if __name__ == '__main__':
    main()
