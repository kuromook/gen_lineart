"""**まとめ方の第1走行**: 線だけで切り分けた細かい面(L4)を、線だけの手がかりでマスクの面(`T_face`)へ束ねる。

**事前登録は 追補207**(走行前に commit `9323fc8`)。
GA = 墨を挟まない接触の長さ ≥ m の組を束ねる(主 m = 9)/ GH = Lk をそのまま / NR = 乱択の束ね(数を GA に合わせる、20 回の最大)/
N1 = 反転した線で GA / N0 = 神託。採点は `m122.score`(T-M49 を直したもの)。**門が1本でも落ちたら表を読まない。**
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402

OUT = Path('results/m124_group_faces')
MS = (1, 5, 9, 17, 33)
MAIN_M = 9
GROW_INTO_INK = 12
N_RAND = 20
THR = m122.THR
MAIN = m122.MAIN
# 追補205 §D(束ねない L4 の T_face F1@0.5)。GG1 と C3 に使う
L4_FACE = {'page0002': 0.258, 'gakusai3': 0.132}


def ok(b):
    return 'PASS' if b else '**FAIL**'


def contacts(lab):
    """4近傍で隣り合う、ラベルの違う画素の組を数える。返り値 {(p, q): 長さ}(p < q)。"""
    out = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        p, q = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        u, c = np.unique(p * (int(lab.max()) + 1) + q, return_counts=True)
        for k, n in zip(u.tolist(), c.tolist()):
            key = (k // (int(lab.max()) + 1), k % (int(lab.max()) + 1))
            out[key] = out.get(key, 0) + n
    return out


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True


def relabel(lab, uf, n):
    roots = np.array([uf.find(i) for i in range(n)], np.int64)
    present = np.unique(roots[np.unique(lab[lab > 0])])
    remap = np.zeros(n, np.int32)
    idx = {r: i + 1 for i, r in enumerate(present.tolist())}
    for i in range(1, n):
        remap[i] = idx.get(int(roots[i]), 0)
    return remap[lab], len(present)


def group_by_contact(lab, m):
    n = int(lab.max()) + 1
    uf = UF(n)
    if m is not None:
        for (p, q), c in contacts(lab).items():
            if c >= m:
                uf.union(p, q)
    return relabel(lab, uf, n)


def full_adjacency(lab, ink):
    """墨を挟む接触も含めた隣接: ラベルを墨の中へ最大 `GROW_INTO_INK` 回育ててから接触を数える。"""
    g = lab.copy()
    for _ in range(GROW_INTO_INK):
        d = cv2.dilate(g.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        grow = (g == 0) & ink & (d > 0)
        if not grow.any():
            break
        g[grow] = d[grow]
    return sorted(contacts(g))


def random_group(lab, pairs, target, seed):
    n = int(lab.max()) + 1
    uf = UF(n)
    groups = len(np.unique(lab[lab > 0]))
    order = np.random.default_rng(seed).permutation(len(pairs))
    for j in order:
        if groups <= target:
            break
        if uf.union(*pairs[j]):
            groups -= 1
    return relabel(lab, uf, n)


def face_score(lab, Fl, U):
    keep, outside = m122.in_domain(lab, U)
    return m122.score(lab, Fl, keep), keep, outside


def run_page(pg, kinds, rows):
    key, H, W, floor_px = pg['key'], pg['H'], pg['W'], pg['floor_px']
    print(f'\n======== {key}  {W}x{H}  床 {floor_px} px')
    Fl, Tl, faces, _ = m122.build_truth(pg, kinds)
    U = Fl > 0
    nF = len(faces)
    ink = pg['ink']
    fails = 0
    L4, _, _ = m122.regions(ink, m122.MAIN_K, floor_px)

    r0 = m122.score(Fl, Fl, np.arange(1, nF + 1))
    g3 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in THR)
    print(f'GG3 N0 神託 F1 ' + ' / '.join(f'{r0[t]["F1"]:.3f}' for t in THR) + f'  {ok(g3)}')
    fails += not g3
    lab_inf, _ = group_by_contact(L4, None)
    rinf, _, _ = face_score(lab_inf, Fl, U)
    g1 = round(rinf[MAIN]['F1'], 3) == L4_FACE[key]
    print(f'GG1 束ねない GA = L4 の T_face F1@0.5 {rinf[MAIN]["F1"]:.3f}(追補205 {L4_FACE[key]})  {ok(g1)}')
    fails += not g1

    res = {}
    print(f'\n  {"腕":<8}{"まとまり":>8}{"域外":>6}{"域内":>6} | ' + ' | '.join(f'@{t} P/R/F1(当たり)' for t in THR))

    def show(nm, lab, n):
        r, keep, outside = face_score(lab, Fl, U)
        res[nm] = r
        print(f'  {nm:<8}{n:>8}{outside:>6}{len(keep):>6} | ' + ' | '.join(
            f'{r[t]["P"]:.3f}/{r[t]["R"]:.3f}/{r[t]["F1"]:.3f}({r[t]["hit"]})' for t in THR))
        rows.append(dict(page=key, arm=nm, groups=n, outside=outside, in_domain=len(keep), n_true=nF,
                         **{f'{q}@{t}': r[t][q] for t in THR for q in ('P', 'R', 'F1', 'hit')}))
        return r

    ga = {}
    for m in MS:
        lab, n = group_by_contact(L4, m)
        g4 = int((lab > 0).sum()) == int((L4 > 0).sum())
        if not g4:
            print(f'GG4 GA m={m} で画素が動いた  **FAIL**')
            fails += 1
        ga[m] = (lab, n)
        show(f'GA m={m}', lab, n)
    print(f'GG4 束ねても画素は動かない(GA の全 m)  {ok(fails == 0)}')

    for k in m122.KS:
        lab, n, _ = m122.regions(ink, k, floor_px)
        r = show(f'GH L{k}', lab, n)
        if k == m122.MAIN_K:
            g2 = abs(r[MAIN]['F1'] - rinf[MAIN]['F1']) < 1e-12
            print(f'GG2 GH L4 = GG1  {ok(g2)}')
            fails += not g2
        del lab

    target = ga[MAIN_M][1]
    pairs = full_adjacency(L4, ink)
    best, counts_ok = None, True
    f1s = []
    for seed in range(N_RAND):
        lab, n = random_group(L4, pairs, target, seed)
        counts_ok &= n == target
        r, _, _ = face_score(lab, Fl, U)
        f1s.append(r[MAIN]['F1'])
        if best is None or r[MAIN]['F1'] > best[0][MAIN]['F1']:
            best = (r, lab, n)
    print(f'GG5 NR のまとまりの数 = GA の {target}(20 回すべて)  {ok(counts_ok)}')
    fails += not counts_ok
    print(f'  NR の F1@0.5: 最大 {max(f1s):.3f} / 中央値 {np.median(f1s):.3f} / 最小 {min(f1s):.3f}(隣接の組 {len(pairs)})')
    show('NR 最大', best[1], best[2])

    ink1 = np.ascontiguousarray(ink[:, ::-1])
    L4m, _, _ = m122.regions(ink1, m122.MAIN_K, floor_px)
    lab, n = group_by_contact(L4m, MAIN_M)
    show('N1 GA', lab, n)
    if fails:
        print(f'\n**門が {fails} 本落ちた。表を読まない**')
        return None
    lab, _ = ga[MAIN_M]
    keep, outside = m122.in_domain(lab, U)
    aP = np.bincount(lab.ravel())
    outs = [i for i in range(1, len(aP)) if aP[i] > 0 and i not in set(keep.tolist())]
    print(f'\n  漏れ(GA m={MAIN_M}): 域外のまとまり {len(outs)} / 画素 {int(sum(aP[i] for i in outs))}'
          f'(L4 の域外 画素 {int(sum(a for i, a in enumerate(np.bincount(L4.ravel())) if i > 0 and i not in set(m122.in_domain(L4, U)[0].tolist())))})')
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    rows, allres = [], {}
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, g6 = loader(kinds)
        if not g6:
            print('**線の読み込みの門が落ちた**')
            return
        res = run_page(pg, kinds, rows)
        if res is None:
            return
        allres[pg['key']] = res
        del pg
    with open(OUT / 'arm_scores.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('\n=== 棒(追補207 §D。**主の腕 GA m=9、T_face に対する F1@0.5、頁ごと・最悪の頁で**)===')
    ga = f'GA m={MAIN_M}'
    c1 = {p: r[ga][MAIN]['F1'] > max(r['NR 最大'][MAIN]['F1'], r['N1 GA'][MAIN]['F1']) for p, r in allres.items()}
    c3 = {p: r[ga][MAIN]['F1'] > L4_FACE[p] for p, r in allres.items()}
    for p, r in allres.items():
        print(f'  {p:<9} GA {r[ga][MAIN]["F1"]:.3f} / NR 最大 {r["NR 最大"][MAIN]["F1"]:.3f} / N1 {r["N1 GA"][MAIN]["F1"]:.3f}'
              f' / 束ねない L4 {L4_FACE[p]:.3f}  C1 {"はい" if c1[p] else "いいえ"} / C3 {"はい" if c3[p] else "いいえ"}')
    print(f'  **C1 {"通る" if all(c1.values()) else "落ちる"} / C3 {"通る" if all(c3.values()) else "落ちる"}**')
    print(f'\n表: {OUT / "arm_scores.csv"}')
    print('**第2の棒 H2(作者の判定)はこの後。C1・C3 は動かさない(規律10)。**')


if __name__ == '__main__':
    main()
