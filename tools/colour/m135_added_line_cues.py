"""追補225 の続き(記述、棒ではない): 作者が判定した足した線(P 24 本・T 24 本)と、計算で取れる量の対応を見る。

**作者の判定を当てる規則を作らない。手がかりの候補を探すだけ。** 48 本しか無いので棒に使わない。
線だけから取れる量(結合の手がかりになりうる):
  w_a, w_b  = 繋いだ2本の線の太さ(端点から 15 px 以内の骨格の距離変換の中央値 ×2)
  split     = 足した線が B の面を割ったか(線の周り 6 px で、B の面の数 < B+X の面の数)
  small     = 割れた片のうち小さい方の面積(px)/ big = 大きい方
  border    = 線に接する B の面が画像の縁に触れるか(背景らしさ。正解を使わない)
  len       = 足した線の長さ
**正解を使う量(手がかりにはしない。記述だけ)**: stop = 線に接する B の面が評価域の外で、B+X の片の1つが評価域の内
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
import m130_t_junctions as m130                                      # noqa: E402
import m133_offset_pairs as m133                                     # noqa: E402
from skimage.morphology import skeletonize                           # noqa: E402

OUT = Path('results/m135_added_line_cues')
VER = {'P': Path('results/m134_p_crops/author_verdicts/verdicts_p'),
       'T': Path('results/m132_t_crops_context/author_verdicts/verdicts_t')}   # T は第2回(追補222)
R_W, R_S = 15, 6


def width_at(pt, sk, dt):
    y, x = int(pt[0]), int(pt[1])
    H, W = sk.shape
    y0, y1, x0, x1 = max(0, y - R_W), min(H, y + R_W + 1), max(0, x - R_W), min(W, x + R_W + 1)
    s = sk[y0:y1, x0:x1].astype(np.uint8)
    yy, xx = np.nonzero(s)
    if len(yy) == 0:
        return float('nan')
    k = int(np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2))
    _, cc = cv2.connectedComponents(s, connectivity=8)
    m = cc == cc[yy[k], xx[k]]
    return float(2 * np.median(dt[y0:y1, x0:x1][m]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    rows = []
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, H, W = pg['key'], pg['H'], pg['W']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        inkB, _, _, _, _ = m130.build_b(pg['ink'])
        sk = skeletonize(inkB)
        dt = cv2.distanceTransform(inkB.astype(np.uint8), cv2.DIST_L2, 5)
        labB = np.load(Path('results/m126_line_ends') / f'{key}_labB.npy')
        keepB, _ = m122.in_domain(labB, U)
        inB = np.zeros(int(labB.max()) + 1, bool); inB[keepB] = True
        bord = np.zeros(int(labB.max()) + 1, bool)
        bord[np.unique(np.concatenate([labB[0], labB[-1], labB[:, 0], labB[:, -1]]))] = True
        areaB = np.bincount(labB.ravel())
        for arm, segf, labf, tag in (('P', m133.OUT / f'{key}_segs.json', m133.OUT / f'{key}_labP.npy', 'p'),
                                     ('T', m130.OUT / f'{key}_segs.json', m130.OUT / f'{key}_labT.npy', 't')):
            segs = json.load(open(segf))['segs']
            labX = np.load(labf)
            keepX, _ = m122.in_domain(labX, U)
            inX = np.zeros(int(labX.max()) + 1, bool); inX[keepX] = True
            areaX = np.bincount(labX.ravel())
            rng = np.random.default_rng(0)
            sel = sorted(rng.permutation(len(segs))[:12].tolist()) if len(segs) > 12 else list(range(len(segs)))
            for n, t in enumerate(sel):
                vid = f'{key}-{tag}{n + 1:02d}'
                vf = VER[arm] / f'{vid}.json'
                v = json.load(open(vf)).get('q1', 'none') if vf.exists() else 'none'
                (ay, ax), (by, bx) = segs[t]
                m = np.zeros((H, W), np.uint8)
                cv2.line(m, (ax, ay), (bx, by), 1, 1 + 2 * R_S)
                m = m.astype(bool) & ~inkB
                lb = [l for l in np.unique(labB[m]) if l > 0]
                lx = [l for l in np.unique(labX[m]) if l > 0]
                pieces = sorted(areaX[lx].tolist()) if lx else [0]
                rows.append(dict(
                    id=vid, arm=arm, page=key, verdict=v,
                    w_a=width_at((ay, ax), sk, dt), w_b=width_at((by, bx), sk, dt),
                    len=float(np.hypot(by - ay, bx - ax)),
                    split=len(lx) > len(lb), nB=len(lb), nX=len(lx),
                    small=int(pieces[0]) if len(lx) > 1 else 0, big=int(pieces[-1]),
                    areaB=int(max(areaB[lb])) if lb else 0,
                    border=bool(bord[lb].any()) if lb else False,
                    stop=bool(lb and (~inB[lb]).any() and inX[lx].any())))
            del labX
        del pg, Fl, U, inkB, sk, dt, labB
    json.dump(rows, open(OUT / 'rows.json', 'w'), ensure_ascii=False, indent=0)
    print(f'{"id":<15}{"腕":<3}{"判定":<8}{"w_a":>5}{"w_b":>5}{"長さ":>6}{"割る":>4}{"小片":>8}{"縁":>3}{"stop*":>6}')
    for r in rows:
        print(f'{r["id"]:<15}{r["arm"]:<3}{r["verdict"]:<8}{r["w_a"]:>5.1f}{r["w_b"]:>5.1f}{r["len"]:>6.1f}'
              f'{"○" if r["split"] else "・":>4}{r["small"]:>8}{"○" if r["border"] else "・":>3}{"○" if r["stop"] else "・":>6}')
    print('\n* stop は正解(マスクの面)を使う量。手がかりにしない。\n')
    for arm in ('P', 'T', '両方'):
        print(f'--- {arm}')
        for v in ('right', 'wrong'):
            g = [r for r in rows if r['verdict'] == v and (arm == '両方' or r['arm'] == arm)]
            if not g:
                continue
            wmin = np.array([min(r['w_a'], r['w_b']) for r in g])
            print(f'  {v:<6} n={len(g):>2}  細い方の太さ 中央値 {np.nanmedian(wmin):.1f}px(四分位 {np.nanpercentile(wmin,25):.1f}–{np.nanpercentile(wmin,75):.1f})'
                  f' / 割る {sum(r["split"] for r in g)} / 縁 {sum(r["border"] for r in g)} / stop* {sum(r["stop"] for r in g)}'
                  f' / 長さ 中央値 {np.median([r["len"] for r in g]):.1f}')
        oth = [r for r in rows if r['verdict'] not in ('right', 'wrong') and (arm == '両方' or r['arm'] == arm)]
        print(f'  わからない・答えなし {len(oth)}(表から外した)')


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def breakdown():
    """F1 の内訳(記述): 当たり / 評価域の提案 / 正解の面。B・B+P・B+T。"""
    kinds = m102.load_kinds()
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key = pg['key']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        prev = None
        for name, f in (('B', Path('results/m126_line_ends') / f'{key}_labB.npy'),
                        ('B+P', m133.OUT / f'{key}_labP.npy'), ('B+T', m130.OUT / f'{key}_labT.npy')):
            lab = np.load(f)
            keep, nout = m122.in_domain(lab, U)
            r = m122.score(lab, Fl, keep)[m122.MAIN]
            extra = ''
            if prev is not None:
                extra = f'  B から: 失った当たり {len(prev - r["hits"])} / 増えた当たり {len(r["hits"] - prev)}'
            else:
                prev = r['hits']
            print(f'{key} {name:<4} F1 {r["F1"]:.3f}  当たり {r["hit"]} / 評価域の提案 {r["n_prop"]} / 正解の面 {r["n_true"]}  評価域の外 {nout}{extra}')
            del lab
        del pg, Fl, U


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'breakdown':
    breakdown()


def new_faces():
    """B に無く B+X で評価域に入った提案は何か(記述): 最良 IoU、覆う正解の面の数(提案の 10% 以上を占める面)、面積。"""
    kinds = m102.load_kinds()
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key = pg['key']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        aT = np.bincount(Fl.ravel())
        labB = np.load(Path('results/m126_line_ends') / f'{key}_labB.npy')
        keepB, _ = m122.in_domain(labB, U)
        inB = np.zeros(int(labB.max()) + 1, bool); inB[keepB] = True
        for name, f in (('B+P', m133.OUT / f'{key}_labP.npy'), ('B+T', m130.OUT / f'{key}_labT.npy')):
            lab = np.load(f)
            keep, _ = m122.in_domain(lab, U)
            r = m122.score(lab, Fl, keep)[m122.MAIN]
            kinds_ = {'まとめすぎ(2面以上)': 0, '1面の一部': 0, '当たり': 0}
            rows = []
            for p in keep:
                m = lab == p
                # B で評価域の内にあった面とほぼ同じなら「新しい」ではない
                lb, cb = np.unique(labB[m], return_counts=True)
                j = int(np.argmax(cb))
                if inB[lb[j]] and cb[j] >= 0.9 * m.sum():
                    continue
                t, c = np.unique(Fl[m], return_counts=True)
                c, t = c[t > 0], t[t > 0]
                ious = c / (m.sum() + aT[t] - c)
                cover = int((c >= 0.1 * m.sum()).sum())
                k = '当たり' if p in r['pids'] else ('まとめすぎ(2面以上)' if cover >= 2 else '1面の一部')
                kinds_[k] += 1
                rows.append((int(m.sum()), cover, float(ious.max()) if len(ious) else 0.0, k))
            print(f'{key} {name}: 新しく評価域に入った提案 {len(rows)} = ' + ' / '.join(f'{a} {b}' for a, b in kinds_.items()))
            for a, cv, io, k in sorted(rows, reverse=True)[:8]:
                print(f'    面積 {a:>8}  覆う正解の面 {cv}  最良 IoU {io:.2f}  {k}')
            del lab
        del pg, Fl, U, labB


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'new':
    new_faces()
