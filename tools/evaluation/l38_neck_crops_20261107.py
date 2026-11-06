"""**首で繋がっている面**を原寸 1:1 で切り出す(作者の裁定を仰ぐため)。

作者(2026-11-06): **「一部レイヤが不自然な結合をしているものもあった」**。
`l38_merge_cause_20261106.py` で **面の 12.0%(136/1,130)が細い首で繋がっている**と出た。
**割るべきかどうかは作者が決める**(規律6)。この器具は**判断の材料を原寸で出すだけ**で、
数も出さない。

**縮小したモンタージュで論じない**(規律6 / 2026-10-24 の方針)。だから:

- **`_face.png`** —— 面の外接矩形を**原寸 1:1**。面の画素を黒、同じコマの他の面を淡い灰
  (**どこまでが1つの面か**が見える)
- **`_neck.png`** —— **首の周りだけを原寸 1:1** で窓に切る(既定 700x700)。
  首の場所は計算で出す: **割れた半径で侵食 → 残った2片をそれぞれ膨らませ、
  面の中で両方に触る画素**が首である。印は付けない(画素そのものを見せる)

**首の場所の出し方は推定である。** 面と窓の画素は原寸のままなので、
**作者が見ているのは器具の判断ではなく原稿の画素**である。

## 2026-11-07 追記: **層の軸で選ぶ口を足した**(`--cell`)

作者の裁定2件(`doc/work_log.md` 2026-11-07)で、**首の半径は両方 1 なのに裁定が逆**だった。
分けていたのは **寄与した塗り層の数(面積1%以上)と支配率**である。**n=2 なので軸とは言わない** ——
**層の軸が作者の裁定を予測するか**を見るために、**2x2 の各セルから数面ずつ**出す口が要る。

| `--cell` | 述語 | 現行40頁での数 |
|---|---|---:|
| `2L_neck` | 層 >= 2 **かつ** 首あり | 22(半径<=2 は 7) |
| `2L_only` | 層 >= 2 **かつ** 首なし | 10 |
| `1L_neck` | 層 == 1 **かつ** 首あり | 114(半径<=2 は 20) |
| `1L_only` | 層 == 1 **かつ** 首なし | 984 |

**`2L_neck` と `1L_neck` を同じ `--max-neck` で出せば、違うのは層の軸だけ**(条例6: 対照は
試験する1軸だけで違う)。**面積は揃わない**(2層の面は大きい)ので、**出した面積を印字する**。
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_arms import fill_mask                                   # noqa: E402
from l14_known_cells import A, PAD, comps                        # noqa: E402
from l17_known_cells import (DEL, MAX_FRAC, MIN_FRAC, arms,      # noqa: E402
                             jobs, zmap)

ROOT = '/home/sh1/deepl/lineart-fill-region'
SRC = f'{ROOT}/results/l38_merge_cause_20261106.json'
OUT = f'{ROOT}/results/l38_neck_crops_20261107'
MINPART = 500

# **セルの述語。** `n_layer` は**面積1%以上を寄与した塗り層の数**(`l38_merge_cause` の定義)、
# `neck` は**その半径で侵食すると 500 画素以上の片が2つ以上残る**最小の半径(無ければ `None`)。
CELLS = {
    '2L_neck': lambda r, mx: r['n_layer'] >= 2 and r['neck'] is not None and r['neck'] <= mx,
    '2L_only': lambda r, mx: r['n_layer'] >= 2 and r['neck'] is None,
    '1L_neck': lambda r, mx: r['n_layer'] == 1 and r['neck'] is not None and r['neck'] <= mx,
    '1L_only': lambda r, mx: r['n_layer'] == 1 and r['neck'] is None,
}


def line_ink(lz, shape):
    """**線層の墨の和集合**。これが無いと面はシルエットになり、絵として読めない。

    2026-11-07、最初の版は面のマスクだけを送って作者に
    **「シルエットの画像に対して何を答えて欲しいのか」**と言われた。**面はベタの連結成分**で、
    **線が無いと何の絵か分からない** —— 判定を頼む画像は**答える対象が絵として読めること**が
    条件である(2026-10-26 の方針: 問う対象に印を付ける)。
    """
    out = np.zeros(shape, bool)
    for lid, path, z in lz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != shape:
            continue
        out |= im < 128
    return out


def render(R, other, lines, box):
    """窓を**絵として**描く。白地 / **問う面を橙で塗る** / 他の面は淡い灰 / **線の墨は黒**。

    線を最後に置く —— 塗りの上に線が乗るので、**何の絵かが読めたまま**印が見える。
    """
    y0, y1, x0, x1 = box
    img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    img[other[y0:y1, x0:x1]] = (235, 235, 235)       # 同じコマの別の面
    img[R[y0:y1, x0:x1]] = (170, 215, 255)           # **問う面(BGR = 淡い橙)**
    img[lines[y0:y1, x0:x1]] = (0, 0, 0)             # 線の墨
    return img


def page_masks(fz):
    out, shape = [], None
    for lid, path in fz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        if shape is None:
            shape = ink.shape
        elif ink.shape != shape or not ink.any():
            continue
        if not ink.any():
            continue
        ys, xs = np.nonzero(ink)
        y0, y1 = max(0, ys.min() - PAD), min(shape[0], ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(shape[1], xs.max() + 1 + PAD)
        m = np.zeros(shape, bool)
        m[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
        out.append((int(lid), m))
    return out, shape


def neck_where(R, r):
    """割れた半径 `r` で侵食し、**残った片の膨張が面の中で出会う画素**を首として返す。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    e = cv2.erode(R.astype(np.uint8), k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(e, 8)
    big = [i for i in range(1, n) if st[i, A] >= MINPART]
    if len(big) < 2:
        return None
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 3, 4 * r + 3))
    hit = np.zeros(R.shape, np.uint8)
    for i in big:
        hit += cv2.dilate((lab == i).astype(np.uint8), kd)
    return (hit >= 2) & R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-neck', type=int, default=2, help='首の半径の上限(既定 2 = 最も細い)')
    ap.add_argument('--cell', choices=sorted(CELLS), default='1L_neck',
                    help='**2x2 のどのセルから出すか**(既定 1L_neck = 首だけで層は1つ)')
    ap.add_argument('--n', type=int, default=2, help='出す面の数')
    ap.add_argument('--win', type=int, default=700, help='首の窓の一辺(原寸の画素)')
    ap.add_argument('--margin', type=int, default=400,
                    help='面の外接矩形に足す余白(原寸の画素)。**周りの絵が入らないと読めない**')
    a = ap.parse_args()
    out = f'{OUT}/{a.cell}'
    os.makedirs(out, exist_ok=True)
    allrows = json.load(open(SRC, encoding='utf-8'))
    pred = CELLS[a.cell]
    rows = [r for r in allrows if pred(r, a.max_neck)]
    rows.sort(key=lambda r: -r['area'])
    print(f'母集団 {len(allrows)} 面(現行40頁)/ セル `{a.cell}` に当たる面 **{len(rows)}**'
          f'(首の半径の上限 {a.max_neck})', flush=True)
    # **同じ頁・同じ層に偏らせない**(1つの事故を n 回見せても材料にならない)
    pick, seen = [], set()
    for r in rows:
        k = (r['work'], r['pid'], r['top_lid'])
        if k in seen:
            continue
        seen.add(k)
        pick.append(r)
        if len(pick) >= a.n:
            break
    print(f'そのうち出すのは **{len(pick)}**'
          f'(頁と層が重ならないように上から取った。面積の大きい順。'
          f'**同じ頁・同じ層で落ちた {len(rows) - len(pick)} 面**は出していない)', flush=True)
    for r in pick:
        print(f"  {r['work']} {r['pid']} comp={r['comp']} region={r['region']} "
              f"面積 {r['area']:,} 首の半径 {r['neck']} 層 {r['n_layer']} layer_id={r['top_lid']}")

    want = {(r['work'], r['pid']) for r in pick}
    zz = zmap()
    fill, line = arms('gated')
    js = [j for j in jobs(zz, fill, line) if (j[1], j[0]) in want]
    print(f'\n走る頁 {len(js)}', flush=True)
    made = 0
    skip = collections.Counter()
    for job in js:
        pid, work, fz, lz, sel = job
        lm, shape = page_masks(fz)
        if shape is None:
            continue
        lines = line_ink(lz, shape)
        print(f'  線層 {len(lz)} 枚を読んだ(墨 {lines.mean():.1%})', flush=True)
        F = np.zeros(shape, bool)
        for _, m in lm:
            F |= m
        H, W = shape
        for fr in sel:
            km = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
            if km is None or km.shape != (H, W):
                continue
            n, lab, st, _ = cv2.connectedComponentsWithStats((km > 127).astype(np.uint8), 8)
            for ci in range(1, n):
                ar = int(st[ci, A])
                if ar < MIN_FRAC * H * W or ar > MAX_FRAC * H * W:
                    continue
                y0 = max(0, st[ci, cv2.CC_STAT_TOP] - 2)
                x0 = max(0, st[ci, cv2.CC_STAT_LEFT] - 2)
                y1 = min(H, y0 + st[ci, cv2.CC_STAT_HEIGHT] + 4)
                x1 = min(W, x0 + st[ci, cv2.CC_STAT_WIDTH] + 4)
                P = (lab[y0:y1, x0:x1] == ci)
                glab, gids, gstat = comps(F[y0:y1, x0:x1] & P, 8)
                for r in pick:
                    if (r['work'], r['pid'], r['comp']) != (work, pid, ci):
                        continue
                    if r['region'] not in gids:
                        continue
                    R = glab == r['region']
                    # **`(work, pid, comp)` は面を同定しない。** `comp`(= `ci`)は
                    # 枠ファイルごとに 1 から振り直されるので、**同じ `comp` を持つ別のコマの
                    # 同じ `region` 番号にも当たる**(2026-11-07、最初の版は 2 面を頼んで
                    # **4 回書こうとした**。同じ衝突を同じ日に3度踏んだ。条例4)。
                    # **面積で同定する** —— 鍵に依らない同一性で、行が持っている値である
                    if int(R.sum()) != r['area']:
                        skip['面積が合わない(`comp` が同じ別のコマ)'] += 1
                        continue
                    other = (glab > 0) & ~R
                    # ---- 面の外接矩形を原寸で
                    ys, xs = np.nonzero(R)
                    # **余白を足す。** 面の外接矩形ぴったりだと、**面の中しか見えない** ——
                    # 大きなベタだと窓の中に線が1画素も入らず、作者が何の絵か読めない
                    # (2026-11-07、ako7 の首の窓が線 0% になった)
                    mg = a.margin
                    by0, by1 = max(0, ys.min() - mg), min(R.shape[0], ys.max() + 1 + mg)
                    bx0, bx1 = max(0, xs.min() - mg), min(R.shape[1], xs.max() + 1 + mg)
                    base = (f"{work}_{pid.split('_', 3)[-1]}_c{ci}_r{r['region']}"
                            f"_area{r['area']}_L{r['n_layer']}_neck{r['neck']}")
                    LN = lines[y0:y1, x0:x1]
                    cv2.imwrite(f'{out}/{base}_a_face.png',
                                render(R, other, LN, (by0, by1, bx0, bx1)))
                    # ---- 首の周りを原寸で
                    # **首が無いセル(`*_only`)では窓を出さない。** 無いものの場所は出せない
                    nk = neck_where(R, r['neck']) if r['neck'] is not None else None
                    if nk is not None and nk.any():
                        ny, nx = np.nonzero(nk)
                        cy, cx = int(ny.mean()), int(nx.mean())
                        h = a.win // 2
                        wy0, wy1 = max(0, cy - h), min(R.shape[0], cy + h)
                        wx0, wx1 = max(0, cx - h), min(R.shape[1], cx + h)
                        cv2.imwrite(f'{out}/{base}_b_neck.png',
                                    render(R, other, LN, (wy0, wy1, wx0, wx1)))
                        # **印の無い版も出す** —— 塗りの色が絵の読みを変えていないかを
                        # 作者自身が確かめられるように(規律6: 読みは作者が決める)
                        plain = np.full((wy1 - wy0, wx1 - wx0, 3), 255, np.uint8)
                        plain[LN[wy0:wy1, wx0:wx1]] = (0, 0, 0)
                        cv2.imwrite(f'{out}/{base}_c_neck_lines_only.png', plain)
                        print(f'  書いた {base}  面 {bx1-bx0}x{by1-by0} / '
                              f'首の窓 {wx1-wx0}x{wy1-wy0}(原寸)', flush=True)
                    elif r['neck'] is None:
                        print(f'  書いた {base}  面 {bx1-bx0}x{by1-by0}(原寸。'
                              f'**首が無いので窓は無し**)', flush=True)
                    else:
                        print(f'  **首の場所が出せなかった**: {base}', flush=True)
                    made += 1
    print(f'\n出した面 {made} / 頼んだ {len(pick)}  -> {out}')
    print('落としたもの(理由別。規律9):')
    for k, v in (skip.most_common() or [('(衝突は起きなかった)', 0)]):
        print(f'  {v}  {k}')
    if made != len(pick):
        print('**頼んだ数と出した数が違う。止める**')
        return 1


if __name__ == '__main__':
    main()
