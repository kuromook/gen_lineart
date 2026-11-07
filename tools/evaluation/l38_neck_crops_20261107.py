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
MININK = 0.002   # 窓の中に最低限要る線の墨(**これを切ったら余白を広げる**)

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
    """窓を**絵として**描く。白地 / 他の面は淡い灰 / **線の墨は黒** / **問う面は半透明の橙**。

    **線を先に置き、問う面を半透明で重ね、最後に輪郭を引く。** 2026-11-07、
    最初の版は**面を塗ってから線を上に置いた**ので、**細い面が線に塗りつぶされて
    「問う面が 1画素も見えない」切り出しが出た**(無作為30面のうち5枚で面が 0.05% 以下、
    1枚は **0.0000%**)。**問う対象が見えない画像は答えられない** ——
    シルエットを送った失敗(同じ日)と同じ病である。

    - 面の画素は **橙を 50% 混ぜる**(線が透けたまま、面が在ることは分かる)
    - **輪郭を濃い橙で引く**(どこまでが1つの面かが、面が細くても分かる。
      **印を付ける**のは 2026-10-26 の方針)
    """
    y0, y1, x0, x1 = box
    img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    img[other[y0:y1, x0:x1]] = (235, 235, 235)       # 同じコマの別の面
    img[lines[y0:y1, x0:x1]] = (0, 0, 0)             # 線の墨(先に置く)
    m = R[y0:y1, x0:x1]
    tint = np.array((170, 215, 255), np.float32)     # BGR = 淡い橙
    img[m] = (0.5 * img[m].astype(np.float32) + 0.5 * tint).astype(np.uint8)
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    th = max(2, int(round(min(img.shape[:2]) / 500)))
    cv2.drawContours(img, cn, -1, (0, 90, 255), th)  # **問う面の輪郭(濃い橙)**
    # **小さい面には在りかを示す枠を足す。** 窓は原寸なので、小さい面だと
    # **広い窓のどこに在るのか分からない**(2026-11-07、面 1,464 画素が
    # 3958x2579 の窓に入った = 窓の 0.001%)。枠は**面の外側**に引くので画素を隠さない
    ys, xs = np.nonzero(m)
    if len(ys):
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if bh * bw < 0.25 * m.size:
            pad = max(40, int(0.02 * min(img.shape[:2])))
            cv2.rectangle(img, (max(0, xs.min() - pad), max(0, ys.min() - pad)),
                          (min(img.shape[1] - 1, xs.max() + pad),
                           min(img.shape[0] - 1, ys.max() + pad)), (0, 90, 255), th)
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
    # **帰無値を取るための口**(2026-11-07)。選んだ面で測った比は母集団の比ではない ——
    # **棒の前に帰無値**(規律9)。素性で選ばず、**母集団から一様に取る**
    ap.add_argument('--random', type=int, default=0,
                    help='**素性で選ばず母集団から一様に N 面**取る(セルの選別は効かない)')
    ap.add_argument('--seed', type=int, default=20261107, help='無作為抽出の種(固定)')
    # **名指しで出す口**(2026-11-07)。選別の規則は別の器具が持ち、ここは描くだけ
    ap.add_argument('--list', help='出す面を名指しした JSON '
                                   '([{work,pid,comp,region,area}, ...])')
    ap.add_argument('--out-name', help='出し先のフォルダー名(既定はセル名か `random`)')
    ap.add_argument('--n', type=int, default=2, help='出す面の数')
    ap.add_argument('--win', type=int, default=700, help='首の窓の一辺(原寸の画素)')
    ap.add_argument('--margin', type=int, default=400,
                    help='面の外接矩形に足す余白(原寸の画素)。**周りの絵が入らないと読めない**')
    a = ap.parse_args()
    out = f'{OUT}/{a.out_name or ("random" if a.random else a.cell)}'
    os.makedirs(out, exist_ok=True)
    allrows = json.load(open(SRC, encoding='utf-8'))
    if a.list:
        want = {(r['work'], r['pid'], r['comp'], r['region'], r['area'])
                for r in json.load(open(a.list, encoding='utf-8'))}
        rows = [r for r in allrows
                if (r['work'], r['pid'], r['comp'], r['region'], r['area']) in want]
        if len(rows) != len(want):
            raise SystemExit(f'名指しの {len(want)} 面に対し素性が {len(rows)} 行。'
                             f'**鍵と面積の両方で引いている**ので、合わなければ止める')
        print(f'母集団 {len(allrows)} 面(現行40頁)/ **名指しの {len(rows)} 面**'
              f'(選別は `{a.list}` の側が持つ)', flush=True)
    elif a.random:
        # **一様。** 大きい面に寄せない・重ならない規則も当てない ——
        # 当てたらそれは母集団の比ではなくなる
        import random as _r
        rows = _r.Random(a.seed).sample(allrows, min(a.random, len(allrows)))
        print(f'母集団 {len(allrows)} 面(現行40頁)/ **一様に {len(rows)} 面**'
              f'(種 {a.seed}。**素性で選んでいない** = 帰無値が取れる)', flush=True)
    else:
        pred = CELLS[a.cell]
        rows = [r for r in allrows if pred(r, a.max_neck)]
        rows.sort(key=lambda r: -r['area'])
        print(f'母集団 {len(allrows)} 面(現行40頁)/ セル `{a.cell}` に当たる面 **{len(rows)}**'
              f'(首の半径の上限 {a.max_neck})', flush=True)
    # **同じ頁・同じ層に偏らせない**(1つの事故を n 回見せても材料にならない)
    if a.random or a.list:
        pick = rows                       # **間引かない**(間引いたら一様でない)
    else:
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
    wid = collections.Counter()
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
                    LN = lines[y0:y1, x0:x1]
                    # **線が入るまで余白を広げる。** 2026-11-07、無作為30面のうち1枚で
                    # **窓の中の線が 0.0000%** になった(周りが白い頁)。**止めて落とすと
                    # 一様でなくなる**ので、**コマの枠まで広げてから**数える(規律9)
                    mg, grew = a.margin, 0
                    while True:
                        by0, by1 = max(0, ys.min() - mg), min(R.shape[0], ys.max() + 1 + mg)
                        bx0, bx1 = max(0, xs.min() - mg), min(R.shape[1], xs.max() + 1 + mg)
                        if LN[by0:by1, bx0:bx1].mean() >= MININK:
                            break
                        if (by1 - by0, bx1 - bx0) == R.shape:
                            skip['**コマ全体でも線が足りない**(広げきった。落としていない)'] += 1
                            break
                        mg, grew = mg * 2, grew + 1
                    if grew:
                        wid[f'余白を {grew} 回広げた'] += 1
                    base = (f"{work}_{pid.split('_', 3)[-1]}_c{ci}_r{r['region']}"
                            f"_area{r['area']}_L{r['n_layer']}_neck{r['neck']}")
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
    for k, v in wid.most_common():
        print(f'  {v}  {k}(**窓の中の線が {MININK:.1%} を切ったので**)')
    print('落としたもの(理由別。規律9):')
    for k, v in (skip.most_common() or [('(衝突は起きなかった)', 0)]):
        print(f'  {v}  {k}')
    if made != len(pick):
        print('**頼んだ数と出した数が違う。止める**')
        return 1


if __name__ == '__main__':
    main()
