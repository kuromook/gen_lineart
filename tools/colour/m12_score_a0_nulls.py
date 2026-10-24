"""M12: 登録集合8ページで **A0 と帰無値 N1/N2/N3 を採点する**。腕を採点する
最初のファイルである。走らせる前にこの docstring を書いて commit した。

----------------------------------------------------------------- 登録(既存)

標準 S1(追補6 §F2): **シルエット層の8連結成分 1つ = 面1つ**。
シルエット層の同定は**構造**(クリップの連鎖のベース、追補5 §E4 B3′)で行い、
**名前では決めない**。同定器は `m11.clip_bases` をそのまま import する
(写すと分かれるので import する)。

面積の床(追補6 §F3): **ページ面積の 1e-5**、**標準側と A0 側で同じ値**。
A0(追補5 §E6 + 追補6 §F3): `CanvasPreview` を**一様量子化 `floor(v*8/256)`**
(チャンネルあたり8段、ディザ無し・パレット当てはめ無し・色空間変換無し)→
**等しい量子化色の8連結領域** → 床 1e-5 → ラベル地図。学習なし。

棒(起案 §3.2。結果を見てから動かさない):
  1. **A0 は N1/N2/N3 のすべてに、ページごとの符号検定で勝つ**(両側 α=0.05)
  2. 学習した腕は A0 に勝たない限り読まない
  3. **既知セルを毎回含める**: 標準を予測として入れて region-F1 **1.0**。
     再現しなければ他の行を読まない
  4. **プールを跨いで平均しない**。ページごとに行を印字する
  5. 判定保留は正直な結論

**この8ページは登録集合である**(追補3 §C8)。切り口はここで選んだので、
**ここからは主張を出さない**。主張は検定集合(872 − 8 ページ)まで出ない。

**n=8 で両側符号検定が α=0.05 を通るのは 8/8 のときだけ**である
(2*(1/2)^8 = 0.0078。7/8 なら 0.070 で通らない)。**これは走らせる前から
分かっている**ので、登録集合の結果は「8/8 か否か」しか情報を持たない。

限界(追補6 §F4。**全ての行に付ける**): 8連結成分は作者の面の**下限**である
(襟と身頃のように隣り合う面は1つの成分になる)。ゆえに **region-F1 は
「少なく大きい面を出す腕」に有利に出る**。A0 が勝っても、それは
「標準の分け足りない向きと A0 の粗さが同じ向き」である可能性を排除しない。

------------------------------------------------- ここで新しく登録すること

採点には**標準と予測を同じ格子に置く**必要があるが、**原寸の合成は存在せず**
(追補2 §B4)、A0 の入力は縮小率が作品ごとに違うプレビューしかない。
**どちらの格子で照合するかは、まだ登録していない。以下を結果を見る前に登録する。**

  R1. **照合は原寸の格子で行う。** 標準を縮小しない。縮小すると 1/8 ページで
      床 1e-5 が 10 画素になり、**作者が描いた面が照合の前に消える**。
      A0 のラベル地図は**最近傍でブロック拡大**する(拡大は情報を増やさないし、
      連結成分を分けも繋ぎもしない。面積は s^2 倍になるだけ)。
  R2. 格子は `プレビュー寸 × s`(s = 縮小率の丸め)。キャンバスとの差は
      **右端・下端の数画素**なので、標準をその格子に切る。**切った行数・列数を
      毎ページ印字する**。
  R3. **重なりの解決**: 成分は層ごとに独立に取る(= 追補6 の 504)。
      ラベル地図は1画素1ラベルなので重なりを決めねばならないが、
      **フォルダを跨ぐ z 順はマニフェストに無い**(フォルダ層は export されず、
      `ancestors` は名前の並びだけで順序を持たない)。ゆえに z 順に依らない
      規則を登録する: **面積の小さい成分が勝つ**(顔は頭のシルエットの上に
      乗る部品で、大きい方に飲ませると部品が丸ごと消える)。面積が同じときは
      `layer_id` の小さい方が先に塗られて勝つ。**争った画素数と、
      解決の後に床を割った成分の数を印字する**。
  R4. 重なりで**2つに割れた成分は1つのラベルのまま**にする(標準の単位は
      「層の成分」であって「解決後の連結成分」ではない)。
  R5. 感受性の格子(追補5 §E6。**毎行印字し、判定には一切使わない**):
      段数 (2,4,8,16,32) × 床 (1e-5,1e-4,1e-3,1e-2)、および IoU (0.25,0.5,0.75)。
      **床を動かすと標準も動く**(§F3: 床は両側で同じ)。格子の中では
      標準を**1e-5 で塗った後にラベル面積で濾して**作る(塗り直さない)。
      **主値の行(段数8・床1e-5・IoU 0.5)だけが塗り直しと一致する。**

器具確認(値は走らせる前に書いた):
  IC-M5 `skimage.measure.label(connectivity=2, background=-1)` は m9 の
        floodFill 歩行と同じ面を返すこと。`page0019` の 5段数 × (総数 +
        4つの床) = 25 セルを m9 の印字と突き合わせる(下の `M9_PAGE0019`)。
        **一致しなければ以降の行を読まない。**
  IC-M6 既知セル(棒3): 標準を予測として入れて region-F1 = 1.0 を**全8ページで**。
  IC-M7 **ブロック拡大の経路そのものの既知セル**: IC-M6 は s=1 なので R1 の
        拡大を通らない。A0 のラベル地図を原寸に拡大したものを標準として
        A0 自身を採点すると region-F1 = 1.0 でなければならない。これが
        通らないかぎり「A0 の点が低い」は腕の性質として読めない(位置ずれ
        でも同じ低い点が出る)。**全8ページで印字する。**

`measure_*` / `judge_*` の分離(追補2 の三点拡張): このファイルの `score_page`
は器具で、勝敗の読みは `judge_*` にしか書かない。
"""
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label as sk_label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_partition_metrics as m2                      # noqa: E402
from m11_silhouette_components import clip_bases       # noqa: E402

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
OUT = Path('results/m12_score'); OUT.mkdir(parents=True, exist_ok=True)

ALPHA_DRAWN = 26          # m11 と同じ
FLOOR = 1e-5              # 追補6 §F3。ページ面積に対する割合
LEVELS = 8                # 追補5 §E6
IOU = 0.5                 # 借りた規則(Track J/L)
SENS_LEVELS = (2, 4, 8, 16, 32)
SENS_FLOORS = (1e-5, 1e-4, 1e-3, 1e-2)
SENS_IOU = (0.25, 0.5, 0.75)

# IC-M5 の期待値: m9 の印字(results/ に残っている m9 の出力そのまま)。
# 段数 -> (床なしの面数, n@1e-5, n@1e-4, n@1e-3, n@1e-2)
M9_PAGE0019 = {2: (35115, 465, 72, 17, 4),
               4: (84850, 868, 89, 26, 4),
               8: (135221, 1156, 114, 29, 7),
               16: (160552, 1119, 112, 31, 7),
               32: (187631, 1083, 114, 33, 6)}


# ----------------------------------------------------------------- measure

def quantise(bgr, levels):
    """m9 と同一。一様量子化 floor(v*levels/256)。"""
    q = (bgr.astype(np.uint16) * levels // 256).astype(np.uint8)
    return q[..., 0].astype(np.int32) * levels * levels + \
        q[..., 1].astype(np.int32) * levels + q[..., 2].astype(np.int32)


def equal_value_components(vals):
    """等しい値の8連結成分。`background=-1` で「背景なし」= 全画素を labelする。"""
    return sk_label(vals, connectivity=2, background=-1)


def filtered_labels(lab, min_px):
    """面積 >= min_px のラベルだけを 1.. に振り直したラベル地図と面積表。

    **ラベル 0 は背景なので必ず落とす**(`m2.areas` と同じ扱い)。落とさないと
    余白が1つの巨大な面として標準に入り、N2(ページ全体を1つの面)がそれと
    照合してしまう。最初の走行はこれを踏んで N2 に 0.6667 と 1.0000 を与えた。
    等値成分(A0 側)は 1 から始まるのでこの行は効かない。"""
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    keep = np.nonzero(cnt >= min_px)[0]
    lut = np.zeros(cnt.size, np.int32)
    lut[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    out = lut[lab]
    return out, {i + 1: int(cnt[k]) for i, k in enumerate(keep)}


def measure_standard(page, grid, floor_px):
    """S1 の原寸ラベル地図。成分は層ごとに独立、重なりは小さい成分が勝つ(R3)。"""
    gh, gw = grid
    lab = np.zeros(grid, np.int32)
    own = np.zeros(grid, np.int32)        # いまその画素を持っている成分の面積
    bases = clip_bases(page)
    nxt, indep, info = 1, 0, []
    for lid, l in sorted(bases.items()):
        im = cv2.imread(str(D / l['file']), cv2.IMREAD_UNCHANGED)
        m = (im[..., 3] >= ALPHA_DRAWN).astype(np.uint8)[:gh, :gw]
        n, cl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        order = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= floor_px]
        indep += len(order)
        order.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
        for i in order:
            a = int(stats[i, cv2.CC_STAT_AREA])
            y, x, h, w = (stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT],
                          stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH])
            sub = cl[y:y + h, x:x + w] == i
            o, lb = own[y:y + h, x:x + w], lab[y:y + h, x:x + w]
            take = sub & ((o == 0) | (o > a))
            o[take], lb[take] = a, nxt
            info.append({'label': nxt, 'layer_id': lid, 'name': l['name'],
                         'area_indep': a})
            nxt += 1
        del im, m, cl
    cnt = np.bincount(lab.ravel(), minlength=nxt)
    for r in info:
        r['area_painted'] = int(cnt[r['label']])
    lost = [r for r in info if r['area_painted'] < floor_px]
    contested = sum(r['area_indep'] for r in info) - int(cnt[1:].sum())
    lab, garea = filtered_labels(lab, floor_px)
    return lab, garea, {'silhouettes': len(bases), 'components_indep': indep,
                        'components_kept': len(garea), 'lost_to_overlap': len(lost),
                        'contested_px': contested,
                        'contested_pages': contested / (gh * gw), 'info': info}


def pair_counts_upsampled(pred_small, s, gt, block=48):
    """ブロック拡大した予測と原寸の標準の交差画素数。行ブロックで数えるので
    原寸のラベル地図を2枚メモリに置かない(R1)。s=1 なら原寸どうし。"""
    ngt = int(gt.max())
    out = collections.Counter()
    H = pred_small.shape[0]
    for y0 in range(0, H, block):
        y1 = min(H, y0 + block)
        ps = pred_small[y0:y1]
        if s != 1:
            ps = np.repeat(np.repeat(ps, s, axis=0), s, axis=1)
        g = gt[y0 * s:y0 * s + ps.shape[0], :ps.shape[1]]
        ps = ps[:g.shape[0], :g.shape[1]]
        m = (ps > 0) & (g > 0)
        if not m.any():
            continue
        key = ps[m].astype(np.int64) * (ngt + 1) + g[m].astype(np.int64)
        c = np.bincount(key)
        for k in np.nonzero(c)[0]:
            out[(int(k) // (ngt + 1), int(k) % (ngt + 1))] += int(c[k])
    return out


def grid_null(shape, n_want):
    """N3: 標準と同じ面数の規則的な格子。面数とページの縦横比の両方に近い
    (行, 列) を選ぶ。実際の面数は毎行印字する。"""
    h, w = shape
    best = None
    for r in range(1, max(2, n_want + 1)):
        c = max(1, round(n_want / r))
        key = (abs(r * c - n_want), abs((h / r) / (w / c) - 1.0))
        if best is None or key < best[0]:
            best = (key, r, c)
    _, r, c = best
    ys = np.linspace(0, h, r + 1).astype(int)
    xs = np.linspace(0, w, c + 1).astype(int)
    lab = np.zeros(shape, np.int32)
    k = 1
    for i in range(r):
        for j in range(c):
            lab[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = k
            k += 1
    return lab, r, c


def score_from_small(pred_small, s, gt, garea, min_px, iou=IOU, pc=None):
    """予測(小格子・ブロック拡大)を原寸の標準に当てて借りた規則で採点する。"""
    if pc is None:
        pc = pair_counts_upsampled(pred_small, s, gt)
    cnt = np.bincount(pred_small.ravel())
    parea = {int(i): int(cnt[i]) * s * s for i in np.nonzero(cnt)[0] if i > 0
             and int(cnt[i]) * s * s >= min_px}
    g = {k: v for k, v in garea.items() if v >= min_px}
    r = dict(m2.match_from_counts(pc, parea, g, iou))
    return r, pc


# ------------------------------------------------------------ 器具確認

def ic_m7(pred_small, s, grid, min_px):
    """ブロック拡大の経路の既知セル。拡大した自分自身を標準にすると 1.0。"""
    up = np.repeat(np.repeat(pred_small, s, axis=0), s, axis=1)[:grid[0], :grid[1]]
    cnt = np.bincount(up.ravel())
    cnt[0] = 0
    garea = {int(i): int(cnt[i]) for i in np.nonzero(cnt)[0] if cnt[i] >= min_px}
    r, _ = score_from_small(pred_small, s, up, garea, min_px)
    del up
    return r


def ic_m5(pages):
    """skimage の等値成分が m9 の floodFill 歩行と一致すること。"""
    p = next(q for q in pages if q['page'].endswith('sunshine2_page0019'))
    im = cv2.imread(str(D / p['preview']['file']), cv2.IMREAD_COLOR)
    n_px = im.shape[0] * im.shape[1]
    ok = True
    for lv, want in M9_PAGE0019.items():
        lab = equal_value_components(quantise(im, lv))
        cnt = np.bincount(lab.ravel())[1:] if lab.min() == 0 else np.bincount(lab.ravel())
        cnt = np.bincount(lab.ravel())
        cnt = cnt[cnt > 0]
        got = (int(cnt.size),) + tuple(int((cnt >= max(1, round(f * n_px))).sum())
                                       for f in SENS_FLOORS)
        good = got == want
        ok &= good
        print(f'  IC-M5 段数{lv:>3}: {got} 期待 {want} -> {"PASS" if good else "FAIL"}')
    print(f'IC-M5  skimage の等値成分 = m9 の floodFill: '
          f'{"PASS" if ok else "FAIL"}')
    return ok


# ----------------------------------------------------------------- judge

def judge_sign_test(wins, losses):
    """両側符号検定。引き分けは落とす(符号検定の定義)。"""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    k = min(wins, losses)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def main():
    pages = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    if not ic_m5(pages):
        print('IC-M5 FAIL -- 以降の行は読まない'); return 1

    rows, grids, t0 = [], [], time.time()
    for p in pages:
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        cw, ch = p['canvas']
        n_px = grid[0] * grid[1]
        floor_px = max(1, round(FLOOR * n_px))
        im = cv2.imread(str(D / p['preview']['file']), cv2.IMREAD_COLOR)
        assert im.shape[:2] == (ph, pw), (im.shape, (ph, pw))

        gt, garea, st = measure_standard(p, grid, floor_px)
        # IC-M6 既知セル: 標準を予測として入れる(原寸どうし、s=1)
        known, _ = score_from_small(gt, 1, gt, garea, floor_px)

        a0_lab = equal_value_components(quantise(im, LEVELS))
        a0, _ = filtered_labels(a0_lab, max(1, round(FLOOR * pw * ph)))
        n1 = np.zeros((ph, pw), np.int32)
        n2 = np.ones((ph, pw), np.int32)
        n3, gr, gc = grid_null((ph, pw), len(garea))

        self_up = ic_m7(a0, s, grid, floor_px)
        r_a0, pc_a0 = score_from_small(a0, s, gt, garea, floor_px)
        r_n1, _ = score_from_small(n1, s, gt, garea, floor_px)
        r_n2, _ = score_from_small(n2, s, gt, garea, floor_px)
        r_n3, _ = score_from_small(n3, s, gt, garea, floor_px)

        row = {'page': p['page'], 'scale': s, 'grid': f'{grid[1]}x{grid[0]}',
               'cropped': f'{cw - grid[1]}x{ch - grid[0]}', 'floor_px': floor_px,
               'silhouettes': st['silhouettes'],
               'gt_indep': st['components_indep'], 'gt_areas': len(garea),
               'gt_lost_overlap': st['lost_to_overlap'],
               'contested_px': st['contested_px'],
               'contested_pages': round(st['contested_pages'], 3),
               'known_cell_f1': round(known['region_f1'], 6),
               'ic_m7_f1': round(self_up['region_f1'], 6),
               'ic_m7_n': self_up['n_pred'],
               'a0_n': r_a0['n_pred'], 'a0_matched': r_a0['matched'],
               'a0_prec': round(r_a0['precision'], 4),
               'a0_rec': round(r_a0['recall'], 4),
               'a0_f1': round(r_a0['region_f1'], 4),
               'n1_f1': round(r_n1['region_f1'], 4),
               'n2_f1': round(r_n2['region_f1'], 4),
               'n3_f1': round(r_n3['region_f1'], 4),
               'n3_shape': f'{gr}x{gc}={gr*gc}'}
        rows.append(row)
        print(f'{p["page"]:44s} s=1/{s} gt {len(garea):>4} '
              f'known {known["region_f1"]:.4f} up {self_up["region_f1"]:.4f} '
              f'A0 {r_a0["region_f1"]:.4f}(n={r_a0["n_pred"]}) '
              f'N1 {r_n1["region_f1"]:.4f} N2 {r_n2["region_f1"]:.4f} '
              f'N3 {r_n3["region_f1"]:.4f}  {time.time()-t0:6.1f}s', flush=True)

        # R5 の格子。判定には使わない。
        g = {'page': p['page']}
        for lv in SENS_LEVELS:
            lab = equal_value_components(quantise(im, lv))
            labf, _ = filtered_labels(lab, max(1, round(min(SENS_FLOORS) * pw * ph)))
            pc = pair_counts_upsampled(labf, s, gt)
            for fl in SENS_FLOORS:
                mp = max(1, round(fl * n_px))
                for th in SENS_IOU:
                    r, _ = score_from_small(labf, s, gt, garea, mp, th, pc=pc)
                    g[f'lv{lv}_f{fl:.0e}_t{th}'] = round(r['region_f1'], 4)
            del lab, labf
        grids.append(g)
        del gt, a0_lab, a0

    import csv
    with open(OUT / 'scores.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / 'sensitivity_grid.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(grids[0].keys()))
        w.writeheader(); w.writerows(grids)

    print('\n== 既知セル(棒3): 標準を予測として入れると region-F1 = 1.0 ==')
    bad = [r for r in rows if abs(r['known_cell_f1'] - 1.0) > 1e-9]
    print(f'  {len(rows)-len(bad)}/{len(rows)} ページで 1.0 -> '
          f'{"PASS" if not bad else "FAIL"}')
    if bad:
        for r in bad:
            print(f'  FAIL {r["page"]} {r["known_cell_f1"]}')
        print('既知セルが再現しない。他の行は読まない。'); return 1

    print('\n== IC-M7 ブロック拡大の経路(拡大した自分自身を標準にすると 1.0) ==')
    bad7 = [r for r in rows if abs(r['ic_m7_f1'] - 1.0) > 1e-9]
    print(f'  {len(rows)-len(bad7)}/{len(rows)} ページで 1.0 -> '
          f'{"PASS" if not bad7 else "FAIL"}')
    if bad7:
        for r in bad7:
            print(f'  FAIL {r["page"]} {r["ic_m7_f1"]} (n={r["ic_m7_n"]})')
        print('拡大の経路が通らない。A0 の低い点は腕の性質として読めない。')
        return 1

    print('\n== ページごと(プールしない。登録集合なので主張は出さない) ==')
    print(f'{"page":44s} {"s":>3} {"gt面":>5} {"A0面":>6} {"照合":>5} '
          f'{"A0prec":>7} {"A0rec":>7} {"A0":>7} {"N1":>7} {"N2":>7} {"N3":>7}')
    for r in rows:
        print(f'{r["page"]:44s} 1/{r["scale"]:<2} {r["gt_areas"]:>5} '
              f'{r["a0_n"]:>6} {r["a0_matched"]:>5} {r["a0_prec"]:>7.4f} '
              f'{r["a0_rec"]:>7.4f} {r["a0_f1"]:>7.4f} {r["n1_f1"]:>7.4f} '
              f'{r["n2_f1"]:>7.4f} {r["n3_f1"]:>7.4f}')
    print('  precision / recall は借りた規則(IoU>=0.5 の貪欲1対1)の列であって、'
          '新しい棒ではない。棒は region-F1 の符号検定のまま。')

    print('\n== judge: ページごとの符号検定(両側。n=8 では 8/8 以外は通らない) ==')
    for k, nm in (('n1_f1', 'N1 blank'), ('n2_f1', 'N2 whole_page'),
                  ('n3_f1', 'N3 grid')):
        win = sum(1 for r in rows if r['a0_f1'] > r[k])
        loss = sum(1 for r in rows if r['a0_f1'] < r[k])
        tie = len(rows) - win - loss
        pv = judge_sign_test(win, loss)
        print(f'  A0 対 {nm:14s} 勝 {win} 負 {loss} 引 {tie}  '
              f'p={pv:.4f} -> {"棒を満たす" if (pv < 0.05 and win > loss) else "満たさない"}')

    print('\n== 標準の形(追補6 §F2/§F3/§F4 の数をこの格子で取り直したもの) ==')
    print(f'  シルエット層 {sum(r["silhouettes"] for r in rows)}  '
          f'成分(層ごとに独立) {sum(r["gt_indep"] for r in rows)}  '
          f'重なり解決後に残った面 {sum(r["gt_areas"] for r in rows)}  '
          f'重なりで床を割った成分 {sum(r["gt_lost_overlap"] for r in rows)}')
    cont = ' '.join('%.2f' % r['contested_pages'] for r in rows)
    print(f'  争った画素(重なりの総量。ページ面積の何倍か): {cont}')
    degen = [r for r in rows if r['gt_areas'] == 0]
    if degen:
        print(f'\n  うち標準が1つも面を持たないページ {len(degen)}/{len(rows)}: '
              f'{[r["page"] for r in degen]}')
        print('  そのページでは「何も出さない」N1 が定義上 1.0 を取る。'
              '登録した比較は全行で行うので除外しない(追補5 §E8)。')
        print('  除外した場合の数を隣に併記する(置き換えない):')
        sub = [r for r in rows if r['gt_areas'] > 0]
        for k, nm in (('n1_f1', 'N1 blank'), ('n2_f1', 'N2 whole_page'),
                      ('n3_f1', 'N3 grid')):
            win = sum(1 for r in sub if r['a0_f1'] > r[k])
            loss = sum(1 for r in sub if r['a0_f1'] < r[k])
            print(f'    (併記) A0 対 {nm:14s} 勝 {win} 負 {loss} 引 '
                  f'{len(sub)-win-loss}  p={judge_sign_test(win, loss):.4f}')

    print('\n限界(追補6 §F4): 8連結成分は作者の面の下限である。隣り合う面は')
    print('1つの成分になるので、region-F1 は少なく大きい面を出す腕に有利に出る。')
    print('この8ページは登録集合である。主張は検定集合(872-8ページ)まで出ない。')
    print(f'\nwrote {OUT/"scores.csv"} / {OUT/"sensitivity_grid.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
