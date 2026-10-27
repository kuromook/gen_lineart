# -*- coding: utf-8 -*-
"""L17 の器具と既知セル KC16〜KC19。PASS しなければ本体は起動しない(登録 §5)。

器具(本体 `l17_fill_adjacency.py` はここから import する):

  assign_top  各墨画素を、それを塗る線層のうち z_global が最大のもの(最も手前)に割り当てる
  adj_sets    ∂R から距離 d 以内(真円のユークリッド距離)・コマの中の**門が通した線の形の層**の墨を層ごとに数える
  purity      max_L share      n_lay90  adj の9割を覆うのに要る層数

**帰無値も棒もここには無い。** ここは器具が壊れていないかだけを見る。
"""
import csv, json, os, sys, zlib
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import (BASE, COV_MAX, DARK_MIN, DROP_TOP, INV, KEEP4,
                            KEEP4_FILL, comps, gated_layers, topf, A)
from pid_work import work_of  # **既定値に落とさない**(T104 の族)
from l14_arms import acc_add
from l15_known_cells import boundary

DEL  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
ZCSV = '/home/sh1/deepl/lineart-fill-region/inbox/layer_z_order_priority65.csv'
PREV = '/home/sh1/deepl/lineart-fill-region/results/l15_zorder_occ/per_region.csv'
DS   = (2, 4, 8, 16)
D1   = 4
MIN_FRAC, MAX_FRAC = 0.002, 0.50
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)
TRIES = 48
MIN_IN, MIN_MOVE = 0.9, 0.10


# ---------------------------------------------------------------- 器具

def assign_top(layers, shape):
    """layers = [(key, ink, z)]。z の低い方から塗り重ねるので、残るのは最も手前の層。

    戻り値 (lab, keys)。lab==i+1 が keys[i] の層。重なる層は二重に数えられない。
    """
    lab = np.zeros(shape, np.int32)
    keys = [k for k, _, _ in sorted(layers, key=lambda t: t[2])]
    for i, (k, ink, z) in enumerate(sorted(layers, key=lambda t: t[2])):
        lab[ink] = i + 1
    return lab, keys


def adj_sets(R, lab, P, ds=DS, bnd=None):
    """{d: (層番号の配列, 画素数の配列)}。距離は真円(DIST_MASK_PRECISE)。"""
    b = boundary(R) if bnd is None else bnd
    if not b.any():
        return {d: (np.zeros(0, np.int32), np.zeros(0, np.int64)) for d in ds}
    dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                               cv2.DIST_MASK_PRECISE)
    out = {}
    for d in ds:
        m = (dt <= d) & P
        v = lab[m]; v = v[v > 0]
        if v.size == 0:
            out[d] = (np.zeros(0, np.int32), np.zeros(0, np.int64))
        else:
            vals, cnt = np.unique(v, return_counts=True)
            out[d] = (vals, cnt)
    return out


def patch_of(glab, gi, gstat):
    yy = int(gstat[gi, cv2.CC_STAT_TOP]); xx = int(gstat[gi, cv2.CC_STAT_LEFT])
    hh = int(gstat[gi, cv2.CC_STAT_HEIGHT]); ww = int(gstat[gi, cv2.CC_STAT_WIDTH])
    return (glab[yy:yy + hh, xx:xx + ww] == gi), yy, xx, hh, ww


def shift_to(patch, ny, nx, shape):
    out = np.zeros(shape, bool)
    hh, ww = patch.shape
    out[ny:ny + hh, nx:nx + ww] = patch
    return out


def place(glab, gi, gstat, P, seed, pid, ci, tries=TRIES,
          min_in=MIN_IN, min_move=MIN_MOVE):
    """面をコマ内のランダムな位置へ平行移動する。置けなければ None。

    **乱数は面ごとに種を切る**ので、どの腕を走らせても同じ面が同じ位置に置かれる。
    (基盤 §5: 対照の面集合が腕と違っていた件の直し。頁単位の rng では
    腕の構成で位置が変わりうる)
    """
    ph, pw = P.shape
    patch, yy, xx, hh, ww = patch_of(glab, gi, gstat)
    if ww + 2 >= pw or hh + 2 >= ph:
        return None
    pn = int(patch.sum())
    # **`hash(str)` は走行ごとに変わる**(PYTHONHASHSEED)。ここに使っていたので、
    # **帰無値も面集合も走行ごとに別物になっていた**(2026-10-25 に発見)。
    # crc32 は過程に依らないので、同じ引数なら必ず同じ位置に置かれる。
    rng = np.random.default_rng([int(seed), zlib.crc32(pid.encode()) % 100000,
                                 int(ci), int(gi)])
    for _ in range(tries):
        nx = int(rng.integers(1, pw - ww - 1)); ny = int(rng.integers(1, ph - hh - 1))
        if abs(nx - xx) + abs(ny - yy) < min_move * (pw + ph):
            continue
        Rs = shift_to(patch, ny, nx, P.shape)
        if int((Rs & P).sum()) < min_in * pn:
            continue
        return Rs
    return None


def purity(cnt):
    t = int(cnt.sum())
    return float(cnt.max()) / t if t else float('nan')


def n_lay90(cnt):
    s = np.sort(np.asarray(cnt))[::-1]; t = int(s.sum())
    if t == 0:
        return 0
    return int(np.searchsorted(np.cumsum(s), 0.9 * t) + 1)


# ---------------------------------------------------------------- 母集団 (本体と共通)

def zmap():
    return {(r['page_id'], r['layer_id']): int(r['z_global'])
            for r in csv.DictReader(open(ZCSV, encoding='utf-8-sig'))}


def jobs(zz, fill, line):
    out = []
    for work in ('ako6', 'ako7'):
        d = json.load(open(f'{DEL}/{work}/koma/koma_manifest.json'))
        for p in d['pages']:
            pid = p['page']
            if (pid, work) not in fill:
                continue
            sel = [fr for fr in p['frames']
                   if int((fr.get('ink') or {}).get('layers_with_png') or 0) > 0]
            if not sel:
                continue
            fz = [(lid, path) for lid, path in sorted(fill[(pid, work)])]
            lz = [(lid, path, zz.get((pid, str(lid)), 0))
                  for lid, path in sorted(line.get((pid, work), []))]
            out.append((pid, work, fz, lz, sel))
    return out


def beta_fill_layers():
    """**作者の納品 `beta_manifest.json` の層**を (pid, work) -> [(lid, path)] で返す。

    L35(登録 2026-10-26 15:07、道1)のための標準の材料。
    `gated_layers()` の fill 側は **こちらの分類**(`keep4 >= KEEP4_FILL`)だったが、
    こちらは**作者が納品したベタ層そのもの**である。496層。

    **落としたものを理由別に印字する**(規律9)—— `beta_manifest` に在るのに
    層 PNG が無い行は、黙って消さずに数える。
    """
    out, cut = {}, {}
    for work in ('ako6', 'ako7'):
        man = json.load(open(f'{DEL}/{work}/fill/beta_manifest.json', encoding='utf-8'))
        for p in man['pages']:
            pid = p['page']
            for b in p.get('beta') or []:
                lid = str(b['layer_id'])
                path = f'{DEL}/{work}/all_layers/{pid}_L{lid}.png'
                if not os.path.exists(path):
                    # **理由を正しく書く**: これは納品漏れではない。層 PNG は
                    # **優先65頁しか発注されていない**ので、manifest の 1,180層のうち
                    # **684層は65頁の外**である(突き合わせ済み: 中 496 / 外 684)。
                    cut['優先65頁の外(層 PNG は65頁しか発注されていない)'] = \
                        cut.get('優先65頁の外(層 PNG は65頁しか発注されていない)', 0) + 1
                    continue
                out.setdefault((pid, work), []).append((int(lid), path))
    n = sum(len(v) for v in out.values())
    print(f'標準の材料 = 納品 beta {n} 層 / {len(out)} 頁  (**こちらの分類ではない**)')
    for k, v in cut.items():
        print(f'  落とした {v}  {k}')
    if not cut:
        print('  落とした 0  (**この点検は空ではない**: 上の 496 を数えている)')
    return out


def panels_and_regions(job, solid='solid'):
    """(pid, work, コマ番号, P, glab, gids, gstat, 線層の (lid, ink, z)) を順に返す。

    母集団の定義は `l15_zorder_occlusion.py` と一字一句同じ。

    `solid` は**標準の作り方**。既定の `'solid'` は `fill_mask(radius=4)` =
    半径4の opening(L12〜L18 のすべての数値がこれ)。`'cover'` は**生の墨の和集合**で、
    **`radius` が器具のどこにも残らない**(L35 = 道1。登録 2026-10-26 15:07)。
    **既定を変えていないので、L18 の数値はこの変更後も再現する。**
    """
    pid, work, fz, lz, sel = job
    F = None
    for lid, path in fz:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, solid)
    if F is None:
        return
    H, W = F.shape
    lines = []
    for lid, path, z in lz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (H, W):
            continue
        lines.append((lid, im < 128, z))
    for fr in sel:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            glab, gids, gstat = comps(f, 8)
            if not gids:
                continue
            sub = [(lid, ink[y0:y1, x0:x1], z) for lid, ink, z in lines]
            yield pid, work, ci, P, glab, gids, gstat, sub


# ---------------------------------------------------------------- 既知セル

def ring(H=200, r=40, lo=1.5, hi=3.5):
    yy, xx = np.mgrid[0:H, 0:H]
    c = H // 2
    dist = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    return (dist <= r), ((dist >= r + lo) & (dist <= r + hi)), ang


def kc16():
    """合成: 円盤を2本の弧(別々の層)で囲むと purity が弧の長さの比に一致する。"""
    R, rg, ang = ring()
    P = np.ones(R.shape, bool)
    ok = True
    print('KC16 器具: purity は弧の比に一致するか (許容 ±0.02)')
    for f in (0.5, 0.8, 0.95, 1.0):
        a1 = rg & (ang < f); a2 = rg & (ang >= f)
        lay = [(1, a1, 0), (2, a2, 1)]
        lab, _ = assign_top(lay, R.shape)
        vals, cnt = adj_sets(R, lab, P, ds=(D1,))[D1]
        pu = purity(cnt); nl = n_lay90(cnt); exp = max(f, 1 - f)
        bad = abs(pu - exp) > 0.02
        ok &= not bad
        print(f'  弧 {f:.2f}/{1-f:.2f}  purity {pu:.4f} (期待 {exp:.2f})  '
              f'n_lay90 {nl}  層数 {len(vals)}  adj {int(cnt.sum())}  '
              f'{"NG" if bad else "ok"}')
    # 1本だけなら 1.0
    lab, _ = assign_top([(1, rg, 0)], R.shape)
    vals, cnt = adj_sets(R, lab, P, ds=(D1,))[D1]
    bad = abs(purity(cnt) - 1.0) > 1e-9 or n_lay90(cnt) != 1
    ok &= not bad
    print(f'  1層のみ      purity {purity(cnt):.4f} (期待 1.0)  '
          f'n_lay90 {n_lay90(cnt)} (期待 1)  {"NG" if bad else "ok"}')
    return ok


def kc17():
    """合成: 同じ画素を2層が塗るとき、z_global の大きい方だけが数えられる。"""
    R, rg, ang = ring()
    P = np.ones(R.shape, bool)
    ok = True
    print('KC17 器具: z の割り当て (二重計数があれば分母が合わない)')
    # (a) 全く同じ画素を2層が塗る -> 上の層だけ、adj = 輪の画素数
    lab, keys = assign_top([(1, rg, 0), (2, rg, 7)], R.shape)
    vals, cnt = adj_sets(R, lab, P, ds=(D1,))[D1]
    tot = int(cnt.sum()); n_rg = int(rg.sum())
    top = keys[int(vals[int(np.argmax(cnt))]) - 1]
    bad = (tot != n_rg) or len(vals) != 1 or top != 2
    ok &= not bad
    print(f'  完全に重なる2層: 層数 {len(vals)} (期待 1)  数えた層 {top} (期待 2=上)  '
          f'adj {tot} = 輪 {n_rg} ? {"NG" if bad else "ok"}')
    # (b) 下の層が輪全体、上の層が半分 -> 半々、合計は輪の画素数のまま
    half = rg & (ang < 0.5)
    lab, keys = assign_top([(1, rg, 0), (2, half, 7)], R.shape)
    vals, cnt = adj_sets(R, lab, P, ds=(D1,))[D1]
    tot = int(cnt.sum()); pu = purity(cnt)
    bad = (tot != n_rg) or abs(pu - 0.5) > 0.02
    ok &= not bad
    print(f'  上が半分を覆う: adj {tot} = 輪 {n_rg} ?  purity {pu:.4f} (期待 0.50)  '
          f'{"NG" if bad else "ok"}')
    return ok


def kc18(limit=6):
    """データ経路: adj の総和が、別経路で数えた「∂R から d 以内の**門が通した層**の墨」と一致する。"""
    zz = zmap(); fill, line = arms('gated')
    js = jobs(zz, fill, line)[:limit]
    print(f'KC18 データ経路: adj の総和を別経路と突き合わせる (頁 {len(js)})')
    nreg = 0; bad = 0
    for job in js:
        for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
            if not sub:
                continue
            lab, _ = assign_top(sub, P.shape)
            U = np.zeros(P.shape, bool)
            for lid, ink, z in sub:
                U |= ink
            for gi in gids:
                R = glab == gi
                b = boundary(R)
                dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)
                got = adj_sets(R, lab, P, ds=DS, bnd=b)
                for d in DS:
                    want = int(((dt <= d) & P & U).sum())
                    if int(got[d][1].sum()) != want:
                        bad += 1
                nreg += 1
    print(f'  面 {nreg}  不一致 {bad}  {"NG" if bad else "ok"}')
    return bad == 0 and nreg > 0


def kc19():
    """データ経路: 面の数が前の走り(l15_zorder_occ の real 行)と一致する。"""
    want = {}
    for r in csv.DictReader(open(PREV)):
        if r['arm'] == 'real':
            want[r['work']] = want.get(r['work'], 0) + 1
    zz = zmap(); fill, line = arms('gated')
    got = {}
    nl = []
    for job in jobs(zz, fill, line):
        for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
            got[work] = got.get(work, 0) + len(gids)
            nl.append(len([1 for lid, ink, z in sub if int(ink.sum()) >= 500]))
    print('KC19 データ経路: 面の数が前の走りと一致するか')
    ok = True
    for w in sorted(set(want) | set(got)):
        bad = want.get(w, 0) != got.get(w, 0)
        ok &= not bad
        print(f'  {w}: 今回 {got.get(w,0)}  前回 {want.get(w,0)}  {"NG" if bad else "ok"}')
    print(f'  合計 今回 {sum(got.values())}  前回 {sum(want.values())}')
    nl.sort()
    print(f'  (縮退の再確認) コマ {len(nl)}  線層>=500px が 1枚以下のコマ '
          f'{sum(1 for x in nl if x <= 1)} ({sum(1 for x in nl if x<=1)/max(len(nl),1):.3f})  '
          f'中央 {nl[len(nl)//2] if nl else 0}')
    return ok


def kc20():
    """合成のみ: 対照の経路で平行移動量0にすると本来の腕と一致する (基盤 §3)。

    **実頁では走らせない** —— 平行移動量0の対照は本来の腕そのもので、見たことになる。
    """
    R, rg, ang = ring()
    P = np.ones(R.shape, bool)
    a1 = rg & (ang < 0.7); a2 = rg & (ang >= 0.7)
    lab, _ = assign_top([(1, a1, 0), (2, a2, 1)], R.shape)
    n, glab, gstat, _ = cv2.connectedComponentsWithStats(R.astype(np.uint8), 8)
    gi = 1
    patch, yy, xx, hh, ww = patch_of(glab, gi, gstat)
    R0 = shift_to(patch, yy, xx, P.shape)         # 平行移動量0 = 対照の経路
    same = bool((R0 == (glab == gi)).all())
    v0, c0 = adj_sets(glab == gi, lab, P, ds=(D1,))[D1]
    v1, c1 = adj_sets(R0, lab, P, ds=(D1,))[D1]
    bad = (not same) or purity(c0) != purity(c1) or n_lay90(c0) != n_lay90(c1) \
        or int(c0.sum()) != int(c1.sum())
    print('KC20 対照の経路 (合成のみ): 平行移動量0 は本来の腕と一致するか')
    print(f'  マスク一致 {same}  purity {purity(c0):.4f} / {purity(c1):.4f}  '
          f'n_lay90 {n_lay90(c0)} / {n_lay90(c1)}  adj {int(c0.sum())} / {int(c1.sum())}  '
          f'{"NG" if bad else "ok"}')
    # 置けない面は None を返すこと (コマより大きい面)
    small = np.zeros(R.shape, bool); small[2:-2, 2:-2] = True
    got = place(glab, gi, gstat, ~small, SEEDS[0], 'kc20', 1)
    print(f'  コマがほぼ無い場合に置かない: {"ok" if got is None else "NG"}')
    return (not bad) and got is None


def main():
    res = [('KC16', kc16()), ('KC17', kc17()), ('KC20', kc20()),
           ('KC18', kc18()), ('KC19', kc19())]
    print('=' * 72)
    for k, v in res:
        print(f'{k}: {"PASS" if v else "FAIL"}')
    if not all(v for _, v in res):
        print('既知セルが破れた。本体は起動しない(登録 §5)')
        sys.exit(1)
    print('既知セル全 PASS。本体を起動してよい')


if __name__ == '__main__':
    main()

# ---------------------------------------------------------------- T131
def _select():
    """`gated_layers()` と**同じ門**を、**落としを理由別に数えながら**通す。

    **`gated_layers()` は書き換えない。** こちらは数えるためだけに在る複製で、
    **両者が同じ集合を返すことを `kc_arms()` が画素ではなく鍵で確かめる。**

    なぜ複製するか: `gated_layers()` は**何を落としたか印字しない。**
    2026-10-26 に、**`cov >= 0.95` の門が 7,362行で1度も発火していない**ことを
    手で数えて初めて見つけた —— **発火しない門は、守っているように見えて何も守らない**
    (T134 (k))。**規律9 を器具の中の `continue` に当てる、の実装である。**
    """
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(KEEP4))}
    fill, line = {}, {}
    cut = {}

    def drop(why):
        cut[why] = cut.get(why, 0) + 1

    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            drop('folder か rendered でない'); continue
        try:
            if float(r['drawn_ratio']) <= 0:
                drop('drawn_ratio <= 0(= 層 PNG が無い。発注の範囲外)'); continue
        except (TypeError, ValueError):
            drop('drawn_ratio が読めない'); continue
        pd = r['preview_dark'].strip()
        if pd == '':
            drop('preview_dark が空'); continue
        if float(pd) < DARK_MIN:
            # **下限の向きの閾**だが、2026-10-26 に曝露を測って **0**
            # (`body_init 255 かつ一部保存` が優先65頁に 0層)
            drop(f'preview_dark < {DARK_MIN}'); continue
        kk = k4.get((r['page_id'], r['layer_id']))
        if kk is None:
            drop('keep4 に無い'); continue
        if float(kk['cov']) >= COV_MAX:
            drop(f'cov >= {COV_MAX}(全面)'); continue
        if topf(r['ancestors']) in DROP_TOP:
            drop(f'最上位フォルダーが {DROP_TOP}'); continue
        pid = r['page_id']
        w = work_of(pid, r.get('page_path'))
        pth = f'{BASE}/{w}/all_layers/{pid}_L{r["layer_id"]}.png'
        d = fill if float(kk['keep4']) >= KEEP4_FILL else line
        d.setdefault((pid, w), []).append((int(r['layer_id']), pth))
    return fill, line, cut


def arms(kind):
    """腕と標準を**1か所で**決める。**`kind` に既定値を置かない**(T104)。

    2026-10-26 に同じ数を2度取り違えた。機序は**「腕」が2か所で定義されていた**こと:

      `gated_layers()`          fill 300 / line 1,681            <- 器具の中
      `l18_propagate.py:355`    `--raw-beta-standard` の枝の中で
                                289層を腕から外す -> line 1,392   <- 呼ぶ側

    **呼び出し行を読まないと、どの走行がどちらの集合を使ったか分からなかった。**

    `kind`:
      `'gated'`         門だけを通した集合。**既定の標準**(L18 / L36 / L37 …)。
                        **fill と line は `keep4` の閾で二分するので交わらない。**
                        **「漏れが無い」のはこの二分の性質で、「素である」ことの性質ではない。**
      `'beta_standard'` 標準を**作者の納品 beta** に替える(L35 の道1)。
                        **標準の層が腕の側にも居るので、両側の層を腕から外す。**

    **名前は「どちらが正しいか」を含意させない** —— `'raw'` / `'scored'` は含意していた。
    **何をしたかを名前にする。**
    """
    if kind not in ('gated', 'beta_standard'):
        raise ValueError(f"kind は 'gated' か 'beta_standard'。既定値は置かない: {kind!r}")
    fill, line, cut = _select()
    nf, nl = sum(len(v) for v in fill.values()), sum(len(v) for v in line.values())
    print(f'arms(kind={kind!r})  門を通した 標準 {nf} / 腕 {nl}')
    for k, v in sorted(cut.items(), key=lambda t: -t[1]):
        print(f'  落とした {v:5d}  {k}')
    # **発火しなかった門も印字する**(T134 (k)。0 を黙らせない)
    for why in (f'cov >= {COV_MAX}(全面)', f'preview_dark < {DARK_MIN}',
                'preview_dark が空', 'keep4 に無い', f'最上位フォルダーが {DROP_TOP}'):
        if why not in cut:
            print(f'  落とした     0  {why}  (**この門は1度も発火していない**)')
    if kind == 'gated':
        both = {(k, lid) for k, v in fill.items() for lid, _ in v} &                {(k, lid) for k, v in line.items() for lid, _ in v}
        print(f'  標準と腕の交差 {len(both)}  '
              f'(**`keep4` の二分なので 0 のはず。0 でなければ読まない**)')
        return fill, line
    bfill = beta_fill_layers()
    bset = {(k, lid) for k, v in bfill.items() for lid, _ in v}
    n_before = sum(len(v) for v in line.values())
    line = {k: [(lid, p) for lid, p in v if (k, lid) not in bset] for k, v in line.items()}
    line = {k: v for k, v in line.items() if v}
    n_after = sum(len(v) for v in line.values())
    print(f'  **両側に居た層を腕から外した**: 腕 {n_before} -> {n_after} '
          f'(**{n_before - n_after} 層が標準の側に移った**)')
    print('  **だから beta_standard の腕は gated の腕と同じものではない。母集団が違う**')
    return bfill, line


def kc_arms():
    """**KC57**: `arms('gated')` が `gated_layers()` と**鍵まで一致**すること。

    **T131 の refactor の安全綱である** —— 置き換えが集合を変えていないことを、
    **呼び手を1本も移す前に**確かめる(KC53 と同じ形を、関数の水準で)。
    """
    f1, l1 = gated_layers()
    f2, l2 = arms('gated')

    # **鍵だけでなくパスまで比べる**(2026-10-26 に強めた)。
    # 鍵だけの比較では、**同じ層を違うファイルから読んでいても PASS する。**
    # 呼び手は `(lid, path)` の組を使うので、**組ごと一致しなければ
    # 「同じ集合を返す」とは言えない。**
    #
    # **2026-10-27 にもう一段強めた。** 呼び手 25本を機械的に置き換える直前に、
    # **この綱が何を比べていないか**を数えたら2つ在った:
    #   1. `sorted(v)` で**順序を潰していた。** 呼び手は `fill[k]` を**そのまま回す**ので、
    #      順序が違えば「同じ集合」でも出力が変わりうる(z 順の先勝ち・割り当ての先勝ち)。
    #   2. `if v` で**空リストの頁を落としていた。** 片方に空で在る頁は映らない。
    # **置き換えが頼っている性質は「同じ集合」ではなく「同じ列」である。**
    # **綱は、頼っている性質を比べなければ綱ではない。**
    def cmp(a, b):
        """**順序まで**比べる。空リストの頁も鍵として比べる。差の理由を返す。"""
        ka, kb = set(a), set(b)
        if ka != kb:
            return f'鍵が違う(片側だけ {len(ka ^ kb)}頁。例 {sorted(ka ^ kb)[:2]})'
        bad = [k for k in ka if a[k] != b[k]]
        if not bad:
            return None
        k = sorted(bad)[0]
        if sorted(a[k]) == sorted(b[k]):
            return f'**順序が違う** {len(bad)}頁(集合は同じ。例 {k})'
        return f'組が違う {len(bad)}頁(例 {k})'

    d_f, d_l = cmp(f1, f2), cmp(l1, l2)
    ok = d_f is None and d_l is None
    n = lambda d: sum(len(v) for v in d.values())
    print(f"KC57 arms('gated') == gated_layers() (**(層, パス) の組・順序・空の頁まで**):  "
          f'標準 {n(f1)} vs {n(f2)}  腕 {n(l1)} vs {n(l2)}  '
          f'{"PASS" if ok else "**FAIL**"}')
    for nm, d in (('標準', d_f), ('腕', d_l)):
        if d:
            print(f'  {nm}: {d}')

    # **この点検が発火することを、同じ走行の中で確かめる**(2026-10-26 の awk の罠。
    # **一致しえない比較の PASS は、「同じ」ではない**)。
    # 腕の1頁だけ順序を入れ替えた複製を作り、**cmp がそれを順序の差として挙げる**ことを見る。
    probe = {k: list(v) for k, v in l1.items()}
    pk = sorted(k for k, v in probe.items() if len(v) > 1)
    if not pk:
        print('  **自己点検ができない**(層が2つ以上在る頁が無い)。PASS を読まない')
        return False
    probe[pk[0]] = list(reversed(probe[pk[0]]))
    fired = cmp(l1, probe)
    print(f'  自己点検(1頁の順序を反転した複製): '
          f'{"**順序の差として発火** " + fired if fired and "順序" in fired else "**発火しない。綱が死んでいる**"}')
    return ok and bool(fired) and '順序' in (fired or '')
