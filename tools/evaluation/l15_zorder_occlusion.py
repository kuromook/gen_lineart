# -*- coding: utf-8 -*-
"""z順で「前にある遮蔽物」を決めて測る (診断。棒なし)。

前回の代用 `OCC ∧ ¬F` は**循環**だった(重なりが無い面では寄与 +0.000、大きく重なる面で
+0.47/+0.55・支持ちょうど 1.000 = 面の輪郭を引き写していた)。今回は納品された
`layer_z_order_priority65.csv` の `z_global`(0 が一番下)で決める。**F を参照しない。**

  遮蔽物の候補 = 障壁にも母集団にも入らなかった層のうち visibility=1 で /sketch set でないもの
  OCC_front(Lf) = ∪ cover(L)  over  z_global(L) > z_global(Lf)
  面 R はその出所の面層 Lf(重なりが最大の層)の z を使う

**先に作者の言明を試験する**: |R ∧ OCC_front| / |R| がほぼ 0 か。
「手前に来るものの下は描いていない」が正しければ 0 に近いはず。これは腕ではなく観察。

  W6 = W2 + OCC_front       W7 = W3 + OCC_front       対照は W7 で取る

**2026-10-26 の直し (T128)**: 遮蔽物の選別が `visibility != '1'` の等値比較で、
**可視の層を落としていた**(`visibility` はビット列。1の位が表示、2の位がマスク)。
ここでは 128層 —— **トーン、つまり遮蔽物の代表** —— が落ちていた。
**この走行の数は、遮蔽物が約2割足りない下で出ている。** 誤りの向きは一方向で、
**遮蔽で説明できる量は過小評価**されている。したがって

    **`doc/work_log.md` の「作者の言明は正しい(重なり中央 0.0000)」は、
    遮蔽物を欠いた母集団の上の数である。印を付けて再走行待ちにした(消していない)。**

**先に書く: 再走行で重なりが上がっても、それは作者の言明への反証ではない** ——
**何を描いているかは作者が決める**(規律6)。上がるのは器具が見ていなかった分である。
"""
import csv, collections, json, os, sys, statistics as st
import zlib
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, A, topf
from l14_arms import acc_add
from l15_known_cells import boundary
from l15_occlusion import excluded_layers
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)
from l15_author_occluder_gates import OCC_KINDS, occ_predicate  # 遮蔽物の述語は1か所

DEL  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
ZCSV = '/home/sh1/deepl/lineart-fill-region/inbox/layer_z_order_priority65.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_zorder_occ'
SEED = 20261026
# 遮蔽物の腕。**既定は `O_all`**(= 2026-10-26 の実装。再現の基準を動かさない)。
# `--occ` で選ぶ。出力は腕ごとに別のファイルに書く(**既定だけ無印**)。
OCC = 'O_all'
ANC = False
MIN_FRAC, MAX_FRAC = 0.002, 0.50


def best(R, lab, stats):
    n = int(R.sum()); v = lab[R]; v = v[v > 0]
    if v.size == 0:
        return 0.0, 0.0
    vals, cnt = np.unique(v, return_counts=True)
    k = int(vals[int(np.argmax(cnt))]); inter = int(cnt.max()); ar = int(stats[k, A])
    return inter / (ar + n - inter), ar / n


def occluder_cover(path, mask_path, shape):
    """**遮蔽物1枚の遮蔽範囲。** マスクが在れば**掛けた後**を返す(O2' / O3 の経路)。

    向きは **`<128` が見せる側**(KC55 でマスクの向きを確かめた。
    反転したマスクは遮蔽を逆に測るので、**ここを間違えると数が逆向きに動く**)。
    返すのは (ink, None) か (None, 落とした理由)。**理由は呼び手が数える**(規律9)。

    作者の定義の「マスクが有効なら適用後の範囲が遮蔽範囲」はここに当たる。
    **有効フラグそのものは分からない**(供給側に無効にした見本が無い)ので、
    **マスクを渡された層は有効として扱う** —— 登録に書いた但し書きである。
    """
    H, W = shape
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, '遮蔽物の PNG が読めない'
    if im.shape != (H, W):
        return None, '遮蔽物の形が頁と違う'
    ink = im < 128
    if mask_path:
        mk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mk is None:
            return None, 'マスクの PNG が読めない'
        if mk.shape != (H, W):
            return None, 'マスクの形が頁と違う'
        ink = ink & (mk < 128)
    return ink, None


def front_by_z(occ, zs, shape):
    """**z ごとの `OCC_front`**(その z より手前の遮蔽物の和集合)を返す。

    2026-10-27 に `one_page` から切り出した。**KC56 がこの関数を呼ぶ** ——
    合成の既知セルは**規則を書き写さず、本体と同じ関数を試験する**(T132、KC48 と同じ作法)。
    切り出す前は `one_page` の中に在ったので、**合成の入力を通す口が無かった。**

    `occ` = [(lid, path, z)](順不同)、`zs` = 面層の z(**降順**)、`shape` = (H, W)。
    **読み込みで落ちた遮蔽物を理由別に数えて返す**(規律9。もとは黙って `continue` していた)。
    """
    H, W = shape
    occ = sorted(occ, key=lambda t: -t[2])
    cum = np.zeros((H, W), bool); front = {}; drop = collections.Counter()
    oi = 0
    for z in zs:
        while oi < len(occ) and occ[oi][2] > z:
            ink, why = occluder_cover(occ[oi][1],
                                      occ[oi][3] if len(occ[oi]) > 3 else None, (H, W))
            if ink is None:
                drop[why] += 1
            else:
                acc_add(cum, None, ink, 'cover')
            oi += 1
        front[z] = cum.copy()
    return front, drop


def one_page(job):
    pid, work, fl, ll, occ, frames = job
    solids = []          # (z, mask) 面層ごと
    F = G = B = None
    for lid, path, z in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool)
        s = np.zeros(ink.shape, bool)
        acc_add(s, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
        F |= s; solids.append((z, s))
    if F is None:
        return [], collections.Counter()
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    H, W = F.shape
    # z の高い遮蔽物から積み上げ、面層の z ごとの OCC_front を作る
    zs = sorted({z for z, _ in solids}, reverse=True)
    front, odrop = front_by_z(occ, zs, (H, W))   # 2026-10-27: 切り出した(KC56 が同じ関数を呼ぶ)
    kl = cv2.imread(f'{DEL}/{work}/koma/{pid}_koma_lines.png', cv2.IMREAD_GRAYSCALE)
    KL = (kl > 127) if (kl is not None and kl.shape == (H, W)) else np.zeros((H, W), bool)
    rng = np.random.default_rng(SEED + zlib.crc32(pid.encode()) % 100000)
    # **`hash(str)` は過程ごとに変わる**ので種に使えない(2026-10-25、罠 T97)。
    # crc32 は過程に依らない。直す前の走行の帰無値は再現しない
    rows = []
    for fr in frames:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2); x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            b = B[y0:y1, x0:x1]; g = G[y0:y1, x0:x1]; kll = KL[y0:y1, x0:x1]
            glab, gids, gstat = comps(f, 8)
            if not gids:
                continue
            ph, pw = P.shape
            glob = {}
            for v, bar in (('W0', b), ('W1', b | g), ('W2', b | g | kll)):
                nn, lb, stt, _ = cv2.connectedComponentsWithStats((P & ~bar).astype(np.uint8), 4)
                glob[v] = (lb, stt, bar)
            for gi in gids:
                R = glab == gi; npx = int(R.sum())
                bn = boundary(R); nb = max(int(bn.sum()), 1)
                # 出所の面層 = 重なりが最大
                zbest, cbest = zs[-1], -1
                for z, s in solids:
                    c = int((R & s[y0:y1, x0:x1]).sum())
                    if c > cbest:
                        cbest, zbest = c, z
                OF = front[zbest][y0:y1, x0:x1]
                rec = dict(pid=pid, work=work, comp=ci, region=gi, area=npx,
                           panel_area=int(P.sum()), arm='real', src_z=zbest,
                           src_share=round(cbest / max(npx, 1), 4),
                           occf_cover=round(int((R & OF).sum()) / max(npx, 1), 4))
                for v in ('W0', 'W1', 'W2'):
                    lb, stt, bar = glob[v]
                    iou, ratio = best(R, lb, stt)
                    rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                    rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                        (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    ).sum()) / nb, 4)
                others = f & ~R
                for v, bar in (('W6', b | g | kll | OF),
                               ('W3', b | g | kll | others),
                               ('W7', b | g | kll | others | OF)):
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~bar).astype(np.uint8), 4)
                    iou, ratio = best(R, lb, stt)
                    rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                    rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                        (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    ).sum()) / nb, 4)
                rows.append(rec)
                yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
                hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
                if ww + 2 >= pw or hh + 2 >= ph:
                    continue
                patch = (glab[yy:yy + hh, xx:xx + ww] == gi); pn = int(patch.sum())
                for _ in range(24):
                    nx = int(rng.integers(1, pw - ww - 1)); ny = int(rng.integers(1, ph - hh - 1))
                    if abs(nx - xx) + abs(ny - yy) < 0.10 * (pw + ph):
                        continue
                    Rs = np.zeros(P.shape, bool); Rs[ny:ny + hh, nx:nx + ww] = patch
                    if int((Rs & P).sum()) < 0.9 * pn:
                        continue
                    bar = b | g | kll | (f & ~Rs) | OF
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~bar).astype(np.uint8), 4)
                    iou, ratio = best(Rs, lb, stt)
                    rows.append(dict(pid=pid, work=work, comp=ci, region=gi, area=npx,
                                     panel_area=int(P.sum()), arm='shuffled',
                                     occf_cover=round(int((Rs & OF).sum()) / max(pn, 1), 4),
                                     W7_iou=round(iou, 4), W7_ratio=round(ratio, 3)))
                    break
    return rows, odrop


def main():
    global OCC, ANC
    ANC = '--anc' in sys.argv   # **祖先のフォルダーの表示も見る(T143。これが作者の定義)**
    for i_, x_ in enumerate(sys.argv):
        if x_ == '--occ':
            OCC = sys.argv[i_ + 1]
    if OCC not in OCC_KINDS:
        sys.exit(f'--occ は {OCC_KINDS} のどれか。**既定に落とさない**(T104)')
    tag = '' if (OCC == 'O_all' and not ANC) else f'_{OCC}' + ('_anc' if ANC else '')
    os.makedirs(OUT, exist_ok=True)
    zz = {(r['page_id'], r['layer_id']): int(r['z_global'])
          for r in csv.DictReader(open(ZCSV, encoding='utf-8-sig'))}
    vis = {(r['page_id'], r['layer_id']): r
           for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    fill, line = arms('gated')
    ex, _ = excluded_layers()
    cut = collections.Counter()
    occs = {}
    # **遮蔽物の述語は `l15_author_occluder_gates.occ_predicate` の1か所**
    # (T131: 選別の入口を2つ置かない)。既定の `O_all` は 2026-10-26 の実装と同じ門で、
    # **理由の文字列も同じにしてある** —— でないと再現の基準が壊れる。
    # `--occ legacy612` が **KC53**(旧来の等値比較の再現)、
    # `--occ O_wide` が **主値**(作者の定義: 乗算は遮蔽物ではない)。
    keep_fn = occ_predicate(OCC, pages={r['page_path'] for r in vis.values()},
                            ancestors=ANC)
    for k, v in ex.items():
        keep = []
        for lid, path, name, reason in v:
            r = vis.get((k[0], str(lid)))
            if r is None:
                cut['在庫に無い'] += 1; continue
            # **T128** (2026-10-26。基盤から。抽出側が M の選別を調べて判明):
            # `visibility` は**ビット列**で、**1の位が表示、2の位がマスク**である。
            # ここは `vi != '1'` の**等値比較**だったので、**値 3/5/7 の層 ——
            # 可視である層 —— を「可視でない」として落としていた。**
            # 在庫 7,362層のうち 2,467層が可視で落ち、うち **2,459層は可視かつ
            # マスク持ち = トーン**。**遮蔽の解析から、遮蔽物の代表が抜けていた。**
            #
            # **落としの理由は発火した条件から導く。横に書き添えない** ——
            # 書き添えた「可視でない」は、2,467層について**嘘を印字していた**。
            # **誤った理由の下で数えられた落としは、数えられていない落としより悪い**
            # (数が、点検が働いている証拠に見える)。
            ok_, why = keep_fn(r['page_path'], lid, r['visibility'],
                               r['ancestors'], r['opacity'])
            if not ok_:
                cut[why] += 1; continue
            z = zz.get((k[0], str(lid)))
            if z is None:
                cut['z順に無い'] += 1; continue
            keep.append((lid, path, z))
        occs[k] = keep
    print(f'遮蔽物の選別 (z順で前後を決める。名前では切らない)  **腕 {OCC}'
          + ('  祖先も見る(T143)**' if ANC else '  祖先は見ない**') + ')')
    print(f'  通した {sum(len(v) for v in occs.values())}')
    for k, v in cut.most_common():
        print(f'  落とした {v:5d}  {k}')
    jobs = []
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
            fz = [(lid, path, zz.get((pid, str(lid)), 0))
                  for lid, path in sorted(fill[(pid, work)])]
            jobs.append((pid, work, fz, sorted(line.get((pid, work), [])),
                         occs.get((pid, work), []), sel))
    print(f'  頁 {len(jobs)}')
    rows = []; odrop = collections.Counter()
    with Pool(4) as pool:
        for i, (r, d) in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r; odrop.update(d)
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  行 {len(rows)}', flush=True)
    keys = ['pid', 'work', 'comp', 'region', 'area', 'panel_area', 'arm', 'src_z',
            'src_share', 'occf_cover'] + \
           [f'{v}_{k}' for v in ('W0', 'W1', 'W2', 'W6', 'W3', 'W7')
            for k in ('iou', 'ratio', 'sup')]
    with open(f'{OUT}/per_region{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)
    real = [r for r in rows if r['arm'] == 'real']
    print(f'-> {OUT}/per_region{tag}.csv  面 {len(real)}')
    # **遮蔽物の読み込みで落ちた分を理由別に印字する**(2026-10-27。規律9)。
    # 切り出す前は黙って `continue` していた。**発火していない理由も印字する**(T134(k))。
    for r_ in ('遮蔽物の PNG が読めない', '遮蔽物の形が頁と違う',
               'マスクの PNG が読めない', 'マスクの形が頁と違う'):
        n_ = odrop.get(r_, 0)
        print(f'   落とした {n_:5d}  {r_}'
              + ('  (**この理由は1度も発火していない**)' if n_ == 0 else ''))
    print('=' * 88)
    print('【先に観察】作者の言明「手前に来るものの下は描いていない」の試験')
    print('  |R ∧ OCC_front| / |R|  —— 0 に近ければ言明どおり。**F を参照していない**')
    for w in ('ako6', 'ako7'):
        s = [r for r in real if r['work'] == w]
        v = sorted(r['occf_cover'] for r in s)
        print(f'  {w}: 面 {len(s)}  中央 {st.median(v):.4f}  平均 {sum(v)/len(v):.4f}  '
              f'90分位 {v[int(.9*len(v))]:.4f}  >0.5 の面 {sum(1 for x in v if x>0.5)} '
              f'({sum(1 for x in v if x>0.5)/len(v):.3f})  >0.1 {sum(1 for x in v if x>0.1)/len(v):.3f}')
    print('\n' + '=' * 88)
    print('腕 (面ごと。棒なし)')
    print(f'{"作品":5} {"障壁":5} {"面":>5} {"IoU中央":>8} {"IoU>=0.5":>9} {"IoU>=0.75":>10} '
          f'{"漏れ":>8} {"支持中央":>9}')
    for w in ('ako6', 'ako7'):
        s = [r for r in real if r['work'] == w]
        if not s:
            continue
        for v in ('W0', 'W1', 'W2', 'W6', 'W3', 'W7'):
            iou = [r[f'{v}_iou'] for r in s]
            print(f'{w:5} {v:5} {len(s):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in s if r[f"{v}_ratio"] > 2)/len(s):8.4f} '
                  f'{st.median([r[f"{v}_sup"] for r in s]):9.4f}')
        c = [r for r in rows if r['work'] == w and r['arm'] == 'shuffled']
        if c:
            iou = [r['W7_iou'] for r in c]
            print(f'{w:5} {"対照W7":5} {len(c):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in c if r["W7_ratio"] > 2)/len(c):8.4f}')
    print('\n循環の点検: 遮蔽物が面に重なる量で層別 (前回はここで +0.000 と +0.47 に割れた)')
    for w in ('ako6', 'ako7'):
        s0 = [r for r in real if r['work'] == w]
        for lab, lo, hi in (('<0.01', 0, 0.01), ('0.01-0.2', 0.01, 0.2),
                            ('0.2-0.5', 0.2, 0.5), ('>=0.5', 0.5, 1.01)):
            s = [r for r in s0 if lo <= r['occf_cover'] < hi]
            if not s:
                continue
            f2 = sum(1 for r in s if r['W2_iou'] >= 0.5) / len(s)
            f6 = sum(1 for r in s if r['W6_iou'] >= 0.5) / len(s)
            print(f'  {w} occf {lab:9} 面 {len(s):5d}  W2 {f2:.3f}  W6 {f6:.3f}  '
                  f'差 {f6-f2:+.3f}  W6支持中央 {st.median([r["W6_sup"] for r in s]):.3f}')


if __name__ == '__main__':
    main()
