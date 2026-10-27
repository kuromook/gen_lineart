# -*- coding: utf-8 -*-
"""L31: **L の面と作者の面がずれる場合の頻度**。登録は `doc/work_log.md § L31 事前登録`。

**この測定は L18 / L17 の判定に入らない。腕も対照も床も動かさない。**
母集団の記述統計であって、棒を当てる対象ではない。

**作者の答の代わりにならない**(規律6)。3行目が稀だったときに言えるのは
「この場合は母集団の中で稀なので、向きの判定を左右しない」までで、
**「作者は1つと見る」ではない。**

  ./venv/bin/python tools/evaluation/l31_region_unit.py --known-cells
  ./venv/bin/python tools/evaluation/l31_region_unit.py
"""
import argparse, collections, csv, os, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from fill_mask import fill_mask
from l14_known_cells import A, MINREG, comps
from l17_known_cells import DEL, MAX_FRAC, MIN_FRAC, jobs, zmap
from pid_work import work_of
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l31_region_unit'
INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
RADIUS = 4.0            # 器具が使っている opening の半径 (fill_mask の既定)
THIN_PX = 16            # **事前登録**: 薄い隙間 = 原寸で 16px 以下
MIN_PX, MIN_SHARE = 50, 0.01    # **事前登録**: 寄与とみなす下限
GAPS = (4, 8, 16, 32, 64)

# ベタ層の名前から材質を引く。**ここに無い名前は「不明」で、既知に混ぜない**(規律9)
MATERIAL = {'線画 hair': '髪', '線画 HEAD hair': '髪', 'hair black': '髪',
            'effect': '効果',
            '線画 cloth': '服', '線画 BODY cloth': '服',
            '線画 skin': '肌', '肌グラデ': '肌',
            'bg base': '背景',
            '描き文字　　白フチ': '描き文字'}


def layer_names():
    return {(r['page_id'], r['layer_id']): r['name']
            for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}


def min_gap(lab, ids):
    """成分どうしの**最小の隙間**(px)。1つなら None。"""
    best = None
    for i in ids:
        other = np.isin(lab, [j for j in ids if j != i])
        if not other.any():
            continue
        d = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3)
        g = float(d[lab == i].min())
        best = g if best is None else min(best, g)
    return best


def thin_pieces(lab, ids):
    """**薄い隙間で割れた側にある成分**の id。各成分について、他のどれかとの
    距離が THIN_PX 以下なら数える(組ではなく成分で数えるのは画素重みのため)。"""
    out = []
    for i in ids:
        other = np.isin(lab, [j for j in ids if j != i])
        if not other.any():
            continue
        d = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3)
        if float(d[lab == i].min()) <= THIN_PX:
            out.append(i)
    return out


def one_page(job):
    pid, work, fz, lz, sel = job
    names = NAMES
    drop = collections.Counter()
    rows_c, rows_l = [], []
    fm, raw = {}, {}
    for lid, path in fz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            drop['層: PNG が読めない'] += 1
            continue
        ink = im < 128
        if not ink.any():
            drop['層: 墨が無い'] += 1
            continue
        raw[lid] = ink
        fm[lid] = fill_mask(ink, radius=RADIUS)
        if not fm[lid].any():
            drop['層: opening で全部消えた'] += 1
    lids = [l for l in fm if fm[l].any()]
    if not lids:
        drop['頁: 残ったベタ層が無い'] += 1
        return rows_c, rows_l, drop
    H, W = fm[lids[0]].shape
    F = np.zeros((H, W), bool)
    for l in lids:
        F |= fm[l]
    for fr in sel:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            drop['コマ: マスクが読めない / 形が合わない'] += 1
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W:
                drop['コマ: 小さすぎ'] += 1
                continue
            if a > MAX_FRAC * H * W:
                drop['コマ: 大きすぎ'] += 1
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            glab, gids, gstat = comps(f, 8)
            drop['成分: 面積の床 (500px) 未満'] += int(
                cv2.connectedComponentsWithStats(f.astype(np.uint8), 8)[0] - 1 - len(gids))
            if not gids:
                continue
            sl = {l: (fm[l][y0:y1, x0:x1] & P) for l in lids}
            # ---------------- 成分ごと: (A) 2行目
            for gi in gids:
                C = (glab == gi)
                ar = int(gstat[gi, A])
                hit, hit_loose = [], []
                for l in lids:
                    px = int((sl[l] & C).sum())
                    if px <= 0:
                        continue
                    hit_loose.append(l)
                    if px >= MIN_PX and px >= MIN_SHARE * ar:
                        hit.append(l)

                def mats(ls):
                    k = {MATERIAL[names.get((pid, str(l)), '')] for l in ls
                         if names.get((pid, str(l)), '') in MATERIAL}
                    unk = [l for l in ls if names.get((pid, str(l)), '') not in MATERIAL]
                    return k, unk
                k, unk = mats(hit)
                kl, unkl = mats(hit_loose)
                rows_c.append(dict(
                    pid=pid, work=work, comp=ci, region=gi, area=ar,
                    n_lay=len(hit), n_lay_loose=len(hit_loose),
                    n_mat=len(k), n_mat_loose=len(kl),
                    n_unknown=len(unk), n_unknown_loose=len(unkl),
                    case2=int(len(k) >= 2),
                    case2_loose=int(len(kl) >= 2),
                    undet=int(len(k) < 2 and len(hit) >= 2 and len(unk) >= 1),
                    mats='+'.join(sorted(k))))
            # ---------------- 層ごと: (B) 3行目 と (C) 4行目
            for l in lids:
                own = sl[l]
                if not own.any():
                    drop['層x コマ: その コマに墨が無い'] += 1
                    continue
                lo, io_, so = comps(own, 8)
                rawc = raw[l][y0:y1, x0:x1] & P
                lr, ir, _sr = comps(rawc, 8)
                g = min_gap(lo, io_) if len(io_) >= 2 else None
                th = thin_pieces(lo, io_) if len(io_) >= 2 else []
                rows_l.append(dict(
                    pid=pid, work=work, comp=ci, lid=l,
                    name=names.get((pid, str(l)), ''),
                    material=MATERIAL.get(names.get((pid, str(l)), ''), ''),
                    px_open=int(own.sum()), px_raw=int(rawc.sum()),
                    n_open=len(io_), n_raw=len(ir),
                    min_gap=-1 if g is None else round(g, 2),
                    thin=int(bool(th)),
                    px_thin=int(sum(int(so[i, A]) for i in th)),
                    opening_split=int(len(io_) > len(ir))))
    return rows_c, rows_l, drop


NAMES = None


def init():
    global NAMES
    NAMES = layer_names()


# ------------------------------------------------------------------ 既知セル

def kc35():
    """1層が隙間で割れたとき、**薄い(8px)か厚い(64px)か**を分けられるか。"""
    ok = True
    for gap, want in ((8, True), (64, False)):
        im = np.zeros((300, 300 + gap), bool)
        im[100:200, 20:120] = True
        im[100:200, 120 + gap:220 + gap] = True
        fmk = fill_mask(im, radius=RADIUS)
        lab, ids, _ = comps(fmk, 8)
        g = min_gap(lab, ids)
        got = len(ids) == 2 and g is not None and g <= THIN_PX
        print(f'  KC35 隙間 {gap}px: 成分 {len(ids)} (期待 2)  最小の隙間 {g}  '
              f'薄い={got} (期待 {want})  {"ok" if got == want else "FAIL"}')
        ok &= got == want
    return ok


def kc36():
    """接する別材質2層を (A) に数え、**離れていれば数えない**か。"""
    ok = True
    for dx, want in ((0, True), (50, False)):
        a = np.zeros((300, 400), bool); a[100:200, 20:120] = True
        b = np.zeros((300, 400), bool); b[100:200, 120 + dx:220 + dx] = True
        fa, fb = fill_mask(a, radius=RADIUS), fill_mask(b, radius=RADIUS)
        lab, ids, stats = comps(fa | fb, 8)
        got = False
        for gi in ids:
            C = (lab == gi); ar = int(stats[gi, A])
            k = set()
            for m, nm in ((fa, '線画 hair'), (fb, '線画 cloth')):
                px = int((m & C).sum())
                if px >= MIN_PX and px >= MIN_SHARE * ar:
                    k.add(MATERIAL[nm])
            if len(k) >= 2:
                got = True
        print(f'  KC36 ずれ {dx}px: 成分 {len(ids)}  別材質が1成分に={got} '
              f'(期待 {want})  {"ok" if got == want else "FAIL"}')
        ok &= got == want
    return ok


def kc37():
    """`fill_mask(radius=4)` の校正: **幅6px の連結部は切れ、幅12px は残る**。"""
    ok = True
    for w, want in ((6, 2), (12, 1)):
        im = np.zeros((300, 400), bool)
        im[100:200, 20:150] = True
        im[100:200, 250:380] = True
        y = 150 - w // 2
        im[y:y + w, 150:250] = True
        n_raw = len(comps(im, 8)[1])
        lab, ids, _ = comps(fill_mask(im, radius=RADIUS), 8)
        print(f'  KC37 連結部 {w}px: 生 {n_raw} 成分 -> opening 後 {len(ids)} '
              f'(期待 {want})  {"ok" if len(ids) == want else "FAIL"}')
        ok &= len(ids) == want
    return ok


def known_cells():
    print('L31 既知セル (登録 §5。**1つでも落ちたら本体を回さない**)')
    r = {'KC35': kc35(), 'KC36': kc36(), 'KC37': kc37()}
    print('=' * 72)
    for k, v in r.items():
        print(f'{k}: {"PASS" if v else "FAIL"}')
    if all(r.values()):
        print('既知セル全 PASS。本体を起動してよい')
        return 0
    print('**落ちた。本体を回さない**')
    return 1


# ------------------------------------------------------------------ 本体

def pct(a, b):
    return f'{a / b:.4f}' if b else 'n/a'


def report(rows_c, rows_l, drop, limit):
    print('\n' + '=' * 92)
    print('**この測定は L18 / L17 の判定に入らない。腕も対照も床も動かさない。**')
    print('**作者の答の代わりにならない**(規律6)。(B) が稀でも言えるのは')
    print('  「この場合は母集団の中で稀なので、向きの判定を左右しない」までで、')
    print('  **「作者は1つと見る」ではない**')
    print()
    print('選別 (規律9。落としたものを理由別に)')
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        if v:
            print(f'  {k}: {v}')
    for w in ('ako6', 'ako7'):
        cc = [r for r in rows_c if r['work'] == w]
        ll = [r for r in rows_l if r['work'] == w]
        if not cc:
            continue
        ar = sum(r['area'] for r in cc)
        print('\n' + '-' * 92)
        print(f'プール {w}   成分 {len(cc)}   墨 {ar:,}px   層x コマ {len(ll)}')
        n2 = sum(r['case2'] for r in cc)
        p2 = sum(r['area'] for r in cc if r['case2'])
        nu = sum(r['undet'] for r in cc)
        pu = sum(r['area'] for r in cc if r['undet'])
        n2l = sum(r['case2_loose'] for r in cc)
        print(f'  (A) 2行目 = 別材質が1成分に入っている')
        print(f'      成分数 {n2}/{len(cc)} = {pct(n2, len(cc))}   '
              f'画素重み {p2:,}/{ar:,} = {pct(p2, ar)}')
        print(f'      (規律9) 2層以上だが材質が不明を含む: 成分数 {nu}/{len(cc)} = '
              f'{pct(nu, len(cc))}  画素重み {pct(pu, ar)}  **既知に混ぜない**')
        print(f'      (併記) >=1px の緩い寄与: 成分数 {n2l}/{len(cc)} = {pct(n2l, len(cc))}')
        nb = sum(r['thin'] for r in ll)
        pb = sum(r['px_thin'] for r in ll)
        po = sum(r['px_open'] for r in ll)
        print(f'  (B) 3行目 = 1層の墨が**薄い隙間 (<= {THIN_PX}px)** で割れている')
        print(f'      層x コマ {nb}/{len(ll)} = {pct(nb, len(ll))}   '
              f'画素重み {pb:,}/{po:,} = {pct(pb, po)}')
        d = collections.Counter()
        for r in ll:
            g = r['min_gap']
            if g < 0:
                d['割れていない'] += 1
            else:
                for t in GAPS:
                    if g <= t:
                        d[f'<= {t}px'] += 1
                        break
                else:
                    d['> 64px'] += 1
        print('      (併記) 最小の隙間の分布: ' +
              '  '.join(f'{k} {v}' for k, v in sorted(d.items())))
        # **(B) の中身を割る**(併記。棒は登録どおり (B) そのものに当てる)。
        # 薄い隙間で割れていても、**生の墨では繋がっていた**なら、
        # それは作者の白抜きではなく **opening が作った隙間**(4行目)である。
        # 3行目(作者の白抜き)に当たるのは「生の墨でも割れていた」側だけ
        th = [r for r in ll if r['thin']]
        by_i = [r for r in th if r['opening_split']]
        by_a = [r for r in th if not r['opening_split']]
        print(f'      (併記) その内訳 —— **器具が作った隙間** '
              f'{len(by_i)}/{len(th) or 1} '
              f'(画素 {sum(r["px_thin"] for r in by_i):,}) / '
              f'**生の墨でも割れていた (= 作者の隙間)** {len(by_a)}/{len(th) or 1} '
              f'(画素 {sum(r["px_thin"] for r in by_a):,})')
        print(f'      3行目に当たるのは後者だけ: 層x コマ '
              f'{pct(len(by_a), len(ll))}  '
              f'画素重み {pct(sum(r["px_thin"] for r in by_a), po)}')
        nc = sum(r['opening_split'] for r in ll)
        pc = sum(r['px_open'] for r in ll if r['opening_split'])
        print(f'  (C) 4行目 = opening (radius {RADIUS:g}) が生の墨より成分を増やした')
        print(f'      層x コマ {nc}/{len(ll)} = {pct(nc, len(ll))}   '
              f'画素重み {pc:,}/{po:,} = {pct(pc, po)}')
    if limit:
        print('\n**--limit の煙試験。棒に当ててはいけない**')
        return
    print('\n' + '=' * 92)
    print(f'棒 (登録 §3。**数を見る前に決めた**。薄い = <= {THIN_PX}px)')
    v = {}
    for w in ('ako6', 'ako7'):
        ll = [r for r in rows_l if r['work'] == w]
        if not ll:
            continue
        po = sum(r['px_open'] for r in ll) or 1
        v[w] = (sum(r['thin'] for r in ll) / len(ll),
                sum(r['px_thin'] for r in ll) / po)
    for w, (a, b) in v.items():
        print(f'  {w}: (B) 層x コマ {a:.4f}  画素重み {b:.4f}')
    print('  **棒は登録どおり (B) そのものに当てる。** 上の「内訳」は併記で、')
    print('  **(B) のうち器具が作った隙間の分は 3行目(作者の白抜き)ではない** ——')
    print('  3行目だけを見たい読み手のために出すが、棒を動かすために使わない')
    if v and all(a < 0.05 and b < 0.05 for a, b in v.values()):
        print('  -> **稀**: 2行目だけで向きが決まる。1,186 は作者の面の数の**下限**。')
        print('     **作者に聞かない**(ただし「作者は1つと見る」と言ったのではない)')
    elif v and any(a >= 0.20 or b >= 0.20 for a, b in v.values()):
        print('  -> **多い**: 向きは決まらない。**作者の時間を使う価値がある**')
    else:
        print('  -> **判定保留** (5〜20%)。作者に出すかの判断を保留し、別途相談する')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--known-cells', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.known_cells:
        return known_cells()
    os.makedirs(OUT, exist_ok=True)
    zz = zmap(); fill, line = arms('gated')
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'L31 単位のずれの頻度   頁 {len(js)}   '
          f'(既知セル KC35-37 は別プロセスで PASS 済みであること)', flush=True)
    rows_c, rows_l, drop = [], [], collections.Counter()
    with Pool(4, initializer=init) as pool:
        for i, (c, l, d) in enumerate(pool.imap_unordered(one_page, js), 1):
            rows_c += c; rows_l += l; drop += d
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  成分 {len(rows_c)}  層x コマ {len(rows_l)}',
                      flush=True)
    tag = f'_limit{a.limit}' if a.limit else ''
    for nm, rows in (('components', rows_c), ('layers', rows_l)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        p = f'{OUT}/{nm}{tag}.csv'
        with open(p, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'-> {p}  行 {len(rows)}')
    report(rows_c, rows_l, drop, a.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
