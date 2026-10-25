# -*- coding: utf-8 -*-
"""L22: `effect` 層の中から**黒消し**を特徴量で切り分けられるか (作者の仮説、2026-10-25)。

> 黒消しはおそらく黒消しだけの独立したレイヤの可能性が高い
> 特徴量で他の effect レイヤ(効果線やトーン等)と切り分けができる可能性が高いとみている

**作者の仮説が立つかどうかを測るだけで、棒は置かない。** 正例は 14 面しかないので、
しきい値を当てて「分けられた」と言える標本ではない。出すのは**分布と、重なりの有無**。

器具 (層ごと。成分は fill_mask の 8連結):
  rect_fill   成分の面積 / 最小外接矩形の面積 (辺に +1。KC29 参照)。べたの塊は 1.0 に寄る
  minor       最小外接矩形の短辺 = **帯の幅**。効果線は細く、黒消しは太い
  ridge       `l21_dirtiness.ridge_width` の中央 (墨重み)
**rect_fill だけでは線と帯を分けられない**(まっすぐな線も 1.0)。KC26 と同じ罠なので、
**minor と対で使う。** KC29 がこれを試験する。

  python tools/evaluation/l22_effect_split.py [--kc] [--name effect]
"""
import argparse, collections, csv, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from fill_mask import fill_mask
from l21_dirtiness import ridge_width, acc_w, quant
from l17_known_cells import gated_layers, jobs, zmap, panels_and_regions, assign_top
from l14_known_cells import comps, A

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l22_effect_split'
Q = '/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'
LABF = '/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'
MINC = 2000            # これ未満の成分は数えない (トーンの粒を1つずつ測らない)


def comp_feats(ink):
    """成分ごとの (画素数, rect_fill, minor, major)。"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(st[i, A])
        if a < MINC:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        # **+1 する** —— minAreaRect は画素の中心を囲むので、各辺が 1px 短く出る。
        # 短辺が小さいほど効き、幅5の線では rect_fill が 1.25 になった (KC29 が掴んだ)
        w, h = w + 1.0, h + 1.0
        rect = w * h
        out.append((a, a / rect if rect > 0 else 0.0, min(w, h), max(w, h)))
    return out


def layer_feats(ink):
    """層の特徴。**画素重み**で中央を取る(大きい塊が1つあっても成分数に埋もれない)。"""
    fm = np.zeros(ink.shape, bool)
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        return None
    y0, y1 = max(0, ys.min() - 8), min(ink.shape[0], ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(ink.shape[1], xs.max() + 9)
    fm[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
    cf = comp_feats(fm)
    w = ridge_width(ink)
    H = acc_w(w)
    d = dict(ink=int(ink.sum()), fill_px=int(fm.sum()), n_comp=len(cf),
             ridge_len=quant(H, 'len')[1] if w.size else None,
             ridge_ink=quant(H, 'ink')[1] if w.size else None)
    if cf:
        a = np.array([c[0] for c in cf], float)
        for j, k in ((1, 'rect_fill'), (2, 'minor'), (3, 'major')):
            v = np.array([c[j] for c in cf], float)
            o = np.argsort(v)
            cw = np.cumsum(a[o]) / a.sum()
            d[k] = float(v[o][np.searchsorted(cw, 0.5)])
        d['bar_px'] = int(sum(c[0] for c in cf if c[1] >= 0.90 and c[2] >= 20))
        d['bar_share'] = round(d['bar_px'] / max(d['fill_px'], 1), 4)
    return d


# ---------------------------------------------------------------- 既知セル

def kc29():
    print('KC29 器具: 形の分かっている塊を入れて rect_fill と minor が返す値')
    ok = True
    cases = []
    for ang in (0, 23, 45):
        im = np.zeros((600, 600), np.uint8)
        r = cv2.boxPoints(((300, 300), (420.0, 60.0), float(ang)))
        cv2.fillPoly(im, [np.int32(r)], 1)
        cases.append((f'帯 420x60 {ang}度', im.astype(bool), 1.00, 60))
    im = np.zeros((600, 600), np.uint8); cv2.circle(im, (300, 300), 120, 1, -1)
    cases.append(('円 半径120', im.astype(bool), 0.785, 240))
    im = np.zeros((600, 600), np.uint8); im[297:302, 60:540] = 1   # 幅5x長480 = 2400px
    cases.append(('線 幅5 (MINC を越える長さ)', im.astype(bool), 1.00, 5))
    im = np.zeros((600, 600), np.uint8)
    for y in range(40, 560, 16):
        for x in range(40, 560, 16):
            cv2.circle(im, (x, y), 4, 1, -1)
    cases.append(('トーンの粒 (成分が小さく数えられない)', im.astype(bool), None, None))
    for nm, m, er, em in cases:
        cf = comp_feats(m)
        if er is None:
            good = len(cf) == 0
            print(f'  {nm:<34}成分 {len(cf)} (期待 0)  {"ok" if good else "FAIL"}')
            ok &= good
            continue
        a, rf, mi, ma = cf[0]
        good = abs(rf - er) <= 0.03 and abs(mi - em) <= 2.0
        ok &= good
        print(f'  {nm:<34}rect_fill {rf:.3f} (期待 {er:.3f})  minor {mi:6.1f}'
              f' (期待 {em})  {"ok" if good else "FAIL"}')
    print('  **線 幅5 も rect_fill 1.0** —— 帯と線は rect_fill では分けられない。'
          'minor と対で使うこと')
    return ok


# ---------------------------------------------------------------- 本体

def kuro_layers():
    """作者が「黒消し」と答えた面の出所の層 id を引く。"""
    q = json.load(open(Q))['queue']
    crop_of = {r['order']: r['crop'] for r in q}
    lab = {}
    for ln in open(LABF):
        r = json.loads(ln)
        if r['choice'] == 'free' and r.get('free'):
            r['choice'], r['word'] = 'word', r['free'].strip()
        lab[crop_of[r['order']]] = r
    want = {c for c, v in lab.items() if v['choice'] == 'word' and v['word'] == '黒消し'}
    ans = {c: (v['word'] if v['choice'] == 'word' else v['choice']) for c, v in lab.items()}
    return want, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kc', action='store_true')
    ap.add_argument('--name', default='effect')
    a = ap.parse_args()
    if not kc29():
        print('既知セルが FAIL。止める'); return 1
    if a.kc:
        return 0
    os.makedirs(OUT, exist_ok=True)
    names = {(r['page_id'], r['layer_id']): r['name']
             for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    want, ans = kuro_layers()
    pages = {c.split('_c')[0].split('_', 1)[1] for c in want}
    fill, line = gated_layers()
    zz = zmap()

    # --- 1. 黒消しの面の出所の層 id
    print(f'\n[1] 作者が「黒消し」と答えた {len(want)} 面の出所の層')
    kuro_lid = set()
    for job in jobs(zz, fill, line):
        if job[0] not in pages:
            continue
        pid, work, fz, lz, sel = job
        fl = []
        for lid, path in fz:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            ink = im < 128
            fm = np.zeros(ink.shape, bool)
            ys, xs = np.nonzero(ink)
            if ys.size == 0:
                continue
            y0, y1 = max(0, ys.min() - 8), min(ink.shape[0], ys.max() + 9)
            x0, x1 = max(0, xs.min() - 8), min(ink.shape[1], xs.max() + 9)
            fm[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
            fl.append((lid, fm))
        k = 0
        from l21_dirtiness import pages_with_window
        from l17_known_cells import adj_sets, D1
        from l15_known_cells import boundary
        for (pid2, work2, ci, P, glab, gids, gstat,
             sub, (y0, x0, y1, x1), _fl, lines) in pages_with_window(job):
            lab2, keys = assign_top(sub, P.shape)
            for gi in gids:
                R = glab == gi
                vals, cnt = adj_sets(R, lab2, P, ds=(D1,), bnd=boundary(R))[D1]
                if cnt.sum() == 0:
                    continue
                k += 1
                nm = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
                if nm not in want:
                    continue
                best, bc = None, 0
                for lid, fm in fl:
                    c = int((fm[y0:y1, x0:x1] & R).sum())
                    if c > bc:
                        best, bc = lid, c
                kuro_lid.add((pid, best))
                print(f'  {nm}  -> L{best} "{names.get((pid, str(best)), "?")}"'
                      f'  面の {bc / max(int(R.sum()), 1):.3f}')
    print(f'  **別々の層 {len(kuro_lid)} 枚**から来ている')

    # --- 2. 同名の層を全部測って、黒消しの層が分かれるか
    print(f'\n[2] 名前が "{a.name}" の層の特徴 (FILL の門を通ったものだけ)')
    rows = []
    for (pid, work), lst in sorted(fill.items()):
        for lid, path in lst:
            if names.get((pid, str(lid))) != a.name:
                continue
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            d = layer_feats(im < 128)
            if d is None:
                continue
            d.update(pid=pid, work=work, lid=lid,
                     kuro=int((pid, lid) in kuro_lid))
            rows.append(d)
    rows.sort(key=lambda r: (-r['kuro'], -r.get('bar_share', 0)))
    hdr = f'  {"黒消":<4}{"頁":<34}{"L":>5}{"墨":>10}{"成分":>5}{"rect":>7}{"minor":>8}{"major":>9}{"帯の割合":>9}'
    print(hdr)
    for r in rows:
        print(f'  {"**" if r["kuro"] else "  ":<4}{r["pid"][:33]:<34}{r["lid"]:>5}'
              f'{r["ink"]:>10,}{r["n_comp"]:>5}'
              f'{r.get("rect_fill", float("nan")):>7.3f}{r.get("minor", float("nan")):>8.1f}'
              f'{r.get("major", float("nan")):>9.1f}{r.get("bar_share", float("nan")):>9.4f}')
    with open(f'{OUT}/{a.name}_layers.csv', 'w') as fh:
        ks = ['work', 'pid', 'lid', 'kuro', 'ink', 'fill_px', 'n_comp', 'rect_fill',
              'minor', 'major', 'bar_px', 'bar_share', 'ridge_len', 'ridge_ink']
        fh.write(','.join(ks) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(k, '')) for k in ks) + '\n')

    # --- 3. 作者が答えた層と、同じ特徴を持つ未確認の層
    print('\n[3] `bar_share` の並び (帯 = rect_fill>=0.90 かつ minor>=20 の成分の墨の割合)')
    srt = sorted(rows, key=lambda r: -r.get('bar_share', 0))
    yes = [r for r in srt if r['kuro']]
    pred = [r for r in srt if not r['kuro'] and r.get('bar_share', 0) >= min(
        (x.get('bar_share', 1) for x in yes), default=1)]
    rest = [r for r in srt if not r['kuro'] and r not in pred]
    print(f'  作者が「黒消し」と答えた層 {len(yes)} 枚: '
          + '  '.join(f'{r["bar_share"]:.4f}' for r in yes))
    print(f'  **同じ範囲に入る未確認の層 {len(pred)} 枚**: '
          + '  '.join(f'{r["bar_share"]:.4f}' for r in pred))
    print(f'  残り {len(rest)} 枚の最大: '
          + (f'{max(r.get("bar_share", 0) for r in rest):.4f}' if rest else '-'))
    for r in pred:
        print(f'    予測 (**作者の確認が要る**): {r["work"]} {r["pid"]} L{r["lid"]}'
              f'  墨 {r["ink"]:,}  rect {r["rect_fill"]:.3f}  minor {r["minor"]:.1f}')
    if yes and rest:
        lo = min(r.get('bar_share', 0) for r in yes + pred)
        hi = max(r.get('bar_share', 0) for r in rest)
        print(f'  **隔たり**: 下側 {lo:.4f} / 上側 {hi:.4f}'
              + ('  (重なり無し)' if lo > hi else '  (重なる)'))
    print('\n  他の特徴の範囲 (読みのため。**棒は置かない**)')
    for k in ('minor', 'rect_fill', 'n_comp', 'ridge_ink'):
        v1 = [r[k] for r in yes + pred if r.get(k) is not None]
        v0 = [r[k] for r in rest if r.get(k) is not None]
        if not v1 or not v0:
            continue
        print(f'    {k:<11} 黒消し+予測 n={len(v1)} [{min(v1):.4g}, {max(v1):.4g}]'
              f'   残り n={len(v0)} [{min(v0):.4g}, {max(v0):.4g}]')
    print(f'\n-> {OUT}/{a.name}_layers.csv')
    print('**正例が 6 枚しかないので、ここでしきい値を決めることはしない。**')
    print('言えるのは「作者が答えた 6 枚と同じ所に、未確認の層が居る」までで、')
    print('**分けられるかは、作者の確認と、他の作品での再現を見てから。**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
