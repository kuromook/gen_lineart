# -*- coding: utf-8 -*-
"""L28 `preview_dark < 0.5` が落とした「線画」層の原寸切り出し。

**これは測定ではない。作者に見せる材料を作るだけである**(規律6: 絵が何を
描いているかは作者が決める)。選ぶ条件は門の内訳から:

  門が `preview_dark < 0.5` で落とした 875 層のうち
    名前に `線画` / `line` を含み (**名前は判定ではなく、確かめる理由**)
    アルファ>=128 の画素が 10,000 以上で
    `black_among_drawn >= 0.95` (ほぼ不透明)
  -> 45 層

**層を1枚だけ描く。** 他の層は描かない —— L23 で「全層を z 順に塗れば原稿の
見た目」が成り立たないことが出ている(`用紙` が全面アルファ255)。合成を見せると
こちらの合成処理の主張になる。**1枚の層に何が入っているか**だけを見せる。

層 PNG は `g = 255 - アルファ` なので、そのまま描くと不透明な画素が黒く出る。
切り出しは**原寸 1:1**(縮小しない。規律6)。墨の重心を中心に取り、
墨が窓に収まらないときは**最も密な窓**を選ぶ。
"""
import csv, os, sys
import numpy as np, cv2
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pid_work import work_of  # **既定値に落とさない**(罠 T104 の族)

HERE = '/home/sh1/deepl/lineart-fill-region'
INV = f'{HERE}/inbox/page_layers_priority65.csv'
LED = f'{HERE}/inbox/line_selection_ledger_priority65_px.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = f'{HERE}/results/l28_preview_dark_crops'
WIN = 1400          # 切り出しの一辺(原寸の画素)
INK_MIN, BAD_MIN = 10000, 0.95


def num(v):
    """**空文字と 0.0 を分ける**(罠 T104: `or` で既定値を入れると 0 が消える)。"""
    v = (v or '').strip()
    if v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def pick():
    led = {(r['page_path'], r['layer_id']): r
           for r in csv.DictReader(open(LED, encoding='utf-8-sig'))}
    out, cut = [], {}
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        dr = num(r['drawn_ratio'])
        if dr is None or dr <= 0:
            continue
        pd = num(r['preview_dark'])
        if pd is None or pd >= 0.5:
            continue
        cut['落とされた全部'] = cut.get('落とされた全部', 0) + 1
        if not ('線画' in r['name'] or 'line' in r['name'].lower()):
            cut['名前に線画/line なし'] = cut.get('名前に線画/line なし', 0) + 1
            continue
        ink = int(led[(r['page_path'], r['layer_id'])]['ink_px'] or 0)
        if ink < INK_MIN:
            cut[f'アルファ>=128 が {INK_MIN:,} 未満'] = cut.get(
                f'アルファ>=128 が {INK_MIN:,} 未満', 0) + 1
            continue
        if (num(r['black_among_drawn']) or 0) < BAD_MIN:
            cut[f'bad < {BAD_MIN}'] = cut.get(f'bad < {BAD_MIN}', 0) + 1
            continue
        out.append((r, ink))
    print('選別 (落としたものを理由別に印字。規律7)')
    for k, v in cut.items():
        print(f'  {k}: {v}')
    print(f'  **残った {len(out)}**')
    return sorted(out, key=lambda t: -t[1])


def best_window(ink):
    """墨が最も密な WIN x WIN の窓の左上。積分画像で全位置を見る。"""
    h, w = ink.shape
    if h <= WIN and w <= WIN:
        return 0, 0
    s = cv2.integral(ink.astype(np.uint8))
    hh, ww = min(WIN, h), min(WIN, w)
    tot = (s[hh:, ww:] + s[:-hh, :-ww] - s[hh:, :-ww] - s[:-hh, ww:])
    y, x = np.unravel_index(int(tot.argmax()), tot.shape)
    return int(y), int(x)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = pick()
    man = []
    for i, (r, ink_px) in enumerate(rows, 1):
        pid, lid = r['page_id'], r['layer_id']
        w = work_of(pid, r.get('page_path'))
        p = f'{BASE}/{w}/all_layers/{pid}_L{lid}.png'
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f'  {i:2d}/{len(rows)} 読めない {p}')
            continue
        ink = g <= 127                      # アルファ >= 128
        y, x = best_window(ink)
        crop = g[y:y + WIN, x:x + WIN]
        safe = r['name'].replace('/', '_').replace(' ', '_')[:20]
        name = f'{i:02d}_{pid}_L{lid}_{safe}.png'
        cv2.imwrite(f'{OUT}/{name}', crop)
        frac = float(ink[y:y + WIN, x:x + WIN].sum()) / max(int(ink.sum()), 1)
        man.append(dict(file=name, page_id=pid, layer_id=lid, name=r['name'],
                        ink_px=ink_px, preview_dark=r['preview_dark'],
                        black_among_drawn=r['black_among_drawn'],
                        crop_x=x, crop_y=y, crop_px=WIN,
                        ink_in_crop=round(frac, 4)))
        print(f'  {i:2d}/{len(rows)} {name}  墨 {ink_px:,}  窓の中に {frac:.1%}', flush=True)
    with open(f'{OUT}/manifest.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(man[0]))
        wr.writeheader()
        wr.writerows(man)
    print(f'-> {OUT}/  ({len(man)} 枚、原寸 {WIN}x{WIN})')
    print(f'-> {OUT}/manifest.csv')


if __name__ == '__main__':
    main()
