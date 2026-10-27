# -*- coding: utf-8 -*-
"""L37 `white` / `MASK` は**下の墨の上に在るか**(登録 `doc/work_log.md` 2026-10-26)。

作者: **`white` はほとんどの場合部位を示さない。エフェクトである場合と、
ホワイト修正に使っていた。** / **`MASK` は塗り用のベースである。部位はきめられない。**

**ホワイト修正は墨を消すために墨の上に不透明な白を置く操作**だが、L の器具は
`ink = (g < 128)`(アルファ >= 128)で読むので**「墨が在る」と数える。**
そして `white` 7層・`MASK` 8層が**ベタの標準(fill)に入っている。**

測る量(**名前で呼ばない。位置関係だけ**):

    ov(W) = |ink_W ∩ below| / |ink_W|      below = z_global が W より小さい層の墨の和

**この試験の力は一方向である**(登録 §2):
  **修正なら ov は高い。だが ov が高いことは修正を意味しない**
  (黒髪の上の白いハイライトも ov が高い)。**ov が低ければ修正ではありえない。**
  **何を描いているかは作者が決める**(規律6)。

棒(登録 §3): `ov <= 0.20` 修正ではありえない / `>= 0.80` 判別できない / 間は判定保留。
"""
import collections
import csv
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import COV_MAX
from l17_known_cells import zmap
from pid_work import work_of
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4 = '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l37_white_over_ink'
LO, HI = 0.20, 0.80            # 登録の棒


def is_target(name):
    n = (name or '')
    return 'white' in n.lower() or 'MASK' in n.upper()


# ---------------------------------------------------------------- 既知セル
def _ov(wink, lowers):
    below = np.zeros(wink.shape, bool)
    for ink in lowers:
        below |= ink
    n = int(wink.sum())
    return (int((wink & below).sum()) / n) if n else None


def known_cells():
    ok = True
    H = W = 200
    low = np.zeros((H, W), bool); low[50:150, 20:100] = True
    on = np.zeros((H, W), bool); on[60:140, 30:90] = True      # 下の墨の中
    beside = np.zeros((H, W), bool); beside[60:140, 120:180] = True   # 隣

    a, b = _ov(on, [low]), _ov(beside, [low])
    print(f'KC49 幾何: 重なる {a:.4f} (==1) / 隣 {b:.4f} (==0)  '
          f'{"PASS" if a == 1.0 and b == 0.0 else "FAIL"}')
    ok &= (a == 1.0 and b == 0.0)

    # KC50: 同じ幾何で z を入れ替える -> below が空になる
    #       (**幾何だけを見ていたらここで落ちる**)
    zs = {'w': 5, 'l': 9}        # 白が下、墨が上
    lowers = [low] if zs['l'] < zs['w'] else []
    c = _ov(on, lowers)
    print(f'KC50 z 順: 墨を白より上にすると {c:.4f} (==0)  '
          f'{"PASS" if c == 0.0 else "FAIL"}')
    ok &= (c == 0.0)

    # KC52: **飽和の点検**(2026-10-26 に足した。最初の走行が全部 1.0 になったので)
    # 画布の全面を覆う層を `below` に積むと、**どんな層でも ov = 1 になる。**
    # **「空でないこと」ではなく「飽和していないこと」が要る点検だった。**
    full = np.ones((H, W), bool)
    d = _ov(beside, [full])
    print(f'KC52 飽和: 全面の層を下に積むと、隣に置いた層でも {d:.4f} (==1) になる  '
          f'{"PASS" if d == 1.0 else "FAIL"}')
    print('     -> **だから全面の層 (被覆 >= 0.95) は `below` に積まない。**'
          ' 本体は各対象の `below_frac` を印字する')
    ok &= (d == 1.0)
    return bool(ok)


# ---------------------------------------------------------------- 本体
def one_page(job):
    """1頁: z の昇順に積みながら、対象の層の `ov` を取る。**層は1回しか読まない。**

    **2026-10-26 の直し**: 最初の走行は **80層すべてが ov = 1.0000** になった。
    発見ではなく欠陥で、**`用紙` が画布の全面でアルファ 255** だからである
    (65頁に1枚ずつ。`color` 12枚と全面トーン9枚も同じ)。
    **下の墨の和集合が常に画布の全面になり、量が自明に 1 になっていた。**

    **この罠は台帳に在った**(L23:「全層を z 順に塗れば原稿の見た目」は成り立たない、
    `用紙` が全面アルファ255)。**持っていて当てなかった。**

    直し: **`below` に積むのは `cov < 0.95` の層だけ**(`keep4.csv` の被覆)。
    `l12` / `l15` が「用紙・全面トーンは遮蔽物にしない」で使っているのと同じ門で、
    **新しい定数を引かない。**

    **登録の定義は字面では「下に在る層の墨の和集合」だったが、
    それは自明に 1 を出す量だった。** 消せるのは**描かれた墨**で、紙ではない ——
    **登録が言い落としていた条件を書き足す**(棒は動かさない。数を読む前)。
    """
    pid, work, layers = job          # layers = [(z, lid, path, name, in_fill, wide)]
    rows, drops = [], collections.Counter()
    below = None
    n_wide = 0
    for z, lid, path, name, in_fill, wide in sorted(layers, key=lambda t: t[0]):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            drops['PNG が読めない'] += 1
            continue
        ink = im < 128
        if below is None:
            below = np.zeros(ink.shape, bool)
        if ink.shape != below.shape:
            drops['形が合わない'] += 1
            continue
        if is_target(name):
            n = int(ink.sum())
            if n == 0:
                drops['墨が 0'] += 1
            else:
                rows.append(dict(pid=pid, work=work, lid=lid, name=name, z=z,
                                 in_fill=int(in_fill), ink_px=n,
                                 over_px=int((ink & below).sum()),
                                 ov=round(int((ink & below).sum()) / n, 4),
                                 # **飽和の点検**: この時点の `below` が画布の何割か。
                                 # 1 に近ければ `ov` は自明で、量は意味を持たない
                                 below_frac=round(float(below.mean()), 4),
                                 n_wide=n_wide))
        if wide:
            n_wide += 1              # **全面の層は下に積まない**(用紙・全面トーン)
        else:
            below |= ink             # **自分も下に積む**(次の対象のために)
    return rows, drops


def main():
    os.makedirs(OUT, exist_ok=True)
    print('L37 `white` / `MASK` は下の墨の上に在るか '
          '(登録 2026-10-26。**既知セルを先に**)')
    if not known_cells():
        print('**既知セルが落ちた。本体を回さない。**')
        sys.exit(1)
    print()

    zz = zmap()
    gfill, _gline = arms('gated')
    F = {(k, lid) for k, v in gfill.items() for lid, _ in v}
    # **全面の層を `below` から外すための被覆**(`l12`/`l15` と同じ門。新しい定数を引かない)
    cov = {(r['page_id'], r['layer_id']): float(r['cov'])
           for r in csv.DictReader(open(KEEP4))}
    n_wide_all = sum(1 for v in cov.values() if v >= COV_MAX)
    print(f'`below` から外す全面の層 (被覆 >= {COV_MAX}): {n_wide_all} 層'
          '  (**`用紙` 65 / `color` 12 / 全面トーン9。これを積むと ov が自明に 1 になる**)')

    per = collections.defaultdict(list)
    cut = collections.Counter()
    n_target = 0
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        dr = r['drawn_ratio'].strip()
        if not dr or float(dr) <= 0:
            cut['`drawn_ratio = 0`(発注の範囲外。墨を持たない)'] += 1
            continue
        pid, lid = r['page_id'], r['layer_id']
        z = zz.get((pid, lid))
        if z is None:
            cut['z 順が決まらない'] += 1
            continue
        w = work_of(pid, r.get('page_path'))
        p = f'{BASE}/{w}/all_layers/{pid}_L{lid}.png'
        if not os.path.exists(p):
            cut['層 PNG が無い'] += 1
            continue
        tgt = is_target(r['name'])
        n_target += tgt
        cv = cov.get((pid, lid))
        wide = (cv is not None and cv >= COV_MAX)
        per[(pid, w)].append((z, lid, p, r['name'],
                              ((pid, w), int(lid)) in F, wide))

    jobs = [(pid, w, v) for (pid, w), v in per.items()
            if any(is_target(x[3]) for x in v)]
    print('選別 (規律9。理由別に)')
    for k, v in cut.most_common():
        print(f'  落とした {v:5d}  {k}')
    print(f'  **対象の層 {n_target}  対象を持つ頁 {len(jobs)} / 全 {len(per)} 頁**')
    if not jobs:
        print('  **この点検は空(対象の層が1つも無い)**')
        sys.exit(1)

    rows, drops = [], collections.Counter()
    nw = int(os.environ.get('L37_WORKERS', '2'))   # L36 が走っているので控える
    with Pool(nw) as pool:
        for i, (rr, dd) in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += rr
            drops.update(dd)
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  対象 {len(rows)}', flush=True)
    for k, v in drops.most_common():
        print(f'  層を落とした {v}  {k}')

    with open(f'{OUT}/over_ink.csv', 'w', newline='', encoding='utf-8') as fh:
        cols = ['pid', 'work', 'lid', 'name', 'z', 'in_fill', 'ink_px',
                'over_px', 'ov', 'below_frac', 'n_wide']
        wr = csv.DictWriter(fh, cols, extrasaction='ignore')
        wr.writeheader()
        for r in sorted(rows, key=lambda r: (-r['in_fill'], r['ov'])):
            wr.writerow(r)
    print(f'-> {OUT}/over_ink.csv  行 {len(rows)}')

    # ---------------- 判定 (登録 §3。棒は動かさない)
    print()
    print('=' * 92)
    print('**ov = 層の墨のうち、下の墨の上に在る割合**  '
          '(**この量を「修正らしさ」と呼ばない。位置関係だけ**)')
    print('**試験の力は一方向**: 修正なら ov は高い。'
          'だが ov が高いことは修正を意味しない(黒髪の上の白いハイライトも高い)。'
          '**ov が低ければ修正ではありえない**')

    def band(v):
        return ('修正ではありえない' if v <= LO else
                '判別できない' if v >= HI else '判定保留')

    for scope, sel in (('**標準(fill)に入っている層**',
                        [r for r in rows if r['in_fill']]),
                       ('(併記) 標準の外の層', [r for r in rows if not r['in_fill']])):
        print()
        print(f'{scope}  {len(sel)} 層')
        if not sel:
            print('  0 層')
            continue
        c = collections.Counter(band(r['ov']) for r in sel)
        for k in ('修正ではありえない', '判定保留', '判別できない'):
            print(f'  {k:12s} {c[k]:4d}')
        for r in sorted(sel, key=lambda r: -r['ov'])[:20]:
            print(f'    ov {r["ov"]:.4f}  {band(r["ov"]):12s}  '
                  f'墨 {r["ink_px"]:>9,}  z {r["z"]:>5}  '
                  f'下の墨が画布の {r["below_frac"]:.4f}  '
                  f'{r["pid"]} L{r["lid"]:>4}  "{r["name"]}"')

    infill = [r for r in rows if r['in_fill']]
    bad = [r for r in infill if r['ov'] >= HI]
    hold = [r for r in infill if LO < r['ov'] < HI]
    print()
    print('=' * 92)
    if not infill:
        print('**標準に `white` / `MASK` の層が1つも入っていない。この軸は空**')
    elif not bad and not hold:
        print(f'-> **標準の {len(infill)} 層すべてが ov <= {LO}。'
              f'この軸での懸念は消える。何も直さない**(登録 §3)')
    else:
        print(f'-> **標準の中に判別できない層が {len(bad)} / 判定保留が {len(hold)}**')
        print('   登録どおり: **原寸1:1の切り出しを作って作者に出す候補にする**'
              '(出す順はユーザー)。**こちらの判断で標準から外さない** —— '
              '外すのは登録の変更で、別の走行になる')
        for r in sorted(bad + hold, key=lambda r: -r['ov']):
            print(f'     ov {r["ov"]:.4f}  {r["pid"]} L{r["lid"]} "{r["name"]}"')


if __name__ == '__main__':
    main()
