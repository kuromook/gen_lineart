# -*- coding: utf-8 -*-
"""L39 の材料を頁ごとに作る(登録 `doc/l39_line_to_fill_registration_20261108.md` §2〜§4)。

**この器具は何も採点しない。** 頁ごとに次を `results/l39_cache/<pid>.pkl` に書く:

  `L4`  線の層(`arms('gated')` の line 側)の墨の和を 1/4 に縮めた濃度(uint8 0..255)= **入力**
  `F4`  塗りの層(fill 側)の `fill_mask(radius=4)` の和 `F` を 1/4 に縮めて 0.5 で2値 = **標的**
  `M4`  コマ成分の和を 1/4 に縮めて 0.5 で2値 = **損失を取る範囲**
  `panels`  原寸のコマ成分ごとに: 外接矩形 / `P` / 天井 `F ∩ P` / 腕C `fill_mask(L ∩ P)` /
            そのコマの面プールの面のマスク(**面ごと。重なりうるのでラベル画像にしない**)

**IC1(面の復元)はここで回す**: 面を `l38_separate_problem_a_20261108.py` と**同じ順で**作り直し、
JSON の行と**順番・鍵・面積が全行一致**しなければ止まる(鍵だけでは一意でない —— 同じ層から
片が複数出るので、`(pid, frame, comp, region, origin, piece_lid)` は 620 通りしか無い)。
**IC3(層の交差)もここ**: 入力の層と標的の層の交差 0。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l39_prepare_20261108.py [頁数]
"""
import collections
import json
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                      # noqa: E402
from fill_mask import fill_mask                                     # noqa: E402
from l14_known_cells import A, comps                                # noqa: E402
from l17_known_cells import DEL, MAX_FRAC, MIN_FRAC, arms, jobs, zmap  # noqa: E402
from l38_merge_cause_20261106 import drop_wip, layer_masks, panels  # noqa: E402
from l38_separate_problem_a_20261108 import pieces_in_face          # noqa: E402

ROOT = '/home/sh1/deepl/lineart-fill-region'
POOL = f'{ROOT}/results/l38_faces_separated_20261108.json'
CACHE = f'{ROOT}/results/l39_cache'
SCALE = 4
PAD = 8                       # 腕C の fill_mask を取る窓の余白(`acc_add` の PAD と同じ役割)


def pack(m):
    return (m.shape, np.packbits(m.astype(bool), axis=None))


def unpack(t):
    shape, b = t
    return np.unpackbits(b, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def down(m, shape4, binary):
    v = cv2.resize(m.astype(np.float32), (shape4[1], shape4[0]), interpolation=cv2.INTER_AREA)
    return (v >= 0.5).astype(np.uint8) if binary else np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)


def all_panels(job, shape, cut):
    """**面の無いコマも含めた**コマ成分 (鍵, 矩形, P)。条件は `panels()` と同じ大きさの門。"""
    pid, work, fz, lz, sel = job
    H, W = shape
    for fr in sel:
        km = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if km is None or km.shape != (H, W):
            cut['コマ枠が読めない/寸法が違う'] += 1
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats((km > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(st[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                cut['コマ成分が小さすぎ/大きすぎ'] += 1
                continue
            y0 = max(0, st[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, st[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + st[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + st[ci, cv2.CC_STAT_WIDTH] + 4)
            yield (fr['file'], int(ci)), (y0, y1, x0, x1), lab[y0:y1, x0:x1] == ci


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    os.makedirs(CACHE, exist_ok=True)
    pool = json.load(open(POOL, encoding='utf-8'))
    print(f'L39 材料。面プール {len(pool)} 行(`{os.path.basename(POOL)}`)', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    # IC3 層の交差
    inter = {(k, lid) for k, v in fill.items() for lid, _ in v} & \
            {(k, lid) for k, v in line.items() for lid, _ in v}
    print(f'IC3 入力の層と標的の層の交差 = {len(inter)}(0 でなければ止まる)')
    assert not inter
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'母集団: 現行の素 {len(js)} 頁')
    for k, v in cut.most_common():
        print(f'  外した {v}  {k}')
    cut.clear()
    if limit:
        js = js[:limit]
        print(f'  **煙試験: 先頭 {limit} 頁だけ。IC1 は先頭の行だけ比べる**')

    rebuilt = []
    leak_num = collections.Counter()
    leak_den = collections.Counter()
    for i, job in enumerate(js, 1):
        pid, work, fz, lz, sel = job
        lm, shape = layer_masks(fz, 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁'] += 1
            continue
        zs = {lid: zz.get((pid, str(lid))) for lid, _ in lm}
        order = sorted(lm, key=lambda t: (zs[t[0]] if zs[t[0]] is not None else -1), reverse=True)
        F = np.zeros(shape, bool)
        for _, m in lm:
            F |= m
        L = np.zeros(shape, bool)
        nl = 0
        for lid, path, z in lz:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape != shape:
                cut['線の層 PNG が読めない/寸法が違う'] += 1
                continue
            L |= im < 128
            nl += 1
        H, W = shape
        shape4 = ((H + SCALE - 1) // SCALE, (W + SCALE - 1) // SCALE)
        Mfull = np.zeros(shape, bool)

        faces_by_panel = collections.defaultdict(list)
        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut,
                                                                  count_cut=False):
            win = [(lid, m[y0:y1, x0:x1] & P) for lid, m in order]
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                if ar < U.FRAGMENT_FLOOR:
                    continue
                pieces, _ = pieces_in_face(R, win)
                if len(pieces) >= 2:
                    for lid, a, m in pieces:
                        rebuilt.append((work, pid, pkey[0], int(ci), int(gi), 'piece', lid, a))
                        faces_by_panel[pkey].append((len(rebuilt) - 1, m))
                else:
                    rebuilt.append((work, pid, pkey[0], int(ci), int(gi), 'kept', None, ar))
                    faces_by_panel[pkey].append((len(rebuilt) - 1, R))

        pans = []
        for pkey, (y0, y1, x0, x1), P in all_panels(job, shape, cut):
            Mfull[y0:y1, x0:x1] |= P
            Lc = L[y0:y1, x0:x1] & P
            # 腕C: 入力の墨に fill_mask(radius=4)。窓は P の矩形そのもの(既に余白 2 つき)
            C = fill_mask(Lc, radius=4) & P if Lc.any() else np.zeros_like(P)
            fl = faces_by_panel.get(pkey, [])
            for _, m in fl:
                leak_num[work] += int((m & Lc).sum())
                leak_den[work] += int(m.sum())
            pans.append(dict(key=pkey, box=(y0, y1, x0, x1), P=pack(P),
                             oracle=pack(F[y0:y1, x0:x1] & P), copy=pack(C),
                             faces=[(idx, pack(m)) for idx, m in fl]))
        got = sum(len(v) for v in faces_by_panel.values())
        placed = sum(len(p['faces']) for p in pans)
        if got != placed:
            cut['面の在るコマが all_panels に無い(面が落ちた)'] += got - placed
        out = dict(pid=pid, work=work, shape=shape, scale=SCALE,
                   L4=down(L, shape4, False), F4=down(F, shape4, True),
                   M4=down(Mfull, shape4, True), panels=pans, n_line_layers=nl,
                   n_fill_layers=len(lm))
        pickle.dump(out, open(f'{CACHE}/{pid}.pkl', 'wb'), protocol=4)
        print(f'  {i}/{len(js)} {pid}  コマ {len(pans)}  面 {placed}  線の層 {nl}  塗りの層 {len(lm)}',
              flush=True)
    for k, v in cut.most_common():
        print(f'  落とした {v}  {k}')
    if not cut:
        print('  落とした 0')

    # IC1 面の復元: 順番・鍵・面積が全行一致
    n = len(rebuilt) if limit else len(pool)
    bad = 0
    for j in range(n):
        r = pool[j] if j < len(pool) else None
        b = rebuilt[j] if j < len(rebuilt) else None
        if r is None or b is None or (r['work'], r['pid'], r['frame'], r['comp'], r['region'],
                                       r['origin'], r['piece_lid'], r['area']) != b:
            bad += 1
    print(f'IC1 面の復元: 作り直し {len(rebuilt)} 行 / JSON {len(pool)} 行、'
          f'比べた {n} 行のうち不一致 {bad}')
    kinds = collections.Counter(b[5] for b in rebuilt)
    print(f'    kept {kinds["kept"]} / piece {kinds["piece"]}(登録: 551 / 105)')
    ok = bad == 0 and (limit or (len(rebuilt) == len(pool) == 656
                                 and kinds['kept'] == 551 and kinds['piece'] == 105))
    print(f'IC1 -> {"PASS" if ok else "FAIL"}')
    for w in sorted(leak_den):
        print(f'NC2 入力の漏れ {w}: 面の画素のうち入力の墨 {leak_num[w]:,}/{leak_den[w]:,}'
              f' = {leak_num[w] / leak_den[w]:.2%}')
    json.dump(dict(ic1=bool(ok), rebuilt=rebuilt), open(f'{CACHE}/_index.json', 'w'),
              ensure_ascii=False)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
