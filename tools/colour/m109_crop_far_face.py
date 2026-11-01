"""`page0005` の「遠い面」が何で囲まれているかを**原寸の切り出し**で見せる。

**規律6**: 画素が根拠の主張は原寸1:1で出す。モンタージュで論じない。

**窓の規則は出力を見る前に決める**(`m106` の `--crops` と同じ作法):
  - 面 = `T4` で墨まで **16px 以上**離れていた面のうち、
    **(あ) `T5` で線率 1.0000 になった群** と **(い) `T5` でも線率 0 のままの群**
    から、**それぞれ面積の中央値に当たる面**を1面ずつ
  - 窓 = その面の重心を中心とした**原寸 1200×1200**(`WIN`)
  - 1面につき3枚: ① 手描きの `線画` の墨 ② `rendering_lineart` だけの墨
    (`ink5 & ~ink4`)③ `T4` の境界
**面の選び方も窓の大きさも、点数を見てから動かさない。**

門は `m108` と同じ既知セル(`T4_*` / `T5_*` の8件)。**落ちたら画像は出さない。**
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402

OUT = Path('results/m109_far_face_crops')
PAGE = 'page0005'
WIN = 1200
FAR = 16.0
IC4 = {'closed': 482, 'med': 0.0000, 'lt05': 344, 'under': 5075}
IC5 = {'closed': 412, 'med': 1.0000, 'lt05': 78, 'under': 1809}


def med_face(rows):
    """面積の中央値に当たる面(**平均ではなく実在する1面**)。"""
    rs = sorted(rows, key=lambda r: r['area'])
    return rs[len(rs) // 2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if PAGE in q['page']][0]
    ex = {}
    ink4, bnd4, floor_px, n_line, n_edge = m106.build(p, byp, kinds, extras=ex)
    ink5 = ex['ink5']
    bnd5 = ink5 | (ex['E'] & m104.open_from(ink5))
    del ex['E']
    rows4, under4 = m106.faces(bnd4, ink4, floor_px)
    rows5, under5 = m106.faces(bnd5, ink5, floor_px)
    ok = True
    for nm, rows, under, ic in (('T4', rows4, under4, IC4),
                                ('T5', rows5, under5, IC5)):
        rs = sorted(r['ratio'] for r in rows)
        m = rs[len(rs)//2] if len(rs) % 2 else (rs[len(rs)//2-1]+rs[len(rs)//2])/2
        got = dict(closed=len(rows), med=round(m, 4),
                   lt05=sum(1 for r in rows if r['ratio'] < 0.5), under=under)
        good = all(abs(got[k] - v) < 1e-9 for k, v in ic.items())
        ok &= good
        print(f'  門 {nm} ' + ' '.join(f'{k}:{v}/{got[k]}' for k, v in ic.items())
              + ('  PASS' if good else '  **FAIL**'))
    if not ok:
        print('  **門が落ちた。画像は出さない**')
        return 1

    # `T4` の遠い面 → `T5` のどの面か(**画素の重なり。純度も印字する**)
    n4, lab4, st4, _ = cv2.connectedComponentsWithStats((~bnd4).astype(np.uint8), 4)
    n5, lab5, _, _ = cv2.connectedComponentsWithStats((~bnd5).astype(np.uint8), 4)
    info5 = {r['face']: r for r in rows5}
    far = [r for r in rows4 if r['d_ink'] >= FAR]
    keep = np.zeros(n4, bool)
    for r in far:
        keep[r['face']] = True
    sel = keep[lab4]
    a4 = lab4[sel].astype(np.int64)
    a5 = lab5[sel].astype(np.int64)
    del sel, lab4, lab5
    key, cnt = np.unique(a4 * n5 + a5, return_counts=True)
    del a4, a5
    best = {}
    for kk, c in zip(key.tolist(), cnt.tolist()):
        f4, f5 = kk // n5, kk % n5
        if c > best.get(f4, (0, -1))[0]:
            best[f4] = (c, f5)
    one, zero = [], []
    for r in far:
        c, f5 = best[r['face']]
        d5 = info5.get(f5)
        if d5 is None:
            continue
        r['t5_ratio'] = d5['ratio']
        (one if d5['ratio'] >= 0.9999 else zero if d5['ratio'] == 0.0
         else []).append(r)
    print(f'  遠い面 {len(far)} / T5 線率1.0 {len(one)} / T5 線率0 {len(zero)}')

    rend = ink5 & ~ink4                       # `rendering_lineart` だけの墨
    for tag, grp in (('bounded_by_rendering', one), ('bounded_by_neither', zero)):
        if not grp:
            print(f'  {tag}: 0面。出さない')
            continue
        f = med_face(grp)
        cx, cy = int(round(f['cx'])), int(round(f['cy']))
        H, W = ink4.shape
        x0 = max(0, min(W - WIN, cx - WIN // 2))
        y0 = max(0, min(H - WIN, cy - WIN // 2))
        print(f'  {tag}: 面{f["face"]} 面積{f["area"]:,}px 線率(T4) {f["ratio"]:.4f}'
              f' → (T5) {f["t5_ratio"]:.4f} 墨まで(T4) {f["d_ink"]:.0f}px'
              f' 窓 x{x0}–{x0+WIN}/y{y0}–{y0+WIN}')
        for nm, img in (('hand_lineart', ink4), ('rendering_lineart', rend),
                        ('boundary_t4', bnd4)):
            crop = (~img[y0:y0+WIN, x0:x0+WIN]).astype(np.uint8) * 255
            cv2.imwrite(str(OUT / f'{tag}__{nm}.png'), crop)
    print(f'  画像: {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
