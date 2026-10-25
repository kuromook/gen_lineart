"""M27: **作者に見てもらう原寸1:1の切り出し**を作る(追補6 §F4 の偏りを測る唯一の道)。

問いは1つだけ: **標準が「1つの面」と数えた箇所は、あなたにとっても1つの面ですか。**
標準 = シルエット層(クリップの連鎖のベース)の **8連結成分**、床 1e-5。
§F4 で登録した限界: **隣り合う面(襟と身頃)は1つの成分になる** ので、
**標準は作者の面を「分け足りない」側に偏っている。** その**大きさ**は作者しか測れない。

**選び方(走らせる前に登録する。結果を見てから選び直さない)**:

| 種類 | 選ぶ規則 | 枚数 |
|---|---|---|
| **K1 色の境目が面の中にある** | 成分の中をプレビュー(1/8)で見て、`>>4` 量子化した色が **share >= 0.15 で2つ以上**。切り出しは**その色の境目**を中心にする | 2 |
| **K2 巨大な面** | ページで最大面積の成分 | 1 |
| **K3 対照(分けて正しい見込み)** | 成分の中の最頻色の share が **>= 0.90**(1色 = 作者の1面の見込み) | 1 |

- **作品が違うページから選ぶ**(同じ作品に偏らせない)。1ページから最大1件
- **縮小しない。** 画素は原寸1:1、窓は最大 1400×1400。縮小モンタージュは作らない(規律6)
- 2枚組で出す: **`_raw`(何も描き込まない原寸)** と
  **`_marked`(成分の輪郭だけマゼンタ)**。**判定は `_raw` で、問いは `_marked` で示す**
- 合成するのは**その成分の元の層だけ**(フォルダーのベースなら子孫も)。
  **ページ全体の合成は原寸では存在しない**(追補2 §B4)ので、作らない

**既知セル**: 8ページの標準が **ベース183 / 成分823 / 面801**(KC-R4c と同じ値)を再現すること。
再現しなければ切り出しを書かない。
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                        # noqa: E402
import m16_score_corrected_standard as m16              # noqa: E402

DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4_known8')
CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
OUT = Path('results/m27_author_crops'); OUT.mkdir(parents=True, exist_ok=True)
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}
WIN = 1400
MINFRAC = 0.002          # 成分がキャンバスのこれ未満なら選ばない
Q = 4                    # 色の量子化(>>4 = 16階調)


def quant(img):
    return (img[..., 0] >> Q).astype(np.int32) * 256 + \
           (img[..., 1] >> Q).astype(np.int32) * 16 + (img[..., 2] >> Q)


def composite(ids, files, box, win):
    """その成分の元の層だけを白地に重ねて、窓を原寸で切り出す。"""
    y0, x0, y1, x1 = win
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    for i in ids:
        im = cv2.imread(str(DELIV / files[i]), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        sub = im[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        a = sub[..., 3:4].astype(np.float32) / 255.0
        out = (sub[..., :3].astype(np.float32) * a +
               out.astype(np.float32) * (1 - a)).astype(np.uint8)
        del im, sub, a
    return out


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
             if l.strip()]
    m12.D = DELIV
    tot, cases = collections.Counter(), []
    for p in pages:
        if not p['layers']:
            continue
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (gt, garea), (gt_a, garea_a), st = m16.measure_standard(
            p, byp[p['source']], grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        prev = cv2.imread(str(DELIV / p['preview']['file']), cv2.IMREAD_UNCHANGED)
        pq = quant(prev)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        work = p['page']
        best = {}
        for c in sorted(comps, key=lambda c: -c['area']):
            if c['area'] < MINFRAC * grid[0] * grid[1]:
                continue
            y, x, h, w = c['box']
            sm = c['mask'][::s, ::s] if s > 1 else c['mask']
            ys, xs = y // s, x // s
            pm = np.zeros(pq.shape, bool)
            hh, ww = sm.shape
            pm[ys:ys + hh, xs:xs + ww] = sm[:pq.shape[0] - ys, :pq.shape[1] - xs]
            vals = pq[pm]
            if vals.size < 50:
                continue
            u, cnt = np.unique(vals, return_counts=True)
            share = cnt / cnt.sum()
            top = share.max()
            big = u[share >= 0.15]
            rec = dict(c, page=p, work=work, top=float(top),
                       ncol=int(len(big)), files=files)
            if len(big) >= 2 and 'K1' not in best:
                # 色の境目: 大きい色2つが隣り合う画素を探す
                m1 = pm & (pq == big[np.argsort(-share[share >= 0.15])[0]])
                m2 = pm & (pq == big[np.argsort(-share[share >= 0.15])[1]])
                d1 = cv2.dilate(m1.astype(np.uint8), np.ones((3, 3), np.uint8))
                edge = (d1 > 0) & m2
                if edge.sum() >= 20:
                    yy, xx = np.nonzero(edge)
                    rec['centre'] = (int(yy.mean()) * s, int(xx.mean()) * s)
                    best['K1'] = rec
            if 'K2' not in best:
                rec2 = dict(rec)
                rec2['centre'] = (y + h // 2, x + w // 2)
                best['K2'] = rec2
            if top >= 0.90 and 'K3' not in best:
                rec3 = dict(rec)
                rec3['centre'] = (y + h // 2, x + w // 2)
                best['K3'] = rec3
        for k, rec in best.items():
            rec['kind'] = k
            cases.append(rec)
        del prev, pq, gt, gt_a, comps
        print(f'  {p["page"]:44s} ベース{st["bases"]:>4} 成分{st["components"]:>4} '
              f'面{len(garea_a):>4}  候補 {sorted(best)}', flush=True)

    print(f'\n== 既知セル ==')
    ok = all(tot[k] == EXPECT[k] for k in EXPECT)
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。切り出しを書かない。**')
        return 1

    # 作品が違うページから、K1 2枚 / K2 1枚 / K3 1枚
    want = {'K1': 2, 'K2': 1, 'K3': 1}
    taken, used_pages, rowsout = collections.Counter(), set(), []
    for k in ('K1', 'K2', 'K3'):
        for rec in sorted([c for c in cases if c['kind'] == k],
                          key=lambda c: -c['area']):
            if taken[k] >= want[k] or rec['work'] in used_pages:
                continue
            taken[k] += 1
            used_pages.add(rec['work'])
            rowsout.append(rec)
    print(f'\n== 切り出す {len(rowsout)} 件 ==')
    for n, rec in enumerate(rowsout, start=1):
        p = rec['page']
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        H, W = ph * s, pw * s
        cy, cx = rec['centre']
        y0 = max(0, min(H - WIN, cy - WIN // 2))
        x0 = max(0, min(W - WIN, cx - WIN // 2))
        y1, x1 = min(H, y0 + WIN), min(W, x0 + WIN)
        win = (y0, x0, y1, x1)
        img = composite(rec['ids'], rec['files'], rec['box'], win)
        y, x, h, w = rec['box']
        full = np.zeros((H, W), bool)
        full[y:y + h, x:x + w] = rec['mask']
        sub = full[y0:y1, x0:x1].astype(np.uint8)
        ed = cv2.morphologyEx(sub, cv2.MORPH_GRADIENT,
                              np.ones((3, 3), np.uint8)) > 0
        mk = img.copy()
        mk[ed] = (255, 0, 255)
        base = (f'case{n}_{rec["kind"]}_{p["page"]}_L{rec["layer_id"]}'
                f'_x{x0}_y{y0}_{x1 - x0}x{y1 - y0}')
        cv2.imwrite(str(OUT / f'{base}_raw.png'), img)
        cv2.imwrite(str(OUT / f'{base}_marked.png'), mk)
        print(f'  {rec["kind"]}  {p["page"]} L{rec["layer_id"]} '
              f'面積{rec["area"]:>9} 最頻色{rec["top"]:.2f} 色{rec["ncol"]}  '
              f'窓 x{x0} y{y0} {x1 - x0}x{y1 - y0}')
        rowsout[n - 1]['file'] = f'{base}_raw.png'
        del full, sub, img, mk
    with open(OUT / 'crops.csv', 'w', newline='', encoding='utf-8') as fh:
        w_ = csv.writer(fh)
        w_.writerow(['種類', 'page_path', 'layer_id', '成分の面積px',
                     '最頻色の share', 'share>=0.15 の色数', '窓', 'ファイル'])
        for rec in rowsout:
            w_.writerow([rec['kind'], rec['page']['source'], rec['layer_id'],
                         rec['area'], f'{rec["top"]:.3f}', rec['ncol'],
                         rec['file'].split('_x')[-1].replace('_raw.png', ''),
                         rec['file']])
    print(f'\nwrote {OUT}/ ({len(rowsout) * 2} 枚 + crops.csv)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
