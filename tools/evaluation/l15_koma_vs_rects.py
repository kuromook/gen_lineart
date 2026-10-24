# -*- coding: utf-8 -*-
"""作者のコマ(納品されたマスク)と、データセットの矩形コマの食い違い (診断。棒なし)。

前回 (l15_koma_mismatch.py) は枠の図形が無かったので「フォルダー配下の層の墨の和」で
代用した。**単位が誤っていた**: 抽出側の納品で `コマ N` フォルダー186個のマスクは
8連結成分で **322個**(1フォルダーに複数コマ)。今回は成分を単位にし、量も墨ではなく
**面積**で測る。

枠の選別は**名前ではなく構造**: マニフェストの `ink.layers_with_png > 0`
(= 配下に描画層がある枠)。`モザイク`(頁全体を囲う)と `/text` のナレーション箱は
この条件でほぼ落ちる。入れ子は overlap_pairs で数えて印字する。
"""
import csv, collections, json, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
DEL  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_koma_vs_rects'
MIN_FRAC = 0.002          # 成分の床 = 頁面積の 0.2%


def one_page(job):
    pid, work, frames, boxes, canvas = job
    H, W = canvas
    floor = int(MIN_FRAC * H * W)
    rect = np.zeros((H, W), np.int32)
    for i, (nm, x0, y0, x1, y1) in enumerate(boxes, 1):
        rect[y0:y1, x0:x1] = i
    rows = []
    comp_masks = []
    for fr in frames:
        p = os.path.join(DEL, work, 'koma', fr['file'])
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mask = m > 127
        if mask.shape != (H, W):
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < floor:
                continue
            C = lab == k
            a = int(C.sum())
            hit = rect[C]
            vals, cnt = np.unique(hit[hit > 0], return_counts=True)
            srt = sorted(zip(cnt.tolist(), vals.tolist()), reverse=True)
            inside = int(cnt.sum()) if len(cnt) else 0
            cum = 0; n90 = 0
            for c, _ in srt:
                cum += c; n90 += 1
                if inside and cum >= 0.9 * inside:
                    break
            rows.append(dict(pid=pid, work=work, layer_id=fr['layer_id'],
                             name=fr['name'], anc=fr['ancestors'], comp=k, area=a,
                             bbox=f"{stats[k,cv2.CC_STAT_LEFT]},{stats[k,cv2.CC_STAT_TOP]},"
                                  f"{stats[k,cv2.CC_STAT_LEFT]+stats[k,cv2.CC_STAT_WIDTH]},"
                                  f"{stats[k,cv2.CC_STAT_TOP]+stats[k,cv2.CC_STAT_HEIGHT]}",
                             inside=inside,
                             outside_frac=round(1 - inside / a, 4),
                             top_share=round((srt[0][0] / a) if srt else 0.0, 4),
                             n_rect=len(srt), n_rect_90=n90,
                             top_rect=(boxes[srt[0][1] - 1][0] if srt else '')))
            comp_masks.append((len(rows) - 1, C))
    # 矩形側: 各矩形が作者の成分にどれだけ覆われるか
    rrows = []
    for i, (nm, x0, y0, x1, y1) in enumerate(boxes, 1):
        ar = (y1 - y0) * (x1 - x0)
        per = []
        for idx, C in comp_masks:
            c = int(C[y0:y1, x0:x1].sum())
            if c:
                per.append((c, idx))
        per.sort(reverse=True)
        cov = sum(c for c, _ in per)
        rrows.append(dict(name=nm, pid=pid, work=work, area=ar,
                          covered=round(min(cov / ar, 1.0), 4),
                          top_share=round((per[0][0] / ar) if per else 0.0, 4),
                          n_comp=len(per)))
    return rows, rrows


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        boxes[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = []
    kept = collections.Counter(); dropped = collections.Counter()
    for work in ('ako6', 'ako7'):
        d = json.load(open(f'{DEL}/{work}/koma/koma_manifest.json'))
        for p in d['pages']:
            pid = p['page']
            if pid not in boxes:
                dropped['L12 の248コマに無い頁'] += 1; continue
            cv_ = p['canvas']
            H, W = (cv_[1], cv_[0]) if cv_[0] > cv_[1] else (cv_[0], cv_[1])
            sel = []
            for fr in p['frames']:
                ink = fr.get('ink') or {}
                if int(ink.get('layers_with_png') or 0) > 0:
                    sel.append(fr); kept['配下に描画層がある枠'] += 1
                else:
                    dropped[f'配下に描画層が無い枠 ({fr["name"]})'] += 1
            if sel:
                jobs.append((pid, work, sel, boxes[pid], (H, W)))
    print('枠の選別 (構造で選ぶ。名前では切らない)')
    for k, v in kept.most_common():
        print(f'  通した {v:5d}  {k}')
    tot = sum(v for k, v in dropped.items())
    print(f'  落とした {tot}')
    for k, v in dropped.most_common(6):
        print(f'    {v:5d}  {k}')
    print(f'  頁 {len(jobs)}')
    R, RR = [], []
    with Pool(4) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(one_page, jobs), 1):
            R += a; RR += b
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  作者のコマ成分 {len(R)}', flush=True)
    for nm, rows in (('per_component', R), ('per_rect', RR)):
        with open(f'{OUT}/{nm}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'-> {OUT}/{nm}.csv  {len(rows)} 行')
    print('\n' + '=' * 84)
    print('作者のコマ成分 -> データセットの矩形 (面積で。棒なし)')
    for w in ('ako6', 'ako7'):
        s = [r for r in R if r['work'] == w]
        if not s:
            continue
        print(f'  {w}: 成分 {len(s)}  面積中央 {st.median([r["area"] for r in s]):,.0f}px')
        print(f'      top_share 中央 {st.median([r["top_share"] for r in s]):.4f}  '
              f'>=0.9 {sum(1 for r in s if r["top_share"]>=0.9)/len(s):.3f}  '
              f'>=0.5 {sum(1 for r in s if r["top_share"]>=0.5)/len(s):.3f}')
        print(f'      面積の9割に要る矩形の数 中央 {st.median([r["n_rect_90"] for r in s]):.1f} / '
              f'触れている数 中央 {st.median([r["n_rect"] for r in s]):.1f}')
        oa = sum(r['area'] - r['inside'] for r in s); ta = sum(r['area'] for r in s)
        print(f'      どの矩形にも入らない面積 {oa:,} / {ta:,} = {oa/ta:.1%}  '
              f'外>=0.95 の成分 {sum(1 for r in s if r["outside_frac"]>=0.95)}')
    print('\nデータセットの矩形 -> 作者のコマ成分')
    for w in ('ako6', 'ako7'):
        s = [r for r in RR if r['work'] == w]
        if not s:
            continue
        print(f'  {w}: 矩形 {len(s)}  作者の成分で覆われる割合 中央 '
              f'{st.median([r["covered"] for r in s]):.4f}  '
              f'最大の成分が占める割合 中央 {st.median([r["top_share"] for r in s]):.4f}')
        print(f'      触れている成分の数 中央 {st.median([r["n_comp"] for r in s]):.1f}  '
              f'0個の矩形 {sum(1 for r in s if r["n_comp"]==0)}')
    print('\n取りこぼしの大きい作者のコマ成分 上位')
    for r in sorted(R, key=lambda r: -(r['area'] - r['inside']))[:10]:
        print(f'  外 {r["outside_frac"]:.2f} 面積 {r["area"]:10,} top {r["top_share"]:.2f}  '
              f'{r["pid"][-22:]:24} L{r["layer_id"]:>4} {r["name"][:10]:10} bbox {r["bbox"]}')


if __name__ == '__main__':
    main()
