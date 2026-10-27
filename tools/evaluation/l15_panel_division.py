# -*- coding: utf-8 -*-
"""作者のコマ区切りは原稿に在るか。データセットのコマとどれだけ食い違うか (診断。棒なし)。

作者の指摘「コマで区切ってる部分は線が途切れている」から。独立したコマ枠レイヤは無く、
区切りは面形レイヤ (keep4>=0.95) のベタ = 断ち落とし+枠 が担っている可能性がある。
その層の**補集合の連結成分**が作者のコマのはず。

データセット側: regions_clip_pairs_v3 の koma は再帰X-Yカットの矩形。
両者を IoU で突き合わせ、食い違いの大きさを出す。
"""
import csv, collections, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, A
from l14_arms import acc_add
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
KEEP4= '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_panel_division'
MIN_FRAC = 0.01     # コマとみなす最小面積 = 頁の 1%


def one_page(job):
    pid, work, cands, boxes = job
    out = []
    for lid, path, cov in cands:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        solid = np.zeros(ink.shape, bool)
        acc_add(solid, None, ink, 'solid')
        H, W = ink.shape
        floor = int(MIN_FRAC * H * W)
        # 補集合の 4連結成分 = 区切られた領域の候補
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            (~solid).astype(np.uint8), 4)
        ids = [i for i in range(1, n) if stats[i, A] >= floor]
        rec = dict(pid=pid, work=work, layer_id=lid, cov=round(cov, 4),
                   page_px=H * W, solid_px=int(solid.sum()),
                   holes=len(ids), n_koma=len(boxes))
        # データセットのコマ矩形と突き合わせる
        best = []
        for nm, x0, y0, x1, y1 in boxes:
            box = np.zeros((H, W), bool); box[y0:y1, x0:x1] = True
            bar = int(box.sum())
            bi, bk = 0.0, -1
            for i in ids:
                C = lab == i
                inter = int((C & box).sum())
                if inter == 0:
                    continue
                iou = inter / (stats[i, A] + bar - inter)
                if iou > bi:
                    bi, bk = iou, i
            best.append((nm, round(bi, 4), bar,
                         int(stats[bk, A]) if bk > 0 else 0))
        rec['koma_iou'] = ';'.join(f'{b:.3f}' for _, b, _, _ in best)
        rec['iou_med'] = round(st.median([b for _, b, _, _ in best]), 4) if best else 0.0
        rec['iou_max'] = round(max([b for _, b, _, _ in best], default=0.0), 4)
        out.append(rec)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fill, _ = arms('gated')
    k4 = {(r['page_id'], r['layer_id']): float(r['cov'])
          for r in csv.DictReader(open(KEEP4))}
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        boxes[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = []
    for (pid, w), v in sorted(fill.items()):
        if pid not in boxes:
            continue
        c = [(lid, p, k4[(pid, str(lid))]) for lid, p in sorted(v)
             if k4[(pid, str(lid))] >= 0.03]
        if c:
            jobs.append((pid, w, c, boxes[pid]))
    print(f'候補のある頁 {len(jobs)} / コマを持つ頁 {len(boxes)}  '
          f'(候補 = 面層で被覆>=0.03)')
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁', flush=True)
    with open(f'{OUT}/per_layer.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/per_layer.csv  {len(rows)} 層\n' + '=' * 84)
    print('作者の区切り候補と、データセットのコマ矩形の一致 (棒なし)')
    print(f'{"頁":28} {"L":>5} {"cov":>6} {"ベタ":>6} {"穴":>4} {"コマ":>5} '
          f'{"IoU中央":>8} {"IoU最大":>8}')
    for r in sorted(rows, key=lambda r: -r['iou_med'])[:25]:
        print(f'{r["pid"][-26:]:28} {r["layer_id"]:5d} {r["cov"]:6.3f} '
              f'{r["solid_px"]/r["page_px"]:6.3f} {r["holes"]:4d} {r["n_koma"]:5d} '
              f'{r["iou_med"]:8.4f} {r["iou_max"]:8.4f}')
    best = {}
    for r in rows:
        if r['pid'] not in best or r['iou_med'] > best[r['pid']]['iou_med']:
            best[r['pid']] = r
    v = sorted(x['iou_med'] for x in best.values())
    print(f'\n頁ごとの最良の層での IoU中央: 頁 {len(v)}  中央 {st.median(v):.4f}  '
          f'>=0.5 の頁 {sum(1 for x in v if x>=0.5)}  >=0.75 の頁 {sum(1 for x in v if x>=0.75)}')
    hv = [x['holes'] for x in best.values()]; kv = [x['n_koma'] for x in best.values()]
    print(f'穴の数 中央 {st.median(hv):.1f} / データセットのコマ数 中央 {st.median(kv):.1f}  '
          f'(一致する頁 {sum(1 for a,b in zip(hv,kv) if a==b)}/{len(hv)})')


if __name__ == '__main__':
    main()
