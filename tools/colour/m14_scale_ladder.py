"""M14: **縮小率が A0 の点をどれだけ決めるか**を、同じページの中で測る。

追補7 §G5 で `page0064`(覆い比 0.994、**s=1/8**)の recall が 0.000、
`page0003`/`page0005`(覆い 0.961/0.959、**s=1/4**)が 0.692/0.530 だった。
**これは縮小率の差とページの差が分かれていない。** 同じページを**さらに縮小**して
同じ標準に当てれば分かれる(標準は原寸のまま動かさない)。

方法: プレビューを `cv2.INTER_AREA` で 1/d(d = 1, 2, 4)に縮め、A0 を作り、
**s*d 倍でブロック拡大して原寸の標準に当てる**(追補7 R1 のまま)。
床は縮めた後の画素数に対する 1e-5(= ページ面積の 1e-5。割合なので不変)。

**これは CSP の `CanvasPreview` の縮小ではなく、こちらで作った縮小である。**
配達されたプレビューがどう作られたかは選べないので、**縮小率の効きの向きを測る
代用**にしかならない。向きだけを読む。

既知セル: **d=1 の行は追補7 §G5 の A0 の列を再現しなければならない**
(`EXPECT_D1`。再現しなければ以降の行を読まない)。

**登録集合8ページなので主張は出ない。** 出すのは検定集合への予測(P18)と、
**抽出側に求めるプレビューの縮小率の根拠**である。
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                      # noqa: E402

D = m12.D
OUT = Path('results/m14_scale'); OUT.mkdir(parents=True, exist_ok=True)
DOWN = (1, 2, 4)

# 追補7 §G5 の A0 の列(ページ名の末尾 -> (照合, recall, region-F1))
EXPECT_D1 = {'sunshine2_page0019': (0, 0.0000, 0.0000),
             'sunshine2_page0002': (1, 0.2000, 0.0006),
             'illust201710_page0003': (9, 0.6923, 0.1029),
             'illust201710_page0005': (35, 0.5303, 0.0657),
             'skima10_page0064': (0, 0.0000, 0.0000),
             'skima10_page0073': (0, 0.0000, 0.0000),
             'skima11_skima12H': (5, 0.0352, 0.0031),
             'skima11_skima12chara': (17, 0.0955, 0.0324)}


def main():
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    rows, t0 = [], time.time()
    for p in pages:
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        im = cv2.imread(str(D / p['preview']['file']), cv2.IMREAD_COLOR)
        gt, garea, st = m12.measure_standard(p, grid, floor_px)
        row = {'page': p['page'], 'scale': s, 'gt_areas': len(garea)}
        for d in DOWN:
            if d == 1:
                sm = im
            else:
                sm = cv2.resize(im, (pw // d, ph // d), interpolation=cv2.INTER_AREA)
            h, w = sm.shape[:2]
            lab = m12.equal_value_components(m12.quantise(sm, m12.LEVELS))
            a0, _ = m12.filtered_labels(lab, max(1, round(m12.FLOOR * h * w)))
            # 拡大率は s*d。端は R2 と同じに切る。
            r, _ = m12.score_from_small(a0, s * d, gt, garea, floor_px)
            row[f'd{d}_eff'] = s * d
            row[f'd{d}_n'] = r['n_pred']
            row[f'd{d}_matched'] = r['matched']
            row[f'd{d}_rec'] = round(r['recall'], 4)
            row[f'd{d}_f1'] = round(r['region_f1'], 4)
            del lab, a0
        rows.append(row)
        print(f'{p["page"]:44s} gt {len(garea):>4} ' + ' '.join(
            f'1/{row[f"d{d}_eff"]}: rec {row[f"d{d}_rec"]:.4f} '
            f'(n={row[f"d{d}_n"]})' for d in DOWN) +
            f'  {time.time()-t0:6.1f}s', flush=True)
        del gt

    bad = []
    for r in rows:
        k = next(k for k in EXPECT_D1 if r['page'].endswith(k))
        want = EXPECT_D1[k]
        got = (r['d1_matched'], r['d1_rec'], r['d1_f1'])
        if got[0] != want[0] or abs(got[1] - want[1]) > 5e-4 or \
                abs(got[2] - want[2]) > 5e-4:
            bad.append((r['page'], got, want))
    print(f'\n既知セル(d=1 が追補7 §G5 を再現): '
          f'{len(rows)-len(bad)}/{len(rows)} -> {"PASS" if not bad else "FAIL"}')
    for b in bad:
        print('  FAIL', b)
    if bad:
        return 1

    import csv
    with open(OUT / 'scale_ladder.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print('\n== 同じページ・同じ標準で縮小率だけを動かした(recall / 面数) ==')
    print(f'{"page":30s} {"gt":>4} ' + ''.join(
        f'{"1/"+str(d)+"x":>22s}' for d in DOWN))
    for r in rows:
        print(f'{r["page"][-28:]:30s} {r["gt_areas"]:>4} ' + ''.join(
            f'  1/{r[f"d{d}_eff"]:<2} {r[f"d{d}_rec"]:>7.4f} n={r[f"d{d}_n"]:<6}'
            for d in DOWN))
    print('\nこれは CSP の縮小ではなく INTER_AREA の縮小である。向きだけを読む。')
    print('登録集合なので主張は出ない。出すのは検定集合への予測と、')
    print('抽出側に求めるプレビューの縮小率の根拠である。')
    print(f'\nwrote {OUT/"scale_ladder.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
