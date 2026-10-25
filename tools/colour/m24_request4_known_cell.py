"""M24: 依頼4 の納品の**既知セル照合**。**2GB を読む前に止まれる唯一の場所**(基盤)。

依頼4 の一覧の先頭8ページ(= パイロットと同じ8ページ)の **303層**について、
**届いたアルファがパイロットの原寸アルファと同じか**を突き合わせ、
**その上で標準を作り直して追補14 の数を再現するか**を見る。

**期待(納品が届く前に書いた。走らせるのは届いた後)**:

| | 内容 | 落ちたら |
|---|---|---|
| **KC-R4a** | 一覧の **303層が 303/303 揃っている** | 残りを読まない |
| **KC-R4b** | **アルファの画素がパイロットと一致** 303/303(バイト一致は印字するが**必須にしない** —— 再エンコードで変わりうる) | 残りを読まない |
| **KC-R4c** | 届いたラスタで標準を作り直すと **ベース183 / 成分823 / 面801** を再現し、**IC-M6(標準を予測として入れて region-F1 = 1.0)が 8/8** | 残りを読まない |

**KC-R4b でバイトが違い画素が同じ場合**は、**PNG の再エンコード**として記録し通す。
**画素が違えば止める** —— 「同じ層のアルファ」が同じでないなら、
**この納品で作る標準は追補14 の標準ではない。**

使い方: `python m24_request4_known_cell.py <納品のディレクトリ>`
(既定は `../lineart/data/incoming_manuscript/colour_request4`)。
ファイル名はパイロットと同じ `<page>_L<id>.png` を想定し、無ければ glob で探す。

**この器具は腕を採点しない。**
"""
import collections
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                       # noqa: E402
import m16_score_corrected_standard as m16             # noqa: E402

LIST = Path('results/m21_request4/request4_layer_list.csv')
PILOT = m12.D
DEFAULT_DIR = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4')
EXPECT = {'layers': 303, 'bases': 183, 'components': 823, 'areas': 801}


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def find(d, page, lid):
    cand = d / f'{page}_L{lid}.png'
    if cand.exists():
        return cand
    hits = sorted(d.glob(f'*{page}*_L{lid}.png')) or \
        sorted(d.glob(f'*{page}*L{lid}.*'))
    return hits[0] if hits else None


def main(argv):
    d = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR
    if not d.exists():
        print(f'納品のディレクトリが無い: {d}')
        print('依頼4 の納品が届いてから走らせる。')
        return 2
    want = [r for r in csv.DictReader(open(LIST, encoding='utf-8'))
            if r['既知セル'] == 'yes']
    pages = list(dict.fromkeys(r['page_path'] for r in want))
    man = {p['source']: p for p in __import__('json').load(
        open(m12.MANIFEST, encoding='utf-8'))['pages']}
    print(f'== KC-R4a 既知セルの層が揃っているか ==')
    print(f'  一覧の既知セル層 {len(want)}(期待 {EXPECT["layers"]})  '
          f'ページ {len(pages)}')
    pilot_files = {}
    for src, p in man.items():
        for l in p['layers']:
            pilot_files[(p['page'], str(l['layer_id']))] = PILOT / l['file']

    missing, byte_same, px_same, px_diff, no_pilot = [], 0, 0, [], 0
    for r in want:
        page = man[r['page_path']]['page']
        f = find(d, page, r['layer_id'])
        if f is None:
            missing.append((page, r['layer_id']))
            continue
        pf = pilot_files.get((page, r['layer_id']))
        if pf is None or not pf.exists():
            no_pilot += 1
            continue
        if md5(f) == md5(pf):
            byte_same += 1
            px_same += 1
            continue
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(pf), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape[:2] != b.shape[:2]:
            px_diff.append((page, r['layer_id'], 'shape'))
            continue
        aa = a[..., 3] if a.ndim == 3 and a.shape[2] == 4 else a
        bb = b[..., 3] if b.ndim == 3 and b.shape[2] == 4 else b
        if np.array_equal(aa, bb):
            px_same += 1
        else:
            px_diff.append((page, r['layer_id'],
                            int(np.count_nonzero(aa != bb))))
    print(f'  見つからない層 {len(missing)} -> '
          f'{"PASS" if not missing else "**FAIL**"}')
    for m in missing[:8]:
        print(f'    欠け {m[0]} L{m[1]}')
    print(f'\n== KC-R4b アルファが一致するか ==')
    print(f'  バイト一致 {byte_same} / 画素一致 {px_same} / '
          f'画素が違う {len(px_diff)} / パイロットに無い {no_pilot}')
    for x in px_diff[:8]:
        print(f'    画素が違う {x[0]} L{x[1]} 差分画素 {x[2]}')
    if missing or px_diff:
        print('\n**既知セルが合わない。残りの納品を読まない。**')
        return 1

    print(f'\n== KC-R4c 届いたラスタで標準を作り直す ==')
    # m16 の measure_standard をそのまま使うには manifest の file 名が要るので、
    # 納品側のファイル名に差し替えた manifest を組む(規則は m16 と同一)。
    import json
    mm = json.load(open(m12.MANIFEST, encoding='utf-8'))
    byp = {p['source']: p for p in mm['pages']}
    rowsrc = {r['page_path'] for r in want}
    tot = collections.Counter()
    bad = []
    for src in rowsrc:
        p = byp[src]
        for l in p['layers']:
            f = find(d, p['page'], str(l['layer_id']))
            if f is not None:
                l['file'] = str(f.relative_to(d)) if f.is_relative_to(d) \
                    else str(f)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        old_D = m12.D
        m12.D = d
        try:
            prows = [r for r in csv.DictReader(
                open('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv',
                     encoding='utf-8')) if r['page_path'] == src]
            (gt, garea), (gt_a, garea_a), st = m16.measure_standard(
                p, prows, grid, floor_px)
            known, _ = m12.score_from_small(gt_a, 1, gt_a, garea_a, floor_px)
        finally:
            m12.D = old_D
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        if abs(known['region_f1'] - 1.0) > 1e-9:
            bad.append((p['page'], known['region_f1']))
        print(f'  {p["page"]:44s} ベース{st["bases"]:>4} 成分{st["components"]:>4} '
              f'面{len(garea_a):>4} IC-M6 {known["region_f1"]:.4f}', flush=True)
    print(f'\n  合計 ベース {tot["bases"]}/{EXPECT["bases"]}  '
          f'成分 {tot["components"]}/{EXPECT["components"]}  '
          f'面 {tot["areas"]}/{EXPECT["areas"]}')
    ok = (tot['bases'] == EXPECT['bases']
          and tot['components'] == EXPECT['components']
          and tot['areas'] == EXPECT['areas'] and not bad)
    print(f'  IC-M6 1.0 でないページ {len(bad)} {bad[:4]}')
    print(f'\n**KC-R4c -> {"PASS" if ok else "FAIL"}**')
    if not ok:
        print('**既知セルが合わない。残りの納品を読まない。**')
        return 1
    print('\n既知セル3つとも PASS。残りの納品を読んでよい。')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
