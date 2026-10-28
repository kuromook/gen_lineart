"""M24: 依頼4 の納品の**既知セル照合**。**2GB を読む前に止まれる唯一の場所**(基盤)。

依頼4 の一覧の先頭8ページ(= パイロットと同じ8ページ)の **303層**について、
**届いたアルファがパイロットの原寸アルファと同じか**を突き合わせ、
**その上で標準を作り直して追補14 の数を再現するか**を見る。

**期待(納品が届く前に書いた。走らせるのは届いた後)**:

| | 内容 | 落ちたら |
|---|---|---|
| **KC-R4a** | 一覧の **303層が 303/303 揃っている** | 残りを読まない |
| **KC-R4b** | **アルファの画素がパイロットと一致** 303/303(バイト一致は印字するが**必須にしない** —— 再エンコードで変わりうる) | 残りを読まない |
| **KC-R4c** | 届いたラスタで標準を作り直すと **ベース183 / 成分823 / 面801**(`standard_all`)を再現し、**IC-M6(標準を予測として入れて region-F1 = 1.0)が 8/8** | 残りを読まない |
| **KC-R4c′**(2026-10-28 に**足した**) | 同じラスタで **`standard_visible` が ベース152 / 成分503 / 面483** を再現し、**その IC-M6 も 8/8** | 残りを読まない |

**KC-R4c′ は棒を緩めずに足したものである**(追補111 §E)。
**作者が 2026-10-28 に `standard_visible` を標準に選んだ**ので、
**納品が再現しなければならない標準が2つになった。** 片方だけ合う納品は通さない。

**KC-R4b でバイトが違い画素が同じ場合**は、**PNG の再エンコード**として記録し通す。
**画素が違えば止める** —— 「同じ層のアルファ」が同じでないなら、
**この納品で作る標準は追補14 の標準ではない。**

使い方: `python m24_request4_known_cell.py <納品のディレクトリ>`
(既定は `../lineart/data/incoming_manuscript/colour_request4`)。

**結合は `source`(= `page_path`)と `layer_id`。ファイル名で結ばない。**
2026-10-25 23:30 の初回走行は、**パイロットのマニフェストの `page` 文字列から
納品のファイル名を組み立てて** 303層のうち 255層を「欠け」と出した。
納品側は 303/303 バイト一致を報告していたので、**疑うべきは配達ではなく比較器**
だった(`formats` を走査子とリストで比べた 2026-10-25 の誤りと同じ形)。
パイロットと納品では `page` 文字列が8ページ中6ページで違う
(`illust201710_illust201710` / `022_` / `skima11_skima11`)。
**いまは納品のマニフェスト(`colour_layers_pages.jsonl`)の `file` を使う。**
**棒(KC-R4a/b/c の内容)は動かしていない。直したのは器具だけ。**

**この器具は腕を採点しない。**
"""
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                       # noqa: E402
import m16_score_corrected_standard as m16             # noqa: E402
import m77_standard_visible as m77                    # noqa: E402

LIST = Path('results/m21_request4/request4_layer_list.csv')
PILOT = m12.D
DEFAULT_DIR = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4')
# **既知セルは世代ごとに1つ。片方に差し替えない**(追補111 §E)。
# **作者は 2026-10-28 に `standard_visible` を標準に選んだ**ので、
# **両方の再現を門にする** —— `standard_all` が合って `standard_visible` が合わない
# (またはその逆)なら、**届いたラスタは追補14 の標準でも新しい標準でもない**。
EXPECT = {'layers': 303, 'bases': 183, 'components': 823, 'areas': 801}
EXPECT_VIS = {'bases': 152, 'components': 503, 'areas': 483}   # 追補105/111


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def delivered(d):
    """納品のマニフェストを `source` で引けるようにする。**名前で結ばない。**"""
    f = d / 'colour_layers_pages.jsonl'
    if not f.exists():
        return None
    pages = {}
    for line in open(f, encoding='utf-8'):
        line = line.strip()
        if line:
            p = json.loads(line)
            pages[p['source']] = p
    return pages


def main(argv):
    d = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR
    if not d.exists():
        print(f'納品のディレクトリが無い: {d}')
        print('依頼4 の納品が届いてから走らせる。')
        return 2
    want = [r for r in csv.DictReader(open(LIST, encoding='utf-8'))
            if r['既知セル'] == 'yes']
    pages = list(dict.fromkeys(r['page_path'] for r in want))
    man = {p['source']: p for p in json.load(
        open(m12.MANIFEST, encoding='utf-8'))['pages']}
    dm = delivered(d)
    if dm is None:
        print(f'納品のマニフェストが無い: {d/"colour_layers_pages.jsonl"}')
        return 2
    dlay = {(src, str(l['layer_id'])): l
            for src, p in dm.items() for l in p['layers']}
    print(f'== KC-R4a 既知セルの層が揃っているか ==')
    print(f'  一覧の既知セル層 {len(want)}(期待 {EXPECT["layers"]})  '
          f'ページ {len(pages)}  納品のマニフェストのページ {len(dm)}')
    pilot_files = {}
    for src, p in man.items():
        for l in p['layers']:
            pilot_files[(src, str(l['layer_id']))] = PILOT / l['file']

    missing, byte_same, px_same, px_diff, no_pilot = [], 0, 0, [], 0
    for r in want:
        src = r['page_path']
        l = dlay.get((src, r['layer_id']))
        f = d / l['file'] if l is not None and l.get('file') else None
        if f is None or not f.exists():
            missing.append((src, r['layer_id']))
            continue
        pf = pilot_files.get((src, r['layer_id']))
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
            px_diff.append((src, r['layer_id'], 'shape'))
            continue
        aa = a[..., 3] if a.ndim == 3 and a.shape[2] == 4 else a
        bb = b[..., 3] if b.ndim == 3 and b.shape[2] == 4 else b
        if np.array_equal(aa, bb):
            px_same += 1
        else:
            px_diff.append((src, r['layer_id'],
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
    rowsrc = {r['page_path'] for r in want}
    tot, vtot = collections.Counter(), collections.Counter()
    bad, vbad = [], []
    for src in sorted(rowsrc):
        p = dm[src]
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
            # **`standard_visible`(作者が選んだ標準)でも作り直す**
            vrows, _ = m77.visible_rows(prows)
            (_, _), (vt_a, varea_a), vst = m16.measure_standard(
                p, vrows, grid, floor_px)
            vknown, _ = m12.score_from_small(vt_a, 1, vt_a, varea_a, floor_px)
            del vt_a
        finally:
            m12.D = old_D
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        vtot['bases'] += vst['bases']
        vtot['components'] += vst['components']
        vtot['areas'] += len(varea_a)
        if abs(known['region_f1'] - 1.0) > 1e-9:
            bad.append((p['page'], known['region_f1']))
        if abs(vknown['region_f1'] - 1.0) > 1e-9:
            vbad.append((p['page'], vknown['region_f1']))
        print(f'  {p["page"]:44s} ベース{st["bases"]:>4} 成分{st["components"]:>4} '
              f'面{len(garea_a):>4} IC-M6 {known["region_f1"]:.4f}  '
              f'| visible ベース{vst["bases"]:>4} 成分{vst["components"]:>4} '
              f'面{len(varea_a):>4} IC-M6 {vknown["region_f1"]:.4f}', flush=True)
    print(f'\n  standard_all      ベース {tot["bases"]}/{EXPECT["bases"]}  '
          f'成分 {tot["components"]}/{EXPECT["components"]}  '
          f'面 {tot["areas"]}/{EXPECT["areas"]}')
    print(f'  **standard_visible**  ベース {vtot["bases"]}/{EXPECT_VIS["bases"]}  '
          f'成分 {vtot["components"]}/{EXPECT_VIS["components"]}  '
          f'面 {vtot["areas"]}/{EXPECT_VIS["areas"]}  '
          f'(**作者が選んだ標準**)')
    ok = (tot['bases'] == EXPECT['bases']
          and tot['components'] == EXPECT['components']
          and tot['areas'] == EXPECT['areas'] and not bad
          and all(vtot[k] == v for k, v in EXPECT_VIS.items()) and not vbad)
    print(f'  IC-M6 1.0 でないページ  all {len(bad)} {bad[:4]} / '
          f'visible {len(vbad)} {vbad[:4]}')
    print(f'\n**KC-R4c -> {"PASS" if ok else "FAIL"}**')
    if not ok:
        print('**既知セルが合わない。残りの納品を読まない。**')
        return 1
    print('\n既知セル3つとも PASS。残りの納品を読んでよい。')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
