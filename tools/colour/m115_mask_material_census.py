"""**納品されたレイヤーマスクで、原案の材料を数え直す。点数は出さない。**

納品: `reply_m_layer_mask_raster_20261101.md`(抽出側、2026-11-01 20:03)。
**111枚 + manifest、sha256 は 112/112 一致。**

**この器具が答えるのは1つだけ** —— **原案(「塗りの入っていないマスク層だけで面を見つける」)の
材料は、マスクが届いた後に何枚あるか。** 追補182 §E では「4頁で20枚、うち15枚は形が取れない」
だった。**形はマスクの側に在ったのだから、数え直す。**

**最初に当てる門は向きである** —— 納品の説明は **「白地に黒。マスクの値をそのまま濃さにした
(黒 = 255 = 層が見える)」**。**PNG に入っている値がマスクの値そのものか、反転(255 − 値)かは、
この文からは決まらない。** 取り違えると**全部のマスクが裏返り、静かに通る**。
**だから `mask_px_ge128`(先方の manifest の数)と突き合わせて、どちらの読みが合うかを
111枚で決める**(規律9。既知セルはこちらの外から来た数である)。

母集団は **`pop_4pages_colour`**(追補163・182)。**`page0019` は母集団の外だが門に使う。**
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m16_score_corrected_standard as m16                           # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m77_standard_visible as m77                                   # noqa: E402

MD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/dataset_colour_pilot_masks_20261101')
MM = MD / 'layer_masks_manifest.json'
CM = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')
OUT = Path('results/m115_mask_material')
FLOOR = 1e-5                      # 追補6 §F3。画布の面積に対する割合
POP = ('page0002', 'page0003', 'skima12H', 'skima12chara')
OUTSIDE = 'page0019'              # 母集団の外。門に使う
MASKFILL = 'mask_fill'

# 先方の便の本文に書かれていた数(**こちらのパイプラインの外**)
IC1_TOTAL = 111
IC1_TEXT_BY_TYPE = {2: 82, 3: 19, 4098: 2}   # **本文の内訳。合計すると 112 になる**
IC1_FOLDER = 9
IC2_INIT = {255: 85, 0: 26}
IC3_P19_MASKFILL = 65             # こちらが先方に伝えた `mask_fill` の数
IC3_P19_WITH_FORM = 19            # 65枚のうち形があるのは19枚(先方の実測)
IC5_OURS_VISIBLE_LEAF_HASMASK = 20   # 追補182 §E。4頁の可視の葉で has_mask=True


def key(page_path):
    return Path(page_path.replace('\\', '/')).stem


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mm = json.load(open(MM, encoding='utf-8'))['layers']
    byp = m16.page_rows()
    cmf = {p['page']: p for p in json.load(open(CM, encoding='utf-8'))['pages']}

    # ---------------------------------------------------------------- 門
    print(f'納品 {MD.name} / マスク {len(mm)} 枚 / 母集団 `pop_4pages_colour`')
    print(f'IC1 枚数 期待 {IC1_TOTAL} 実測 {len(mm)}'
          f'  {"PASS" if len(mm) == IC1_TOTAL else "FAIL"}')
    nf = sum(1 for r in mm if r['folder'])
    tf = collections.Counter(int(r['layer_type']) for r in mm if r['folder'])
    tl = collections.Counter(int(r['layer_type']) for r in mm if not r['folder'])
    ok = (sum(tf.values()) + sum(tl.values()) == len(mm)) and nf == IC1_FOLDER
    print(f'IC1b フォルダー 期待 {IC1_FOLDER} 実測 {nf} / '
          f'葉の種別 {dict(sorted(tl.items()))} + フォルダーの種別 {dict(sorted(tf.items()))} '
          f'= {len(mm)}  {"PASS" if ok else "FAIL"}')
    s = sum(IC1_TEXT_BY_TYPE.values()) + IC1_FOLDER
    print(f'    **先方の本文の内訳は合計 {s} で、納品の {len(mm)} と 1 枚ずれる**'
          f'(本文 `layer_type=2` {IC1_TEXT_BY_TYPE[2]} 対 実測 葉 {tl[2]} + フォルダー {tf[2]})。'
          '**納品そのものは内部で辻褄が合っている。本文の数を照会する**')
    ini = collections.Counter(int(r['init_color']) for r in mm)
    ok = all(ini[k] == v for k, v in IC2_INIT.items())
    print(f'IC2 `init_color` 期待 {IC2_INIT} 実測 {dict(sorted(ini.items()))}'
          f'  {"PASS" if ok else "FAIL"}')

    # IC4 向き(**取り違えると全部裏返る**)
    direct = inverted = 0
    for r in mm:
        g = cv2.imread(str(MD / r['mask_file']), cv2.IMREAD_UNCHANGED)
        if g is None:
            print(f'  **読めない: {r["mask_file"]}**')
            continue
        if g.ndim == 3:
            g = g[..., 0]
        n_ge = int((g >= 128).sum())
        if n_ge == r['mask_px_ge128']:
            direct += 1
        elif int((g <= 127).sum()) == r['mask_px_ge128']:
            inverted += 1
        r['_png_ge128'] = n_ge       # 門のためだけに持つ。**分類には使わない**
    print(f'IC4 向き: **PNG の値 >= 128 が `mask_px_ge128` と一致 {direct} / '
          f'反転が一致 {inverted} / 111**  '
          f'{"PASS" if max(direct, inverted) == len(mm) else "FAIL"}')
    if direct == len(mm):
        print('    **PNG に入っているのはマスクの値そのもの。>= 128 が「層が見える」側である**')
    elif inverted == len(mm):
        print('    **PNG は反転している(255 − マスクの値)。<= 127 が「層が見える」側である**')
    else:
        print('    **決まらない。読みを止める**')
        return

    vis_side = (lambda g: g >= 128) if direct == len(mm) else (lambda g: g <= 127)

    # IC3(母集団の外。`page0019`)
    kinds = m102.load_kinds()
    pp19 = [k for k in byp if key(k) == OUTSIDE][0]
    vis19, _ = m77.visible_rows(byp[pp19])
    visid19 = {int(r['layer_id']) for r in vis19}
    mf19 = [r for r in mm if key(r['page_path']) == OUTSIDE and not r['folder']
            and kinds.get((r['page_path'], str(r['layer_id']))) == MASKFILL]
    p19 = [r for r in mf19 if r['layer_id'] in visid19]
    form = [r for r in p19 if r['mask_px_ge128'] > 0]
    ok = len(p19) == IC3_P19_MASKFILL and len(form) == IC3_P19_WITH_FORM
    print(f'IC3(**母集団の外。門として当てる**)`page0019` の**可視の葉**の `mask_fill` '
          f'期待 {IC3_P19_MASKFILL} 実測 {len(p19)} / 形がある 期待 {IC3_P19_WITH_FORM} '
          f'実測 {len(form)}  {"PASS" if ok else "FAIL"}')
    inv = [r for r in mf19 if r['layer_id'] not in visid19]
    print(f'    **可視を外すと {len(mf19)} 枚になり門が落ちる** —— '
          f'差の {len(inv)} 枚は非表示の層(L{"・L".join(str(r["layer_id"]) for r in inv)})。'
          '**先方が使った 65 はこちらが渡した「可視の葉」の数である**')

    # IC5 こちらの以前の数
    ours = 0
    for pg in POP:
        rows = [r for r in byp[[k for k in byp if key(k) == pg][0]]]
        vis, _ = m77.visible_rows(rows)
        kids = {int(r['parent_id']) for r in rows if r['parent_id']}
        ours += sum(1 for r in vis
                    if r['has_mask'] in ('True', 'true', '1')
                    and int(r['layer_id']) not in kids)
    print(f'IC5 4頁の**可視の葉**で `has_mask=True` 期待 {IC5_OURS_VISIBLE_LEAF_HASMASK}(追補182 §E)'
          f' 実測 {ours}  {"PASS" if ours == IC5_OURS_VISIBLE_LEAF_HASMASK else "FAIL"}')
    print()

    # ------------------------------------------------- 材料(頁ごと。条例2)
    print('=== A. 原案の材料(**頁ごと。プールしない**。条例2)===')
    print(f'{"頁":<14}{"納品の":>7}{"葉":>5}{"可視":>5}{"形あり":>7}'
          f'{"全面通す":>9}{"全面隠す":>9}{"成分":>7}{"床超":>7}')
    rowsout, tot = [], collections.Counter()
    for pg in list(POP) + [OUTSIDE]:
        pp = [k for k in byp if key(k) == pg][0]
        rows = byp[pp]
        idx = {int(r['layer_id']): r for r in rows}
        vis, _ = m77.visible_rows(rows)
        visid = {int(r['layer_id']) for r in vis}
        kids = {int(r['parent_id']) for r in rows if r['parent_id']}
        cv = cmf[[p for p in cmf if p.endswith(pg)][0]]['canvas']
        area = cv[0] * cv[1]
        floor = area * FLOOR
        mine = [r for r in mm if key(r['page_path']) == pg]
        drop = collections.Counter()
        n_leaf = n_vis = n_form = n_all = n_none = n_comp = n_big = 0
        for r in mine:
            lid = int(r['layer_id'])
            if r['folder'] or lid in kids:
                drop['フォルダー'] += 1
                continue
            n_leaf += 1
            if lid not in visid:
                drop['非表示'] += 1
                continue
            n_vis += 1
            ge = r['mask_px_ge128']      # **見える側の画素数。PNG の生の値ではない**
            if ge == 0:
                n_none += 1
                drop['マスクが全面を隠す(形が無い)'] += 1
                continue
            if ge >= area:
                n_all += 1
                drop['マスクが全面を通す(切っていない)'] += 1
                continue
            n_form += 1
            g = cv2.imread(str(MD / r['mask_file']), cv2.IMREAD_UNCHANGED)
            if g.ndim == 3:
                g = g[..., 0]
            nlab, _, st, _ = cv2.connectedComponentsWithStats(
                vis_side(g).astype(np.uint8), connectivity=8)
            a = st[1:, cv2.CC_STAT_AREA]
            n_comp += len(a)
            n_big += int((a >= floor).sum())
            rowsout.append({'page': pg, 'layer_id': lid,
                            'name': idx[lid]['name'] if lid in idx else '',
                            'layer_type': r['layer_type'], 'clip': r['clip'],
                            'init_color': r['init_color'],
                            'mask_px_ge128': ge, 'mask_ratio': ge / area,
                            'components': len(a),
                            'components_over_floor': int((a >= floor).sum())})
        mark = '' if pg in POP else '  (**母集団の外**)'
        print(f'{pg:<14}{len(mine):>7}{n_leaf:>5}{n_vis:>5}{n_form:>7}'
              f'{n_all:>9}{n_none:>9}{n_comp:>7}{n_big:>7}{mark}')
        if pg in POP:
            for k in ('納品の', '葉', '可視', '形あり', '成分', '床超'):
                pass
            tot['mask'] += len(mine); tot['leaf'] += n_leaf; tot['vis'] += n_vis
            tot['form'] += n_form; tot['all'] += n_all; tot['none'] += n_none
            tot['comp'] += n_comp; tot['big'] += n_big
        print(f'{"":<14}  落とした数(理由別。規律9): '
              + ' / '.join(f'{k} {v}' for k, v in sorted(drop.items())) or '  なし')
    print(f'{"4頁の合計":<14}{tot["mask"]:>7}{tot["leaf"]:>5}{tot["vis"]:>5}'
          f'{tot["form"]:>7}{tot["all"]:>9}{tot["none"]:>9}'
          f'{tot["comp"]:>7}{tot["big"]:>7}')
    print()
    print('**「形あり」= 可視の葉で、マスクが全面を隠しも通しもしない層**'
          ' = **原案が面を提案できる層**')
    print('**「床超」= その層のマスクの8連結成分のうち、画布の 1e-5 以上のもの**'
          ' = **提案される面の数**')

    with open(OUT / 'mask_material.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsout[0]))
        w.writeheader(); w.writerows(rowsout)
    print(f'\n表: {OUT / "mask_material.csv"}')
    print('**この走行は点数を出していない。** 原案の点数・帰無値・棒は 追補183 の K8 で、'
          'NC2 の実装と同じ走行に入れる')


if __name__ == '__main__':
    main()
