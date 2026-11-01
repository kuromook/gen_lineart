"""**採用4頁(`colour_pilot2`)で、原案の材料を数える。点数は出さない。**

**事前登録は 追補193**(走行前に commit `24c6b49`)。定義は `m115` と同じで、
作者の裁定(線の層を外す 追補189 / ハイライトを外す 追補190)を重ねる。
**`filter`(補正の層)は材料に数えず、別の行に印字する** —— 作者が未裁定。保留であって破棄ではない。

構造の行は **`m_colour_page_layers_v2_20261102.csv`**(known8 に新しい4頁は無い)。
**頁は `page_path` の完全一致で引く** —— `sketch\\sketch\\page0025` は 2022 と 2022b の双子が両方在る。
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
import m77_standard_visible as m77                                   # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402

D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_pilot2')
MM = D / 'layer_masks_manifest_pilot2.json'
CM = D / 'colour_layers_manifest_pilot2.json'
V2 = Path('inbox/m_colour_page_layers_v2_20261102.csv')
OUT = Path('results/m116_mask_material_adopted4')
FLOOR = 1e-5
MASKFILL = 'mask_fill'
LINE = ('線画',)
HIGHLIGHT = ('high', 'ハイライト', 'ﾊｲﾗｲﾄ', '発光', '光', 'glow', 'shine', '艶')
STRUCT = ('layer_id', 'parent_id', 'order_in_parent', 'visibility', 'has_mask',
          'layer_type', 'clip', 'folder', 'name')

# 門の期待値(**こちらのパイプラインの外から来た数**。追補193)
IC2_LAYERS = {'page0025': 40, 'top': 50, 'CDrev2': 77, '新規キャンバス': 45}
IC3_MASKS = {'page0025': 1, 'top': 4, 'CDrev2': 4, '新規キャンバス': 12}
IC5_VIS_LEAF_HM = {'page0025': 1, 'top': 3, 'CDrev2': 2, '新規キャンバス': 11}
IC6_PASS_ALL = {('page0025', 52), ('top', 108), ('CDrev2', 82), ('CDrev2', 83),
                ('新規キャンバス', 17)}
OLD4 = {'層': 6, '床超': 12}            # 追補190 §F(`m115` + 裁定2件)


def key(pp):
    return Path(pp.replace('\\', '/')).stem


def ok(b):
    return 'PASS' if b else '**FAIL**'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.load(open(CM, encoding='utf-8'))['pages']
    mm = json.load(open(MM, encoding='utf-8'))['layers']
    kinds = m102.load_kinds()
    v2 = collections.defaultdict(list)
    for r in csv.DictReader(open(V2, encoding='utf-8-sig')):
        v2[r['page_path']].append(r)
    k8 = m16.page_rows()
    fails = 0

    # ---------------------------------------------------------------- 門
    bad = 0
    for pp, rows in k8.items():
        a = {r['layer_id']: tuple(r[c] for c in STRUCT) for r in rows}
        b = {r['layer_id']: tuple(r[c] for c in STRUCT) for r in v2.get(pp, [])}
        bad += (a != b)
    print(f'IC1 構造の列が known8 と一致する頁 {len(k8) - bad} / {len(k8)}  {ok(bad == 0)}')
    fails += bad > 0

    got = {key(p['source']): len(v2[p['source']]) for p in pages}
    print(f'IC2 層の数 期待 {IC2_LAYERS} 実測 {got}  {ok(got == IC2_LAYERS)}')
    fails += got != IC2_LAYERS

    nm = collections.Counter(key(r['page_path']) for r in mm)
    err = sum(1 for r in mm if r.get('error'))
    print(f'IC3 マスク 期待 {IC3_MASKS} 実測 {dict(nm)} / 出せなかった {err}'
          f'  {ok(dict(nm) == IC3_MASKS and err == 0)}')
    fails += not (dict(nm) == IC3_MASKS and err == 0)

    direct = inverted = 0
    for r in mm:
        g = cv2.imread(str(D / r['mask_file']), cv2.IMREAD_UNCHANGED)
        g = g[..., 0] if g.ndim == 3 else g
        if int((g >= 128).sum()) == r['mask_px_ge128']:
            direct += 1
        if int((g <= 127).sum()) == r['mask_px_ge128']:
            inverted += 1
    one = (inverted == len(mm)) != (direct == len(mm))
    print(f'IC4 向き: 生の値が一致 {direct} / 反転が一致 {inverted} / {len(mm)}  {ok(one)}')
    if not one:
        print('    **向きが決まらない。読みを止める**')
        return
    vis_side = (lambda g: g <= 127) if inverted == len(mm) else (lambda g: g >= 128)
    print('    **' + ('反転している。<= 127 が「層が見える」側' if inverted == len(mm)
                      else '生の値のまま。>= 128 が「層が見える」側') + '**(前便と同じか: '
          + ('同じ' if inverted == len(mm) else '**違う**') + ')')

    full = {(key(r['page_path']), r['layer_id']) for r in mm
            if r['mask_px_ge128'] >= r['canvas'][0] * r['canvas'][1]}
    print(f'IC6 全面を通すマスク 期待 {sorted(IC6_PASS_ALL)} 実測 {sorted(full)}'
          f'  {ok(full == IC6_PASS_ALL)}')
    fails += full != IC6_PASS_ALL

    vis_by, hm_vis = {}, {}
    for p in pages:
        rows = v2[p['source']]
        vis, _ = m77.visible_rows(rows)
        kids = {int(r['parent_id']) for r in rows if r['parent_id']}
        vis_by[p['source']] = (vis, kids)
        hm_vis[key(p['source'])] = sum(
            1 for r in vis if r['has_mask'] in ('True', 'true', '1')
            and int(r['layer_id']) not in kids and r['folder'] != 'True')
    print(f'IC5 可視の葉で has_mask 期待 {IC5_VIS_LEAF_HM} 実測 {hm_vis}'
          f'  {ok(hm_vis == IC5_VIS_LEAF_HM)}')
    fails += hm_vis != IC5_VIS_LEAF_HM
    if fails:
        print(f'\n**門が {fails} 本落ちた。§A 以下は読まない**')
    print()

    # ------------------------------------------------- 材料(頁ごと。条例2)
    print('=== A. 原案の材料(**頁ごと。プールしない**。条例2)===')
    hdr = ('納品', '葉', '可視', '全面隠す', '全面通す', '線', 'ﾊｲﾗｲﾄ', 'filter', '材料', '床超')
    print(f'{"頁":<16}' + ''.join(f'{h:>8}' for h in hdr))
    detail, tot = [], collections.Counter()
    for p in pages:
        pp, pg = p['source'], key(p['source'])
        vis, kids = vis_by[pp]
        idx = {int(r['layer_id']): r for r in v2[pp]}
        visid = {int(r['layer_id']) for r in vis}
        area = p['canvas'][0] * p['canvas'][1]
        floor = area * FLOOR
        c = collections.Counter()
        for r in [x for x in mm if x['page_path'] == pp]:
            lid = int(r['layer_id'])
            c['納品'] += 1
            row = idx[lid]
            if r['folder'] or lid in kids:
                c['_フォルダー'] += 1
                continue
            c['葉'] += 1
            if lid not in visid:
                c['_非表示'] += 1
                continue
            c['可視'] += 1
            ge = r['mask_px_ge128']                    # **見える側の画素数。PNG の生の値ではない**
            name = row['name'] or ''
            kind = kinds.get((pp, str(lid)), '(表に無い)')
            why = None
            if ge == 0:
                why = '全面隠す'
            elif ge >= area:
                why = '全面通す'
            elif any(t in name for t in LINE):
                why = '線'
            elif any(t in name.lower() or t in name for t in HIGHLIGHT):
                why = 'ﾊｲﾗｲﾄ'
            elif kind == 'filter':
                why = 'filter'
            sizes = []
            if why in (None, '線', 'ﾊｲﾗｲﾄ', 'filter'):
                g = cv2.imread(str(D / r['mask_file']), cv2.IMREAD_UNCHANGED)
                g = g[..., 0] if g.ndim == 3 else g
                _, _, st, _ = cv2.connectedComponentsWithStats(
                    vis_side(g).astype(np.uint8), connectivity=8)
                a = np.sort(st[1:, cv2.CC_STAT_AREA])[::-1]
                sizes = [x / area for x in a if x >= floor]
            if why is None:
                c['材料'] += 1
                c['床超'] += len(sizes)
            else:
                c[why] += 1
            detail.append(dict(page=pg, layer_id=lid, name=name, kind=kind,
                               layer_type=r['layer_type'], clip=r['clip'],
                               init_color=r['init_color'], mask_ratio=ge / area,
                               verdict=why or '材料', n_over_floor=len(sizes),
                               sizes=' / '.join(f'{x*100:.2f}%' for x in sizes[:6])))
        print(f'{pg[:16]:<16}' + ''.join(f'{c[h]:>8}' for h in hdr))
        print(f'{"":<16}  落とした数(理由別。規律9): フォルダー {c["_フォルダー"]} / 非表示 {c["_非表示"]}'
              f' / 全面隠す {c["全面隠す"]} / 全面通す {c["全面通す"]} / 線 {c["線"]}'
              f' / ハイライト {c["ﾊｲﾗｲﾄ"]} / **filter(保留) {c["filter"]}**')
        tot.update({h: c[h] for h in hdr})
    print(f'{"採用4頁の合計":<14}' + ''.join(f'{tot[h]:>8}' for h in hdr))
    print()
    print('=== B. 形を見る層(**作者が裁定できるように全部並べる。同定しない**)===')
    print(f'{"頁":<14}{"層":>5}  {"判定":<8}{"種別":<12}{"種":>6}{"ｸﾘｯﾌﾟ":>6}{"割合":>9}  床超の面(大きい順)  名前')
    for d in detail:
        if d['verdict'] in ('全面隠す', '全面通す'):
            continue
        print(f'{d["page"][:14]:<14}{d["layer_id"]:>5}  {d["verdict"]:<8}{d["kind"]:<12}'
              f'{d["layer_type"]:>6}{d["clip"]:>6}{d["mask_ratio"]*100:>8.2f}%  '
              f'{d["sizes"] or "—"}  {d["name"]}')
    print()
    print('=== C. 8頁(`pop_8pages_colour_20261101`)===')
    print(f'  既存4頁(追補190 §F)  材料 {OLD4["層"]} 層 / 床超 {OLD4["床超"]} 面')
    print(f'  採用4頁(本走行)      材料 {tot["材料"]} 層 / 床超 {tot["床超"]} 面'
          f'  (**filter 保留 {tot["filter"]} 層は入れていない**)')
    print(f'  8頁                  材料 {OLD4["層"] + tot["材料"]} 層 / 床超 {OLD4["床超"] + tot["床超"]} 面')
    print('  **recall の上界は出さない** —— 採用4頁の標準の面(b)はまだ作っていない(追補193)')

    with open(OUT / 'mask_material.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    print(f'\n表: {OUT / "mask_material.csv"}')
    print('**この走行は点数を出していない。**')


if __name__ == '__main__':
    main()
