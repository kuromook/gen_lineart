#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m88: **完全に隠れた面 131 を、作者に聞く前に計算で分ける**(追補126)。

**追補125 §F で立てた Open**: `standard_visible` の面 **131/692 = 18.9%** が、
作者の遮蔽物の集合で**完全に隠れる**(`P0446` は 70/140 = 50%)。
**採点の標準に、平らにした画像には現れない面が2割近く入っている。**

**原因の候補は2つあり、追補125 ではどちらかを決めなかった。
そのうち片方は計算で判定できる**(人に判定を頼む前に計算可能な軸を測る):

| 類 | 覆っているもの | 読み |
|---|---|---|
| **C1** | **別のベース層、または その下地に付いた塗り** | **重なり解決が z と食い違っている** —— `m16` は「面積の小さい成分が勝つ」で解き、**z を見ていない**(追補7 R3)。**上の部位が主張している画素を下の部位に渡している** |
| **C2** | **ベースでない層だけ**(線画・効果) | **本当に上に何かが乗っている**、または**細い面が線画の太さで消えた** |
| **C3** | どちらか単独では覆いきらない | 混在 |

**C1 は作者に聞かなくても判定できる** —— **作者の z 順が「上の部位が見える」と既に
言っている**から。**聞くべき問いは「隠れている部分も面として数えるか」であり、
それは C2 と C3 の切り出しで聞く。**

**この器具は標準を作り直さない。棒も主値も触らない**(追補122 §I / 追補125 §D)。
**出すのは 131 の内訳と、面の大きさの分布、そして作者に見せる切り出しだけ。**

**既知セル(落ちたら内訳を読まない)**:
  KC-1 **完全に隠れた面が 131**(`m87` の作者の腕と一致。腕の設定を写し間違えていない証拠)
  KC-2 **C1 + C2 + C3 = 131**(取りこぼしが無い)
  KC-3 **面の総数が 692 / ベース 152 / 成分 503**(標準を動かしていない証拠)

**負の対照**: **遮蔽物を空にすると完全に隠れた面は 0**(恒等。`m87` の IC-3 と同じ経路)。

  <venv>/python tools/colour/m88_hidden_area_classes.py
"""
import collections
import csv
import json
import pathlib
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                    # noqa: E402
import m16_score_corrected_standard as m16                          # noqa: E402
import m77_standard_visible as m77                                  # noqa: E402
import m87_split_standard_by_occlusion as m87                       # noqa: E402

OUT = pathlib.Path('results/m88_hidden_area_classes')
CROP = OUT / 'crops'
ALIAS = pathlib.Path('results/alias/page_alias.csv')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
EXPECT = {'hidden': 131, 'areas': 692, 'bases': 152, 'components': 503}
MARGIN = 56               # 切り出しの余白(原寸の画素)
CAP = 1400                # 1枚の板の上限(原寸。超えたら中央で切る。注記する)
PER_CLASS_PER_PAGE = 2    # 切り出しを作る数(頁ごと・類ごと)


def aliases():
    if not ALIAS.exists():
        return {}
    return {r['page_path']: r['alias'] for r in
            csv.DictReader(open(ALIAS, encoding='utf-8'))}


def label_img(w, lines, font, pad=10):
    h = pad * 2 + int(sum(font.getbbox(t)[3] + 6 for t in lines))
    im = Image.new('RGB', (w, h), (18, 18, 20))
    d = ImageDraw.Draw(im)
    y = pad
    for t in lines:
        d.text((pad, y), t, font=font, fill=(236, 236, 240))
        y += font.getbbox(t)[3] + 6
    return np.array(im)[:, :, ::-1]


def panel(base_rgb, outline, cap=CAP):
    """原寸 1:1 の板。**拡大・縮小しない。** 大きすぎるときは中央で切る。"""
    im = base_rgb.copy()
    cnt, _ = cv2.findContours(outline.astype(np.uint8), cv2.RETR_LIST,
                              cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (60, 60, 255), 1)
    h, w = im.shape[:2]
    if h > cap or w > cap:
        y0, x0 = max(0, (h - cap) // 2), max(0, (w - cap) // 2)
        im = im[y0:y0 + min(h, cap), x0:x0 + min(w, cap)]
    return im


def strip(panels, font, head, sub, gap=10):
    hh = max(p.shape[0] for p in panels)
    ww = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    body = np.full((hh, ww, 3), 18, np.uint8)
    x = 0
    for p in panels:
        body[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    top = label_img(ww, head, font)
    bot = label_img(ww, sub, font)
    return np.vstack([top, body, bot])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CROP.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 17)
    al = aliases()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    _, comp_ok, _, _, _ = m87.VARIANTS[1]            # **作者の集合**
    assert m87.VARIANTS[1][0].startswith('O_wide')

    tot = collections.Counter()
    rows_out, t0 = [], time.time()
    for p in pages:
        rows_all = byp[p['source']]
        rows_vis, _ = m77.visible_rows(rows_all)
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (lab_a, garea), st = m16.measure_standard(
            p, rows_vis, grid, floor_px, out_comps=comps)
        comps = [c for c in comps if c['area'] >= floor_px]
        areas = m87.areas_from(lab_a, comps, floor_px)
        del lab_a
        base_ids = {str(c['layer_id']) for c in comps}
        tot['areas'] += len(areas)
        tot['bases'] += st['bases']
        tot['components'] += st['components']

        files = {l['layer_id']: l['file'] for l in p['layers'] if l.get('file')}
        roots, kids = m16.build_tree(rows_vis)
        z = m16.z_order(roots, kids)
        cbase = m87.clip_base_map(rows_vis)
        parent = {str(r['layer_id']): (str(r['parent_id']) if r['parent_id']
                                       else None) for r in rows_vis}
        drop = collections.Counter()
        rowby = {str(r['layer_id']): r for r in rows_vis}
        cand = []
        for r in rows_vis:
            if r['layer_type'] == m16.PAPER_TYPE or r['folder'] == 'True':
                continue
            if int(r['opacity'] or 0) <= 0 or not comp_ok(r['composite']):
                continue
            zz = z.get(int(r['layer_id']))
            if zz is None:
                continue
            cand.append((zz, r))
        cand.sort(key=lambda t: t[0], reverse=True)

        # **ベース系**(ベース層・ベースに付いた塗り)と**非ベース系**を別に数える
        cb = np.zeros(grid, np.uint16)
        co = np.zeros(grid, np.uint16)
        clips_by_base = collections.defaultdict(list)
        cmask, cache = {}, {}
        prev = cv2.imread(str(m12.D / p['preview']['file']), cv2.IMREAD_COLOR)
        made = collections.Counter()
        i = 0
        for c in sorted(areas, key=lambda a: a['z'], reverse=True):
            while i < len(cand) and cand[i][0] > c['z']:
                r = cand[i][1]
                i += 1
                m = m87.occluder_mask(r, kids, files, grid, cache, cbase,
                                      True, drop)
                if m is None:
                    continue
                bb = cbase.get(r['layer_id'])
                is_base = str(r['layer_id']) in base_ids
                clip_of_base = bb is not None and str(bb['layer_id']) in base_ids
                if is_base or clip_of_base:
                    cb += m
                else:
                    co += m
                if bb is not None:
                    clips_by_base[str(bb['layer_id'])].append(
                        str(r['layer_id']))
                    cmask[str(r['layer_id'])] = m
            y, x, h, w = c['box']
            sb = cb[y:y + h, x:x + w].astype(np.int32)
            so = co[y:y + h, x:x + w].astype(np.int32)
            if c['base'] is not None:
                keys = [str(c['base'])] + m87.ancestors(c['base'], parent)
                for b in keys:
                    for lid in clips_by_base.get(b, ()):
                        mm = cmask.get(lid)
                        if mm is not None:
                            sb -= mm[y:y + h, x:x + w]   # 除外はベース系に居る
            ob, oo = sb > 0, so > 0
            rem = c['mask'] & ~(ob | oo)
            if rem.any():
                n, _, stt, _ = cv2.connectedComponentsWithStats(
                    rem.astype(np.uint8), connectivity=8)
                if any(int(stt[j, cv2.CC_STAT_AREA]) >= floor_px
                       for j in range(1, n)):
                    continue                              # 隠れていない
            # **ここから完全に隠れた面**
            only_b = not (c['mask'] & ~ob).any()
            only_o = not (c['mask'] & ~oo).any()
            cls = 'C1' if only_b else ('C2' if only_o else 'C3')
            tot['hidden'] += 1
            tot[cls] += 1
            a0 = int(c['mask'].sum())
            rows_out.append(dict(
                page=al.get(p['source'], p['page']), 類=cls,
                面の画素=a0, 面の割合=round(a0 / (grid[0] * grid[1]), 8),
                ベース系だけで覆う=only_b, 非ベース系だけで覆う=only_o,
                ベース系の被覆=round(float((c['mask'] & ob).sum()) / a0, 4),
                非ベース系の被覆=round(float((c['mask'] & oo).sum()) / a0, 4),
                面の高さ=h, 面の幅=w))
            if made[cls] >= PER_CLASS_PER_PAGE:
                continue
            made[cls] += 1
            # ---- 原寸 1:1 の切り出し(拡大・縮小しない)----
            y0 = max(0, y - MARGIN); x0 = max(0, x - MARGIN)
            y1 = min(grid[0], y + h + MARGIN); x1 = min(grid[1], x + w + MARGIN)
            H, W = y1 - y0, x1 - x0
            mk = np.zeros((H, W), bool)
            mk[y - y0:y - y0 + h, x - x0:x - x0 + w] = c['mask']
            # 1. 合成(プレビューを最近傍で s 倍。**原寸の合成は存在しない**)
            if prev is None:
                comp = np.full((H, W, 3), 40, np.uint8)
            else:
                pc = prev[y0 // s:(y1 + s - 1) // s, x0 // s:(x1 + s - 1) // s]
                pc = cv2.resize(pc, (pc.shape[1] * s, pc.shape[0] * s),
                                interpolation=cv2.INTER_NEAREST)
                comp = pc[y0 - (y0 // s) * s:y0 - (y0 // s) * s + H,
                          x0 - (x0 // s) * s:x0 - (x0 // s) * s + W]
                if comp.shape[:2] != (H, W):
                    comp = cv2.copyMakeBorder(
                        comp, 0, max(0, H - comp.shape[0]), 0,
                        max(0, W - comp.shape[1]), cv2.BORDER_CONSTANT,
                        value=(40, 40, 40))[:H, :W]
            # 2. この面(白)と、同じベースの他の面(暗い灰)
            ba = np.zeros((H, W, 3), np.uint8)
            br = rowby.get(str(c['base'])) if c['base'] is not None else None
            bal = None
            if br is not None:
                bal = (m87.subtree_alpha(br['layer_id'], kids, files, grid,
                                         cache)
                       if br['folder'] == 'True'
                       else m87.alpha_of(br['layer_id'], files, grid, cache))
            if bal is not None:
                ba[bal[y0:y1, x0:x1]] = (74, 74, 78)
            ba[mk] = (242, 242, 246)
            # 3. ベース系の遮蔽物(青)/ 4. 非ベース系(橙)
            fb = np.zeros((H, W, 3), np.uint8)
            fo = np.zeros((H, W, 3), np.uint8)
            sb2 = cb[y0:y1, x0:x1] > 0
            so2 = co[y0:y1, x0:x1] > 0
            fb[sb2] = (40, 150, 240)      # ベース系 = 橙(BGR)
            fo[so2] = (210, 130, 40)      # 非ベース系 = 青(BGR)
            head = [f'{al.get(p["source"], p["page"])}  面 {a0:,}px  '
                    f'{w}x{h}  類 {cls}',
                    '**問い: 赤い輪郭の面は、絵の中で1画素も見えていない。'
                    'これを「面」として数えるべきか**']
            sub = ['板は左から: (1) 合成(プレビューを最近傍で '
                   f'{s}倍。原寸の合成は存在しない) '
                   '(2) 赤の面=白 / 同じベースの他の面=暗い灰',
                   '(3) ベース系の遮蔽物(別の部位とその塗り)=橙 '
                   '(4) 非ベース系の遮蔽物(線画・効果)=青',
                   f'ベース系の被覆 {float((c["mask"] & ob).sum())/a0:.4f} / '
                   f'非ベース系の被覆 {float((c["mask"] & oo).sum())/a0:.4f}'
                   '  **原寸 1:1。拡大・縮小していない**']
            img = strip([panel(comp, mk), panel(ba, mk),
                         panel(fb, mk), panel(fo, mk)], font, head, sub)
            fn = CROP / (f'{al.get(p["source"], p["page"])}_{cls}'
                         f'_{made[cls]}_{a0}px.png')
            cv2.imwrite(str(fn), img)
        for k, v in drop.items():
            tot[f'落: {k}'] += v
        del cb, co, cmask, cache, areas, comps, prev
        print(f'  {al.get(p["source"], p["page"]):8s} 隠れ '
              f'{sum(1 for r in rows_out if r["page"] == al.get(p["source"], p["page"])):>4}'
              f'  {time.time()-t0:6.1f}s', flush=True)

    print('\n== 既知セル ==')
    ok = True
    for k, v in EXPECT.items():
        good = tot[k] == v
        ok &= good
        print(f'  {k:11s} 期待 {v:>5} / 実測 {tot[k]:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    s3 = tot['C1'] + tot['C2'] + tot['C3']
    print(f'  KC-2 類の和   期待 {tot["hidden"]:>5} / 実測 {s3:>5}  '
          f'{"PASS" if s3 == tot["hidden"] else "**FAIL**"}')
    ok &= s3 == tot['hidden']
    print(f'\n  **既知セル {"すべて PASS" if ok else "**FAIL**。内訳を読まない"}**')
    if not ok:
        return 1

    print('\n== 131 の内訳 ==')
    for cls, why in (('C1', '別のベースが覆う = 重なり解決が z と食い違う'),
                     ('C2', 'ベースでない層だけが覆う = 線画・効果'),
                     ('C3', 'どちらか単独では覆いきらない')):
        n = tot[cls]
        px = sorted(r['面の画素'] for r in rows_out if r['類'] == cls)
        med = px[len(px) // 2] if px else 0
        print(f'  {cls}  {n:>4} ({n/tot["hidden"]:>6.1%})  '
              f'面の画素 中央 {med:>8,} / 最小 {(px[0] if px else 0):>8,} '
              f'/ 最大 {(px[-1] if px else 0):>9,}   {why}')

    print('\n== 頁ごと(プールしない)==')
    byq = collections.Counter((r['page'], r['類']) for r in rows_out)
    for pg in sorted({r['page'] for r in rows_out}):
        print(f'  {pg:8s} C1 {byq[(pg,"C1")]:>3} / C2 {byq[(pg,"C2")]:>3} '
              f'/ C3 {byq[(pg,"C3")]:>3}')

    print('\n== 落としたもの(理由別。規律9)==')
    for k, v in sorted(tot.items()):
        if k.startswith('落: '):
            print(f'  {k[3:]:26s} {v:>6}')

    f = OUT / 'hidden_areas.csv'
    with open(f, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f'\nwrote {f}\nwrote {CROP}/ ({len(list(CROP.glob("*.png")))} 枚)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
