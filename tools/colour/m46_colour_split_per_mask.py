"""M46: **1つのマスクの中に、色で分かれる領域がいくつあるか**(作者の「複数の面」の数)。

作者(2026-10-26): > **複数の面　マスクは全部の面をカバーするように細かく切らない**
> **それをやるとレイヤーが途方もない数になってしまう　なので いくつかの面は一つのマスクであつかう**

**作者の追記(2026-10-26。逐語)**:

> **もうひとついうと、１マスクの中に面をひとまとめにといったが、
> 髪と肌はそれぞれ１マスクであることが多い**

→ **束ね方は一様ではない。材質の種類で違う。**
**「1成分が作者の面いくつに当たるか」の係数は1つの数ではなく、種類ごとに違う。**
→ **層別してから測る**(プールを跨いで平均しない。moat 8)。

**層の作り方(2段)**:

1. **候補は作者の語で立てる** —— 入口に `HAIR` / `SKIN` / `EYE` / `CLOTH` / `UNDERWARE` が
   実際に在る(`m45`)。**作者が「髪と肌」と名指しした**ので、仮説を立てる理由としては正当
2. **判定は名前でしない** —— 名前は7つめの落ちた錨。
   **結果は「名前が `HAIR`/`SKIN` の層」として印字する。「髪のマスク」とは書かない**

**層ごとの向き(走らせる前に書く)**:

| 層 | 作者の言葉からの予測 |
|---|---|
| **名前が `HAIR` / `SKIN`**(`hair`/`skin`/`髪`/`肌`) | **1成分あたり 1 に近い**(それぞれ1マスク) |
| **その他** | **>= 2**(いくつかの面を1マスクで扱う) |

**2群が分かれなければ**、**(a) 名前が材質を指していない / (b) 色では面の境目が残っていない /
(c) 作者の「多い」がこの8頁では当たらない** の3つが残り、**どれかは決められない** ——
**それも印字する。**

**全体の中央値は出さない**(出すなら層ごとの行の下に
「層を跨いだ数であって母集団の数ではない」と添える)。

**向き(走らせる前に書く)**:

> **作者の「複数」が正しければ、1成分あたりの「色で分かれる領域」の数の中央値は >= 2。**
> **1.0 に近ければ、色では面の境目が残っていない**(塗りが平坦でない、あるいは境目が
> 同じ色で塗られている)ことになり、**機械的な細分の道は閉じる。**

**測れる範囲(行ごとに印字する)**:

| ベースの種類 | 色をどこから取るか | 偏り |
|---|---|---|
| **フォルダーのベース** | **子孫の合成(原寸)** —— 子孫は届いている | 偏りなし |
| **非フォルダーのベース** | **プレビュー(縮小)** —— 色を持つクリップ層が届いていない | **縮小は近傍の平均なので、細い面や境目ぎわの面が消える = 面の数を過小に数える側に偏る**(基盤) |

**原寸の行とプレビューの行を1つの中央値にまとめない。**

**色で分かれる領域の数え方**: 成分の中の画素を `>>4` で量子化し、
**その成分の画素の 5% 以上を占める色の数**。
(5% 未満は反エイリアスの縁と細部。**閾値は結果を見てから動かさない。**)

**既知セル**: 8頁の標準が **ベース183 / 成分823 / 面801**。
"""
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402

DELIV = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4_known8')
OUT = Path('results/m46_colour_split'); OUT.mkdir(parents=True, exist_ok=True)
SHARE = 0.05
Q = 4
MAT = __import__('re').compile(r'(hair|skin|髪|肌)', __import__('re').I)
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}


def quant(img):
    return ((img[..., 0] >> Q).astype(np.int32) * 256
            + (img[..., 1] >> Q).astype(np.int32) * 16 + (img[..., 2] >> Q))


def main():
    rows = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8') if l.strip()]
    m12.D = DELIV
    tot = collections.Counter()
    rec = []
    for p in pages:
        if not p['layers']:
            continue
        prows = byp[p['source']]
        byid = {str(r['layer_id']): r for r in prows}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        files = {l['layer_id']: l['file'] for l in p['layers']}
        comps = []
        (_, _), (_, garea_a), st = m16.measure_standard(
            p, prows, grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        prev = cv2.imread(str(DELIV / p['preview']['file']), cv2.IMREAD_COLOR)
        pq = quant(prev)
        for c in comps:
            y, x, h, w = c['box']
            r = byid.get(str(c['layer_id']), {})
            folder = r.get('folder') == 'True'
            ids = c.get('ids') or []
            n = None
            how = ''
            if folder and ids:
                sub = np.full((h, w, 3), 255, np.uint8)
                for i in ids:
                    f = files.get(i)
                    if not f:
                        continue
                    im = cv2.imread(str(DELIV / f), cv2.IMREAD_UNCHANGED)
                    if im is None:
                        continue
                    t = im[y:y + h, x:x + w]
                    if t.size == 0:
                        continue
                    a = t[..., 3:4].astype(np.float32) / 255.0
                    sub = (t[..., :3].astype(np.float32) * a
                           + sub.astype(np.float32) * (1 - a)).astype(np.uint8)
                    del im, t, a
                q = quant(sub)[c['mask']]
                how = '原寸(子孫の合成)'
            else:
                sm = c['mask'][::s, ::s] if s > 1 else c['mask']
                ys, xs = y // s, x // s
                hh = min(sm.shape[0], pq.shape[0] - ys)
                ww = min(sm.shape[1], pq.shape[1] - xs)
                if hh <= 0 or ww <= 0:
                    continue
                m2 = np.zeros(pq.shape, bool)
                m2[ys:ys + hh, xs:xs + ww] = sm[:hh, :ww]
                q = pq[m2]
                how = f'プレビュー 1/{s}(**面の数を過小に数える側に偏る**)'
            if q.size < 20:
                continue
            u, cnt = np.unique(q, return_counts=True)
            n = int((cnt / cnt.sum() >= SHARE).sum())
            nm = r.get('name', '?') or '?'
            rec.append({'page': p['page'], 'layer_id': c['layer_id'],
                        'name': nm, 'folder': folder,
                        'area_px': c['area'], 'colours': n, 'how': how,
                        'stratum': ('名前が HAIR/SKIN' if MAT.search(nm)
                                    else 'その他の名前')})
        print(f'  {p["page"]:44s} 成分{st["components"]:>4} 面{len(garea_a):>4}',
              flush=True)
        del comps, prev, pq

    ok = all(tot[k] == EXPECT[k] for k in EXPECT)
    print(f'\n== 既知セル ==')
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if not ok:
        print('**既知セルが合わない。表を書かない。**')
        return 1

    def show(sel, lab):
        v = sorted(x['colours'] for x in sel)
        print(f'  {lab:46s} {len(sel):>4} 成分  '
              f'中央 **{statistics.median(v)}** / 平均 {sum(v) / len(v):.2f} / '
              f'最大 {v[-1]} / 1 のもの {sum(1 for x in v if x <= 1) / len(v):.0%} / '
              f'2以上 {sum(1 for x in v if x >= 2) / len(v):.0%}')

    for how_lab, how_sel in (('原寸(フォルダーのベース)', '原寸'),
                             ('プレビュー(非フォルダー。**過小に偏る**)', 'プレビュー')):
        grp = [x for x in rec if x['how'].startswith(how_sel)]
        print(f'\n== {how_lab}: {len(grp)} 成分 ==')
        if not grp:
            print('  0 成分')
            continue
        for st_lab in ('名前が HAIR/SKIN', 'その他の名前'):
            sel = [x for x in grp if x['stratum'] == st_lab]
            if sel:
                show(sel, st_lab)
            else:
                print(f'  {st_lab:46s}    0 成分')
        print(f'  **向き(登録済み)**: 名前が HAIR/SKIN は **1 に近い**(それぞれ1マスク)/ '
              f'その他は **>= 2**(いくつかの面を1マスクで扱う)')
        print(f'  **分かれなければ**: (a) 名前が材質を指していない / '
              f'(b) 色では境目が残っていない / (c) 作者の「多い」がこの8頁では当たらない '
              f'—— **どれかは決められない**')
    print('\n  **原寸の行とプレビューの行は1つの中央値にまとめない**(基盤の条件)')
    with open(OUT / 'colour_split.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec[0].keys()))
        w.writeheader()
        w.writerows(rec)
    print(f'\nwrote {OUT}/colour_split.csv  '
          f'(**この8頁での数。既知セルは `illust*` が半分の層で corpus の数ではない**)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
