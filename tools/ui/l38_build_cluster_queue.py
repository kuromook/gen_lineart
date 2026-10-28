# -*- coding: utf-8 -*-
"""L38: **面のまとまり**に語を当てる UI の待ち行列と画像を作る(2026-10-28)。

作者の指示(2026-10-28):

> **これはUIの出番だ 複数の面を表示し、ユーザがまとまりを選択して語をわりあてる**

そして作者の言明(2026-10-26):

> 割れているなら別の面 基本単位として これは線の単位を1本と数えるのと同じ意味での単位
> **複数の面のまとまりとなってはじめて意味を持たせられるものも多い**

L18 が閉じたのは **面1つに語1つを割り当て、連結した墨で運ぶ形**である
(2026-10-26 21:08、A/B/C すべて FAIL)。**閉じたのは運び手で、面と語の対応ではない。**
**まとまりを単位にした形は一度も測っていない** —— 単位が無かったからで、
**その単位は作者が選ぶことでしか作れない**(規律6)。ここはその単位を作る器具である。

## 画面の単位(作者が選んだ。2026-10-28)

- **コマ1つ**。1コマの面を全部並べ、その中で束ねる。**まとまりはコマを跨がない**
  (この制限は記録に残す。跨ぐ必要が出たら、それは次の版の設計である)
- **束ねる面は「既に語を付けた面」と同じ標準**(`arms('gated')`)。
  だから**面ごとの 628件の答がそのまま画面に出る**。L35 の納品ベタ層の面ではない

## 出力

```
results/l38_cluster_ui/panels/<pan>_sel.png     選択用(長辺 1200)。面ごとに淡い色と番号
results/l38_cluster_ui/panels/<pan>_nat.png     **原寸**(縮めない。規律6 / 原寸の切り出し)
results/l38_cluster_ui/panels/<pan>_map.png     16bit・**原寸**。画素の値 = 面の番号(0 = 面でない)
results/l38_cluster_ui/queue.json               コマごとの面と、既存の面ごとの答
```

**`_map.png` が当たり判定の正本である。** 画面の座標から面を引くのはサーバ側で、
**原寸の地図で引く**(画面の座標を `1/scale` で戻す)。
最初は縮めた地図で引いていたが、**1画素より細い面が消えた**(65頁で 1,563面中 2面)。
**地図は縮めない。** 残る危険は丸めだけで、それを既知セルが往復で試す。

## 既知セル(この器具が自分で持つもの。サーバ側でも回す)

1. **当たり判定**: 面ごとに**縮小前に**決めた代表点(境界から最も遠い画素)を
   縮小後の座標へ移し、`_map.png` がその面を返すこと。**返さない面を理由別に数える**
2. **結び付き**: 面の名前(`{work}_{pid}_c{ci}_r{gi}_{k:03d}.png`)が
   L19 の 551面を**全部含むこと**。`k` は **L19 と同じ条件・同じ順**で振る
   (`adj > 0` の面だけ数える)。**含まなければ名前が合っていない = 答が結べない**
3. **面の数**: 画面に描いた面の数と、待ち行列に載せた面の数が一致すること

**`adj = 0` の面(縁に門を通した層の墨が1枚も触っていない面)も選べる。**
ただし **`k` は振らない** —— 振ると L19 の名前が全部ずれて 628件の答が結べなくなる。
**答を持たない面として、`region` 番号で載せる。**
"""
import argparse, json, os, sys, csv, collections
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, os.path.join(HERE, '..', 'evaluation'))
sys.path.insert(0, HERE)
from l15_known_cells import boundary
from l17_known_cells import (arms, D1, adj_sets, assign_top, jobs,
                             panels_and_regions, zmap)

ROOT = '/home/sh1/deepl/lineart-fill-region'
OUT = f'{ROOT}/results/l38_cluster_ui'
L19Q = f'{ROOT}/results/l19_ui/queue.json'
L19A = f'{ROOT}/labels/fill_words.jsonl'
MAXSIDE = 1200
# 面の色(BGR、淡い)。番号で見分けるので色は巡回でよい
PAL = [(245, 215, 215), (215, 245, 215), (215, 225, 250), (245, 235, 205),
       (240, 215, 245), (205, 245, 245), (225, 225, 245), (235, 245, 215),
       (250, 225, 235), (220, 240, 230)]


def one_page(job):
    """1頁のコマを作る。**`k` は L19 と同じ条件・同じ順**で振る。"""
    out = []
    k = 0
    pan = 0          # 頁の中の通し番号(`ci` は枠ファイルごとなので頁内で再出する。KC46)
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        pan += 1
        lab, keys = assign_top(sub, P.shape)
        inkP = (lab > 0) & P
        faces = []
        for gi in gids:
            R = glab == gi
            if not R.any():
                continue
            b = boundary(R)
            vals, cnt = adj_sets(R, lab, P, ds=(D1,), bnd=b)[D1]
            adj = int(cnt.sum())
            nm = ''
            if adj:
                # **L19 と同じ条件**。ここで `k` を振る順が名前を決める
                k += 1
                nm = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
            # 代表点 = 面の中で境界から最も遠い画素(縮小に耐える)
            dt = cv2.distanceTransform(R.astype(np.uint8), cv2.DIST_L2, 3)
            iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
            faces.append(dict(region=int(gi), crop=nm, adj=adj,
                              area=int(R.sum()), py=int(iy), px=int(ix),
                              inr=float(dt[iy, ix])))
        if not faces:
            continue
        # ---- 画像3枚
        nat = np.full(P.shape + (3,), 255, np.uint8)
        nat[~P] = (238, 238, 238)                 # コマの外
        nat[inkP] = (200, 200, 200)               # 他の層の墨は淡い灰で退く
        fmap = np.zeros(P.shape, np.uint16)
        for i, f in enumerate(faces, 1):
            R = glab == f['region']
            nat[R] = PAL[(i - 1) % len(PAL)]
            fmap[R] = i
            # 縁を濃くする(面の切れ目が見えないと束ねられない)
            nat[boundary(R)] = (90, 90, 90)
        ys, xs = np.nonzero(P)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        nat, fmap = nat[y0:y1, x0:x1], fmap[y0:y1, x0:x1]
        h, w = fmap.shape
        sc = min(1.0, MAXSIDE / max(h, w))
        sel = (cv2.resize(nat, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
               if sc < 1.0 else nat.copy())
        # **地図は縮めない。** 縮めた地図で当たり判定をすると、
        # **1画素より細い面が消えて押せなくなる**(2026-10-28、65頁で 1,563面中 2面が消えた)。
        # 画面の座標を 1/scale で原寸へ戻して引く —— **捨てる情報が無い**。
        smap = fmap
        # **番号は画像に焼かない。** 小さい面が隣り合うと重なって読めなくなる
        # (2026-10-28、7コマの煙試験で 13/15/16/17 が重なった)。
        # 代表点の座標だけ渡し、**番号の札は画面側(DOM)で置く** ——
        # 重なりを避けられ、選択の状態も札に出せる。
        for i, f in enumerate(faces, 1):
            cy, cx = int((f['py'] - y0) * sc), int((f['px'] - x0) * sc)
            f['sy'], f['sx'] = cy, cx
            f['py'], f['px'] = int(f['py'] - y0), int(f['px'] - x0)
        tag = f'{work}_{pid}_p{pan:02d}'
        cv2.imwrite(f'{OUT}/panels/{tag}_sel.png', sel)
        cv2.imwrite(f'{OUT}/panels/{tag}_nat.png', nat)
        cv2.imwrite(f'{OUT}/panels/{tag}_map.png', smap)
        out.append(dict(work=work, pid=pid, pan=pan, comp=int(ci), tag=tag,
                        faces=faces, scale=round(float(sc), 5),
                        nat_h=int(h), nat_w=int(w),
                        sel_h=int(smap.shape[0]), sel_w=int(smap.shape[1])))
    return out


def answers_by_crop():
    """L19 の面ごとの答を `crop` 名で引ける形にする。

    **`order` では引かない**(T144)。L19 のサーバが持っている中身照合を
    **そのまま輸入して使う** —— 2本目を書くと、2本目が間違う。
    """
    sys.path.insert(0, HERE)
    from l19_fill_label_server import match_row
    q = json.load(open(L19Q, encoding='utf-8'))['queue']
    out, proj, why = {}, {}, collections.Counter()
    n = 0
    for ln in open(L19A, encoding='utf-8'):
        if not ln.strip():
            continue
        r = json.loads(ln)
        n += 1
        o, w = match_row(r, q, proj)
        if o is None:
            why[w] += 1
            continue
        out[q[o]['crop']] = r
    print(f'L19 の答 {n} 行 -> **{len(out)} 面に結んだ**(行列 {len(q)} 面)')
    for w, c in why.items():
        print(f'   結ばなかった {c}  {w}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='頁を切る(煙試験)')
    a = ap.parse_args()
    os.makedirs(f'{OUT}/panels', exist_ok=True)
    print('L38: **面のまとまり**の待ち行列。標準は `arms(\'gated\')` '
          '(= 面ごとの 628件と同じ標準)', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'頁 {len(js)}', flush=True)
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, js), 1):
            rows += r
            if i % 5 == 0 or i == len(js):
                print(f'  {i}/{len(js)} 頁  コマ {len(rows)}', flush=True)
    # **並びを決定的にする**(Pool の戻り順は不定。作り直すたびに order がずれると
    # 保存済みの答と対応しなくなる。L19 で同じ手当てをしている)
    rows.sort(key=lambda r: (r['work'], r['pid'], r['pan']))
    ans = answers_by_crop()
    nface = nans = 0
    for o, r in enumerate(rows):
        r['order'] = o
        for f in r['faces']:
            nface += 1
            g = ans.get(f['crop']) if f['crop'] else None
            f['word'] = (g.get('word') or '') if g else ''
            f['choice'] = (g.get('choice') or '') if g else ''
            if g:
                nans += 1
    # ---- 既知セル2: L19 の面が全部在るか(名前が合っていることの試験)
    # **走った頁に限って比べる。** 頁を切った走行で全 551面を要求すると
    # **原理的に通れない門**になる —— 2026-10-28 に KC30 で直したのと同じ病を、
    # この器具を書いた同じ日にもう一度踏んだ(`--limit 2` で 18/551 で FAIL)。
    # **頁で絞るのは水準を下げることではない**: 走った頁の上では「全部ある」が要求のまま。
    seen = {r['pid'] for r in rows}
    want = {x['crop'] for x in json.load(open(L19Q, encoding='utf-8'))['queue']
            if x['pid'] in seen}
    got = {f['crop'] for r in rows for f in r['faces'] if f['crop']}
    ok2 = want <= got
    print(f'\n既知セル2 名前: 走った {len(seen)}頁 の L19 の {len(want)} 面が'
          f'今回の {len(got)} 面に全部あるか: '
          f'{"ok" if ok2 else "**FAIL**"}  欠け {len(want - got)}')
    if not ok2:
        print('**`k` の振り方が L19 と違う。答が結べないので止める**')
        return 1
    # ---- 既知セル1: 当たり判定の**往復**
    # 代表点(原寸)-> 画面の座標 -> 1/scale で原寸へ戻す -> 地図を引く。
    # **丸めの往復がここで試される**(地図は縮めていないので、残る危険は丸めだけ)。
    # **2つを分ける**(直すためでなく、機序が違うから):
    #   別の面を返す = **致命**。押した面と違う面が選ばれ、作者の答が黙って壊れる
    #   0 を返す     = **押せないだけ**。一覧からは選べるので、印を付けて数える
    bad, unreach = collections.Counter(), 0
    for r in rows:
        m = cv2.imread(f'{OUT}/panels/{r["tag"]}_map.png', cv2.IMREAD_UNCHANGED)
        if m is None:
            bad['地図が読めない'] += len(r['faces']); continue
        for i, f in enumerate(r['faces'], 1):
            y = int(round(f['sy'] / r['scale']))
            x = int(round(f['sx'] / r['scale']))
            v = (int(m[y, x]) if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] else -1)
            f['clickable'] = bool(v == i)
            if v == i:
                continue
            if v == 0 or v == -1:
                unreach += 1
            else:
                bad['**別の面を返す(致命)**'] += 1
    print(f'既知セル1 当たり判定の往復: 面 {nface}  '
          f'**別の面を返した {sum(bad.values())}**  画像から押せない {unreach}')
    for w, c in (bad.items() if bad else [('(致命の理由は発火していない)', 0)]):
        print(f'   {w}: {c}')
    if bad:
        print('**押した面と違う面が選ばれる。止める**')
        return 1
    # ---- 既知セル3: 面の数(待ち行列と地図の最大値)
    mism = 0
    for r in rows:
        m = cv2.imread(f'{OUT}/panels/{r["tag"]}_map.png', cv2.IMREAD_UNCHANGED)
        if m is None or int(m.max()) != len(r['faces']):
            mism += 1
    print(f'既知セル3 面の数: 待ち行列と地図が食い違うコマ **{mism}**')
    q = dict(n=len(rows), n_face=nface, n_face_answered=nans,
             n_unclickable=int(unreach),
             standard='gated', unit='panel', cross_panel=False,
             built='l38_build_cluster_queue.py', queue_id='l38_panel_v1',
             kc=dict(names_ok=bool(ok2), hit_fatal=int(sum(bad.values())),
                     unclickable=int(unreach), count_mismatch=int(mism)),
             queue=rows)
    with open(f'{OUT}/queue.json', 'w', encoding='utf-8') as fh:
        json.dump(q, fh, ensure_ascii=False)
    sz = sum(os.path.getsize(f'{OUT}/panels/{f}') for f in os.listdir(f'{OUT}/panels'))
    print(f'\n-> {OUT}/queue.json  コマ {len(rows)}  面 {nface}  '
          f'**面ごとの答が在る面 {nans}**')
    print(f'-> {OUT}/panels/  画像 {len(rows) * 3} 枚  {sz / 2**20:.0f} MiB')
    print('**まとまりはコマを跨がない**(作者が選んだ画面の単位。2026-10-28)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
