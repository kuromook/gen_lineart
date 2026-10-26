"""M47: **「この1成分は、あなたにとって何面ですか」を聞く UI の queue** を作る。

**サーバは書かない。** Track L の `l19_fill_label_server.py` を**そのまま流用**する
(L25 が同じ形で流用している実績がある):

```
python ../lineart-fill-region/tools/ui/l19_fill_label_server.py \
    --host 0.0.0.0 --queue results/m47_ui/queue.json \
    --save labels/area_count.jsonl --extra ''
-> http://<host>:8475
```

**なぜ UI か**(基盤、2026-10-26): 作者の回答で **「いくつかの面は一つのマスクで扱う」**
(= 標準は作者の面を束ねている)と **「離れていれば別の面」**(= 作者の面は連結)が確定し、
**成分 <= 作者の面** が導出された。**係数(1成分が何面か)は測るしかなく、
作者の判定以外に経路が無い。** 切り出し2件では「複数」という定性の答までしか出ない。

**設計(走らせる前に登録。条件は基盤の7つ)**:

1. **出す絵**: 原寸1:1 + マゼンタの輪郭。窓(1400×1400)に収まらない成分は
   **(a) 形の把握用の縮小 + (b) 判定用の原寸(くびれ)** の2枚組(`m35` と同じ登録)
2. **答の形**: **1 / 2 / 3 / 4以上**(キー 1〜4)+ **自由入力**(どこで分かれるか)+ **保留**
3. **画面に出さないもの**: **層の名前・こちらの量**(`drawn_ratio` などの数字)。
   **出すと答が名前に汚染される**(L25/L27 の64問が汚染されていない理由)
4. **門(既知セル)**: **作者が既に答えた2件**を混ぜる ——
   `illust201710\page0003` L50(件1b のスーツ)と `skima11\skima12chara` L269(件2 の胴と両脚)。
   **位置は先に記録し、画面には印を出さない**
5. **標本**: **層別**(**名前が `HAIR`/`SKIN` の層からの成分** と **その他**)。
   各層 **14件**(+ 既知セル2件 = **30件**)。**種は 20261026 で固定**
6. **落としたものを理由別に数えて印字**(規律9)
7. **作者の時間**: L の実績(中央 2.0秒/問、34問で5.1分)から、**30問で5分前後**を見込む

**向き(走らせる前に登録)**:

| | 予測 |
|---|---|
| **どの成分も 1以上** | **定義上**(作者の面は連結、連結集合は1成分に収まる)。**0 が出たら器具の誤り** |
| 名前が `HAIR`/`SKIN` の層 | **1 が多い**(作者「髪と肌はそれぞれ1マスクであることが多い」) |
| その他の層 | **2以上が多い** |

**記録する(作者には見せない)**: **その成分の頁が「標準が頁全体を覆う群」かどうか**
(追補34: 8頁中3頁)。

---

## 2026-10-26 修理(作者の「ボタンが無い・キーが効かない」)

**原因**: 流用した `l19_fill_label_server.py` の JS が読む鍵のうち **3つを入れていなかった**。

| 鍵 | 使用箇所 | 入れないと |
|---|---|---|
| **`area`**(`area_px` という名で入れていた) | `it.area.toLocaleString()` | **TypeError で `go()` が中断** → 解答欄も「その他」も DOM に足されず、`keymap`/`okeys` が空のまま。**ボタンが無く、キーも効かない**(症状の両方が1つの原因) |
| `scale` | `const sc=ctx?it.ctx_scale:it.scale` | 縮小表示でも **「(原寸)」と出る**(作者に嘘をつく) |
| `ctx_scale` | 同上 | 同上 |

**この罠の形**: **器具を流用したとき、行列の鍵の名前が唯一の接点**。
「起動の門を通った / JS は構文検査を通った / HTTP 200」は**どれもこの不一致を知らない**
(`it.area` が undefined かどうかは構文検査の外)。
**点検は鍵の名前に当てる** → `m49_check_queue_keys.py`(JS の `it.<鍵>` を全部拾い、
`payload()` が足す鍵 + 行列の鍵と差分を取る)。**m47 はこれを通ってから配る。**

**同時に見つけた2つ(鍵とは別の、絵の側の欠陥)**:

1. **マゼンタが窓の縁にしか無い 2件**(`sunshine2\page0002` L61/L69)。成分が**画布を
   丸ごと覆う**(45,884,640px = 100%)ためで、原寸1400窓では
   **「どこまでが1つの面か」が見えないまま数を聞かれる**。
   → **落とす。** 縮小して頁ぜんたいを出す案を先に書いたが、取り下げた ——
   **マゼンタの枠が「頁の外周」と一致してしまうので、
   どの面について聞かれているのかが絵から分からない**(層の名前を出さない §3 と両立しない)。
   **「1」という尤もらしい答が、別の問いに対して記録される**ほうが、
   答が無いことより悪い(T106 と同じ形)。
   **登録(この UI の限界)**: **標準が頁全体を覆う群の「面」は、
   切り出しの問いにはできない。** 作者に聞く道が要るなら、
   **出せるのは「そのマスクで実際に見えている塗り」(量2)であって標準の面(量1′)ではない** ——
   **別の量に差し替えることになるので、ここでは聞かない**
2. **マゼンタが窓の 66.2% を埋めた1件**(`sunshine2\page0019` L3 = 網点の層)。
   点ごとに輪郭が出て**絵が見えない**。**今夜の網点はこれで3度目**(m27 で2度)。
   → **外周だけ(`RETR_EXTERNAL`)で描き直し、それでも窓の 10% を超えるなら落とす。**

**読める絵の門(走らせる前に登録。3つ)**:

| | 門 | 落ちると |
|---|---|---|
| **(0)** | **窓が単色**(`m27.flat`: share >= 0.05 の色が2つ未満) | 情報の無い切り出し |
| **(1)** | **輪郭が窓の `MAXMG = 0.10` を超える**(外周だけで描き直しても) | 網点の層。絵が見えない |
| **(2)** | **窓の内側にマゼンタが1画素も無い**(輪郭が窓/頁の縁と一致) | どの面を聞かれているか分からない |

**門(0) は m27 が既に持っていた**(`m27_author_crops.py:98` `flat()`、**220行目で呼んでいる**)。
**m47 は流用のときにこれを落としていた。器具を作り直すと、前の器具が通った門が黙って消える。**

**根拠の差し替え(2026-10-26。基盤の差し戻し。規律3: 撤回はパス)**:
最初こちらは「**m27 の走行ログに出ている**」と書いたが、
**基盤が `results/m27_author_crops` を探して 0 件**だった —— 走行ログは
**セッションの scratchpad にしか無く、`results/` には保存していなかった**。
**器具に残るコメントが、ディスクから歩けない出所を指していた**(T106 と同じ族)。
→ **(1) 3本の走行ログを `results/m27_author_crops/run{2,3,4}.log` に移した**(これで歩ける)。
**(2) 基盤が示したより強い根拠に差し替えた**:

| | 歩ける根拠 |
|---|---|
| 門が m27 に在った | `m27_author_crops.py:98-102` の `flat()`、**220行目で実際に呼んでいる**(コードで閉じる) |
| **同じ成分を m27 は落としていた** | `run{2,3,4}.log` に **`捨てる(窓が単色) 000_2017_illustration_sunshine2_page0002 L61`** —— **m47 が問0 として出していたのと同じ層** |
| **m47 の出力自身が証拠** | **問0・問1 は色数2**(シアン 99.14%)。**`flat()` を通していれば必ず落ちる = 通していない**(出力で閉じる) |

**出力自身が門を通っていない証拠になる**という形がいちばん強い ——
**走行ログが消えても残る。**
**基盤が独立に「uniq色 2」で同じ2件に当たった**のは、門(2) と門(0) が同じものを捕まえるからで、
**どちらか片方ではなく両方を印字する**(片方の門が将来の別の形を見落とすため)。

`MAXMG = 0.10` の位置について: 
**観測値は 66.2% と、残り17件すべて 2.0% 以下**(間に何も無い)。
**この門は「絵が読めるか」であって面の数ではない。ただし落ちた成分は
「面が多い側」に偏りうる** —— 落とした数と理由を印字し、ここに残す。
**既知セル(作者が既に答えた2件)は門で落とさない**(落ちたら門の方を疑う)。
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'colour'))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402
import m27_author_crops as m27                           # noqa: E402
import m35_case1_remake as m35                           # noqa: E402

DELIV = m27.DELIV
OUT = Path('results/m47_ui'); (OUT / 'crops').mkdir(parents=True, exist_ok=True)
SEED = 20261026
PER = 14
WIN = 1400
MAT = re.compile(r'(hair|skin|髪|肌)', re.I)
MINFRAC = 0.0005
MAXMG = 0.10            # 輪郭が窓のこれを超えたら「絵が読めない」
KNOWN = [('000_2018_illust201710_page0003', 50),
         ('000_2018_skima11_skima12chara', 269)]
DOMINATED = {'000_2017_illustration_sunshine2_page0002',
             '000_2018_skima11_skima12H', '000_2018_skima11_skima12chara'}
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}


def main():
    rows = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
             if l.strip()]
    m12.D = DELIV
    tot, drop = collections.Counter(), collections.Counter()
    cand = []
    for p in pages:
        if not p['layers']:
            drop['ベース層が無い頁'] += 1
            continue
        prows = byp[p['source']]
        byid = {str(r['layer_id']): r for r in prows}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (_, garea_a), st = m16.measure_standard(
            p, prows, grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        zmap = {c['layer_id']: c['z'] for c in comps}
        page_ids = [i for i, _ in sorted(
            ((l['layer_id'], zmap.get(l['layer_id'], 0)) for l in p['layers']),
            key=lambda t: t[1])]
        for c in comps:
            if c['area'] < MINFRAC * grid[0] * grid[1]:
                drop['成分が小さすぎる(画布の0.05%未満)'] += 1
                continue
            r = byid.get(str(c['layer_id']), {})
            cand.append(dict(c, page=p['page'], source=p['source'], scale=s,
                             grid=grid, files=files, page_ids=page_ids,
                             name=r.get('name', '?') or '?'))
        del comps
        print(f'  {p["page"]:44s} 成分{st["components"]:>4} 面{len(garea_a):>4}',
              flush=True)

    ok = all(tot[k] == EXPECT[k] for k in EXPECT)
    print(f'\n== 既知セル ==')
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if not ok:
        print('**既知セルが合わない。queue を書かない。**')
        return 1

    for c in cand:
        c['stratum'] = '名前が HAIR/SKIN' if MAT.search(c['name']) \
            else 'その他の名前'
    rng = np.random.default_rng(SEED)
    prev = {l['page']: l['preview']['file'] for l in pages}
    q, rej, diag = [], collections.Counter(), []

    def render(c, keep=False):
        """1件を描き、**読めるかどうかを判定する**。読めなければ None。

        `keep=True`(既知セル)は門で落とさない。**落ちたら門の方を疑う。**
        """
        n = len(q) + 1
        H, W = c['grid']
        s = c['scale']
        y, x, h, w = c['box']
        full = np.zeros((H, W), bool)
        full[y:y + h, x:x + w] = c['mask']
        fits = h <= WIN and w <= WIN
        if fits:
            cy, cx = y + h // 2, x + w // 2
        else:
            wst = m35.find_waist(c, c['mask'].astype(np.uint8))
            cy, cx = (wst[0], wst[1]) if wst else (y + h // 2, x + w // 2)
        y0 = max(0, min(H - WIN, cy - WIN // 2))
        x0 = max(0, min(W - WIN, cx - WIN // 2))
        y1, x1 = min(H, y0 + WIN), min(W, x0 + WIN)
        img = m27.composite(c['page_ids'], c['files'], c['box'], (y0, x0, y1, x1))
        sub = full[y0:y1, x0:x1].astype(np.uint8)
        base = f'q{n:03d}_{c["page"]}_L{c["layer_id"]}'
        # --- 門(0): **窓が単色**。`m27.flat`(share >= 0.05 の色が2つ未満)を
        # **そのまま使う** —— **m27 が既に持っていた門を、m47 は流用のときに落としていた。**
        # **歩ける根拠(2つ。基盤の差し戻しで差し替えた)**:
        #  (a) `results/m27_author_crops/run{2,3,4}.log` に
        #      **`捨てる(窓が単色) 000_2017_illustration_sunshine2_page0002 L61`** —— 
        #      **m47 が問0 として出していたのと同じ成分を、m27 は落としていた**
        #  (b) **m47 の出力自身が証拠**: 問0・問1 は色数2(シアン 99.14%)。
        #      `flat()` を通していれば必ず落ちる = **通していない**
        # **器具を作り直すと、前の器具が通った門が黙って消える。**
        nc = int((np.unique(m27.quant(img), return_counts=True)[1] /
                  (img.shape[0] * img.shape[1]) >= 0.05).sum())
        if m27.flat(img) and not keep:
            rej['窓が単色(情報の無い切り出し)'] += 1
            print(f'  **落とす** {c["page"]} L{c["layer_id"]} '
                  f'窓が単色(share>=5% の色 {nc})', flush=True)
            return None
        # --- 原寸の窓に輪郭を描く。**窓を埋めたら外周だけで描き直す**(網点の層)
        mode, mk, mg, frac = '', None, None, 1.0
        for mode, retr in (('全部の輪郭', cv2.RETR_CCOMP),
                           ('外周だけ', cv2.RETR_EXTERNAL)):
            mk = img.copy()
            cs, _ = cv2.findContours(sub, retr, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mk, cs, -1, (255, 0, 255), 3)
            mg = (mk[..., 2] > 200) & (mk[..., 0] > 200) & (mk[..., 1] < 60)
            frac = float(mg.mean())
            if frac <= MAXMG:
                break
        if frac > MAXMG and not keep:
            rej[f'絵が読めない(輪郭が窓の{MAXMG:.0%}超)'] += 1
            print(f'  **落とす** {c["page"]} L{c["layer_id"]} '
                  f'輪郭が窓の {frac:.1%}(外周だけでも埋まる = 網点の層)',
                  flush=True)
            return None
        if frac > MAXMG:
            print(f'  **既知セルが門に落ちた(残す)** {c["page"]} '
                  f'L{c["layer_id"]} {frac:.1%}', flush=True)
        # --- 頁の中での形と位置(縮小)
        pv = cv2.imread(str(DELIV / prev[c['page']]), cv2.IMREAD_COLOR)
        sm = full[::s, ::s].astype(np.uint8)
        sh, sw = min(sm.shape[0], pv.shape[0]), min(sm.shape[1], pv.shape[1])
        shape = pv.copy()
        cs2, _ = cv2.findContours(sm[:sh, :sw], cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shape, cs2, -1, (255, 0, 255), 2)
        ctxim = shape.copy()
        cv2.rectangle(ctxim, (x0 // s, y0 // s), (x1 // s, y1 // s),
                      (0, 180, 255), 3)
        cv2.imwrite(str(OUT / 'crops' / f'{base}_ctx.png'), ctxim)
        # --- 窓の縁にしかマゼンタが無い = 原寸の窓では面の切れ目が見えない
        inner = mg.copy()
        inner[:6, :] = False; inner[-6:, :] = False
        inner[:, :6] = False; inner[:, -6:] = False
        border_only = bool(inner.sum() == 0)
        if border_only and not keep:
            # **輪郭が頁の縁と一致する** = 成分が画布を丸ごと覆う。
            # 縮小して頁ぜんたいを出しても、**マゼンタの枠は「頁の外周」でしかなく、
            # どの面について聞かれているのかが絵から分からない**
            # (層の名前は出さない規律3 と両立しない)。
            # **「1」という尤もらしい答が、別の問いに対して記録される**のが最悪で、
            # 答が無いことより悪い(T106 と同じ形)。→ **聞かない。**
            rej['絵から問いが分からない(成分が画布を覆い、輪郭が頁の縁)'] += 1
            print(f'  **落とす** {c["page"]} L{c["layer_id"]} '
                  f'輪郭が頁の縁と一致(成分が画布の {c["area"] / (H * W):.1%})',
                  flush=True)
            return None
        cv2.imwrite(str(OUT / 'crops' / f'{base}_crop.png'), mk)
        scale = 1.0
        if border_only:
            print(f'  **既知セルの輪郭が頁の縁(残す)** {c["page"]} '
                  f'L{c["layer_id"]}', flush=True)
        diag.append((base, round(frac, 4), mode, scale, bool(fits), nc))
        print(f'  描いた {n:>3}  {base}  輪郭 {frac:>6.2%}  {mode}  '
              f'表示 {scale}', flush=True)
        del full, sub, img, mk, pv, ctxim, shape
        return {'order': n - 1, 'crop': f'{base}_crop.png',
                'ctx': f'{base}_ctx.png',
                # **サーバの JS が読む鍵**(m49 が機械的に突き合わせる)
                'area': int(c['area']), 'scale': scale,
                'ctx_scale': round(1.0 / s, 4),
                # **画面に出さない記録**
                'pid': c['source'], 'lid': c['layer_id'],
                'why': ('known' if (c['page'], c['layer_id']) in KNOWN
                        else c['stratum']),
                'work': c['page'], 'fits_window': bool(fits),
                'outline': mode,
                'page_covered_by_one_area': c['page'] in DOMINATED,
                'area_px': int(c['area']),   # 旧名も残す(配った記録との照合用)
                'cands': [], 'word_share': {}, 'top_word': '', 'purity': 0.0,
                'n_lay': 0, 'n_lay90': 0, 'auto': 0, 'comp': 0,
                'region': c['layer_id']}

    for st_lab in ('名前が HAIR/SKIN', 'その他の名前'):
        pool = sorted([c for c in cand if c['stratum'] == st_lab],
                      key=lambda c: (c['page'], c['layer_id'], -c['area']))
        got = 0
        for i in rng.permutation(len(pool)):   # **順も種も前の走行と同じ**
            if got >= PER:
                break
            it = render(pool[int(i)])
            if it is not None:
                q.append(it); got += 1
        print(f'  層 {st_lab:20s} 候補 {len(pool):>4} → 引いた {got}')
    for pg, lid in KNOWN:
        hit = [c for c in cand if c['page'] == pg and c['layer_id'] == lid]
        if not hit:
            print(f'  **既知セルが候補に無い**: {pg} L{lid}')
            continue
        if any(x['work'] == pg and x['lid'] == lid for x in q):
            continue
        it = render(max(hit, key=lambda c: c['area']), keep=True)
        if it is not None:
            q.append(it)
            print(f'  既知セルを足した: {pg} L{lid}')

    json.dump(dict(n=len(q), n_ask=len(q), n_check=0, n_auto=0, n_all=len(q),
                   words=['1', '2', '3', '4以上'],
                   ask='マゼンタで囲んだ範囲は、何面ですか?',
                   help='原寸 1:1 の切り出し。マゼンタ = こちらが「1つの面」と'
                        '数えた範囲の外周。C を押すと頁のどこかが分かる'
                        '(縮小。オレンジの枠が今見ている範囲)。'
                        '分かれると思う数を押し、どこで分かれるかは T で一言。',
                   others=[['free', 'どこで分かれるか(一言)', 'T'],
                           ['mixed', '保留(後で見直す)', 'M']],
                   queue=q),
              open(OUT / 'queue.json', 'w'), ensure_ascii=False)
    print(f'\n== 落としたもの(理由別)==')
    for k, v in drop.most_common():
        print(f'  {k:34s} {v:>5}')
    print(f'\n== 標本 ==')
    for k, v in collections.Counter(x['why'] for x in q).most_common():
        print(f'  {k:20s} {v:>3} 件')
    print(f'  窓に収まらない成分(2枚組になる) '
          f'{sum(1 for x in q if not x["fits_window"])}')
    print(f'  標準が頁全体を覆う群の成分 '
          f'{sum(1 for x in q if x["page_covered_by_one_area"])}'
          f'(**作者には見せない記録**)')
    print(f'\n== 絵の読みやすさ(門 {MAXMG:.0%})==')
    for base, frac, mode, scale, fits, nc in diag:
        print(f'  {base:56s} 輪郭 {frac:>7.2%}  色 {nc:>2}  '
              f'表示 {scale:<7} {mode}')
    print(f'\nwrote {OUT}/queue.json  ({len(q)} 問)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
