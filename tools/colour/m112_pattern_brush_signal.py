"""**パターンブラシの層**を見分ける手がかりを測る(**作者の裁定、2026-11-01 夕**)。

作者の言明(逐語): **「この３枚の切り出しの星は clipstduiopaint のパターンブラシだ
特定の像をパターンとして登録したブラシで、このブラシで線を引くとパターンをストロークに
沿って描く 筆圧やペンの傾き・軸の回転によって濃度やサイズを変わるようにすることもできる
線というものをグループ化するならこれは分けて考えるべき対象だろう 作者の絵でもなければ
３Dの出力でもないものであり、機械的（数学的）に作られるものもあれば手描きのパターンも有り、
分類にとってはノイズになる」**

→ **線の出自は4つになった**: ① 作者の手描き ② 3D の機械生成(`rendering_lineart`)
③ 塗り・マスクの境目 ④ **パターンブラシ**。**④ は分類にとってノイズ**(作者)。

**この器具が測るのは「見分けられるか」だけで、棒ではない。**

**手がかり(走る前に決めた)**: パターンブラシは**同じ像を繰り返し押す**ので、
層のアルファの連結成分に**合同な形が何度も現れる**はずである。
  - 対象 = **可視の葉**(`folder != True`)でラスタが在る層。クリップは問わず、**印字する**
  - 成分 = アルファの **8連結**。**面積 `MINA` 未満の破片は落とし、落とした数を印字する**(規律9)
  - 署名 = **(外接箱の高さ, 幅, 面積)** の厳密一致
  - 指標 **`rep_frac`** = 署名が2回以上現れる成分の画素が、その層の墨に占める割合

**この手がかりは下界である。** 作者が言うとおり、パターンブラシは
**筆圧・傾き・軸の回転で濃度やサイズが変わり得る**。そうなると署名は一致しないので、
**`rep_frac` は本来のパターンブラシを取りこぼす。高い値は証拠だが、低い値は反証ではない。**

**棒を置かない。** 分布を印字し、層238(作者が名指しした層)がどこに来るかを見る。
**作者が名指しした1枚は既知セルであって、母集団ではない**(規律6。判断は作者のもの)。
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m77_standard_visible as m77                                   # noqa: E402
import m87_split_standard_by_occlusion as m87                        # noqa: E402
import m99_b_standard as m99                                         # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402

OUT = Path('results/m112_pattern_brush')
MINA = 16            # この面積未満の成分は落とす(**落とした数を印字する**)
KNOWN = ('000_2018_illust201710_illust201710_page0005', '238')   # 作者が名指しした層


def signal(a):
    """1層のアルファから、繰り返しの手がかりを出す。"""
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = a[y0:y1, x0:x1].astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    sig = collections.Counter()
    px = collections.Counter()
    small = 0
    for i in range(1, n):
        ar = int(st[i, cv2.CC_STAT_AREA])
        if ar < MINA:
            small += 1
            continue
        k = (int(st[i, cv2.CC_STAT_HEIGHT]), int(st[i, cv2.CC_STAT_WIDTH]), ar)
        sig[k] += 1
        px[k] += ar
    kept = sum(sig.values())
    tot = sum(px.values())
    if kept == 0:
        return dict(n_comp=0, dropped_small=small, rep_frac=0.0, rep_comp=0,
                    top_sig='', top_n=0, ink_px=int(a.sum()), box=f'{y1-y0}x{x1-x0}')
    rep = [k for k, c in sig.items() if c >= 2]
    return dict(n_comp=kept, dropped_small=small,
                rep_frac=sum(px[k] for k in rep) / tot,
                rep_comp=sum(sig[k] for k in rep),
                top_sig='x'.join(map(str, sig.most_common(1)[0][0])),
                top_n=sig.most_common(1)[0][1],
                ink_px=int(a.sum()), box=f'{y1-y0}x{x1-x0}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    v3 = {r['page'] for r in csv.DictReader(
        open('results/m104_b_standard_v3/b_standard_v3.csv', encoding='utf-8-sig'))}
    pages = [p for p in pages if p['page'] in v3]
    print(f'母集団 `pop_6pages_lineart` = {len(pages)} 頁 / 面積の床 {MINA} px\n')

    rows, drop = [], collections.Counter()
    for p in pages:
        vis, _ = m77.visible_rows(byp[p['source']])
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        n_leaf = 0
        for r in vis:
            if r['folder'] == 'True':
                drop['フォルダー'] += 1
                continue
            if r['layer_type'] == m104.PAPER:
                drop['用紙'] += 1
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, {})
            if a is None:
                drop['ラスタが無い'] += 1
                continue
            g = signal(a)
            del a
            if g is None:
                drop['アルファが空'] += 1
                continue
            n_leaf += 1
            g.update(page=p['page'], layer_id=str(r['layer_id']),
                     parent_id=r['parent_id'], order_in_parent=r['order_in_parent'],
                     clip=r['clip'], layer_type=r['layer_type'],
                     formats=r['formats'], sat_mean=r['sat_mean'],
                     colours_q16=r['colours_q16'], drawn_ratio=r['drawn_ratio'],
                     v_layer_kind=kinds.get((p['source'], str(r['layer_id'])), ''),
                     is_line=int(m99.is_line(r)), name=r['name'])
            rows.append(g)
        print(f'{p["page"]:44s} 測った葉 {n_leaf:>3}')
    print('\n落とした数(理由別。規律9): ' + ' / '.join(f'{k} {v}'
                                                 for k, v in drop.most_common()))

    with open(OUT / 'pattern_signal.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\n=== IC 作者が名指しした層(既知セル。**母集団ではない**) ===')
    k = [r for r in rows if (r['page'], r['layer_id']) == KNOWN]
    if not k:
        print('  **層238 が表に無い。手がかりは読まない**')
        return 1
    r = k[0]
    print(f'  層238 `{r["name"]}` 成分 {r["n_comp"]:,} / 落とした破片 '
          f'{r["dropped_small"]:,} / **rep_frac {r["rep_frac"]:.4f}** / '
          f'繰り返す成分 {r["rep_comp"]:,} / 最頻の署名 {r["top_sig"]} が '
          f'{r["top_n"]} 回 / 墨 {r["ink_px"]:,}px / 箱 {r["box"]}')

    print('\n=== A. `rep_frac` の上位20(**全6頁の葉をまとめて並べる。頁ごとは §B**) ===')
    print(f'{"頁":<14}{"層":>6}{"ｸﾘｯﾌﾟ":>6}{"線":>4}{"rep_frac":>10}{"成分":>8}'
          f'{"繰返":>7}{"最頻の署名":>14}{"回":>5}{"墨px":>12}  名前')
    for r in sorted(rows, key=lambda x: -x['rep_frac'])[:20]:
        print(f'{r["page"][-13:]:<14}{r["layer_id"]:>6}'
              f'{("1" if r["clip"] == "1" else "-"):>6}'
              f'{("線" if r["is_line"] else "-"):>4}{r["rep_frac"]:>10.4f}'
              f'{r["n_comp"]:>8,}{r["rep_comp"]:>7,}{r["top_sig"]:>14}'
              f'{r["top_n"]:>5}{r["ink_px"]:>12,}  {r["name"]}')

    print('\n=== B. 頁ごとの分布(**プールしない**。条例2) ===')
    print(f'{"頁":<44}{"葉":>5}{"rep_frac ≥0.5":>14}{"≥0.2":>7}{"中央":>8}{"最大":>8}')
    for p in pages:
        g = [r for r in rows if r['page'] == p['page']]
        if not g:
            continue
        v = sorted(r['rep_frac'] for r in g)
        print(f'{p["page"]:<44}{len(g):>5}{sum(1 for x in v if x >= 0.5):>14}'
              f'{sum(1 for x in v if x >= 0.2):>7}'
              f'{v[len(v)//2]:>8.3f}{v[-1]:>8.3f}')

    print('\n=== C. 線と名付けた層との重なり(**`線画` にパターンブラシは混ざっているか**) ===')
    ln = [r for r in rows if r['is_line']]
    print(f'  `線画` の葉 {len(ln)} 枚 / rep_frac ≥0.5 が {sum(1 for r in ln if r["rep_frac"]>=0.5)} 枚'
          f' / ≥0.2 が {sum(1 for r in ln if r["rep_frac"]>=0.2)} 枚')
    for r in sorted(ln, key=lambda x: -x['rep_frac'])[:5]:
        print(f'    {r["page"][-13:]:<14}{r["layer_id"]:>6} rep_frac {r["rep_frac"]:.4f}'
              f' 成分 {r["n_comp"]:,} 最頻 {r["top_sig"]} x{r["top_n"]}  {r["name"]}')

    print('\n**この手がかりは下界である** —— 筆圧・傾き・軸の回転で濃度やサイズが変わると'
          '署名が一致しないので、**高い値は証拠だが低い値は反証ではない**(作者の言明)')
    print(f'\n表: {OUT / "pattern_signal.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
