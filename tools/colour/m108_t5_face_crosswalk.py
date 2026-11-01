"""`T4` の面と `T5` の面を**画素で突き合わせ**、線率 0 → 1 の跳びの正体を分ける。

**これは診断であって棒ではない。** `T4` も `T5` も第3版の値をそのまま使う
(規律10。結果を見てから棒を動かさない)。`T5` の線率の上がりは**定義による**ので、
第3版の登録どおり**棒の証拠にしない**。

**何を分けるのか**(追補170 §読み): `page0005` で `T4` の線率の中央は 0.0000、
`T5` では 1.0000 に跳ぶ。この跳びには**別々の二つの機構**がありうる。
  (あ) **同じ面の線率が上がった** —— 面はそのまま在って、その境目にあった
       非墨の境界が `rendering_lineart` の墨に変わった。**定義どおりの上がり**
  (い) **面の集団が入れ替わった** —— 線率0 だった面が面として消え(新しい墨に
       吸われる・外周に通じて開く・床下に落ちる)、残った面の多数派が別物になった
**中央値だけでは (あ) と (い) を区別できない**(罠 T-M34)。だから画素で突き合わせる。

**走る前の予想**(2026-11-01、出力は1行も読んでいない。**これは後知恵ではない**):
  予想1 `T4` の遠い面(墨まで 16px 以上)254面のうち、**過半数は `T5` では
        「閉じた面」として残らない** —— (い) が主である
  予想2 残ったものの線率は 0 から上がる —— (あ) も同時に起きる
  予想3 `T5` の 412面には、`T4` に対応する閉じた面が無い**新しい面**が含まれる
  **反証**: もし 254面がほぼ全部 `T5` でも閉じた面として残り、線率だけ 0 → 1 に
  上がっていたなら (あ) だけで説明が付き、**予想1 は外れる**。**両方印字する**。

**門は頁ごとに当てる**(条例3)。既知セルは第3版の走行の行
(`results/m104_b_standard_v3/b_standard_v3.csv`)の `T4_*` と `T5_*`:
  閉じた面 / 線率 中央 / 線率<0.5 の面 / 床で落ちた数 —— **8つ × 6頁 = 48件**。
**1頁でも落ちたら、その頁の突き合わせは読まない。**

**母集団は4つ全部印字する**(規律9)。
"""
import collections
import csv
import os
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402
import m107_face_strata_by_ink_distance as m107                      # noqa: E402

OUT = Path('results/m108_t5_crosswalk')
V3 = Path('results/m104_b_standard_v3/b_standard_v3.csv')
NEAR = m107.NEAR          # 「墨に接している」の境(px)。**m107 と同じ値を借りる**
FAR = m107.FAR            # 「離れている」の境(px)
POPS = m107.POPS

# `T5` での行き先の名前(**落とした先を理由別に数える**。規律9)
DEST = ('墨に吸われた', '閉じた面', '開いた', '床下')


def med(v):
    return m107.med(v)


def labels(bnd):
    n, lab, st, _ = cv2.connectedComponentsWithStats((~bnd).astype(np.uint8),
                                                     connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    return n, lab, st, border


def classify(n, st, border, kept, floor_px):
    """ラベル → 行き先の名前。**0 は境界(墨か遮蔽物の縁)**。"""
    cls = np.empty(n, object)
    cls[0] = '墨に吸われた'
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < floor_px:
            cls[i] = '床下'
        elif i in border:
            cls[i] = '開いた'
        elif i in kept:
            cls[i] = '閉じた面'
        else:
            cls[i] = '開いた'            # 床以上・外周に触れていない・なのに残っていない
    return cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = {}
    for r in csv.DictReader(open(V3, encoding='utf-8-sig')):
        known[r['page']] = {
            f'{s}_{k}': v for s in ('T4', 'T5')
            for k, v in (('closed', int(r[f'{s}_closed_n'])),
                         ('med', round(float(r[f'{s}_ratio_med']), 4)),
                         ('lt05', int(r[f'{s}_ratio_lt05_n'])),
                         ('under', int(r[f'{s}_under_floor'])))}
    print(f'既知セル: 第3版の走行の {len(known)} 頁 × 8件 = {len(known)*8} 件')

    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    pages = [p for p in pages if p['page'] in known]
    only = os.environ.get('M108_ONLY')       # smoke 用。**絞ったら必ず印字する**
    if only:
        pages = [p for p in pages if only in p['page']]
        print(f'**M108_ONLY={only} で頁を絞った(smoke)。'
              f'母集団の表は読まない** —— {len(pages)} 頁')
    else:
        print(f'母集団 `pop_6pages_lineart` = {len(pages)} 頁\n')

    per_page, gate_fail, rows_out = {}, [], []
    for p in pages:
        ex = {}
        ink4, bnd4, floor_px, n_line, n_edge = m106.build(p, byp, kinds, extras=ex)
        ink5 = ex['ink5']
        om5 = m104.open_from(ink5)
        bnd5 = ink5 | (ex['E'] & om5)
        del om5, ex['E']

        rows4, under4 = m106.faces(bnd4, ink4, floor_px)
        rows5, under5 = m106.faces(bnd5, ink5, floor_px)
        got = {}
        for s, rows, under in (('T4', rows4, under4), ('T5', rows5, under5)):
            got[f'{s}_closed'] = len(rows)
            got[f'{s}_med'] = round(med([r['ratio'] for r in rows]), 4)
            got[f'{s}_lt05'] = sum(1 for r in rows if r['ratio'] < 0.5)
            got[f'{s}_under'] = under
        k = known[p['page']]
        ok = all(abs(got[x] - k[x]) < 1e-9 for x in k)
        print(f'{p["page"]:44s} 線 {n_line:>3} / 縁 {n_edge:>3} / '
              f'R7b {ex["n_r7b"]:>2} / 床 {floor_px:>6} px')
        for s in ('T4', 'T5'):
            print(f'    門 {s} ' + ' '.join(
                f'{x.split("_")[1]}:{k[s+"_"+x.split("_")[1]]}/{got[x]}'
                f'{"" if abs(got[x]-k[x])<1e-9 else "**FAIL**"}'
                for x in (f'{s}_closed', f'{s}_med', f'{s}_lt05', f'{s}_under')))
        print('    ' + ('**PASS**' if ok else '**落ちた。この頁の突き合わせは読まない**'))
        if not ok:
            gate_fail.append(p['page'])
            del ink4, bnd4, ink5, bnd5
            continue

        # ---- 画素で突き合わせる
        n4, lab4, st4, b4 = labels(bnd4)
        n5, lab5, st5, b5 = labels(bnd5)
        del bnd4, bnd5
        keep5 = {r['face'] for r in rows5}
        cls5 = classify(n5, st5, b5, keep5, floor_px)
        info5 = {r['face']: r for r in rows5}

        keep4 = np.zeros(n4, bool)
        for r in rows4:
            keep4[r['face']] = True
        sel = keep4[lab4]
        a4 = lab4[sel].astype(np.int64)
        a5 = lab5[sel].astype(np.int64)
        del sel
        key, cnt = np.unique(a4 * n5 + a5, return_counts=True)
        del a4, a5, lab4, lab5, ink4, ink5
        hist = collections.defaultdict(list)
        for kk, c in zip(key.tolist(), cnt.tolist()):
            hist[kk // n5].append((c, kk % n5))

        for r in rows4:
            h = sorted(hist[r['face']], reverse=True)
            c, dst = h[0]
            tot = sum(x[0] for x in h)
            r['dest'] = cls5[dst]
            r['dest_frac'] = c / tot
            d5 = info5.get(dst)
            r['t5_ratio'] = d5['ratio'] if (d5 is not None
                                            and r['dest'] == '閉じた面') else ''
            r['t5_d_ink'] = d5['d_ink'] if (d5 is not None
                                            and r['dest'] == '閉じた面') else ''
            r['t5_face'] = dst if r['dest'] == '閉じた面' else ''
            r['page'] = p['page']
            rows_out.append(r)
        claimed = {r['t5_face'] for r in rows4 if r['t5_face'] != ''}
        per_page[p['page']] = dict(rows4=rows4, rows5=rows5,
                                   new5=[r for r in rows5
                                         if r['face'] not in claimed])
        del hist, key, cnt, st4, st5, cls5

    if gate_fail:
        print(f'\n**門が落ちた頁 {len(gate_fail)}: ' + ' / '.join(gate_fail) + '**')
    if not per_page:
        return 1

    with open(OUT / 'crosswalk_6pages.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['page'] + [c for c in rows_out[0]
                                                     if c != 'page'])
        w.writeheader()
        w.writerows(rows_out)

    print('\n\n=== A. `T4` の面は `T5` でどこへ行ったか(**頁ごと。近い/遠いで分ける**) ===')
    print(f'{"頁":<30}{"群":<5}{"面":>5}' + ''.join(f'{d:>13}' for d in DEST)
          + f'{"残った面の線率 T4→T5(中央)":>24}')
    for pg, d in per_page.items():
        for lab, grp in (('近い', [r for r in d['rows4'] if r['d_ink'] < NEAR]),
                         ('遠い', [r for r in d['rows4'] if r['d_ink'] >= FAR]),
                         ('中間', [r for r in d['rows4']
                                   if NEAR <= r['d_ink'] < FAR])):
            if not grp:
                continue
            c = collections.Counter(r['dest'] for r in grp)
            st = [r for r in grp if r['dest'] == '閉じた面']
            arrow = (f'{med([r["ratio"] for r in st]):.4f} → '
                     f'{med([r["t5_ratio"] for r in st]):.4f}') if st else '—'
            print(f'{pg[-24:]:<30}{lab:<5}{len(grp):>5}'
                  + ''.join(f'{c.get(x, 0):>13}' for x in DEST)
                  + f'{arrow:>24}')

    print('\n=== B. `T5` の面の出自(**どちらの機構で中央値が動いたか**) ===')
    print(f'{"頁":<30}{"T5 面":>7}{"T4 の近い面から":>17}{"T4 の遠い面から":>17}'
          f'{"T4 の中間から":>15}{"新しい面":>10}{"新しい面の線率":>16}'
          f'{"T4 P2":>8}{"T5 P2":>8}')
    judge = {}
    for pg, d in per_page.items():
        src = {}
        for lab, lo, hi in (('近', 0, NEAR), ('遠', FAR, 1e18),
                            ('中', NEAR, FAR)):
            src[lab] = sum(1 for r in d['rows4']
                           if lo <= r['d_ink'] < hi and r['dest'] == '閉じた面')
        new = d['new5']
        p2_4 = med([r['ratio'] for r in d['rows4']])
        p2_5 = med([r['ratio'] for r in d['rows5']])
        judge[pg] = dict(n5=len(d['rows5']), new=len(new), src=src,
                         p2_4=p2_4, p2_5=p2_5)
        nm = f'{med([r["ratio"] for r in new]):.4f}' if new else '—'
        print(f'{pg[-24:]:<30}{len(d["rows5"]):>7}{src["近"]:>17}{src["遠"]:>17}'
              f'{src["中"]:>15}{len(new):>10}{nm:>16}'
              f'{p2_4:>8.3f}{p2_5:>8.3f}')

    print('\n=== C. 母集団ごと(**4つ全部印字する**。規律9) ===')
    for name, excl in POPS.items():
        sel = [pg for pg in per_page if not (excl and any(e in pg for e in excl))]
        if not sel:
            print(f'  {name:<26} 0 頁')
            continue
        a = ' '.join('%.2f' % judge[pg]['p2_4'] for pg in sel)
        b = ' '.join('%.2f' % judge[pg]['p2_5'] for pg in sel)
        c = ' '.join(str(judge[pg]['new']) for pg in sel)
        print(f'  {name:<26} {len(sel)} 頁 / T4 P2 {a} / T5 P2 {b} / 新しい面 {c}')
    print('  **頁ごとの列を必ず並べる**(幅だけで動きを読まない。条例2)')

    print('\n=== D. 予想の答え合わせ(**棒ではない。診断である**) ===')
    for pg, d in per_page.items():
        far = [r for r in d['rows4'] if r['d_ink'] >= FAR]
        if len(far) < 10:
            print(f'  {pg[-24:]:<26} 遠い面 {len(far)} —— **10面未満。判定保留**')
            continue
        gone = sum(1 for r in far if r['dest'] != '閉じた面')
        st = [r for r in far if r['dest'] == '閉じた面']
        up = sum(1 for r in st if r['t5_ratio'] > 0)
        j = judge[pg]
        print(f'  {pg[-24:]:<26} 遠い面 {len(far)}')
        print(f'    予想1 過半数が閉じた面として残らない: '
              f'{gone}/{len(far)} = {gone/len(far)*100:.1f}% '
              f'→ **{"当たり" if gone > len(far)/2 else "外れ"}**')
        print(f'    予想2 残ったものの線率が 0 から上がる: '
              f'{up}/{len(st)} 面 → **'
              f'{"当たり" if st and up > len(st)/2 else ("外れ" if st else "残り 0 面。判定不能")}**')
        print(f'    予想3 `T5` に新しい面が在る: {j["new"]}/{j["n5"]} '
              f'→ **{"当たり" if j["new"] > 0 else "外れ"}**')
    print('\n  **`T5` の線率の上がりは定義による**(`rendering_lineart` を墨に入れた)。'
          '第3版の登録どおり、**棒の証拠にしない**')
    print(f'\n  表: {OUT / "crosswalk_6pages.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
