"""面を「最も近い `線画` の墨までの距離」で層別して、頁ごとに線率を印字する。

**これは診断であって棒ではない。** 主値 `P2`(頁ごとの線率の中央値)は**動かさない** ——
第3版の値はそのまま有効である(規律10。結果を見てから棒を動かさない)。

**なぜ測るのか**(追補169 §B): `page0005` で面を墨までの距離で束ねたら、
**墨に接している 226面は線率 中央 0.7744・線率ちょうど0 が 0面**、
**16px 以上離れた 256面は全部 線率ちょうど0** という二山が出た。
**中央値が 0 に落ちたのは、ゼロの面が16面多いだけ**(482面の 3.3%)。
→ **この二山が 3D の頁だけの性質か、どの頁にも在るかは未測定だった。** それを測る。

**後知恵であることを明記する**: この層別は `page0005` を見てから思いついた。
**予想ではない。** だから**これを棒にはしない**。

**門は頁ごとに当てる**(条例3)。既知セルは第3版の走行の行
(`results/m104_b_standard_v3/b_standard_v3.csv` の `T4_*`):
  閉じた面 / 線率 中央 / 線率<0.5 の面 / 床で落ちた数 —— **4つ × 6頁 = 24件**。
**1頁でも落ちたら、その頁の層別は読まない。**

**母集団は4つ全部印字する**(規律9。選別は測定):
  `pop_6pages_lineart` / `pop_5pages_no_page0005` / `pop_4pages_colour` /
  `pop_3pages_workflow`(現行)
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402

import json                                                          # noqa: E402

OUT = Path('results/m107_face_strata')
V3 = Path('results/m104_b_standard_v3/b_standard_v3.csv')
BINS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 10 ** 9)]
NEAR = 1.0          # 「墨に接している」の境(px)。**層別の境は先に決める**
FAR = 16.0          # 「離れている」の境(px)

# 母集団(登録どおり。頁の名前で引く。**外した理由は作者の裁定で、点数と独立**)
POPS = {
    'pop_6pages_lineart': None,
    'pop_5pages_no_page0005': {'page0005'},
    'pop_4pages_colour': {'page0005', 'page0019'},
    'pop_3pages_workflow': {'page0005', 'page0019', 'page0002'},
}


def med(v):
    v = sorted(v)
    if not v:
        return float('nan')
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = {}
    for r in csv.DictReader(open(V3, encoding='utf-8-sig')):
        known[r['page']] = dict(closed=int(r['T4_closed_n']),
                                med=round(float(r['T4_ratio_med']), 4),
                                lt05=int(r['T4_ratio_lt05_n']),
                                under=int(r['T4_under_floor']))
    print(f'既知セル: 第3版の走行の {len(known)} 頁 × 4件 = {len(known)*4} 件')

    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    pages = [p for p in pages if p['page'] in known]
    print(f'母集団 `pop_6pages_lineart` = {len(pages)} 頁\n')

    rows_out, per_page, gate_fail = [], {}, []
    for p in pages:
        ink, bnd, floor_px, n_line, n_edge = m106.build(p, byp, kinds)
        rows, under = m106.faces(bnd, ink, floor_px)
        got = dict(closed=len(rows), med=round(med([r['ratio'] for r in rows]), 4),
                   lt05=sum(1 for r in rows if r['ratio'] < 0.5), under=under)
        k = known[p['page']]
        ok = all(abs(got[x] - k[x]) < 1e-9 for x in k)
        print(f'{p["page"]:44s} 線 {n_line:>3} / 縁 {n_edge:>3} / 床 {floor_px:>6} px')
        print('    門 ' + ' '.join(
            f'{x}:{k[x]}/{got[x]}{"" if abs(got[x]-k[x])<1e-9 else "**FAIL**"}'
            for x in ('closed', 'med', 'lt05', 'under'))
            + ('  **PASS**' if ok else '  **落ちた。この頁の層別は読まない**'))
        if not ok:
            gate_fail.append(p['page'])
            continue
        for r in rows:
            r['page'] = p['page']
            rows_out.append(r)
        per_page[p['page']] = rows
        del ink, bnd

    if gate_fail:
        print(f'\n**門が落ちた頁 {len(gate_fail)}: ' + ' / '.join(gate_fail) + '**')
    if not per_page:
        return 1

    with open(OUT / 'faces_6pages.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['page'] + [c for c in rows_out[0]
                                                     if c != 'page'])
        w.writeheader()
        w.writerows(rows_out)

    print('\n\n=== A. 距離で束ねた線率(**頁ごと。プールしない**) ===')
    hdr = '    '.join(f'{lo}–{hi if hi < 10**9 else "∞"}' for lo, hi in BINS)
    print(f'{"頁":<44}{hdr}')
    for pg, rows in per_page.items():
        cells = []
        for lo, hi in BINS:
            sel = [r['ratio'] for r in rows if lo <= r['d_ink'] < hi]
            cells.append(f'{len(sel):>4}/{med(sel):.2f}' if sel else '   -/-  ')
        print(f'{pg:<44}' + ' '.join(cells))
    print('    (各欄 = 面の数 / その束の線率の中央値)')

    print('\n=== B. 二山か(**頁ごと**。近い = 墨まで < 1px / 遠い = 16px 以上) ===')
    print(f'{"頁":<44}{"近い面":>7}{"線率中央":>9}{"ちょうど0":>9}'
          f'{"遠い面":>7}{"線率中央":>9}{"ちょうど0":>9}{"全面":>6}{"P2":>7}{"余裕":>6}')
    summ = {}
    for pg, rows in per_page.items():
        near = [r for r in rows if r['d_ink'] < NEAR]
        far = [r for r in rows if r['d_ink'] >= FAR]
        nz = sum(1 for r in near if r['ratio'] == 0.0)
        fz = sum(1 for r in far if r['ratio'] == 0.0)
        z = sum(1 for r in rows if r['ratio'] == 0.0)
        n = len(rows)
        # 中央値が 0 から動くのに必要な「ゼロの面の減り」
        slack = z - (n // 2 - 1 + (0 if n % 2 else 0)) if z > 0 else None
        slack = (z - (n - 1) // 2) if n % 2 else (z - (n // 2 - 1))
        p2 = med([r['ratio'] for r in rows])
        summ[pg] = dict(near=len(near), near_med=med([r['ratio'] for r in near]),
                        near_zero=nz, far=len(far),
                        far_med=med([r['ratio'] for r in far]), far_zero=fz,
                        all=n, p2=p2, zero=z,
                        slack=(slack if p2 == 0 else None))
        s = summ[pg]
        print(f'{pg:<44}{s["near"]:>7}{s["near_med"]:>9.4f}{nz:>9}'
              f'{s["far"]:>7}{s["far_med"]:>9.4f}{fz:>9}{n:>6}{p2:>7.3f}'
              + (f'{s["slack"]:>6}' if s['slack'] is not None else '     -'))
    print('    「余裕」= **中央値が 0 から動くために、線率0 の面が何面減れば足りるか**'
          '(`P2` が 0 の頁だけ)')

    print('\n=== C. 母集団ごと(**4つ全部印字する**。規律9) ===')
    for name, excl in POPS.items():
        sel = [pg for pg in per_page
               if not (excl and any(e in pg for e in excl))]
        if not sel:
            print(f'  {name:<26} 0 頁')
            continue
        nm = [summ[pg]['near_med'] for pg in sel]
        fm = [summ[pg]['far_med'] for pg in sel]
        print(f'  {name:<26} {len(sel)} 頁 / '
              f'近い面の線率の中央(頁ごとの値を並べる) '
              f'{" ".join(f"{x:.2f}" for x in nm)} / '
              f'遠い面 {" ".join(f"{x:.2f}" for x in fm)}')
    print('  **頁ごとの列を必ず並べる**(幅だけで動きを読まない。条例2)')

    print('\n=== D. 判定(**棒ではない。診断である**) ===')
    two = [pg for pg, s in summ.items()
           if s['near'] >= 10 and s['far'] >= 10 and s['near_med'] >= 0.5
           and s['far_med'] == 0.0]
    print(f'  二山(近い面の線率の中央 ≥ 0.5 かつ 遠い面の中央 = 0、両側10面以上): '
          f'**{len(two)} / {len(per_page)} 頁**')
    for pg in two:
        print(f'    {pg}')
    nofar = [pg for pg, s in summ.items() if s['far'] < 10]
    print(f'  遠い面が10面未満の頁: {len(nofar)} —— ' + (' / '.join(nofar) or 'なし'))
    print(f'\n  表: {OUT / "faces_6pages.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
