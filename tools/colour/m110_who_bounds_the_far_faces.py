"""`page0005` の面の縁を**層ごとに**割り当て、「どちらの線からも遠い43面」を誰が囲んでいるか出す。

**なぜ測るのか**(追補171 §C): `T4` の遠い254面のうち 191面は 3D の機械生成線画
(`rendering_lineart`)で囲まれていた。残る **43面は「手描きの線でも 3D の線でもない何か」**
で囲まれている。**原案(塗りの入っていないマスク層だけで面を見つける)に直接効く** ——
その「何か」がマスク層なら、原案の正解源はこの43面を説明できるはずである。

**層の同定は構造で行う**(条例4)。`layer_id` / `parent_id` / `order_in_parent` /
`clip` / `folder` / `layer_type` と、納品の層種別 `v_layer_kind`(census)。
**名前は参考として隣に置くだけで、同定には使わない。**

**門は3つ**(**落ちたら層の割り当ては読まない**):
  IC1 `m106` の既知セル4件(閉じた面 482 / 線率 中央 0.0000 / 線率<0.5 が 344 / 床下 5075)
  IC2 **縁の画素数が 482面すべてで `m106.faces` の `ring` と一致する**
      (縁を数え直しているので、その数え方が同じであることを当てる)
  IC3 **線率0 の面では、縁の画素のうち墨に当たるものが0でなければならない**(恒等)

**出すもの**: 群ごとに「縁の画素を出した層」の表。**1画素が複数の層の縁になり得る**ので、
**延べ数と、その群の縁の画素数に対する割合(100% を超え得る)の両方を印字する**。
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
import m99_b_standard as m99                                         # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402

OUT = Path('results/m110_far_face_owners')
PAGE = 'page0005'
FAR = 16.0
XW = Path('results/m108_t5_crosswalk/crosswalk_6pages.csv')
TOP = 8                     # 表に出す層の数(**残りは「その他」で合計して印字する**)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if PAGE in q['page']][0]

    # ---- 第1巡: 境界と面(**`m106` の既知セルで門を当てる**)
    ink, bnd, floor_px, n_line, n_edge = m106.build(p, byp, kinds)
    rows, under = m106.faces(bnd, ink, floor_px)
    rs = sorted(r['ratio'] for r in rows)
    m = rs[len(rs)//2] if len(rs) % 2 else (rs[len(rs)//2-1]+rs[len(rs)//2])/2
    got = dict(closed=len(rows), med=round(m, 4),
               lt05=sum(1 for r in rows if r['ratio'] < 0.5), under=under)
    ic1 = all(abs(got[k] - v) < 1e-9 for k, v in m106.IC.items())
    print('IC1 ' + ' '.join(f'{k}:{v}/{got[k]}' for k, v in m106.IC.items())
          + ('  PASS' if ic1 else '  **FAIL**'))
    if not ic1:
        print('**門が落ちた。層の割り当ては読まない**')
        return 1

    # ---- 縁の画素の座標を面ごとに取る(**`m106.faces` と同じ式**)
    n, lab, st, _ = cv2.connectedComponentsWithStats((~bnd).astype(np.uint8),
                                                     connectivity=4)
    k3 = np.ones((3, 3), np.uint8)
    idx = {r['face']: i for i, r in enumerate(rows)}
    ys, xs, fid = [], [], []
    bad = 0
    for r in rows:
        y0, y1, x0, x1 = r['y0'], r['y1'], r['x0'], r['x1']
        mm = (lab[y0:y1, x0:x1] == r['face'])
        ring = cv2.dilate(mm.astype(np.uint8), k3).astype(bool) & ~mm \
            & bnd[y0:y1, x0:x1]
        yy, xx = np.nonzero(ring)
        if len(yy) != r['ring']:
            bad += 1
        ys.append(yy + y0)
        xs.append(xx + x0)
        fid.append(np.full(len(yy), idx[r['face']], np.int32))
        del mm, ring
    del lab
    ys = np.concatenate(ys)
    xs = np.concatenate(xs)
    fid = np.concatenate(fid)
    print(f'IC2 縁の画素数が合わない面 {bad} / {len(rows)}'
          + ('  PASS' if bad == 0 else '  **FAIL**'))
    if bad:
        print('**門が落ちた。層の割り当ては読まない**')
        return 1

    # IC3 線率0 の面の縁に墨は無い(恒等)
    on_ink = ink[ys, xs]
    zero = np.array([r['ratio'] == 0.0 for r in rows])[fid]
    ic3 = int((on_ink & zero).sum())
    print(f'IC3 線率0 の面の縁に当たる墨の画素 {ic3}'
          + ('  PASS' if ic3 == 0 else '  **FAIL**'))
    if ic3:
        print('**門が落ちた。層の割り当ては読まない**')
        return 1
    om = m104.open_from(ink)
    print(f'  縁の画素 合計 {len(ys):,} / 線 {n_line} / 縁を出した層 {n_edge}\n')

    # ---- 第2巡: 同じ順序で縁を層ごとに受け取り、面ごとに数える
    tally = {}
    seen = []

    def on_edge(r, e):
        hit = e[ys, xs] & om[ys, xs]
        if not hit.any():
            seen.append((r['layer_id'], 0))
            return
        tally[str(r['layer_id'])] = np.bincount(fid[hit], minlength=len(rows))
        seen.append((r['layer_id'], int(hit.sum())))

    ink2, bnd2, _, _, n_edge2 = m106.build(p, byp, kinds,
                                           extras={'on_edge': on_edge})
    same = bool((bnd2 == bnd).all()) and n_edge2 == n_edge
    print(f'IC4 第2巡の境界が第1巡と同一 {same} / 縁を出した層 {n_edge2} '
          f'(第1巡 {n_edge})' + ('  PASS' if same else '  **FAIL**'))
    del ink2, bnd2, om
    if not same:
        print('**門が落ちた。層の割り当ては読まない**')
        return 1
    print(f'  縁の画素を1つでも出した層 {len(tally)} / '
          f'出さなかった層 {sum(1 for _, c in seen if c == 0)}\n')

    # ---- 群を作る(`m108` の突き合わせから。**定義を変えない**)
    xw = {int(r['face']): r for r in csv.DictReader(open(XW))
          if PAGE in r['page']}
    grp = {'近い(墨まで<1px)': [], '3Dで囲まれた(T5 線率1.0)': [],
           'どちらの線からも遠い(T5 線率0)': [], 'その他': []}
    for i, r in enumerate(rows):
        x = xw.get(r['face'])
        t5 = (float(x['t5_ratio']) if x and x['t5_ratio'] != '' else None)
        if r['d_ink'] < 1.0:
            grp['近い(墨まで<1px)'].append(i)
        elif r['d_ink'] >= FAR and t5 is not None and t5 >= 0.9999:
            grp['3Dで囲まれた(T5 線率1.0)'].append(i)
        elif r['d_ink'] >= FAR and t5 == 0.0:
            grp['どちらの線からも遠い(T5 線率0)'].append(i)
        else:
            grp['その他'].append(i)

    meta = {str(r['layer_id']): r for r in byp[p['source']]}
    out_rows = []
    for gname, ids in grp.items():
        if not ids:
            continue
        sel = np.zeros(len(rows), bool)
        sel[ids] = True
        ringtot = int(sum(rows[i]['ring'] for i in ids))
        per = [(int(v[sel].sum()), lid) for lid, v in tally.items()]
        per = sorted((c, lid) for c, lid in per if c > 0)[::-1]
        print(f'=== {gname} —— {len(ids)} 面 / 縁の画素 {ringtot:,} ===')
        print(f'{"層":>8} {"親":>8} {"位":>4} {"ｸﾘｯﾌﾟ":>6} {"種":>5} '
              f'{"v_layer_kind":<14}{"縁の画素":>10}{"割合":>8}  名前(参考)')
        for c, lid in per[:TOP]:
            r = meta.get(lid, {})
            kind = kinds.get((p['source'], lid), '(表に無い)')
            print(f'{lid:>8} {r.get("parent_id", "?"):>8} '
                  f'{r.get("order_in_parent", "?"):>4} '
                  f'{"1" if r.get("clip") == "1" else "-":>6} '
                  f'{("ﻌ" if r.get("folder") == "True" else r.get("layer_type", "?")):>5} '
                  f'{kind:<14}{c:>10,}{c/ringtot*100:>7.1f}%  {r.get("name", "?")}')
            out_rows.append(dict(group=gname, faces=len(ids), ring_total=ringtot,
                                 layer_id=lid, parent_id=r.get('parent_id', ''),
                                 order_in_parent=r.get('order_in_parent', ''),
                                 clip=r.get('clip', ''), folder=r.get('folder', ''),
                                 layer_type=r.get('layer_type', ''),
                                 v_layer_kind=kind, name=r.get('name', ''),
                                 ring_px=c, ring_frac=c/ringtot))
        rest = sum(c for c, _ in per[TOP:])
        print(f'{"その他":>8} {len(per[TOP:]):>8} 層' + ' ' * 31
              + f'{rest:>10,}{rest/ringtot*100:>7.1f}%')
        print(f'  延べ {sum(c for c, _ in per):,} / 縁 {ringtot:,} = '
              f'{sum(c for c, _ in per)/ringtot*100:.1f}%'
              f'(**1画素が複数の層の縁になり得るので 100% を超える**)')
        byk = collections.Counter()
        for c, lid in per:
            byk[kinds.get((p['source'], lid), '(表に無い)')] += c
        print('  層種別ごと: ' + ' / '.join(f'{k} {v:,}({v/ringtot*100:.1f}%)'
                                            for k, v in byk.most_common()) + '\n')

    with open(OUT / 'owners.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f'表: {OUT / "owners.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
