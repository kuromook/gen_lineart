"""追補232・233: W(τ = 6.0)の標本外の確認。8頁、土台 B / B+P / B+P+T、何も選び直さない。

正解は追補233 §A の規則(親をまたぐ重なり順 / 祖先のフォルダーのマスク / トーンを外す)。
門: G-旧(2頁で旧器具と画素ごとに一致)/ G-向き / G-材料 / G-線、続けて W1′・W2〜W5。
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m77_standard_visible as m77                                    # noqa: E402
import m99_b_standard as m99                                          # noqa: E402
import m102_b_standard_v2 as m102                                     # noqa: E402
import m119_mask_faces_and_lines as m119                              # noqa: E402
import m122_line_only_faces as m122                                   # noqa: E402
import m126_line_ends as m126                                         # noqa: E402
import m130_t_junctions as m130                                       # noqa: E402
import m136_thin_line_merge as m136                                   # noqa: E402

TAU = 6.0
m136.TAU = TAU
D = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/holdout8_20261107')
ROWS = Path('inbox/m_holdout8_page_layers_20261107.csv')
CEN = Path('inbox/mask_material_pages_20261107.csv')
DRAWN = Path('results/m138_holdout_pages/drawn_pages.csv')
OUT = Path('results/m139_w_holdout8')
ALPHA = 26
ok = m130.ok


def tr(v):
    return str(v) in ('True', 'true', '1')


def chain(lid, idx):
    """根から lid までの祖先(lid を含まない)の layer_id の列。"""
    out, seen = [], set()
    p = idx[lid]['parent_id']
    while p and p in idx and p not in seen:
        seen.add(p)
        out.append(p)
        p = idx[p]['parent_id']
    return out[::-1]


def zkey(lid, idx):
    return tuple(int(idx[a]['order_in_parent'] or 0) for a in chain(lid, idx) + [lid])


def folder_vis(pg, idx):
    """祖先に効くフォルダーのマスク {layer_id(str): 見える側}。"""
    H, W = pg['H'], pg['W']
    out = {}
    for r in pg['mm']:
        if r.get('folder') is True or tr(r.get('folder')):
            out[str(r['layer_id'])] = m119.vis_side(pg['mdir'] / r['mask_file'], H, W)
    return out


def anc_mask(lid, idx, fv, H, W):
    m = None
    for a in chain(lid, idx):
        if a in fv:
            m = fv[a].copy() if m is None else (m & fv[a])
    return m


def build_truth2(pg, kinds, drop_tone=True, log=print):
    H, W, floor_px = pg['H'], pg['W'], pg['floor_px']
    idx = {str(r['layer_id']): r for r in pg['rows']}
    mat, other, drop = m119.classify(pg['rows'], pg['mm'], H * W, kinds, pg['pp'])
    n_mat_all = len(mat)
    faint = sum(1 for r, _ in other if int(r['mask_px_ge128']) == 0 and int(r.get('mask_px_1_127') or 0) > 0)
    tone = [x for x in mat if tr(x[1].get('tone_enabled'))]
    if drop_tone:
        mat = [x for x in mat if not tr(x[1].get('tone_enabled'))]
    fv = folder_vis(pg, idx)
    n_clip = sum(int(row['clip'] or 0) != 0 for _, row in mat)
    log(f'  材料 {n_mat_all}(落とした: ' + ' / '.join(f'{k} {v}' for k, v in sorted(drop.items())) +
        f')→ トーンを外す {len(tone) if drop_tone else 0} → {len(mat)} / clip の材料 {n_clip} / フォルダーのマスク {len(fv)}')
    gdir = True
    info = {str(r['layer_id']): row for r, row in mat}
    files = {str(r['layer_id']): r['mask_file'] for r, _ in mat}
    want = {str(r['layer_id']): int(r['mask_px_ge128']) for r, _ in mat}
    order = sorted(info, key=lambda l: zkey(l, idx))
    Fl = np.zeros((H, W), np.int32)
    faces = []
    above = np.zeros((H, W), bool)
    n_anc = 0
    for l in reversed(order):
        v = m119.vis_side(pg['mdir'] / files[l], H, W)
        gdir &= int(v.sum()) == want[l]
        am = anc_mask(l, idx, fv, H, W)
        if am is not None:
            v &= am
            n_anc += 1
        fm = v & ~above
        above |= v
        lb, st, big, _ = m119.components(fm, 8, floor_px)
        for i in big:
            fid = len(faces) + 1
            Fl[lb == i] = fid
            faces.append(dict(fid=fid, layer=int(l), name=info[l]['name']))
        del lb, v, fm
    log(f'  G-向き 見える側の画素 = mask_px_ge128(材料 {len(order)} 層){"" if gdir else ""}  {ok(gdir)} / 祖先のフォルダーのマスクを掛けた材料 {n_anc}')
    return Fl, faces, dict(faint=faint, n_mat_all=n_mat_all, n_tone=len(tone), gdir=gdir, fv=fv, idx=idx)


def old_pages_gate(kinds):
    good = True
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        F0, _, _, _ = m122.build_truth(pg, kinds)
        F1, _, _ = build_truth2(pg, kinds, drop_tone=False, log=lambda *a: None)
        same = bool((F0 == F1).all())
        print(f'G-旧 {pg["key"]}: 新しい規則の正解 = 旧器具の正解(画素ごと、面 {int(F0.max())})  {ok(same)}')
        good &= same
        del pg, F0, F1
    return good


def load_holdout(no, page_path, kinds):
    rows = [r for r in csv.DictReader(open(ROWS, encoding='utf-8-sig')) if r['page_path'] == page_path]
    cm = next(p for p in json.load(open(D / 'colour_layers_manifest_holdout8.json', encoding='utf-8'))['pages']
              if p['source'] == page_path)
    mm = [r for r in json.load(open(D / 'layer_masks_manifest_holdout8.json', encoding='utf-8'))['layers']
          if r['page_path'] == page_path]
    W, H = cm['canvas']
    files = {str(l['layer_id']): l['file'] for l in cm['layers'] if l.get('file')}
    idx = {str(r['layer_id']): r for r in rows}
    pg = dict(key=f'h{no}', pp=page_path, rows=rows, mm=mm, mdir=D, H=H, W=W,
              floor_px=max(1, int(round(H * W * m119.FLOOR))))
    fv = folder_vis(pg, idx)
    vis, _ = m77.visible_rows(rows)
    ink = np.zeros((H, W), bool)
    n_want, n_read, no_file, n_own_mask = 0, 0, [], 0
    for r in vis:
        if not m99.is_line(r):
            continue
        n_want += 1
        n_own_mask += tr(r['has_mask'])
        f = files.get(str(r['layer_id']))
        if f is None:
            no_file.append(r['layer_id'])
            continue
        im = cv2.imread(str(D / f), cv2.IMREAD_UNCHANGED)
        a = np.zeros((H, W), bool)
        aa = im[..., 3] >= ALPHA
        a[:aa.shape[0], :aa.shape[1]] = aa[:H, :W]
        am = anc_mask(str(r['layer_id']), idx, fv, H, W)
        if am is not None:
            a &= am
        ink |= a
        n_read += 1
        del im, a, aa
    skipped = {str(s['layer_id']): s['reason'] for s in cm.get('skipped', [])}
    why = {}
    for l in no_file:
        k = skipped.get(str(l), '理由なし')
        why[k] = why.get(k, 0) + 1
    g = n_read + len(no_file) == n_want and n_read > 0 and all(k != '理由なし' for k in why)
    print(f'G-線 h{no}: 可視の `線画` の葉 {n_want} = 読んだ {n_read} + ラスタ無し {len(no_file)}'
          f'({" / ".join(f"{k} {v}" for k, v in why.items()) or "—"})墨 {int(ink.sum())} px / 自身のマスクを持つ線の層 {n_own_mask}(掛けない)  {ok(g)}')
    pg['ink'] = ink
    return pg, g


def evaluate(lab, Fl, U):
    return m136.evaluate(lab, Fl, U)


def run_page(pg, Fl, faces, cen):
    key, floor_px = pg['key'], pg['floor_px']
    U = Fl > 0
    fails = 0
    r0 = m122.score(Fl, Fl, np.arange(1, len(faces) + 1))
    w2 = all(abs(r0[t]['F1'] - 1) < 1e-12 for t in m122.THR)
    print(f'W2 神託 F1 = 1(面 {len(faces)})  {ok(w2)}')
    fails += not w2
    if len(faces) == 0:
        print('  **正解の面が 0。この頁は読めない(判定保留)**')
        return None
    inkB, inkP = m136.bases(pg)
    _, pts, dirs, _, free = m130.build_b(pg['ink'])
    segsT, _ = m130.extend(inkB, pts, dirs, free, m130.D_MAIN)
    inkPT = m130.draw_segs(inkP, segsT)
    w1 = bool((~pg['ink'] | inkB).all() and (~inkB | inkP).all() and (~inkP | inkPT).all())
    print(f"W1′ 線 ⊆ B ⊆ B+P ⊆ B+P+T(T の延長 {len(segsT)} 本)  {ok(w1)}")
    fails += not w1
    inkM = np.ascontiguousarray(pg['ink'][:, ::-1])
    pgM = dict(pg, ink=inkM)
    iBM, iPM = m136.bases(pgM)
    _, pM, dM, _, fM = m130.build_b(inkM)
    sM, _ = m130.extend(iBM, pM, dM, fM, m130.D_MAIN)
    iPTM = m130.draw_segs(iPM, sM)
    out = {}
    for name, ink_, inkM_ in (('B', inkB, iBM), ('B+P', inkP, iPM), ('B+P+T', inkPT, iPTM)):
        lab, _, _ = m122.regions(ink_, m136.K, floor_px)
        f0, lk0, h0, n0 = evaluate(lab, Fl, U)
        sw = m136.seam_widths(ink_, lab)
        bord = m136.border_faces(lab)
        cand = m136.candidates(sw, bord, m136.MIN_PX)
        mp = m136.w_pairs(sw, cand, TAU)
        labW = m136.merge(lab, mp)
        w4 = m136.nfaces(labW) <= m136.nfaces(lab)
        fails += not w4
        fW, lkW, hW, nW = evaluate(labW, Fl, U)
        nf, w5 = [], True
        rng = np.random.default_rng(0)
        for s in range(m136.N_RAND):
            pick = [cand[i] for i in rng.permutation(len(cand))[:len(mp)]]
            w5 &= len(pick) == len(mp)
            nf.append(evaluate(m136.merge(lab, pick), Fl, U)[0])
        fails += not w5
        labM, _, _ = m122.regions(inkM_, m136.K, floor_px)
        swM = m136.seam_widths(inkM_, labM)
        f1M = evaluate(m136.merge(labM, m136.w_pairs(swM, m136.candidates(swM, m136.border_faces(labM), m136.MIN_PX), TAU)), Fl, U)[0]
        del labM
        auc, ns, nd = describe_auc(lab, sw, bord, Fl, U)
        print(f'--- {name}: 候補 {len(cand)} → 結合 {len(mp)}  W4 {ok(w4)} / W5 {ok(w5)}\n'
              f'  F1 {f0:.3f} → +W {fW:.3f}  当たり {h0} → {hW}  提案 {n0} → {nW}  漏れ(U) {lk0:.3f} → {lkW:.3f}\n'
              f'  NW 最大 {max(nf):.3f} / 中央値 {np.median(nf):.3f}  N1 {f1M:.3f}  AUC {auc}(同じ面 {ns} / 違う面 {nd})')
        out[name] = dict(f0=f0, fW=fW, nw=max(nf), nwm=float(np.median(nf)), n1=f1M, h0=h0, hW=hW, n0=n0, nW=nW,
                         cand=len(cand), merged=len(mp), auc=auc)
        del lab, labW
    if fails:
        print(f'**門が {fails} 本落ちた。この頁の表を読まない**')
        return None
    return out


def describe_auc(lab, sw, bord, Fl, U):
    keep, _ = m122.in_domain(lab, U)
    n = int(lab.max()) + 1
    inside = np.zeros(n, bool)
    inside[keep] = True
    maj = np.zeros(n, int)
    for p in keep:
        t, c = np.unique(Fl[lab == p], return_counts=True)
        c, t = c[t > 0], t[t > 0]
        maj[p] = t[np.argmax(c)]
    s, d = [], []
    for (a, b), v in sw.items():
        if len(v) < m136.MIN_PX or not (inside[a] and inside[b]) or bord[a] or bord[b]:
            continue
        (s if maj[a] == maj[b] else d).append(float(np.median(v)))
    s, d = np.array(s), np.array(d)
    if not len(s) or not len(d):
        return 'なし', len(s), len(d)
    auc = float(((s[:, None] < d[None]).sum() + 0.5 * (s[:, None] == d[None]).sum()) / (len(s) * len(d)))
    return f'{auc:.2f}', len(s), len(d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    good, res = m136.unit_test()
    print(f'W3 単体試験(τ = {m136.TAU:g}){res}  {ok(good)}')
    kinds = m102.load_kinds()
    if not (good and old_pages_gate(kinds)):
        print('**門が落ちた。走らない**')
        return
    cen = {r['page_path']: r for r in csv.DictReader(open(CEN, encoding='utf-8-sig'))}
    drawn = list(csv.DictReader(open(DRAWN, encoding='utf-8')))
    allr = {}
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    for d in drawn:
        if only and d['no'] not in only:
            continue
        no, pp = d['no'], d['page_path']
        print(f'\n======== h{no}  {pp}')
        pg, g = load_holdout(no, pp, kinds)
        Fl, faces, inf = build_truth2(pg, kinds)
        # 差の内訳(追補234): 値が 1〜127 だけのマスクは census では材料、M では全面隠す(見える側 0 画素)
        gm = inf['n_mat_all'] + inf['faint'] == int(cen[pp]['n_material'])
        print(f'  G-材料 トーンを外す前の材料 {inf["n_mat_all"]} + 値が 1〜127 だけのマスク {inf["faint"]} = census {cen[pp]["n_material"]}  {ok(gm)}  床 {pg["floor_px"]} px / 面 {len(faces)}')
        if not (g and gm and inf['gdir']):
            print('  **頁の門が落ちた。この頁の表を読まない**')
            allr[no] = None
            continue
        r = run_page(pg, Fl, faces, cen)
        allr[no] = r
        if r:
            r['tone'] = int(cen[pp]['n_material_tone'])
        json.dump(r, open(OUT / f'h{no}.json', 'w'), ensure_ascii=False)
        del pg, Fl
    print('\n=== 棒(追補232 §E、τ = 6.0、土台ごと、最悪の頁で読む)===')
    good_pages = {k: v for k, v in allr.items() if v}
    print(f'読めた頁 {len(good_pages)} / 8(読めない: {[k for k, v in allr.items() if not v]})')
    for b in ('B', 'B+P', 'B+P+T'):
        n1 = n2 = 0
        for k, r in good_pages.items():
            x = r[b]
            wb1, wb2 = x['fW'] > x['f0'], x['fW'] > max(x['nw'], x['n1'])
            n1 += wb1
            n2 += wb2
            print(f'  {b:<6} h{k}(トーン {r["tone"]:>2}) F1 {x["f0"]:.3f} → {x["fW"]:.3f} 当たり {x["h0"]}→{x["hW"]} 結合 {x["merged"]:>3}/{x["cand"]:<4}'
                  f' NW 最大 {x["nw"]:.3f} N1 {x["n1"]:.3f} AUC {x["auc"]}  WB1 {"はい" if wb1 else "いいえ"} / WB2 {"はい" if wb2 else "いいえ"}')
        a = lambda n: '通る' if n == len(good_pages) and n else '落ちる'  # noqa: E731
        print(f'  **{b}: WB1-標本外 {a(n1)}({n1}/{len(good_pages)}頁)/ WB2-標本外 {a(n2)}({n2}/{len(good_pages)}頁)**')


if __name__ == '__main__':
    main()
