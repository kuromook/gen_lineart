# -*- coding: utf-8 -*-
"""L18 本体: ベタに接した語は、墨を伝って線の束ごと動くか。

登録は `doc/work_log.md § 2026-10-25 L18 事前登録` (1fb1128)。器具と既知セルは
`l18_known_cells.py`(KC21〜KC24 が PASS していなければ回さない)。
**棒の定数は基盤が決める。** この走りが出すのは縮退・帰無値・引数だけ。

  --arms degen     縮退の点検だけ (成分の単位。棒の前に出す)

**棒は腕と同じ走行の中で計算する**(2026-10-25 の新しい規律)。帰無値の関数である床は
走行をまたいで凍結しない —— **凍結した引きの運が棒に入る**。凍結してよいのは
絶対の定数(棒A 0.30 / 棒C 0.50)だけ。
  --arms control   対照(帰無値)だけ      --arms real  本来の腕   --arms both
"""
import argparse, re, csv, os, statistics as st, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l15_known_cells import boundary
from l17_known_cells import (D1, assign_top, gated_layers, jobs, panels_and_regions,
                             place, SEEDS, TRIES, zmap)
from l18_known_cells import (GS, comp_pur, geo_spread, ink_components, prop_full,
                             seed_of)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l18_propagate'
EXCLUDE = None       # 事前登録の母集団定義 (--exclude-gapfill)
INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
ARMS = 'both'


def n_lay90_of(C, lab):
    v = lab[C]; v = v[v > 0]
    if v.size == 0:
        return 0, 0
    vals, cnt = np.unique(v, return_counts=True)
    s = np.sort(cnt)[::-1]
    return int(np.searchsorted(np.cumsum(s), 0.9 * s.sum()) + 1), int(len(vals))


def gapfill_crops():
    """作者が「隙間埋め」と答えた面の L19 切り出し名。**事前登録の母集団定義**。

    名前は「判定のある 551 面の中で見つかった隙間埋けだけを除いた母集団」。
    **残り 635 面の隙間埋けは未知**(基盤 `ruling_l18_authorised...` §2-2)。
    """
    import json as _j
    q = _j.load(open('/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'))
    crop_of = {r['order']: r['crop'] for r in q['queue']}
    out, lab = set(), {}
    for ln in open('/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'):
        r = _j.loads(ln)
        lab[crop_of[r['order']]] = r
    for c, r in lab.items():
        if str(r['choice']).startswith('gapfill'):
            out.add(c)
    return out


def one_page(job):
    comps, regs, pans = [], [], []
    k = 0                 # **L19 と同じ頁内の通し番号**。面の鍵は一意でないので名前で照合
    names_all = []
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        lab, keys = assign_top(sub, P.shape)
        ink, cl, st_, n = ink_components(lab, P)
        tot = int(ink.sum())
        if tot == 0:
            continue
        # ---- 面の名前を、**腕に依らず本来の面の形から**振る (L19 と同じ条件・同じ順)
        cname, drop_gf = {}, 0
        for gi in gids:
            _s, _L, _adj = seed_of(glab == gi, lab, P)
            if _adj == 0:
                continue
            k += 1
            nm = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
            names_all.append(nm)
            if EXCLUDE and nm in EXCLUDE:
                drop_gf += 1
                continue          # **両腕から同じ面を落とす**(L17 §5 と同じ規律)
            cname[gi] = nm
        # ---- 縮退 (成分の単位。登録 §3)
        # 2026-10-26: ここは `pans[-1]['drop_gapfill'] = drop_gf` だった ——
        # **append の前に [-1] を書いていた**ので、最初のコマで IndexError、
        # 落ちなければ**1つ前のコマに数を付けていた**。
        # `--exclude-gapfill` は一度も走ったことがなかった(T102 の族: 走らない経路)。
        areas = sorted((int(st_[i, A]) for i in range(1, n)), reverse=True)
        pans.append(dict(pid=pid, work=work, comp=ci, ink=tot, n_comp=n - 1,
                         top_share=round(areas[0] / tot, 4) if areas else 0,
                         med_comp=areas[len(areas) // 2] if areas else 0,
                         drop_gapfill=drop_gf))
        if ARMS == 'degen':
            for i in range(1, n):
                C = cl == i
                nl, nraw = n_lay90_of(C, lab)
                comps.append(dict(pid=pid, work=work, comp=ci, cid=i,
                                  area=int(st_[i, A]), share=round(st_[i, A] / tot, 5),
                                  n_lay90=nl, n_lay=nraw, arm='degen'))
            continue
        # ---- 腕
        for arm in (('real',) if ARMS in ('both', 'real') else ()) + \
                   (('shuffled',) if ARMS in ('both', 'control') else ()):
            seeds = SEEDS if arm == 'shuffled' else (0,)
            for sd in seeds:
                acc = np.zeros(P.shape, bool)
                seedlab = np.zeros(P.shape, np.uint16)
                owner = {}
                for gi in gids:
                    if EXCLUDE is not None and gi not in cname:
                        continue
                    if arm == 'real':
                        R = glab == gi
                        if place(glab, gi, gstat, P, SEEDS[0], pid, ci) is None:
                            continue          # 両腕で同じ面集合 (L17 §5)
                    else:
                        R = place(glab, gi, gstat, P, sd, pid, ci)
                        if R is None:
                            continue
                    s, L, adj = seed_of(R, lab, P)
                    if adj == 0 or L == 0:
                        continue
                    C, ids = prop_full(s, cl)
                    acc |= C
                    seedlab[s & (seedlab == 0)] = gi
                    for x in ids:
                        owner.setdefault(int(x), []).append((gi, L, adj))
                    regs.append(dict(pid=pid, work=work, comp=ci, region=gi, arm=arm,
                                     seed=sd, adj=adj, prop=int(C.sum()),
                                     n_comp=int(len(ids))))
                # 成分ごとの正直さ -> cov_honest (基盤 band_form_l18 §1)
                honest = 0; conf_px = 0; lab_of = {}
                for cid, lst in owner.items():
                    C = cl == cid
                    lst.sort(key=lambda t: -t[2])
                    L = lst[0][1]
                    lab_of[cid] = L
                    ar = int(st_[cid, A])
                    cf = int(len({t[1] for t in lst}) > 1)
                    pu = comp_pur(C, lab, L)
                    nl, nraw = n_lay90_of(C, lab)
                    # **競合した成分は cov1 に入れ cov_honest には入れない**
                    if cf:
                        conf_px += ar
                    elif pu >= 0.90:
                        honest += ar
                    comps.append(dict(pid=pid, work=work, comp=ci, cid=cid, arm=arm,
                                      seed=sd, area=ar, share=round(ar / tot, 5),
                                      pur=round(pu, 4), n_lay90=nl, n_lay=nraw,
                                      conflict=cf, n_owner=len(lst)))
                rec = dict(pid=pid, work=work, comp=ci, arm=arm, seed=sd, ink=tot,
                           cov=int(acc.sum()), honest=honest, conflict_px=conf_px,
                           n_comp_owned=len(owner),
                           top_share=round(max((int(st_[i, A]) for i in range(1, n)),
                                               default=0) / tot, 4))
                # 感度: 測地距離の打ち切り (判定に使わない。単位は成分のまま)
                if seedlab.any():
                    sn = geo_spread(seedlab, ink, GS)
                    for g in GS:
                        reach = sn[g] > 0
                        rec[f'cov{g}'] = int(reach.sum())
                        h = 0
                        for cid, L in lab_of.items():
                            Cg = (cl == cid) & reach
                            kg = int(Cg.sum())   # **k は頁の通し番号。潰さない**
                            if kg == 0 or len({t[1] for t in owner[cid]}) > 1:
                                continue
                            if int((Cg & (lab == L)).sum()) / kg >= 0.90:
                                h += kg
                        rec[f'honest{g}'] = h
                else:
                    for g in GS:
                        rec[f'cov{g}'] = 0; rec[f'honest{g}'] = 0
                pans.append(rec)
    return comps, regs, pans, names_all


def med(v):
    return st.median(v) if v else float('nan')


FULL_LOG = '/home/sh1/deepl/lineart-fill-region/results/l18_propagate/run_full.log'


def parse_full_abc():
    """`run_full.log` から全面の版の 腕・対照・r0・床 を読む。

    行の形: `B 床 = 対照 0.0876 + 3r0 0.0765 = **0.1641**  本来 0.0953 ...`
    (印字は `3r0` なので **3で割って r0 に戻す**)。
    """
    txt = open(FULL_LOG, encoding='utf-8').read()
    out, work = {}, None
    for ln in txt[txt.index('A / B / C ('):].splitlines():
        t = ln.strip()
        for w in ('ako6', 'ako7'):
            if t.startswith(w + ':'):
                work = w
        m = re.search(r'B 床 = 対照 ([\d.]+) \+ 3r0 ([\d.]+) = \*\*([\d.]+)\*\*'
                      r'\s+本来 ([\d.]+)', t)
        if m and work:
            ch0, three_r0, floor, ch1 = (float(x) for x in m.groups())
            out[work] = dict(arm=ch1, ctl=ch0, r0=three_r0 / 3.0, floor=floor)
    return out


def print_both_populations(res):
    """**2つの版を、それぞれ自分の対照と自分の床と一緒に並べる**(基盤 2026-10-26)。

    **除外版は第1の走行の部分集合解析ではなく、小さい母集団の上の第2の測定である。**
    面を外すと語が1つ消え、どの成分が語を受けるかと純度が**上流で**変わる。
    だから**対照も床も版ごとに違う** —— 床はその走行の中の対照の関数だからで(T97)、
    全面版の床1つに腕2つを当てて読むと、今夜3回出た形をもう一度踏む。
    """
    full = parse_full_abc()
    print('\n' + '=' * 92)
    print('**2つの版。母集団が違うので、対照も床も別である**')
    print('  **除外版は全面版の部分集合解析ではない** —— 面を外すと語が1つ消え、'
          'どの成分が語を受けるかと純度が上流で変わる。')
    print('  **小さい母集団の上の、第2の測定である。**')
    print('  全面版の数は **前の走行の記録**(同じ器具 crc32・同じ seed・KC30 PASS)。'
          'この走行で計算し直したものではない')
    print()
    print(f'  {"版":<28}{"プール":<7}{"腕":>9}{"対照":>9}{"r0":>9}'
          f'{"床=対照+3r0":>13}  判定')
    for w in ('ako6', 'ako7'):
        f = full.get(w)
        if f:
            # **`r0` に † を付ける**: 前の走行の印字は `3r0` なので、これは
            # 3で割って得た**導出値**である。この走行の `r0` は生の値で出す
            print(f'  {"全面 (前の走行の記録)":<24}{w:<7}{f["arm"]:>9.4f}{f["ctl"]:>9.4f}'
                  f'{f["r0"]:>8.4f}†{f["floor"]:>13.4f}  '
                  f'{"床の上" if f["arm"] >= f["floor"] else "床の下"}')
        a1 = res.get(('real', w, 'ch')); a0 = res.get(('shuffled', w, 'ch'))
        if a1 and a0:
            a1, a0 = a1[0], a0[0]
            fl = a0['ch'] + 3 * a0['r0']
            print(f'  {"隙間埋けを外した版 (この走行)":<21}{w:<7}{a1["ch"]:>9.4f}'
                  f'{a0["ch"]:>9.4f}{a0["r0"]:>9.4f}{fl:>13.4f}  '
                  f'{"床の上" if a1["ch"] >= fl else "床の下"}')
        print()
    print('  † 前の走行の印字は `3r0` なので、この `r0` は3で割って得た**導出値**。'
          'この走行の `r0` は生の値')


def main():
    global ARMS, EXCLUDE
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both',
                    choices=('both', 'real', 'control', 'degen'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--exclude-gapfill', action='store_true',
                    help='**事前登録**: 判定のある551面の中で見つかった隙間埋けだけを外す'
                         ' (残り635面は未知)')
    a = ap.parse_args()
    if a.exclude_gapfill:
        # **起動の門**: 全面の版の記録が読めなければ、2時間走ってから気付かない
        if not os.path.exists(FULL_LOG) or 'A / B / C (' not in open(
                FULL_LOG, encoding='utf-8').read():
            sys.exit(f'止める: 全面の版の記録 {FULL_LOG} が読めない。'
                     ' 外した版だけを出すことは許可されていない(基盤 2026-10-26)')
        EXCLUDE = gapfill_crops()
        print(f'母集団: **判定のある551面の中で見つかった隙間埋け {len(EXCLUDE)} 面だけを'
              f'除いた版**。**残り 635 面の隙間埋けは未知**', flush=True)
    else:
        print('母集団: **全面の版**(登録された一次)', flush=True)
    ARMS = a.arms
    os.makedirs(OUT, exist_ok=True)
    zz = zmap(); fill, line = gated_layers()
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'腕 {ARMS}  頁 {len(js)}  (既知セル KC21-24 は別プロセスで PASS 済みであること)')
    comps, regs, pans, names = [], [], [], []
    with Pool(4) as pool:
        for i, (c, r, p, nm) in enumerate(pool.imap_unordered(one_page, js), 1):
            comps += c; regs += r; pans += p; names += nm
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  成分 {len(comps)}', flush=True)
    # ---- KC30 データ経路: 面の通し番号が L19 と同じか (**除外の鍵が正しいことの点検**)
    if not a.limit:
        import json as _j
        q = _j.load(open('/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'))
        want = {r['crop'] for r in q['queue']}
        got = set(names)
        ok = want <= got
        print(f'KC30 データ経路: L19 の 551 面の名前が今回の名前に全部あるか: '
              f'{"ok" if ok else "FAIL"}  (面 {len(got)} / 欠け {len(want - got)})')
        if not ok:
            print('**除外の鍵が合っていない。止める**')
            return 1
        if EXCLUDE:
            print(f'  選別(規律9-1): 隙間埋けとして外した面 {len(EXCLUDE & got)} / '
                  f'指定 {len(EXCLUDE)}')
    for nm, rows in (('comps', comps), ('regions', regs), ('panels', pans)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(f'{OUT}/{nm}_{ARMS}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'-> {OUT}/{nm}_{ARMS}.csv  行 {len(rows)}')

    if ARMS == 'degen':
        print('\n' + '=' * 92)
        print('縮退の点検 (登録 §3。**棒の前に出す**。単位は成分)')
        for w in ('ako6', 'ako7'):
            pp = [r for r in pans if r['work'] == w]
            cc = [r for r in comps if r['work'] == w]
            if not cc:
                continue
            print(f'  {w}: コマ {len(pp)}  成分 {len(cc)}')
            print(f'    最大成分がコマの墨に占める割合: 中央 '
                  f'{med([r["top_share"] for r in pp]):.4f}  '
                  f'90分位 {sorted(r["top_share"] for r in pp)[int(.9*len(pp))]:.4f}  '
                  f'>0.5 のコマ {sum(1 for r in pp if r["top_share"]>0.5)/len(pp):.3f}')
            print(f'    1コマの成分の数: 中央 {med([r["n_comp"] for r in pp]):.0f}')
            ar = sorted(r['area'] for r in cc)
            print(f'    成分の画素: 中央 {med(ar):.0f}  90分位 {ar[int(.9*len(ar))]}  '
                  f'最大 {ar[-1]}')
            for lab2, lo, hi in (('全成分', 0, 10**12), ('>=1000px', 1000, 10**12),
                                 ('>=10000px', 10000, 10**12)):
                s = [r for r in cc if lo <= r['area'] < hi]
                if not s:
                    continue
                nl = [r['n_lay90'] for r in s]
                print(f'    {lab2:10} {len(s):6d}  1成分が跨ぐ層 n_lay90 中央 '
                      f'{med(nl):.1f}  <=1 {sum(1 for x in nl if x<=1)/len(nl):.3f}  '
                      f'>=3 {sum(1 for x in nl if x>=3)/len(nl):.3f}  '
                      f'生の層数 中央 {med([r["n_lay"] for r in s]):.1f}')
            # **門は画素重み中央** (基盤 band_form_l18 §4。成分を数えると巨大成分1個が映らない)
            tw = sum(r['area'] for r in cc)
            ss = sorted(cc, key=lambda r: r['n_lay90'])
            acc2 = 0; pwm = ss[-1]['n_lay90']
            for r in ss:
                acc2 += r['area']
                if acc2 >= tw / 2:
                    pwm = r['n_lay90']; break
            print(f'    **門: 成分が跨ぐ層数の画素重み中央 {pwm}** '
                  f'(>=3 なら「縮退のため測定にならない」で閉じる)  '
                  f'成分数での中央 {med([r["n_lay90"] for r in cc]):.1f}')
            for k in (1, 2, 3):
                print(f'    墨で重み付け: n_lay90 <= {k} の成分が墨に占める割合 '
                      f'{sum(r["area"] for r in cc if r["n_lay90"]<=k)/tw:.4f}')
        return

    print('\n' + '=' * 92)
    print('基盤 band_form_l18_20261025.md の引数 (一次は **画素重み**。分母は P36 と同じ)')
    res = {}
    for arm in ('real', 'shuffled'):
        pp = [r for r in pans if r.get('arm') == arm]
        if not pp:
            continue
        print(f'\n{"本来の腕" if arm=="real" else "対照 = 帰無値"} ({arm})')
        for w in ('ako6', 'ako7'):
            s0 = [r for r in pp if r['work'] == w]
            if not s0:
                continue
            per = []
            for sd in sorted({r['seed'] for r in s0}):
                t = [r for r in s0 if r['seed'] == sd]
                ink = sum(r['ink'] for r in t)
                c1 = sum(r['cov'] for r in t) / ink
                ch = sum(r['honest'] for r in t) / ink
                per.append(dict(seed=sd, cov1=c1, ch=ch,
                                rate=ch / c1 if c1 else float('nan'),
                                cf=sum(r['conflict_px'] for r in t) / ink))
                line_ = (f'  {w} seed {sd}: cov1 {c1:.4f}  **cov_honest {ch:.4f}**  '
                         f'honest_rate {per[-1]["rate"]:.4f}  競合 {per[-1]["cf"]:.4f}')
                print(line_)
            for k in ('cov1', 'ch', 'rate'):
                v = [q[k] for q in per]
                sd_ = st.stdev(v) if len(v) > 1 else 0.0
                res[(arm, w, k)] = (med(v), max(v) - min(v), sd_)
            if len(per) > 1:
                print(f'    seed間: cov_honest 中央 {res[(arm,w,"ch")][0]:.4f}  '
                      f'**r0 = 最大−最小 {res[(arm,w,"ch")][1]:.4f}**  '
                      f'SD {res[(arm,w,"ch")][2]:.4f}  '
                      f'honest_rate 中央 {res[(arm,w,"rate")][0]:.4f}')
                print('      (**最大−最小は seed 本数に単調**なので、本数の違う走行の'
                      ' r0 は別の量。SD は併記、主値は登録どおり r0)')
            # コマの2階級 (基盤 §4。**判定はプール×階級の4セル**)
            for lab2, lo, hi in (('最大成分<0.5', 0, 0.5), ('最大成分>=0.5', 0.5, 1.01)):
                t = [r for r in s0 if lo <= r['top_share'] < hi]
                if not t:
                    continue
                # **階級別 r0** = その階級の中での seed 最大−最小 (対照から出る)
                per2 = []
                for sd in sorted({r['seed'] for r in t}):
                    u = [r for r in t if r['seed'] == sd]
                    ik = sum(r['ink'] for r in u)
                    per2.append((sum(r['cov'] for r in u) / ik,
                                 sum(r['honest'] for r in u) / ik))
                chs = [q[1] for q in per2]
                r0c = (max(chs) - min(chs)) if len(chs) > 1 else 0.0
                ink = sum(r['ink'] for r in t)
                c1 = sum(r['cov'] for r in t) / ink
                ch = sum(r['honest'] for r in t) / ink
                rt = ch / c1 if c1 else float('nan')
                res[(arm, w, lab2)] = dict(cov1=c1, ch=ch, rate=rt, r0=r0c,
                                           ink_share=ink / sum(r['ink'] for r in s0))
                print(f'    [{lab2:12}] コマ行 {len(t):5d}  '
                      f'墨 {res[(arm,w,lab2)]["ink_share"]:.3f}  '
                      f'cov1 {c1:.4f}  cov_honest {ch:.4f}  honest_rate {rt:.4f}'
                      + (f'  **階級別 r0 {r0c:.4f}**' if len(chs) > 1 else ''))
            # 測地距離の格子 (開示。判定に使わない)
            ink = sum(r['ink'] for r in s0)
            print('    感度 g (判定に使わない。g=∞ が一次)')
            for g in list(GS) + ['inf']:
                if g == 'inf':
                    c1 = sum(r['cov'] for r in s0) / ink
                    ch = sum(r['honest'] for r in s0) / ink
                else:
                    c1 = sum(r[f'cov{g}'] for r in s0) / ink
                    ch = sum(r[f'honest{g}'] for r in s0) / ink
                print(f'      g={str(g):>4}  cov1 {c1:.4f}  cov_honest {ch:.4f}  '
                      f'honest_rate {ch/c1 if c1 else float("nan"):.4f}')
            cc = [r for r in comps if r['work'] == w and r['arm'] == arm]
            if cc:
                pu = [r['pur'] for r in cc]
                print(f'    (併記) 語を受けた成分 {len(cc)}  '
                      f'成分数で share(comp_pur>=0.90) '
                      f'{sum(1 for x in pu if x>=0.90)/len(pu):.4f}  中央 {med(pu):.4f}')
                for lab2, lo, hi in (('<1000', 0, 1000), ('1000-9999', 1000, 10000),
                                     ('>=10000', 10000, 10**12)):
                    t = [r for r in cc if lo <= r['area'] < hi]
                    if not t:
                        continue
                    pu2 = [r['pur'] for r in t]
                    aw = sum(r['area'] for r in t)
                    print(f'      成分 {lab2:10} {len(t):6d}  share>=0.90 '
                          f'{sum(1 for x in pu2 if x>=0.90)/len(pu2):.4f}  '
                          f'画素重み {sum(r["pur"]*r["area"] for r in t)/aw:.4f}')
    if ('real', 'ako6', 'ch') in res and ('shuffled', 'ako6', 'ch') in res:
        print('\n' + '=' * 92)
        print('A / B / C (基盤 band_form_l18_20261025.md §2。式は動かさない)')
        for w in ('ako6', 'ako7'):
            if ('real', w, 'ch') not in res:
                continue
            ch1 = res[('real', w, 'ch')][0]; ch0 = res[('shuffled', w, 'ch')][0]
            r0 = res[('shuffled', w, 'ch')][1]
            rate = res[('real', w, 'rate')][0]
            sd0 = res[('shuffled', w, 'ch')][2]
            floor = ch0 + 3 * r0        # **対照 + 3r0**。裸の 3r0 は対照自身が通る
            floor_sd = ch0 + 3 * sd0    # 参考(基盤の指示で併記。主値ではない)
            print(f'  {w}: A cov_honest {ch1:.4f} >= 0.30 -> '
                  f'{"PASS" if ch1 >= 0.30 else "FAIL"}')
            print(f'      B 床 = 対照 {ch0:.4f} + 3r0 {3*r0:.4f} = **{floor:.4f}**  '
                  f'本来 {ch1:.4f} -> {"床の上" if ch1 >= floor else "床の下"}   '
                  f'比 {ch1/ch0 if ch0 else float("nan"):.2f}x')
            print(f'        (参考) 対照 + 3SD = {floor_sd:.4f} -> '
                  f'{"床の上" if ch1 >= floor_sd else "床の下"}  **主値ではない**')
            print(f'      C honest_rate {rate:.4f} >= 0.50 -> '
                  f'{"PASS" if rate >= 0.50 else "FAIL"}')
        # ---- **プール×階級の4セル** (基盤の裁定で必須)
        print('\n  プール x 階級の4セル (**判定はセル単位**。片方だけ通ったら'
              '「その階級では通った」で、プール全体の PASS ではない)')
        print(f'    {"プール":<6}{"階級":<14}{"墨":>7}{"A cov_honest":>14}{"床=対照+3r0":>14}'
              f'{"C rate":>9}  判定')
        for w in ('ako6', 'ako7'):
            for lab2 in ('最大成分<0.5', '最大成分>=0.5'):
                a1 = res.get(('real', w, lab2)); a0 = res.get(('shuffled', w, lab2))
                if not a1 or not a0:
                    continue
                fl = a0['ch'] + 3 * a0['r0']
                vA = a1['ch'] >= 0.30
                vB = a1['ch'] >= fl
                vC = a1['rate'] >= 0.50
                verdict = ('A' + ('o' if vA else 'x') + ' B' + ('o' if vB else 'x')
                           + ' C' + ('o' if vC else 'x'))
                print(f'    {w:<6}{lab2:<14}{a1["ink_share"]:>7.3f}{a1["ch"]:>14.4f}'
                      f'{fl:>14.4f}{a1["rate"]:>9.4f}  {verdict}'
                      + ('  **このセルは通った**' if (vA and vB and vC) else ''))
    if EXCLUDE:
        print_both_populations(res)


if __name__ == '__main__':
    main()
