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
import argparse
import time, re, csv, os, statistics as st, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l15_known_cells import boundary
from l17_known_cells import (D1, arms, assign_top, beta_fill_layers, jobs,
                             panels_and_regions,
                             place, SEEDS, TRIES, zmap)
from l18_known_cells import (GS, comp_pur, geo_spread, ink_components, prop_full,
                             seed_of)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l18_propagate'
EXCLUDE = None       # 事前登録の母集団定義 (--exclude-gapfill)
PAIRED = False       # L36: 対応のある読みを併記する (登録 2026-10-26 15:23)
STANDARD = 'solid'   # 標準の作り方。'solid' = fill_mask(radius=4) / 'cover' = 生の墨
                     # (L35 = 道1。登録 2026-10-26 15:07。**既定は変えない**)
LIMIT = 0            # --limit (煙試験。出力の名前に入れる)
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
    pan = 0               # **枠をまたいだコマの通し番号**(2026-10-26、KC46 が見つけた):
                          # `ci` は **koma_manifest の枠ごと**の成分番号なので、
                          # 同じ頁の別の枠で**同じ ci が再出する**。
                          # これが T109(鍵 `(pid, comp, region)` が一意でない)の機序。
                          # **同定のための列で、腕・棒・標準は動かさない。**
    names_all = []
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job, STANDARD):
        if not sub:
            continue
        pan += 1
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
        pans.append(dict(pid=pid, work=work, pan=pan, comp=ci, ink=tot, n_comp=n - 1,
                         top_share=round(areas[0] / tot, 4) if areas else 0,
                         med_comp=areas[len(areas) // 2] if areas else 0,
                         drop_gapfill=drop_gf))
        if ARMS == 'degen':
            for i in range(1, n):
                C = cl == i
                nl, nraw = n_lay90_of(C, lab)
                comps.append(dict(pid=pid, work=work, pan=pan, comp=ci, cid=i,
                                  area=int(st_[i, A]), share=round(st_[i, A] / tot, 5),
                                  n_lay90=nl, n_lay=nraw, arm='degen'))
            continue
        # ---- 腕
        # **L36**(登録 2026-10-26 15:23): `PAIRED` のとき
        #   (1) **名前の無い面をどちらの腕も採点しない**。この門は前から在ったが
        #       `EXCLUDE is not None` のときしか発火せず、全面の版では対照だけが
        #       **名前の無い面**(本来の位置では線に接していない面)を採点していた。
        #       これが朝に見つけた「対照だけ 229鍵」と同じ欠陥である
        #   (2) **seed ごとに両腕の交差の上で計算し直す**(対応のある読み)。
        #       **主値は置き換えない。併記である**
        gate = (EXCLUDE is not None) or PAIRED

        def run(arm, sd, allowed=None):
            """1つの腕・1つの seed を採点する。`allowed` が在ればその面だけ。

            返すのは (採点した面の gi の集合, regs 行, comps 行, pans 行)。
            **この関数は `lab` / `cl` / `P` を読むだけで作り直さない** ——
            両腕が同じ配列を見ることの保証がここに在る(基盤への答 2026-10-26 15:20)。
            """
            acc = np.zeros(P.shape, bool)
            seedlab = np.zeros(P.shape, np.uint16)
            owner, scored, rrows, crows = {}, set(), [], []
            for gi in gids:
                if gate and gi not in cname:
                    continue
                if allowed is not None and gi not in allowed:
                    continue
                if arm == 'real':
                    R = glab == gi
                    if place(glab, gi, gstat, P, SEEDS[0], pid, ci) is None:
                        continue          # 両腕で同じ面集合 (L17 §5)
                else:
                    R = place(glab, gi, gstat, P, sd, pid, ci)
                    if R is None:
                        continue
                s_, L, adj = seed_of(R, lab, P)
                if adj == 0 or L == 0:
                    continue
                C, ids = prop_full(s_, cl)
                acc |= C
                seedlab[s_ & (seedlab == 0)] = gi
                scored.add(gi)
                for x in ids:
                    owner.setdefault(int(x), []).append((gi, L, adj))
                rrows.append(dict(pid=pid, work=work, pan=pan, comp=ci, region=gi, arm=arm,
                                  seed=sd, adj=adj, prop=int(C.sum()),
                                  n_comp=int(len(ids)),
                                  name=cname.get(gi, '')))
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
                crows.append(dict(pid=pid, work=work, pan=pan, comp=ci, cid=cid, arm=arm,
                                  seed=sd, area=ar, share=round(ar / tot, 5),
                                  pur=round(pu, 4), n_lay90=nl, n_lay=nraw,
                                  conflict=cf, n_owner=len(lst)))
            rec = dict(pid=pid, work=work, pan=pan, comp=ci, arm=arm, seed=sd, ink=tot,
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
            return scored, rrows, crows, rec

        arms = (('real',) if ARMS in ('both', 'real') else ()) + \
               (('shuffled',) if ARMS in ('both', 'control') else ())
        got = {}
        for arm in arms:
            for sd in (SEEDS if arm == 'shuffled' else (0,)):
                sc, rr, cr, rec = run(arm, sd)
                got[(arm, sd)] = sc
                regs += rr; comps += cr; pans.append(rec)
        # ---- 対応のある読み (登録 §3。**主値を置き換えない。arm に印を付けて併記**)
        if PAIRED and ARMS == 'both':
            for sd in SEEDS:
                inter = got.get(('real', 0), set()) & got.get(('shuffled', sd), set())
                for arm in ('real', 'shuffled'):
                    _sc, rr, cr, rec = run(arm, sd if arm == 'shuffled' else 0, inter)
                    for row in rr:
                        row['arm'] = f'{arm}_paired'; row['seed'] = sd
                    for row in cr:
                        row['arm'] = f'{arm}_paired'; row['seed'] = sd
                    rec['arm'] = f'{arm}_paired'; rec['seed'] = sd
                    rec['n_inter'] = len(inter)
                    rec['n_real'] = len(got.get(('real', 0), set()))
                    regs += rr; comps += cr; pans.append(rec)
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
        # `res[(arm, w, 'ch')]` は **(中央, 最大−最小, SD) の3つ組**(380行)。
        # 2026-10-26: ここを dict だと思って `a0['ch']` と書いて煙試験で落ちた
        t1 = res.get(('real', w, 'ch')); t0 = res.get(('shuffled', w, 'ch'))
        if t1 and t0:
            arm, ctl, r0c, sd0 = t1[0], t0[0], t0[1], t0[2]
            fl = ctl + 3 * r0c
            print(f'  {"隙間埋けを外した版 (この走行)":<21}{w:<7}{arm:>9.4f}'
                  f'{ctl:>9.4f}{r0c:>9.4f}{fl:>13.4f}  '
                  f'{"床の上" if arm >= fl else "床の下"}'
                  f'   (併記・判定に使わない: 3SD = {3*sd0:.4f})')
        print()
    print('  † 前の走行の印字は `3r0` なので、この `r0` は3で割って得た**導出値**。'
          'この走行の `r0` は生の値')
    # **表だけが流通したときに読み違えられる1行**(基盤 2026-10-26)。
    # 「対照が 0.1001 -> 0.0893 に下がった」は独立の所見として読まれるが、
    # 実際には 45面の寄与と、名前の無い面(T110)の寄与が分離していない
    print('  **対照の列は版の中でだけ意味を持つ。版をまたいで引き算しない** —— '
          '除外版では')
    print('  除外の門が対照からも**名前の無い面**を落とすので(T110。'
          '`tools/evaluation/l18_t110_populations.py` で数えられる)、')
    print('  対照と床の版をまたいだ差には**45面の寄与と名前の無い面の寄与が'
          '混ざっていて分離できない。**')
    print('  **腕の比較は読める**(本来の腕はどちらの版でも名前のある面だけを見ている)。'
          '**各版の判定は自分の走行の中で完結している**')


def _num(v):
    """CSV の文字列を**走行中と同じ型**に戻す。`int` を先に試す(`seed` は整数)。

    空欄はそのまま空文字で返す —— コマの行には**2種類**あり
    (縮退の行と、腕 x seed ごとに採点した行)、`DictWriter` が
    片方に無い列を空で埋めている。**空を 0 に倒すと、無い量が 0 として読まれる。**
    """
    if v is None or v == '':
        return v
    for f in (int, float):
        try:
            return f(v)
        except (TypeError, ValueError):
            pass
    return v


def load_run(arms, tag):
    """既に書かれた CSV から、走行の3つの表と面の名前を読み戻す。

    **走行をやり直さずに報告だけを出し直すための経路**(2026-10-27)。
    checkpoint が頁ごとに書いているので、**本走行の実測はファイルに全部在る** ——
    足りなかったのは最後の集計だけだった。

    **この経路自身の既知セル**: 前の走行の印字を再現できること。
    `--report-from _rawbeta_limit6` が 2026-10-26 の 6頁の印字と一致し、
    `--report-from _rawbeta_limit2` が同日の煙試験の印字と一致すること
    (**集計の算術は走行と同じコードを通る。試されるのは読み戻しの方である**)。
    """
    out = []
    for nm in ('comps', 'regions', 'panels'):
        path = f'{OUT}/{nm}_{arms}{tag}.csv'
        rows = [{k: _num(v) for k, v in r.items()}
                for r in csv.DictReader(open(path, encoding='utf-8'))]
        print(f'<- {path}  行 {len(rows)}')
        out.append(rows)
    npath = f'{OUT}/names_{arms}{tag}.txt'
    names = [x.strip() for x in open(npath, encoding='utf-8') if x.strip()]
    print(f'<- {npath}  面 {len(names)}')
    return out[0], out[1], out[2], names


def kc30(names, raw_beta, exclude, limit=0):
    """KC30 データ経路の点検。**標準ごとに当たる水準が違う**(2026-10-27 に直した)。

    もとは「L19 の 551面の名前が今回の名前に全部あるか」の1本だけだった。
    **これは `fill_mask` の標準でしか通らない。** 面の名前は
    `{work}_{pid}_c{ci}_r{gi}_{k:03d}.png` で、`r{gi}` は**標準の面番号**、
    `k` は**頁内の通し番号** —— どちらも標準から出る量である。
    L35(道1)は標準を納品 beta 層の生の墨に差し替えるので、
    **名前は原理的に一致しない。**

    2026-10-27 の本走行(65頁、2.81時間)が**この門で止まった**。実測:

    ```
    名前まるごと    一致  52 / 551     <- 止まったのはこの水準
    k を外す        一致 223 / 491
    r も外す(コマ) 一致 136 / 139
    頁              一致  64 /  64     <- **データ経路は切れていない**
    ```

    > **通らない点検は、何も測らない点検と同じだけ悪い。**
    > 前者は走行を止め、後者は黙って通す。**どちらも腕について何も言っていない。**

    そして**煙試験はこの門を見られなかった** —— `if not a.limit` で囲まれていたので、
    2頁の煙試験は門を飛ばして表まで印字する。**煙試験が構造的に盲だった門である。**

    水準を下げたのではない。**名前の一致は道1では定義されない量**で、
    道1で当たるのは頁の水準である。落ちた数は理由別に印字する(規律9)。
    """
    import json as _j
    import re as _re
    q = _j.load(open('/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'))
    want = {r['crop'] for r in q['queue']}
    got = set(names)
    pg = lambda n: n.split('_c')[0]
    km = lambda n: _re.sub(r'_r\d+_\d{3}\.png$', '', n)
    if limit:
        # **部分走行では門にしない。だが印字する**(T134 の族: 発火しない門は、
        # 印字しなければ在ることが分からない)。L19 の 551面は 64頁に散っているので、
        # **頁を切った走行では「全部ある」は原理的に満たせない** ——
        # ここで止めると煙試験が常に落ちる。
        # **煙試験が門を見られないのと、門が煙試験を落とすのは別の病で、どちらも直す。**
        print(f'KC30 データ経路: **{limit}頁の部分走行なので門にしない**(印字のみ)')
        print(f'  この走行の頁 {len({pg(n) for n in got})}  面 {len(got)}  '
              f'L19 の面との名前の一致 {len(want & got)}/{len(want)}'
              f'{"(道1 では名前は別の量。読まない)" if raw_beta else ""}')
        return 0
    if not raw_beta:
        ok = want <= got
        print(f'KC30 データ経路: L19 の 551 面の名前が今回の名前に全部あるか: '
              f'{"ok" if ok else "FAIL"}  (面 {len(got)} / 欠け {len(want - got)})')
        if not ok:
            print('**除外の鍵が合っていない。止める**')
            return 1
        if exclude:
            print(f'  選別(規律9-1): 隙間埋けとして外した面 {len(exclude & got)} / '
                  f'指定 {len(exclude)}')
        return 0
    wp, gp = {pg(n) for n in want}, {pg(n) for n in got}
    wk, gk = {km(n) for n in want}, {km(n) for n in got}
    ok = wp <= gp
    print(f'KC30(道1 の形): **名前の一致は道1では定義されない** —— '
          f'標準が違うので `r{{gi}}` と `k` が別の量になる'
          f'(名前での一致 {len(want & got)}/{len(want)} は**読まない**)')
    print(f'  **頁の水準(これが門)**: L19 の {len(wp)}頁が今回の {len(gp)}頁に'
          f'全部あるか: {"ok" if ok else "FAIL"}  欠けた頁 {len(wp - gp)}')
    print(f'  コマの水準(参考。**棒ではない**): 一致 {len(wk & gk)} / L19 {len(wk)}  '
          f'欠け {len(wk - gk)}  (標準が違えば1面も残らないコマが出る)')
    if not ok:
        print('**頁が欠けている。データ経路が切れている。止める**')
        return 1
    return 0


def main():
    global ARMS, LIMIT, EXCLUDE, PAIRED
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both',
                    choices=('both', 'real', 'control', 'degen'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--exclude-gapfill', action='store_true',
                    help='**事前登録**: 判定のある551面の中で見つかった隙間埋けだけを外す'
                         ' (残り635面は未知)')
    ap.add_argument('--paired', action='store_true',
                    help='**L36**(登録 2026-10-26 15:23): 名前の無い面をどちらの腕も'
                         '採点しないようにし、**seed ごとに両腕の交差の上での '
                         '`cov_honest` を併記**する。**主値・棒・床・標準は動かさない**')
    ap.add_argument('--report-from', default='',
                    help='**走行をやり直さず**、既に書かれた CSV から報告だけを'
                         '出し直す。値は `comps_<arms><tag>.csv` の `<tag>`'
                         '(例 `_rawbeta`、`_rawbeta_limit6`)。'
                         '**前の走行の印字を再現できることが門**(`load_run`)')
    ap.add_argument('--raw-beta-standard', action='store_true',
                    help='**L35 = 道1**(登録 2026-10-26 15:07): 標準を '
                         '`fill_mask` から外し、**納品 beta 496層の生の墨**から作る。'
                         '`radius` が器具のどこにも残らない')
    a = ap.parse_args()
    if a.exclude_gapfill and a.raw_beta_standard:
        # **鍵は標準から出る**(KC30 の docstring)。道1 では1面も当たらないので、
        # 黙って「0面を外した版」になる。**名前で結ぶ器具は、標準が変わると空振りする**
        sys.exit('止める: `--exclude-gapfill` の鍵は `fill_mask` の標準での'
                 '**面の名前**なので、道1(`--raw-beta-standard`)では1面も当たらない')
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
    LIMIT = a.limit
    PAIRED = a.paired
    if PAIRED:
        print('**L36**: 名前の無い面は**どちらの腕も採点しない**(この門は前から'
              '在ったが `--exclude-gapfill` のときしか発火しなかった)。'
              'そして **seed ごとに両腕の交差の上で計算し直して併記**する', flush=True)
        print('  **主値は登録のまま**(各腕の現在の面集合の上の `cov_honest`)。'
              '**対応のある読みは棒ではない**(登録 §3)', flush=True)
    os.makedirs(OUT, exist_ok=True)
    # **T131**(2026-10-26): 腕と標準の決め方を `arms(kind)` 1か所に寄せた。
    # それまでは `gated_layers()` が素の集合を返し、**289層の除去が下の
    # `--raw-beta-standard` の枝の中に在った** —— **どの走行がどちらの集合を
    # 使ったかが、呼び出し行を読まないと分からなかった**(同じ数を2度取り違えた)。
    zz = zmap()
    fill, line = arms('beta_standard' if a.raw_beta_standard else 'gated')
    if a.raw_beta_standard:
        global STANDARD
        STANDARD = 'cover'
        print('標準: **納品 beta 層の生の墨の8連結成分**(L35 = 道1)。'
              '**`fill_mask` を通さない —— `radius` は器具のどこにも残らない**',
              flush=True)
        print('  **L18 の数値と引き算しない。標準が違うので別の量である**(T110)',
              flush=True)
        print('  **継承する不確かさ**: この496層は抽出側の門 '
              '(`preview_dark >= 0.5` ほか) で選ばれていて、T116/T120 で '
              '**その門は margin 未検証**。L33 の窓の上側14層は14層ともこの496に'
              '入っている = **標準の境界に 14/496 = 2.8% の既知の不確かさ**。'
              '**2.8% 未満の差は読まない**(登録 §5)', flush=True)
        print('  **道1は道2の代わりにならない** —— これが答えるのは'
              '「作者のベタ層はどんな形をしているか」で、'
              '**「作者が面と呼ぶものは何か」には答えない**(登録 §6)', flush=True)
        # **標準の差し替えと289層の除去は `arms('beta_standard')` が済ませている**
        # (T131。上の呼び出し行を見よ)。**ここで二度やらない。**
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'腕 {ARMS}  頁 {len(js)}  (既知セル KC21-24 は別プロセスで PASS 済みであること)')

    # **出力の名前に母集団と頁数を入れる**(2026-10-26)。
    # それまでは `comps_both.csv` 固定で、
    #   ・2頁の煙試験が **65頁の走行の CSV を上書きした**(実際に失った)
    #   ・隙間埋けを外した版も、全面の版の CSV を上書きする形だった
    # 走行の種類が違うものを同じ名前に書かない。
    tag = '_rawbeta' if a.raw_beta_standard else ''
    if a.paired:
        tag += '_paired'
    if EXCLUDE:
        tag += '_nogapfill'
    if LIMIT:
        tag += f'_limit{LIMIT}'
    if a.report_from:
        tag = a.report_from
        print(f'**報告のみ**: {tag} の CSV から読み戻す(走行はしない)', flush=True)

    # ---- checkpoint (2026-10-26)。**長い走行の最後に落ちて全部失わないため。**
    # それまでは**最後に一度だけ**書いていた。29時間の走行ではこれは取れない。
    # 頁が1つ終わるたびに追記する。**列は最初の頁で決まる**ので、
    # 後の頁が新しい列を出したら**警告して落とす**(黙って欠けさせない)。
    ck = {}

    def checkpoint(nm, rows):
        if not rows:
            return
        path = f'{OUT}/{nm}_{ARMS}{tag}.csv'
        if nm not in ck:
            keys = sorted({k for r in rows for k in r})
            fh = open(path, 'w', newline='', encoding='utf-8')
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            ck[nm] = (fh, w, set(keys), path, 0)
        fh, w, keys, path, n = ck[nm]
        extra = {k for r in rows for k in r} - keys
        if extra:
            print(f'**止める**: {nm} の列が頁の途中で増えた {sorted(extra)} —— '
                  '**最初の頁で決まった見出しに無い列は黙って落ちる**', flush=True)
            raise SystemExit(3)
        for r in rows:
            w.writerow(r)
        fh.flush()
        ck[nm] = (fh, w, keys, path, n + len(rows))

    comps, regs, pans, names = [], [], [], []
    # **走行の頭で切り詰める**(2026-10-26、T131 の再現確認で出た)。
    # CSV は `checkpoint()` が最初の頁で `'w'` で開くので置き換わるのに、
    # **名前のファイルだけ `'a'` で追記し続けていた** —— 同じ引数で2度走らせると
    # **中身が静かに2倍になる**(CSV は正しいので気付かない)。
    # **同じ走行の出力なのに、ファイルによって「1回ぶん」の意味が違っていた。**
    if a.report_from:
        comps, regs, pans, names = load_run(ARMS, tag)
    else:
        open(f'{OUT}/names_{ARMS}{tag}.txt', 'w', encoding='utf-8').close()
        t0 = time.time()
        with Pool(4) as pool:
            for i, (c, r, p, nm) in enumerate(pool.imap_unordered(one_page, js), 1):
                comps += c; regs += r; pans += p; names += nm
                checkpoint('comps', c); checkpoint('regions', r); checkpoint('panels', p)
                with open(f'{OUT}/names_{ARMS}{tag}.txt', 'a', encoding='utf-8') as fh:
                    for x in nm:
                        fh.write(x + '\n')
                el = time.time() - t0
                # **見積りは最初の数頁では使えない**(2026-10-26)。`Pool(4)` なので
                # 4頁が同時に走っていて、`i=1` の時点の `el/i` は**並列を無視した値**。
                # i が 4 の数倍になるまで**過大に出る**ので、そのときは印を付ける
                # (付けないと、その数が引用される)。
                warm = '  **見積りは温まっていない(並列4本、i<8)**' if i < 8 else ''
                print(f'  {i}/{len(js)} 頁  成分 {len(comps)}  '
                      f'経過 {el / 60:.1f}分  1頁 {el / i / 60:.2f}分  '
                      f'残り見積り {(len(js) - i) * el / i / 60:.0f}分{warm}',
                      flush=True)
    for nm_, v in ck.items():
        v[0].close()
        print(f'-> {v[3]}  行 {v[4]}  (**checkpoint で頁ごとに追記済み**)')
    # ---- KC30 データ経路(**標準ごとに当たる水準が違う**。2026-10-27 に直した)
    # 煙試験が見られない門にしない: `--limit` でも回す。
    if kc30(names, a.raw_beta_standard, EXCLUDE, LIMIT):
        return 1
    # **出力の名前に母集団と頁数を入れる**(2026-10-26)。
    # それまでは `comps_both.csv` 固定で、
    #   ・2頁の煙試験が **65頁の走行の CSV を上書きした**(実際に失った)
    #   ・隙間埋けを外した版も、全面の版の CSV を上書きする形だった
    # 走行の種類が違うものを同じ名前に書かない。
    # (CSV は上の checkpoint で既に書かれている)

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
    # ---- 対応のある読み (L36。登録 2026-10-26 15:23 §3。**棒ではない。副値**)
    if PAIRED:
        print('\n' + '=' * 92)
        print('**対応のある読み**(L36 の副値。**棒ではない。主値を置き換えない**)')
        print('  seed ごとに「本来の腕が採点した面 ∩ その seed の対照が採点した面」の上で')
        print('  両腕を計算し直したもの。**本来の腕の値も seed ごとに変わる**(交差が違うから)')
        for w in ('ako6', 'ako7'):
            rows = [r for r in pans if r['work'] == w
                    and str(r.get('arm', '')).endswith('_paired')]
            if not rows:
                continue
            print(f'\n  プール {w}')
            per_r, per_s = [], []
            for sd in sorted({r['seed'] for r in rows}):
                out = {}
                for arm in ('real_paired', 'shuffled_paired'):
                    t = [r for r in rows if r['seed'] == sd and r['arm'] == arm]
                    if not t:
                        continue
                    ink = sum(r['ink'] for r in t)
                    if not ink:
                        continue
                    out[arm] = (sum(r['honest'] for r in t) / ink,
                                sum(r['honest'] for r in t) /
                                max(sum(r['cov'] for r in t), 1),
                                sum(r['conflict_px'] for r in t) / ink)
                ni = sum(r.get('n_inter', 0) for r in rows
                         if r['seed'] == sd and r['arm'] == 'real_paired')
                nr = sum(r.get('n_real', 0) for r in rows
                         if r['seed'] == sd and r['arm'] == 'real_paired')
                if 'real_paired' in out and 'shuffled_paired' in out:
                    per_r.append(out['real_paired'][0])
                    per_s.append(out['shuffled_paired'][0])
                    print(f'    seed {sd}: 交差 {ni}/{nr} = '
                          f'{ni/nr if nr else 0:.4f}   '
                          f'本来 cov_honest {out["real_paired"][0]:.4f} '
                          f'(rate {out["real_paired"][1]:.4f}, '
                          f'競合 {out["real_paired"][2]:.4f})   '
                          f'対照 {out["shuffled_paired"][0]:.4f} '
                          f'(rate {out["shuffled_paired"][1]:.4f}, '
                          f'競合 {out["shuffled_paired"][2]:.4f})')
            if per_r and per_s:
                r0p = max(per_s) - min(per_s)
                ch1 = sum(per_r) / len(per_r)      # seed ごとの交差が違うので平均
                ch0 = med(per_s)
                floor = ch0 + 3 * r0p
                # **本来の腕にも seed 間のばらつきが在る**(2026-10-26 の煙試験で
                # 0.1396〜0.5602 = 幅 0.42 が出た)。交差が seed ごとに違うので
                # **本来の腕の母集団も seed ごとに違う**。
                # **対照の r0 だけを印字すると、本来の腕が1点のように見える。**
                r0r = max(per_r) - min(per_r)
                print(f'    **対応のある読み**: 本来 平均 {ch1:.4f} '
                      f'(seed ごとに母集団が違うので中央でなく平均)  '
                      f'**本来の腕の幅 {min(per_r):.4f}〜{max(per_r):.4f} '
                      f'= r0 {r0r:.4f}**  '
                      f'対照 中央 {ch0:.4f}  r0 {r0p:.4f}  床 {floor:.4f}')
                if r0r >= abs(ch1 - ch0):
                    print(f'      **本来の腕のばらつき {r0r:.4f} が '
                          f'腕と対照の差 {abs(ch1-ch0):.4f} 以上** —— '
                          f'この読みでは腕と対照を分けられない')
                print(f'      A {ch1:.4f} >= 0.30 -> '
                      f'{"PASS" if ch1 >= 0.30 else "FAIL"}   '
                      f'B {"床の上" if ch1 >= floor else "床の下"}')
                print('      **これは棒ではない。登録 §4 の帰結を読むための副値である**')
                print('      (交差が本来の腕の5割未満なら「対応のある読みは成立しない」'
                      '= 登録 §4 の3行目)')
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
