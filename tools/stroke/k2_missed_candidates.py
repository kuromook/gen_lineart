#!/usr/bin/env python
"""K2(2026-10-16 事前登録 + 修正 A1〜A7): 同じ部位の中の「取り逃がしの候補」を数える。

**これは真値ではなく候補の数である。「取り逃がし率」と書いてはならない。**

3 箱に分けて**合算しない**(A4):
  箱 A  同じ(畳み込み後の)節点に出ている腕の対で、門を通るのに**現行規則が対にしなかった**もの
  箱 C  両端とも節点付きだが**節点が別**な tip の対(J の topology 側)
  箱 B  **自由端どうし**の tip の対で、襟で切った線分上の墨が 0(A2)

門(A6-2、J の `stroke_chain.py` から読むだけで import): 角 **30 度以内**(J の THETA_MAX=60 では
ない。A3)、横ずれ **3.0px 以内**(J の D_MAX)、接線は `TAN_SKIP=2` の襟を飛ばして先端で評価。
距離の上限 `GAP_MAX=24px`(箱 C と箱 B)。

率の定義(A6-3、結果を見る前に固定): 箱ごとに
  分母 = 構造条件を満たす対の数 / 分子 = 門を通った対の数
  同じ部位 = 2 つの token が作者の同じグループ / またぎ = 別のグループ
  どちらのグループにも入っていない線は分母にも入れない。同じ token の中の対は数えない
NC-k2: **箱ごとに**「同じ部位の率 >= またぐ率の 2.0 倍」。下回れば共線の判定は何も測っていない
IC: IC-k5a(複製 = 原本)+ 本数が pack と一致(A6-1。IC-k5b/IC-k7 は既に合格済み)
"""
import argparse, csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
TRACK_J = Path("/home/sh1/deepl/lineart-solid-fill")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))
sys.path.insert(0, str(TRACK_J / "tools/evaluation"))

THR, GROW, MAX_BEND = 128, 200, 35.0        # 作者の線を作った経路(IC-k5b で判明)
ANGLE_MAX = 30.0                            # K の登録値(J の THETA_MAX ではない)
D_MAX_K = 3.0                               # = J の較正値 D_MAX
GAP_MAX = 24.0                              # A1 で登録
NC_FACTOR = 2.0                             # NC-k2 の合格条件(箱ごと)
IC_K5A_PANELS = 20                          # IC-k5a を当てるコマ数(先頭から機械的に)

J_SHA = "53b8406cfbb4f4d1e4c53a15102ec8079a4f541e77e0523b6836af9d5a9a56a6"


def tip_geometry(path):
    """両端の (tip, T, K, frame) を J の arm_geometry で取る。path は (n,2) float。"""
    import stroke_chain as sc
    if len(path) < 2:
        return None, None
    return sc.arm_geometry(path, 0), sc.arm_geometry(path, 1)


def gate(ai, aj):
    """(通ったか, 角度, 横ずれ)。J の pair_cost と同じ式で、門だけ K の値にする。"""
    import stroke_chain as sc
    ti, Ti, _Ki, Fi = ai
    tj, Tj, _Kj, Fj = aj
    if Ti is None or Tj is None:
        return False, float("nan"), float("nan")
    dot = float(np.clip(Ti @ (-Tj), -1.0, 1.0))
    theta = float(np.degrees(np.arccos(dot)))
    d = 0.5 * (sc.perp_offset(Fi, tj) + sc.perp_offset(Fj, ti))
    return bool(theta <= ANGLE_MAX and d <= D_MAX_K), theta, float(d)


def seg_points(a, b, step=1.0):
    n = max(int(np.ceil(np.hypot(*(b - a)) / step)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def work(t):
    pid, source, name, do_ic = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths, order_path
    import strokes_instrumented as SI
    import strokes as F_ORIG                                  # 読むだけ(IC-k5a 用)

    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    ink_plain = SI.ink_mask(g)                                 # A7-2: 素の gray<128 も併記
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    iw = dist * 2.0
    sk = skeletonize(ink)
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(sk, iw, max_bend=MAX_BEND)

    ic_same = None
    if do_ic:                                                  # IC-k5a: 複製 = 原本
        o = F_ORIG.strokes_from_skeleton(skeletonize(ink), dist * 2.0, max_bend=MAX_BEND)
        ic_same = bool(len(o[0]) == len(spans)
                       and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                               for a, b in zip(o[0], spans))
                       and np.array_equal(o[1], sos) and np.array_equal(o[2], keep)
                       and o[3] == info)

    # --- token(作者の線)の番号付け: export_panel_tokens_v2 と同じ順序 ---
    gr = {}
    for i in range(len(spans)):
        gr.setdefault(int(sos[i]), []).append(i)
    tok_spans, span_tok = [], {}
    for k, mem in gr.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        tid = len(tok_spans)
        tok_spans.append(mem)
        for i in mem:
            span_tok[i] = tid

    # --- span ごとの順序付き経路と両端の幾何 ---
    paths, tips, geo = {}, {}, {}
    for i in range(len(spans)):
        rr, cc = spans[i]
        if len(rr) == 0:
            continue
        p = order_path(np.stack([rr, cc], 1)).astype(np.float64)
        if len(p) < 2:
            continue
        paths[i] = p
        tips[i] = (p[0], p[-1])
        geo[i] = tip_geometry(p)

    # --- 付着点を最も近い tip に割り当てる(A7-4) ---
    node_root = extra["node_root"]
    tip_node = {}                      # (span, end) -> 畳み込み後の節点
    att_by_span = defaultdict(list)
    for s, n, ar, ac in extra["att"]:
        att_by_span[s - 1].append((n, np.array([ar, ac], float)))
    n_two_on_one_tip = n_att_no_path = 0
    for i, lst in att_by_span.items():
        if i not in tips:
            n_att_no_path += len(lst); continue
        got = {}
        for n, pt in lst:
            e = 0 if np.hypot(*(tips[i][0] - pt)) <= np.hypot(*(tips[i][1] - pt)) else 1
            if e in got:
                n_two_on_one_tip += 1
                if np.hypot(*(tips[i][e] - pt)) >= got[e][1]:
                    continue
            got[e] = (n, float(np.hypot(*(tips[i][e] - pt))))
        for e, (n, _d) in got.items():
            tip_node[(i, e)] = int(node_root.get(int(n), int(n)))

    # --- 骨格画素 -> token(A7-1 の crossed 判定に使う) ---
    sk_pts, sk_tok = [], []
    for i, (rr, cc) in enumerate(spans):
        tid = span_tok.get(i, -1)
        if tid < 0 or len(rr) == 0:
            continue
        sk_pts.append(np.stack([rr, cc], 1).astype(np.float64))
        sk_tok.append(np.full(len(rr), tid, np.int32))
    sk_pts = np.concatenate(sk_pts) if sk_pts else np.zeros((0, 2))
    sk_tok = np.concatenate(sk_tok) if len(sk_pts) else np.zeros(0, np.int32)
    sk_tree = cKDTree(sk_pts) if len(sk_pts) else None

    linked = set(tuple(x) for x in extra["linked_pairs"])
    bridge_flag = extra["bridge_flag"]
    H, W = ink.shape

    def tip_w(i, e):
        r, c = tips[i][e]
        return float(iw[int(np.clip(round(r), 0, H - 1)), int(np.clip(round(c), 0, W - 1))])

    rows = []       # 1 対 1 行。箱・token 対・門の結果・付随する量

    # ---------------- 箱 A: 同じ節点に出ている腕の対 ----------------
    cos_lim = -np.cos(np.deg2rad(MAX_BEND))        # 原本の門(記録用。判定には使わない)
    for node, lst in extra["arms"].items():
        arms = [(s - 1, s) for s, _v in lst]
        fv = {s: np.asarray(v, float) for s, v in lst}
        used_here = {x for pr in linked for x in pr if x in fv}
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                i, si = arms[a]; j, sj = arms[b]
                if i == j or i not in geo or j not in geo:
                    continue
                ti = span_tok.get(i, -1); tj = span_tok.get(j, -1)
                if ti < 0 or tj < 0 or ti == tj:
                    continue
                # この節点に付いている端を選ぶ
                ei = next((e for e in (0, 1) if tip_node.get((i, e)) == int(node)), None)
                ej = next((e for e in (0, 1) if tip_node.get((j, e)) == int(node)), None)
                if ei is None or ej is None:
                    continue
                ok, th, d = gate(geo[i][ei], geo[j][ej])
                fdot = float(fv[si] @ fv[sj])
                rows.append(dict(box="A", t0=ti, t1=tj, s0=i, s1=j, e0=ei, e1=ej,
                                 node0=int(node), node1=int(node),
                                 gap=float(np.hypot(*(tips[i][ei] - tips[j][ej]))),
                                 theta=th, dperp=d, passed=int(ok),
                                 linked=int((min(si, sj), max(si, sj)) in linked),
                                 ink_on_seg=-1, ink_on_seg_plain=-1, crossed=-1,
                                 seg_n=0, collar=float("nan"),
                                 br0=int(bridge_flag[si]), br1=int(bridge_flag[sj]),
                                 f_dot=round(fdot, 4), f_would=int(fdot <= cos_lim),
                                 f_arm_used=int(si in used_here or sj in used_here)))

    # ---------------- 箱 C / 箱 B: tip の対(24px 以内) ----------------
    keys = [(i, e) for i in paths for e in (0, 1) if span_tok.get(i, -1) >= 0]
    if keys:
        pts = np.array([tips[i][e] for i, e in keys])
        tree = cKDTree(pts)
        for a, b in tree.query_pairs(GAP_MAX):
            (i, ei), (j, ej) = keys[a], keys[b]
            if i == j:
                continue
            ti = span_tok[i]; tj = span_tok[j]
            if ti == tj:
                continue
            ni = tip_node.get((i, ei)); nj = tip_node.get((j, ej))
            gap = float(np.hypot(*(pts[a] - pts[b])))
            if ni is not None and nj is not None:
                if ni == nj:
                    continue                                   # 箱 A の領分
                ok, th, d = gate(geo[i][ei], geo[j][ej])
                rows.append(dict(box="C", t0=ti, t1=tj, s0=i, s1=j, e0=ei, e1=ej,
                                 node0=int(ni), node1=int(nj), gap=gap, theta=th, dperp=d,
                                 passed=int(ok), linked=0, ink_on_seg=-1,
                                 ink_on_seg_plain=-1, crossed=-1, seg_n=0, collar=float("nan"),
                                 br0=int(bridge_flag[i + 1]), br1=int(bridge_flag[j + 1]),
                                 f_dot=float("nan"), f_would=-1, f_arm_used=-1))
            elif ni is None and nj is None:
                ok, th, d = gate(geo[i][ei], geo[j][ej])
                # A2: 襟で切った線分上の墨画素数(主は grow=200、素の ink も併記)
                pa, pb = pts[a], pts[b]
                col = max(2.0, 0.5 * max(tip_w(i, ei), tip_w(j, ej)))
                v = pb - pa; L = float(np.hypot(*v))
                nink = nink_p = -1; seg_n = 0
                if L > 2 * col + 1e-6:
                    u = v / L
                    q0, q1 = pa + u * col, pb - u * col
                    sp = seg_points(q0, q1)
                    rr = np.clip(np.round(sp[:, 0]).astype(int), 0, H - 1)
                    cc = np.clip(np.round(sp[:, 1]).astype(int), 0, W - 1)
                    nink = int(ink[rr, cc].sum()); nink_p = int(ink_plain[rr, cc].sum())
                    seg_n = len(sp)
                # A7-1: 第三の token の骨格が線分のすぐ脇を通るか
                crossed = 0
                if sk_tree is not None:
                    r = max(2.0, 0.5 * (tip_w(i, ei) + tip_w(j, ej)) / 2.0)
                    for idx in sk_tree.query_ball_point(seg_points(pa, pb, 2.0), r):
                        if any(sk_tok[m] not in (ti, tj) for m in idx):
                            crossed = 1; break
                rows.append(dict(box="B", t0=ti, t1=tj, s0=i, s1=j, e0=ei, e1=ej,
                                 node0=-1, node1=-1, gap=gap, theta=th, dperp=d,
                                 passed=int(ok), linked=0, ink_on_seg=nink,
                                 ink_on_seg_plain=nink_p, crossed=crossed,
                                 seg_n=seg_n, collar=round(col, 2),
                                 br0=int(bridge_flag[i + 1]), br1=int(bridge_flag[j + 1]),
                                 f_dot=float("nan"), f_would=-1, f_arm_used=-1))

    diag = dict(n_tokens=len(tok_spans), n_spans=len(spans), ic_k5a_same=ic_same,
                n_att=len(extra["att"]),
                n_two_on_one_tip=n_two_on_one_tip, n_att_no_path=n_att_no_path,
                n_free_tips=int(sum(1 for i, e in keys if tip_node.get((i, e)) is None)) if keys else 0,
                n_node_tips=int(sum(1 for i, e in keys if tip_node.get((i, e)) is not None)) if keys else 0)
    return pid, (rows, diag)


def rate(num, den):
    return round(num / den, 5) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/k2_missed_candidates_20261017")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    import hashlib
    sha = hashlib.sha256((TRACK_J / "tools/evaluation/stroke_chain.py").read_bytes()).hexdigest()
    if sha != J_SHA:
        print(f"J の stroke_chain.py が登録時と違う: {sha}\n止めて相談する"); return

    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    recs = sorted([r for r in G.S["faces"].values()
                   if r["usage"] == "train" and r["status"] == "done"],
                  key=lambda r: (r["panel"], r["order"]))
    if a.limit:
        recs = recs[:a.limit]
    n_broken = sum("broken" in r for r in recs)
    pids = sorted({r["panel"] for r in recs})
    P = G.S["panels"]
    print(f"顔 {len(recs)}(broken の項目あり {n_broken} / 無し {len(recs) - n_broken})"
          f" / コマ {len(pids)}", flush=True)

    tasks = [(p, P[p]["source"], P[p]["name"], k < IC_K5A_PANELS) for k, p in enumerate(pids)]
    byp = {}
    with Pool(a.workers) as pool:
        for i, (pid, o) in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            byp[pid] = o
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    # --- 器具確認 ---
    ic5a = [o[1]["ic_k5a_same"] for o in byp.values() if o and o[1]["ic_k5a_same"] is not None]
    ic_count_bad = [p for p in pids
                    if byp.get(p) is None or byp[p][1]["n_tokens"] != int(P[p]["n"])]
    chk = dict(ic_k5a_copy_equals_original=dict(n_panels=len(ic5a), all_same=bool(all(ic5a)),
                                                passed=bool(ic5a and all(ic5a))),
               ic_token_count_equals_pack=dict(n_panels=len(pids), n_bad=len(ic_count_bad),
                                               bad=ic_count_bad[:10], passed=not ic_count_bad),
               j_stroke_chain_sha256=sha,
               registered=dict(angle_max=ANGLE_MAX, d_max=D_MAX_K, gap_max=GAP_MAX,
                               nc_factor=NC_FACTOR, thr=THR, grow=GROW, max_bend=MAX_BEND))
    chk["passed"] = bool(chk["ic_k5a_copy_equals_original"]["passed"]
                         and chk["ic_token_count_equals_pack"]["passed"])
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    if not chk["passed"]:
        print("器具確認: 不合格 —— 止めて相談する"); return
    print("器具確認: 合格", flush=True)

    # --- 顔ごとに、対の両側の token をグループで色分けする ---
    rows = []
    for r in recs:
        o = byp.get(r["panel"])
        if o is None:
            continue
        grp = {}
        for gi, g_ in enumerate(r["groups"]):
            for sid in g_["strokes"]:
                grp[int(sid)] = gi
        if not grp:
            continue
        for d in o[0]:
            g0, g1 = grp.get(d["t0"]), grp.get(d["t1"])
            if g0 is None or g1 is None:            # どちらかがグループ外 -> 分母に入れない
                continue
            rows.append(dict(order=r["order"], panel=r["panel"], same=int(g0 == g1), **d))

    if not rows:
        print("対が 1 つも無い"); return
    with open(out / "per_pair.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    box = lambda b: [x for x in rows if x["box"] == b]
    res = {}
    for b in ("A", "C", "B"):
        R = box(b)
        d = {}
        for lab, sel in (("same_part", [x for x in R if x["same"]]),
                         ("across_parts", [x for x in R if not x["same"]])):
            den = len(sel)
            if b == "A":
                num = [x for x in sel if x["passed"] and not x["linked"]]
            elif b == "B":
                num = [x for x in sel if x["passed"] and x["ink_on_seg"] == 0]
            else:
                num = [x for x in sel if x["passed"]]
            e = dict(denominator=den, numerator=len(num), rate=rate(len(num), den))
            if b == "A":
                e["passed_gate_but_linked"] = sum(1 for x in sel if x["passed"] and x["linked"])
                e["passed_gate"] = sum(1 for x in sel if x["passed"])
                # A8: 候補の内訳 —— 原本が候補にすらしなかったのか、貪欲が腕を別に使ったのか
                e["breakdown_A8"] = dict(
                    f_rejected_direction=sum(1 for x in num if not x["f_would"]),
                    f_candidate_but_arm_used=sum(1 for x in num if x["f_would"] and x["f_arm_used"]),
                    f_candidate_arm_free=sum(1 for x in num if x["f_would"] and not x["f_arm_used"]))
            if b == "B":
                e["passed_gate"] = sum(1 for x in sel if x["passed"])
                e["crossed_of_numerator"] = sum(1 for x in num if x["crossed"] == 1)
                e["not_crossed_of_numerator"] = sum(1 for x in num if x["crossed"] == 0)
            if num:
                e["gap_px"] = dict(p10=round(float(np.percentile([x["gap"] for x in num], 10)), 2),
                                   median=round(float(np.median([x["gap"] for x in num])), 2),
                                   p90=round(float(np.percentile([x["gap"] for x in num], 90)), 2))
                e["n_faces"] = len({x["order"] for x in num})
                e["bridge_side"] = sum(1 for x in num if x["br0"] or x["br1"])
            d[lab] = e
        rs, ra = d["same_part"]["rate"], d["across_parts"]["rate"]
        na, da = d["across_parts"]["numerator"], d["across_parts"]["denominator"]
        ns, ds = d["same_part"]["numerator"], d["same_part"]["denominator"]
        from scipy.stats import beta as _beta, fisher_exact as _fe
        # またぎの率の 95% 片側上限(Clopper-Pearson)。0/84 のような分母の小ささを隠さない
        up = 1.0 if da == 0 else (1.0 if na == da else
                                  float(_beta.ppf(0.95, na + 1, da - na)))
        _, pf = _fe([[ns, ds - ns], [na, da - na]], alternative="greater") if da else (0, float("nan"))
        if da == 0:
            fac, verdict = float("nan"), "判定できない(またぎの対が 0 件)"
        elif ra == 0:
            fac = float("inf")
            verdict = ("合格(点推定では無限。ただし分母が小さいので検出力に注意)"
                       if ns else "判定できない(どちらも 0 件)")
        else:
            fac = round(rs / ra, 3)
            verdict = "合格" if fac >= NC_FACTOR else "不合格(2.0 倍を下回る)"
        d["nc_k2"] = dict(factor=fac, bar=NC_FACTOR,
                          across_rate_upper95=round(up, 5),
                          # 厳しい版(登録より厳しいので、緩めたことにはならない)
                          separation_established=bool(up > 0 and rs / 2.0 >= up),
                          fisher_p_one_sided=float(f"{pf:.3e}") if da else float("nan"),
                          verdict=verdict,
                          passed=bool(verdict.startswith("合格")))
        res[b] = d

    # A2: 箱 B の墨画素数の分布(0 の山と非 0 の山が分かれるか)
    Bp = [x for x in box("B") if x["same"] and x["passed"]]
    hist = lambda v: {str(k): int(sum(1 for x in v if x == k)) for k in range(0, 6)} | \
                     {"6-10": int(sum(1 for x in v if 6 <= x <= 10)),
                      "11-20": int(sum(1 for x in v if 11 <= x <= 20)),
                      "21+": int(sum(1 for x in v if x > 20)), "n": len(v)}
    fr = [x["ink_on_seg"] / x["seg_n"] for x in Bp if x["ink_on_seg"] >= 0 and x["seg_n"] > 0]
    frac = {f"{lo:.1f}-{lo+0.1:.1f}": int(sum(1 for v in fr if lo <= v < lo + 0.1 + (1e-9 if lo >= 0.9 else 0)))
            for lo in [i / 10 for i in range(10)]}
    inkd = dict(fraction_of_segment=frac | {"n": len(fr)},
                seg_n=dict(median=float(np.median([x["seg_n"] for x in Bp if x["seg_n"] > 0])) if Bp else None),
                grow200=hist([x["ink_on_seg"] for x in Bp if x["ink_on_seg"] >= 0]),
                plain=hist([x["ink_on_seg_plain"] for x in Bp if x["ink_on_seg_plain"] >= 0]),
                note="門を通った同部位の箱 B 候補の、襟で切った線分上の墨画素数。"
                     "0 の山と非 0 の山が分かれなければ襟の取り方が悪い(A2)")

    diag = dict(n_pairs=len(rows), n_att=sum(o[1]["n_att"] for o in byp.values() if o),
                n_two_on_one_tip=sum(o[1]["n_two_on_one_tip"] for o in byp.values() if o),
                n_att_no_path=sum(o[1]["n_att_no_path"] for o in byp.values() if o),
                n_free_tips=sum(o[1]["n_free_tips"] for o in byp.values() if o),
                n_node_tips=sum(o[1]["n_node_tips"] for o in byp.values() if o))
    summary = dict(registered=chk["registered"],
                   population=dict(n_faces=len({x["order"] for x in rows}), n_faces_read=len(recs),
                                   n_faces_with_broken=n_broken, n_panels=len(pids)),
                   note="候補の数である。『取り逃がし率』ではない。箱は合算しない(A4)。"
                        "条件 (d) が効くのは箱 A だけ(A6-4)",
                   boxes=res, ink_on_segment=inkd, diagnostics=diag)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
