#!/usr/bin/env python
"""K2-B'(2026-10-18 事前登録。基盤 `inbox/reply_k2_k4_k5_20261018.md` の 1 の形):

箱 B の 182 件を、**2 値の墨数ではなく灰値のプロファイル**で 3 分類する。
`ink_mask(thr=128, grow=200)` は灰 200 以上を墨にしないので、**紙(250〜255)と
かすれ(200〜250)がどちらも「墨 0」に入る**。飛ばし線とかすれの境目がそこに落ちていた。

**停止条件は素材の形ではなく器具について書く**(基盤の訂正。素材の分布が二山かどうかは
我々が決められる量ではない)。構成した標的 T1/T2/T3 が**書いた規則で正しい箱に入ること**。
1 つでも外れたら止める。素材の分布の形は**印字するだけ**で合否に使わない。

母集団は動かさない: `k2_missed_candidates_20261017/per_pair.csv` の `box=B,same=1,passed=1`
= 182 件。襟も `collar` 列の値をそのまま使う。
"""
import argparse, csv, hashlib, json, subprocess, sys
from collections import Counter, defaultdict
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

THR, GROW, MAX_BEND = 128, 200, 35.0
G_PAPER, G_INK = 240, 128          # 登録した閾値(紙 / かすれ / 墨)
ANGLE_MAX, D_MAX_K = 30.0, 3.0     # 門(K の登録値)
BG_MODE_MIN = 245                  # 前提の検査: 背景の最頻灰
WIN, BLOB_D, CROP = 14.0, 5, 512   # 標的の窓長・塊の直径・切り出し
N_TARGET, SEED = 30, 20261018
N_SITE_POOL = 300                  # B'-3: 構成が 30 件に達するまで site を足す
KINDS = ("T1", "T2", "T3", "T4touch", "T4sep")
WANT = {"T1": "紙", "T2": "かすれ", "T3": "墨", "T4touch": "墨", "T4sep": "墨"}
N_PER_KIND = {"T1": 30, "T2": 30, "T3": 30, "T4touch": 30, "T4sep": 30}
T4_KINDS = ("T4touch", "T4sep")
SEP_LEN, SEP_MARGIN = 14.0, 2.0     # T4sep: 第三の線の長さと、分離を確かめる余白
CLEAN_R, MIN_SPAN = 6.0, 40        # 標的に使う span の選び方
J_SHA = "53b8406cfbb4f4d1e4c53a15102ec8079a4f541e77e0523b6836af9d5a9a56a6"


def seg_points(a, b, step=1.0):
    n = max(int(np.ceil(np.hypot(*(b - a)) / step)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def klass(gmin):
    return "紙" if gmin > G_PAPER else ("かすれ" if gmin > G_INK else "墨")


def pipeline(g):
    """作者の経路(IC-k5b)をそのまま回し、span / token / tip / 骨格木を返す。"""
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths, order_path
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    iw = dist * 2.0
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(skeletonize(ink), iw, max_bend=MAX_BEND)
    gr = defaultdict(list)
    for i in range(len(spans)):
        gr[int(sos[i])].append(i)
    span_tok, ntok = {}, 0
    for k, mem in gr.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        for i in mem:
            span_tok[i] = ntok
        ntok += 1
    paths, tips = {}, {}
    for i in range(len(spans)):
        rr, cc = spans[i]
        if len(rr) == 0:
            continue
        p = order_path(np.stack([rr, cc], 1)).astype(np.float64)
        if len(p) < 2:
            continue
        paths[i] = p
        tips[i] = (p[0], p[-1])
    sk_pts, sk_tok = [], []
    for i, (rr, cc) in enumerate(spans):
        t = span_tok.get(i, -1)
        if t < 0 or len(rr) == 0:
            continue
        sk_pts.append(np.stack([rr, cc], 1).astype(np.float64))
        sk_tok.append(np.full(len(rr), t, np.int32))
    sk_pts = np.concatenate(sk_pts) if sk_pts else np.zeros((0, 2))
    sk_tok = np.concatenate(sk_tok) if len(sk_pts) else np.zeros(0, np.int32)
    sp_pts, sp_idx = [], []                     # 全 span(token に入らないものも)
    for i, (rr, cc) in enumerate(spans):
        if len(rr) == 0 or i not in tips:
            continue
        sp_pts.append(np.stack([rr, cc], 1).astype(np.float64))
        sp_idx.append(np.full(len(rr), i, np.int32))
    sp_pts = np.concatenate(sp_pts) if sp_pts else np.zeros((0, 2))
    sp_idx = np.concatenate(sp_idx) if len(sp_pts) else np.zeros(0, np.int32)
    ncc, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    return dict(ink=ink, iw=iw, lab=lab, spans=spans, span_tok=span_tok, paths=paths, tips=tips,
                sk_pts=sk_pts, sk_tok=sk_tok, sp_idx=sp_idx,
                sk_tree=cKDTree(sk_pts) if len(sk_pts) else None,
                sp_tree=cKDTree(sp_pts) if len(sp_pts) else None)


def comp_at(P, q):
    """点 q の連結成分のラベル。ink の外なら 0。tip が 1px ずれている場合に備えて 3x3 を見る。"""
    H, W = P["lab"].shape
    r, c = int(round(q[0])), int(round(q[1]))
    r, c = int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))
    if P["lab"][r, c]:
        return int(P["lab"][r, c])
    sub = P["lab"][max(0, r - 1):r + 2, max(0, c - 1):c + 2]
    nz = sub[sub > 0]
    return int(np.bincount(nz).argmax()) if len(nz) else 0


def comp_collar(P, pa, pb, step=0.5):
    """C-1: tip から弦に沿って進み、**自分の連結成分を出た点**から測り始める。
    戻り値 (a2, b2, t_a, t_b, reason)。測れないときは a2 が None で reason に理由。
    `step` は標本の刻みであって判定の閾値ではない。"""
    v = pb - pa
    L = float(np.hypot(*v))
    if L < 1e-6:
        return None, None, -1.0, -1.0, "弦の長さが 0"
    u = v / L
    la, lb = comp_at(P, pa), comp_at(P, pb)
    t_a = t_b = None
    t = step
    while t <= L:
        if t_a is None and comp_at(P, pa + u * t) != la:
            t_a = t
        if t_b is None and comp_at(P, pb - u * t) != lb:
            t_b = t
        if t_a is not None and t_b is not None:
            break
        t += step
    if t_a is None or t_b is None:
        return None, None, -1.0, -1.0, "自分の連結成分を出ないまま他方の tip に達した"
    if L - t_b <= t_a:
        return None, None, float(t_a), float(t_b), "両端が出た後の区間が空"
    return pa + u * t_a, pa + u * (L - t_b), float(t_a), float(t_b), ""


def profile(g, pa, pb, col):
    """襟で切った線分の灰値。標本 0 個なら切る前の中点 1 点(登録どおり別に数える)。"""
    H, W = g.shape
    v = pb - pa
    L = float(np.hypot(*v))
    if L > 2 * col + 1e-6:
        u = v / L
        sp = seg_points(pa + u * col, pb - u * col)
        mid_only = 0
    else:
        sp = ((pa + pb) / 2.0)[None, :]
        mid_only = 1
    rr = np.clip(np.round(sp[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(sp[:, 1]).astype(int), 0, W - 1)
    return g[rr, cc].astype(int), sp, mid_only


def own_ink_mask(P, sp, vals, i, j):
    """B'-1b: 標本ごとに『自分の墨』か判定する。
    半径 max(2.0, iw[最近傍の自分の骨格画素]/2 + 1.0) 以内に候補の 2 span の骨格があり、
    **かつ他のどの span の骨格も無い**ときだけ『自分の墨』。帰属不明は『他の墨』(除外しない)。

    半径の素は **標本点の `iw` ではなく、最も近い「自分の骨格画素」の `iw`**(B'-1b、
    2026-10-18 登録)。根拠は定義: 画素がその線の body に属するかはその線の太さで決まり、
    線の太さは線の性質であって画素の位置の性質ではない。`iw` は距離変換の 2 倍なので
    縁で 0〜2 に潰れ、弦が掠めるのは必ず縁 → 半径が最小になる場所が半径を最も要る場所
    という向きの逆転が起きていた。最近傍 40 画素に自分の骨格が無ければ半径内にも無いので
    『自分の墨』にはしない。"""
    own = np.zeros(len(vals), bool)
    if P["sp_tree"] is None:
        return own
    H, W = P["iw"].shape
    kq = min(40, len(P["sp_idx"]))
    for k in range(len(vals)):
        if vals[k] > G_PAPER:
            continue
        q = sp[k]
        _dd, mm = P["sp_tree"].query(q, k=kq)
        w = None
        for m_ in np.atleast_1d(mm):
            if int(P["sp_idx"][m_]) in (i, j):
                y, x = P["sp_tree"].data[m_]
                w = float(P["iw"][int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
                break
        if w is None:                      # 最近傍 40 画素に自分の骨格が無い
            continue
        r = max(2.0, w / 2.0 + 1.0)
        idx = P["sp_tree"].query_ball_point(q, r)
        if not idx:
            continue
        near = {int(P["sp_idx"][m]) for m in idx}
        own[k] = near.issubset({i, j})
    return own


def measure(g, P, i, ei, j, ej, col, ti, tj):
    """1 対の灰値プロファイルと分類、および記録用の列。
    襟は C-1(自分の連結成分を出た点から測る)。自分の墨の除外は廃止し、
    旧規則が除外したはずの標本数だけ `n_would_exclude` に印字する。"""
    pa, pb = P["tips"][i][ei], P["tips"][j][ej]
    a2, b2, t_a, t_b, why = comp_collar(P, pa, pb)
    if a2 is None:
        return dict(cls=None, cls_raw=None, untestable=why, g_min=-1, g_min_excl=-1,
                    n_own_dark=-1, n_would_exclude=-1, d_own=-1.0, d_other=-1.0,
                    g_med=-1, g_max=-1, frac_gt240=-1.0, seg_n=0, midpoint_only=0,
                    crossed=-1, near_other=-1, cls_second=None, g_second=-1,
                    gap=round(float(np.hypot(*(pb - pa))), 3), t_a=t_a, t_b=t_b,
                    comp_same=int(comp_at(P, pa) == comp_at(P, pb)))
    H, W = g.shape
    sp = seg_points(a2, b2)
    rr = np.clip(np.round(sp[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(sp[:, 1]).astype(int), 0, W - 1)
    vals = g[rr, cc].astype(int)
    mid_only = 0
    own = own_ink_mask(P, sp, vals, i, j)          # 分類には使わない。印字のみ
    free = np.ones(len(vals), bool)
    gmin_excl = int(vals.min())
    k = int(np.argmin(vals))
    gmin = int(vals.min())
    cls = klass(gmin)
    # 墨の下位区分と、かすれの near_other_token(いずれも記録のみ)
    crossed = 0
    near_other = 0
    if P["sk_tree"] is not None:
        for idx in P["sk_tree"].query_ball_point(seg_points(pa, pb, 2.0), col):
            if any(P["sk_tok"][m] not in (ti, tj) for m in idx):
                crossed = 1
                break
        idx = P["sk_tree"].query_ball_point(sp[k], col)
        near_other = int(any(P["sk_tok"][m] not in (ti, tj) for m in idx))
    fv = np.sort(vals)
    second = int(fv[1]) if len(fv) > 1 else int(fv[0])
    # 最暗点(除外後)から、自分の span / 他の span までの距離(罠が発火したか見るため)
    d_own = d_other = -1.0
    if P["sp_tree"] is not None:
        dd, mm = P["sp_tree"].query(sp[k], k=min(40, len(P["sp_idx"])))
        for dist_, m_ in zip(np.atleast_1d(dd), np.atleast_1d(mm)):
            sid = int(P["sp_idx"][m_])
            if sid in (i, j) and d_own < 0:
                d_own = float(dist_)
            if sid not in (i, j) and d_other < 0:
                d_other = float(dist_)
    return dict(g_min=gmin, g_min_excl=gmin_excl, n_own_dark=int(own.sum()),
                n_would_exclude=int(own.sum()), untestable="",
                t_a=round(t_a, 2), t_b=round(t_b, 2),
                comp_same=int(comp_at(P, pa) == comp_at(P, pb)),
                cls_raw=klass(gmin), d_own=round(d_own, 2), d_other=round(d_other, 2),
                g_med=int(np.median(vals)), g_max=int(vals.max()),
                frac_gt240=round(float((vals > G_PAPER).mean()), 4),
                seg_n=len(vals), midpoint_only=mid_only, cls=cls,
                cls_second=klass(second), g_second=second,
                crossed=crossed, near_other=near_other,
                gap=round(float(np.hypot(*(pb - pa))), 3))


# ---------------- 素材(箱 B の 182 件) ----------------

def work_material(t):
    pid, source, name, rows = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None, None
    P = pipeline(g)
    bg = int(np.argmax(np.bincount(g.ravel(), minlength=256)))
    out = []
    for r in rows:
        i, j = int(r["s0"]), int(r["s1"])
        ei, ej = int(r["e0"]), int(r["e1"])
        if i not in P["tips"] or j not in P["tips"]:
            out.append(dict(order=r["order"], panel=pid, s0=i, s1=j, missing=1,
                            t0=int(r["t0"]), t1=int(r["t1"])))
            continue
        ti, tj = int(r["t0"]), int(r["t1"])
        d = measure(g, P, i, ei, j, ej, float(r["collar"]), ti, tj)
        d.update(order=r["order"], panel=pid, s0=i, s1=j, e0=ei, e1=ej, missing=0,
                 t0=ti, t1=tj,
                 gap_csv=float(r["gap"]), ink_csv=int(r["ink_on_seg"]),
                 crossed_csv=int(r["crossed"]))
        out.append(d)
    # 標的に使える site(改変前に窓の周りが空いている span の中央)
    sites = []
    for i, p in P["paths"].items():
        if len(p) < MIN_SPAN or P["span_tok"].get(i, -1) < 0:
            continue
        mid = len(p) // 2
        c = p[mid]
        if P["sk_tree"] is not None:
            idx = P["sk_tree"].query_ball_point(c, CLEAN_R)
            if any(P["sk_tok"][m] != P["span_tok"][i] for m in idx):
                continue
        H, W = g.shape
        if not (CROP // 2 + 4 < c[0] < H - CROP // 2 - 4 and CROP // 2 + 4 < c[1] < W - CROP // 2 - 4):
            continue
        sites.append((pid, i, mid))
    return pid, out, dict(bg_mode=bg, sites=sites)


# ---------------- 構成した標的(T1/T2/T3) ----------------

def build_target(g, p, mid, iw, kind):
    """窓を塗る。戻り値は (改変した画像, 窓の両端の点)。"""
    out = g.copy()
    s = e = mid
    while s > 0 and np.hypot(*(p[mid] - p[s - 1])) < WIN / 2:
        s -= 1
    while e < len(p) - 1 and np.hypot(*(p[mid] - p[e + 1])) < WIN / 2:
        e += 1
    fill = 220 if kind == "T2" else 255
    for q in p[s:e + 1]:
        rad = int(round(float(iw[int(q[0]), int(q[1])]) / 2.0 + 1.5))
        cv2.circle(out, (int(round(q[1])), int(round(q[0]))), max(rad, 1), int(fill), -1)
    if kind == "T3":
        cv2.circle(out, (int(round(p[mid][1])), int(round(p[mid][0]))), BLOB_D // 2, 0, -1)
    if kind in T4_KINDS:
        wa, wb = p[s], p[e]
        v = wb - wa
        L = float(np.hypot(*v))
        if L < 1e-6:
            return out, (wa, wb), (s, e)
        u = v / L
        c = wa + u * (L * 0.5)                      # どちらも弦の中点で交わる
        th = np.deg2rad(60.0)
        d = np.array([u[0] * np.cos(th) - u[1] * np.sin(th),
                      u[0] * np.sin(th) + u[1] * np.cos(th)])
        w0 = float(iw[int(p[mid][0]), int(p[mid][1])])
        if kind == "T4touch":                       # 長い: 候補の墨まで届いて 8 近傍で繋がる
            half, wdt = L, max(int(round(2.0 * w0)), 3)
        else:                                       # 短い: 白く塗った窓の中だけ(分離を狙う)
            half, wdt = SEP_LEN / 2.0, max(int(round(w0)), 3)
        a2, b2 = c - d * half, c + d * half
        cv2.line(out, (int(round(a2[1])), int(round(a2[0]))),
                 (int(round(b2[1])), int(round(b2[0]))), 0, wdt)
    return out, (p[s], p[e]), (s, e)


def work_target(t):
    pid, source, name, sites = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return []
    P0 = pipeline(g)                      # 改変前(iw と経路を取るため)
    res = []
    for rank, (_pid, i, mid) in sites:
        if i not in P0["paths"]:
            continue
        p = P0["paths"][i]
        c = p[mid]
        r0 = int(c[0]) - CROP // 2
        c0 = int(c[1]) - CROP // 2
        for kind in KINDS:
            mod, (wa, wb), (si, ei_) = build_target(g, p, mid, P0["iw"], kind)
            crop = mod[r0:r0 + CROP, c0:c0 + CROP]
            Pc = pipeline(crop)
            off = np.array([r0, c0], float)
            # 由来で同定する: 窓の外側(塗った円の外)の原経路の点から、改変後の span を引く
            best, anch = {}, {}
            for tag, (end_k, step) in ((0, (si, -1)), (1, (ei_, +1))):
                rad = float(P0["iw"][int(p[end_k][0]), int(p[end_k][1])]) / 2.0 + 1.5
                k = end_k
                while 0 <= k + step < len(p) and np.hypot(*(p[k] - p[end_k])) < rad + 2.0:
                    k += step
                anchor = p[k] - off
                anch[tag] = (anchor, p[end_k] - off)
                sp = None
                if Pc["sp_tree"] is not None:
                    idx = Pc["sp_tree"].query_ball_point(anchor, 3.0)
                    if idx:
                        dd = [(float(np.hypot(*(Pc["sp_tree"].data[m] - anchor))), int(Pc["sp_idx"][m]))
                              for m in idx]
                        sp = min(dd)[1]
                if sp is None:
                    best[tag] = None
                    continue
                ta, tb = Pc["tips"][sp]
                we = p[end_k] - off
                e_ = 0 if np.hypot(*(ta - we)) <= np.hypot(*(tb - we)) else 1
                best[tag] = (sp, e_, float(np.hypot(*(Pc["tips"][sp][e_] - we))))
            ka = best.get(0); kb = best.get(1)
            row = dict(panel=pid, span=i, kind=kind, site_order=rank,
                       d_tip0=round(ka[2], 2) if ka else -1.0,
                       d_tip1=round(kb[2], 2) if kb else -1.0)
            if ka is None or kb is None:
                row.update(constructed=0, reason="窓の外側に span が残っていない(剪定された)")
                res.append(row); continue
            if ka[0] == kb[0]:
                row.update(constructed=0, reason="両側が同じ span(隙間が出来ていない)")
                res.append(row); continue
            import stroke_chain as sc
            ai = sc.arm_geometry(Pc["paths"][ka[0]], ka[1])
            aj = sc.arm_geometry(Pc["paths"][kb[0]], kb[1])
            if ai[1] is None or aj[1] is None:
                row.update(constructed=0, reason="接線が取れない"); res.append(row); continue
            dot = float(np.clip(ai[1] @ (-aj[1]), -1.0, 1.0))
            th = float(np.degrees(np.arccos(dot)))
            dp = 0.5 * (sc.perp_offset(ai[3], aj[0]) + sc.perp_offset(aj[3], ai[0]))
            if not (th <= ANGLE_MAX and dp <= D_MAX_K):
                row.update(constructed=0, reason=f"門を通らない(角{th:.1f} 横{dp:.2f})")
                res.append(row); continue
            H2, W2 = crop.shape
            pa, pb = Pc["tips"][ka[0]][ka[1]], Pc["tips"][kb[0]][kb[1]]
            wa_ = float(Pc["iw"][int(np.clip(pa[0], 0, H2 - 1)), int(np.clip(pa[1], 0, W2 - 1))])
            wb_ = float(Pc["iw"][int(np.clip(pb[0], 0, H2 - 1)), int(np.clip(pb[1], 0, W2 - 1))])
            col = max(2.0, 0.5 * max(wa_, wb_))
            d = measure(crop, Pc, ka[0], ka[1], kb[0], kb[1], col,
                        Pc["span_tok"].get(ka[0], -1), Pc["span_tok"].get(kb[0], -1))
            blob_is_token = 0
            if kind == "T3" and Pc["sk_tree"] is not None:
                idx = Pc["sk_tree"].query_ball_point(np.array([CROP / 2.0, CROP / 2.0]), 4.0)
                blob_is_token = int(any(Pc["sk_tok"][m] not in
                                        (Pc["span_tok"].get(ka[0], -1), Pc["span_tok"].get(kb[0], -1))
                                        for m in idx))
            want = WANT[kind]
            # B'-2: 構成後の弦の全長 + 3px の帯に、候補の 2 span と(T4 の)第三の線以外が無いこと
            allow = {ka[0], kb[0]}
            if kind in T4_KINDS:
                cc_ = (pa + pb) * 0.5
                nb_ = Pc["sp_tree"].query_ball_point(cc_, 4.0) if Pc["sp_tree"] is not None else []
                allow |= {int(Pc["sp_idx"][m]) for m in nb_}
            dirty = set()
            if Pc["sp_tree"] is not None:
                for mlist in Pc["sp_tree"].query_ball_point(seg_points(pa, pb, 1.0), 3.0):
                    dirty |= {int(Pc["sp_idx"][m]) for m in mlist}
            dirty -= allow
            if dirty:
                row.update(constructed=0, reason="弦の帯に他の span がいる(B'-2)")
                res.append(row); continue
            # C-2: 第三の線が候補と 8 近傍で連結しているかを**実測**して振り分ける。
            # 描き方で決めつけない。狙いと実測が食い違う構成は「構成できず」に数える。
            if kind in T4_KINDS:
                lc = comp_at(Pc, (pa + pb) * 0.5)
                la, lb = comp_at(Pc, pa), comp_at(Pc, pb)
                touching = bool(lc and (lc == la or lc == lb))
                row["third_touches"] = int(touching)
                if kind == "T4touch" and not touching:
                    row.update(constructed=0, reason="第三の線が候補に触れていない(touch 不成立)")
                    res.append(row); continue
                if kind == "T4sep" and touching:
                    row.update(constructed=0, reason="第三の線が候補に触れた(sep 不成立)")
                    res.append(row); continue
            got = d["cls"] if d["cls"] is not None else f"測れない:{d['untestable']}"
            row.update(constructed=1, theta=round(th, 2), dperp=round(dp, 3),
                       want=want, got=got, ok=int(d["cls"] == want),
                       untestable_target=int(d["cls"] is None),
                       blob_is_token=blob_is_token,
                       **{k_: d[k_] for k_ in ("g_min", "g_min_excl", "n_own_dark",
                                               "n_would_exclude", "t_a", "t_b", "comp_same",
                                               "cls_raw", "d_own", "d_other", "g_med",
                                               "frac_gt240", "seg_n", "midpoint_only",
                                               "gap", "crossed")})
            res.append(row)
    return res


def hist_counter(c, keys):
    return {k: int(c.get(k, 0)) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pairs", default="results/k2_missed_candidates_20261017/per_pair.csv")
    ap.add_argument("--snapshot", default="results/k1_broken_vs_fill_20261016/labels_snapshot.jsonl")
    ap.add_argument("--out", default="results/k2_box_b_gray_20261018")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--band", choices=["a", "b", "c"], default="c",
                    help="a,b=診断の帯(置換の 0..299 / 300..599) / c=検査の帯(置換の 900..1199)")
    ap.add_argument("--pool-start", type=int, default=None, help="既定は帯から決まる")
    ap.add_argument("--pool-n", type=int, default=N_SITE_POOL)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256((TRACK_J / "tools/evaluation/stroke_chain.py").read_bytes()).hexdigest()
    j_head = subprocess.run(["git", "-C", str(TRACK_J), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"J の stroke_chain.py sha256 一致: {sha == J_SHA}  (J head {j_head})", flush=True)
    assert sha == J_SHA, "J の連鎖器が変わっている。止める。"

    rows = [r for r in csv.DictReader(open(ROOT / a.pairs))
            if r["box"] == "B" and r["same"] == "1" and r["passed"] == "1"]
    byp = defaultdict(list)
    for r in rows:
        byp[int(r["panel"])].append(r)
    lab = {}
    for l in open(ROOT / a.snapshot):
        d = json.loads(l)
        for gg in d.get("groups", []):
            for s in gg["strokes"]:
                lab[(int(d["order"]), int(s))] = gg["label"]

    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"], rs) for p, rs in byp.items()]
    print(f"素材: 箱 B の候補 {len(rows)} 件 / コマ {len(tasks)} 枚", flush=True)

    mat, bgs, sites = [], {}, []
    with Pool(a.workers) as pool:
        for pid, o, extra in pool.imap_unordered(work_material, tasks, chunksize=2):
            if o is None:
                print(f"  コマ {pid} 読めない", flush=True); continue
            mat += o
            bgs[pid] = extra["bg_mode"]
            sites += extra["sites"]
    for r in mat:
        o = int(r["order"])
        r["label"] = lab.get((o, int(r.get("t0", -1)))) or "(無)"

    # --- 内部検査(登録外。span 番号が前回と一致しているかの確認) ---
    bad_gap = [r for r in mat if not r["missing"] and abs(r["gap"] - r["gap_csv"]) > 0.01]
    bad_cross = [r for r in mat if not r["missing"] and r["crossed_csv"] >= 0
                 and r["crossed"] != r["crossed_csv"]]
    low_bg = {k: v for k, v in bgs.items() if v <= BG_MODE_MIN}
    chk = dict(n_rows=len(mat), n_missing=sum(r["missing"] for r in mat),
               gap_mismatch=len(bad_gap), crossed_mismatch=len(bad_cross),
               bg_mode_min=min(bgs.values()) if bgs else None,
               panels_bg_at_or_below_245=low_bg,
               premise_ok=bool(not low_bg and not bad_gap))
    print("内部検査:", json.dumps(chk, ensure_ascii=False), flush=True)

    # --- 構成した標的 ---
    rng = np.random.default_rng(SEED)
    sites.sort()
    perm = rng.permutation(len(sites)) if len(sites) else np.zeros(0, int)
    # 帯は **1 本の置換の連続した区間**で割る(独立な第 2 seed では帯が重なるため)。
    # 帯(a)=0..299(診断、検査ではない) / 帯(b)=300..599(検査)。素集合は構成として保証される。
    start = a.pool_start if a.pool_start is not None else \
        {"a": 0, "b": N_SITE_POOL, "c": 3 * N_SITE_POOL}[a.band]
    order = perm[start:start + a.pool_n]
    pick = [(rank, sites[k]) for rank, k in enumerate(order, start=start)]
    bys = defaultdict(list)
    for rank, s_ in pick:
        bys[s_[0]].append((rank, s_))
    ttasks = [(p, P[p]["source"], P[p]["name"], ss) for p, ss in bys.items()]
    band_role = ("検査(棒はこの帯で判定)" if a.band == "c" else "診断(検査ではない)")
    print(f"帯({a.band}) {band_role}: 置換の {start}..{start + len(order) - 1} 番目、"
          f"site {len(sites)} 候補から {len(pick)} 個を試し、構成できた順に "
          f"{N_PER_KIND} を採る(B'-3)", flush=True)
    tg = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(work_target, ttasks, chunksize=1):
            tg += r

    with open(out / "per_pair_gray.csv", "w", newline="") as fh:
        fn = ["panel", "order", "t0", "t1", "s0", "s1", "e0", "e1", "missing", "label", "cls",
              "untestable", "t_a", "t_b", "comp_same", "n_would_exclude",
              "cls_raw", "g_min", "g_min_excl", "n_own_dark", "d_own", "d_other",
              "g_med", "g_max", "frac_gt240", "seg_n", "midpoint_only", "crossed", "near_other",
              "cls_second", "g_second", "gap", "gap_csv", "ink_csv", "crossed_csv"]
        w = csv.DictWriter(fh, fieldnames=fn, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(mat)
    with open(out / "targets.csv", "w", newline="") as fh:
        fn = ["panel", "span", "kind", "site_order", "constructed", "reason", "want", "got", "ok",
              "untestable_target", "third_touches", "t_a", "t_b", "comp_same", "n_would_exclude",
              "g_min", "g_min_excl", "cls_raw", "n_own_dark", "d_own", "d_other",
              "g_med", "frac_gt240", "seg_n", "midpoint_only", "gap", "theta", "dperp",
              "crossed", "blob_is_token", "d_tip0", "d_tip1"]
        w = csv.DictWriter(fh, fieldnames=fn, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(tg)

    # --- 停止条件の判定(器具について) ---
    tsum, selected = {}, {}
    for kind in KINDS:
        k = [r for r in tg if r["kind"] == kind]
        c = sorted([r for r in k if r.get("constructed")], key=lambda r: r["site_order"])
        need = N_PER_KIND[kind]
        sel = c[:need]
        selected[kind] = sel
        okn = sum(r["ok"] for r in sel)
        tsum[kind] = dict(n_sites_tried=len(k), n_constructed=len(c), need=need,
                          n_selected=len(sel), n_ok=okn, n_wrong=len(sel) - okn,
                          short=max(0, need - len(sel)),
                          wrong=[dict(panel=r["panel"], span=r["span"], got=r["got"],
                                      g_min=r["g_min"], g_min_excl=r["g_min_excl"],
                                      n_own_dark=r["n_own_dark"], d_own=r["d_own"],
                                      d_other=r["d_other"], seg_n=r["seg_n"])
                                 for r in sel if not r["ok"]],
                          not_constructed=dict(Counter(
                              (r.get("reason", "") or "")[:28] for r in k if not r.get("constructed"))),
                          n_untestable=sum(r.get("untestable_target", 0) for r in sel),
                          blob_is_token=sum(r.get("blob_is_token", 0) for r in sel) if kind == "T3" else None,
                          third_touches=sum(r.get("third_touches", 0) for r in sel)
                          if kind in T4_KINDS else None)
    # C-2: T4 は touch / sep を**合算しない**。5 行そのまま。
    rows5 = {k: tsum[k] for k in KINDS}
    stop_ok = all(v["n_selected"] >= v["need"] and v["n_wrong"] == 0 for v in rows5.values())

    # --- 素材の記述(合否に使わない) ---
    rows_all = [r for r in mat if not r["missing"]]
    # C-3: 測れない箱は独立。紙にも墨にも足さない。
    untest = [r for r in rows_all if r.get("untestable")]
    ok_rows = [r for r in rows_all if not r.get("untestable")]
    trim = ok_rows
    midp = []
    def describe(sel):
        c = Counter(r["cls"] for r in sel)
        oth = Counter(r["cls"] for r in sel if r["label"] == "face.other")
        mou = Counter(r["cls"] for r in sel if r["label"] == "face.mouth")
        gm = [r["g_min"] for r in sel]
        return dict(n=len(sel), cls=hist_counter(c, ["紙", "かすれ", "墨"]),
                    cls_of_face_other=hist_counter(oth, ["紙", "かすれ", "墨"]),
                    cls_of_face_mouth=hist_counter(mou, ["紙", "かすれ", "墨"]),
                    g_min_hist={k: int(v) for k, v in sorted(Counter(
                        ("<=128" if x <= 128 else "129-199" if x < 200 else "200-240" if x <= 240
                         else "241-249" if x < 250 else "250-254" if x < 255 else "255")
                        for x in gm).items())},
                    changed_by_second_min=sum(1 for r in sel if r["cls"] != r["cls_second"]),
                    cls_raw=hist_counter(Counter(r["cls_raw"] for r in sel), ["紙", "かすれ", "墨"]),
                    moved_by_B1=sum(1 for r in sel if r["cls"] != r["cls_raw"]),
                    moved_by_B1_detail=dict(Counter(f"{r['cls_raw']}→{r['cls']}"
                                                    for r in sel if r["cls"] != r["cls_raw"])),
                    ink0_csv_of_paper=sum(1 for r in sel if r["cls"] == "紙" and r["ink_csv"] == 0),
                    paper_of_ink0_csv=sum(1 for r in sel if r["ink_csv"] == 0 and r["cls"] == "紙"),
                    n_ink0_csv=sum(1 for r in sel if r["ink_csv"] == 0))
    desc = dict(trimmed=describe(trim), midpoint_only=describe(midp))
    desc["untestable_box"] = dict(
        n=len(untest), n_measured=len(ok_rows), n_total=len(rows_all),
        reasons=dict(Counter(r["untestable"] for r in untest)),
        label=dict(Counter(r["label"] for r in untest).most_common(6)),
        note="紙にも墨にも足していない(C-3)")
    sub = Counter((r["cls"], r["crossed"], r["near_other"]) for r in ok_rows)
    desc["subsplit"] = {f"{k[0]}/crossed={k[1]}/near_other={k[2]}": int(v) for k, v in sorted(
        sub.items(), key=lambda t: (-t[1],))}
    lab_tab = defaultdict(Counter)
    for r in ok_rows:
        lab_tab[r["cls"]][r["label"]] += 1
    desc["label_by_class"] = {k: dict(v.most_common(6)) for k, v in lab_tab.items()}

    summary = dict(
        registered=dict(g_paper=G_PAPER, g_ink=G_INK, win=WIN, blob_d=BLOB_D, crop=CROP,
                        n_target=N_TARGET, seed=SEED, clean_r=CLEAN_R, min_span=MIN_SPAN,
                        collar="per_pair.csv の collar 列をそのまま(襟は動かさない)",
                        collar_rule="C-1: tip から弦に沿って進み、自分の 8 近傍連結成分を"
                                    "出た点から測る。自分の墨の除外(B'-1/B'-1b)は廃止",
                        band=a.band, band_role=band_role,
                        band_slice=[int(start), int(start + len(order))],
                        note="停止条件は構成した標的のみ。素材の分布の形は合否に使わない。"
                             "帯(a) は診断で検査ではない。棒の判定は帯(b) のみ。2 帯を合算しない"),
        instrument=dict(targets_5rows=rows5, stop_condition_passed=stop_ok, internal=chk,
                        j_sha_ok=sha == J_SHA, j_head=j_head),
        material_description=desc,
        verdict=(f"診断の帯なので合否は出さない(帯 {a.band})" if a.band != "c" else
                 ("器具の停止条件に合格(5 行とも 30 件を構成し、全部正しい箱に入った)" if stop_ok
                  else "**止める**: 登録した「5 行とも各 30 件がすべて正しい箱に入る」を満たしていない")))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
