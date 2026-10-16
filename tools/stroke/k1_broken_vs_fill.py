#!/usr/bin/env python
"""K1(2026-10-16 事前登録): 過接続の印(Track H の `broken`)は、ベタ領域に隣接する線に偏るか。

閾値・合否・陰性対照は `doc/work_log.md` の「K1 事前登録」に、数値を1つも見る前に書いてある。
ここはそれをそのまま実行するだけで、判定の条件分岐は持たない。

読むだけの他 track: Track F の panel_pack(線)、Track J の fill_mask(ベタ)、Track H の印。
印のファイルは作者が書き込み中なので、複製を `--labels` で渡して固定する。

出力: summary.json / per_stroke.csv / per_face.csv / check.json
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
TRACK_J = Path("/home/sh1/deepl/lineart-solid-fill")
sys.path.insert(0, str(TRACK_H / "tools/face"))
sys.path.insert(0, str(TRACK_J / "tools/evaluation"))
import group_server as G                                              # noqa: E402
from fill_mask import fill_mask, ink_of, FILL_HALF_WIDTH_PX           # noqa: E402

# --- 事前登録した定数(doc/work_log.md 2026-10-16)。結果を見て動かさない ---
ADJ_PX = 4.0                      # = J の FILL_HALF_WIDTH_PX、native px
ADJ_FRAC = 0.25                   # 標本点のこの割合以上が近ければ「ベタ隣接」
SAMPLE_STEP = 1.0                 # 折れ線の標本化の刻み [px]
GRID_PX = (2.0, 4.0, 8.0, 16.0)   # 感度(副)
GRID_FRAC = (0.10, 0.25, 0.50)
N_NEG = 200                       # NC-k1 の引き直し回数
SEED = 20261016
RR_SUPPORT, P_SUPPORT, RR_REFUTE = 1.5, 0.01, 1.2
IC_K1_FACTOR = 3.0                # コマ全体の面積割合の中央値が J の何倍以内か
IC_K2_ONINK, IC_K2_TRANSPOSED = 0.85, 0.5
LOW_FILL_FRAC = 0.01              # (c) 行: 顔の窓のベタ面積割合がこれ未満の顔を除く


def densify(poly):
    """(16, 2) の折れ線を SAMPLE_STEP 刻みに標本化して (m, 2) で返す。"""
    seg = np.diff(poly, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    out = [poly[:1]]
    for i, L in enumerate(ln):
        if L < 1e-9:
            continue
        k = max(int(np.ceil(L / SAMPLE_STEP)), 1)
        t = np.arange(1, k + 1)[:, None] / k
        out.append(poly[i] + t * seg[i])
    return np.vstack(out), float(ln.sum())


def panel_fields(pid, radius=FILL_HALF_WIDTH_PX):
    """コマ 1 枚ぶん: ベタまでの距離、インク、面積割合。"""
    bgr = G.panel_img(pid)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    ink = ink_of(gray)
    fill = fill_mask(ink, radius=radius)
    dist = distance_transform_edt(~fill) if fill.any() else np.full(fill.shape, np.inf)
    return dist, ink, float(fill.mean()), float(ink.mean())


def sample_rows(recs, out_csv, radius=FILL_HALF_WIDTH_PX):
    """線ごとに 1 行。印の有無・ベタ距離の分布・長さ・位置合わせの確認を一度に取る。"""
    rows, faces = [], []
    cache_pid, cache = None, None
    for n, r in enumerate(recs, 1):
        pid = r["panel"]
        if pid != cache_pid:
            cache, cache_pid = panel_fields(pid, radius), pid
        dist, ink, panel_fill_frac, panel_ink_frac = cache
        H, W = ink.shape
        pl = G.face_payload(r["order"])
        broken = set(r.get("broken", []))
        x0, y0, x1, y1 = pl["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = G.PICK_K * (x1 - x0) / 2, G.PICK_K * (y1 - y0) / 2
        wx0, wy0 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
        wx1, wy1 = int(min(cx + hw, W)), int(min(cy + hh, H))
        win = np.s_[wy0:wy1, wx0:wx1]
        fill_win = (dist[win] == 0)
        ids = [s["id"] for s in pl["strokes"]]
        stray = sorted(broken - set(ids))
        onink, transposed = [], []
        for s in pl["strokes"]:
            p = np.asarray(s["p"], np.float64).reshape(16, 2)
            pts, length = densify(p)
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            d = dist[yi, xi]
            oi = float(ink[yi, xi].mean())
            ti = float(ink[np.clip(xi, 0, H - 1), np.clip(yi, 0, W - 1)].mean())
            onink.append(oi)
            transposed.append(ti)
            rows.append(dict(order=r["order"], panel=pid, sid=s["id"], marked=int(s["id"] in broken),
                             n_pts=len(pts), length=round(length, 2), width=s["w"],
                             cluster=s["c"], csize=s["cs"], on_ink=round(oi, 4), on_ink_transposed=round(ti, 4),
                             **{f"f{int(t)}": round(float((d <= t).mean()), 4) for t in GRID_PX}))
        faces.append(dict(order=r["order"], panel=pid, n_strokes=len(ids), n_marks=len(broken & set(ids)),
                          n_stray=len(stray), face_fill_frac=round(float(fill_win.mean()), 5),
                          panel_fill_frac=round(panel_fill_frac, 5), panel_ink_frac=round(panel_ink_frac, 5),
                          on_ink_median=round(float(np.median(onink)), 4),
                          on_ink_transposed_median=round(float(np.median(transposed)), 4)))
        if n % 20 == 0:
            print(f"  {n}/{len(recs)} 顔", flush=True)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    return rows, faces


def two_by_two(marked, adj):
    a = int((marked & adj).sum()); b = int((marked & ~adj).sum())
    c = int((~marked & adj).sum()); d = int((~marked & ~adj).sum())
    p_m = a / (a + b) if a + b else float("nan")
    p_u = c / (c + d) if c + d else float("nan")
    rr = (p_m / p_u) if p_u else float("nan")
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return dict(marked_adj=a, marked_not=b, unmarked_adj=c, unmarked_not=d,
                p_adj_marked=round(p_m, 4), p_adj_unmarked=round(p_u, 4),
                rr=round(rr, 3), fisher_p=float(f"{p:.3e}"))


def mh_rr(marked, adj, length, q=5):
    """長さの 5 分位で層別した Mantel-Haenszel の RR。交絡の検査(副)。"""
    edges = np.quantile(length, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    num = den = 0.0
    strata = []
    for i in range(q):
        m = (length > edges[i]) & (length <= edges[i + 1])
        a = float((marked & adj & m).sum()); b = float((marked & ~adj & m).sum())
        c = float((~marked & adj & m).sum()); d = float((~marked & ~adj & m).sum())
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * (c + d) / n
        den += c * (a + b) / n
        strata.append(dict(stratum=i, lo=round(float(edges[i]) if i else float(length.min()), 1),
                           hi=round(float(edges[i + 1]) if i < q - 1 else float(length.max()), 1),
                           n_marked=int(a + b), n_unmarked=int(c + d),
                           p_adj_marked=round(a / (a + b), 4), p_adj_unmarked=round(c / (c + d), 4)))
    return (round(num / den, 3) if den else float("nan")), strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="固定した face_groups.jsonl の複製")
    ap.add_argument("--out", default="results/k1_broken_vs_fill_20261016")
    ap.add_argument("--limit", type=int, default=0, help="smoke test 用。先頭 n 顔だけ")
    ap.add_argument("--fill-radius", type=float, default=FILL_HALF_WIDTH_PX,
                    help="K1b(2026-10-16 宣言)。ベタマスクの半径だけを動かす。既定 = K1 の登録値 4.0")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    G.load_state(Path(a.labels))
    all_faces = list(G.S["faces"].values())
    train_done = [r for r in all_faces if r["usage"] == "train" and r["status"] == "done"]
    recs = sorted([r for r in train_done if "broken" in r], key=lambda r: (r["panel"], r["order"]))
    no_means = sorted(r["order"] for r in train_done if "broken" not in r)
    if a.limit:
        recs = recs[:a.limit]
    print(f"対象 {len(recs)} 顔 / 手段が無かった {len(no_means)} 顔 / コマ {len({r['panel'] for r in recs})} 枚", flush=True)

    rows, faces = sample_rows(recs, out / "per_stroke.csv", a.fill_radius)
    with open(out / "per_face.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(faces[0].keys())); wr.writeheader(); wr.writerows(faces)

    marked = np.array([r["marked"] for r in rows], bool)
    length = np.array([r["length"] for r in rows], float)
    order = np.array([r["order"] for r in rows], int)
    frac = {t: np.array([r[f"f{int(t)}"] for r in rows], float) for t in GRID_PX}
    adj = frac[ADJ_PX] >= ADJ_FRAC

    # --- 器具確認 ---
    jcsv = TRACK_J / "results/fill_koma_20261010/per_panel_held.csv"
    j_area = sorted({(r["panel"], float(r["gt_area_frac"])) for r in csv.DictReader(open(jcsv))})
    j_med = float(np.median([v for _, v in j_area]))
    my_panels = sorted({(f["panel"], f["panel_fill_frac"]) for f in faces})
    my_panel_med = float(np.median([v for _, v in my_panels]))
    ratio = my_panel_med / j_med if j_med else float("inf")
    ic1 = bool(1.0 / IC_K1_FACTOR <= ratio <= IC_K1_FACTOR)
    onink_med = float(np.median([f["on_ink_median"] for f in faces]))
    trans_med = float(np.median([f["on_ink_transposed_median"] for f in faces]))
    ic2 = bool(onink_med >= IC_K2_ONINK and trans_med <= IC_K2_TRANSPOSED)
    ic3 = sum(f["n_stray"] for f in faces)

    # --- NC-k1: 偽の印 ---
    rng = np.random.default_rng(SEED)
    idx_by_face, nmark_by_face = {}, {}
    for o in np.unique(order):
        sel = np.flatnonzero(order == o)
        idx_by_face[o] = sel
        nmark_by_face[o] = int(marked[sel].sum())
    neg_rr = []
    for _ in range(N_NEG):
        fake = np.zeros_like(marked)
        for o, sel in idx_by_face.items():
            k = nmark_by_face[o]
            if k:
                fake[rng.choice(sel, k, replace=False)] = True
        neg_rr.append(two_by_two(fake, adj)["rr"])
    neg_rr = np.array(neg_rr, float)
    lo, hi = float(np.percentile(neg_rr, 2.5)), float(np.percentile(neg_rr, 97.5))

    main_tab = two_by_two(marked, adj)
    outside = bool(main_tab["rr"] < lo or main_tab["rr"] > hi)
    mh, strata = mh_rr(marked, adj, length)

    # (c) ベタがほぼ無い顔を除く
    low = {f["order"] for f in faces if f["face_fill_frac"] < LOW_FILL_FRAC}
    keep = ~np.isin(order, list(low))
    tab_c = two_by_two(marked[keep], adj[keep]) if keep.any() else None

    sens = {f"px{int(t)}_frac{fr:.2f}": two_by_two(marked, frac[t] >= fr)
            for t in GRID_PX for fr in GRID_FRAC}

    verdict = ("支持" if (main_tab["rr"] >= RR_SUPPORT and main_tab["fisher_p"] < P_SUPPORT and outside)
               else "反証" if (main_tab["rr"] <= RR_REFUTE and not outside) else "判定保留")
    chk = dict(ic_k1_panel_area=dict(my_panel_median=round(my_panel_med, 5), track_j_median=round(j_med, 5),
                                     n_j_panels=len(j_area), n_my_panels=len(my_panels), ratio=round(ratio, 3), factor_bar=IC_K1_FACTOR, passed=ic1),
               ic_k2_alignment=dict(on_ink_median=round(onink_med, 4), transposed_median=round(trans_med, 4),
                                    bars=[IC_K2_ONINK, IC_K2_TRANSPOSED], passed=ic2),
               ic_k3_stray_marks=dict(n_stray=ic3, passed=ic3 == 0),
               nc_k1_fake_marks=dict(n_draws=N_NEG, rr_mean=round(float(neg_rr.mean()), 3),
                                     rr_p2_5=round(lo, 3), rr_p97_5=round(hi, 3),
                                     real_rr=main_tab["rr"], real_outside_band=outside))
    chk["passed"] = bool(ic1 and ic2 and ic3 == 0)
    summary = dict(registered=dict(ADJ_PX=ADJ_PX, ADJ_FRAC=ADJ_FRAC, SAMPLE_STEP=SAMPLE_STEP,
                                   fill_radius=a.fill_radius, seed=SEED, n_neg=N_NEG,
                                   bars=dict(rr_support=RR_SUPPORT, p_support=P_SUPPORT, rr_refute=RR_REFUTE)),
                   population=dict(labels=str(a.labels), n_faces=len(recs), n_panels=len({r["panel"] for r in recs}),
                                   n_strokes=len(rows), n_marks=int(marked.sum()),
                                   mark_rate=round(float(marked.mean()), 5),
                                   faces_with_mark=sum(1 for f in faces if f["n_marks"]),
                                   no_means_faces=len(no_means), no_means_orders=no_means),
                   main=main_tab, verdict=verdict,
                   k1_strat=dict(mh_rr=mh, strata=strata),
                   row_c_low_fill_excluded=dict(n_faces_excluded=len(low), threshold=LOW_FILL_FRAC, table=tab_c),
                   face_fill_frac=dict(median=round(float(np.median([f["face_fill_frac"] for f in faces])), 5),
                                       p10=round(float(np.percentile([f["face_fill_frac"] for f in faces], 10)), 5),
                                       p90=round(float(np.percentile([f["face_fill_frac"] for f in faces], 90)), 5)),
                   sensitivity=sens)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    print("器具確認:", "合格" if chk["passed"] else "不合格 —— 止めて相談する")
    if chk["passed"]:
        print(json.dumps({k: summary[k] for k in ("population", "main", "verdict", "k1_strat",
                                                  "row_c_low_fill_excluded", "face_fill_frac")},
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
