#!/usr/bin/env python
"""K4a(2026-10-16 事前登録): 畳み込まれた尾根のインクが、全てのストロークから落ちている量。

ラベルを 1 枚も使わない。棒を置くのは「太い交差点ほど損失が大きい」の一点だけで、
他は記述として出す。閾値・棒・陰性対照は doc/work_log.md の K4a 登録のまま。

IC-k5a 複製が原本と一致すること(機械的に選んだ先頭 20 コマで、返り値の最初の 4 つが完全一致)
IC-k5b 残ったストロークの本数が panel_pack_20260919 の n と一致すること(177 コマの 95% 以上)
NC-k4a コマ内で局所インク幅を橋 span 間で無作為に入れ替えると相関が消えること

出力: summary.json / check.json / per_bridge.csv / per_panel.csv
"""
import argparse, csv, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
PACK = TRACK_F / "results/panel_pack_20260919"
CACHE = TRACK_F / "results/panel_skeleton_20260917"
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))        # 複製
sys.path.insert(0, str(TRACK_F / "tools/stroke"))               # 原本(読むだけ)
CACHE = CACHE  # 未使用。pack は skeletonize をその場で行う経路から作られている

# IC-k5b が落ちて判明(2026-10-16): 作者の線を作ったのは tokenize_panels.py ではなく
# export_panel_tokens_v2.py で、インクは grow=200 のヒステリシス、骨格はその場で skeletonize。
# panel_skeleton_20260917 のキャッシュでも plain ink_mask でもない。ここはその経路をなぞる。
THR, GROW = 128, 200
MAX_BEND, MIN_LEN, RHO_BAR, P_BAR, NC_BAR = 35.0, 8, 0.30, 0.01, 0.10
IC_K5B_BAR, N_IC_K5A, SEED = 0.95, 20, 20261016
W_KNEE = 2.0          # B1: max(3.0, 1.5w) の折れ点。ここ未満は閾値が定数
RATIO_BAR = 0.90      # B1 主値 A: len/w の中央値がこれ以上で、かつ両帰無模型の帯の外なら支持
N_NULL = 200          # 帰無模型の引き直し回数


def run_panel(t):
    pid, source, name, n_pack, want_original = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return dict(pid=pid, ok=False, why="missing image")
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    wd = dist * 2
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(sk, wd, max_bend=MAX_BEND)
    # pack の n は keep の本数から「鎖にしたとき 2 点未満」を落としたもの(export_panel_tokens_v2)
    groups = {}
    for i in range(len(spans)):
        groups.setdefault(int(sos[i]), []).append(i)
    n_repro = 0
    for k, mem in groups.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) >= 2:
            n_repro += 1

    same = None
    if want_original:
        import strokes as F                                      # Track F の原本
        o_spans, o_sos, o_keep, o_info = F.strokes_from_skeleton(sk, wd, max_bend=MAX_BEND)
        same = bool(len(o_spans) == len(spans)
                    and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                            for a, b in zip(o_spans, spans))
                    and np.array_equal(o_sos, sos) and np.array_equal(o_keep, keep) and o_info == info)

    skel_px = int(sk.sum())
    # B1 主値 B: 2 つの相異なる節点を繋ぐ全ての span(選別を外した母集団)
    two = [(npx, wid) for npx, wid, nnode in extra["spans_all"] if nnode == 2]
    rows = []
    l1_n = l1_px = l2_n = l2_px = 0
    for s, npx, w, thr in extra["bridges"]:
        kept = bool(keep[sos[s - 1]])
        rows.append(dict(pid=pid, span=s, n_px=npx, ink_width=round(w, 3), thr=round(thr, 3),
                         ratio=round(npx / thr, 4), kept=int(kept)))
        if kept:
            l2_n += 1; l2_px += npx
        else:
            l1_n += 1; l1_px += npx
    bpx = l1_px + l2_px
    prow = dict(pid=pid, source=source, name=name, skel_px=skel_px, spans=info["spans"],
                nodes=info["nodes"], strokes=info["strokes"], bridges=info["bridges"],
                n_pack=n_pack, n_repro=n_repro, count_match=int(n_repro == n_pack),
                bridge_px=bpx, bridge_px_frac=round(bpx / max(skel_px, 1), 5),
                loss1_dropped_n=l1_n, loss1_dropped_px=l1_px,
                loss1_px_frac=round(l1_px / max(skel_px, 1), 5),
                loss2_token_n=l2_n, loss2_token_px=l2_px,
                loss2_px_frac=round(l2_px / max(skel_px, 1), 5),
                n_two_node=len(two), ic_k5a_same=same)
    return dict(pid=pid, ok=True, prow=prow, rows=rows, two=[(pid, a, b) for a, b in two])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="K1 と同じコマ集合を取るための固定ラベル")
    ap.add_argument("--out", default="results/k4a_bridge_loss_20261016")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    recs = [r for r in G.S["faces"].values() if r["usage"] == "train" and r["status"] == "done" and "broken" in r]
    pids = sorted({r["panel"] for r in recs})
    if a.limit:
        pids = pids[:a.limit]
    panels = G.S["panels"]
    ic_a = set(pids[:N_IC_K5A])                                  # 機械的に先頭 20 コマ
    tasks = [(p, panels[p]["source"], panels[p]["name"], int(panels[p]["n"]), p in ic_a) for p in pids]
    print(f"コマ {len(tasks)} 枚 / IC-k5a は先頭 {len(ic_a)} 枚", flush=True)

    prows, brows, twos, bad = [], [], [], []
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_panel, tasks, chunksize=2), 1):
            if not res["ok"]:
                bad.append(res); continue
            prows.append(res["prow"]); brows.extend(res["rows"]); twos.extend(res["two"])
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    prows.sort(key=lambda r: r["pid"])

    ic5a = [r["ic_k5a_same"] for r in prows if r["ic_k5a_same"] is not None]
    ic5a_ok = bool(ic5a) and all(ic5a)
    match = np.array([r["count_match"] for r in prows], bool)
    ic5b_rate = float(match.mean()) if len(match) else 0.0
    ic5b_ok = bool(ic5b_rate >= IC_K5B_BAR)

    w = np.array([r["ink_width"] for r in brows], float)
    n = np.array([r["n_px"] for r in brows], float)
    thr = np.array([r["thr"] for r in brows], float)
    pid = np.array([r["pid"] for r in brows], int)
    rng = np.random.default_rng(SEED)

    # --- 主値 B: 選別を外した母集団(2 つの相異なる節点を繋ぐ全 span) ---
    t_pid = np.array([t[0] for t in twos], int)
    t_n = np.array([t[1] for t in twos], float)
    t_w = np.array([t[2] for t in twos], float)

    def by_knee(mask_w, mask_n, mask_pid, label):
        out = {}
        for key, m in (("all", np.ones(len(mask_w), bool)),
                       ("w_lt_2.0", mask_w < W_KNEE), ("w_ge_2.0", mask_w >= W_KNEE)):
            if m.sum() < 10:
                out[key] = dict(n=int(m.sum()), rho=None, p=None, nc_rho=None); continue
            r_, p_ = spearmanr(mask_w[m], mask_n[m])
            perm = np.empty(int(m.sum()))
            ww, pp = mask_w[m], mask_pid[m]
            for q in np.unique(pp):
                sel = pp == q
                perm[sel] = rng.permutation(ww[sel])
            nr, np_ = spearmanr(perm, mask_n[m])
            out[key] = dict(n=int(m.sum()), rho=round(float(r_), 4), p=float(f"{p_:.3e}"),
                            nc_rho=round(float(nr), 4), nc_p=float(f"{np_:.3e}"))
        return out

    main_b = by_knee(t_w, t_n, t_pid, "two_node")
    side_selected = by_knee(w, n, pid, "bridges_only")
    b_ge = main_b["w_ge_2.0"]
    nc_ok = bool(b_ge["nc_rho"] is not None and abs(b_ge["nc_rho"]) <= NC_BAR)
    b_ok = bool(b_ge["rho"] is not None and b_ge["rho"] >= RHO_BAR and b_ge["p"] < P_BAR and nc_ok)

    # --- 主値 A: len/w の分布を 2 つの帰無模型と比べる(比例域のみ) ---
    prop = w >= W_KNEE
    ratio = n[prop] / w[prop]
    obs_med = float(np.median(ratio))
    pile = float(((ratio > 1.35) & (ratio <= 1.5)).mean())
    thr_p, w_p = thr[prop], w[prop]
    uni = [float(np.median(rng.uniform(0, thr_p) / w_p)) for _ in range(N_NULL)]
    emp_pool = t_n[t_n > 0]
    emp = []
    for _ in range(N_NULL):
        draw = rng.choice(emp_pool, size=(len(thr_p), 8))        # 閾値で切り詰めて引く
        ok_ = draw <= thr_p[:, None]
        first = np.where(ok_.any(1), draw[np.arange(len(thr_p)), ok_.argmax(1)], thr_p)
        emp.append(float(np.median(first / w_p)))
    def band(v):
        v = np.array(v); return dict(median=round(float(np.median(v)), 4),
                                     p2_5=round(float(np.percentile(v, 2.5)), 4),
                                     p97_5=round(float(np.percentile(v, 97.5)), 4))
    nb_uni, nb_emp = band(uni), band(emp)
    outside = bool((obs_med < nb_uni["p2_5"] or obs_med > nb_uni["p97_5"])
                   and (obs_med < nb_emp["p2_5"] or obs_med > nb_emp["p97_5"]))
    a_verdict = "支持" if (obs_med >= RATIO_BAR and outside) else ("反証" if not outside else "判定保留")

    chk = dict(ic_k5a_copy_equals_original=dict(n_panels=len(ic5a), all_same=ic5a_ok, passed=ic5a_ok),
               ic_k5b_count_matches_pack=dict(n_panels=len(prows), rate=round(ic5b_rate, 4),
                                              bar=IC_K5B_BAR, passed=ic5b_ok),
               nc_k4b_width_shuffled_two_node=dict(rho=b_ge["nc_rho"], bar=NC_BAR, passed=nc_ok),
               missing_panels=len(bad))
    chk["passed"] = bool(ic5a_ok and ic5b_ok and nc_ok)

    def frac(key):
        v = np.array([r[key] for r in prows], float)
        return dict(median=round(float(np.median(v)), 5), p10=round(float(np.percentile(v, 10)), 5),
                    p90=round(float(np.percentile(v, 90)), 5), mean=round(float(v.mean()), 5))
    summary = dict(
        registered=dict(max_bend=MAX_BEND, min_len=MIN_LEN, rho_bar=RHO_BAR, p_bar=P_BAR,
                        nc_bar=NC_BAR, ic_k5b_bar=IC_K5B_BAR, seed=SEED,
                        w_knee=W_KNEE, ratio_bar=RATIO_BAR, n_null=N_NULL,
                        w_provenance="dist*2, dist = distanceTransform(ink_mask(g, thr=128, grow=200)) "
                                     "-- F の畳み込み閾値が使う ink_width と同一の配列",
                        pipeline="export_panel_tokens_v2.py の経路(ヒステリシス + その場の skeletonize)"),
        population=dict(n_panels=len(prows), n_bridges=len(brows), n_two_node_spans=len(twos),
                        skel_px_total=int(sum(r["skel_px"] for r in prows))),
        primary_A_ratio=dict(n=int(prop.sum()), obs_median_len_over_w=round(obs_med, 4),
                             pile_up_1p35_to_1p5=round(pile, 4),
                             null_uniform=nb_uni, null_empirical=nb_emp,
                             bar=RATIO_BAR, outside_both_bands=outside, verdict=a_verdict),
        primary_B_two_node_rho=dict(by_ink_width=main_b, bars=dict(rho=RHO_BAR, p=P_BAR, nc=NC_BAR),
                                    verdict=("支持" if b_ok else "不支持")),
        side_bridges_only_rho=dict(by_ink_width=side_selected,
                                   note="選別規則 len<=max(3.0,1.5w) が範囲制限で正の相関を作るため、"
                                        "この ρ と入れ替え対照は人工物と信号を分離できない。記録のみ"),
        loss_1_dropped=dict(n=int(sum(r["loss1_dropped_n"] for r in prows)),
                            px=int(sum(r["loss1_dropped_px"] for r in prows)), frac=frac("loss1_px_frac")),
        loss_2_spurious_token=dict(n=int(sum(r["loss2_token_n"] for r in prows)),
                                   px=int(sum(r["loss2_token_px"] for r in prows)), frac=frac("loss2_px_frac")),
        bridge_total=dict(n=len(brows), frac=frac("bridge_px_frac")),
        bridge_len=dict(median=float(np.median(n)), p10=float(np.percentile(n, 10)),
                        p90=float(np.percentile(n, 90)), max=float(n.max())),
        ink_width=dict(median=float(np.median(w)), p10=float(np.percentile(w, 10)),
                       p90=float(np.percentile(w, 90))))
    with open(out / "per_panel.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(prows[0].keys())); wr.writeheader(); wr.writerows(prows)
    with open(out / "per_bridge.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(brows[0].keys())); wr.writeheader(); wr.writerows(brows)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    print("器具確認:", "合格" if chk["passed"] else "不合格 —— 止めて相談する")
    if chk["passed"]:
        print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
