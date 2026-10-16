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

MAX_BEND, MIN_LEN, RHO_BAR, P_BAR, NC_BAR = 35.0, 8, 0.30, 0.01, 0.10
IC_K5B_BAR, N_IC_K5A, SEED = 0.95, 20, 20261016


def run_panel(t):
    pid, source, name, n_pack, want_original = t
    line = DATASET / source / "line" / name
    skel = CACHE / "skeleton" / source / name
    g = cv2.imread(str(line), 0)
    sk = cv2.imread(str(skel), 0)
    if g is None or sk is None:
        return dict(pid=pid, ok=False, why="missing image" if g is None else "missing skeleton")
    import strokes_instrumented as SI
    sk = sk > 0
    ink = SI.ink_mask(g)
    wd = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5) * 2
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(sk, wd, max_bend=MAX_BEND)

    same = None
    if want_original:
        import strokes as F                                      # Track F の原本
        o_spans, o_sos, o_keep, o_info = F.strokes_from_skeleton(sk, wd, max_bend=MAX_BEND)
        same = bool(len(o_spans) == len(spans)
                    and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                            for a, b in zip(o_spans, spans))
                    and np.array_equal(o_sos, sos) and np.array_equal(o_keep, keep) and o_info == info)

    skel_px = int(sk.sum())
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
                n_pack=n_pack, count_match=int(info["strokes"] == n_pack),
                bridge_px=bpx, bridge_px_frac=round(bpx / max(skel_px, 1), 5),
                loss1_dropped_n=l1_n, loss1_dropped_px=l1_px,
                loss1_px_frac=round(l1_px / max(skel_px, 1), 5),
                loss2_token_n=l2_n, loss2_token_px=l2_px,
                loss2_px_frac=round(l2_px / max(skel_px, 1), 5),
                ic_k5a_same=same)
    return dict(pid=pid, ok=True, prow=prow, rows=rows)


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

    prows, brows, bad = [], [], []
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_panel, tasks, chunksize=2), 1):
            if not res["ok"]:
                bad.append(res); continue
            prows.append(res["prow"]); brows.extend(res["rows"])
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
    pid = np.array([r["pid"] for r in brows], int)
    rho, pv = spearmanr(w, n)
    rng = np.random.default_rng(SEED)
    perm = np.empty_like(w)
    for p in np.unique(pid):
        m = pid == p
        perm[m] = rng.permutation(w[m])
    nc_rho, nc_p = spearmanr(perm, n)
    nc_ok = bool(abs(nc_rho) <= NC_BAR)

    chk = dict(ic_k5a_copy_equals_original=dict(n_panels=len(ic5a), all_same=ic5a_ok, passed=ic5a_ok),
               ic_k5b_count_matches_pack=dict(n_panels=len(prows), rate=round(ic5b_rate, 4),
                                              bar=IC_K5B_BAR, passed=ic5b_ok),
               nc_k4a_width_shuffled=dict(rho=round(float(nc_rho), 4), p=float(f"{nc_p:.3e}"),
                                          bar=NC_BAR, passed=nc_ok),
               missing_panels=len(bad))
    chk["passed"] = bool(ic5a_ok and ic5b_ok and nc_ok)

    def frac(key):
        v = np.array([r[key] for r in prows], float)
        return dict(median=round(float(np.median(v)), 5), p10=round(float(np.percentile(v, 10)), 5),
                    p90=round(float(np.percentile(v, 90)), 5), mean=round(float(v.mean()), 5))
    summary = dict(
        registered=dict(max_bend=MAX_BEND, min_len=MIN_LEN, rho_bar=RHO_BAR, p_bar=P_BAR,
                        nc_bar=NC_BAR, ic_k5b_bar=IC_K5B_BAR, seed=SEED),
        population=dict(n_panels=len(prows), n_bridges=len(brows),
                        skel_px_total=int(sum(r["skel_px"] for r in prows))),
        primary=dict(spearman_rho=round(float(rho), 4), p=float(f"{pv:.3e}"),
                     bars=dict(rho=RHO_BAR, p=P_BAR),
                     verdict=("支持" if (rho >= RHO_BAR and pv < P_BAR and nc_ok) else "不支持")),
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
