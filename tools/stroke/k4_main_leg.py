#!/usr/bin/env python
"""K4 主脚(2026-10-16 事前登録。棒は結果を見る前に固定済み):
印のある線は**畳まれた橋を通る**か。

主値: 長さ 5 分位で層別した MH-RR = P(畳まれた橋を 1 つ以上通る | 印あり) / (| 印なし)
支持: MH-RR >= 2.0 かつ Fisher 両側 p < 0.01 かつ NC-k4 の 95% 帯の外
反証: 印のある線の尾根長が印のない線のそれと区別できない(2 標本 KS で p >= 0.05)
NC-k4: 偽の印 200 回(K1 と同じ引き方。顔ごとに印と同数を無作為に)
IC-k5: 橋 span は Track F の `strokes_from_skeleton` から取る。自作しない
対象: K1 と同じ 179 顔(学習用・完了・`broken` の項目あり)。検証用は読まない

「通る」の定義(実装): F は橋 span を節点に吸収して 2 つの節点を 1 つにする。
線の中の 2 つの span が**その畳み込み後の節点で対にされた**とき、その線は畳まれた橋を通る。
尾根長は F が記録した橋 span の画素数、`ratio` = 画素数 / F の畳み込み閾値
`max(3.0, 1.5 x median(局所インク幅))`(K4 の変数は J の MERGE_PX ではない)。
"""
import argparse, csv, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import fisher_exact, ks_2samp

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))

THR, GROW, MAX_BEND = 128, 200, 35.0
MH_BAR, P_BAR, KS_BAR = 2.0, 0.01, 0.05
N_NEG, SEED, Q = 200, 20261016, 5


def work(t):
    pid, source, name, do_ic = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths
    import strokes_instrumented as SI
    import strokes as F_ORIG                                   # 読むだけ(IC-k5a)
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(skeletonize(ink), dist * 2.0,
                                                             max_bend=MAX_BEND)
    ic_same = None
    if do_ic:
        o = F_ORIG.strokes_from_skeleton(skeletonize(ink), dist * 2.0, max_bend=MAX_BEND)
        ic_same = bool(len(o[0]) == len(spans)
                       and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                               for a, b in zip(o[0], spans))
                       and np.array_equal(o[1], sos) and np.array_equal(o[2], keep)
                       and o[3] == info)

    mb = {int(k): v for k, v in extra["merge_bridges"].items()}
    at = {tuple(int(x) for x in k.split("-")): int(v) for k, v in extra["linked_at"].items()}
    gr = {}
    for i in range(len(spans)):
        gr.setdefault(int(sos[i]), []).append(i)
    out = []
    for k, mem in gr.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        ms = set(s + 1 for s in mem)
        lens, ratios = [], []
        for (a, b), node in at.items():                  # この線の中で作られた対だけ
            if a in ms and b in ms and node in mb:
                for _s, npx, _w, thr in mb[node]:
                    lens.append(float(npx)); ratios.append(float(npx) / float(thr))
        out.append((int(bool(lens)), lens, ratios))
    return pid, (out, ic_same, int(info["bridges"]))


def mh_rr(y, x, length, q=Q):
    """長さ q 分位で層別した Mantel-Haenszel の RR(y=印、x=橋を通る)。"""
    edges = np.quantile(length, np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    num = den = 0.0
    strata = []
    for i in range(q):
        m = (length > edges[i]) & (length <= edges[i + 1])
        a = float((y & x & m).sum()); b = float((y & ~x & m).sum())
        c = float((~y & x & m).sum()); d = float((~y & ~x & m).sum())
        n = a + b + c + d
        if n == 0:
            continue
        num += a * (c + d) / n
        den += c * (a + b) / n
        strata.append(dict(stratum=i, lo=round(float(edges[i]) if i else float(length.min()), 1),
                           hi=round(float(edges[i + 1]) if i < q - 1 else float(length.max()), 1),
                           n_marked=int((y & m).sum()), n_unmarked=int((~y & m).sum()),
                           p_thru_marked=round(float(a / (a + b)), 4) if a + b else None,
                           p_thru_unmarked=round(float(c / (c + d)), 4) if c + d else None))
    return (round(num / den, 4) if den else float("nan")), strata


def table(y, x):
    a = int((y & x).sum()); b = int((y & ~x).sum())
    c = int((~y & x).sum()); d = int((~y & ~x).sum())
    pm = a / (a + b) if a + b else float("nan")
    pu = c / (c + d) if c + d else float("nan")
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return dict(marked_thru=a, marked_not=b, unmarked_thru=c, unmarked_not=d,
                p_thru_marked=round(pm, 4), p_thru_unmarked=round(pu, 4),
                rr=round(pm / pu, 4) if pu else float("nan"), fisher_p=float(f"{p:.3e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/k4_main_leg_20261017")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    from k1_broken_vs_fill import densify
    G.load_state(Path(a.labels))
    recs = sorted([r for r in G.S["faces"].values()
                   if r["usage"] == "train" and r["status"] == "done" and "broken" in r],
                  key=lambda r: (r["panel"], r["order"]))
    if a.limit:
        recs = recs[:a.limit]
    pids = sorted({r["panel"] for r in recs})
    P = G.S["panels"]
    print(f"顔 {len(recs)} / コマ {len(pids)}", flush=True)
    byp = {}
    tasks = [(p, P[p]["source"], P[p]["name"], k < 20) for k, p in enumerate(pids)]
    with Pool(a.workers) as pool:
        for i, (pid, o) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            byp[pid] = o
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    ic5 = [o[1] for o in byp.values() if o and o[1] is not None]
    bad = [p for p in pids if byp.get(p) is None or len(byp[p][0]) != int(P[p]["n"])]
    chk = dict(ic_k5a_copy_equals_original=dict(n_panels=len(ic5), all_same=bool(all(ic5)),
                                                passed=bool(ic5 and all(ic5))),
               ic_token_count_equals_pack=dict(n_panels=len(pids), n_bad=len(bad),
                                               bad=bad[:10], passed=not bad),
               registered=dict(mh_bar=MH_BAR, p_bar=P_BAR, ks_bar=KS_BAR, n_neg=N_NEG,
                               seed=SEED, quantiles=Q, thr=THR, grow=GROW, max_bend=MAX_BEND))
    chk["passed"] = bool(chk["ic_k5a_copy_equals_original"]["passed"]
                         and chk["ic_token_count_equals_pack"]["passed"])
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    if not chk["passed"]:
        print("器具確認: 不合格 —— 止めて相談する"); return
    print("器具確認: 合格", flush=True)

    rows = []
    for r in recs:
        o = byp.get(r["panel"])
        if o is None:
            continue
        pl = G.face_payload(r["order"])
        broken = set(r.get("broken", []))
        for s in pl["strokes"]:
            sid = s["id"]
            thru, lens, ratios = o[0][sid]
            _pts, length = densify(np.asarray(s["p"], np.float64).reshape(16, 2))
            rows.append(dict(order=r["order"], panel=r["panel"], sid=sid,
                             marked=int(sid in broken), length=round(length, 2), thru=thru,
                             n_bridges=len(lens),
                             ridge_med=round(float(np.median(lens)), 3) if lens else "",
                             ratio_med=round(float(np.median(ratios)), 4) if ratios else ""))
    with open(out / "per_stroke.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    y = np.array([r["marked"] for r in rows], bool)
    x = np.array([r["thru"] for r in rows], bool)
    L = np.array([r["length"] for r in rows], float)
    order = np.array([r["order"] for r in rows], int)
    main = table(y, x)
    mh, strata = mh_rr(y, x, L)

    rng = np.random.default_rng(SEED)
    by_face = {o_: (np.flatnonzero(order == o_), int(y[order == o_].sum()))
               for o_ in np.unique(order)}
    neg_mh, neg_rr = [], []
    for _ in range(N_NEG):
        fake = np.zeros(len(rows), bool)
        for o_, (m, k) in by_face.items():
            if k:
                fake[rng.choice(m, k, replace=False)] = True
        neg_mh.append(mh_rr(fake, x, L)[0])
        neg_rr.append(table(fake, x)["rr"])
    nm = np.array([v for v in neg_mh if np.isfinite(v)], float)
    nr = np.array([v for v in neg_rr if np.isfinite(v)], float)
    lo, hi = float(np.percentile(nm, 2.5)), float(np.percentile(nm, 97.5))
    outside = bool(mh < lo or mh > hi)

    # 反証(登録どおり): 尾根長の分布が印の有無で区別できるか。線ごとに中央値を 1 つ
    rm = np.array([r["ridge_med"] for r in rows if r["thru"] and r["marked"]], float)
    ru = np.array([r["ridge_med"] for r in rows if r["thru"] and not r["marked"]], float)
    ks = ks_2samp(rm, ru) if len(rm) >= 2 and len(ru) >= 2 else None
    qm = np.array([r["ratio_med"] for r in rows if r["thru"] and r["marked"]], float)
    qu = np.array([r["ratio_med"] for r in rows if r["thru"] and not r["marked"]], float)
    ks_ratio = ks_2samp(qm, qu) if len(qm) >= 2 and len(qu) >= 2 else None

    support = bool(mh >= MH_BAR and main["fisher_p"] < P_BAR and outside)
    refuted = bool(ks is not None and ks.pvalue >= KS_BAR)
    verdict = ("支持" if support and not refuted else
               "反証" if refuted and not support else
               "支持の条件と反証の条件が同時に成立(登録の不備として報告する)" if support and refuted
               else "判定保留")
    summary = dict(registered=chk["registered"],
                   population=dict(n_faces=len({r["order"] for r in rows}), n_panels=len(pids),
                                   n_strokes=len(rows), n_marks=int(y.sum()),
                                   n_thru=int(x.sum()), thru_share=round(float(x.mean()), 4)),
                   main=main, mh_rr=mh, strata=strata,
                   nc_k4=dict(n_draws=len(nm), mh_mean=round(float(nm.mean()), 4),
                              mh_p2_5=round(lo, 4), mh_p97_5=round(hi, 4),
                              rr_mean=round(float(nr.mean()), 4),
                              rr_p2_5=round(float(np.percentile(nr, 2.5)), 4),
                              rr_p97_5=round(float(np.percentile(nr, 97.5)), 4),
                              real_mh=mh, outside=outside),
                   ridge=dict(n_marked=len(rm), n_unmarked=len(ru),
                              med_marked=round(float(np.median(rm)), 2) if len(rm) else None,
                              med_unmarked=round(float(np.median(ru)), 2) if len(ru) else None,
                              ks_p=float(f"{ks.pvalue:.3e}") if ks is not None else None,
                              ks_stat=round(float(ks.statistic), 4) if ks is not None else None,
                              ratio_med_marked=round(float(np.median(qm)), 4) if len(qm) else None,
                              ratio_med_unmarked=round(float(np.median(qu)), 4) if len(qu) else None,
                              ks_p_ratio=float(f"{ks_ratio.pvalue:.3e}") if ks_ratio is not None else None),
                   verdict=verdict)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
