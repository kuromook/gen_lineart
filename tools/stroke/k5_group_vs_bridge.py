#!/usr/bin/env python
"""K5(2026-10-16 事前登録): ベタ由来の屑は作者のパーツグループに入らないか。

前提: IC-k7(pack の線番号と私の再現した線番号の一致)が合格していること。
棒・陰性対照・母集団は doc/work_log.md の K5 登録のまま。ここは実行するだけ。

主脚   RR  = P(損失2 | グループ内) / P(損失2 | グループ外)。支持 RR<=0.5、反証 RR∈[0.80,1.25]
第2脚  RR2 = P(損失2 | 1〜2本のグループ) / P(損失2 | 3本以上)。支持 RR2>=1.5
両脚とも in_fill の内外に分けて出す(修正 B3)。NC-k5 は同じ大きさの偽グループ 200 回。
"""
import argparse, csv, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))
sys.path.insert(0, "/home/sh1/deepl/lineart-solid-fill/tools/evaluation")

THR, GROW, MAX_BEND, IN_FILL_BAR = 128, 200, 35.0, 0.5
RR_SUPPORT, RR_REFUTE, P_BAR, RR2_SUPPORT = 0.5, (0.80, 1.25), 0.01, 1.5
N_NEG, SEED, SMALL_GROUP = 200, 20261016, 2


def work(t):
    """コマ 1 枚: 出力された線の番号ごとに「畳まれた橋か」「ベタの中か」を返す。"""
    pid, source, name = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None
    from skimage.morphology import skeletonize
    from stroke_path import chain, span_paths
    from fill_mask import fill_mask, ink_of
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    fill = fill_mask(ink_of(g))
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(skeletonize(ink), dist * 2, max_bend=MAX_BEND)
    bridge = extra["bridge_flag"]
    groups = {}
    for i in range(len(spans)):
        groups.setdefault(int(sos[i]), []).append(i)
    out = []
    for k, mem in groups.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        isb = len(mem) == 1 and bool(bridge[mem[0] + 1])        # 畳まれた橋がそのままトークンに
        rr, cc = spans[mem[0]]
        ov = float(fill[rr, cc].mean()) if isb else float("nan")
        out.append((int(isb), int(ov >= IN_FILL_BAR) if isb else 0))
    return pid, out


def tab(a_mask, b_mask, y):
    a = int(y[a_mask].sum()); b = int((~y[a_mask]).sum())
    c = int(y[b_mask].sum()); d = int((~y[b_mask]).sum())
    pa = a / (a + b) if a + b else float("nan")
    pb = c / (c + d) if c + d else float("nan")
    rr = pa / pb if pb else float("nan")
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return dict(n_left=a + b, n_right=c + d, p_left=round(pa, 5), p_right=round(pb, 5),
                rr=round(rr, 4), fisher_p=float(f"{p:.3e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/k5_group_vs_bridge_20261016")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
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
    with Pool(a.workers) as pool:
        for i, (pid, o) in enumerate(pool.imap_unordered(work, [(p, P[p]["source"], P[p]["name"]) for p in pids],
                                                         chunksize=2), 1):
            byp[pid] = o
            if i % 25 == 0 or i == len(pids):
                print(f"  {i}/{len(pids)}", flush=True)

    rows, ic_bad = [], []
    for r in recs:
        o = byp.get(r["panel"])
        if o is None:
            ic_bad.append(r["panel"]); continue
        if len(o) != int(P[r["panel"]]["n"]):                    # IC: 本数が pack と一致
            ic_bad.append(r["panel"]); continue
        pl = G.face_payload(r["order"])
        size = {}
        for gi, grp in enumerate(r["groups"]):
            for sid in grp["strokes"]:
                size[sid] = len(grp["strokes"])
        for s in pl["strokes"]:
            sid = s["id"]
            isb, inf = o[sid]
            rows.append(dict(order=r["order"], panel=r["panel"], sid=sid, loss2=isb, in_fill=inf,
                             in_group=int(sid in size), group_size=size.get(sid, 0)))
    if ic_bad:
        print(json.dumps(dict(ic_failed_panels=sorted(set(ic_bad))[:20], passed=False), ensure_ascii=False))
        print("器具確認: 不合格 —— 止めて相談する"); return

    y = np.array([r["loss2"] for r in rows], bool)
    ing = np.array([r["in_group"] for r in rows], bool)
    inf = np.array([r["in_fill"] for r in rows], bool)
    gsz = np.array([r["group_size"] for r in rows], int)
    order = np.array([r["order"] for r in rows], int)

    def legs(sel, label):
        d = dict(n=int(sel.sum()))
        d["main"] = tab(sel & ing, sel & ~ing, y)
        small = sel & ing & (gsz <= SMALL_GROUP)
        big = sel & ing & (gsz > SMALL_GROUP)
        d["leg2_small_vs_big"] = tab(small, big, y)
        return d

    allr = np.ones(len(rows), bool)
    # 「ベタの外」の腕: ベタ由来の損失2 を母集団から除く。残った y は「ベタの外の損失2」を意味する
    out_fill_pop = ~(y & inf)
    res = dict(all=legs(allr, "all"),
               out_fill=legs(out_fill_pop, "out_fill"),
               out_fill_note="母集団からベタの中の損失2を除いた。この腕の y は『ベタの外の損失2』",
               loss2_in_fill_only=dict(n=int((y & inf).sum()), n_loss2=int(y.sum()),
                                       share_in_fill=round(float((y & inf).sum() / max(y.sum(), 1)), 4)))

    rng = np.random.default_rng(SEED)
    by_face = {}
    for o_ in np.unique(order):
        m = np.flatnonzero(order == o_)
        by_face[o_] = (m, int(ing[m].sum()))
    neg = []
    for _ in range(N_NEG):
        fake = np.zeros(len(rows), bool)
        for o_, (m, k) in by_face.items():
            if k:
                fake[rng.choice(m, k, replace=False)] = True
        neg.append(tab(fake, ~fake, y)["rr"])
    neg = np.array([v for v in neg if np.isfinite(v)], float)
    lo, hi = float(np.percentile(neg, 2.5)), float(np.percentile(neg, 97.5))
    rr = res["all"]["main"]["rr"]
    outside = bool(rr < lo or rr > hi)
    verdict = ("支持" if (rr <= RR_SUPPORT and res["all"]["main"]["fisher_p"] < P_BAR and outside)
               else "反証" if (RR_REFUTE[0] <= rr <= RR_REFUTE[1] and not outside) else "判定保留")
    rr2 = res["all"]["leg2_small_vs_big"]["rr"]
    v2 = ("支持" if (rr2 >= RR2_SUPPORT and res["all"]["leg2_small_vs_big"]["fisher_p"] < P_BAR)
          else "不支持")

    summary = dict(registered=dict(rr_support=RR_SUPPORT, rr_refute=RR_REFUTE, p_bar=P_BAR,
                                   rr2_support=RR2_SUPPORT, n_neg=N_NEG, seed=SEED,
                                   small_group_max=SMALL_GROUP, in_fill_bar=IN_FILL_BAR),
                   population=dict(n_faces=len({r["order"] for r in rows}), n_panels=len(pids),
                                   n_strokes=len(rows), n_loss2=int(y.sum()),
                                   n_in_group=int(ing.sum()),
                                   in_group_share=round(float(ing.mean()), 4)),
                   results=res,
                   nc_k5=dict(n_draws=len(neg), rr_mean=round(float(neg.mean()), 4),
                              p2_5=round(lo, 4), p97_5=round(hi, 4),
                              real_rr=rr, outside=outside),
                   verdict=dict(main=verdict, leg2=v2))
    with open(out / "per_stroke.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
