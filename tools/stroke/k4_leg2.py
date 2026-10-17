#!/usr/bin/env python
"""K4 第 2 脚(2026-10-16 事前登録、K2 の後に実行):
過接続と過分割は「尾根が吸収閾値のどちら側に落ちるか」の 2 症状か。

合格条件(登録のまま): **印のある線が通る畳まれた橋の `ratio` 中央値**と、
**K2 箱 C の候補の間にある尾根の `ratio` 中央値**が、ブートストラップ 95% 区間で重ならず、
**前者 <= 1 < 後者**であること。符号の反転が見えて初めて「1 つの機構の 2 症状」と書ける。
直列の 35 度判定(f0 -> 尾根 -> f1 の 2 回)の通過/不通過の内訳を併記する(基盤の訂正)。

`ratio` = 尾根 span の画素数 / Track F の畳み込み閾値 `max(3.0, 1.5 x median(局所インク幅))`。
**J の `MERGE_PX` ではない。印の線を作ったのは F だから。**

箱 C の候補の「尾根」の定め方(**測る前にここで固定**): 2 つの(畳み込み後の)節点の両方に
付いている span。複数あるときは**最も短いもの**を取る —— 2 つの節点が 1 つに畳まれるには
その最短の橋が吸収される必要があるので、機構に対応するのはそれである。
直接つながる span が無い対は「尾根なし」として**別に数え、中央値には入れない**。
"""
import argparse, csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))

THR, GROW, MAX_BEND = 128, 200, 35.0
N_BOOT, SEED = 10000, 20261016


def work(t):
    pid, source, name, pairs = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return pid, None
    from skimage.morphology import skeletonize
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, info, extra = SI.strokes_from_skeleton(skeletonize(ink), dist * 2.0,
                                                             max_bend=MAX_BEND)
    root = {int(k): int(v) for k, v in extra["node_root"].items()}
    spans_all = extra["spans_all"]                       # (画素数, 局所インク幅中央値, 節点数)
    # span -> その span が付いている(畳み込み後の)節点の集合
    roots_of = defaultdict(set)
    for s, n, _ar, _ac in extra["att"]:
        roots_of[int(s)].add(root.get(int(n), int(n)))
    # (節点 a, 節点 b) -> その両方に付いている span
    between = defaultdict(list)
    for s, rs in roots_of.items():
        rs = sorted(rs)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                between[(rs[i], rs[j])].append(s)
    arms = {int(k): {int(s): np.asarray(v, float) for s, v in lst}
            for k, lst in extra["arms"].items()}
    at = {tuple(int(x) for x in k.split("-")): int(v) for k, v in extra["linked_at"].items()}
    partner = defaultdict(set)                       # (節点, span) -> F が実際に対にした相手
    for (x, y), nd in at.items():
        partner[(nd, x)].add(y); partner[(nd, y)].add(x)
    cos_lim = -np.cos(np.deg2rad(MAX_BEND))

    out = []
    for d in pairs:
        n0, n1 = sorted((int(d["node0"]), int(d["node1"])))
        s0, s1 = int(d["s0"]) + 1, int(d["s1"]) + 1          # span 番号は 1 始まり
        cand = between.get((n0, n1), [])
        if not cand:
            out.append(dict(ridge=None)); continue
        best = min(cand, key=lambda s: spans_all[s - 1][0])   # 最短の橋
        npx, wmed, _nn = spans_all[best - 1]
        thr = max(3.0, 1.5 * float(wmed))
        # 直列の 35 度判定: f0 -> 尾根(節点 n0 で)、尾根 -> f1(節点 n1 で)
        def pass35(node, a, b):
            A = arms.get(node, {})
            if a not in A or b not in A:
                return None
            return bool(float(A[a] @ A[b]) <= cos_lim)
        na, nb = (n0, n1) if n0 in roots_of[s0] else (n1, n0)
        other = sorted((partner[(na, best)] | partner[(nb, best)]) - {s0, s1})
        ends_taken = int(bool((partner[(na, s0)] - {best}) or (partner[(nb, s1)] - {best})))
        out.append(dict(ridge=float(npx), thr=float(thr), ratio=float(npx) / thr,
                        n_between=len(cand),
                        g0=pass35(na, s0, best), g1=pass35(nb, best, s1),
                        ridge_linked_other=int(bool(other)), ends_taken=ends_taken))
    return pid, out


def boot_median(v, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = np.median(rng.choice(v, (n, len(v)), replace=True), axis=1)
    return round(float(np.median(v)), 4), round(float(np.percentile(m, 2.5)), 4), \
        round(float(np.percentile(m, 97.5)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--k2", default="results/k2_missed_candidates_20261017/per_pair.csv")
    ap.add_argument("--k4", default="results/k4_main_leg_20261017/per_stroke.csv")
    ap.add_argument("--out", default="results/k4_leg2_20261017")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    # 箱 C の候補(同部位・門を通ったもの)
    cpairs = [r for r in csv.DictReader(open(ROOT / a.k2))
              if r["box"] == "C" and r["same"] == "1" and r["passed"] == "1"]
    byp = defaultdict(list)
    for r in cpairs:
        byp[int(r["panel"])].append(r)
    # 印のある線が通った畳まれた橋の ratio
    marked = [float(r["ratio_med"]) for r in csv.DictReader(open(ROOT / a.k4))
              if r["marked"] == "1" and r["thru"] == "1" and r["ratio_med"]]

    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"], rs) for p, rs in byp.items()]
    print(f"箱 C の候補 {len(cpairs)} 件 / コマ {len(tasks)} 枚、印の線 {len(marked)} 本", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for pid, res in pool.imap_unordered(work, tasks, chunksize=2):
            if res is None:
                continue
            for r, d in zip(byp[pid], res):
                rows.append(dict(panel=pid, order=r["order"], s0=r["s0"], s1=r["s1"],
                                 gap=r["gap"], **d))
    with open(out / "per_candidate.csv", "w", newline="") as fh:
        fn = ["panel", "order", "s0", "s1", "gap", "ridge", "thr", "ratio", "n_between", "g0", "g1", "ridge_linked_other", "ends_taken"]
        w = csv.DictWriter(fh, fieldnames=fn, restval=""); w.writeheader(); w.writerows(rows)

    withr = [r for r in rows if r.get("ridge") is not None]
    box_c = [r["ratio"] for r in withr]
    mm, mlo, mhi = boot_median(marked)
    cm, clo, chi = boot_median(box_c) if box_c else (None, None, None)
    overlap = not (mhi < clo or chi < mlo) if box_c else None
    sign = bool(mm <= 1.0 < cm) if box_c else None
    g_both = sum(1 for r in withr if r["g0"] and r["g1"])
    g_one = sum(1 for r in withr if (r["g0"] or r["g1"]) and not (r["g0"] and r["g1"]))
    g_none = sum(1 for r in withr if not r["g0"] and not r["g1"])
    g_na = sum(1 for r in withr if r["g0"] is None or r["g1"] is None)
    both = [r for r in withr if r["g0"] and r["g1"]]
    g_both_other = sum(1 for r in both if r.get("ridge_linked_other"))
    g_both_free = [r for r in both if not r.get("ridge_linked_other")]
    g_both_ends = sum(1 for r in g_both_free if r.get("ends_taken"))
    summary = dict(
        registered=dict(condition="ブートストラップ 95% 区間が重ならず、印の線の中央値 <= 1 < 箱 C の中央値",
                        n_boot=N_BOOT, seed=SEED, max_bend=MAX_BEND),
        marked_strokes=dict(n=len(marked), median_ratio=mm, ci95=[mlo, mhi]),
        box_c=dict(n_candidates=len(rows), n_with_ridge=len(withr),
                   n_no_ridge=len(rows) - len(withr),
                   median_ratio=cm, ci95=[clo, chi],
                   ridge_px_median=round(float(np.median([r["ridge"] for r in withr])), 2) if withr else None,
                   n_between_median=float(np.median([r["n_between"] for r in withr])) if withr else None,
                   ratio_gt1=sum(1 for r in withr if r["ratio"] > 1.0)),
        serial_35deg=dict(both_pass=g_both, one_fails=g_one, both_fail=g_none,
                          arm_missing=g_na,
                          both_pass_but_ridge_arm_taken=g_both_other,
                          both_pass_arm_free=g_both - g_both_other,
                          of_those_end_arm_taken=g_both_ends,
                          unexplained=len(g_both_free) - g_both_ends,
                          note="尾根が橋として吸収されている場合、原本は尾根の腕を候補から外すので "
                               "arm_missing になる。これ自体が『F では中継が残らない』の現れ"),
        verdict=("支持(符号の反転が見えた)" if (sign and overlap is False) else
                 "不支持(区間が重なる)" if overlap else
                 "不支持(符号の条件を満たさない)"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
