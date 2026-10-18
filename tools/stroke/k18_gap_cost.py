#!/usr/bin/env python
"""K18 登録 (G): 細線化が作る隙間 ——「割らない」側の足跡と費用(2026-10-19 事前登録)。

**設計の探索であって検定ではない。**材料は train(作者の 40 件は既読)。
Track F は封存、変種は K の tree。閾値・棒・帰無値・器具確認は `doc/work_log.md` の
「登録 (G)」に、数値を 1 つも見る前に書いてある。ここはそれを実行するだけ。

述語「墨の隙間」の定義は登録どおり:
  tip から reach=10px の span 画素の平均で外向き接線 → 0.5px 刻みで進み
  **器具自身の `ink_mask(thr=128, grow=200)` の墨**を出たら止める / 自分の連結成分を
  出たら止める / 弦の長さで打ち切る。墨の隙間 = max(0, L - ext_a - ext_b)。

出力: summary.json / per_pair.csv / montage_extension.png
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-grammar/tools/stroke")
import k2_box_b_gray as K                                            # noqa: E402

REACH, STEP = 10.0, 0.5
TH_SK = [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, float("inf")]
TH_INK = [0.0, 1.0, 2.0, 4.0, 6.0, float("inf")]
N_NEG, SEED = 200, 20261019
ANS = {1: "飛ばし線", 2: "かすれ", 3: "切れていない", 4: "判断できない"}
NULL_PRECISION = 9 / 40          # 帰無値(統計量 = 精度)。登録に書いた値


def cp95(k, n):
    if n == 0:
        return None, None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return round(lo, 4), round(hi, 4)


def outward_tangent(P, i, e):
    """tip から外側を向く単位接線。`arm_direction` と同じ規約の逆向き。"""
    p = P["paths"][i]
    tip = p[0] if e == 0 else p[-1]
    d = np.hypot(p[:, 0] - tip[0], p[:, 1] - tip[1])
    sel = p[d <= REACH]
    if len(sel) < 2:
        sel = p[np.argsort(d)[: max(2, min(len(p), 4))]]
    v = tip - sel.mean(0)
    nv = float(np.hypot(*v))
    return (v / nv) if nv > 1e-6 else None


def extend(P, i, e, L):
    """登録 (G) の伸ばし。返り値 (伸びた長さ, 止めた理由)。"""
    p = P["paths"][i]
    tip = p[0] if e == 0 else p[-1]
    u = outward_tangent(P, i, e)
    if u is None:
        return 0.0, "接線が退化"
    own = K.comp_at(P, tip)
    H, W = P["ink"].shape
    t = 0.0
    while t + STEP <= L:
        q = tip + (t + STEP) * u
        r, c = int(round(q[0])), int(round(q[1]))
        if not (0 <= r < H and 0 <= c < W):
            return t, "画像の外"
        v = int(P["lab"][r, c])
        if v == 0:
            return t, "墨の外"
        if own and v != own:
            return t, "別成分"
        t += STEP
    return t, "弦で打ち切り"


PAIRS = {}


def work(pid_src_name):
    pid, source, nm = pid_src_name
    g = cv2.imread(str(K.DATASET / source / "line" / nm), 0)
    if g is None:
        return pid, None
    P = K.pipeline(g)
    out = []
    for r in PAIRS[pid]:
        i, j, ei, ej = int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"])
        if i not in P["tips"] or j not in P["tips"]:
            out.append(dict(r, missing_tip=1)); continue
        pa = P["paths"][i][0] if ei == 0 else P["paths"][i][-1]
        pb = P["paths"][j][0] if ej == 0 else P["paths"][j][-1]
        L = float(np.hypot(*(pb - pa)))
        ea, ra_ = extend(P, i, ei, L)
        eb, rb_ = extend(P, j, ej, L)
        out.append(dict(r, missing_tip=0, gap_sk=L, ext_a=ea, ext_b=eb,
                        stop_a=ra_, stop_b=rb_,
                        ink_gap=max(0.0, L - ea - eb),
                        tok_a=int(P["span_tok"].get(i, -1)),
                        tok_b=int(P["span_tok"].get(j, -1))))
    return pid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(K.TRACK_H / "labels/state.json"))
    ap.add_argument("--groups", default=str(K.TRACK_H / "labels/face_groups.jsonl"))
    ap.add_argument("--contig", default="results/k8_contiguity_20261018/per_pair_contig.csv")
    ap.add_argument("--k10", default="results/k10_author_request_20261018")
    ap.add_argument("--out", default="results/k18_gap_cost_20261019")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(ROOT / a.contig)) if r["missing"] == "0"]
    boxb = [r for r in rows if r["box"] == "B"]
    k8_contig = sum(1 for r in boxb if r["contiguous"] == "1")
    print(f"箱 B(missing=0) {len(boxb)} 件 / K8 の `contiguous=1` は {k8_contig} 件", flush=True)

    # --- 作者の 40 件を (コマ, s0, s1, 隙間) で照合(K10 の訂正 4 の一般形) ---
    key = json.load(open(ROOT / a.k10 / "key.json"))["key"]
    ansj = json.load(open(ROOT / a.k10 / "answers.json"))["answers"]
    judged = {}
    for did, it in key.items():
        if it["kind"] != "main":
            continue
        judged[(int(it["panel"]), int(it["s0"]), int(it["s1"]), round(float(it["gap"]), 2))] = \
            int(ansj[did])
    print(f"作者が答えた main {len(judged)} 件", flush=True)

    # --- 印と部位群(append-only を last-wins で) ---
    last = {}
    for l in open(a.groups):
        r = json.loads(l); last[r["order"]] = r
    pop = [r for r in last.values() if r.get("usage") == "train"
           and r.get("status") == "done" and "broken" in r]
    marked, grp_of = defaultdict(set), defaultdict(dict)
    for r in pop:
        p = int(r["panel"])
        for s in set(r.get("broken") or []):
            marked[p].add(int(s))
        for gi, gg in enumerate(r.get("groups", [])):
            for s in gg["strokes"]:
                grp_of[p][int(s)] = f'{r["order"]}:{gi}'

    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    for r in boxb:
        PAIRS.setdefault(int(r["panel"]), []).append(r)
    pids = sorted(PAIRS)
    print(f"コマ {len(pids)}", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for n, (pid, o) in enumerate(pool.imap_unordered(
                work, [(p, P[p]["source"], P[p]["name"]) for p in pids], chunksize=1), 1):
            if o: res.extend(o)
            if n % 20 == 0 or n == len(pids): print(f"  {n}/{len(pids)}", flush=True)

    # --- 器具確認 ---
    miss = [r for r in res if r["missing_tip"]]
    ok = [r for r in res if not r["missing_tip"]]
    for r in ok:
        k_ = (int(r["panel"]), int(r["s0"]), int(r["s1"]), round(float(r["gap"]), 2))
        r["answer"] = judged.get(k_, 0)
    matched = sum(1 for r in ok if r["answer"])
    stops = Counter([r["stop_a"] for r in ok] + [r["stop_b"] for r in ok])
    ic = dict(ic1_n_boxb=len(ok), ic1_pass=(len(ok) == 182),
              ic2_judged_matched=matched, ic2_pass=(matched == len(judged)),
              ic3_ink_gap_zero=sum(1 for r in ok if r["ink_gap"] <= 0),
              ic3_k8_contiguous=k8_contig,
              ic4_stop_reasons=dict(stops), missing_tip=len(miss))
    print("器具確認:", json.dumps(ic, ensure_ascii=False), flush=True)

    ext = np.array([r["ext_a"] for r in ok] + [r["ext_b"] for r in ok])
    retr = dict(n_tips=len(ext), median=round(float(np.median(ext)), 3),
                q25=round(float(np.quantile(ext, .25)), 3),
                q75=round(float(np.quantile(ext, .75)), 3),
                max=round(float(ext.max()), 3),
                zero_share=round(float((ext == 0).mean()), 4))
    gs = np.array([r["gap_sk"] for r in ok]); gi = np.array([r["ink_gap"] for r in ok])
    gaps = dict(gap_sk_median=round(float(np.median(gs)), 3),
                ink_gap_median=round(float(np.median(gi)), 3),
                shrink_median=round(float(np.median(gs - gi)), 3))

    # --- 腕 ---
    rng = np.random.default_rng(SEED)
    jrows = [r for r in ok if r["answer"]]
    jans = np.array([r["answer"] for r in jrows])
    perms = [rng.permutation(jans) for _ in range(N_NEG)]

    def arm(sel_key, th):
        sel = [r for r in ok if r[sel_key] <= th]
        js = [n for n, r in enumerate(jrows) if jrows[n][sel_key] <= th]
        kasure = sum(1 for n in js if jans[n] == 2)
        tobashi = sum(1 for n in js if jans[n] == 1)
        dk = sum(1 for n in js if jans[n] == 4)
        notcut = sum(1 for n in js if jans[n] == 3)
        prec = (kasure / len(js)) if js else None
        lo, hi = cp95(kasure, len(js))
        nn = [float(np.mean(p[js] == 2)) for p in perms] if js else []
        cross = same = unk = extend_marked = 0
        for r in sel:
            pid = int(r["panel"]); ga = grp_of[pid].get(r["tok_a"]); gb = grp_of[pid].get(r["tok_b"])
            if ga is None or gb is None: unk += 1
            elif ga != gb: cross += 1
            else: same += 1
            if r["tok_a"] in marked[pid] or r["tok_b"] in marked[pid]: extend_marked += 1
        return dict(axis=sel_key, th=("inf" if th == float("inf") else th),
                    footprint=len(sel), judged=len(js),
                    kasure=kasure, tobashi=tobashi, not_cut=notcut, dont_know=dk,
                    precision=(round(prec, 4) if prec is not None else None),
                    cp95_lo=lo, cp95_hi=hi,
                    ratio=(round((1 - prec) / prec, 3) if prec else None),
                    nc_median=(round(float(np.median(nn)), 4) if nn else None),
                    nc_lo=(round(float(np.quantile(nn, .025)), 4) if nn else None),
                    nc_hi=(round(float(np.quantile(nn, .975)), 4) if nn else None),
                    cost_cross_group=cross, cost_same_group=same, cost_group_unknown=unk,
                    cost_extends_marked=extend_marked)

    arms = [arm("gap_sk", t) for t in TH_SK] + [arm("ink_gap", t) for t in TH_INK]
    summary = dict(
        registered=dict(primary="精度 = 閉じた対のうち「かすれ」の割合、棒 精度 > 50%",
                        bar_identity="比 = (1-精度)/精度。K17 の棒と同一",
                        null_value=NULL_PRECISION, null_statistic="精度",
                        denominator="判断できないを含める",
                        note="設計の探索であって検定ではない。40 件は既読。"
                             "隙間長の軸と face.other の軸は同じ 7 件(K10 訂正 3)",
                        th_sk=[("inf" if t == float("inf") else t) for t in TH_SK],
                        th_ink=[("inf" if t == float("inf") else t) for t in TH_INK]),
        instrument_checks=ic, retraction=retr, gaps=gaps,
        author=dict(n_judged=len(jrows), by_answer={ANS[k]: int((jans == k).sum())
                                                    for k in sorted(ANS)}),
        arms=arms)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "per_pair.csv", "w", newline="") as fh:
        f = ["panel", "order", "label", "box", "s0", "s1", "e0", "e1", "gap_sk", "ext_a", "ext_b",
             "ink_gap", "stop_a", "stop_b", "tok_a", "tok_b", "answer"]
        w = csv.DictWriter(fh, fieldnames=f, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"},
                     ensure_ascii=False, indent=1))
    print(f'{"軸":<9}{"閾値":>6}{"足跡":>6}{"判定":>5}{"かすれ":>7}{"飛ばし":>7}{"不明":>5}'
          f'{"精度":>8}{"CP95":>18}{"比":>8}{"NC":>8}{"群跨ぎ":>7}{"印延長":>7}')
    for r in arms:
        cp = f'[{r["cp95_lo"]}, {r["cp95_hi"]}]' if r["cp95_lo"] is not None else "-"
        print(f'{r["axis"]:<9}{str(r["th"]):>6}{r["footprint"]:>6}{r["judged"]:>5}'
              f'{r["kasure"]:>7}{r["tobashi"]:>7}{r["dont_know"]:>5}'
              f'{str(r["precision"]):>8}{cp:>18}{str(r["ratio"]):>8}'
              f'{str(r["nc_median"]):>8}{r["cost_cross_group"]:>7}{r["cost_extends_marked"]:>7}')


if __name__ == "__main__":
    main()
