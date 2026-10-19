#!/usr/bin/env python
"""K21 費用器具の作り直し(2026-10-19、基盤 `reply_k_bridging_approved` §2 の 3 件目)。

部位群による費用は**飽和している**(全腕で群跨ぎ 0)。箱 B の対は定義上ほぼ同じ部位に
居るので、「**同じ部位の中で別の線を融合する**」という本当の誤りを見られない。
作者の時間を使わずに測れる別の費用を 2 本立てる:

  (1) **曖昧さ** —— 1 つの自由端が複数の相手を持つか。持つなら規則は選べない
  (2) **幅の不一致** —— 太さの違う 2 本を繋ぐのは怪しい。
      基準は **F 自身が結んだ対(箱 A)の分布**。F が受け入れている範囲の外なら費用

(2) は「器具は自分自身を測ってはいけない」を満たす: 基準が F の挙動であって K の仮説ではない。
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-grammar/tools/stroke")
import k2_box_b_gray as K

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results/k21_cost_rebuild_20261019"
TH = [4.0, 6.0, 8.0, 12.0, float("inf")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(ROOT / "results/k8_contiguity_20261018/per_pair_contig.csv"))
            if r["missing"] == "0"]
    gap = {(r["panel"], r["s0"], r["s1"], r["e0"], r["e1"]): float(r["gap_sk"])
           for r in csv.DictReader(open(ROOT / "results/k18_gap_cost_20261019/per_pair.csv"))}
    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(K.TRACK_H / "labels/state.json")
    P = G.S["panels"]
    byp = defaultdict(list)
    for r in rows:
        byp[int(r["panel"])].append(r)
    out = []
    pids = sorted(byp)
    for n, pid in enumerate(pids, 1):
        g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
        if g is None:
            continue
        Q = K.pipeline(g)
        for r in byp[pid]:
            i, j, ei, ej = int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"])
            if i not in Q["tips"] or j not in Q["tips"]:
                continue
            pa, pb = Q["tips"][i][ei], Q["tips"][j][ej]
            wa, wb = [float(Q["iw"][int(round(p[0])), int(round(p[1]))]) for p in (pa, pb)]
            k_ = (r["panel"], r["s0"], r["s1"], r["e0"], r["e1"])
            out.append(dict(panel=r["panel"], box=r["box"], s0=r["s0"], s1=r["s1"],
                            e0=r["e0"], e1=r["e1"], label=r["label"],
                            gap=gap.get(k_, float(r["gap"])), w_a=wa, w_b=wb,
                            w_mis=abs(wa - wb) / max(1e-6, (wa + wb) / 2)))
        if n % 25 == 0 or n == len(pids):
            print(f"  {n}/{len(pids)}", flush=True)
    with open(OUT / "per_pair_width.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    A = np.array([r["w_mis"] for r in out if r["box"] == "A"])
    B = [r for r in out if r["box"] == "B"]
    q95 = float(np.quantile(A, .95)); q99 = float(np.quantile(A, .99))
    print(f"\n基準 = 箱 A(F 自身が結んだ対) n={len(A)}  "
          f"幅の不一致 中央値 {np.median(A):.3f}  95% {q95:.3f}  99% {q99:.3f}")
    tipc = defaultdict(list)
    for r in B:
        tipc[(r["panel"], r["s0"], r["e0"])].append(r)
        tipc[(r["panel"], r["s1"], r["e1"])].append(r)
    res = []
    print(f'\n{"θ":>6}{"対":>5}{"曖昧な対":>9}{"":>2}{"幅不一致中央":>13}{"95%超":>8}{"99%超":>8}')
    for th in TH:
        sel = [r for r in B if r["gap"] <= th]
        if not sel:
            continue
        t2 = defaultdict(int)
        for r in sel:
            t2[(r["panel"], r["s0"], r["e0"])] += 1
            t2[(r["panel"], r["s1"], r["e1"])] += 1
        amb = sum(1 for r in sel if t2[(r["panel"], r["s0"], r["e0"])] > 1
                  or t2[(r["panel"], r["s1"], r["e1"])] > 1)
        wm = np.array([r["w_mis"] for r in sel])
        row = dict(th=("inf" if th == float("inf") else th), n=len(sel),
                   ambiguous=amb, ambiguous_share=round(amb / len(sel), 4),
                   w_mis_median=round(float(np.median(wm)), 3),
                   over_q95=int((wm > q95).sum()), over_q95_share=round(float((wm > q95).mean()), 4),
                   over_q99=int((wm > q99).sum()))
        res.append(row)
        print(f'{str(row["th"]):>6}{row["n"]:>5}{amb:>9}{"":>2}{row["w_mis_median"]:>13.3f}'
              f'{row["over_q95"]:>8}{row["over_q99"]:>8}')
    json.dump(dict(reference=dict(box="A", n=len(A), median=round(float(np.median(A)), 4),
                                  q95=round(q95, 4), q99=round(q99, 4),
                                  note="基準は F 自身が結んだ対。K の仮説ではない"),
                   arms=res,
                   note="部位群の費用は飽和していた(全腕で群跨ぎ 0)ので、"
                        "作者の時間を使わない費用を 2 本に作り直した"),
              open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n出力: {OUT}")


if __name__ == "__main__":
    main()
