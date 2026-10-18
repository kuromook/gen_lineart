#!/usr/bin/env python
"""K7 部位ラベルの軸(2026-10-18 事前登録)。

問い: 部位ラベルは、対が箱 B(両端が自由端)に落ちることを予測するか。
口を選んだのは**作者の視認**であって数ではないので、確認的に検定してよい。

主値は **AFe = (RR-1)/RR**(曝露側の寄与割合)。**因果の上限としてのみ記述する。**
CI は order 単位のクラスタ・ブートストラップ(対は order 内で従属)。

器具確認(1 つでも外れたら止める):
  1. 再現: 箱 A 521 / 箱 B 182 / 箱 C 79、ラベル付与 782/782、両端一致 782/782
  2. 負の対照: order 内でラベルを入れ替える置換 1000 回で RR の中央値が [0.95, 1.05]
  3. 自明な腕: order 層別の MH-RR(素の RR が order 構成で説明されるなら 1 に寄る)
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent.parent
N_BOOT, N_PERM, SEED = 10000, 1000, 20261018
EXPOSURE = "face.mouth"
PERM_OK = (0.95, 1.05)
EXPECT = {"A": 521, "B": 182, "C": 79}


def rr_of(expo, out):
    """RR = P(out | expo) / P(out | ~expo)。片方が 0 なら None。"""
    a, b = int((expo & out).sum()), int((expo & ~out).sum())
    c, d = int((~expo & out).sum()), int((~expo & ~out).sum())
    if (a + b) == 0 or (c + d) == 0 or (c == 0):
        return None
    return (a / (a + b)) / (c / (c + d))


def mh_rr(expo, out, strata):
    """order 層別の Mantel-Haenszel RR。"""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, b = float((expo & out & m).sum()), float((expo & ~out & m).sum())
        c, d = float((~expo & out & m).sum()), float((~expo & ~out & m).sum())
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * (c + d) / n
        den += c * (a + b) / n
    return (num / den) if den > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/k2_missed_candidates_20261017/per_pair.csv")
    ap.add_argument("--snapshot", default="results/k1_broken_vs_fill_20261016/labels_snapshot.jsonl")
    ap.add_argument("--out", default="results/k7_part_axis_20261018")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    lab = {}
    for l in open(ROOT / a.snapshot):
        d = json.loads(l)
        for g in d.get("groups", []):
            for s in g["strokes"]:
                lab[(int(d["order"]), int(s))] = g["label"]
    rows = [r for r in csv.DictReader(open(ROOT / a.pairs))
            if r["same"] == "1" and r["passed"] == "1"]
    recs = []
    for r in rows:
        o = int(r["order"])
        la, lb = lab.get((o, int(r["t0"]))), lab.get((o, int(r["t1"])))
        recs.append(dict(order=o, panel=int(r["panel"]), box=r["box"], gap=float(r["gap"]),
                         label=la, label_ok=int(la is not None and la == lb)))
    # --- 器具確認 1: 再現 ---
    chk = dict(box_counts=dict(Counter(r["box"] for r in recs)),
               n_labelled=sum(1 for r in recs if r["label"]),
               n_agree=sum(r["label_ok"] for r in recs), n_total=len(recs))
    chk["ok"] = (chk["box_counts"] == EXPECT and chk["n_labelled"] == len(recs)
                 and chk["n_agree"] == len(recs))
    print("器具確認1 再現:", json.dumps(chk, ensure_ascii=False), flush=True)
    if not chk["ok"]:
        json.dump(dict(instrument=dict(reproduce=chk, passed=False),
                       verdict="**止める**: 母集団が再現しない"),
                  open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
        print("止める: 母集団が再現しない"); return

    order = np.array([r["order"] for r in recs])
    expo = np.array([r["label"] == EXPOSURE for r in recs])
    outc = np.array([r["box"] == "B" for r in recs])
    rr = rr_of(expo, outc)
    afe = (rr - 1.0) / rr if rr else None
    _od, pf = fisher_exact([[int((expo & outc).sum()), int((expo & ~outc).sum())],
                            [int((~expo & outc).sum()), int((~expo & ~outc).sum())]])
    # --- CI: order 単位のクラスタ・ブートストラップ ---
    rng = np.random.default_rng(SEED)
    uo = np.unique(order)
    idx_of = {o: np.where(order == o)[0] for o in uo}
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uo, len(uo), replace=True)
        ii = np.concatenate([idx_of[o] for o in pick])
        v = rr_of(expo[ii], outc[ii])
        if v:
            boot.append(v)
    boot = np.array(boot)
    rr_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    afe_b = (boot - 1.0) / boot
    afe_ci = [float(np.percentile(afe_b, 2.5)), float(np.percentile(afe_b, 97.5))]
    # --- 器具確認 2: order 内でラベルを入れ替える置換 ---
    perm = []
    for _ in range(N_PERM):
        e2 = expo.copy()
        for o in uo:
            ii = idx_of[o]
            e2[ii] = rng.permutation(expo[ii])
        v = rr_of(e2, outc)
        if v:
            perm.append(v)
    perm = np.array(perm)
    perm_med = float(np.median(perm))
    perm_ok = PERM_OK[0] <= perm_med <= PERM_OK[1]
    perm_p = float((np.abs(np.log(perm)) >= abs(np.log(rr))).mean()) if rr else None
    # --- 器具確認 3 / 自明な腕: order 層別 MH-RR ---
    mh = mh_rr(expo, outc, order)

    by_label = defaultdict(Counter)
    for r in recs:
        by_label[r["label"]][r["box"]] += 1
    tab = {k: dict(n=sum(v.values()), **{b: v.get(b, 0) for b in "ABC"},
                   box_b_share=round(v.get("B", 0) / sum(v.values()), 4))
           for k, v in sorted(by_label.items(), key=lambda t: -sum(t[1].values()))}

    with open(out / "per_pair_label.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["order", "panel", "box", "gap", "label", "label_ok"])
        w.writeheader(); w.writerows(recs)

    summary = dict(
        registered=dict(exposure=EXPOSURE, outcome="箱 B(両端が自由端)",
                        primary="AFe = (RR-1)/RR、因果の上限としてのみ記述する",
                        ci="order 単位のクラスタ・ブートストラップ", n_boot=N_BOOT,
                        n_perm=N_PERM, perm_ok_band=list(PERM_OK), seed=SEED,
                        note="口を選んだのは作者の視認であって数ではない。"
                             "部位の偏りはどの対が飛ばし線かを 1 件も決めない"),
        instrument=dict(reproduce=chk,
                        negative_control=dict(perm_median_rr=round(perm_med, 4),
                                              band=list(PERM_OK), ok=bool(perm_ok),
                                              perm_p_two_sided=perm_p),
                        trivial_arm=dict(mh_rr_by_order=round(mh, 4) if mh else None,
                                         note="素の RR が order 構成で説明されるなら 1 に寄る"),
                        passed=bool(perm_ok)),
        primary=dict(n_exposed=int(expo.sum()), n_exposed_boxB=int((expo & outc).sum()),
                     n_unexposed=int((~expo).sum()), n_unexposed_boxB=int((~expo & outc).sum()),
                     rr=round(rr, 4) if rr else None, rr_ci95=[round(x, 4) for x in rr_ci],
                     afe=round(afe, 4) if afe else None,
                     afe_ci95=[round(x, 4) for x in afe_ci],
                     fisher_p=float(pf),
                     reading="AFe は因果の上限。口であることが箱 B を作っているのではなく、"
                             "口の線の長さ・孤立しやすさ・描き方を全部含んだ上限値"),
        by_label_posthoc=tab,
        verdict=("器具確認に合格" if perm_ok else "**止める**: 負の対照が帯を外れた"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
