#!/usr/bin/env python
"""位置ビンの歪み: 10 本の結果に事前登録の判定を当てる(2026-10-09 事前登録どおり)。

判定(結果を見る前に固定):
  大きい  : V1/V2/V3 のいずれかが反転、または 5 数値のいずれかの変化が >= 0.270 bits
  小さい  : どれも反転せず、かつ 5 数値すべての変化が < 0.090 bits
  中間    : その間(記録は差し替えるが再実行の理由にはならない)
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "results/posbins_20261009"
BIG, SMALL = 0.270, 0.090
REC = {"cloze_none": 6.314605334367116, "cloze_tags": 6.353530597693522,
       "cloze_type": 6.387861989437070, "twostage": 6.658, "setar": 6.680,
       "marginal": 7.562884283205894}
KEY = {"cloze_none": ("cloze_orig_none", "cloze_fixed_none"),
       "cloze_tags": ("cloze_orig_tags", "cloze_fixed_tags"),
       "cloze_type": ("cloze_orig_type", "cloze_fixed_type"),
       "twostage": ("train_2stage_orig", "train_2stage_fixed"),
       "setar": ("train_set_orig", "train_set_fixed")}


def main():
    runs = json.load(open(OUT / "runs.json"))["runs"]
    chk = json.load(open(OUT / "check.json"))
    marg = {"orig": chk["K1b"]["pos_marginal"], "fixed": chk["N2_demo"]["marginal_fixed"]}
    rep = {"marginal": dict(orig=marg["orig"], fixed=marg["fixed"],
                            delta=round(marg["fixed"] - marg["orig"], 5),
                            recorded=REC["marginal"])}
    drift, deltas = {}, {"marginal": marg["fixed"] - marg["orig"]}
    for name, (ko, kf) in KEY.items():
        if ko not in runs or kf not in runs or runs[ko].get("failed") or runs[kf].get("failed"):
            rep[name] = dict(incomplete=True); continue
        o, f = runs[ko]["pos_bits"], runs[kf]["pos_bits"]
        rep[name] = dict(orig_rerun=round(o, 6), fixed=round(f, 6), delta=round(f - o, 5),
                         recorded=REC[name], drift_vs_recorded=round(o - REC[name], 5),
                         best_epoch=[runs[ko].get("best_epoch"), runs[kf].get("best_epoch")])
        drift[name] = o - REC[name]; deltas[name] = f - o

    # V1/V2/V3 の合否(補正後の値どうしで判定しなおす)
    mf = marg["fixed"]
    v = {}
    if "cloze_type" in deltas:
        band_f = max(rep[k]["fixed"] for k in ("cloze_none", "cloze_tags", "cloze_type"))
        v["V1"] = dict(desc="cloze pos < pos 周辺", before=True,
                       after=bool(band_f < mf), cloze_band_fixed=round(band_f, 5), marginal_fixed=mf)
    if "twostage" in deltas:
        t = rep["twostage"]["fixed"]
        v["V2"] = dict(desc="2段 pos < pos 周辺", before=True, after=bool(t < mf), twostage_fixed=round(t, 5))
        if "V1" in v:
            v["V3"] = dict(desc="2段 pos <= cloze 帯 への不通過", before=False,
                           after=bool(t <= v["V1"]["cloze_band_fixed"]),
                           note="before=False は『不通過』。after=True なら判定が反転する")
    flipped = [k for k, d in v.items() if d["before"] != d["after"]]
    mx = max(abs(x) for x in deltas.values()) if deltas else 0.0
    verdict = ("大きい" if (flipped or mx >= BIG) else
               "小さい" if mx < SMALL else "中間")
    rep["verdict"] = dict(verdict=verdict, flipped=flipped, max_abs_delta=round(mx, 5),
                          thresholds=dict(big=BIG, small=SMALL), V=v,
                          rerun_drift_vs_recorded={k: round(x, 5) for k, x in drift.items()},
                          drift_max_abs=round(max((abs(x) for x in drift.values()), default=0), 5))
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
