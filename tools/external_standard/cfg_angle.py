"""The CFG angle, as registered: doc/prereg_generation_setting_cfg_20261103.md.

Per axis and per level x in {4.0, 9.0}:

    delta_x = AUC23(E_x) - AUC23(E6)        (A2 = 1, external = 0, tile unit)

Nulls, printed before any bar:
  N2  -- per prompt, the E_x and E6 tiles of that prompt swap sides with
         probability 1/2 (whole prompt at a time), 1000 draws. Band 2.5-97.5%.
  N1  -- delta_seed = AUC23(S1) - AUC23(E6): same settings, another seed.

Checks: IC2 reproduces the third pool's AUC23 table (+-0.0001), IC3 self AUC,
IC4 tile counts and drops by reason, NC1 the swap null is centred, NC2 the
ink amount beside every axis and the Spearman rho with ink for C1 axes.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_auc import AXES, DISCRETE_AXES, NULL_AXIS, auc, tie_structure  # noqa: E402

LEVELS = ("4.0", "9.0")
MOVE, NEAR, SEED_UNSTABLE, SEED_UNSTABLE_AXES = 0.10, 0.05, 0.10, 3   # registered
DRAWS = 1000
IC2_TOL = 0.0001
# result_third_pool_20261031.md section 4, AUC23 column (animagine), registered as IC2.
THIRD_POOL_AUC23 = {
    "line_width_p50": 0.9423, "background_ratio": 0.8650,
    "faint_mean_dist_to_ink": 0.2453, "faint_of_drawn_ratio": 0.0189,
    "long_line_ratio": 0.7532, "width_consistency": 0.5730,
    "deep_black_ratio": 0.5491, "orientation_entropy": 0.4004,
    "faint_near_ink_ratio": 0.7329, "long_component_ratio": 0.3157,
    "components_per_1k_ink_px": 0.2845, "ink_ratio": 0.2433,
}
# purpose_sorting_20261103.md section 1: the box each axis is in now.
BOX = {
    "long_line_ratio": "model", "long_component_ratio": "model",
    "orientation_entropy": "model",
    "faint_of_drawn_ratio": "process", "faint_near_ink_ratio": "process",
    "faint_mean_dist_to_ink": "process", "deep_black_ratio": "process",
    "components_per_1k_ink_px": "hold", "background_ratio": "hold",
    "line_width_p50": "discrete", "width_consistency": "discrete",
    "ink_ratio": "null",
}


def prompt_of(sample):
    """external_animagine_xl_31_0123_seed..._w0000_0000.png -> '0123'."""
    return sample.split("_seed")[0].rsplit("_", 1)[1]


def load(path, labels):
    out = {lab: collections.defaultdict(list) for lab in labels}
    for r in csv.DictReader(open(path)):
        if r["label"] in out:
            author = r["label"] in ("a1_author_analogue", "a2_author_digital")
            key = r["sample"] if author else prompt_of(r["sample"])
            out[r["label"]][key].append(r)
    return out


def vals(groups, axis, keys=None):
    keys = groups.keys() if keys is None else keys
    return np.array([float(r[axis]) for k in keys for r in groups[k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--profile-new", default="results/cfg_angle_20261103/profile_cfg.csv")
    ap.add_argument("--tiling-reports", default="data/external_standard")
    ap.add_argument("--out", default="results/cfg_angle_20261103")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()

    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    third = load(args.profile_third, ["a1_author_analogue", "a2_author_digital", "a3_animagine"])
    new = load(args.profile_new, ["e6_remeasure", "e4", "e9", "s1"])
    A1, A2, E6 = third["a1_author_analogue"], third["a2_author_digital"], third["a3_animagine"]
    E = {"4.0": new["e4"], "9.0": new["e9"]}
    S1 = new["s1"]

    p("# CFG の角度 —— 登録 doc/prereg_generation_setting_cfg_20261103.md")
    p()
    p("## 器具検査")
    n = {k: sum(len(v) for v in g.values()) for k, g in
         [("A1", A1), ("A2", A2), ("E6", E6), ("E6再測", new["e6_remeasure"]),
          ("E4", E["4.0"]), ("E9", E["9.0"]), ("S1", S1)]}
    p("タイル数: " + " / ".join(f"{k} {v}" for k, v in n.items()))
    # IC4: the tiler's own report, drops by reason
    for tag in ("cfg4p0", "cfg9p0"):
        rp = Path(args.tiling_reports) / f"tiling_report_tiles_animagine_xl_31_{tag}.json"
        if rp.exists():
            p(f"IC4 {tag}: {json.dumps(json.load(open(rp)), ensure_ascii=False)}")
        else:
            p(f"IC4 {tag}: 報告が無い({rp})")
    # extra check (not registered): the remeasured E6 equals the third pool's E6 tile by tile
    old = {r["sample"]: r for g in E6.values() for r in g}
    worst = 0.0
    missing = 0
    for g in new["e6_remeasure"].values():
        for r in g:
            o = old.get(r["sample"])
            if o is None:
                missing += 1
                continue
            worst = max(worst, max(abs(float(r[a]) - float(o[a])) for a in AXES))
    p(f"(登録外の点検)E6 を測り直した値と第3のプールの値: 最大の差 {worst:.3e} / 片方にしか無いタイル {missing}")

    ic2_ok = True
    ic3_ok = True
    for a in AXES:
        v = auc(vals(A2, a), vals(E6, a))
        ok = abs(v - THIRD_POOL_AUC23[a]) <= IC2_TOL + 5e-5   # table printed to 4 places
        ic2_ok &= ok
        s = auc(vals(E6, a), vals(E6, a))
        ic3_ok &= abs(s - 0.5) < 1e-12
        if not ok:
            p(f"IC2 不一致 {a}: {v:.5f} 対 {THIRD_POOL_AUC23[a]}")
    p(f"IC2 第3のプールの AUC23 を再現: {'通った' if ic2_ok else '通らない'}")
    p(f"IC3 自分自身との AUC 0.5: {'通った' if ic3_ok else '通らない'}")

    rng = np.random.default_rng(20261103)
    a2 = {a: vals(A2, a) for a in AXES}
    a1 = {a: vals(A1, a) for a in AXES}
    base23 = {a: auc(a2[a], vals(E6, a)) for a in AXES}
    base13 = {a: auc(a1[a], vals(E6, a)) for a in AXES}

    p()
    p("## 帰無値(棒の前に)")
    seed_delta = {a: auc(a2[a], vals(S1, a)) - base23[a] for a in AXES}
    unstable = [a for a in AXES if abs(seed_delta[a]) >= SEED_UNSTABLE]
    p("N1 seed Δ = AUC23(S1) − AUC23(E6):")
    for a in AXES:
        p(f"  {a:28s} {seed_delta[a]:+.4f}")
    p(f"  |Δ_seed| ≥ {SEED_UNSTABLE} の軸 {len(unstable)} 本 {unstable}")

    null = {}
    nc1_ok = True
    for lv in LEVELS:
        Ex = E[lv]
        shared = sorted(set(Ex) & set(E6))
        only_x = sorted(set(Ex) - set(E6))
        only_6 = sorted(set(E6) - set(Ex))
        p(f"N2 CFG {lv}: 対になるプロンプト {len(shared)} / E_x にだけ {len(only_x)} / E6 にだけ {len(only_6)}"
          "(片側だけのプロンプトは入れ替えずに元の側に置く)")
        null[lv] = {}
        for a in AXES:
            px = [vals(Ex, a, [k]) for k in shared]
            p6 = [vals(E6, a, [k]) for k in shared]
            fx, f6 = vals(Ex, a, only_x), vals(E6, a, only_6)
            out = np.empty(args.draws)
            for d in range(args.draws):
                flip = rng.random(len(shared)) < 0.5
                xs = [p6[i] if f else px[i] for i, f in enumerate(flip)]
                ss = [px[i] if f else p6[i] for i, f in enumerate(flip)]
                nx = np.concatenate(xs + [fx])
                n6 = np.concatenate(ss + [f6])
                out[d] = auc(a2[a], nx) - auc(a2[a], n6)
            med, lo, hi = np.median(out), np.percentile(out, 2.5), np.percentile(out, 97.5)
            null[lv][a] = (float(med), float(lo), float(hi))
            if abs(med) >= 0.02:
                nc1_ok = False
            p(f"  {a:28s} 中央 {med:+.4f}  帯 [{lo:+.4f}, {hi:+.4f}]")
    p(f"NC1 入れ替えの中央が全軸 |·|<0.02: {'通った' if nc1_ok else '通らない'}")

    p()
    p("## 主値 Δ_x = AUC23(E_x) − AUC23(E6)(副: AUC13 の Δ、隣に墨の量)")
    ink_d = {lv: auc(a2[NULL_AXIS], vals(E[lv], NULL_AXIS)) - base23[NULL_AXIS] for lv in LEVELS}
    p(f"墨の量 ink_ratio の Δ: CFG 4.0 {ink_d['4.0']:+.4f} / CFG 9.0 {ink_d['9.0']:+.4f}"
      "(予想: 4.0 で正、9.0 で負)")
    pooled = collections.defaultdict(list)
    for g in (E6, E["4.0"], E["9.0"]):
        for a in AXES:
            pooled[a].extend(vals(g, a).tolist())

    result = {"ic2": ic2_ok, "ic3": ic3_ok, "nc1": nc1_ok, "seed_delta": {a: float(v) for a, v in seed_delta.items()},
              "unstable_axes": unstable, "axes": {}}
    c4 = len(unstable) >= SEED_UNSTABLE_AXES
    instrument_ok = ic2_ok and nc1_ok and ic3_ok
    for a in AXES:
        rows = {}
        for lv in LEVELS:
            v23 = auc(a2[a], vals(E[lv], a))
            v13 = auc(a1[a], vals(E[lv], a))
            d = v23 - base23[a]
            med, lo, hi = null[lv][a]
            c1 = abs(d) >= MOVE and not (lo <= d <= hi) and abs(d) > abs(seed_delta[a])
            crosses = (v23 - 0.5) * (base23[a] - 0.5) < 0
            c2 = c1 and (crosses or abs(v23 - 0.5) <= NEAR)
            rho = float(spearmanr(pooled[a], pooled[NULL_AXIS]).statistic) if c1 and a != NULL_AXIS else None
            rows[lv] = {"auc23": v23, "delta": d, "auc13": v13, "delta13": v13 - base13[a],
                        "C1": bool(c1), "C2": bool(c2), "rho_ink": rho}
        result["axes"][a] = {"box": BOX[a], "auc23_e6": base23[a], **rows}
        p(f"{a:28s} [{BOX[a]}] E6 {base23[a]:.4f}")
        for lv in LEVELS:
            r = rows[lv]
            extra = f"  ρ(墨) {r['rho_ink']:+.3f}" + ("  墨の量と一緒に動く" if abs(r['rho_ink']) >= 0.7 else "") \
                if r["rho_ink"] is not None else ""
            p(f"    CFG {lv}: AUC23 {r['auc23']:.4f}  Δ {r['delta']:+.4f}  "
              f"(AUC13 Δ {r['delta13']:+.4f})  C1 {'○' if r['C1'] else '·'}  C2 {'○' if r['C2'] else '·'}{extra}")
        if a in DISCRETE_AXES:
            for lab, g in (("A2", A2), ("E6", E6), ("E4", E["4.0"]), ("E9", E["9.0"])):
                nv, big, frac = tie_structure(vals(g, a).tolist())
                p(f"    tie {lab}: 相異なる値 {nv} / 最大の同値塊 {big}({frac:.1%})")

    p()
    p("## 判定")
    if not instrument_ok:
        p("C5 器具: IC2 / IC3 / NC1 のどれかが通らない —— 何も読まない")
    elif c4:
        p(f"C4 発火: |Δ_seed| ≥ {SEED_UNSTABLE} の軸が {len(unstable)} 本 —— 仕分けを動かさない(判定保留)。"
          "§1 の表に「外部側は1回の引き」と印を付ける")
    else:
        n_c2 = 0
        for a in AXES:
            r = result["axes"][a]
            c1 = any(r[lv]["C1"] for lv in LEVELS)
            c2 = any(r[lv]["C2"] for lv in LEVELS)
            n_c2 += c2 and BOX[a] != "null"
            which = ",".join(lv for lv in LEVELS if r[lv]["C2"]) or ",".join(lv for lv in LEVELS if r[lv]["C1"])
            box = BOX[a]
            if box == "null":
                say = "帰無値の腕 —— 印字だけ"
            elif box == "discrete":
                say = "判定保留のまま(離散、条例4)" + (f" —— C{'2' if c2 else '1'} は通った({which})" if c1 else "")
            elif c2:
                say = {"model": "他(生成設定)に根拠つきで移る",
                       "process": "工程でも設定でも説明できる(箱は変わらない)",
                       "hold": "他(生成設定)に移る"}[box] + f"({which})"
            elif c1:
                say = f"設定で動くが差は埋まらない({which})。箱はそのまま"
            else:
                say = "動かない。箱はそのまま" + ("。モデルの違いは CFG では再現しない" if box == "model" else "")
            p(f"  {a:28s} {say}")
        p(f"C2 を通った軸(帰無値の腕を除く) {n_c2} 本" +
          (" —— 7本以上: 外部側は設定1つでどちらにも置ける" if n_c2 >= 7 else ""))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg_angle.txt").write_text("\n".join(lines) + "\n")
    json.dump(result, open(out / "cfg_angle.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved: {out}/cfg_angle.{{txt,json}}")


if __name__ == "__main__":
    main()
