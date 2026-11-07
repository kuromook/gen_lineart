"""The generator angle, as registered: doc/prereg_generator_comfy_20261107.md.

Per axis and per model k in {E (animagine), I (illustrious)}:

    delta_g = AUC23(G_k) - AUC23(F_k)     G = ComfyUI pool, F = Forge seed2 pool

Nulls before any bar: N2 per-prompt swap of G_k and F_k (1000 draws, band
2.5-97.5%), N1 delta_seed = AUC23(S1_k) - AUC23(F_k). Same machinery as
cfg_angle.py, widened to two models. PC: the CFG 9.0 pool's deep_black delta.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_angle import load, vals  # noqa: E402
from compute_auc import AXES, DISCRETE_AXES, auc, tie_structure  # noqa: E402
from face_scale_angle import known_table  # noqa: E402

MOVE, NEAR, SEED_UNSTABLE, SEED_UNSTABLE_AXES = 0.10, 0.05, 0.10, 3   # registered
DRAWS = 1000
TOL = 0.0001
PC_AXIS, PC_VALUE = "deep_black_ratio", -0.2239   # result_generation_setting_cfg_20261104.md section 2
MODELS = {"E": ("a3_animagine", "ec", "s1e", "animagine_xl_31", "auc_three_arms_animagine.txt"),
          "I": ("a3_illustrious", "ic", "s1i", "illustrious_xl_v01", "auc_three_arms_illustrious.txt")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--profile-new", default="results/generator_comfy_20261107/profile_comfy.csv")
    ap.add_argument("--profile-cfg", default="results/cfg_angle_20261103/profile_cfg.csv")
    ap.add_argument("--out", default="results/generator_comfy_20261107")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    third = load(args.profile_third, ["a1_author_analogue", "a2_author_digital", "a3_animagine", "a3_illustrious"])
    new = load(args.profile_new, ["ec", "ic", "s1e", "s1i"])
    E9 = load(args.profile_cfg, ["e9"])["e9"]
    A1, A2 = third["a1_author_analogue"], third["a2_author_digital"]
    a1 = {a: vals(A1, a) for a in AXES}
    a2 = {a: vals(A2, a) for a in AXES}

    p("# 生成器の角度(Forge → ComfyUI)—— 登録 doc/prereg_generator_comfy_20261107.md")
    p("\n## 器具検査")
    ok = {"IC2": True, "IC3": True, "PC": True, "NC1": True}
    for k, (fl, gl, sl, name, table) in MODELS.items():
        F, G, S = third[fl], new[gl], new[sl]
        p(f"[{k}] タイル: F {sum(map(len, F.values()))} / G {sum(map(len, G.values()))} / S1 {sum(map(len, S.values()))}")
        rp = Path(f"data/external_standard/tiling_report_tiles_{name}_comfy.json")
        p(f"  IC4 {json.dumps(json.load(open(rp)), ensure_ascii=False) if rp.exists() else '報告が無い'}")
        known = known_table(f"results/third_pool_20261031/{table}")
        for a in AXES:
            v = auc(a2[a], vals(F, a))
            if abs(v - known[a][1]) > TOL + 5e-5:
                ok["IC2"] = False
                p(f"  IC2 不一致 {a}: {v:.5f} 対 {known[a][1]}")
            ok["IC3"] &= abs(auc(vals(F, a), vals(F, a)) - 0.5) < 1e-12
    p(f"IC2 第3のプールの AUC23 を再現: {'通った' if ok['IC2'] else '通らない'} / IC3 自己 AUC: {'通った' if ok['IC3'] else '通らない'}")

    rng = np.random.default_rng(20261107)

    def swap_null(G, F, a, ref):
        shared = sorted(set(G) & set(F))
        pg = [vals(G, a, [s]) for s in shared]
        pf = [vals(F, a, [s]) for s in shared]
        fg, ff = vals(G, a, sorted(set(G) - set(F))), vals(F, a, sorted(set(F) - set(G)))
        out = np.empty(args.draws)
        for d in range(args.draws):
            flip = rng.random(len(shared)) < 0.5
            xs = np.concatenate([pf[i] if f else pg[i] for i, f in enumerate(flip)] + [fg])
            ys = np.concatenate([pg[i] if f else pf[i] for i, f in enumerate(flip)] + [ff])
            out[d] = auc(ref, xs) - auc(ref, ys)
        return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(shared)

    # PC: the instrument sees a known setting change (CFG 9.0, animagine).
    F_E = third["a3_animagine"]
    d_pc = auc(a2[PC_AXIS], vals(E9, PC_AXIS)) - auc(a2[PC_AXIS], vals(F_E, PC_AXIS))
    m, lo, hi, _ = swap_null(E9, F_E, PC_AXIS, a2[PC_AXIS])
    s1d = auc(a2[PC_AXIS], vals(new["s1e"], PC_AXIS)) - auc(a2[PC_AXIS], vals(F_E, PC_AXIS))
    pc_g1 = abs(d_pc) >= MOVE and not (lo <= d_pc <= hi) and abs(d_pc) > abs(s1d)
    ok["PC"] = abs(d_pc - PC_VALUE) <= TOL + 5e-5 and pc_g1
    p(f"PC CFG 9.0 の {PC_AXIS} Δ {d_pc:+.4f}(既知 {PC_VALUE})帯 [{lo:+.4f},{hi:+.4f}] G1 式 {'○' if pc_g1 else '·'} → {'通った' if ok['PC'] else '通らない'}")

    result = {"checks": ok, "models": {}}
    p("\n## 帰無値(棒の前に)")
    nulls, seeds = {}, {}
    for k, (fl, gl, sl, name, _) in MODELS.items():
        F, G, S = third[fl], new[gl], new[sl]
        nulls[k], seeds[k] = {}, {}
        for a in AXES:
            seeds[k][a] = auc(a2[a], vals(S, a)) - auc(a2[a], vals(F, a))
            nulls[k][a] = swap_null(G, F, a, a2[a])
            ok["NC1"] &= abs(nulls[k][a][0]) < 0.02
        p(f"[{k}] 対になるプロンプト {nulls[k][AXES[0]][3]}")
        for a in AXES:
            m, lo, hi, _ = nulls[k][a]
            p(f"  {a:28s} N2 中央 {m:+.4f} 帯 [{lo:+.4f},{hi:+.4f}]  N1 Δ_seed {seeds[k][a]:+.4f}")
    p(f"NC1 入れ替えの中央が全軸 |·|<0.02: {'通った' if ok['NC1'] else '通らない'}")
    unstable = {k: [a for a in AXES if abs(seeds[k][a]) >= SEED_UNSTABLE] for k in MODELS}
    c4 = any(len(v) >= SEED_UNSTABLE_AXES for v in unstable.values())
    p(f"C4 の条件: |Δ_seed| ≥ {SEED_UNSTABLE} の軸 E {unstable['E']} / I {unstable['I']} → {'発火' if c4 else '発火しない'}")

    p("\n## 主値 Δ_g = AUC23(G) − AUC23(F)(副: AUC13 の Δ / AUC(G 対 F))")
    verdict = {}
    for a in AXES:
        row = []
        for k, (fl, gl, sl, name, _) in MODELS.items():
            F, G = third[fl], new[gl]
            base, v23 = auc(a2[a], vals(F, a)), auc(a2[a], vals(G, a))
            d = v23 - base
            d13 = auc(a1[a], vals(G, a)) - auc(a1[a], vals(F, a))
            direct = auc(vals(G, a), vals(F, a))
            m, lo, hi, _ = nulls[k][a]
            g1 = abs(d) >= MOVE and not (lo <= d <= hi) and abs(d) > abs(seeds[k][a])
            g0 = abs(d) <= max(NEAR, abs(seeds[k][a]))
            cls = "G1 動く" if g1 else ("G0 揃う" if g0 else "小さく動く")
            verdict[(a, k)] = cls
            result["models"].setdefault(k, {})[a] = {"auc23_F": base, "auc23_G": v23, "delta": d, "delta13": d13,
                                                     "auc_G_vs_F": direct, "band": [lo, hi], "seed": seeds[k][a],
                                                     "class": cls}
            row.append(f"{k} {base:.4f}→{v23:.4f} Δ {d:+.4f} {'外' if not (lo <= d <= hi) else '中'} (Δ13 {d13:+.4f}, G対F {direct:.4f}) {cls}")
        p(f"{a:28s} | " + " | ".join(row))
        if a in DISCRETE_AXES:
            for lab, g in (("A2", A2), ("E6", third["a3_animagine"]), ("EC", new["ec"]),
                           ("I6", third["a3_illustrious"]), ("IC", new["ic"])):
                nv, big, frac = tie_structure(vals(g, a).tolist())
                p(f"    tie {lab}: 相異なる値 {nv} / 最大の同値塊 {big}({frac:.1%})")

    p("\n## 判定")
    instrument = all(ok.values())
    if not instrument:
        p(f"C5: {ok} —— 何も読まない")
        call = "C5"
    elif c4:
        p("C4: 外部側が1回の引きで安定しない —— 判定保留")
        call = "C4"
    else:
        g1 = sorted({a for (a, k), c in verdict.items() if c.startswith("G1")})
        not0 = sorted({a for (a, k), c in verdict.items() if not c.startswith("G0")})
        if g1:
            call = "代わりに使えない"
            p(f"運用: **代わりに使えない** —— G1 の軸 {g1}")
        elif not0:
            call = "近いが同じではない"
            p(f"運用: **近いが同じではない** —— G0 でない軸 {not0}")
        else:
            call = "代わりに使える"
            p("運用: **代わりに使える(この2モデル・この設定で)** —— 12軸すべて両モデルで G0。既存のプールとは混ぜない")
    result["call"] = call
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generator_angle.txt").write_text("\n".join(lines) + "\n")
    json.dump(result, open(out / "generator_angle.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
