"""The stratified null, and the yardstick on the digital side.

doc/prereg_null_and_yardstick_20261031.md. The registration is closed; this
file only implements it, and it may not write C1 or C2 whatever it finds:
the main values (E2 0.9274, E3 0.7745) were already seen.

Part 1 asks whether a permutation that keeps each pseudo-class's arm mixture
fixed gives a null that sits at 0.5 instead of splitting in two.
Part 2 asks how far the author's own digital works sit from each other, on the
same scale as the analogue-side yardstick (max |d| 0.278).
"""

import argparse
import collections
import itertools
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compute_auc import AXES, auc  # noqa: E402
from multivariate_transfer import (  # noqa: E402
    A1, A2, CV_FOLDS, F9, FEATURE_SETS, PERM_SEED, Pool, arm_pool, columns,
    digest_mod, eval_E1, eval_E2, eval_E3, eval_E4, group_bootstrap, group_cv,
    halves, join, load, permute_labels_by_group, score_auc, train_test,
    works_of,
)

A2_ROWS = ["2018/ako7", "2019/ako7", "2019b/ako8", "2020/ako9", "2020b/ako10"]
A2_HELD = "2022b/ako10"          # 2 groups; declared out before the run
KNOWN_N0 = (0.4417, 0.0535, 0.9597)      # IC-1, from the previous run
KNOWN_Y_A1_MAX = 0.278                   # IC-2
BAND_MAX = 0.30                          # Pass criterion A1/A3
SAME = (0.178, 0.378)                    # Pass criteria Y1/Y2/Y3


def stratified_permutation(pool, arm_of, rng):
    """N1: draw the pseudo-class inside each arm, so the mixture is fixed.

    The unstratified permutation (N0) lets a pseudo-class fill up with one arm,
    and the fitted direction then leans on the real A1/A3 separation -- which is
    why its null came out bimodal. Here each arm contributes the same share.
    """
    groups = sorted(set(pool.g))
    arm = {g: arm_of[g] for g in groups}
    truth = {g: pool.y[pool.g == g][0] for g in groups}
    p = np.mean([truth[g] for g in groups])
    table = {}
    for a in sorted(set(arm.values())):
        gs = [g for g in groups if arm[g] == a]
        k = int(round(len(gs) * p))
        pick = set(rng.choice(gs, size=k, replace=False)) if 0 < k < len(gs) else (
            set(gs) if k >= len(gs) else set())
        for g in gs:
            table[g] = 1.0 if g in pick else 0.0
    y = np.array([table[g] for g in pool.g], dtype=float)
    return Pool(pool.X, y, pool.g, pool.r), table, arm


def arm_map(*pairs):
    """group name -> arm name, for the stratifier."""
    out = {}
    for pool, name in pairs:
        for g in set(pool.g):
            out[g] = name
    return out


def pools_for(arms, a3key, which):
    """The (train, test) pair for one evaluation, plus the arm map."""
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    if which == "E2":
        tr = [(arm_pool(a1, 1), "A1"), (arm_pool(a3, 0, h1), "A3")]
        te = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
    elif which == "E3":
        tr = [(arm_pool(a2, 1), "A2"), (arm_pool(a3, 0, h1), "A3")]
        te = join(arm_pool(a1, 1), arm_pool(a3, 0, h2))
    elif which == "E1":
        return [(w, [(arm_pool(a1, 1, a1["r"] != w), "A1"), (arm_pool(a3, 0, h1), "A3")],
                 join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a3, 0, h2)))
                for w in works_of(a1)]
    else:  # E4
        f2 = np.array([digest_mod(g, CV_FOLDS) for g in a2["g"]])
        return [(w, [(arm_pool(a1, 1, a1["r"] != w), "A1"), (arm_pool(a2, 0, f2 != i), "A2")],
                 join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a2, 0, f2 == i)))
                for i, w in enumerate(works_of(a1))]
    return [(None, tr, te)]


def stratified_null(arms, a3key, feats, which, draws, C, seed=PERM_SEED, check=None):
    rng = np.random.default_rng(seed)
    parts = pools_for(arms, a3key, which)
    out = []
    for _ in range(draws):
        per = []
        for _fold, tr_parts, te in parts:
            train = join(*[p for p, _ in tr_parts])
            amap = arm_map(*tr_parts)
            perm, table, arm = stratified_permutation(train, amap, rng)
            if check is not None:
                shares = {a: np.mean([table[g] for g in table if arm[g] == a])
                          for a in set(arm.values())}
                check.append(max(shares.values()) - min(shares.values()))
            per.append(train_test(perm, te, feats, C)["auc"])
        out.append(float(np.mean(per)))
    a = np.array(out)
    return {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "width": float(np.percentile(a, 97.5)
                                                                - np.percentile(a, 2.5)),
            "draws": len(a)}


def replay_rng_prefix(arms, a3key, rng):
    """IC-1: consume the previous run's draws in the same order, without fitting."""
    e2, e3 = eval_E2(arms, a3key, F9), eval_E3(arms, a3key, F9)
    group_bootstrap(e2, rng=rng)
    group_bootstrap(e3, rng=rng)
    a1, a3 = arms[A1], arms[a3key]
    h1, _ = halves(a3)
    for _ in range(200):
        for w in works_of(a1):
            train = join(arm_pool(a1, 1, a1["r"] != w), arm_pool(a3, 0, h1))
            permute_labels_by_group(train, rng)
    return e2


def unstratified_null_E2(arms, a3key, draws, C, rng):
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    train = join(arm_pool(a1, 1), arm_pool(a3, 0, h1))
    test = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
    vals = [train_test(permute_labels_by_group(train, rng), test, F9, C)["auc"]
            for _ in range(draws)]
    a = np.array(vals)
    return {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5))}


# ----------------------------------------------------------------- part 2

def row_pool(arm, rows, pos):
    """Two printed rows of one arm as a two-class pool (pos = label 1)."""
    keep = np.isin(arm["r"], rows)
    y = np.where(arm["r"][keep] == pos, 1.0, 0.0)
    return Pool(arm["X"][keep], y, arm["g"][keep], arm["r"][keep])


def yardstick(arm, rows, feats, tile_folds=False):
    """Pairwise distance between an arm's own rows, single-axis and multivariate."""
    single = collections.defaultdict(list)
    multi = {}
    for a, b in itertools.combinations(rows, 2):
        pool = row_pool(arm, [a, b], a)
        for ax in AXES:
            v = columns(pool, [ax])[:, 0]
            single[ax].append(abs(score_auc(v, pool.y) - 0.5))
        if tile_folds:
            folds = np.arange(len(pool.y)) % CV_FOLDS
            recs = []
            for f in range(CV_FOLDS):
                tr, te = folds != f, folds == f
                recs.append(train_test(pool.take(tr), pool.take(te), feats)["auc"])
        else:
            recs = [r["auc"] for r in group_cv(pool, feats)]
        multi[f"{a} 対 {b}"] = float(np.mean(recs))
    return single, multi


def band_of_pair(arm, a, b, axis, rng, draws=1000):
    """Group bootstrap on one pair's single-axis AUC (NC5 uses it)."""
    pool = row_pool(arm, [a, b], a)
    v = columns(pool, [axis])[:, 0]
    by_group = collections.defaultdict(list)
    for i, g in enumerate(pool.g):
        by_group[g].append(i)
    groups = np.array(sorted(by_group))
    out = []
    for _ in range(draws):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(set(pool.y[idx])) < 2:
            continue
        out.append(score_auc(v[idx], pool.y[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# -------------------------------------------------------------------- main

def part1(arms, a3key, draws, out):
    p = out.append
    res = {}
    e1 = eval_E1(arms, a3key, F9)
    real = {"E1": float(np.mean([r["auc"] for r in e1])),
            "E2": eval_E2(arms, a3key, F9)["auc"],
            "E3": eval_E3(arms, a3key, F9)["auc"],
            "E4": float(np.mean([r["auc"] for r in eval_E4(arms, F9)]))}
    C = {"E1": collections.Counter(r["C"] for r in e1).most_common(1)[0][0],
         "E2": eval_E2(arms, a3key, F9)["C"], "E3": eval_E3(arms, a3key, F9)["C"],
         "E4": collections.Counter(r["C"] for r in eval_E4(arms, F9)).most_common(1)[0][0]}

    p("# 第1部 帰無値を作り直す(層化置換)")
    p("")
    p("## IC-1 前の走行の `N0` を、同じ乱数の列で引き直す")
    rng = np.random.default_rng(PERM_SEED)
    replay_rng_prefix(arms, a3key, rng)
    n0 = unstratified_null_E2(arms, a3key, draws, C["E2"], rng)
    ok1 = all(abs(a - b) < 1e-4 for a, b in
              zip((n0["median"], n0["lo"], n0["hi"]), KNOWN_N0))
    p(f"  既知 中央 {KNOWN_N0[0]:.4f} 帯 [{KNOWN_N0[1]:.4f}, {KNOWN_N0[2]:.4f}]")
    p(f"  今回 中央 {n0['median']:.4f} 帯 [{n0['lo']:.4f}, {n0['hi']:.4f}]"
      f"  —— {'一致' if ok1 else '**一致しない**'}")
    res["IC1"], res["N0"] = ok1, n0

    p("")
    p("## IC-3 層化が効いているか(擬似クラス1に入る割合の、腕どうしの差)")
    check = []
    stratified_null(arms, a3key, F9, "E2", 5, C["E2"], check=check)
    p(f"  5回の抽選で、最大の差 {max(check):.4f}(期待 0.02 未満)")
    res["IC3"] = float(max(check))

    for name, feats in FEATURE_SETS.items():
        d = draws if name == "F9" else max(1, draws // 4)
        p("")
        p(f"## N1 層化置換 —— 特徴集合 {name}({d}回)")
        res.setdefault("N1", {})[name] = {}
        for which in ("E1", "E2", "E3", "E4"):
            n = stratified_null(arms, a3key, feats, which, d, C[which])
            outside = real[which] < n["lo"] or real[which] > n["hi"]
            p(f"  {which}  中央 {n['median']:.4f}  帯 [{n['lo']:.4f}, {n['hi']:.4f}]"
              f"  幅 **{n['width']:.4f}**   実測 {real[which]:.4f} は"
              f" {'**帯の外**' if outside else '帯の中'}")
            n["real"], n["outside"] = real[which], bool(outside)
            res["N1"][name][which] = n

    p("")
    p("## N3 信号の無い既知の対(A3 の H1 対 H2)に N1 を当てる")
    a3 = arms[a3key]
    h1, _ = halves(a3)
    pool = Pool(a3["X"], np.where(h1, 1.0, 0.0), a3["g"], a3["r"])
    rng3 = np.random.default_rng(PERM_SEED)
    vals = []
    for _ in range(max(1, draws // 4)):
        perm, _t, _a = stratified_permutation(pool, {g: "A3" for g in set(pool.g)}, rng3)
        vals.append(float(np.mean([r["auc"] for r in group_cv(perm, F9)])))
    a = np.array(vals)
    res["N3"] = {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
                 "hi": float(np.percentile(a, 97.5))}
    p(f"  中央 {res['N3']['median']:.4f}  帯 [{res['N3']['lo']:.4f}, {res['N3']['hi']:.4f}]"
      f"(期待 0.5 ± 0.05)")

    f9 = res["N1"]["F9"]
    med_ok = all(abs(f9[w]["median"] - 0.5) <= 0.05 for w in f9)
    wid_ok = all(f9[w]["width"] <= BAND_MAX for w in f9)
    n3_ok = abs(res["N3"]["median"] - 0.5) <= 0.05
    n4_ok = f9["E1"]["outside"]
    p("")
    p("## 判定(第1部。**登録 §Pass criteria(1)**)")
    p(f"  中央値が 0.5±0.05: {'満たす' if med_ok else '満たさない'}"
      f" / 帯の幅 ≤ {BAND_MAX}: {'満たす' if wid_ok else '満たさない'}"
      f" / N3 が 0.5±0.05: {'満たす' if n3_ok else '満たさない'}"
      f" / N4(E1 の実測)が帯の外: {'満たす' if n4_ok else '満たさない'}")
    if med_ok and wid_ok and n3_ok and n4_ok:
        v = "A1 帰無値として機能する"
    elif not n4_ok:
        v = "A2 細すぎる(信号まで消している)"
    elif not wid_ok:
        v = "A3 まだ広い(test AUC がこの材料に合わない)"
    else:
        v = "A4 判定保留"
    p(f"  **判定: {v}**")
    p("  **この走行では `C1` `C2` とは書かない** —— 主値は見た後だから(登録の冒頭)")
    res["verdict1"] = v
    return res


def part2(arms, out):
    p = out.append
    res = {}
    p("")
    p("# 第2部 ものさしをデジタル側でも作る")
    for side, arm, rows in (("Y-A1 アナログ(A1 の5作品)", arms[A1], works_of(arms[A1])),
                            ("Y-A2 デジタル(A2 の5行)", arms[A2], A2_ROWS)):
        single, multi = yardstick(arm, rows, F9)
        per_axis = {ax: (float(np.median(v)), float(np.max(v))) for ax, v in single.items()}
        overall = max(v[1] for v in per_axis.values())
        top = max(per_axis.items(), key=lambda kv: kv[1][1])
        p("")
        p(f"## {side}  対 {len(multi)}")
        p(f"  軸ごとの `|AUC-0.5|`: 中央の中央 "
          f"{np.median([v[0] for v in per_axis.values()]):.4f} / "
          f"**最大 {overall:.4f}**({top[0]})")
        for ax in AXES:
            p(f"    {ax:26s} 中央 {per_axis[ax][0]:.4f}  最大 {per_axis[ax][1]:.4f}")
        p(f"  多変量(9軸、群分割5折)の対ごと:")
        for k, v in sorted(multi.items(), key=lambda kv: -kv[1]):
            p(f"    {k:34s} AUC {v:.4f}")
        p(f"  多変量の中央 {np.median(list(multi.values())):.4f} / "
          f"最大 {max(multi.values()):.4f}")
        res[side.split()[0]] = {"axis": per_axis, "max": overall, "multi": multi}

    # the known value was printed to three decimals, so the tolerance is that
    ic2 = abs(res["Y-A1"]["max"] - KNOWN_Y_A1_MAX) <= 0.0005
    p("")
    p(f"## IC-2 アナログ側のものさしの引き直し: {res['Y-A1']['max']:.4f} "
      f"対 既知 {KNOWN_Y_A1_MAX:.4f} —— {'一致' if ic2 else '**一致しない**'}")
    res["IC2"] = ic2

    p("")
    p("## NC4 群でなくタイルで折を切る(誤りの再現)")
    _s, multi_tile = yardstick(arms[A2], A2_ROWS, F9, tile_folds=True)
    grp = res["Y-A2"]["multi"]
    diffs = [multi_tile[k] - grp[k] for k in grp]
    p(f"  対ごとの差(タイル折 − 群折)の中央 {np.median(diffs):+.4f} / "
      f"最大 {max(diffs):+.4f}(期待: 上がる)")
    res["NC4"] = float(np.median(diffs))

    p("")
    p(f"## NC5 `{A2_HELD}` を入れたらどうなるか(入れないと先に決めた理由)")
    rng = np.random.default_rng(PERM_SEED)
    wide = 0
    for other in A2_ROWS:
        lo, hi = band_of_pair(arms[A2], A2_HELD, other, "faint_of_drawn_ratio", rng)
        wide += (hi - lo) > 0.30
        p(f"  {A2_HELD} 対 {other:14s} faint_of_drawn_ratio の帯 [{lo:.4f}, {hi:.4f}]"
          f"  幅 {hi-lo:.4f}")
    res["NC5_wide"] = int(wide)
    p(f"  帯の幅が 0.30 を超えた対 {wide}/{len(A2_ROWS)}")

    a1m, a2m = res["Y-A1"]["max"], res["Y-A2"]["max"]
    p("")
    p("## 判定(第2部。**登録 §Pass criteria(2)**)")
    if SAME[0] <= a2m <= SAME[1]:
        v = "Y1 ものさしは工程で変わらない"
    elif a2m > SAME[1]:
        v = "Y2 デジタルの方が作品間で動く"
    else:
        v = "Y3 デジタルの方が作品間で安定"
    p(f"  アナログ側 最大 {a1m:.4f} / デジタル側 最大 {a2m:.4f} → **判定: {v}**")
    res["verdict2"] = v

    p("")
    p("## 前後の棒を両方印字する(規律10。**ものさしを足したのは床を厳しくする向き**)")
    p("  前の走行の C2 候補3本の `|d23|`(animagine)を、両方のものさしと並べる:")
    for ax, d23 in (("long_line_ratio", 0.253), ("components_per_1k_ink_px", 0.215),
                    ("long_component_ratio", 0.184)):
        p(f"    {ax:26s} |d23| {d23:.3f}   A1 のものさし "
          f"{res['Y-A1']['axis'][ax][1]:.3f}   A2 のものさし "
          f"{res['Y-A2']['axis'][ax][1]:.3f}")
    p("  **前の判定(C2 = 0本)は更新しない** —— どのみち 0 本で、救われるものも落ちるものも無い")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--model", default="animagine", choices=["animagine", "illustrious"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--part", default="both", choices=["1", "2", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    arms = load(args.csv)
    out = [f"# 帰無値の作り直しと、ものさし({args.model})",
           f"# 登録 doc/prereg_null_and_yardstick_20261031.md / seed {PERM_SEED}",
           "# **主値は見た後。この走行では C1・C2 とは書かない**", ""]
    res = {}
    if args.part in ("1", "both"):
        res.update(part1(arms, f"a3_{args.model}", args.draws, out))
    if args.part in ("2", "both"):
        res.update(part2(arms, out))
    out.append("")
    out.append(f"# 所要 {time.time()-t0:.1f} 秒")
    text = "\n".join(out)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
        pathlib.Path(args.out).with_suffix(".json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
