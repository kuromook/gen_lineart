"""Distances inside the ako lineage, on the same scale as the external pools.

doc/prereg_ako_within_series_20261031.md. The registration is closed; this file
only implements it.

The author ruled 2025/ako5 (ako5ver2koma) into the FANBOX arc on 2026-10-31,
so the lineage now holds two arcs: FANBOX (ako5, analogue-dominant, 2025) and
学祭 (ako7-ako10, digital, 2018-2022b). Arc, process, era and extraction path
all move together in that contrast -- the run measures distance, it does not
name a cause.
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
from compute_auc import AXES  # noqa: E402
from multivariate_transfer import (  # noqa: E402
    A1, A2, CV_FOLDS, F9, Pool, columns, digest_mod, load, score_auc,
    train_test,
)

# The rows, in the order the registration prints them.
ROWS = [
    ("K5  ako5(2025)", A1, "ako5ver2koma", "FANBOX編", "アナログ主体"),
    ("K7a 2018/ako7", A2, "2018/ako7", "学祭編", "デジタル"),
    ("K7b 2019/ako7", A2, "2019/ako7", "学祭編", "デジタル"),
    ("K8  2019b/ako8", A2, "2019b/ako8", "学祭編", "デジタル"),
    ("K9  2020/ako9", A2, "2020/ako9", "学祭編", "デジタル"),
    ("K10 2020b/ako10", A2, "2020b/ako10", "学祭編", "デジタル"),
]
REF_WORKS = ["fitnesskoma", "gakuenkoma", "hamlabikoma", "houseikoma"]
EXTERNAL = [("X1 animagine", "a3_animagine"), ("X2 illustrious", "a3_illustrious")]
GAKUSAI = [r[0] for r in ROWS if r[3] == "学祭編"]
KNOWN_A2_MAX_D = 0.4832          # IC-2, measured earlier today
KNOWN_GAKUSAI = (0.8947, 0.9972)  # IC-3, median and max
C_A, C_B = 0.05, 0.02             # Pass criteria, written before the run
DRAWS_SINGLE, DRAWS_MULTI, DRAWS_BOOT = 1000, 50, 1000


def subset(arms, arm_key, row):
    a = arms[arm_key]
    m = a["r"] == row
    return a["X"][m], a["g"][m]


def whole(arms, arm_key):
    a = arms[arm_key]
    return a["X"], a["g"]


def pair_pool(one, two):
    (X1, g1), (X2, g2) = one, two
    return Pool(np.concatenate([X1, X2]),
                np.concatenate([np.ones(len(g1)), np.zeros(len(g2))]),
                np.concatenate([g1, g2]),
                np.concatenate([np.full(len(g1), "pos"), np.full(len(g2), "neg")]))


def single_axis(pool):
    return {ax: score_auc(columns(pool, [ax])[:, 0], pool.y) for ax in AXES}


def permute_groups(pool, rng):
    groups = sorted(set(pool.g))
    lab = np.array([pool.y[pool.g == g][0] for g in groups])
    table = dict(zip(groups, rng.permutation(lab)))
    return np.array([table[g] for g in pool.g], dtype=float)


def single_null(pool, rng, draws=DRAWS_SINGLE):
    vals = collections.defaultdict(list)
    cols = {ax: columns(pool, [ax])[:, 0] for ax in AXES}
    for _ in range(draws):
        y = permute_groups(pool, rng)
        for ax in AXES:
            vals[ax].append(abs(score_auc(cols[ax], y) - 0.5))
    return {ax: (float(np.median(v)), float(np.percentile(v, 97.5)))
            for ax, v in vals.items()}


def multi_cv(pool, feats=F9, C=None):
    """Mean test AUC over grouped folds, plus the out-of-fold score vector.

    C is chosen inside the folds for the real data, and handed to the null as
    the registration says (§費用: the permutation gets the C the real data
    chose). Re-selecting it inside every draw costs 21 fits instead of 1.
    """
    folds = np.array([digest_mod(g, CV_FOLDS) for g in pool.g])
    per, oof, cs = [], np.full(len(pool.y), np.nan), []
    for f in range(CV_FOLDS):
        tr, te = folds != f, folds == f
        if len(set(pool.y[tr])) < 2 or len(set(pool.y[te])) < 2:
            continue
        rec = train_test(pool.take(tr), pool.take(te), feats, C)
        per.append(rec["auc"])
        cs.append(rec["C"])
        oof[te] = rec["scores"]
    mode = collections.Counter(cs).most_common(1)[0][0] if cs else 1.0
    return float(np.mean(per)), oof, len(per), mode


def boot_band(pool, oof, rng, draws=DRAWS_BOOT):
    ok = ~np.isnan(oof)
    by_group = collections.defaultdict(list)
    for i in np.where(ok)[0]:
        by_group[pool.g[i]].append(i)
    groups = np.array(sorted(by_group))
    out = []
    for _ in range(draws):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(set(pool.y[idx])) < 2:
            continue
        out.append(score_auc(oof[idx], pool.y[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def multi_null(pool, rng, C, draws=DRAWS_MULTI):
    vals = []
    for _ in range(draws):
        y = permute_groups(pool, rng)
        vals.append(multi_cv(Pool(pool.X, y, pool.g, pool.r), C=C)[0])
    return (float(np.median(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    arms = load(args.csv)
    rng = np.random.default_rng(20261031)
    out = ["# `ako` 系の中の距離(登録 doc/prereg_ako_within_series_20261031.md)",
           "# 判定の単位はタイル1枚 / この track は絵を判定しない", ""]
    p = out.append

    pools = {name: subset(arms, key, row) for name, key, row, _a, _pr in ROWS}
    for name, key in EXTERNAL:
        pools[name] = whole(arms, key)
    for w in REF_WORKS:
        pools[f"(参考) {w}"] = subset(arms, A1, w)

    p("## IC-4 行(タイル / 群)")
    for name, (X, g) in pools.items():
        p(f"  {name:22s} タイル {len(g):5d}  群 {len(set(g)):4d}")
    p("")
    p("## IC-1 自己 AUC(恒等)")
    self_pool = pair_pool(pools["K7b 2019/ako7"], pools["K7b 2019/ako7"])
    p(f"  K7b を両側に置く: {score_auc(columns(self_pool, ['ink_ratio'])[:, 0], self_pool.y):.4f}"
      "(期待 0.5000 ちょうど)")

    groups_of_pairs = [
        ("1. 編をまたぐ(FANBOX編 対 学祭編)",
         [("K5  ako5(2025)", b) for b in GAKUSAI]),
        ("2. 編が同じ(学祭編どうし)", list(itertools.combinations(GAKUSAI, 2))),
        ("3. 外部との対", [(a, x) for a, _k, _r, _ar, _p in ROWS for x, _ in EXTERNAL]),
        ("(参考) K5 対 ako 系外のアナログ4作",
         [("K5  ako5(2025)", f"(参考) {w}") for w in REF_WORKS]),
    ]

    res, multi_by_group = {}, {}
    for title, pairs in groups_of_pairs:
        p("")
        p(f"# {title}")
        rows_out = []
        for a, b in pairs:
            print(f"[{time.time()-t0:7.1f}s] {a} 対 {b}", file=sys.stderr, flush=True)
            pool = pair_pool(pools[a], pools[b])
            sa = single_axis(pool)
            d = {ax: abs(v - 0.5) for ax, v in sa.items()}
            top = max(d, key=d.get)
            nul = single_null(pool, rng)
            m, oof, nf, c_pick = multi_cv(pool)
            lo, hi = boot_band(pool, oof, rng)
            mn = multi_null(pool, rng, c_pick)
            judged = (mn[2] - mn[1]) <= 0.30
            rows_out.append((a, b, m, lo, hi, d[top], top, nul[top][1], mn, judged,
                             float(np.median(list(d.values())))))
            res[f"{a} 対 {b}"] = {
                "multi": m, "band": [lo, hi], "max_d": d[top], "max_axis": top,
                "median_d": float(np.median(list(d.values()))),
                "null_multi": list(mn), "judged": bool(judged),
                "single": {ax: sa[ax] for ax in AXES},
                "null_single_97.5": {ax: nul[ax][1] for ax in AXES},
            }
        multi_by_group[title] = [r[2] for r in rows_out]
        for a, b, m, lo, hi, dmax, top, nul97, mn, judged, dmed in sorted(
                rows_out, key=lambda r: -r[2]):
            p(f"  {a:22s} 対 {b:22s} 多変量 **{m:.4f}** 帯 [{lo:.4f}, {hi:.4f}]"
              f"  単軸 |d| 中央 {dmed:.3f} 最大 {dmax:.3f}({top}、置換の 97.5% {nul97:.3f})"
              + ("" if judged else "   **帯が足りない**"))
        p(f"  → 中央 {np.median([r[2] for r in rows_out]):.4f} / "
          f"最大 {max(r[2] for r in rows_out):.4f} / 最小 {min(r[2] for r in rows_out):.4f}")

    p("")
    p("## IC-2 / IC-3 既知の値との一致")
    gak = multi_by_group["2. 編が同じ(学祭編どうし)"]
    a2_max = max(v["max_d"] for k, v in res.items()
                 if all(x in GAKUSAI for x in k.split(" 対 ")))
    p(f"  学祭編どうしの多変量: 中央 {np.median(gak):.4f} / 最大 {max(gak):.4f}"
      f"   既知 {KNOWN_GAKUSAI[0]:.4f} / {KNOWN_GAKUSAI[1]:.4f}")
    p(f"  学祭編どうしの単軸 最大 |d|: {a2_max:.4f}   既知 {KNOWN_A2_MAX_D:.4f}")

    cross = multi_by_group["1. 編をまたぐ(FANBOX編 対 学祭編)"]
    ext = multi_by_group["3. 外部との対"]
    ako_pairs = cross + gak
    diff = float(np.median(cross) - np.median(gak))
    p("")
    p("## 判定(登録 §Pass criteria)")
    p(f"  M(編またぎ) {np.median(cross):.4f} − M(編同じ) {np.median(gak):.4f} = **{diff:+.4f}**")
    if diff >= C_A:
        v1 = "C-A 編が見える"
    elif diff <= C_B:
        v1 = "C-B 編は埋もれる"
    else:
        v1 = "C-C 判定保留"
    p(f"  **判定: {v1}**")
    p(f"  `ako` どうしの最大 {max(ako_pairs):.4f} 対 `ako` 対外部の最小 {min(ext):.4f}")
    v2 = "C-D 作者の中が外部との差を超える" if max(ako_pairs) > min(ext) else \
         "C-E 超えない"
    p(f"  **判定: {v2}**")
    res["verdict"] = [v1, v2, diff]

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
