"""Does the rough side split into the two hills the author describes?

doc/prereg_rough_side_mixture_20261101.md. The registration is closed; this
file only implements it.

Each work is cut in two by its groups, then each half is measured against the
roughs of the works the author called analogue, with the work itself held out
of that pool. The read is the gap between the two halves, compared with the
same gap taken inside the analogue works. The split is a mixture of two
diagonal Gaussians fitted by EM, written here because sklearn is not installed
in this environment -- so it carries its own check against data with a known
answer, printed before anything else.
"""

import argparse
import collections
import csv
import itertools
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compute_auc import AXES  # noqa: E402
from multivariate_transfer import CV_FOLDS, F9, digest_mod, load  # noqa: E402
from ako_series_auc import boot_band, multi_cv, pair_pool  # noqa: E402

BAND_MAX = 0.30
C_GAP = 0.05               # registered bar for C1, and for NC3
SEED = 20261101
STAGE2_JSON = "results/process_stage2_20261101/process_stage2.json"
F9_IDX = [AXES.index(a) for a in F9]
INK_IDX = AXES.index("ink_ratio")


def gmm_diag(X, k, rng, iters=300, reg=1e-6):
    """EM for a k-component diagonal-covariance Gaussian mixture. Returns
    hard labels, BIC and the log-likelihood."""
    n, d = X.shape
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=p)])
    mu = np.array(centers, dtype=float)
    var = np.tile(X.var(0) + reg, (k, 1))
    pi = np.full(k, 1.0 / k)
    ll_old, r, ll = -np.inf, np.full((n, k), 1.0 / k), -np.inf
    for _ in range(iters):
        lp = (-0.5 * (((X[:, None, :] - mu[None]) ** 2) / var[None]).sum(-1)
              - 0.5 * np.log(var).sum(1)[None] - 0.5 * d * np.log(2 * np.pi)
              + np.log(pi)[None])
        m = lp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        r = np.exp(lp - lse[:, None])
        ll = float(lse.sum())
        nk = r.sum(0) + 1e-12
        pi = nk / n
        mu = (r.T @ X) / nk[:, None]
        var = (r.T @ (X ** 2)) / nk[:, None] - mu ** 2 + reg
        if ll - ll_old < 1e-7 * abs(ll):
            break
        ll_old = ll
    n_par = k * 2 * d + (k - 1)
    return r.argmax(1), float(-2 * ll + n_par * np.log(n)), ll


def gmm_check(rng):
    """Instrument check from outside the pipeline: two Gaussians 6 sigma apart
    must be recovered, and BIC must prefer two components over one."""
    a = rng.normal(0, 1, (300, 3))
    b = rng.normal(6, 1, (300, 3))
    X = np.concatenate([a, b])
    truth = np.concatenate([np.zeros(300), np.ones(300)])
    lab, bic2, _ = gmm_diag(X, 2, rng)
    _l1, bic1, _ = gmm_diag(X, 1, rng)
    acc = max((lab == truth).mean(), (lab != truth).mean())
    return acc, bic1 - bic2


def load_rough(profile_csv, manifest_csv):
    meta = {r["tile"]: r for r in csv.DictReader(open(manifest_csv))}
    by_work = collections.defaultdict(list)
    for rec in csv.DictReader(open(profile_csv)):
        stem = rec["sample"].rsplit(".", 1)[0]
        by_work[meta[stem]["work"]].append((rec, meta[stem]))
    pools, facts = {}, {}
    for work, items in by_work.items():
        X = np.array([[float(r[a]) for a in AXES] for r, _ in items], dtype=float)
        g = np.array([m["group"] for _, m in items])
        pools[work] = (X, g)
        m = items[0][1]
        facts[work] = {"rough": m["rough"], "line_process": m["line_process"],
                       "device": m["device"], "tag": m["tag"],
                       "tiles": len(items), "groups": len(set(g))}
    return pools, facts


def group_means(X, g, cols):
    gs = sorted(set(g))
    return gs, np.array([X[g == k][:, cols].mean(0) for k in gs])


def split_work(X, g, how, rng, cols, k=2):
    """Return a boolean mask over tiles, plus (delta BIC 1-vs-2, BIC of 3)."""
    if how == "parity":                                  # NC3
        return np.array([digest_mod(x, 2) == 0 for x in g]), (float("nan"),) * 2
    if how == "tile":                                    # NC2
        Z = X[:, cols]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        lab, bic2, _ = gmm_diag(Z, 2, rng)
        _l, bic1, _ = gmm_diag(Z, 1, rng)
        return lab == 0, (bic1 - bic2, float("nan"))
    gs, M = group_means(X, g, cols)                      # main arm, and NC4
    M = (M - M.mean(0)) / (M.std(0) + 1e-12)
    lab, bic2, _ = gmm_diag(M, 2, rng)
    _l, bic1, _ = gmm_diag(M, 1, rng)
    _l3, bic3, _ = gmm_diag(M, 3, rng) if len(gs) > 6 else (None, float("nan"), None)
    table = dict(zip(gs, lab))
    return np.array([table[x] == 0 for x in g]), (bic1 - bic2, bic2 - bic3)


def one_sided_p(null, real):
    return float((np.sum(null >= real) + 1) / (len(null) + 1))


def measurable(pl):
    """Can multi_cv fit this pair at all? (amendment 1, 2026-11-02)

    Mirrors the fold rules of multi_cv (outer) and select_C (inner): every
    usable outer fold must leave at least one inner fold with both labels,
    otherwise select_C returns C=None. Decided from the groups alone.
    """
    folds = np.array([digest_mod(g, CV_FOLDS) for g in pl.g])
    outer = 0
    for f in range(CV_FOLDS):
        tr, te = folds != f, folds == f
        if len(set(pl.y[tr])) < 2 or len(set(pl.y[te])) < 2:
            continue
        outer += 1
        if not any(len(set(pl.y[tr & (folds != h)])) == 2
                   and len(set(pl.y[tr & (folds == h)])) == 2
                   for h in range(CV_FOLDS)):
            return False
    return outer > 0


def run_arm(name, how, pools, facts, works, analogue, rng, t0, out, cols):
    p = out.append
    p("")
    p(f"### {name}")
    rows, dropped_cv = {}, []
    for w in works:
        X, g = pools[w]
        mask, bics = split_work(X, g, how, rng, cols)
        if mask.all() or (~mask).all():
            p(f"  {w:20s} 片側が空(割れなかった)")
            continue
        pool_works = [a for a in analogue if a != w]
        PX = np.concatenate([pools[a][0] for a in pool_works])
        Pg = np.concatenate([pools[a][1] for a in pool_works])
        pls = [pair_pool((X[h], g[h]), (PX, Pg)) for h in (mask, ~mask)]
        if not all(measurable(pl) for pl in pls):
            p(f"  {w:20s} 山 {int(mask.sum())}/{int((~mask).sum())}枚 —— "
              "**山が群の5分割で測れない → 外す**(修正1)")
            dropped_cv.append(w)
            continue
        aucs, bands = [], []
        for half, pl in zip((mask, ~mask), pls):
            print(f"[{time.time()-t0:7.1f}s] {name} {w} 半分 {int(half.sum())}枚",
                  file=sys.stderr, flush=True)
            m, oof, _nf, _c = multi_cv(pl)
            aucs.append(m)
            bands.append(boot_band(pl, oof, rng))
        gap = abs(aucs[0] - aucs[1])
        wide = max(hi - lo for lo, hi in bands) > BAND_MAX
        rows[w] = {"gap": gap, "auc": aucs, "bands": bands,
                   "n": [int(mask.sum()), int((~mask).sum())],
                   "dbic_1v2": bics[0], "dbic_2v3": bics[1], "wide": bool(wide)}
        p(f"  {w:20s} 下絵 {facts[w]['rough']:5s} 札 {facts[w]['tag']:6s}"
          f" 山 {int(mask.sum()):4d}/{int((~mask).sum()):4d}枚"
          f"  AUC {aucs[0]:.4f} [{bands[0][0]:.4f}, {bands[0][1]:.4f}]"
          f" / {aucs[1]:.4f} [{bands[1][0]:.4f}, {bands[1][1]:.4f}]"
          f"  **差 {gap:.4f}**  ΔBIC(1→2) {bics[0]:+.1f}"
          + ("   **帯が足りない**" if wide else ""))
    mixed = [r["gap"] for w, r in rows.items() if facts[w]["rough"] == "混在"]
    anal = [r["gap"] for w, r in rows.items() if facts[w]["rough"] == "アナログ"]
    read = float(np.median(mixed) - np.median(anal)) if mixed and anal else float("nan")
    p(f"  → 混在 {len(mixed)}作 中央 {np.median(mixed):.4f}"
      f" / アナログ {len(anal)}作 中央 {np.median(anal):.4f}"
      f"  **読む差 = {read:+.4f}**")
    ws = [w for w in works if w in rows]
    gaps = np.array([rows[w]["gap"] for w in ws])
    n_mix = sum(1 for w in ws if facts[w]["rough"] == "混在")
    null, drop = [], 0
    for pick in itertools.combinations(range(len(ws)), n_mix):
        m = np.zeros(len(ws), dtype=bool)
        m[list(pick)] = True
        if m.all() or (~m).all():
            drop += 1
            continue
        null.append(float(np.median(gaps[m]) - np.median(gaps[~m])))
    null = np.array(null)
    pv = one_sided_p(null, read)
    p(f"  山が測れず外した作品(修正1) {len(dropped_cv)}: {', '.join(dropped_cv) or 'なし'}"
      f" / 帰無に残った作品 {len(ws)}(混在 {n_mix})")
    p(f"  交換の帰無 {len(null)}+外し{drop} 通り: 中央 {np.median(null):+.4f}"
      f" / 97.5% {np.percentile(null, 97.5):+.4f} / 片側 p = {pv:.4f}")
    wide_mixed = sum(1 for w, r in rows.items()
                     if r["wide"] and facts[w]["rough"] == "混在")
    p(f"  帯が足りない作品: 混在 {wide_mixed} / 全 {sum(r['wide'] for r in rows.values())}")
    return {"rows": rows, "read": read, "p": pv, "wide_mixed": wide_mixed,
            "dropped_cv": dropped_cv,
            "null_975": float(np.percentile(null, 97.5)),
            "mixed_median": float(np.median(mixed)) if mixed else None,
            "analogue_median": float(np.median(anal)) if anal else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="results/rough_mixture_20261101/profile_rough_norm.csv")
    ap.add_argument("--raw", default="results/rough_mixture_20261101/profile_rough_raw.csv")
    ap.add_argument("--manifest", default="data/author_rough_20261101/pool_manifest.csv")
    ap.add_argument("--line", default="results/process_stage2_20261101/profile_s2.csv")
    ap.add_argument("--line-manifest", default="data/author_process_20261101/pool_manifest.csv")
    ap.add_argument("--out", default="results/rough_mixture_20261101/rough_mixture.txt")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = ["# 下絵の側の混合(登録 doc/prereg_rough_side_mixture_20261101.md)",
           "# 判定の単位はタイル1枚 / この track は絵を判定しない", ""]
    p = out.append
    summary = {}

    acc, dbic = gmm_check(rng)
    p("## 登録外の器具点検 —— 答えの分かっている材料で GMM を回す(sklearn が無いので自作)")
    p(f"  6σ 離れた2つの正規分布 600点: 当たり {acc:.4f}(期待 1.0000)"
      f" / ΔBIC(1→2) {dbic:+.1f}(期待 大きく正)"
      f"  → {'通った' if acc > 0.99 and dbic > 0 else '**通らない**'}")
    summary["gmm_check"] = {"acc": acc, "dbic": dbic}

    pools, facts = load_rough(args.norm, args.manifest)
    works = sorted(w for w in facts
                   if facts[w]["line_process"] == "デジタル"
                   and facts[w]["device"] == "未記載")
    dropped = sorted(set(facts) - set(works))
    analogue = [w for w in works if facts[w]["rough"] == "アナログ"]
    p("")
    p(f"## IC3 母集団 {len(works)}作(混在 {len(works)-len(analogue)} / アナログ {len(analogue)})")
    for w in works:
        f = facts[w]
        p(f"  {w:20s} 下絵 {f['rough']:5s} 札 {f['tag']:6s}"
          f" タイル {f['tiles']:4d} 群 {f['groups']:4d}")
    p(f"  登録に従って外した作品 {len(dropped)}: {', '.join(dropped)}")
    p("  落ちたタイルと紙の水準は data/author_rough_20261101/build_report.json(墨0が1枚)")

    # IC5: the line-art side must reproduce stage 2 exactly.
    p("")
    p("## IC5 線画の側が第2段と一致するか(`multi_cv` は決定的)")
    lp, lf = load_rough(args.line, args.line_manifest)
    old = json.loads(pathlib.Path(STAGE2_JSON).read_text())["pairs"]
    a, b = sorted(works)[0], sorted(works)[1]
    got = multi_cv(pair_pool(lp[a], lp[b]))[0]
    key = f"{a} 対 {b}"
    exp = old[key]["multi"] if key in old else float("nan")
    p(f"  {key}: いま {got:.6f} / 第2段 {exp:.6f} / 差 {abs(got-exp):.3e}"
      f" → {'通った' if abs(got - exp) <= 1e-9 else '**通らない**'}")
    summary["IC5"] = {"pair": key, "now": got, "stage2": exp}

    # IC1
    pl = pair_pool(pools[works[0]], pools[works[0]])
    from multivariate_transfer import columns  # noqa: E402
    from multivariate_transfer import score_auc as _sa  # noqa: E402
    ic1 = _sa(columns(pl, ["ink_ratio"])[:, 0], pl.y)
    p("")
    p(f"## IC1 同じ山を両側に置く: {ic1:.4f}(期待 0.5000)"
      f" {'通った' if abs(ic1-0.5) < 1e-9 else '**通らない**'}")

    p("")
    p("## 主値と NC")
    summary["main_norm"] = run_arm("主の腕(白点を揃えた側・群単位の2成分 GMM)",
                                   "group", pools, facts, works, analogue, rng, t0,
                                   out, F9_IDX)
    summary["NC3_parity"] = run_arm("NC3 群の偶奇で割る(割り方そのものの対照)",
                                    "parity", pools, facts, works, analogue, rng, t0,
                                    out, F9_IDX)
    summary["NC4_ink"] = run_arm("NC4 ink_ratio 1本だけで割る",
                                 "group", pools, facts, works, analogue, rng, t0,
                                 out, [INK_IDX])
    summary["NC2_tile"] = run_arm("NC2 群を壊してタイル単位で割る",
                                  "tile", pools, facts, works, analogue, rng, t0,
                                  out, F9_IDX)

    # IC4: the same read on the un-equalized pixels.
    pools_raw, facts_raw = load_rough(args.raw, args.manifest)
    summary["main_raw"] = run_arm("IC4 白点を揃えない側(同じ読み)",
                                  "group", pools_raw, facts_raw, works, analogue,
                                  rng, t0, out, F9_IDX)

    # NC5: how often is three components better than two?
    p("")
    p("## NC5 成分を3にすると良くなる作品の数")
    better = [w for w in works
              if summary["main_norm"]["rows"].get(w, {}).get("dbic_2v3", -1) > 0]
    p(f"  2→3 で BIC が良くなった作品 {len(better)} / {len(works)}: {', '.join(better)}")
    p(f"  読み方(登録): 半分を超えたら「2つの山」という読みを取り下げる"
      f" → {'**取り下げる**' if len(better) > len(works)/2 else '取り下げない'}")
    summary["NC5"] = {"better_with_3": better}

    # verdict
    m, nc3, nc4, raw = (summary["main_norm"], summary["NC3_parity"],
                        summary["NC4_ink"], summary["main_raw"])
    p("")
    p("## 判定(登録 §Pass criteria)")
    c1 = bool(m["read"] >= C_GAP and m["read"] > m["null_975"])
    c4 = bool(nc3["read"] >= C_GAP or len(better) > len(works) / 2)
    p(f"  C1 混合: 読む差 {m['read']:+.4f} 棒 {C_GAP:+.4f}"
      f" 帰無 97.5% {m['null_975']:+.4f} p {m['p']:.4f} → **{'C1 混合が見える' if c1 else 'C2 届かない'}**")
    if not c1:
        p("     **「下絵は映らない」とは書かない** —— 書けるのは「この器具では混合は見えない」まで")
    p(f"  C3 紙: 揃えた側 {m['read']:+.4f} 対 揃えない側 {raw['read']:+.4f}"
      f" → **{'紙の色の話 —— 揃えない側だけに在る' if (raw['read'] >= C_GAP and not c1) else '紙だけの話ではない'}**")
    p(f"  C4 割り方の産物: NC3 {nc3['read']:+.4f}(棒 {C_GAP:+.4f})"
      f" / 3成分が良い作品 {len(better)} → **{'倒れた' if c4 else '倒れていない'}**")
    p(f"  NC4 1軸だけ: {nc4['read']:+.4f} —— 主値に並ぶなら見ているのは墨の密度")
    p(f"  C5 判定保留: 混在のうち帯が足りない作品 {m['wide_mixed']}"
      f" → **{'判定保留' if m['wide_mixed'] >= 4 else '読める'}**")
    summary["verdict"] = {"C1": c1, "C4_topple": c4,
                          "C5_hold": bool(m["wide_mixed"] >= 4)}

    p("")
    p(f"# 所要 {time.time() - t0:.1f} 秒")
    pathlib.Path(args.out).write_text("\n".join(out) + "\n")
    pathlib.Path(args.out).with_suffix(".json").write_text(
        json.dumps({"summary": summary, "facts": facts}, indent=1,
                   ensure_ascii=False, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    main()
