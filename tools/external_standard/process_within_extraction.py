"""Does the author's line-art process split tiles more than the arc does?

doc/prereg_process_within_extraction_20261031.md. The registration is closed;
this file only implements it. The extraction is fixed to clip_pairs_v3, so the
extraction path is held still -- and therefore cannot be measured here.

Every pair's multivariate AUC is computed once. The nulls for the cell
differences are exact enumerations of the work-level label exchange (165 ways
for the process, 4,620 for the arc), which only reassign cells and never
recompute an AUC, so they are free. The author's ruling is given per work, so
the work is the unit the exchange has to use.
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
from multivariate_transfer import (  # noqa: E402
    A2, CV_FOLDS, Pool, columns, digest_mod, load, score_auc,
)
from ako_series_auc import (  # noqa: E402
    boot_band, multi_cv, multi_null, pair_pool, permute_groups, single_axis,
    single_null,
)

TAGS = ["2015", "2016", "2017", "2018", "2019", "2019b", "2020", "2020b",
        "2021", "2022", "2022b", "2023b", "2024", "2024b", "2024c"]
MATCH = 3                      # registered: era distance <= 3 is "matched"
BAND_MAX = 0.30                # registered: wider than this is "帯が足りない"
C_D = 0.05                     # registered bar for both C1 and C2
KNOWN_GAKUSAI = (0.8947, 0.9972)   # IC2, from doc/result_ako_series_20261031.md
IC2_TOL = 5e-4
GAKUSAI_A2 = ["2018/ako7", "2019/ako7", "2019b/ako8", "2020/ako9", "2020b/ako10"]
EXTERNAL = [("X1 animagine", "a3_animagine"), ("X2 illustrious", "a3_illustrious")]
SEED = 20261031


def era(work_tag):
    return TAGS.index(work_tag)


def load_ruled(profile_csv, manifest_csv):
    """The eleven ruled works, joined to the author's rulings by tile name."""
    meta = {r["tile"]: r for r in csv.DictReader(open(manifest_csv))}
    by_work = collections.defaultdict(list)
    for rec in csv.DictReader(open(profile_csv)):
        stem = rec["sample"].rsplit(".", 1)[0]
        m = meta[stem]
        by_work[m["work"]].append((rec, m))
    pools, facts = {}, {}
    for work, items in by_work.items():
        X = np.array([[float(r[a]) for a in AXES] for r, _ in items], dtype=float)
        g = np.array([m["group"] for _, m in items])
        pools[work] = (X, g)
        m = items[0][1]
        facts[work] = {"arc": m["arc"], "process": m["line_process"],
                       "device": m["device"], "tag": m["tag"],
                       "tiles": len(items), "groups": len(set(g))}
    return pools, facts


def cell_of(a, b, facts, key_process="process", key_arc="arc"):
    same_p = facts[a][key_process] == facts[b][key_process]
    same_a = facts[a][key_arc] == facts[b][key_arc]
    return "S1" if (same_p and same_a) else "S2" if same_p else \
           "S3" if same_a else "S4"


def median(vals):
    return float(np.median(vals)) if len(vals) else float("nan")


def diff_from_labels(pairs_auc, dist, labels, arcs, matched_only):
    """median(S3)-median(S1) and median(S2)-median(S1) under given labels."""
    cells = collections.defaultdict(list)
    for (a, b), auc in pairs_auc.items():
        if matched_only and dist[(a, b)] > MATCH:
            continue
        same_p = labels[a] == labels[b]
        same_a = arcs[a] == arcs[b]
        k = "S1" if (same_p and same_a) else "S2" if same_p else \
            "S3" if same_a else "S4"
        cells[k].append(auc)
    if not cells["S1"]:
        return float("nan"), float("nan")
    return (median(cells["S3"]) - median(cells["S1"]),
            median(cells["S2"]) - median(cells["S1"]))


def exchange_null_process(pairs_auc, dist, facts, matched_only):
    works = sorted(facts)
    arcs = {w: facts[w]["arc"] for w in works}
    n_analogue = sum(1 for w in works if facts[w]["process"] == "アナログ")
    out, dropped = [], 0
    for pick in itertools.combinations(works, n_analogue):
        lab = {w: ("アナログ" if w in pick else "デジタル") for w in works}
        d, _ = diff_from_labels(pairs_auc, dist, lab, arcs, matched_only)
        if np.isnan(d):
            dropped += 1          # that assignment leaves S1 or S3 empty
        else:
            out.append(d)
    return np.array(out), dropped


def exchange_null_arc(pairs_auc, dist, facts, matched_only):
    works = sorted(facts)
    procs = {w: facts[w]["process"] for w in works}
    multiset = sorted(facts[w]["arc"] for w in works)
    seen, out, dropped = set(), [], 0
    for perm in set(itertools.permutations(multiset)):
        arcs = dict(zip(works, perm))
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        _, d = diff_from_labels(pairs_auc, dist, procs, arcs, matched_only)
        if np.isnan(d):
            dropped += 1
        else:
            out.append(d)
    return np.array(out), dropped


def one_sided_p(null, real):
    return float((np.sum(null >= real) + 1) / (len(null) + 1))


def pair_row(pools, a, b, rng, t0, tag=""):
    print(f"[{time.time() - t0:7.1f}s] {tag}{a} 対 {b}", file=sys.stderr, flush=True)
    pool = pair_pool(pools[a], pools[b])
    sa = single_axis(pool)
    d = {ax: abs(v - 0.5) for ax, v in sa.items()}
    top = max(d, key=d.get)
    nul = single_null(pool, rng)
    m, oof, _nf, c_pick = multi_cv(pool)
    lo, hi = boot_band(pool, oof, rng)
    mn = multi_null(pool, rng, c_pick)
    return {"multi": m, "band": [lo, hi], "null_multi": list(mn),
            "judged": bool((mn[2] - mn[1]) <= BAND_MAX),
            "max_d": d[top], "max_axis": top,
            "median_d": float(np.median(list(d.values()))),
            "null_single_97.5": nul[top][1], "C": c_pick}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/process_within_20261031/profile_ruled.csv")
    ap.add_argument("--manifest", default="data/author_process_20261031/pool_manifest.csv")
    ap.add_argument("--third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--out", default="results/process_within_20261031/process_within.txt")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pools, facts = load_ruled(args.csv, args.manifest)
    arms = load(args.third)
    for name, key in EXTERNAL:
        pools[name] = (arms[key]["X"], arms[key]["g"])

    works = sorted(facts)
    dist = {(a, b): abs(era(facts[a]["tag"]) - era(facts[b]["tag"]))
            for a, b in itertools.combinations(works, 2)}

    out = ["# 同じ抽出の中の工程と編(登録 doc/prereg_process_within_extraction_20261031.md)",
           "# 判定の単位はタイル1枚 / この track は絵を判定しない", ""]
    p = out.append

    p("## IC3 採用した行(タイル / 群 / 頁は build_report.json)")
    for w in works:
        f = facts[w]
        p(f"  {w:16s} {f['arc']:8s} 線画 {f['process']:5s} 機材 {f['device']:10s}"
          f" 札 {f['tag']:6s} タイル {f['tiles']:4d} 群 {f['groups']:4d}")
    p(f"  計 タイル {sum(f['tiles'] for f in facts.values())}"
      f" 群 {sum(f['groups'] for f in facts.values())}")
    for name, key in EXTERNAL:
        p(f"  {name:16s} タイル {len(arms[key]['g']):5d} 群 {len(set(arms[key]['g'])):4d}")
    p("")

    p("## IC1 / NC1 自己 AUC(恒等)")
    self_pool = pair_pool(pools[works[0]], pools[works[0]])
    ic1 = score_auc(columns(self_pool, ["ink_ratio"])[:, 0], self_pool.y)
    p(f"  {works[0]} を両側に置く(ink_ratio): {ic1:.4f}(期待 0.5000 ちょうど)"
      f"  {'通った' if abs(ic1 - 0.5) < 1e-9 else '**通らない**'}")
    p("")

    # --- the 55 own pairs ---
    pairs_auc, res = {}, {}
    for a, b in itertools.combinations(works, 2):
        r = pair_row(pools, a, b, rng, t0)
        r["cell"] = cell_of(a, b, facts)
        r["era"] = dist[(a, b)]
        pairs_auc[(a, b)] = r["multi"]
        res[f"{a} 対 {b}"] = r
        pathlib.Path(args.out).with_suffix(".partial.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False, default=float))

    cells = collections.defaultdict(list)
    for (a, b), auc in pairs_auc.items():
        cells[cell_of(a, b, facts)].append((a, b, auc, dist[(a, b)]))

    NAMES = {"S1": "S1 工程が同じ・編が同じ", "S2": "S2 工程が同じ・編がちがう",
             "S3": "S3 工程がちがう・編が同じ", "S4": "S4 工程もちがう・編もちがう"}
    for k in ["S1", "S2", "S3", "S4"]:
        p("")
        p(f"# {NAMES[k]}({len(cells[k])}対)")
        for a, b, auc, dd in sorted(cells[k], key=lambda r: -r[2]):
            r = res[f"{a} 対 {b}"]
            flag = "" if r["judged"] else "   **帯が足りない**"
            p(f"  {a:16s} 対 {b:16s} 多変量 **{auc:.4f}**"
              f" 帯 [{r['band'][0]:.4f}, {r['band'][1]:.4f}] 時期距離 {dd:2d}"
              f"  単軸 最大 {r['max_d']:.3f}({r['max_axis']}、置換 97.5% {r['null_single_97.5']:.3f})"
              f"{flag}")
        aucs = [r[2] for r in cells[k]]
        eras = [r[3] for r in cells[k]]
        p(f"  → 中央 {median(aucs):.4f} / 最大 {max(aucs):.4f} / 最小 {min(aucs):.4f}"
          f" / 時期距離の中央 {median(eras):.1f}")

    # --- external pairs ---
    p("")
    p("# S5 外部との対(22対)")
    ext_rows = []
    for w in works:
        for name, _k in EXTERNAL:
            r = pair_row(pools, w, name, rng, t0, tag="外部 ")
            res[f"{w} 対 {name}"] = r
            ext_rows.append((w, name, r))
    for w, name, r in sorted(ext_rows, key=lambda t: -t[2]["multi"]):
        flag = "" if r["judged"] else "   **帯が足りない**"
        p(f"  {w:16s} 対 {name:16s} 多変量 **{r['multi']:.4f}**"
          f" 帯 [{r['band'][0]:.4f}, {r['band'][1]:.4f}]"
          f"  単軸 最大 {r['max_d']:.3f}({r['max_axis']}){flag}")
    ext = [r["multi"] for _w, _n, r in ext_rows]
    own = list(pairs_auc.values())
    p(f"  → 中央 {median(ext):.4f} / 最大 {max(ext):.4f} / 最小 {min(ext):.4f}")

    # --- the registered differences and their exchange nulls ---
    p("")
    p("## 主値 —— セルの差と、作品ラベルの交換の帰無(厳密な数え上げ)")
    summary = {}
    for matched in (False, True):
        lab = {w: facts[w]["process"] for w in works}
        arcs = {w: facts[w]["arc"] for w in works}
        d_proc, d_arc = diff_from_labels(pairs_auc, dist, lab, arcs, matched)
        np_, np_drop = exchange_null_process(pairs_auc, dist, facts, matched)
        na_, na_drop = exchange_null_arc(pairs_auc, dist, facts, matched)
        pp = one_sided_p(np_, d_proc)
        pa = one_sided_p(na_, d_arc)
        tag = "matched(時期距離 ≤ 3)" if matched else "素(全対)"
        sub = collections.defaultdict(int)
        for (a, b) in pairs_auc:
            if not matched or dist[(a, b)] <= MATCH:
                sub[cell_of(a, b, facts)] += 1
        p("")
        p(f"### {tag} —— 対の数 S1 {sub['S1']} / S2 {sub['S2']} / S3 {sub['S3']} / S4 {sub['S4']}")
        p(f"  **D(工程) = 中央(S3) − 中央(S1) = {d_proc:+.4f}**"
          f"   交換の帰無 {len(np_)}+外し{np_drop} 通り: 中央 {np.median(np_):+.4f}"
          f" / 97.5% {np.percentile(np_, 97.5):+.4f} / 片側 p = {pp:.4f}")
        p(f"  **D(編)   = 中央(S2) − 中央(S1) = {d_arc:+.4f}**"
          f"   交換の帰無 {len(na_)}+外し{na_drop} 通り: 中央 {np.median(na_):+.4f}"
          f" / 97.5% {np.percentile(na_, 97.5):+.4f} / 片側 p = {pa:.4f}")
        summary[tag] = {"d_process": d_proc, "d_arc": d_arc, "p_process": pp,
                        "p_arc": pa, "exchange_dropped": [np_drop, na_drop],
                        "null_process_975": float(np.percentile(np_, 97.5)),
                        "null_arc_975": float(np.percentile(na_, 97.5)),
                        "pairs": dict(sub)}

    # --- NC2 group-breaking ---
    p("")
    p("## NC2 群を壊す(タイル単位で置換)")
    a, b = "2023b/akocultB", "2024/ako3B"
    pool = pair_pool(pools[a], pools[b])
    _m, _oof, _nf, c_pick = multi_cv(pool)
    tile_null = []
    for _ in range(50):
        y = rng.permutation(pool.y)
        tile_null.append(multi_cv(Pool(pool.X, y, pool.g, pool.r), C=c_pick)[0])
    gn = res[f"{a} 対 {b}"]["null_multi"]
    p(f"  {a} 対 {b}")
    p(f"  群単位の帰無: 中央 {gn[0]:.4f} 帯 [{gn[1]:.4f}, {gn[2]:.4f}](幅 {gn[2]-gn[1]:.4f})")
    p(f"  タイル単位の帰無: 中央 {np.median(tile_null):.4f}"
      f" 帯 [{np.percentile(tile_null, 2.5):.4f}, {np.percentile(tile_null, 97.5):.4f}]"
      f"(幅 {np.percentile(tile_null, 97.5) - np.percentile(tile_null, 2.5):.4f})")
    p("  期待(登録): タイル単位は帯が狭くなり中央が 0.5 から上にずれる。発火が正常")
    summary["NC2"] = {"group_null": gn, "tile_null_median": float(np.median(tile_null)),
                      "tile_null_band": [float(np.percentile(tile_null, 2.5)),
                                         float(np.percentile(tile_null, 97.5))]}

    # --- NC3 device ---
    p("")
    p("## NC3 機材(iPadと明記 / 未記載)—— デジタル8作の中だけ")
    dig = [w for w in works if facts[w]["process"] == "デジタル"]
    dpairs = {k: v for k, v in pairs_auc.items() if k[0] in dig and k[1] in dig}
    same = [v for k, v in dpairs.items() if facts[k[0]]["device"] == facts[k[1]]["device"]]
    diff = [v for k, v in dpairs.items() if facts[k[0]]["device"] != facts[k[1]]["device"]]
    e_same = [dist[k] for k in dpairs if facts[k[0]]["device"] == facts[k[1]]["device"]]
    e_diff = [dist[k] for k in dpairs if facts[k[0]]["device"] != facts[k[1]]["device"]]
    d_dev = median(diff) - median(same)
    n_ipad = sum(1 for w in dig if facts[w]["device"] == "iPadと明記")
    dev_null = []
    for pick in itertools.combinations(sorted(dig), n_ipad):
        s = [v for k, v in dpairs.items() if (k[0] in pick) == (k[1] in pick)]
        t = [v for k, v in dpairs.items() if (k[0] in pick) != (k[1] in pick)]
        if s and t:
            dev_null.append(median(t) - median(s))
    dev_null = np.array(dev_null)
    p(f"  機材が同じ {len(same)}対 中央 {median(same):.4f}(時期距離の中央 {median(e_same):.1f})")
    p(f"  機材がちがう {len(diff)}対 中央 {median(diff):.4f}(時期距離の中央 {median(e_diff):.1f})")
    p(f"  **D(機材) = {d_dev:+.4f}** 交換の帰無 {len(dev_null)} 通り:"
      f" 97.5% {np.percentile(dev_null, 97.5):+.4f} / 片側 p = {one_sided_p(dev_null, d_dev):.4f}")
    p(f"  外した割り当て(S1 か S3 が空): {len(list(itertools.combinations(sorted(dig), n_ipad))) - len(dev_null)}")
    p("  読み方(登録): 発火したら「工程」の読みは機材と交絡する。**この対比も時期と交絡している**")
    summary["NC3"] = {"d_device": d_dev, "p": one_sided_p(dev_null, d_dev),
                      "era_same": median(e_same), "era_diff": median(e_diff)}

    # --- NC4 era only ---
    p("")
    p("## NC4 時期距離だけで多変量 AUC を回帰(属性を1つも使わない腕)")
    x = np.array([dist[k] for k in pairs_auc], dtype=float)
    y = np.array([pairs_auc[k] for k in pairs_auc], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    perm = [abs(np.polyfit(rng.permutation(x), y, 1)[0]) for _ in range(10000)]
    p_slope = float((np.sum(np.array(perm) >= abs(slope)) + 1) / 10001)
    p(f"  傾き {slope:+.5f} / 切片 {intercept:.4f} / R² {r2:.4f} / 置換 p = {p_slope:.4f}")
    p(f"  読み方(登録): R² > 0.30 なら素の D(工程) は時期の言い換え"
      f" → {'**越えた**' if r2 > 0.30 else '越えていない'}")
    summary["NC4"] = {"slope": float(slope), "r2": float(r2), "p": p_slope}

    # --- NC5 tile counts ---
    p("")
    p("## NC5 枚数(最小 2024/ako3B 83 と 最大 2019/succor 491)")
    key = ("2019/succor", "2024/ako3B") if ("2019/succor", "2024/ako3B") in pairs_auc \
        else ("2024/ako3B", "2019/succor")
    vals = sorted(pairs_auc.values(), reverse=True)
    rank = vals.index(pairs_auc[key]) + 1
    p(f"  その対の多変量 {pairs_auc[key]:.4f} —— 55対のうち {rank} 位"
      f"(中央 {median(list(pairs_auc.values())):.4f})")
    p("  読み方(登録): 特別高ければ器具が枚数を読んでいる")
    summary["NC5"] = {"auc": pairs_auc[key], "rank": rank}

    # --- IC2 ---
    p("")
    p("## IC2 走行の外から既に知っている値(学祭編どうしの A2 行)")
    a2 = {}
    for row in GAKUSAI_A2:
        m = arms[A2]["r"] == row
        a2[row] = (arms[A2]["X"][m], arms[A2]["g"][m])
    g_aucs = []
    for x1, x2 in itertools.combinations(GAKUSAI_A2, 2):
        print(f"[{time.time()-t0:7.1f}s] IC2 {x1} 対 {x2}", file=sys.stderr, flush=True)
        g_aucs.append(multi_cv(pair_pool(a2[x1], a2[x2]))[0])
    got = (median(g_aucs), max(g_aucs))
    ok = all(abs(g - k) <= IC2_TOL for g, k in zip(got, KNOWN_GAKUSAI))
    p(f"  中央 {got[0]:.4f} / 最大 {got[1]:.4f}   既知 {KNOWN_GAKUSAI[0]:.4f} / {KNOWN_GAKUSAI[1]:.4f}"
      f"   許容 {IC2_TOL} → {'通った' if ok else '**通らない**'}")
    summary["IC2"] = {"median": got[0], "max": got[1], "pass": bool(ok)}

    # --- 参考: fitness across extractions, and the five work-level rulings ---
    p("")
    p("## 参考(読みは保留)")
    fk = ("fitnesskoma", "a1_author_analogue")
    m = arms[fk[1]]["r"] == fk[0]
    if m.any():
        r = pair_row(pools | {"(A1) fitnesskoma": (arms[fk[1]]["X"][m], arms[fk[1]]["g"][m])},
                     "2024b/fitness", "(A1) fitnesskoma", rng, t0, tag="参考 ")
        p(f"  2024b/fitness(v3)対 (A1) fitnesskoma 多変量 **{r['multi']:.4f}**"
          f" 帯 [{r['band'][0]:.4f}, {r['band'][1]:.4f}]")
        p("  **census が「同じ作品」と答えるまで読まない**(登録 §cannot settle)")
        summary["fitness_cross_extraction"] = r

    # --- verdict ---
    p("")
    p("## 判定(登録 §Pass criteria)")
    sm = summary["matched(時期距離 ≤ 3)"]
    sr = summary["素(全対)"]
    c1 = (sm["d_process"] >= C_D) and (sm["d_process"] > sm["null_process_975"])
    c2 = (sm["d_arc"] >= C_D) and (sm["d_arc"] > sm["null_arc_975"])
    unjudged = [k for k, v in res.items() if not v["judged"]]
    p(f"  C1 工程: matched D {sm['d_process']:+.4f} 棒 {C_D:+.4f}"
      f" 帰無 97.5% {sm['null_process_975']:+.4f} → **{'C1 工程が見える' if c1 else '届かない'}**")
    p(f"  C2 編  : matched D {sm['d_arc']:+.4f} 棒 {C_D:+.4f}"
      f" 帰無 97.5% {sm['null_arc_975']:+.4f} → **{'C2 編が見える' if c2 else '届かない'}**")
    p(f"  (素の D: 工程 {sr['d_process']:+.4f} / 編 {sr['d_arc']:+.4f} —— 時期と交絡)")
    p(f"  C4 外部の床: min(S5) {min(ext):.4f} 対 max(作者どうし) {max(own):.4f}"
      f" → **{'C4 再現する' if min(ext) <= max(own) else 'C4 再現しない'}**")
    p(f"  C5 帯が足りない対: {len(unjudged)} / {len(res)}  S3m は 3対のまま")
    if not (c1 or c2):
        p("  **C3 判定保留 —— この設計では作者の裁定が画素に映らない**")
    summary["verdict"] = {"C1": bool(c1), "C2": bool(c2),
                          "C4": bool(min(ext) <= max(own)),
                          "unjudged": len(unjudged), "pairs": len(res)}

    p("")
    p(f"# 所要 {time.time() - t0:.1f} 秒")
    pathlib.Path(args.out).write_text("\n".join(out) + "\n")
    pathlib.Path(args.out).with_suffix(".json").write_text(
        json.dumps({"pairs": res, "summary": summary, "facts": facts},
                   indent=1, ensure_ascii=False, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    main()
