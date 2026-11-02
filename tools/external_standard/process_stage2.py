"""Do the rough stage and the device split tiles inside one year?

doc/prereg_process_stage2_20261101.md. The registration is closed; this file
only implements it.

Each of the three axes is read on a population where the other two labels are
held still, so what is held still cannot be measured here. Every pair's
multivariate AUC is computed once; the cell-difference nulls are exact
enumerations of the work-level label exchange (31,465 / 293,930 / 5,005 ways),
which only reassign cells and never recompute an AUC, so they are free. The
author's ruling is given per work, so the work is the unit the exchange uses.
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
    A2, Pool, columns, digest_mod, load, score_auc,
)
from ako_series_auc import (  # noqa: E402
    boot_band, multi_cv, multi_null, pair_pool, single_axis, single_null,
)

TAGS = ["2015", "2016", "2017", "2018", "2019", "2019b", "2020", "2020b",
        "2021", "2022", "2022b", "2023b", "2024", "2024b", "2024c"]
MATCH = 3                      # registered: era distance <= 3 is "matched"
BAND_MAX = 0.30                # registered: wider than this is "帯が足りない"
C_D = 0.05                     # registered bar for C1 and C2
MIN_CELL = 8                   # registered: a read cell under this is 判定保留
KNOWN_GAKUSAI = (0.8947, 0.9972)   # IC2, from doc/result_ako_series_20261031.md
IC2_TOL = 5e-4
IC2B_TOL = 1e-9                # IC2b: multi_cv is deterministic
STAGE1_JSON = "results/process_within_20261031/process_within.json"
GAKUSAI_A2 = ["2018/ako7", "2019/ako7", "2019b/ako8", "2020/ako9", "2020b/ako10"]
EXTERNAL = [("X1 animagine", "a3_animagine"), ("X2 illustrious", "a3_illustrious")]
SPLIT_WORKS = ["2017/gakusai1-3DL", "2019/succor", "2018/ako6"]   # NC5
NC2_PAIRS = [("2023b/akocultB", "2024/ako3B"),        # 第1段と同じ対
             ("2017/ako5", "2017/gakusai1-3DL")]      # B の距離0で最大の対
SEED = 20261101

# name, label key, population filter, (label A, label B)
AXIS_DEFS = [
    ("A 線画(アナログ / デジタル)", "line_process",
     lambda f: True, ("アナログ", "デジタル")),
    ("B 下絵(混在 / アナログ)—— 線画=デジタル・機材=未記載 に固定", "rough",
     lambda f: f["line_process"] == "デジタル" and f["device"] == "未記載",
     ("混在", "アナログ")),
    ("C 機材(iPadと明記 / 未記載)—— 線画=デジタル・下絵=アナログ に固定", "device",
     lambda f: f["line_process"] == "デジタル" and f["rough"] == "アナログ",
     ("iPadと明記", "未記載")),
]
READS = [("距離0", lambda d: d == 0), ("matched(≤3)", lambda d: d <= MATCH),
         ("素(全対)", lambda d: True)]


def era(tag):
    return TAGS.index(tag)


def median(vals):
    return float(np.median(vals)) if len(vals) else float("nan")


def load_pool(profile_csv, manifest_csv):
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
        facts[work] = {"arc": m["arc"], "rough": m["rough"],
                       "line_process": m["line_process"], "device": m["device"],
                       "tag": m["tag"], "tiles": len(items), "groups": len(set(g))}
    return pools, facts


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


def axis_table(works, pairs_auc, dist):
    """Index the pairs of one axis population once, as numpy arrays."""
    pos = {w: i for i, w in enumerate(works)}
    ai, bi, auc, dd = [], [], [], []
    for (a, b), v in pairs_auc.items():
        if a in pos and b in pos:
            ai.append(pos[a])
            bi.append(pos[b])
            auc.append(v)
            dd.append(dist[(a, b)])
    return (np.array(ai), np.array(bi), np.array(auc, dtype=float),
            np.array(dd, dtype=int))


def d_of(tab, keep, is_a):
    """median(label differs) - median(label same) on the kept pairs."""
    ai, bi, auc, _dd = tab
    same = is_a[ai[keep]] == is_a[bi[keep]]
    v = auc[keep]
    if not same.any() or same.all():
        return float("nan")
    return float(np.median(v[~same]) - np.median(v[same]))


def exchange_null(tab, keep, n_works, n_a):
    out, dropped = [], 0
    for pick in itertools.combinations(range(n_works), n_a):
        is_a = np.zeros(n_works, dtype=bool)
        is_a[list(pick)] = True
        d = d_of(tab, keep, is_a)
        if np.isnan(d):
            dropped += 1
        else:
            out.append(d)
    return np.array(out), dropped


def one_sided_p(null, real):
    return float((np.sum(null >= real) + 1) / (len(null) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/process_stage2_20261101/profile_s2.csv")
    ap.add_argument("--manifest", default="data/author_process_20261101/pool_manifest.csv")
    ap.add_argument("--third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--out", default="results/process_stage2_20261101/process_stage2.txt")
    ap.add_argument("--works", default=None,
                    help="smoke test only: comma-separated subset of works")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pools, facts = load_pool(args.csv, args.manifest)
    arms = load(args.third)
    for name, key in EXTERNAL:
        pools[name] = (arms[key]["X"], arms[key]["g"])

    works = sorted(facts)
    if args.works:                      # smoke test path, never the real run
        keep_w = set(args.works.split(","))
        works = [w for w in works if w in keep_w]
        facts = {w: facts[w] for w in works}
        print(f"SMOKE: {len(works)} works only", file=sys.stderr)
    dist = {(a, b): abs(era(facts[a]["tag"]) - era(facts[b]["tag"]))
            for a, b in itertools.combinations(works, 2)}

    out = ["# 下絵と機材 —— 同じ年の中で(登録 doc/prereg_process_stage2_20261101.md)",
           "# 判定の単位はタイル1枚 / この track は絵を判定しない", ""]
    p = out.append
    summary = {}

    p("## IC3 採用した行(門と紙の水準は build.log / build_report.json)")
    for w in works:
        f = facts[w]
        p(f"  {w:20s} 札 {f['tag']:6s} 下絵 {f['rough']:5s} 線画 {f['line_process']:5s}"
          f" 機材 {f['device']:10s} 編 {f['arc']:10s}"
          f" タイル {f['tiles']:4d} 群 {f['groups']:4d}")
    p(f"  計 {len(works)}作 タイル {sum(f['tiles'] for f in facts.values())}"
      f" 群 {sum(f['groups'] for f in facts.values())}")
    for name, key in EXTERNAL:
        p(f"  {name:20s} タイル {len(arms[key]['g']):5d} 群 {len(set(arms[key]['g'])):4d}")
    p("")

    p("## IC1 / NC1 自己 AUC(恒等)")
    self_pool = pair_pool(pools[works[0]], pools[works[0]])
    ic1 = score_auc(columns(self_pool, ["ink_ratio"])[:, 0], self_pool.y)
    p(f"  {works[0]} を両側に置く(ink_ratio): {ic1:.4f}(期待 0.5000 ちょうど)"
      f"  {'通った' if abs(ic1 - 0.5) < 1e-9 else '**通らない**'}")
    summary["IC1"] = ic1

    # --- every own pair ---
    pairs_auc, res = {}, {}
    partial = pathlib.Path(args.out).with_suffix(".partial.json")
    for a, b in itertools.combinations(works, 2):
        r = pair_row(pools, a, b, rng, t0)
        r["era"] = dist[(a, b)]
        pairs_auc[(a, b)] = r["multi"]
        res[f"{a} 対 {b}"] = r
        partial.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))

    # --- IC2b: the 55 pairs of stage 1 must reproduce exactly ---
    p("")
    p("## IC2b 第1段の55対の多変量 AUC が再現するか(`multi_cv` は決定的)")
    old = json.loads(pathlib.Path(STAGE1_JSON).read_text())["pairs"]
    worst, n_cmp = 0.0, 0
    for k, v in old.items():
        if k in res and " 対 X" not in k:
            d = abs(res[k]["multi"] - v["multi"])
            worst = max(worst, d)
            n_cmp += 1
    ic2b = worst <= IC2B_TOL
    p(f"  比べた対 {n_cmp} / 最大の差 {worst:.3e} / 許容 {IC2B_TOL:.0e}"
      f" → **{'通った' if ic2b else '通らない —— 走行を読まない'}**")
    summary["IC2b"] = {"pairs": n_cmp, "worst": worst, "pass": bool(ic2b)}

    # --- the three axes ---
    p("")
    p("## 主値 —— 軸ごとの D と、作品ラベルの交換の厳密な帰無")
    for title, key, filt, (la, lb) in AXIS_DEFS:
        ws = [w for w in works if filt(facts[w])]
        tab = axis_table(ws, pairs_auc, dist)
        is_a = np.array([facts[w][key] == la for w in ws])
        n_a = int(is_a.sum())
        p("")
        p(f"### {title}")
        p(f"  母集団 {len(ws)}作({la} {n_a} / {lb} {len(ws) - n_a})"
          f"  対 {len(tab[0])}  交換の通り数 C({len(ws)},{n_a})")
        rec = {}
        for rname, pred in READS:
            keep = np.array([pred(d) for d in tab[3]], dtype=bool)
            if not keep.any():
                p(f"  {rname:12s} 対が0")
                continue
            d_real = d_of(tab, keep, is_a)
            same = is_a[tab[0][keep]] == is_a[tab[1][keep]]
            n_same, n_diff = int(same.sum()), int((~same).sum())
            if np.isnan(d_real):
                p(f"  {rname:12s} 片側のセルが空(同じ {n_same} / ちがう {n_diff})")
                rec[rname] = {"d": None, "n_same": n_same, "n_diff": n_diff}
                continue
            nul, drop = exchange_null(tab, keep, len(ws), n_a)
            pv = one_sided_p(nul, d_real)
            med_e_same = median(tab[3][keep][same])
            med_e_diff = median(tab[3][keep][~same])
            p(f"  {rname:12s} 同じ {n_same:4d}対 中央 "
              f"{median(tab[2][keep][same]):.4f}(時期距離の中央 {med_e_same:.1f})"
              f" / ちがう {n_diff:4d}対 中央 "
              f"{median(tab[2][keep][~same]):.4f}(距離の中央 {med_e_diff:.1f})")
            p(f"  {'':12s} **D = {d_real:+.4f}**  帰無 {len(nul)}+外し{drop} 通り:"
              f" 中央 {np.median(nul):+.4f} / 97.5% {np.percentile(nul, 97.5):+.4f}"
              f" / 片側 p = {pv:.4f}")
            bands = [res[f"{ws[i]} 対 {ws[j]}"]["judged"]
                     for i, j in zip(tab[0][keep], tab[1][keep])]
            # C5 reads the different-label pairs only (registration §C5:
            # "読む側のセル(B の距離0 10対 / C の matched 8対)"). Until
            # 2026-11-03 this counted the whole cell; see result §3 C5.
            bands_diff = [ok for ok, sm in zip(bands, same) if not sm]
            p(f"  {'':12s} 帯が足りない対 {bands.count(False)} / {len(bands)}"
              f"(うち「ちがう」対 {bands_diff.count(False)} / {len(bands_diff)})")
            rec[rname] = {"d": d_real, "p": pv, "n_same": n_same, "n_diff": n_diff,
                          "null_975": float(np.percentile(nul, 97.5)),
                          "null_median": float(np.median(nul)),
                          "ways": len(nul), "dropped": drop,
                          "era_same": med_e_same, "era_diff": med_e_diff,
                          "unjudged": bands_diff.count(False),
                          "unjudged_whole_cell": bands.count(False)}
        summary[title] = rec

    # --- identification NC: ako3 and ako3B may be the same work ---
    p("")
    p("## 同定の NC —— `2024/ako3` と `2024/ako3B` が同じ作品だった場合(`ako3` を外して数え直す)")
    ws = [w for w in works if w != "2024/ako3"]
    tab = axis_table(ws, pairs_auc, dist)
    is_a = np.array([facts[w]["line_process"] == "アナログ" for w in ws])
    for rname, pred in READS:
        keep = np.array([pred(d) for d in tab[3]], dtype=bool)
        d_real = d_of(tab, keep, is_a) if keep.any() else float("nan")
        if np.isnan(d_real):
            p(f"  {rname:12s} 片側のセルが空")
            continue
        nul, drop = exchange_null(tab, keep, len(ws), int(is_a.sum()))
        p(f"  {rname:12s} D(線画) = {d_real:+.4f}"
          f"  帰無 97.5% {np.percentile(nul, 97.5):+.4f} / p = {one_sided_p(nul, d_real):.4f}")
    p("  **主値は「別の2作」のまま**(登録時点で裁定が無い)。作者の答えが来たら両方の棒を印字する")

    # --- external pairs ---
    p("")
    p(f"# S5 外部との対({len(works) * 2}対)")
    ext_rows = []
    for w in works:
        for name, _k in EXTERNAL:
            r = pair_row(pools, w, name, rng, t0, tag="外部 ")
            res[f"{w} 対 {name}"] = r
            ext_rows.append((w, name, r))
            partial.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    for w, name, r in sorted(ext_rows, key=lambda t: -t[2]["multi"]):
        flag = "" if r["judged"] else "   **帯が足りない**"
        p(f"  {w:20s} 対 {name:16s} 多変量 **{r['multi']:.4f}**"
          f" 帯 [{r['band'][0]:.4f}, {r['band'][1]:.4f}]"
          f"  単軸 最大 {r['max_d']:.3f}({r['max_axis']}){flag}")
    ext = [r["multi"] for _w, _n, r in ext_rows]
    own = list(pairs_auc.values())
    p(f"  → 中央 {median(ext):.4f} / 最大 {max(ext):.4f} / 最小 {min(ext):.4f}")

    # --- NC5: one work split in half by group parity ---
    p("")
    p("## NC5 同じ作品を2つに割る(群の digest 偶奇)—— 分かれる理由が「群の集合」しか無い対")
    nc5 = {}
    for w in SPLIT_WORKS:
        X, g = pools[w]
        par = np.array([digest_mod(x, 2) for x in g])
        half = {f"{w}#even": (X[par == 0], g[par == 0]),
                f"{w}#odd": (X[par == 1], g[par == 1])}
        r = pair_row(pools | half, f"{w}#even", f"{w}#odd", rng, t0, tag="NC5 ")
        nc5[w] = r
        p(f"  {w:20s} 偶 {int((par == 0).sum()):4d}枚 / 奇 {int((par == 1).sum()):4d}枚"
          f"  多変量 **{r['multi']:.4f}** 帯 [{r['band'][0]:.4f}, {r['band'][1]:.4f}]"
          f"  単軸 最大 {r['max_d']:.3f}({r['max_axis']})")
    worst5 = max(r["multi"] for r in nc5.values())
    p(f"  期待(登録): ≤ 0.80。**0.90 を超えたら、どの軸も名乗れない** → 最大 {worst5:.4f}"
      f" → **{'倒れた' if worst5 > 0.90 else '倒れていない'}**")
    summary["NC5"] = {w: r["multi"] for w, r in nc5.items()}

    # --- NC2: break the group ---
    p("")
    p("## NC2 群を壊す(タイル単位で置換)—— 期待は「帯が 0.30 より狭くなる」だけ")
    nc2 = {}
    for a, b in NC2_PAIRS:
        pool = pair_pool(pools[a], pools[b])
        _m, _oof, _nf, c_pick = multi_cv(pool)
        tile_null = [multi_cv(Pool(pool.X, rng.permutation(pool.y), pool.g, pool.r),
                              C=c_pick)[0] for _ in range(50)]
        gn = res[f"{a} 対 {b}"]["null_multi"] if f"{a} 対 {b}" in res \
            else res[f"{b} 対 {a}"]["null_multi"]
        lo, hi = np.percentile(tile_null, 2.5), np.percentile(tile_null, 97.5)
        p(f"  {a} 対 {b}")
        p(f"    群単位 : 中央 {gn[0]:.4f} 帯 [{gn[1]:.4f}, {gn[2]:.4f}](幅 {gn[2]-gn[1]:.4f})")
        p(f"    タイル : 中央 {np.median(tile_null):.4f} 帯 [{lo:.4f}, {hi:.4f}]"
          f"(幅 {hi-lo:.4f}) → {'狭くなった' if (hi-lo) < (gn[2]-gn[1]) else '狭くならない'}")
        nc2[f"{a} 対 {b}"] = {"group": gn, "tile_median": float(np.median(tile_null)),
                              "tile_band": [float(lo), float(hi)]}
    summary["NC2"] = nc2

    # --- NC3 tile counts / NC4 era only ---
    keys = list(pairs_auc)
    y = np.array([pairs_auc[k] for k in keys], dtype=float)
    p("")
    p("## NC3 枚数 —— log10(小さい側のタイル数)だけで回帰")
    x3 = np.log10([min(facts[a]["tiles"], facts[b]["tiles"]) for a, b in keys])
    s3, i3 = np.polyfit(x3, y, 1)
    r2_3 = 1.0 - np.sum((y - (s3 * x3 + i3)) ** 2) / np.sum((y - y.mean()) ** 2)
    p(f"  傾き {s3:+.5f} / 切片 {i3:.4f} / R² {r2_3:.4f}"
      f"  期待 R² < 0.20 → {'**越えた —— 枚数で層別して読み直す**' if r2_3 >= 0.20 else '越えていない'}")
    summary["NC3"] = {"slope": float(s3), "r2": float(r2_3)}
    if r2_3 >= 0.20:
        med_t = float(np.median(x3))
        for title, key, filt, (la, lb) in AXIS_DEFS:
            ws = [w for w in works if filt(facts[w])]
            tab = axis_table(ws, pairs_auc, dist)
            is_a = np.array([facts[w][key] == la for w in ws])
            small = np.log10([min(facts[ws[i]]["tiles"], facts[ws[j]]["tiles"])
                              for i, j in zip(tab[0], tab[1])])
            for lab, m in (("枚数 小", small <= med_t), ("枚数 大", small > med_t)):
                for rname, pred in READS:
                    keep = m & np.array([pred(d) for d in tab[3]], dtype=bool)
                    if keep.sum() < 2:
                        continue
                    d = d_of(tab, keep, is_a)
                    if not np.isnan(d):
                        p(f"    {title[:1]} {lab} {rname:12s} D = {d:+.4f}({int(keep.sum())}対)")

    p("")
    p("## NC4 時期距離だけで回帰(属性を1つも使わない腕)")
    x4 = np.array([dist[k] for k in keys], dtype=float)
    s4, i4 = np.polyfit(x4, y, 1)
    r2_4 = 1.0 - np.sum((y - (s4 * x4 + i4)) ** 2) / np.sum((y - y.mean()) ** 2)
    perm = [abs(np.polyfit(rng.permutation(x4), y, 1)[0]) for _ in range(10000)]
    p4 = float((np.sum(np.array(perm) >= abs(s4)) + 1) / 10001)
    p(f"  傾き {s4:+.5f} / 切片 {i4:.4f} / R² {r2_4:.4f} / 置換 p = {p4:.4f}"
      f"  ({len(keys)}対。第1段は 88対で R² 0.2557)")
    p(f"  読み方(登録): R² > 0.30 なら素の D は時期の言い換え"
      f" → {'**越えた**' if r2_4 > 0.30 else '越えていない'}")
    summary["NC4"] = {"slope": float(s4), "r2": float(r2_4), "p": p4}

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
    p(f"  中央 {got[0]:.4f} / 最大 {got[1]:.4f}   既知 {KNOWN_GAKUSAI[0]:.4f} /"
      f" {KNOWN_GAKUSAI[1]:.4f}   許容 {IC2_TOL} → {'通った' if ok else '**通らない**'}")
    summary["IC2"] = {"median": got[0], "max": got[1], "pass": bool(ok)}

    # --- verdict ---
    p("")
    p("## 判定(登録 §Pass criteria)")
    b = summary[AXIS_DEFS[1][0]].get("距離0", {})
    c = summary[AXIS_DEFS[2][0]].get("matched(≤3)", {})
    a_ = summary[AXIS_DEFS[0][0]].get("matched(≤3)", {})
    c1 = bool(b.get("d") is not None and b.get("d", -1) >= C_D
              and b["d"] > b["null_975"])
    c2 = bool(c.get("d") is not None and c.get("d", -1) >= C_D
              and c["d"] > c["null_975"])
    p(f"  C1 下絵(距離0): D {b.get('d', float('nan')):+.4f} 棒 {C_D:+.4f}"
      f" 帰無 97.5% {b.get('null_975', float('nan')):+.4f} p {b.get('p', float('nan')):.4f}"
      f" → **{'C1 下絵が見える' if c1 else '届かない'}**")
    p(f"  C2 機材(matched): D {c.get('d', float('nan')):+.4f} 棒 {C_D:+.4f}"
      f" 帰無 97.5% {c.get('null_975', float('nan')):+.4f} p {c.get('p', float('nan')):.4f}"
      f" → **{'C2 機材が見える' if c2 else '届かない'}**")
    p(f"  C3 線画(matched): D {a_.get('d', float('nan')):+.4f}"
      f" p {a_.get('p', float('nan')):.4f} —— **印字するが名乗らない**"
      "(距離0の対が0、アナログ線画は 2024〜2025 だけ)")
    p(f"  C4 外部の床: min(S5) {min(ext):.4f} 対 max(作者どうし) {max(own):.4f}"
      f" → **{'C4 再現する' if min(ext) <= max(own) else 'C4 再現しない'}**")
    unj = [k for k, v in res.items() if not v["judged"]]
    small_b = b.get("n_diff", 0) < MIN_CELL or b.get("unjudged", 0) > 0
    small_c = c.get("n_diff", 0) < MIN_CELL or c.get("unjudged", 0) > 0
    p(f"  C5 判定保留: B の読むセル {b.get('n_diff')}対(帯不足 {b.get('unjudged')})"
      f" / C の読むセル {c.get('n_diff')}対(帯不足 {c.get('unjudged')})"
      f" → B {'判定保留' if small_b else '読める'} / C {'判定保留' if small_c else '読める'}")
    p(f"  帯が足りない対: {len(unj)} / {len(res)}")
    p(f"  NC5 の最大 {worst5:.4f} → {'**全体が倒れた**' if worst5 > 0.90 else '倒れていない'}")
    summary["verdict"] = {"C1": c1, "C2": c2, "C4": bool(min(ext) <= max(own)),
                          "C5_B_hold": bool(small_b), "C5_C_hold": bool(small_c),
                          "NC5_max": worst5, "unjudged": len(unj), "pairs": len(res)}

    p("")
    p(f"# 所要 {time.time() - t0:.1f} 秒")
    pathlib.Path(args.out).write_text("\n".join(out) + "\n")
    pathlib.Path(args.out).with_suffix(".json").write_text(
        json.dumps({"pairs": res, "summary": summary, "facts": facts},
                   indent=1, ensure_ascii=False, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    main()
