"""The face-scale angle, as registered: doc/prereg_face_scale_20261105.md.

Per author pool A in {A1 (primary), A2 (secondary)}, external pool k in {E6, I6}:
match the source face height in bins [250,300) [300,350) [350,400) [400,500)
[500,600) by thinning the larger side to the smaller count (tile level),
200 matching draws, AUC_m = median. Null N1: same counts drawn from all
face-detected tiles regardless of size, author side keeping per-work counts,
whole sources at a time; N0 the same without works (NC1 only).
"""

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_auc import AXES, DISCRETE_AXES, NULL_AXIS, auc, tie_structure  # noqa: E402

BINS = [250, 300, 350, 400, 500, 600]
M, DRAWS, SEED = 200, 1000, 20261105
SHRINK, MIN_GAP, NC_MED = 0.10, 0.15, 0.02
PC_FULL, PC_LO, PC_HI = 0.85, 0.40, 0.60
IC_TOL = 0.0001
LABEL = {"a1_author_analogue": "a1", "a2_author_digital": "a2", "a3_animagine": "e6", "a3_illustrious": "i6"}
EXT_FILE = {"e6": "animagine", "i6": "illustrious"}
FACE = "__face__"


def work_of(pool, tile):
    if pool == "a1":
        return tile.split("koma")[0]
    if pool == "a2":
        return re.match(r"a2_([0-9]+b?_[a-z0-9]+)__", tile).group(1)
    return "ext"


def known_table(path):
    """AUC13, AUC23 per axis from the third pool's '## 軸ごと(全体)' block."""
    out, on = {}, False
    for line in open(path):
        if line.startswith("## 軸ごと"):
            on = True
            continue
        if on and line.startswith("## "):
            break
        parts = line.split()
        if on and parts and parts[0] in AXES:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
    return out


def bin_of(h):
    for i in range(len(BINS) - 1):
        if BINS[i] <= h < BINS[i + 1]:
            return i
    return None


def draw_by_source(rng, idx_by_src, n):
    """Whole sources in random order until n tiles; trim the last source at random."""
    srcs = list(idx_by_src)
    rng.shuffle(srcs)
    out = []
    for s in srcs:
        if len(out) >= n:
            break
        t = list(idx_by_src[s])
        need = n - len(out)
        if len(t) > need:
            t = list(rng.choice(t, need, replace=False))
        out.extend(t)
    return out, max(0, n - len(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--faces", default="results/face_scale_20261104/face_px_est.csv")
    ap.add_argument("--out", default="results/face_scale_20261104")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    out = Path(args.out)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    face = {}
    for r in csv.DictReader(open(args.faces)):
        face[(r["pool"], r["tile"])] = (float(r["source_face_px_est"]) if r["source_face_px_est"] else None, r["source"])
    pools = {k: {"tile": [], "src": [], "work": [], "h": [], "vals": collections.defaultdict(list)} for k in LABEL.values()}
    unmatched = 0
    for r in csv.DictReader(open(args.profile)):
        k = LABEL.get(r["label"])
        if k is None:
            continue
        stem = Path(r["sample"]).stem
        if (k, stem) not in face:
            unmatched += 1
            continue
        h, src = face[(k, stem)]
        P = pools[k]
        P["tile"].append(stem)
        P["src"].append(src)
        P["work"].append(work_of(k, stem))
        P["h"].append(h if h is not None else np.nan)
        for a in AXES:
            P["vals"][a].append(float(r[a]))
    for k, P in pools.items():
        P["h"] = np.array(P["h"])
        P["vals"] = {a: np.array(v) for a, v in P["vals"].items()}
        P["vals"][FACE] = P["h"]
        P["face"] = np.where(~np.isnan(P["h"]))[0]
    n_face_rows = sum(1 for _ in face)
    n_join = sum(len(P["tile"]) for P in pools.values())

    p("# 顔の大きさを揃える角度 —— 登録 doc/prereg_face_scale_20261105.md")
    p("## 母集団")
    for k, P in pools.items():
        p(f"  {k}: タイル {len(P['tile'])} / 顔のある元の絵の窓 {len(P['face'])} / 外した(顔の検出なし) {len(P['tile']) - len(P['face'])}")
    ic2 = unmatched == 0 and n_join == n_face_rows
    p(f"## IC2 突き合わせ: profile 側で突き合わない {unmatched} / 顔の CSV {n_face_rows} 行・突き合った {n_join} → {'通った' if ic2 else '通らない'}")

    p("## IC1 絞らない AUC が第3のプールの表を再現(AUC13 / AUC23)")
    ic1 = True
    for k in ("e6", "i6"):
        kt = known_table(f"results/third_pool_20261031/auc_three_arms_{EXT_FILE[k]}.txt")
        bad = []
        for a in AXES:
            for j, A in enumerate(("a1", "a2")):
                if abs(auc(pools[A]["vals"][a], pools[k]["vals"][a]) - kt[a][j]) > IC_TOL + 5e-5:
                    bad.append((A, a))
        ic1 &= not bad and len(kt) == len(AXES)
        p(f"  {k}: 外れた {bad if bad else 'なし'}(表の軸 {len(kt)})")

    rng = np.random.default_rng(SEED)
    axes = list(AXES) + [FACE]
    results = {}
    ic3 = True
    for A in ("a1", "a2"):
        for k in ("e6", "i6"):
            PA, PK = pools[A], pools[k]
            binsA = collections.defaultdict(list)
            binsK = collections.defaultdict(list)
            for i in PA["face"]:
                b = bin_of(PA["h"][i])
                if b is not None:
                    binsA[b].append(i)
            for i in PK["face"]:
                b = bin_of(PK["h"][i])
                if b is not None:
                    binsK[b].append(i)
            nb = [min(len(binsA[b]), len(binsK[b])) for b in range(len(BINS) - 1)]
            matches = []
            for _ in range(M):
                ia, ik = [], []
                for b, n in enumerate(nb):
                    ia += list(rng.choice(binsA[b], n, replace=False))
                    ik += list(rng.choice(binsK[b], n, replace=False))
                cnt = [sum(1 for i in ia if bin_of(PA["h"][i]) == b) == sum(1 for i in ik if bin_of(PK["h"][i]) == b) == nb[b]
                       for b in range(len(nb))]
                ic3 &= all(cnt)
                matches.append((np.array(ia), np.array(ik)))
            auc_m = {a: float(np.median([auc(PA["vals"][a][ia], PK["vals"][a][ik]) for ia, ik in matches])) for a in axes}
            auc_face = {a: auc(PA["vals"][a][PA["face"]], PK["vals"][a][PK["face"]]) for a in axes}
            auc_all = {a: auc(PA["vals"][a], PK["vals"][a]) for a in AXES}
            # Null draws.
            src_by_work = collections.defaultdict(lambda: collections.defaultdict(list))
            for i in PA["face"]:
                src_by_work[PA["work"][i]][PA["src"][i]].append(i)
            src_all_A = collections.defaultdict(list)
            for i in PA["face"]:
                src_all_A[PA["src"][i]].append(i)
            src_K = collections.defaultdict(list)
            for i in PK["face"]:
                src_K[PK["src"][i]].append(i)
            n1 = {a: np.empty(args.draws) for a in axes}
            n0 = {a: np.empty(args.draws) for a in axes}
            short = 0
            for t in range(args.draws):
                ia_m, ik_m = matches[t % M]
                wc = collections.Counter(PA["work"][i] for i in ia_m)
                ia = []
                for w, n in wc.items():
                    got, miss = draw_by_source(rng, src_by_work[w], n)
                    ia += got
                    short += miss
                ik, _ = draw_by_source(rng, src_K, len(ik_m))
                ia0, _ = draw_by_source(rng, src_all_A, len(ia_m))
                ia, ik, ia0 = np.array(ia), np.array(ik), np.array(ia0)
                for a in axes:
                    n1[a][t] = auc(PA["vals"][a][ia], PK["vals"][a][ik])
                    n0[a][t] = auc(PA["vals"][a][ia0], PK["vals"][a][ik])
            band = {a: (float(np.median(n1[a])), float(np.percentile(n1[a], 2.5)), float(np.percentile(n1[a], 97.5))) for a in axes}
            n0med = {a: float(np.median(n0[a])) for a in axes}
            wc0 = collections.Counter(PA["work"][i] for i in matches[0][0])
            wcf = collections.Counter(PA["work"][i] for i in PA["face"])
            results[(A, k)] = dict(nb=nb, auc_m=auc_m, auc_face=auc_face, auc_all=auc_all, band=band, n0med=n0med,
                                   short=short, wc0=wc0, wcf=wcf)
    p(f"## IC3 揃えた組でビンごとの数が両側で等しい(全 {M}回): {'通った' if ic3 else '通らない'}")

    p("## 組ごとの揃えた数と作品の比")
    for (A, k), R in results.items():
        p(f"  {A}×{k}: ビン {R['nb']} = {sum(R['nb'])}  N1 で作品の中の窓が足りなかった延べ {R['short']}")
        p(f"     揃えた組(1回目)の作品: {dict(R['wc0'])}")
        p(f"     顔のある窓全体の作品:   {dict(R['wcf'])}")

    p("## PC 顔の高さそのものの AUC(揃え方の検査。発見ではない)")
    pc = True
    for k in ("e6", "i6"):
        R = results[("a1", k)]
        ok = R["auc_face"][FACE] >= PC_FULL and PC_LO <= R["auc_m"][FACE] <= PC_HI
        pc &= ok
        p(f"  a1×{k}: 顔のある窓全体 {R['auc_face'][FACE]:.4f} / 揃えた {R['auc_m'][FACE]:.4f} → {'通った' if ok else '通らない'}")
    for k in ("e6", "i6"):
        R = results[("a2", k)]
        p(f"  (副) a2×{k}: 顔のある窓全体 {R['auc_face'][FACE]:.4f} / 揃えた {R['auc_m'][FACE]:.4f}")

    p("## 帰無値(棒の前)。N1 = 作品の数を保つ / N0 = 保たない")
    nc1 = True
    for A in ("a1", "a2"):
        for k in ("e6", "i6"):
            R = results[(A, k)]
            p(f"  {A}×{k}")
            for a in AXES:
                g0, lo, hi = R["band"][a]
                d0 = R["n0med"][a] - R["auc_face"][a]
                if A == "a1":
                    nc1 &= abs(d0) < NC_MED
                p(f"    {a:26s} 顔全体 {R['auc_face'][a]:.4f}  N0中央−顔全体 {d0:+.4f}  N1 中央 {g0:.4f} 帯 [{lo:.4f}, {hi:.4f}]  作品の比が動かす分 {g0 - R['auc_face'][a]:+.4f}")
    p(f"  NC1 (A1、全軸・両モデル |N0 中央 − 顔全体| < {NC_MED}): {'通った' if nc1 else '通らない'}")

    ok = ic1 and ic2 and ic3 and pc and nc1
    p("")
    p(f"## 主値と判定(C5 の条件: IC1 {ic1} / IC2 {ic2} / IC3 {ic3} / PC {pc} / NC1 {nc1})")
    verdicts = {}
    for A in ("a1", "a2"):
        p(f"### {A}{'(主)' if A == 'a1' else '(副。箱は動かさない)'}")
        p("  axis                       | 絞らない  顔全体  N1中央   帯              揃えた    s     | ... 各モデル | 判定")
        for a in AXES:
            cells, st = [], {}
            for k in ("e6", "i6"):
                R = results[(A, k)]
                g0, lo, hi = R["band"][a]
                gm = R["auc_m"][a]
                s = abs(g0 - 0.5) - abs(gm - 0.5)
                outside = not (lo <= gm <= hi)
                cross = (gm - 0.5) * (g0 - 0.5) < 0
                st[k] = dict(g0=g0, gm=gm, s=s, outside=outside, cross=cross)
                cells.append(f"{k} {R['auc_all'][a]:.4f} {R['auc_face'][a]:.4f} {g0:.4f} [{lo:.4f},{hi:.4f}] {gm:.4f} {s:+.4f} {'外' if outside else '中'}")
            S = list(st.values())
            if not ok:
                v = "C5 —— 何も読まない"
            elif a == NULL_AXIS:
                v = "帰無値の腕(読まない)"
            elif a in DISCRETE_AXES:
                v = "判定保留(離散。条例4)"
            elif all(x["outside"] and not x["cross"] and x["s"] >= SHRINK for x in S):
                v = "C2 縮む"
            elif all(x["outside"] and not x["cross"] and x["s"] <= -SHRINK for x in S):
                v = "C3 開く"
            elif all(x["outside"] and x["cross"] and abs(x["gm"] - 0.5) >= 0.10 for x in S):
                v = "C4 向きが反転(判定保留)"
            elif all(not x["outside"] and abs(x["g0"] - 0.5) >= MIN_GAP for x in S):
                v = "C1 残る"
            else:
                v = "判定保留"
            verdicts[(A, a)] = v
            p(f"  {a:26s} | " + " | ".join(cells) + f" | {v}")
        for a in DISCRETE_AXES:
            for P_ in (A, "e6", "i6"):
                n_val, big, frac = tie_structure(pools[P_]["vals"][a][pools[P_]["face"]])
                p(f"    条例4 {a} {P_}(顔のある窓): 相異なる値 {n_val} / 最大の同値塊 {big}({frac:.1%})")
    (out / "face_scale_angle.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
