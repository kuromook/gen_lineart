"""The close-up angle, as registered: doc/prereg_closeup_20261106.md.

Same construction as face_scale_angle.py, with the external side of the
matched comparison drawn from the close-up pools (E6c / I6c) and the bins
moved to the author's typical face size: [600,700) [700,800) [800,1000).

  AUC_m : A x Kc matched on those bins (thin the larger side), 200 draws, median
  N1    : A's face tiles (any size, per-work counts kept) x the EXISTING K's
          face tiles (any size), whole sources, same counts, 1000 draws
  N0    : the same without works (NC1)
  W     : Kc x K matched on the previous run's bins 250..600 -- does the
          close-up wording alone move an axis at a fixed face size?
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_auc import AXES, DISCRETE_AXES, NULL_AXIS, auc, tie_structure  # noqa: E402
from face_scale_angle import EXT_FILE, draw_by_source, known_table, work_of  # noqa: E402

HI_BINS = [600, 700, 800, 1000]
LO_BINS = [250, 300, 350, 400, 500, 600]
M, DRAWS, SEED = 200, 1000, 20261107
SHRINK, MIN_GAP, NC_MED = 0.10, 0.15, 0.02
PC_FULL, PC_LO, PC_HI = 0.85, 0.40, 0.60
W_LO, W_HI = 0.40, 0.60
MIN_MATCHED = 150
IC_TOL = 0.0001
FACE = "__face__"
LABEL = {"a1_author_analogue": "a1", "a2_author_digital": "a2", "a3_animagine": "e6", "a3_illustrious": "i6",
         "e6c": "e6c", "i6c": "i6c"}
KC = {"e6": "e6c", "i6": "i6c"}


def bin_of(h, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= h < bins[i + 1]:
            return i
    return None


def load(profiles, faces):
    face = {}
    for path in faces:
        for r in csv.DictReader(open(path)):
            face[(r["pool"], r["tile"])] = (float(r["source_face_px_est"]) if r["source_face_px_est"] else None, r["source"])
    pools = {k: {"tile": [], "src": [], "work": [], "h": [], "vals": collections.defaultdict(list)} for k in set(LABEL.values())}
    unmatched = collections.Counter()
    for path in profiles:
        for r in csv.DictReader(open(path)):
            k = LABEL.get(r["label"])
            if k is None:
                continue
            stem = Path(r["sample"]).stem
            if (k, stem) not in face:
                unmatched[k] += 1
                continue
            h, src = face[(k, stem)]
            P = pools[k]
            P["tile"].append(stem)
            P["src"].append(src)
            P["work"].append(work_of(k, stem))
            P["h"].append(h if h is not None else np.nan)
            for a in AXES:
                P["vals"][a].append(float(r[a]))
    for P in pools.values():
        P["h"] = np.array(P["h"])
        P["vals"] = {a: np.array(v) for a, v in P["vals"].items()}
        P["vals"][FACE] = P["h"]
        P["face"] = np.where(~np.isnan(P["h"]))[0]
    return pools, face, unmatched


def matched(rng, PA, PK, bins, axes):
    bA, bK = collections.defaultdict(list), collections.defaultdict(list)
    for i in PA["face"]:
        b = bin_of(PA["h"][i], bins)
        if b is not None:
            bA[b].append(i)
    for i in PK["face"]:
        b = bin_of(PK["h"][i], bins)
        if b is not None:
            bK[b].append(i)
    nb = [min(len(bA[b]), len(bK[b])) for b in range(len(bins) - 1)]
    raw = [(len(bA[b]), len(bK[b])) for b in range(len(bins) - 1)]
    draws, ok = [], True
    for _ in range(M):
        ia, ik = [], []
        for b, n in enumerate(nb):
            ia += list(rng.choice(bA[b], n, replace=False)) if n else []
            ik += list(rng.choice(bK[b], n, replace=False)) if n else []
        for b, n in enumerate(nb):
            ok &= sum(bin_of(PA["h"][i], bins) == b for i in ia) == sum(bin_of(PK["h"][i], bins) == b for i in ik) == n
        draws.append((np.array(ia, int), np.array(ik, int)))
    med = {a: float(np.median([auc(PA["vals"][a][ia], PK["vals"][a][ik]) for ia, ik in draws])) for a in axes}
    return med, draws, nb, raw, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--profile-new", default="results/closeup_20261106/profile_closeup.csv")
    ap.add_argument("--faces-old", default="results/face_scale_20261104/face_px_est.csv")
    ap.add_argument("--faces-new", default="results/closeup_20261106/face_px_est_closeup.csv")
    ap.add_argument("--out", default="results/closeup_20261106")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    pools, face, unmatched = load([args.profile_third, args.profile_new], [args.faces_old, args.faces_new])
    axes = list(AXES) + [FACE]
    p("# アップの角度 —— 登録 doc/prereg_closeup_20261106.md")
    p("## 母集団")
    for k in ("a1", "a2", "e6", "i6", "e6c", "i6c"):
        P = pools[k]
        p(f"  {k}: タイル {len(P['tile'])} / 顔のある元の絵の窓 {len(P['face'])} / 外した(顔の検出なし) {len(P['tile']) - len(P['face'])}"
          f" / 元の絵 {len(set(P['src']))}(うち顔なし {len(set(P['src'])) - len({P['src'][i] for i in P['face']})})")
    n_face_rows = len(face)
    n_join = sum(len(P["tile"]) for P in pools.values())
    ic2 = sum(unmatched.values()) == 0 and n_join == n_face_rows and all(len(pools[k]["tile"]) > 0 for k in ("e6c", "i6c"))
    p(f"## IC2 突き合わせ: profile 側で突き合わない {dict(unmatched)} / 顔の CSV {n_face_rows} 行・突き合った {n_join} → {'通った' if ic2 else '通らない'}")

    p("## IC1 既存のプールで絞らない AUC が第3のプールの表を再現(AUC13 / AUC23)")
    ic1 = True
    for k in ("e6", "i6"):
        kt = known_table(f"results/third_pool_20261031/auc_three_arms_{EXT_FILE[k]}.txt")
        bad = [(A, a) for a in AXES for j, A in enumerate(("a1", "a2"))
               if abs(auc(pools[A]["vals"][a], pools[k]["vals"][a]) - kt[a][j]) > IC_TOL + 5e-5]
        ic1 &= not bad and len(kt) == len(AXES)
        p(f"  {k}: 外れた {bad if bad else 'なし'}")

    rng = np.random.default_rng(SEED)
    R, ic3, ic4 = {}, True, True
    for A in ("a1", "a2"):
        for k in ("e6", "i6"):
            PA, PKc, PK = pools[A], pools[KC[k]], pools[k]
            med, draws, nb, raw, ok = matched(rng, PA, PKc, HI_BINS, axes)
            ic3 &= ok
            ic4 &= sum(nb) >= MIN_MATCHED
            face_ref = {a: auc(PA["vals"][a][PA["face"]], PK["vals"][a][PK["face"]]) for a in axes}
            all_ref = {a: auc(PA["vals"][a], PK["vals"][a]) for a in AXES}
            by_work = collections.defaultdict(lambda: collections.defaultdict(list))
            allA = collections.defaultdict(list)
            for i in PA["face"]:
                by_work[PA["work"][i]][PA["src"][i]].append(i)
                allA[PA["src"][i]].append(i)
            srcK = collections.defaultdict(list)
            for i in PK["face"]:
                srcK[PK["src"][i]].append(i)
            n1 = {a: np.empty(args.draws) for a in axes}
            n0 = {a: np.empty(args.draws) for a in axes}
            short = 0
            for t in range(args.draws):
                ia_m, ik_m = draws[t % M]
                ia = []
                for w, n in collections.Counter(PA["work"][i] for i in ia_m).items():
                    got, miss = draw_by_source(rng, by_work[w], n)
                    ia += got
                    short += miss
                ik, _ = draw_by_source(rng, srcK, len(ik_m))
                ia0, _ = draw_by_source(rng, allA, len(ia_m))
                ia, ik, ia0 = np.array(ia, int), np.array(ik, int), np.array(ia0, int)
                for a in axes:
                    n1[a][t] = auc(PA["vals"][a][ia], PK["vals"][a][ik])
                    n0[a][t] = auc(PA["vals"][a][ia0], PK["vals"][a][ik])
            band = {a: (float(np.median(n1[a])), float(np.percentile(n1[a], 2.5)), float(np.percentile(n1[a], 97.5))) for a in axes}
            R[(A, k)] = dict(med=med, nb=nb, raw=raw, face=face_ref, all=all_ref, band=band,
                             n0={a: float(np.median(n0[a])) for a in axes}, short=short,
                             wc=collections.Counter(PA["work"][i] for i in draws[0][0]),
                             wcf=collections.Counter(PA["work"][i] for i in PA["face"]))
    p(f"## IC3 揃えた組でビンごとの数が両側で等しい(全 {M}回): {'通った' if ic3 else '通らない'}")
    p(f"## IC4 揃えた数が組ごとに ≥ {MIN_MATCHED}: {'通った' if ic4 else '通らない'}")
    for (A, k), r in R.items():
        p(f"  {A}×{KC[k]}: ビン {HI_BINS} 作者/外部 {r['raw']} → 揃えた {r['nb']} = {sum(r['nb'])}  N1 で作品の窓が足りなかった延べ {r['short']}")
        p(f"     揃えた組(1回目)の作品: {dict(r['wc'])}")
        p(f"     顔のある窓全体の作品:   {dict(r['wcf'])}")

    p("## PC 顔の高さの AUC(揃え方の検査。発見ではない)")
    pc = True
    for k in ("e6", "i6"):
        r = R[("a1", k)]
        ok = r["face"][FACE] >= PC_FULL and PC_LO <= r["med"][FACE] <= PC_HI
        pc &= ok
        p(f"  a1: 対 既存{k} の顔のある窓 {r['face'][FACE]:.4f} / 対 {KC[k]} 揃えた {r['med'][FACE]:.4f} → {'通った' if ok else '通らない'}")
    for k in ("e6", "i6"):
        r = R[("a2", k)]
        p(f"  (副) a2: {r['face'][FACE]:.4f} / {r['med'][FACE]:.4f}")

    p("## 副 W: 同じ顔の大きさ(250〜600)で、アップの語 対 既存の語")
    W = {}
    for k in ("e6", "i6"):
        med, _, nb, raw, _ = matched(rng, pools[KC[k]], pools[k], LO_BINS, axes)
        W[k] = med
        p(f"  {KC[k]}×{k}: ビン {LO_BINS} 件数 {raw} → 揃えた {nb} = {sum(nb)}(0 のビンは外れる)")
        for a in AXES:
            p(f"    {a:26s} AUC {med[a]:.4f} {'中' if W_LO <= med[a] <= W_HI else '外'}")

    p("## 帰無値(棒の前)")
    nc1 = True
    for (A, k), r in R.items():
        p(f"  {A}(対 既存 {k})")
        for a in AXES:
            g0, lo, hi = r["band"][a]
            d0 = r["n0"][a] - r["face"][a]
            if A == "a1":
                nc1 &= abs(d0) < NC_MED
            p(f"    {a:26s} 顔全体 {r['face'][a]:.4f}  N0中央−顔全体 {d0:+.4f}  N1 中央 {g0:.4f} 帯 [{lo:.4f}, {hi:.4f}]  作品の比が動かす分 {g0 - r['face'][a]:+.4f}")
    p(f"  NC1 (A1、|N0 中央 − 顔全体| < {NC_MED}): {'通った' if nc1 else '通らない'}")

    ok_all = ic1 and ic2 and ic3 and ic4 and pc and nc1
    p("")
    p(f"## 主値と判定(C5 の条件: IC1 {ic1} / IC2 {ic2} / IC3 {ic3} / IC4 {ic4} / PC {pc} / NC1 {nc1})")
    for A in ("a1", "a2"):
        p(f"### {A}{'(主)' if A == 'a1' else '(副。箱は動かさない)'}")
        p("  axis | モデルごと: 絞らない 顔全体 N1中央 [帯] 揃えた s 帯 | W | 判定")
        for a in AXES:
            cells, S = [], []
            for k in ("e6", "i6"):
                r = R[(A, k)]
                g0, lo, hi = r["band"][a]
                gm = r["med"][a]
                s = abs(g0 - 0.5) - abs(gm - 0.5)
                x = dict(g0=g0, gm=gm, s=s, out=not (lo <= gm <= hi), cross=(gm - 0.5) * (g0 - 0.5) < 0)
                S.append(x)
                cells.append(f"{k} {r['all'][a]:.4f} {r['face'][a]:.4f} {g0:.4f} [{lo:.4f},{hi:.4f}] {gm:.4f} {s:+.4f} {'外' if x['out'] else '中'}")
            w_in = all(W_LO <= W[k][a] <= W_HI for k in ("e6", "i6"))
            if not ok_all:
                v = "C5 —— 何も読まない"
            elif a == NULL_AXIS:
                v = "帰無値の腕(読まない)"
            elif a in DISCRETE_AXES:
                v = "判定保留(離散。条例4)"
            elif all(x["out"] and not x["cross"] and x["s"] >= SHRINK for x in S):
                v = "C2 縮む —— " + ("構図のスケール" if w_in else "構図のスケール、またはアップの語")
            elif all(x["out"] and not x["cross"] and x["s"] <= -SHRINK for x in S):
                v = "C3 開く"
            elif all(x["out"] and x["cross"] and abs(x["gm"] - 0.5) >= 0.10 for x in S):
                v = "C4 向きが反転(判定保留)"
            elif all(not x["out"] and abs(x["g0"] - 0.5) >= MIN_GAP for x in S):
                v = "C1 残る"
            else:
                v = "判定保留"
            p(f"  {a:26s} | " + " | ".join(cells) + f" | W {'中' if w_in else '外'} | {v}")
        for a in DISCRETE_AXES:
            for P_ in (A, "e6c", "i6c"):
                n_val, big, frac = tie_structure(pools[P_]["vals"][a][pools[P_]["face"]])
                p(f"    条例4 {a} {P_}(顔のある窓): 相異なる値 {n_val} / 最大の同値塊 {big}({frac:.1%})")
    (out / "closeup_angle.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
