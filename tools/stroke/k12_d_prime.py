#!/usr/bin/env python
"""K12 登録 D の主値(2026-10-18 事前登録、帯は基盤 `reply_d_bars_20261018.md`)。

D' : トークンの骨格から `grow` だけのインク画素を除くと、骨格が 2 つ以上に分かれる
     (= そのトークンは自分の連結性を `grow` に負っている)。主値の述語。
D  : `grow` だけの画素を 1 つ以上通る。副。合算しない。

棒: AFe >= 0.30。RR* = (0.3 + 0.7p)/(0.7p) を**実測の p で引き直す**。
反証: MH-RR の 95% CI 上限が RR* を下回ったら棄却。
棄権: 帰無のもとで期待される曝露印・非曝露印のどちらかが 20 件未満なら**主値を読まない**。
層別は長さ十分位のみ。顔はクラスタ(ブートストラップ)。負の対照は十分位内で印を置換、帰無値 1。
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K

Q_STRATA, N_BOOT, N_PERM, SEED = 10, 10000, 1000, 20261018
AFE_BAR, MIN_EXPECTED = 0.30, 20
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def n_components(pts):
    """8 近傍での連結成分数。pts は {(r,c)} の集合。"""
    seen, n = set(), 0
    for p in pts:
        if p in seen:
            continue
        n += 1
        stack = [p]; seen.add(p)
        while stack:
            r, c = stack.pop()
            for dr, dc in NB:
                q = (r + dr, c + dc)
                if q in pts and q not in seen:
                    seen.add(q); stack.append(q)
    return n


def work(t):
    pid, source, name, sids = t
    g = cv2.imread(str(K.DATASET / source / "line" / name), 0)
    if g is None:
        return pid, []
    Q = K.pipeline(g)
    grow_only = Q["ink"].astype(bool) & ~(g < K.THR)
    tok_px = defaultdict(list)
    for i, (rr, cc) in enumerate(Q["spans"]):
        tk = Q["span_tok"].get(i, -1)
        if tk < 0 or len(rr) == 0:
            continue
        tok_px[tk].append((rr, cc))
    out = []
    for sid in sids:
        if sid not in tok_px:
            out.append(dict(panel=pid, sid=sid, found=0)); continue
        rr = np.concatenate([a for a, _ in tok_px[sid]])
        cc = np.concatenate([b for _, b in tok_px[sid]])
        gro = grow_only[rr, cc]
        pts = set(zip(rr.tolist(), cc.tolist()))
        keep = set(zip(rr[~gro].tolist(), cc[~gro].tolist()))
        n_before = n_components(pts)
        n_after = n_components(keep) if keep else 0
        d_prime = int(len(keep) > 0 and n_after > n_before)
        # 切り口の画素(残った成分を 2 つ以上つないでいた grow 画素)の灰と幅
        cut_gray = cut_iw = -1
        if d_prime:
            comp, seen = {}, set()
            cid = 0
            for p in keep:
                if p in seen: continue
                cid += 1; st = [p]; seen.add(p)
                while st:
                    r0, c0 = st.pop(); comp[(r0, c0)] = cid
                    for dr, dc in NB:
                        q = (r0 + dr, c0 + dc)
                        if q in keep and q not in seen:
                            seen.add(q); st.append(q)
            cuts = []
            for p in (set(pts) - keep):
                nb = {comp[q] for dr, dc in NB
                      if (q := (p[0] + dr, p[1] + dc)) in comp}
                if len(nb) >= 2:
                    cuts.append(p)
            if cuts:
                cut_gray = int(max(int(g[p]) for p in cuts))
                cut_iw = round(float(np.median([Q["iw"][p] for p in cuts])), 2)
        out.append(dict(panel=pid, sid=sid, found=1, n_px=int(len(rr)),
                        n_px_grow=int(gro.sum()), d_simple=int(gro.sum() > 0),
                        n_comp_before=n_before, n_comp_after=n_after,
                        all_grow=int(len(keep) == 0), d_prime=d_prime,
                        cut_gray=cut_gray, cut_iw=cut_iw))
    return pid, out


def mh_rr(y, x, strata):
    num = den = 0.0
    rows = []
    for s in np.unique(strata):
        m = strata == s
        a = float((y & x & m).sum()); b = float((y & ~x & m).sum())
        c = float((~y & x & m).sum()); d = float((~y & ~x & m).sum())
        n = a + b + c + d
        if n == 0:
            continue
        num += a * (c + d) / n
        den += c * (a + b) / n
        rows.append(dict(stratum=int(s), n=int(n), n_marked=int(a + b),
                         informative=bool(a + b > 0 and c + d > 0
                                          and (a + c) > 0 and (b + d) > 0)))
    return ((num / den) if den > 0 else None), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--k4", default="results/k4_main_leg_20261017/per_stroke.csv")
    ap.add_argument("--out", default="results/k12_d_prime_20261018")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    k4 = list(csv.DictReader(open(ROOT / a.k4)))
    bysid = {}
    for r in k4:
        bysid[(int(r["panel"]), int(r["sid"]))] = r
    byp = defaultdict(set)
    for (p, s) in bysid:
        byp[p].add(s)
    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"], sorted(s)) for p, s in byp.items()]
    print(f"母集団: 一意な (コマ, sid) {len(bysid)} / コマ {len(tasks)}", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for pid, res in pool.imap_unordered(work, tasks, chunksize=2):
            for d in res:
                k = bysid[(d["panel"], d["sid"])]
                rows.append(dict(d, order=k["order"], marked=int(k["marked"]),
                                 length=float(k["length"])))
    with open(out / "per_token.csv", "w", newline="") as fh:
        fn = ["panel", "order", "sid", "found", "marked", "length", "n_px", "n_px_grow",
              "d_simple", "n_comp_before", "n_comp_after", "all_grow", "d_prime",
              "cut_gray", "cut_iw"]
        w = csv.DictWriter(fh, fieldnames=fn, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["found"]]
    y = np.array([bool(r["marked"]) for r in ok])
    xp = np.array([bool(r["d_prime"]) for r in ok])
    xs = np.array([bool(r["d_simple"]) for r in ok])
    L = np.array([r["length"] for r in ok])
    face = np.array([int(r["order"]) for r in ok])
    p_prime, p_simple = float(xp.mean()), float(xs.mean())
    n_marks = int(y.sum())
    exp_e, exp_u = n_marks * p_prime, n_marks * (1 - p_prime)
    rr_star = (AFE_BAR + (1 - AFE_BAR) * p_prime) / ((1 - AFE_BAR) * p_prime) if p_prime else None
    gate = dict(n_tokens=len(ok), n_not_found=len(rows) - len(ok), n_marks=n_marks,
                prevalence_d_prime=round(p_prime, 4), prevalence_d_simple=round(p_simple, 4),
                all_grow=int(sum(r["all_grow"] for r in ok)),
                expected_exposed_marks=round(exp_e, 2),
                expected_unexposed_marks=round(exp_u, 2),
                min_expected=MIN_EXPECTED,
                rr_star=round(rr_star, 4) if rr_star else None,
                abstain=bool(exp_e < MIN_EXPECTED or exp_u < MIN_EXPECTED))
    print("棄権条件:", json.dumps(gate, ensure_ascii=False), flush=True)
    summary = dict(registered=dict(afe_bar=AFE_BAR, rr_star_formula="(0.3+0.7p)/(0.7p)",
                                   strata="長さ十分位のみ", cluster="顔(order)",
                                   null="長さ十分位内で印を置換 → MH-RR の帰無値 1",
                                   abstain_rule=f"期待曝露印・非曝露印のどちらかが {MIN_EXPECTED} 未満なら主値を読まない"),
                   gate=gate)
    if gate["abstain"]:
        summary["verdict"] = "**主値を読まない(棄権)**: 期待される印が足りない"
        json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(summary, ensure_ascii=False, indent=1)); return

    edges = np.quantile(L, np.linspace(0, 1, Q_STRATA + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    st = np.clip(np.searchsorted(edges, L, side="right") - 1, 0, Q_STRATA - 1)
    mh, srows = mh_rr(y, xp, st)
    info = [r for r in srows if r["informative"]]
    rng = np.random.default_rng(SEED)
    uf = np.unique(face)
    idx_of = {f: np.where(face == f)[0] for f in uf}
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(uf, len(uf), replace=True)
        ii = np.concatenate([idx_of[f] for f in pick])
        v, _ = mh_rr(y[ii], xp[ii], st[ii])
        if v:
            boot.append(v)
    boot = np.array(boot)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    perm = []
    for _ in range(N_PERM):
        y2 = y.copy()
        for s in range(Q_STRATA):
            m = st == s
            y2[m] = rng.permutation(y[m])
        v, _ = mh_rr(y2, xp, st)
        if v:
            perm.append(v)
    perm = np.array(perm)
    afe = (p_prime * (mh - 1)) / (p_prime * mh + 1 - p_prime) if mh else None
    afe_b = (p_prime * (boot - 1)) / (p_prime * boot + 1 - p_prime)
    summary.update(
        strata=dict(n_informative=len(info), n_total=len(srows),
                    marks_in_informative=sum(r["n_marked"] for r in info),
                    marks_total=n_marks, detail=srows),
        negative_control=dict(perm_median=round(float(np.median(perm)), 4),
                              perm_ci=[round(float(np.percentile(perm, 2.5)), 4),
                                       round(float(np.percentile(perm, 97.5)), 4)],
                              null_value=1.0),
        primary=dict(mh_rr=round(mh, 4), ci95=[round(x, 4) for x in ci],
                     afe=round(afe, 4), afe_ci95=[round(float(np.percentile(afe_b, 2.5)), 4),
                                                  round(float(np.percentile(afe_b, 97.5)), 4)],
                     rr_star=round(rr_star, 4), afe_bar=AFE_BAR),
        secondary_d_simple=dict(prevalence=round(p_simple, 4),
                                mh_rr=round(mh_rr(y, xs, st)[0], 4)),
        verdict=("支持(AFe >= 0.30)" if afe >= AFE_BAR else
                 "**棄却**(MH-RR の 95% CI 上限が RR* を下回る)" if ci[1] < rr_star else
                 "不支持(棒に届かないが棄却にも至らない)"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
