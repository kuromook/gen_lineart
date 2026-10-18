#!/usr/bin/env python
"""K16 登録 (0) の確認: 取り分け(test_panel / test_series)で「畳まれた橋を通る」を測る。
2026-10-18 事前登録(`doc/work_log.md`)。帯は 1 回しか開かない。

曝露 = 畳まれた橋を通る / 転帰 = 作者の broken 印 / 層別 = 長さ十分位(各帯の中)
顔 = CI のクラスタ / 主値 = PAF(Miettinen、長さ調整)、Levin は感度
棒: CI 下限 >= 0.30 支持 / 上限 < 0.30 棄却 / その間 判定保留(「ほぼ届いた」と書かない)
帰無値: 長さ十分位の中で印を置換 -> MH-RR = 1
**両帯を同時に 1 回だけ読み、合算せず別々に報告。逐次にしない。**
読む前の印字: 比例した棄権条件 / 粗い表の 2 式一致 / 層の寄与
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/sh1/deepl/lineart-stroke-grammar") / "tools/stroke"))

Q_STRATA, N_BOOT, N_PERM, SEED = 10, 10000, 1000, 20261018
PAF_BAR, DISCOVERY_RR = 0.30, 4.8785
BANDS = ("test_panel", "test_series")


def reach(n, p, n_marks, rr):
    """帯の n・曝露有病率・印の数と、仮定した効果量から、棒に届くかを転帰を読まずに計算する。"""
    m = n_marks / n
    r_un = m / (p * rr + (1 - p))
    a = n * p * rr * r_un          # 曝露側の印
    c = n * (1 - p) * r_un         # 非曝露側の印
    se = np.sqrt(1 / a - 1 / (n * p) + 1 / c - 1 / (n * (1 - p)))
    lo, hi = rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)
    pc = a / (a + c)
    def pf(r): return pc * (r - 1) / r if r > 1 else 0.0
    return dict(assumed_rr=round(rr, 4), exposed_marks=round(a, 1), unexposed_marks=round(c, 1),
                rr_ci=[round(float(lo), 3), round(float(hi), 3)],
                paf=round(float(pf(rr)), 4), paf_ci=[round(float(pf(lo)), 4), round(float(pf(hi)), 4)])


def mh(expo, outc, st):
    num = den = 0.0
    rows = []
    for s in np.unique(st):
        m = st == s
        a = float((expo & outc & m).sum()); b = float((expo & ~outc & m).sum())
        c = float((~expo & outc & m).sum()); d = float((~expo & ~outc & m).sum())
        n = a + b + c + d
        if n == 0: continue
        num += a * (c + d) / n; den += c * (a + b) / n
        rows.append(dict(stratum=int(s), n=int(n), n_marks=int(a + c),
                         informative=bool((a + b) and (c + d) and (a + c) and (b + d))))
    return ((num / den) if den > 0 else None), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(TRACK_H / "labels/state.json"))
    ap.add_argument("--groups", default=str(TRACK_H / "labels/face_groups.jsonl"))
    ap.add_argument("--out", default="results/k16_holdout_bridge_20261018")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    from k4_main_leg import work
    from k1_broken_vs_fill import densify
    G.load_state(Path(a.labels))
    # face_groups.jsonl は**追記式のログ**で、同じ order が最大 3 行ある(417 行 / 一意 301)。
    # **後勝ち**で畳む。この規則で train は K4 の 179 顔 / 185 印を完全に再現する(器具確認)。
    last = {}
    for l in open(a.groups):
        r = json.loads(l); last[r["order"]] = r
    recs = sorted([r for r in last.values() if r.get("usage") in BANDS
                   and r.get("status") == "done" and "broken" in r],
                  key=lambda r: (r["panel"], r["order"]))
    pids = sorted({r["panel"] for r in recs})
    print(f"取り分けの顔 {len(recs)}({dict(Counter(r['usage'] for r in recs))}) / コマ {len(pids)}",
          flush=True)
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"], False) for p in pids]
    byp = {}
    with Pool(a.workers) as pool:
        for i, (pid, o) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            byp[pid] = o
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    bad = [p for p in pids if byp.get(p) is None or len(byp[p][0]) != int(P[p]["n"])]
    print("器具確認 トークン数がパックと一致:", "合格" if not bad else f"不合格 {bad[:5]}", flush=True)
    assert not bad, "トークン数が合わない。止める。"

    rows = []
    for r in recs:
        o = byp.get(r["panel"])
        if o is None: continue
        pl = G.face_payload(r["order"])
        broken = set(r.get("broken", []))
        for s in pl["strokes"]:
            sid = s["id"]
            thru, _l, _rt = o[0][sid]
            _p, length = densify(np.asarray(s["p"], np.float64).reshape(16, 2))
            rows.append(dict(band=r["usage"], order=r["order"], panel=r["panel"], sid=sid,
                             thru=int(thru), marked=int(sid in broken), length=round(length, 2)))
    with open(out / "per_token.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["band", "order", "panel", "sid", "thru", "marked", "length"])
        w.writeheader(); w.writerows(rows)

    # ---- 読む前の印字: 比例した棄権条件(転帰の分布は使わない) ----
    gate = {}
    for b in BANDS:
        sub = [r for r in rows if r["band"] == b]
        n = len(sub); p = float(np.mean([r["thru"] for r in sub]))
        nm = int(sum(r["marked"] for r in sub))
        sup = reach(n, p, nm, DISCOVERY_RR)
        ref = reach(n, p, nm, 1.0000001)
        gate[b] = dict(n_tokens=n, prevalence=round(p, 4), n_marks=nm,
                       support_side=sup, null_side=ref,
                       support_reaches=bool(sup["paf_ci"][0] >= PAF_BAR),
                       refutation_reaches=bool(ref["paf_ci"][1] < PAF_BAR))
        gate[b]["read"] = bool(gate[b]["support_reaches"] and gate[b]["refutation_reaches"])
    print("\n=== 棄権条件(材料の転帰を読む前) ===")
    print(json.dumps(gate, ensure_ascii=False, indent=1), flush=True)

    summary = dict(registered=dict(exposure="畳まれた橋を通る", outcome="作者の broken 印",
                                   strata="長さ十分位(各帯の中)", cluster="顔",
                                   primary="PAF(Miettinen、長さ調整)", sensitivity="Levin",
                                   bar="CI 下限 >= 0.30 支持 / 上限 < 0.30 棄却 / その間 判定保留",
                                   null="長さ十分位の中で印を置換 -> MH-RR = 1",
                                   discovery_rr=DISCOVERY_RR, note="両帯を合算しない"),
                   gate=gate, bands={})
    rng = np.random.default_rng(SEED)
    for b in BANDS:
        if not gate[b]["read"]:
            summary["bands"][b] = dict(skipped="棄権条件に届かない"); continue
        sub = [r for r in rows if r["band"] == b]
        y = np.array([bool(r["marked"]) for r in sub]); x = np.array([bool(r["thru"]) for r in sub])
        L = np.array([r["length"] for r in sub]); f = np.array([r["order"] for r in sub])
        e = np.quantile(L, np.linspace(0, 1, Q_STRATA + 1)); e[0], e[-1] = -np.inf, np.inf
        st = np.clip(np.searchsorted(e, L, side="right") - 1, 0, Q_STRATA - 1)
        # 粗い表の 2 式一致(器具確認)
        rc = y[x].mean() / y[~x].mean() if y[~x].any() else None
        pp, pc = float(x.mean()), float(x[y].mean())
        lev = pp * (rc - 1) / (pp * (rc - 1) + 1); mie = pc * (rc - 1) / rc
        rr, srows = mh(x, y, st)
        info = [r for r in srows if r["informative"]]
        uf = np.unique(f); idx = {u: np.where(f == u)[0] for u in uf}
        bo = []
        for _ in range(N_BOOT):
            pick = rng.choice(uf, len(uf), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            v, _ = mh(x[ii], y[ii], st[ii])
            if v and y[ii].any() and x[ii][y[ii]].size:
                bo.append(float(x[ii][y[ii]].mean()) * (v - 1) / v)
        bo = np.array(bo)
        ci = [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
        perm = []
        for _ in range(N_PERM):
            y2 = y.copy()
            for s in range(Q_STRATA):
                m = st == s
                y2[m] = rng.permutation(y[m])
            v, _ = mh(x, y2, st)
            if v: perm.append(v)
        paf = pc * (rr - 1) / rr
        verdict = ("支持" if ci[0] >= PAF_BAR else "**棄却**" if ci[1] < PAF_BAR else "判定保留")
        summary["bands"][b] = dict(
            crude=dict(rr=round(float(rc), 4), levin=round(lev, 6), miettinen=round(mie, 6),
                       agree=bool(abs(lev - mie) < 1e-6)),
            strata=dict(n_informative=len(info), n_total=len(srows),
                        marks_in_informative=sum(r["n_marks"] for r in info)),
            negative_control=dict(perm_median=round(float(np.median(perm)), 4),
                                  perm_ci=[round(float(np.percentile(perm, 2.5)), 4),
                                           round(float(np.percentile(perm, 97.5)), 4)],
                                  null_value=1.0),
            primary=dict(mh_rr=round(rr, 4), paf_miettinen=round(paf, 4),
                         paf_ci95=[round(x_, 4) for x_ in ci],
                         paf_levin=round(pp * (rr - 1) / (pp * (rr - 1) + 1), 4)),
            verdict=verdict)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
