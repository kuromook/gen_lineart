#!/usr/bin/env python
"""実験 H6(2026-10-12 事前登録): Track F のまとまりと、線単位のグループ(正解)のずれ。学習用だけ。

出力: summary.json、per_group.csv、check.json
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402

MAX_ORDER = 104
KEEP = (3, 32)
TH = 0.8


def measure(strokes, cl, csize, same_label=()):
    """strokes: G の線番号。cl: コマの線ごとのまとまり番号。same_label: 同じ目印の別グループの線"""
    ids = np.asarray(sorted(strokes))
    c = cl[ids]
    cnt = Counter(c[c >= 0].tolist())
    n_parts = len(cnt) + int((c < 0).sum())                    # まとまりなしの線は 1 本ずつ別と数える
    if not cnt:
        return dict(n=len(ids), n_clusters=n_parts, completeness=0.0, purity=0.0, purity_label=0.0, cstar=-1, csize=0, kept=False)
    cstar, k = max(cnt.items(), key=lambda t: (t[1], -t[0]))
    size = int(csize[cstar])
    same = sum(1 for i in same_label if cl[i] == cstar)
    return dict(n=len(ids), n_clusters=n_parts, completeness=k / len(ids), purity=k / size, purity_label=(k + same) / size,
                cstar=int(cstar), csize=size, kept=KEEP[0] <= size <= KEEP[1])


def as_is(m):
    return m["completeness"] >= TH and m["purity"] >= TH and m["kept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h6_cluster_vs_groups_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    G.load_state(G.LAB / "face_groups.jsonl")
    recs = [r for o, r in sorted(G.S["faces"].items()) if o <= MAX_ORDER and r["usage"] == "train" and r["status"] == "done"]
    rng = np.random.default_rng(20261012)

    def panel(pid):
        _p, _w, cl = G.panel_strokes(pid)
        cl = np.asarray(cl).astype(int)
        return cl, np.bincount(cl[cl >= 0], minlength=int(cl.max()) + 2)

    # 道具確認 1: 合成
    cl, cs = panel(recs[0]["panel"])
    big = [c for c in np.flatnonzero(cs >= 3)][:2]
    A, B = np.flatnonzero(cl == big[0]), np.flatnonzero(cl == big[1])
    m1, m2 = measure(A, cl, cs), measure(np.concatenate([A, B]), cl, cs)
    ok1 = (m1["completeness"] == 1 and m1["purity"] == 1 and m1["n_clusters"] == 1 and m2["n_clusters"] == 2
           and abs(m2["completeness"] - max(len(A), len(B)) / (len(A) + len(B))) < 1e-9 and m2["purity"] == 1)
    # 道具確認 2: UI の cs と一致
    ok2 = True
    for o in rng.choice(len(G.S["queue"]), 10, replace=False):
        pl = G.face_payload(int(o)); cl2, cs2 = panel(pl["panel"])
        ok2 &= all((s["c"] < 0 and s["cs"] == 0) or s["cs"] == cs2[s["c"]] for s in pl["strokes"])

    rows, neg = [], []
    for r in recs:
        cl, cs = panel(r["panel"])
        pick = [s["id"] for s in G.face_payload(r["order"])["strokes"]]
        broken = set(r.get("broken", []))
        for gi, g in enumerate(r["groups"]):
            same = [i for gj, h in enumerate(r["groups"]) if gj != gi and h["label"] == g["label"] for i in h["strokes"]]
            m = measure(g["strokes"], cl, cs, same)
            m.update(order=r["order"], label=g["label"], has_broken=bool(broken & set(g["strokes"])), broken_known="broken" in r,
                     as_is=as_is(m))
            rows.append(m)
            neg.append(as_is(measure(rng.choice(pick, len(g["strokes"]), replace=False), cl, cs)))
    neg_rate = float(np.mean(neg))
    chk = dict(synthetic_ok=bool(ok1), ui_csize_ok=bool(ok2), negative_as_is=round(neg_rate, 4), n_faces=len(recs), n_groups=len(rows))
    chk["passed"] = bool(ok1 and ok2 and neg_rate <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格")
    if not chk["passed"]:
        return
    cols = ["order", "label", "n", "n_clusters", "completeness", "purity", "purity_label", "cstar", "csize", "kept", "has_broken",
            "broken_known", "as_is"]
    with open(out / "per_group.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        wr.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()} for m in rows])

    def summ(sel):
        if not sel:
            return None
        frag = [m["completeness"] < TH for m in sel]; mix = [m["purity"] < TH for m in sel]; drop = [not m["kept"] for m in sel]
        return dict(n=len(sel), as_is=round(float(np.mean([m["as_is"] for m in sel])), 3),
                    fragmented=round(float(np.mean(frag)), 3), mixed=round(float(np.mean(mix)), 3),
                    mixed_other_label=round(float(np.mean([m["purity_label"] < TH for m in sel])), 3),
                    not_kept=round(float(np.mean(drop)), 3),
                    completeness_median=round(float(np.median([m["completeness"] for m in sel])), 2),
                    purity_median=round(float(np.median([m["purity"] for m in sel])), 2),
                    one_cluster=round(float(np.mean([m["n_clusters"] == 1 for m in sel])), 3),
                    csize_too_small=round(float(np.mean([m["csize"] < KEEP[0] for m in sel])), 3),
                    csize_too_big=round(float(np.mean([m["csize"] > KEEP[1] for m in sel])), 3))
    S = {"all": summ(rows), "eye (open+closed)": summ([m for m in rows if m["label"].startswith("face.eye")])}
    for lb in G.LABELS:
        S[lb] = summ([m for m in rows if m["label"] == lb])
    S["size 1-2"] = summ([m for m in rows if m["n"] <= 2]); S["size 3+"] = summ([m for m in rows if m["n"] >= 3])
    known = [m for m in rows if m["broken_known"]]
    S["with broken line"] = summ([m for m in known if m["has_broken"]]); S["without broken line"] = summ([m for m in known if not m["has_broken"]])
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for k, v in S.items():
        if v:
            print(f"{k:22s} n={v['n']:3d} そのまま={v['as_is']:.2f} 断片化={v['fragmented']:.2f} 混在={v['mixed']:.2f}"
                  f"(別の目印と {v['mixed_other_label']:.2f}) 残らない={v['not_kept']:.2f}(小 {v['csize_too_small']:.2f} 大 {v['csize_too_big']:.2f})"
                  f" まとまり具合 中央値 {v['completeness_median']:.2f} 純度 中央値 {v['purity_median']:.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
