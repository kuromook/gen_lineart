#!/usr/bin/env python
"""K8 弦に沿った墨の地続き(2026-10-18 事前登録)。

述語: C-1 の襟(tip から弦に沿って進み、自分の 8 近傍連結成分を出た点から測る)が
**標本区間を返さない**こと。下位区分は
  `自分の連結成分を出ないまま他方の tip に達した` / `両端が出た後の区間が空`。

**「隙間と呼べるのか」とは呼ばない**(T4-touch で構成した本物の埋まった隙間 30 件中
4 件が同じ箱に落ちたため)。

- 自明な腕: 同じ述語を箱 A(連鎖器が繋いだ 521 件)に当てる
- 層別: 隙間長のビンは測る前に固定(下の GAP_BINS)
- 分割: comp_same(2 つの tip が同じ連結成分か)

帯も帰無値も無い記述。器具確認は再現のみ。
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K                                  # pipeline / comp_collar / comp_at

GAP_BINS = [0, 2, 4, 8, 12, 16, 20, 24, 10 ** 9]
EXPECT = {"A": 521, "B": 182, "C": 79}


def bin_of(g):
    for i in range(len(GAP_BINS) - 1):
        if GAP_BINS[i] <= g < GAP_BINS[i + 1]:
            hi = GAP_BINS[i + 1]
            return f"[{GAP_BINS[i]},{'inf' if hi > 10 ** 8 else hi})"
    return "?"


def work(t):
    pid, source, name, rows = t
    g = cv2.imread(str(K.DATASET / source / "line" / name), 0)
    if g is None:
        return []
    P = K.pipeline(g)
    out = []
    for r in rows:
        i, j, ei, ej = int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"])
        if i not in P["tips"] or j not in P["tips"]:
            out.append(dict(r, missing=1)); continue
        pa, pb = P["tips"][i][ei], P["tips"][j][ej]
        a2, _b2, t_a, t_b, why = K.comp_collar(P, pa, pb)
        out.append(dict(r, missing=0, contiguous=int(a2 is None), reason=why,
                        t_a=round(t_a, 2), t_b=round(t_b, 2),
                        comp_same=int(K.comp_at(P, pa) == K.comp_at(P, pb)),
                        gap_px=round(float(np.hypot(*(pb - pa))), 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pairs", default="results/k2_missed_candidates_20261017/per_pair.csv")
    ap.add_argument("--snapshot", default="results/k1_broken_vs_fill_20261016/labels_snapshot.jsonl")
    ap.add_argument("--out", default="results/k8_contiguity_20261018")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    lab = {}
    for l in open(ROOT / a.snapshot):
        d = json.loads(l)
        for gg in d.get("groups", []):
            for s in gg["strokes"]:
                lab[(int(d["order"]), int(s))] = gg["label"]
    rows = [r for r in csv.DictReader(open(ROOT / a.pairs))
            if r["same"] == "1" and r["passed"] == "1"]
    chk = dict(box_counts=dict(Counter(r["box"] for r in rows)), n=len(rows))
    chk["ok"] = chk["box_counts"] == EXPECT and chk["n"] == sum(EXPECT.values())
    print("器具確認 再現:", json.dumps(chk, ensure_ascii=False), flush=True)
    assert chk["ok"], "母集団が再現しない。止める。"

    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    byp = defaultdict(list)
    for r in rows:
        byp[int(r["panel"])].append(r)
    tasks = [(p, P[p]["source"], P[p]["name"], rs) for p, rs in byp.items()]
    print(f"対 {len(rows)} 件 / コマ {len(tasks)} 枚", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for part in pool.imap_unordered(work, tasks, chunksize=2):
            res += part
    for r in res:
        r["label"] = lab.get((int(r["order"]), int(r["t0"])), "?")
        r["gap_bin"] = bin_of(float(r["gap"]))

    fn = ["order", "panel", "box", "label", "gap", "gap_px", "gap_bin", "missing",
          "contiguous", "reason", "t_a", "t_b", "comp_same", "s0", "s1", "e0", "e1"]
    with open(out / "per_pair_contig.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fn, extrasaction="ignore", restval="")
        w.writeheader(); w.writerows(res)

    ok = [r for r in res if not r["missing"]]
    def rate(sel):
        n = len(sel)
        c = sum(r["contiguous"] for r in sel)
        return dict(n=n, contiguous=c, rate=round(c / n, 4) if n else None)
    by_box = {b: rate([r for r in ok if r["box"] == b]) for b in "ABC"}
    bins = [bin_of(x + 0.5) for x in GAP_BINS[:-1]]
    by_box_bin = {b: {bn: rate([r for r in ok if r["box"] == b and r["gap_bin"] == bn])
                      for bn in bins} for b in "ABC"}
    boxb = [r for r in ok if r["box"] == "B"]
    cont_b = [r for r in boxb if r["contiguous"]]
    split = dict(n=len(cont_b),
                 by_reason=dict(Counter(r["reason"] for r in cont_b)),
                 by_comp_same=dict(Counter(r["comp_same"] for r in cont_b)),
                 by_reason_comp=dict(Counter(f"{r['reason']} / comp_same={r['comp_same']}"
                                             for r in cont_b)),
                 by_label=dict(Counter(r["label"] for r in cont_b).most_common(8)),
                 by_gap_bin=dict(Counter(r["gap_bin"] for r in cont_b)))
    summary = dict(
        registered=dict(predicate="C-1 の襟が標本区間を返さない(弦に沿って墨が地続き)",
                        name="2 本の tip の墨が、直線の弦に沿って地続きである",
                        rejected_name="「ここは隙間と呼べるのか」(T4-touch の本物の埋まった隙間 "
                                      "30 件中 4 件が同じ箱に落ちるため言い過ぎ)",
                        gap_bins=GAP_BINS, trivial_arm="同じ述語を箱 A に当てる",
                        note="帯も帰無値も無い記述。合否判定は置かない"),
        instrument=dict(reproduce=chk, n_missing=sum(r["missing"] for r in res)),
        by_box=by_box, by_box_and_gap_bin=by_box_bin, box_b_contiguous_split=split)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
