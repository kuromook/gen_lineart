#!/usr/bin/env python
"""実験 H9c(2026-10-12 事前登録): 作者の読み方(張り付き・縦長・耳)を向きの手がかりにする。学習用で向きの目印がある顔だけ。

出力: summary.json、check.json、faces.csv
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h2_face_detect as H2                                            # noqa: E402
import h9a_head_detect as H9                                           # noqa: E402

N_LINES = 225


def cues(face, head, shape=None):
    fx0, fy0, fx1, fy1 = face; hx0, hy0, hx1, hy1 = head[:4]
    hw = hx1 - hx0
    gl, gr = (fx0 - hx0) / hw, (hx1 - fx1) / hw
    d = dict(aspect=(fx1 - fx0) / (fy1 - fy0), gap_l=gl, gap_r=gr, stick=min(gl, gr), side="left" if gl < gr else "right")
    if shape is not None:
        H, W = shape[:2]
        d["cut"] = bool(hx0 <= 0.02 * W or hy0 <= 0.02 * H or hx1 >= 0.98 * W or hy1 >= 0.98 * H)
    return d


def pr(pred, y):
    tp = int((pred & y).sum())
    p = tp / pred.sum() if pred.any() else float("nan"); r = tp / max(int(y.sum()), 1)
    return p, r, (2 * p * r / (p + r) if tp else 0.0)


def fit_A(stick, asp, y):
    T1, T2 = np.unique(stick), np.unique(asp)                    # 全部の組を一度に計算(最初に最大になる組を採る)
    P = (stick[None] <= T1[:, None])[:, None, :] & (asp[None] <= T2[:, None])[None, :, :]
    tp = (P & y).sum(-1); F = np.where(tp > 0, 2 * tp / np.maximum(P.sum(-1) + y.sum(), 1), 0.0)
    i, j = np.unravel_index(int(F.argmax()), F.shape)
    return T1[i], T2[j]


def fit_B(stick, y):
    best = None
    for t in np.unique(stick):
        p, r, _ = pr(stick <= t, y)
        if p >= 0.9 and (best is None or r > best[0]):
            best = (r, t)
    return best[1] if best else float(np.min(stick))


def run(stick, asp, side, yaw, dr, grp):
    predA = np.zeros(len(yaw), bool); predB = np.zeros(len(yaw), bool)
    for g in sorted(set(grp)):
        te = grp == g; tr = ~te
        t1, t2 = fit_A(stick[tr], asp[tr], yaw[tr] == "profile")
        predA[te] = (stick[te] <= t1) & (asp[te] <= t2)
        predB[te] = stick[te] <= fit_B(stick[tr], yaw[tr] != "front")
    return predA, predB


def report(predA, predB, side, yaw, dr, sel=None):
    m = np.ones(len(yaw), bool) if sel is None else sel
    pa = pr(predA[m], (yaw == "profile")[m]); pb = pr(predB[m], (yaw != "front")[m])
    c = m & predB & np.isin(yaw, ["oblique", "profile"]) & (dr != "")
    return dict(n=int(m.sum()),
                A_profile=dict(precision=round(pa[0], 3), recall=round(pa[1], 3), f=round(pa[2], 3), n_pred=int(predA[m].sum()), n_true=int((yaw == "profile")[m].sum())),
                B_not_front=dict(precision=round(pb[0], 3), recall=round(pb[1], 3), n_pred=int(predB[m].sum()), n_true=int((yaw != "front")[m].sum()),
                                 pred_by_yaw=dict(Counter(yaw[m & predB].tolist()))),
                C_side=dict(n=int(c.sum()), accuracy=round(float((side[c] == dr[c]).mean()), 3) if c.any() else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h9c_pose_cues_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    G.load_state(G.LAB / "face_groups.jsonl")
    faces = {}
    for k, ln in enumerate(open(G.LAB / "face_groups.jsonl")):               # H9b と同じ時点(最初の 225 行)
        if k < N_LINES and ln.strip():
            d = json.loads(ln); faces[int(d["order"])] = d
    recs = [r for o, r in sorted(faces.items()) if r["usage"] == "train" and r["status"] == "done" and r.get("pose")]

    s1 = cues((30, 20, 70, 80), (0, 0, 100, 100)); s2 = cues((60, 20, 100, 100), (0, 0, 100, 100))
    ok_syn = (abs(s1["stick"] - 0.3) < 1e-9 and abs(s1["aspect"] - 40 / 60) < 1e-9 and s2["stick"] == 0 and s2["side"] == "right" and abs(s2["aspect"] - 0.5) < 1e-9)

    try:
        H2.CONF = float(json.load(open(H9.hf_hub_download(H9.REPO, f"{H9.MODEL}/threshold.json")))["threshold"])
    except Exception:
        H2.CONF = 0.3
    det = H9.HeadDetector(); cache = {}
    rows, boxes = [], []
    for r in recs:
        q = G.S["queue"][r["order"]]; pid = int(q["panel"]); img = G.panel_img(pid)
        if pid not in cache:
            cache[pid] = det(img)
        face = tuple(float(q[c]) for c in ("x0", "y0", "x1", "y1"))
        h = H9.match(face, cache[pid])
        if not h:
            continue
        c = cues(face, h, img.shape); boxes.append((pid, face, h))
        cxf = (face[0] + face[2]) / 2
        ears = []
        pts, _w, _cl = G.panel_strokes(pid)
        for g in r["groups"]:
            if g["label"] == "face.ear":
                ears.append("left" if pts[g["strokes"]].reshape(-1, 2)[:, 0].mean() < cxf else "right")
        rows.append(dict(order=r["order"], group=q["group"], yaw=r["pose"]["yaw"], dir=r["pose"]["dir"] or "", pitch=r["pose"]["pitch"],
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}, n_ears=len(ears), ear_sides="+".join(sorted(ears))))
    with open(out / "faces.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    stick = np.array([t["stick"] for t in rows]); asp = np.array([t["aspect"] for t in rows]); side = np.array([t["side"] for t in rows])
    yaw = np.array([t["yaw"] for t in rows]); dr = np.array([t["dir"] for t in rows]); grp = np.array([t["group"] for t in rows]); cut = np.array([t["cut"] for t in rows])

    rng = np.random.default_rng(20261012)
    predA, predB = run(stick, asp, side, yaw, dr, grp)
    real_f = report(predA, predB, side, yaw, dr)["A_profile"]["f"]
    nf, nc = [], []
    for _ in range(200):                                         # 2026-10-12 に直した陰性対照(かき混ぜ 200 回の分布)
        p = rng.permutation(len(rows))
        nA, nB = run(stick, asp, side, yaw[p], dr[p], grp)
        neg = report(nA, nB, side, yaw[p], dr[p])
        nf.append(neg["A_profile"]["f"])
        if neg["C_side"]["accuracy"] is not None:
            nc.append(neg["C_side"]["accuracy"])
    p95 = float(np.percentile(nf, 95)); cmean = float(np.mean(nc)) if nc else None
    chk = dict(n_faces=len(rows), synthetic_ok=bool(ok_syn), real_A_f=real_f, shuffled_A_f_median=round(float(np.median(nf)), 3),
               shuffled_A_f_p95=round(p95, 3), shuffled_C_accuracy_mean=None if cmean is None else round(cmean, 3), shuffled_C_n_runs=len(nc))
    chk["passed"] = bool(ok_syn and real_f > p95 and (cmean is None or 0.35 <= cmean <= 0.65))
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    S = dict(all=report(predA, predB, side, yaw, dr), no_cut=report(predA, predB, side, yaw, dr, ~cut),
             D_cut=dict(n_cut=int(cut.sum()), by_yaw=dict(Counter(yaw[cut].tolist()))),
             thresholds_all_data=dict(A=[round(float(v), 3) for v in fit_A(stick, asp, yaw == "profile")], B=round(float(fit_B(stick, yaw != "front")), 3)),
             medians={y: dict(n=int((yaw == y).sum()), stick=round(float(np.median(stick[yaw == y])), 3), aspect=round(float(np.median(asp[yaw == y])), 3)) for y in ("front", "oblique", "profile")},
             E_ears={y: dict(Counter(min(t["n_ears"], 2) for t in rows if t["yaw"] == y)) for y in ("front", "oblique", "profile")})
    one = [t for t in rows if t["n_ears"] == 1 and t["dir"]]
    S["E_one_ear_side_vs_dir"] = dict(n=len(one), ear_on_opposite_side_of_facing=sum(t["ear_sides"] != t["dir"] for t in one))
    S["passed"] = dict(A=bool(S["all"]["A_profile"]["precision"] >= 0.8 and S["all"]["A_profile"]["recall"] >= 0.6),
                       B=bool(S["all"]["B_not_front"]["precision"] >= 0.85),
                       C=bool(S["all"]["C_side"]["accuracy"] is not None and S["all"]["C_side"]["accuracy"] >= 0.85))
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, ensure_ascii=False, indent=1))

    import cv2
    CELL, tiles = 230, []
    code = {"front": "F", "oblique": "O", "profile": "P", "back": "B"}
    for t, (pid, f, h), pa, pb in zip(rows, boxes, predA, predB):
        xs, ys = [f[0], f[2], h[0], h[2]], [f[1], f[3], h[1], h[3]]
        sd = 1.25 * max(max(xs) - min(xs), max(ys) - min(ys)); cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        k = CELL / sd
        A = np.array([[k, 0, CELL / 2 - cx * k], [0, k, CELL / 2 - cy * k]])
        cell = cv2.warpAffine(G.panel_img(pid), A, (CELL, CELL), flags=cv2.INTER_AREA, borderValue=(236, 236, 236))
        for b, col in ((f, (222, 141, 60)), (h, (30, 130, 235))):
            q = (np.array([[b[0], b[1]], [b[2], b[3]]]) @ A[:, :2].T + A[:, 2]).round().astype(int)
            cv2.rectangle(cell, tuple(q[0]), tuple(q[1]), col, 2, cv2.LINE_AA)
        head = np.full((22, CELL, 3), 255, np.uint8)
        lab = code[t["yaw"]] + ("-" + t["dir"][0].upper() if t["dir"] else "") + "/" + t["pitch"]
        cv2.putText(head, f"face {t['order'] + 1}  {lab}" + ("  cut" if t["cut"] else "") + ("  [stuck]" if pb else ""), (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(np.vstack([head, cell]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 9:
        tiles.append(np.full_like(tiles[0], 255))
    g = np.vstack([np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)])
    title = np.full((28, g.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "blue = face box, orange = head box.  label = artist's pose (F front / O oblique / P profile, -L/-R facing, level/up/down).  cut = head box at panel edge.  [stuck] = judged 'stuck' by rule B",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "all_faces.png"), np.vstack([title, g])); print("->", out / "all_faces.png")


if __name__ == "__main__":
    main()
