#!/usr/bin/env python
"""実験 H9b(2026-10-12 事前登録): 顔の枠と頭の枠の位置関係から、頭の向きを推定する。学習用で向きの目印がある顔だけ。

出力: summary.json、check.json、faces.csv、scatter.png
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h2_face_detect as H2                                            # noqa: E402
import h9a_head_detect as H9                                           # noqa: E402

YAW3 = ["front", "oblique", "profile"]
PITCH3 = ["level", "up", "down"]


def cues(face, head):
    fx0, fy0, fx1, fy1 = face; hx0, hy0, hx1, hy1 = head[:4]
    hw, hh = hx1 - hx0, hy1 - hy0
    return dict(a=((fx0 + fx1) / 2 - (hx0 + hx1) / 2) / (hw / 2), b=((fy0 + fy1) / 2 - (hy0 + hy1) / 2) / (hh / 2),
                rw=(fx1 - fx0) / hw, rh=(fy1 - fy0) / hh)


def loo_centroid(X, y, groups, classes):
    """シリーズ群を 1 つずつ抜く。クラスの平均にいちばん近いものを選ぶ(手がかりは学ぶ側のばらつきで割る)"""
    pred = np.empty(len(y), dtype=object)
    for g in sorted(set(groups)):
        te = groups == g; tr = ~te
        sd = X[tr].std(0) + 1e-9
        cs = [c for c in classes if (y[tr] == c).any()]
        M = np.array([X[tr][y[tr] == c].mean(0) for c in cs])
        D = np.linalg.norm((X[te][:, None] - M[None]) / sd, axis=-1)
        pred[te] = [cs[k] for k in D.argmin(1)]
    return pred


def score(y, pred, classes):
    acc = float((y == pred).mean()); maj = max(Counter(y.tolist()).values()) / len(y)
    rec = {c: (round(float((pred[y == c] == c).mean()), 3) if (y == c).any() else None) for c in classes}
    conf = {c: dict(Counter(pred[y == c].tolist())) for c in classes}
    return dict(n=len(y), accuracy=round(acc, 3), majority_baseline=round(maj, 3), mean_recall=round(float(np.mean([v for v in rec.values() if v is not None])), 3),
                recall=rec, confusion_truth_to_pred=conf, counts=dict(Counter(y.tolist())), passed=bool(acc >= 0.70 and acc - maj >= 0.15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h9b_head_pose_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    n_lines = sum(1 for _ in open(G.LAB / "face_groups.jsonl"))
    G.load_state(G.LAB / "face_groups.jsonl")
    recs = [r for o, r in sorted(G.S["faces"].items()) if r["usage"] == "train" and r["status"] == "done" and r.get("pose")]

    # 道具確認: 合成(球の模型)
    k, b0, r0 = 0.5, 0.25, 0.7
    psi = np.radians(np.arange(-90, 91, 15))
    syn = [cues((k * np.sin(p) - r0 * max(np.cos(p), 0.3), b0 - 0.6, k * np.sin(p) + r0 * max(np.cos(p), 0.3), b0 + 0.6), (-1, -1, 1, 1))["a"] for p in psi]
    ok_syn = bool(np.all(np.sign(syn) == np.sign(psi)) and np.all(np.diff(np.abs(syn)[psi >= 0]) > 0))

    try:
        H2.CONF = float(json.load(open(H9.hf_hub_download(H9.REPO, f"{H9.MODEL}/threshold.json")))["threshold"])
    except Exception:
        H2.CONF = 0.3
    det = H9.HeadDetector(); cache = {}
    rows = []
    for r in recs:
        q = G.S["queue"][r["order"]]; pid = int(q["panel"])
        if pid not in cache:
            cache[pid] = det(G.panel_img(pid))
        face = tuple(float(q[c]) for c in ("x0", "y0", "x1", "y1"))
        h = H9.match(face, cache[pid])
        row = dict(order=r["order"], group=q["group"], yaw=r["pose"]["yaw"], dir=r["pose"]["dir"] or "", pitch=r["pose"]["pitch"], head=bool(h))
        row.update({c: round(v, 4) for c, v in cues(face, h).items()} if h else dict(a="", b="", rw="", rh=""))
        rows.append(row)
    with open(out / "faces.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    R = [t for t in rows if t["head"]]
    A = np.array([t["a"] for t in R]); B = np.array([t["b"] for t in R]); RW = np.array([t["rw"] for t in R]); RH = np.array([t["rh"] for t in R])
    yaw = np.array([t["yaw"] for t in R]); dr = np.array([t["dir"] for t in R]); pitch = np.array([t["pitch"] for t in R]); grp = np.array([t["group"] for t in R])

    side = np.isin(yaw, ["oblique", "profile"]) & np.isin(dr, ["left", "right"])
    dir_acc = float(((A[side] > 0) == (dr[side] == "right")).mean())
    rng = np.random.default_rng(20261012)
    perm = rng.permutation(len(A)); neg1 = float(((A[perm][side] > 0) == (dr[side] == "right")).mean())
    negm = float(np.mean([((A[rng.permutation(len(A))][side] > 0) == (dr[side] == "right")).mean() for _ in range(200)]))
    chk = dict(jsonl_lines=n_lines, n_train_with_pose=len(rows), n_with_head=len(R), synthetic_ok=ok_syn,
               negative_direction_accuracy=round(neg1, 3), negative_direction_accuracy_mean200=round(negm, 3))
    chk["passed"] = bool(ok_syn and 0.35 <= neg1 <= 0.65)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    S = dict(counts=dict(yaw=dict(Counter(zip(yaw.tolist(), dr.tolist())).most_common()) and {f"{y}/{d}": v for (y, d), v in Counter(zip(yaw.tolist(), dr.tolist())).items()},
                         pitch=dict(Counter(pitch.tolist()))))
    S["1_direction"] = dict(n=int(side.sum()), accuracy=round(dir_acc, 3), passed=bool(dir_acc >= 0.85),
                            by_yaw={y: dict(n=int((side & (yaw == y)).sum()), accuracy=round(float(((A[side & (yaw == y)] > 0) == (dr[side & (yaw == y)] == "right")).mean()), 3))
                                    for y in ("oblique", "profile") if (side & (yaw == y)).any()})
    m3 = np.isin(yaw, YAW3)
    S["2_yaw_degree"] = score(yaw[m3], loo_centroid(np.column_stack([np.abs(A), RW])[m3], yaw[m3], grp[m3], YAW3), YAW3)
    S["2_yaw_degree"]["n_back_excluded"] = int((yaw == "back").sum())
    S["3_pitch"] = score(pitch, loo_centroid(np.column_stack([B, RH]), pitch, grp, PITCH3), PITCH3)
    rank = np.array([YAW3.index(y) for y in yaw[m3]])
    S["4_descriptive"] = dict(spearman_absa_vs_yaw=round(float(spearmanr(np.abs(A[m3]), rank)[0]), 3), spearman_rw_vs_yaw=round(float(spearmanr(RW[m3], rank)[0]), 3),
                              abs_a_median={y: round(float(np.median(np.abs(A[yaw == y]))), 3) for y in YAW3 if (yaw == y).any()},
                              rw_median={y: round(float(np.median(RW[yaw == y])), 3) for y in YAW3 if (yaw == y).any()},
                              b_median={p: round(float(np.median(B[pitch == p])), 3) for p in PITCH3 if (pitch == p).any()},
                              rh_median={p: round(float(np.median(RH[pitch == p])), 3) for p in PITCH3 if (pitch == p).any()})
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S, ensure_ascii=False, indent=1))

    # 図: 小さな図 2 枚(左 = 左右、右 = 上下)。色は目印、形は使わず凡例で示す
    W = 460; pad = 50
    def panel(xs, ys, labs, order, cols, xl, yl, xr, yr, title):
        im = np.full((W + 80, W + 40, 3), 255, np.uint8)
        def px(x, y):
            return (int(pad + (x - xr[0]) / (xr[1] - xr[0]) * (W - pad - 10)), int(30 + (yr[1] - y) / (yr[1] - yr[0]) * (W - 40)))
        cv2.rectangle(im, px(xr[0], yr[1]), px(xr[1], yr[0]), (200, 200, 200), 1)
        if xr[0] < 0 < xr[1]:
            cv2.line(im, px(0, yr[0]), px(0, yr[1]), (225, 225, 225), 1)
        for lab in order:
            for x, y in zip(xs[labs == lab], ys[labs == lab]):
                cv2.circle(im, px(np.clip(x, *xr), np.clip(y, *yr)), 5, cols[lab], -1, cv2.LINE_AA); cv2.circle(im, px(np.clip(x, *xr), np.clip(y, *yr)), 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, title, (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(im, f"{xl}  [{xr[0]}, {xr[1]}]", (pad, W + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(im, f"vertical: {yl}  [{yr[0]}, {yr[1]}]", (pad, W + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        for j, lab in enumerate(order):
            x, y = pad + 150 * (j % 3), W + 48 + 18 * (j // 3)
            cv2.circle(im, (x + 6, y), 5, cols[lab], -1, cv2.LINE_AA)
            cv2.putText(im, f"{lab} ({int((labs == lab).sum())})", (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        return im
    yl = np.array([y if y in ("front", "back") else f"{y}-{d}" for y, d in zip(yaw, dr)])
    c1 = {"front": (120, 120, 120), "oblique-left": (200, 150, 60), "oblique-right": (60, 140, 230), "profile-left": (130, 70, 20), "profile-right": (20, 60, 190), "back": (60, 170, 60)}
    im1 = panel(A, RW, yl, [k for k in c1 if (yl == k).any()], c1, "a = face offset in head box (+ = right)", "rw = face width / head width", (-1.0, 1.0), (0.3, 1.1), "left-right")
    c2 = {"level": (120, 120, 120), "up": (60, 140, 230), "down": (200, 150, 60)}
    im2 = panel(B, RH, pitch, PITCH3, c2, "b = face offset in head box (+ = lower)", "rh = face height / head height", (-0.4, 0.8), (0.3, 1.1), "up-down")
    cv2.imwrite(str(out / "scatter.png"), np.hstack([im1, im2])); print("->", out / "scatter.png")


if __name__ == "__main__":
    main()
