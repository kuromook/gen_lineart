#!/usr/bin/env python
"""実験 H10(2026-10-16 事前登録): 顔の数で伸びるか(a)/ 向きを線から学ぶ(b)/ 向きをパーツの手がかりに足す(c)。学習用だけ。

出力: summary.json、check.json
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h2_face_detect as H2                                            # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h9a_head_detect as H9                                           # noqa: E402

N_LINES, MAX_ORDER = 417, 300
TAU = 0.04
YAW5 = [("front", None), ("oblique", "left"), ("oblique", "right"), ("profile", "left"), ("profile", "right")]
PITCH3 = ["level", "up", "down"]
EYE = H.CLASSES.index("eye")
OTHERS = [H.CLASSES.index(c) for c in ("brow", "nose", "mouth", "ear")]
GRID = 6


def yaw_index(p):
    return YAW5.index((p["yaw"], p.get("dir") if p["yaw"] in ("oblique", "profile") else None))


def layout(f, flip=False):
    """顔の枠(1.2 倍)を 6 × 6 に区切った線の長さの割合、非対称さ、左右の量の差"""
    x0, y0, x1, y1 = f["box"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; hw, hh = 0.6 * (x1 - x0), 0.6 * (y1 - y0)
    p = f["pts"].copy()
    if flip:
        p[..., 0] = 2 * cx - p[..., 0]
    mid = (p[:, 1:] + p[:, :-1]) / 2; L = np.hypot(*(p[:, 1:] - p[:, :-1]).transpose(2, 0, 1))
    u = (mid[..., 0] - (cx - hw)) / (2 * hw); v = (mid[..., 1] - (cy - hh)) / (2 * hh)
    m = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    Hm = np.zeros((GRID, GRID))
    np.add.at(Hm, ((v[m] * GRID).astype(int), (u[m] * GRID).astype(int)), L[m])
    Hm /= max(Hm.sum(), 1e-9)
    return Hm, float(np.abs(Hm - Hm[:, ::-1]).sum()), float(Hm[:, GRID // 2:].sum() - Hm[:, :GRID // 2].sum())


def box_cues(f, head, flip=False):
    fx0, fy0, fx1, fy1 = f["box"]; hx0, hy0, hx1, hy1 = head[:4]; hw, hh = hx1 - hx0, hy1 - hy0
    a = ((fx0 + fx1) / 2 - (hx0 + hx1) / 2) / (hw / 2)
    gl, gr = (fx0 - hx0) / hw, (hx1 - fx1) / hw
    return np.array([-a if flip else a, ((fy0 + fy1) / 2 - (hy0 + hy1) / 2) / (hh / 2), (fx1 - fx0) / hw, (fy1 - fy0) / hh,
                     (fx1 - fx0) / (fy1 - fy0), min(gl, gr)])


def face_feats(f, head, flip=False):
    Hm, asym, lr = layout(f, flip)
    return np.concatenate([Hm.reshape(-1), [asym, lr], box_cues(f, head, flip)])


FLIP5 = [0, 2, 1, 4, 3]


def logreg(Xtr, ytr, Xte, ncls, dev, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev); B = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    y = torch.tensor(ytr, dtype=torch.long, device=dev)
    torch.manual_seed(20261016 + seed)
    lin = torch.nn.Linear(A.shape[1], ncls).to(dev); opt = torch.optim.AdamW(lin.parameters(), lr=0.05, weight_decay=0.05)
    for _ in range(500):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(A), y).backward(); opt.step()
    with torch.no_grad():
        return torch.softmax(lin(B), 1).cpu().numpy()


def pose_cv(F, Ff, y5, y3, fold, tr_idx, te_idx, dev, seed=0):
    """tr_idx で学び(左右反転を足す)、te_idx の確率を返す"""
    Xtr = np.vstack([F[tr_idx], Ff[tr_idx]])
    p5 = logreg(Xtr, np.concatenate([y5[tr_idx], [FLIP5[k] for k in y5[tr_idx]]]), F[te_idx], 5, dev, seed)
    p3 = logreg(Xtr, np.concatenate([y3[tr_idx], y3[tr_idx]]), F[te_idx], 3, dev, seed + 1)
    return p5, p3


def pose_scores(p5, y5, p3, y3):
    pred5 = p5.argmax(1); t = [YAW5[k] for k in y5]; q = [YAW5[k] for k in pred5]
    side = np.array([a[0] in ("oblique", "profile") for a in t])
    pdir = np.array(["right" if (p5[i, 2] + p5[i, 4]) > (p5[i, 1] + p5[i, 3]) else "left" for i in range(len(y5))])
    dir_acc = float(np.mean([pdir[i] == t[i][1] for i in np.flatnonzero(side)]))
    deg = lambda a: a[0]
    tdeg = np.array([deg(a) for a in t]); qdeg = np.array([deg(a) for a in q])
    acc3 = float((tdeg == qdeg).mean()); maj3 = max(Counter(tdeg.tolist()).values()) / len(tdeg)
    pp = p3.argmax(1); accp = float((pp == y3).mean()); majp = max(Counter(y3.tolist()).values()) / len(y3)
    recp = {PITCH3[c]: round(float((pp[y3 == c] == c).mean()), 3) for c in range(3) if (y3 == c).any()}
    rec3 = {c: round(float((qdeg[tdeg == c] == c).mean()), 3) for c in ("front", "oblique", "profile")}
    return dict(direction=dict(n=int(side.sum()), accuracy=round(dir_acc, 3), passed=bool(dir_acc >= 0.85)),
                yaw_degree=dict(accuracy=round(acc3, 3), majority=round(maj3, 3), recall=rec3, passed=bool(acc3 >= 0.70 and acc3 - maj3 >= 0.15)),
                pitch=dict(accuracy=round(accp, 3), majority=round(majp, 3), recall=recp, passed=bool(accp >= 0.70 and accp - majp >= 0.15)))


def parts_cv(faces, fold, feat, dev, subset=None, seed0=10):
    """H7 の cv_probs と同じ。subset(k) = 分割 k の学ぶ側に使う顔の番号(None なら全部)"""
    probs = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = [i for i in range(len(faces)) if fold[i] != k] if subset is None else subset(k)
        te = [i for i in range(len(faces)) if fold[i] == k]
        P = H.fit_predict(np.vstack([feat(i, k) for i in tr]), np.concatenate([faces[i]["y"] for i in tr]),
                          np.vstack([feat(i, k) for i in te]), dev, seed0 + k)
        s = 0
        for i in te:
            n = len(faces[i]["y"]); probs[i] = P[s:s + n]; s += n
    return [p.argmax(1) for p in probs]


def part_eval(faces, preds):
    rows, n_pred, n_hit = H.group_eval(faces, preds, TAU)
    y = np.concatenate([f["y"] for f in faces]); p = np.concatenate(preds)
    out = dict(all=H.rate(rows)[0], eye=H.rate(rows, lambda c, n: c == EYE)[0], others=H.rate(rows, lambda c, n: c in OTHERS)[0],
               others_size12=H.rate(rows, lambda c, n: c in OTHERS and n <= 2)[0], pred_group_precision=round(n_hit / max(n_pred, 1), 3))
    for c in OTHERS + [EYE]:
        out[H.CLASSES[c]] = H.rate(rows, lambda cc, n, c=c: cc == c)[0]
        tp = int(((p == c) & (y == c)).sum())
        out[H.CLASSES[c] + "_stroke_PR"] = [round(tp / max(int((p == c).sum()), 1), 3), round(tp / max(int((y == c).sum()), 1), 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h10_scale_pose_20261016")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faces = H.load_faces(MAX_ORDER, N_LINES)
    assert all(f["pose"] for f in faces)
    fold, g2f = H.folds_of(faces)
    FB = [H.feats_B(f) for f in faces]
    rng = np.random.default_rng(20261016)

    # 頭の枠
    try:
        H2.CONF = float(json.load(open(H9.hf_hub_download(H9.REPO, f"{H9.MODEL}/threshold.json")))["threshold"])
    except Exception:
        H2.CONF = 0.3
    det = H9.HeadDetector(); cache, heads = {}, []
    for f in faces:
        if f["panel"] not in cache:
            cache[f["panel"]] = det(G.panel_img(f["panel"]))
        h = H9.match(f["box"], cache[f["panel"]])
        heads.append(h if h else (f["box"][0] - 0.15 * (f["box"][2] - f["box"][0]), f["box"][1] - 0.3 * (f["box"][3] - f["box"][1]),
                                  f["box"][2] + 0.15 * (f["box"][2] - f["box"][0]), f["box"][3], 0.0))   # 見つからなければ顔の枠から仮に作る
    n_nohead = sum(1 for f in faces if not H9.match(f["box"], cache[f["panel"]]))
    PF = np.array([face_feats(f, h) for f, h in zip(faces, heads)]); PFf = np.array([face_feats(f, h, True) for f, h in zip(faces, heads)])
    y5 = np.array([yaw_index(f["pose"]) for f in faces]); y3 = np.array([PITCH3.index(f["pose"]["pitch"]) for f in faces])

    # 道具確認
    ok1 = all(len({fold[i] for i in range(len(faces)) if faces[i]["group"] == g}) == 1 for g in g2f)
    tp = [f["y"].copy() for f in faces]; ok2 = all(v["precision"] == 1 and v["recall"] == 1 for v in H.stroke_pr(faces, tp).values() if v["n"])
    Hm, asym, lr = layout(faces[0]); Hf, asf, lrf = layout(faces[0], True)
    ok4 = bool(np.allclose(Hf, Hm[:, ::-1]) and abs(asym - asf) < 1e-9 and abs(lr + lrf) < 1e-9)
    def pose_oof(y5_, y3_, seed=0):
        P5 = np.zeros((len(faces), 5)); P3 = np.zeros((len(faces), 3))
        for k in sorted(set(fold.tolist())):
            tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
            P5[te], P3[te] = pose_cv(PF, PFf, y5_, y3_, fold, tr, te, dev, seed + 7 * k)
        return P5, P3
    perm = rng.permutation(len(faces))
    n5, n3 = pose_oof(y5[perm], y3[perm], seed=500)
    neg = pose_scores(n5, y5[perm], n3, y3[perm])
    ok3 = (0.35 <= neg["direction"]["accuracy"] <= 0.65 and neg["yaw_degree"]["accuracy"] <= neg["yaw_degree"]["majority"] + 0.05
           and neg["pitch"]["accuracy"] <= neg["pitch"]["majority"] + 0.05)
    # 学ぶ側を減らしても当てる側が同じことの確認は、parts_cv の作り(当てる側 = fold == k、学ぶ側 = 与えた subset)で保証。subset が当てる側と重ならないことを確かめる
    subsets = {}
    for frac in (0.25, 0.5, 0.75):
        for rep in range(3):
            for k in sorted(set(fold.tolist())):
                tr = np.flatnonzero(fold != k)
                subsets[(frac, rep, k)] = sorted(rng.choice(tr, max(1, int(round(frac * len(tr)))), replace=False).tolist())
    ok1b = all(not any(fold[i] == k for i in v) for (fr, rp, k), v in subsets.items())
    chk = dict(n_faces=len(faces), n_strokes=int(sum(len(f["y"]) for f in faces)), n_no_head_box=n_nohead, folds_disjoint=bool(ok1 and ok1b),
               truth_pr_is_1=bool(ok2), flip_ok=ok4, negative_pose=neg)
    chk["passed"] = bool(ok1 and ok1b and ok2 and ok3 and ok4)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    S = {}
    # H10a
    full = {s0: part_eval(faces, parts_cv(faces, fold, lambda i, k: FB[i], dev, seed0=s0)) for s0 in (10, 110, 210)}
    spread = [v["all"] for v in full.values()]; spread_o = [v["others"] for v in full.values()]
    S["a_seed_spread"] = dict(all=spread, others=spread_o, range_all=round(max(spread) - min(spread), 3), range_others=round(max(spread_o) - min(spread_o), 3))
    curve = {"1.0": full[10]}
    for frac in (0.25, 0.5, 0.75):
        reps = [part_eval(faces, parts_cv(faces, fold, lambda i, k: FB[i], dev, subset=lambda k, fr=frac, rp=rep: subsets[(fr, rp, k)], seed0=10))
                for rep in range(3)]
        curve[str(frac)] = {key: (round(float(np.mean([r[key] for r in reps])), 3) if not isinstance(reps[0][key], list) else reps[0][key]) for key in reps[0]}
    S["a_curve"] = curve
    S["a_decision_delta_100_minus_50"] = round(curve["1.0"]["all"] - curve["0.5"]["all"], 3)
    print("H10a", json.dumps(dict(spread=S["a_seed_spread"], curve={k: {kk: v[kk] for kk in ("all", "eye", "others", "others_size12", "pred_group_precision")} for k, v in curve.items()}), ensure_ascii=False), flush=True)

    # H10b
    P5, P3 = pose_oof(y5, y3)
    S["b_pose"] = pose_scores(P5, y5, P3, y3)
    print("H10b", json.dumps(S["b_pose"], ensure_ascii=False), flush=True)

    # H10c: 学ぶ側の向きの確率は、学ぶ側の中でもう一度分割して作る
    inner = {}
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k)
        sub = [faces[i] for i in tr]; subfold, _ = H.folds_of(sub, 4)
        I5 = np.zeros((len(tr), 5)); I3 = np.zeros((len(tr), 3))
        for kk in sorted(set(subfold.tolist())):
            a_ = tr[subfold != kk]; b_ = tr[subfold == kk]
            I5[subfold == kk], I3[subfold == kk] = pose_cv(PF, PFf, y5, y3, None, a_, b_, dev, 900 + 10 * k + kk)
        inner[k] = {int(i): np.concatenate([I5[j], I3[j]]) for j, i in enumerate(tr)}
    def withpose(i, k):
        v = np.concatenate([P5[i], P3[i]]) if fold[i] == k else inner[k][i]
        return np.column_stack([FB[i], np.tile(v, (len(FB[i]), 1))])
    def withtrue(i, k):
        v = np.concatenate([np.eye(5)[y5[i]], np.eye(3)[y3[i]]])
        return np.column_stack([FB[i], np.tile(v, (len(FB[i]), 1))])
    S["c_parts"] = dict(B=full[10], B_plus_predicted_pose=part_eval(faces, parts_cv(faces, fold, withpose, dev, seed0=10)),
                        B_plus_true_pose=part_eval(faces, parts_cv(faces, fold, withtrue, dev, seed0=10)))
    S["c_delta_others_predicted"] = round(S["c_parts"]["B_plus_predicted_pose"]["others"] - full[10]["others"], 3)
    S["c_delta_others_true"] = round(S["c_parts"]["B_plus_true_pose"]["others"] - full[10]["others"], 3)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("H10c", json.dumps(S["c_parts"], ensure_ascii=False), S["c_delta_others_predicted"], S["c_delta_others_true"]); print("->", out)


if __name__ == "__main__":
    main()
