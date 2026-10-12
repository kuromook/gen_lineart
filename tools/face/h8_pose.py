#!/usr/bin/env python
"""実験 H8(2026-10-12 事前登録): 正解の目から作った「目の枠組み」(向きの代わり)を教えると、眉・鼻・口・耳は当たるようになるか。学習用だけ。

出力: summary.json、check.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402

EYE = H.CLASSES.index("eye")
OTHERS = [H.CLASSES.index(c) for c in ("brow", "nose", "mouth", "ear")]
TAU = 0.04


def eye_frame(f):
    """正解の目のグループから。返る: dict(n, E (x, y), d, ang, w) か None"""
    gs = sorted({int(g) for g in f["gid"][(f["y"] == EYE) & (f["gid"] >= 0)]})
    if not gs:
        return dict(n=0)
    info = []
    for g in gs:
        P = f["pts"][f["gid"] == g].reshape(-1, 2)
        info.append((int((f["gid"] == g).sum()), P.mean(0), float(np.ptp(P[:, 0]))))
    info.sort(key=lambda t: -t[0])
    w = float(np.mean([t[2] for t in info[:2]]))
    if len(info) == 1:
        return dict(n=1, E=info[0][1], d=0.0, ang=0.0, w=w)
    a, b = sorted([info[0][1], info[1][1]], key=lambda c: c[0])
    v = b - a
    return dict(n=2, E=(a + b) / 2, d=float(np.hypot(*v)), ang=float(np.arctan2(v[1], v[0])), w=w)


def feats_D(f, fr):
    n = len(f["pts"]); s = f["s"]
    B = H.feats_B(f)
    if fr["n"] == 0:
        return np.column_stack([B, np.tile([1, 0, 0], (n, 1)), np.zeros((n, 14))])
    c, sn = np.cos(-fr["ang"]), np.sin(-fr["ang"])
    R = np.array([[c, -sn], [sn, c]])
    def rel(xy):
        return (xy - fr["E"]) @ R.T
    mid, a, b = rel(f["pts"].mean(1)), rel(f["pts"][:, 0]), rel(f["pts"][:, -1])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    Euv = H.uv(f, np.asarray(fr["E"]))
    glob = [fr["n"] == 1, fr["n"] == 2, Euv[0], Euv[1], fr["d"] / s, np.sin(fr["ang"]), np.cos(fr["ang"]), fr["w"] / s]
    bydist = mid / fr["d"] if fr["d"] > 0 else np.zeros_like(mid)
    return np.column_stack([B, np.zeros(n), np.tile(np.array(glob, float), (n, 1)), mid / s, lo / s, hi / s, bydist])


def others_eval(faces, preds, sel=None):
    idx = [i for i in range(len(faces)) if sel is None or sel[i]]
    fs = [faces[i] for i in idx]; ps = [preds[i] for i in idx]
    y = np.concatenate([f["y"] for f in fs]); p = np.concatenate(ps)
    out = {}
    for name, cs in [("brow+nose+mouth+ear", OTHERS)] + [(H.CLASSES[c], [c]) for c in OTHERS]:
        tp = int(sum(((p == c) & (y == c)).sum() for c in cs)); npred = int(np.isin(p, cs).sum()); npos = int(np.isin(y, cs).sum())
        rows, _n, _h = H.group_eval(fs, ps, TAU)
        r, ng = H.rate(rows, lambda c, n, cs=cs: c in cs)
        r12, n12 = H.rate(rows, lambda c, n, cs=cs: c in cs and n <= 2)
        out[name] = dict(stroke_precision=round(tp / max(npred, 1), 3), stroke_recall=round(tp / max(npos, 1), 3), n_strokes=npos,
                         group_recovered=r, n_groups=ng, group_recovered_size12=r12, n_groups_size12=n12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h8_pose_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faces = H.load_faces()
    fold, _ = H.folds_of(faces)
    FB = [H.feats_B(f) for f in faces]
    frames = [eye_frame(f) for f in faces]
    FD = [feats_D(f, fr) for f, fr in zip(faces, frames)]
    rng = np.random.default_rng(20261012)
    perm = rng.permutation(len(faces))
    while np.any(perm == np.arange(len(faces))):
        perm = rng.permutation(len(faces))
    def moved(i):                                               # 別の顔の枠組みを、この顔の枠の中の同じ相対位置に置く
        fr = dict(frames[perm[i]])
        if fr["n"]:
            g = faces[perm[i]]; x0, y0, x1, y1 = faces[i]["box"]; uvE = H.uv(g, np.asarray(fr["E"])); k = faces[i]["s"] / g["s"]
            fr.update(E=np.array([x0 + uvE[0] * (x1 - x0), y0 + uvE[1] * (y1 - y0)]), d=fr["d"] * k, w=fr["w"] * k)
        return fr
    FN = [feats_D(faces[i], moved(i)) for i in range(len(faces))]

    pB = [p.argmax(1) for p in H.cv_probs(faces, fold, lambda i, k: FB[i], dev, seed0=10)]
    h7 = json.load(open(G.ROOT / "results/h7_stroke_parts_20261012/summary.json"))["variants"]["B"]["stroke"]
    ok1 = H.stroke_pr(faces, pB) == h7
    pD = [p.argmax(1) for p in H.cv_probs(faces, fold, lambda i, k: FD[i], dev, seed0=10)]
    pN = [p.argmax(1) for p in H.cv_probs(faces, fold, lambda i, k: FN[i], dev, seed0=10)]
    eB, eD, eN = others_eval(faces, pB), others_eval(faces, pD), others_eval(faces, pN)
    key = "brow+nose+mouth+ear"
    dneg = eN[key]["group_recovered"] - eB[key]["group_recovered"]
    two = [fr["n"] == 2 for fr in frames]
    chk = dict(n_faces=len(faces), eye_groups={str(k): int(sum(fr["n"] == k for fr in frames)) for k in (0, 1, 2)},
               B_matches_H7=bool(ok1), negative_minus_B=round(dneg, 3))
    chk["passed"] = bool(ok1 and dneg <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    S = dict(part1_B_by_face_type={"two_or_more_eyes": dict(n_faces=int(sum(two)), **others_eval(faces, pB, two)),
                                   "one_or_no_eye": dict(n_faces=int(len(two) - sum(two)), **others_eval(faces, pB, [not t for t in two]))},
             part2=dict(B=eB, D_true_eye_frame=eD, D_swapped_eye_frame=eN),
             part2_D_by_face_type={"two_or_more_eyes": others_eval(faces, pD, two), "one_or_no_eye": others_eval(faces, pD, [not t for t in two])},
             decision_delta=round(eD[key]["group_recovered"] - eB[key]["group_recovered"], 3))
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    def line(e):
        return "  ".join(f"{k}: 線 {v['stroke_precision']:.2f}/{v['stroke_recall']:.2f} 復元 {v['group_recovered']:.2f}({v['n_groups']})"
                         f" 1-2本 {'-' if v['group_recovered_size12'] is None else format(v['group_recovered_size12'], '.2f')}({v['n_groups_size12']})"
                         for k, v in e.items() if k != "n_faces")
    print("\n部分 1: B を顔の種類で分ける")
    for k, v in S["part1_B_by_face_type"].items():
        print(f" {k} ({v['n_faces']} 顔)\n   " + line(v))
    print("\n部分 2")
    for k, v in S["part2"].items():
        print(f" {k}\n   " + line(v))
    print("\nD を顔の種類で分ける")
    for k, v in S["part2_D_by_face_type"].items():
        print(f" {k}\n   " + line(v))
    print("\n判定に使う差(D − B、眉・鼻・口・耳のグループ復元率):", S["decision_delta"]); print("->", out)


if __name__ == "__main__":
    main()
