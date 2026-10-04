#!/usr/bin/env python
"""実験 H5(2026-10-04 事前登録): 線の向きを手がかりに足す。学習用の目印だけ。

向き: 線分の向き φ を長さで重みづけ、2φ で平均。θ = 主方向(0 = 水平、90 = 垂直)、c = 揃い具合(0〜1)。
比較: A 位置+大きさ(H4 の再掲) / D 位置+大きさ+向き。1 個抜き・近傍 5 個の多数決。
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_eye_sizes as C                                           # noqa: E402
import h4_rule_draft as H4                                             # noqa: E402


def orientation(lines):
    """lines: [(n, 2) の (x, y)]。返る: θ(度、-90〜90。画像の y は下向きなので右下がりが正)、c"""
    s = 0j; tot = 0.0
    for ln in lines:
        d = np.diff(np.asarray(ln, float), axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        phi = np.arctan2(d[:, 1], d[:, 0])
        s += (L * np.exp(2j * phi)).sum(); tot += L.sum()
    if tot == 0:
        return 0.0, 0.0
    return float(np.degrees(0.5 * np.angle(s))), float(abs(s) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h5_orientation_20261004")
    a = ap.parse_args()
    out = C.ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, 100, 17)[:, None]
    synth = dict(horizontal=orientation([np.hstack([t, 0 * t])]), vertical=orientation([np.hstack([0 * t, t])]),
                 diag45=orientation([np.hstack([t, t])]),
                 cross=orientation([np.hstack([t, 0 * t]), np.hstack([0 * t, t])]))
    ok1 = (abs(synth["horizontal"][0]) < 1 and abs(abs(synth["vertical"][0]) - 90) < 1 and abs(abs(synth["diag45"][0]) - 45) < 1
           and min(synth[k][1] for k in ("horizontal", "vertical", "diag45")) > 0.99 and synth["cross"][1] < 0.01)

    rows = H4.load_train()
    meta = list(csv.DictReader(open(C.CLUSTERS / "meta.csv")))
    P = np.load(C.CLUSTERS / "pts.npy", mmap_mode="r"); M = np.load(C.CLUSTERS / "mask.npy", mmap_mode="r")
    oc = np.array([orientation(C.inst_points(P, M, meta, r["row"])) for r in rows])
    TH, CO = oc[:, 0], oc[:, 1]
    U = np.array([r["u"] for r in rows]); V = np.array([r["v"] for r in rows]); R = np.array([r["r"] for r in rows])
    FA = np.stack([U, V, np.log(R)], 1)
    FD = np.column_stack([FA, CO * np.cos(2 * np.radians(TH)), CO * np.sin(2 * np.radians(TH))])
    XA, XD = FA / FA.std(0), FD / FD.std(0)
    Y = {lb: np.array([lb in r["labs"] for r in rows]) for lb in H4.SHOW}
    rng = np.random.default_rng(20261004)

    h4 = {(t_["label"], t_["cue"]): t_ for t_ in csv.DictReader(open(C.ROOT.parent / "results/h4_rule_draft_20261004/loo.csv"))}
    table, neg, ok2 = [], {}, True
    for lb in H4.TARGETS:
        y = Y[lb]; base = float(y.mean())
        pa, ra, na = H4.pr(H4.knn_votes(XA, y) >= 3, y)
        pd_, rd, nd = H4.pr(H4.knn_votes(XD, y) >= 3, y)
        ref = h4[(lb, "A position+size")]
        ok2 &= round(pa, 3) == float(ref["precision"]) and round(ra, 3) == float(ref["recall"])
        sh = []
        for _ in range(20):
            ys = y[rng.permutation(len(y))]
            p, _r, _n = H4.pr(H4.knn_votes(XD, ys) >= 3, ys)
            sh.append(base if np.isnan(p) else p)
        neg[lb] = dict(base_rate=round(base, 3), shuffled_precision=round(float(np.mean(sh)), 3),
                       ok=bool(abs(np.mean(sh) - base) <= 0.10))
        for name, (p, r_, n) in (("A position+size", (pa, ra, na)), ("D position+size+orientation", (pd_, rd, nd))):
            table.append(dict(label=lb, n_pos=int(y.sum()), base_rate=round(base, 3), cue=name, precision=round(p, 3),
                              recall=round(r_, 3), n_pred=n, note="参考値(正例 < 20)" if y.sum() < 20 else ""))
    chk = dict(synthetic={k: [round(v[0], 1), round(v[1], 3)] for k, v in synth.items()}, synthetic_ok=bool(ok1),
               h4_match=bool(ok2), negative_control=neg, n_train=len(rows))
    chk["passed"] = bool(ok1 and ok2 and all(v["ok"] for v in neg.values()))
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False), flush=True)
    print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    with open(out / "loo.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    for t_ in table:
        print(f"{t_['label']:16s} n={t_['n_pos']:3d}  {t_['cue']:28s} P={t_['precision']:.2f} R={t_['recall']:.2f} "
              f"(pred {t_['n_pred']}) {t_['note']}", flush=True)

    # 眉と前髪(枠の上側)の主方向
    top = V < 0.45
    brow = top & Y["face.brow"]
    hair = top & Y["not_face"] & ~np.any([Y[lb] for lb in H4.SHOW if lb != "not_face"], axis=0)
    dist = {}
    for name, sel in (("brow (v<0.45)", brow), ("not_face only (v<0.45)", hair)):
        ab = np.abs(TH[sel])
        dist[name] = dict(n=int(sel.sum()), abs_theta_median=round(float(np.median(ab)), 1),
                          near_horizontal_lt30=int((ab < 30).sum()), middle_30_60=int(((ab >= 30) & (ab < 60)).sum()),
                          near_vertical_ge60=int((ab >= 60).sum()), coherence_median=round(float(np.median(CO[sel])), 2))
        print(name, dist[name], flush=True)
    json.dump(dist, open(out / "brow_vs_hair.json", "w"), indent=1, ensure_ascii=False)

    # 図: 位置に主方向を短い線で描く(目印ごとの小さな図)
    S, PAD = 300, 14
    tiles = []
    for lb in H4.SHOW:
        im = np.full((S, S, 3), 255, np.uint8)
        for k in (1, 2):
            cv2.line(im, (S * k // 3, 0), (S * k // 3, S), (232, 232, 232), 1)
            cv2.line(im, (0, S * k // 3), (S, S * k // 3), (232, 232, 232), 1)
        cv2.rectangle(im, (0, 0), (S - 1, S - 1), (150, 150, 150), 1)
        for i in np.flatnonzero(Y[lb]):
            h = 5 + 9 * CO[i]
            dx, dy = h * np.cos(np.radians(TH[i])), h * np.sin(np.radians(TH[i]))
            p0 = (int(round((U[i] * S - dx) * 4)), int(round((V[i] * S - dy) * 4)))
            p1 = (int(round((U[i] * S + dx) * 4)), int(round((V[i] * S + dy) * 4)))
            cv2.line(im, p0, p1, (140, 80, 30), 2, cv2.LINE_AA, 2)
        head = np.full((26, S, 3), 255, np.uint8)
        cv2.putText(head, f"{lb}  n={int(Y[lb].sum())}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(np.vstack([head, im]), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    g = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
    title = np.full((34, g.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"train labels only (n={len(rows)}): each tick = one cluster at its position in the face box, "
                       "drawn along its main line direction (longer = more aligned)", (PAD, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "orientation.png"), np.vstack([title, g]))
    print("->", out / "orientation.png", flush=True)


if __name__ == "__main__":
    main()
