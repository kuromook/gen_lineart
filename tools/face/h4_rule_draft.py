#!/usr/bin/env python
"""実験 H4(2026-10-04 事前登録): 学習用の目印だけで規則の下書きを作る。検証用の目印は読み込まない。

手がかりの比較(学習用の中の 1 個抜き): A 位置+大きさの近傍 5 / B 語だけ / C 位置+大きさ+語。
出力: loo.csv、positions.png、words.csv、check.json
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_eye_sizes as C                                           # noqa: E402

LABELS_CSV = C.ROOT.parent / "labels/strategic_labels.csv"
QUEUE_CSV = C.ROOT.parent / "labels/queue_v1.csv"
TARGETS = ["face.eye.open", "face.brow", "face.mouth", "face.nose", "face.ear"]
SHOW = TARGETS + ["face.eye.closed", "face.other", "not_face"]
K = 5


def load_train():
    q = {int(t["order"]): t for t in csv.DictReader(open(QUEUE_CSV))}
    rows = []
    for t in csv.DictReader(open(LABELS_CSV)):
        if t["usage"] != "train":                                  # 検証用は読み込まない
            continue
        labs = set(t["labels"].split(";"))
        if "unsure" in labs:
            continue
        qq = q[int(t["order"])]
        assert qq["usage"] == "train" and qq["row"] == t["row"]
        rows.append(dict(row=int(t["row"]), u=float(qq["u"]), v=float(qq["v"]), r=float(qq["r"]), labs=labs))
    return rows


def knn_votes(X, y, groups=None):
    """1 個抜きの近傍 K 個の正例数。groups があれば同じ group の中だけで探す(K 個に満たなければ全体)"""
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    votes = np.zeros(len(X), int)
    for i in range(len(X)):
        d = D[i]
        if groups is not None:
            same = np.flatnonzero((groups == groups[i]) & np.isfinite(d))
            if len(same) >= K:
                votes[i] = y[same[np.argsort(d[same])[:K]]].sum(); continue
        votes[i] = y[np.argsort(d)[:K]].sum()
    return votes


def pr(pred, y):
    tp = int((pred & y).sum())
    return (tp / max(int(pred.sum()), 1) if pred.any() else float("nan"), tp / max(int(y.sum()), 1), int(pred.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h4_rule_draft_20261004")
    a = ap.parse_args()
    out = C.ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = load_train()
    words_all, _meta, _P, _M = C.encode_all(torch.device("cuda"))
    w = np.array([int(words_all[r["row"]]) for r in rows])
    U = np.array([r["u"] for r in rows]); V = np.array([r["v"] for r in rows]); R = np.array([r["r"] for r in rows])
    F = np.stack([U, V, np.log(R)], 1)
    X = F / F.std(0)
    Y = {lb: np.array([lb in r["labs"] for r in rows]) for lb in SHOW}
    rng = np.random.default_rng(20261004)

    chk = dict(n_train=len(rows), counts={lb: int(Y[lb].sum()) for lb in SHOW}, n_words=int(len(set(w.tolist()))))
    table, neg = [], {}
    for lb in TARGETS:
        y = Y[lb]
        base = float(y.mean())
        pa = knn_votes(X, y) >= 3
        pb = np.array([(lambda o: len(o) > 0 and y[o].mean() > 0.5)(np.flatnonzero((w == w[i]) & (np.arange(len(w)) != i)))
                       for i in range(len(w))])
        pc = knn_votes(X, y, groups=w) >= 3
        sh = []
        for _ in range(20):
            ys = y[rng.permutation(len(y))]
            p, _r, _n = pr(knn_votes(X, ys) >= 3, ys)
            sh.append(base if np.isnan(p) else p)
        neg[lb] = dict(base_rate=round(base, 3), shuffled_precision=round(float(np.mean(sh)), 3),
                       ok=bool(abs(np.mean(sh) - base) <= 0.10))
        for name, pred in (("A position+size", pa), ("B word only", pb), ("C position+size+word", pc)):
            p, r_, n = pr(pred, y)
            table.append(dict(label=lb, n_pos=int(y.sum()), base_rate=round(base, 3), cue=name,
                              precision=round(p, 3), recall=round(r_, 3), n_pred=n,
                              note="参考値(正例 < 20)" if y.sum() < 20 else ""))
    chk["negative_control"] = neg
    chk["passed"] = bool(all(v["ok"] for v in neg.values()))
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False), flush=True)
    print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    with open(out / "loo.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    for t in table:
        print(f"{t['label']:16s} n={t['n_pos']:3d} base={t['base_rate']:.2f}  {t['cue']:22s} "
              f"P={t['precision']:.2f} R={t['recall']:.2f} (pred {t['n_pred']}) {t['note']}", flush=True)

    wt = []
    for lb in SHOW:
        c = Counter(w[Y[lb]].tolist())
        wt.append(dict(label=lb, n=int(Y[lb].sum()), n_words=len(c),
                       top_words=" ".join(f"w{k}:{v}" for k, v in c.most_common(6)),
                       r_median=round(float(np.median(R[Y[lb]])), 3) if Y[lb].any() else "",
                       r_p10_p90=" ".join(f"{x:.2f}" for x in np.percentile(R[Y[lb]], [10, 90])) if Y[lb].any() else ""))
    with open(out / "words.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(wt[0])); wr.writeheader(); wr.writerows(wt)
    for t in wt:
        print(t, flush=True)

    # 図: 目印ごとの小さな図(灰 = 学習用の全部、濃色 = その目印)。色は 1 色、識別は図の題で行う
    S, PAD = 300, 14
    tiles = []
    for lb in SHOW:
        im = np.full((S, S, 3), 255, np.uint8)
        for t in (1, 2):
            cv2.line(im, (S * t // 3, 0), (S * t // 3, S), (232, 232, 232), 1)
            cv2.line(im, (0, S * t // 3), (S, S * t // 3), (232, 232, 232), 1)
        cv2.rectangle(im, (0, 0), (S - 1, S - 1), (150, 150, 150), 1)
        for i in range(len(rows)):
            cv2.circle(im, (int(U[i] * S), int(V[i] * S)), 3, (205, 205, 205), -1, cv2.LINE_AA)
        for i in np.flatnonzero(Y[lb]):
            rad = int(np.clip(4 + R[i] * 14, 4, 11))
            cv2.circle(im, (int(U[i] * S), int(V[i] * S)), rad + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(im, (int(U[i] * S), int(V[i] * S)), rad, (140, 80, 30), -1, cv2.LINE_AA)
        head = np.full((26, S, 3), 255, np.uint8)
        cv2.putText(head, f"{lb}  n={int(Y[lb].sum())}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(np.vstack([head, im]), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    g = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
    title = np.full((34, g.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"train labels only (n={len(rows)}): position inside the face box. gray = all train, "
                       "blue = this label (dot size = relative size r)", (PAD, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "positions.png"), np.vstack([title, g]))
    print("->", out / "positions.png", flush=True)


if __name__ == "__main__":
    main()
