#!/usr/bin/env python
"""実験 H7(2026-10-12 事前登録): 線 1 本ごとにパーツを当て、近さでつないでグループにする。学習用だけ。

手がかり A 線そのもの / B + 顔の枠の中の位置 / C + 目に対する位置と周り。シリーズ群で 5 分割。
出力: summary.json、check.json、strokes.csv
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402

MAX_ORDER = 104
CLASSES = ["eye", "brow", "nose", "mouth", "ear", "other_face", "none"]
PARTS = CLASSES[:-1]
TAUS = (0.02, 0.04, 0.08)
TH = 0.8
NFOLD, NSEED = 5, 5


def cls_of(label):
    if label == "glasses":                                       # 2026-10-12 に足した目印。H7・H8 の時点にはなく、パーツには数えない
        return "none"
    return "eye" if label.startswith("face.eye") else {"face.other": "other_face"}.get(label, label.replace("face.", ""))


def load_faces(max_order=MAX_ORDER, n_lines=None):
    """返る: 顔ごとの dict(order, group, pts (n,16,2), ids, y (クラス番号), gid (正解グループ番号, -1 = なし), s, box, cl, csize, pose)。
    n_lines: 保存ファイルの最初の何行までを使うか(追記式なので、ある時点の内容を再現できる)"""
    G.load_state(G.LAB / "face_groups.jsonl")
    if n_lines is not None:
        G.S["faces"] = {}
        for k, ln in enumerate(open(G.LAB / "face_groups.jsonl")):
            if k < n_lines and ln.strip():
                d = json.loads(ln); G.S["faces"][int(d["order"])] = d
    faces = []
    for o, r in sorted(G.S["faces"].items()):
        if o > max_order or r["usage"] != "train" or r["status"] != "done":      # 検証用は使わない
            continue
        q = G.S["queue"][o]
        pl = G.face_payload(o)
        ids = np.array([s["id"] for s in pl["strokes"]])
        pts = np.array([s["p"] for s in pl["strokes"]], float).reshape(-1, 16, 2)
        wid = np.array([s["w"] for s in pl["strokes"]], float)
        pos = {i: k for k, i in enumerate(ids)}
        y = np.full(len(ids), CLASSES.index("none")); gid = np.full(len(ids), -1)
        for gi, g in enumerate(r["groups"]):
            for i in g["strokes"]:
                y[pos[i]] = CLASSES.index(cls_of(g["label"])); gid[pos[i]] = gi
        x0, y0, x1, y1 = pl["box"]
        _p, _w, cl = G.panel_strokes(pl["panel"]); cl = np.asarray(cl).astype(int)
        faces.append(dict(order=o, group=q["group"], pts=pts, wid=wid, ids=ids, y=y, gid=gid, box=(x0, y0, x1, y1),
                          s=float(np.sqrt((x1 - x0) * (y1 - y0))), cl=cl[ids],
                          csize=np.bincount(cl[cl >= 0], minlength=cl.max() + 2), pose=r.get("pose"), panel=pl["panel"]))
    return faces


def feats_A(f):
    p, s = f["pts"], f["s"]
    d = np.diff(p, axis=1); L = np.hypot(d[..., 0], d[..., 1])
    arc = L.sum(1); chord = np.hypot(*(p[:, -1] - p[:, 0]).T)
    phi = np.arctan2(d[..., 1], d[..., 0])
    z = (L * np.exp(2j * phi)).sum(1) / np.maximum(arc, 1e-6)
    turn = np.abs(np.angle(np.exp(1j * np.diff(phi, axis=1)))).sum(1)
    bw = np.ptp(p[..., 0], axis=1); bh = np.ptp(p[..., 1], axis=1)
    return np.column_stack([np.log(arc / s + 1e-3), chord / s, chord / np.maximum(arc, 1e-6), z.real, z.imag, turn,
                            np.log(f["wid"] / s + 1e-4), bw / s, bh / s])


def uv(f, xy):
    x0, y0, x1, y1 = f["box"]
    return np.stack([(xy[..., 0] - x0) / (x1 - x0), (xy[..., 1] - y0) / (y1 - y0)], -1)


def feats_B(f):
    p = f["pts"]
    mid = uv(f, p.mean(1)); a = uv(f, p[:, 0]); b = uv(f, p[:, -1])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.column_stack([feats_A(f), mid, lo, hi])


def feats_C(f, p_eye):
    p, s = f["pts"], f["s"]
    mid = p.mean(1)
    arc = np.hypot(*np.diff(p, axis=1).transpose(2, 0, 1)).sum(1)
    w = np.where(p_eye > 0.5, p_eye, 0) * arc
    dist = np.linalg.norm(mid[:, None] - mid[None], axis=-1)
    near = ((dist < 0.08 * s).sum(1) - 1) / 10.0
    if w.sum() <= 0:
        rel = np.zeros((len(p), 6))
    else:
        E = (w[:, None] * mid).sum(0) / w.sum()
        dx = np.sqrt((w * (mid[:, 0] - E[0]) ** 2).sum() / w.sum()) + 0.02 * s
        r = mid - E
        rel = np.column_stack([r / s, r / dx, np.full(len(p), dx / s), np.ones(len(p))])
    return np.column_stack([feats_B(f), rel, near, np.full(len(p), len(p) / 100.0)])


def fit_predict(Xtr, ytr, Xte, dev, seed0=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    out = 0
    for k in range(NSEED):
        torch.manual_seed(20261012 + seed0 * 100 + k)
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, len(CLASSES))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        for _ in range(400):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(Xtr_t), ytr_t).backward(); opt.step()
        with torch.no_grad():
            out = out + torch.softmax(net(Xte_t), 1).cpu().numpy()
    return out / NSEED


def folds_of(faces, n=NFOLD):
    cnt = Counter(f["group"] for f in faces)
    order = sorted(cnt, key=lambda g: (-cnt[g], g))
    load = [0] * n; g2f = {}
    for g in order:                                             # 大きい群から、いま一番少ない分割へ
        k = int(np.argmin(load)); g2f[g] = k; load[k] += cnt[g]
    return np.array([g2f[f["group"]] for f in faces]), g2f


def cv_probs(faces, fold, feat, dev, labels=None, seed0=0):
    """返る: 顔ごとの確率 (n, 7)。feat(f, i) = 顔 i の特徴"""
    probs = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = [i for i in range(len(faces)) if fold[i] != k]; te = [i for i in range(len(faces)) if fold[i] == k]
        Xtr = np.vstack([feat(i, k) for i in tr]); ytr = np.concatenate([(labels or {}).get(i, faces[i]["y"]) for i in tr])
        P = fit_predict(Xtr, ytr, np.vstack([feat(i, k) for i in te]), dev, seed0 + k)
        s = 0
        for i in te:
            n = len(faces[i]["y"]); probs[i] = P[s:s + n]; s += n
    return probs


def components(f, pred, tau):
    """同じ予測パーツの線を、点どうしの最短距離 < tau·s でつなぐ。返る: [(クラス番号, 線の位置の集合)]"""
    out = []
    for c in range(len(PARTS)):
        idx = np.flatnonzero(pred == c)
        if not len(idx):
            continue
        P = f["pts"][idx]
        D = np.linalg.norm(P[:, None, :, None] - P[None, :, None, :], axis=-1).min((2, 3))
        par = list(range(len(idx)))
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for a, b in zip(*np.nonzero(np.triu(D < tau * f["s"], 1))):
            par[find(a)] = find(b)
        comp = defaultdict(set)
        for a in range(len(idx)):
            comp[find(a)].add(int(idx[a]))
        out += [(c, v) for v in comp.values()]
    return out


def group_eval(faces, preds, tau):
    """返る: 正解グループごとの (クラス, 本数, 復元されたか)、予測グループ数、対応が付いた予測グループ数"""
    rows, n_pred, n_hit = [], 0, 0
    for f, pred in zip(faces, preds):
        comps = components(f, pred, tau)
        truth = [(int(f["y"][f["gid"] == g][0]), set(np.flatnonzero(f["gid"] == g).tolist())) for g in sorted(set(f["gid"][f["gid"] >= 0].tolist()))]
        hit = set()
        for c, Gs in truth:
            ok = False
            for j, (pc, Ps) in enumerate(comps):
                if pc == c and len(Gs & Ps) >= TH * len(Gs) and len(Gs & Ps) >= TH * len(Ps):
                    ok = True; hit.add(j); break
            rows.append((c, len(Gs), ok))
        n_pred += len(comps); n_hit += len(hit)
    return rows, n_pred, n_hit


def baseline(faces):
    """元のまとまり: 大きさと目印を問わず、まとまり具合 ≥ 0.8 かつ 純度 ≥ 0.8(純度はコマ全体のまとまりの大きさで)"""
    rows = []
    for f in faces:
        for g in sorted(set(f["gid"][f["gid"] >= 0].tolist())):
            m = f["gid"] == g; c = f["cl"][m]; cnt = Counter(c[c >= 0].tolist()); ok = False
            if cnt:
                cs, k = max(cnt.items(), key=lambda t: t[1])
                ok = k >= TH * m.sum() and k >= TH * f["csize"][cs]
            rows.append((int(f["y"][m][0]), int(m.sum()), ok))
    return rows


def rate(rows, cond=lambda c, n: True):
    v = [ok for c, n, ok in rows if cond(c, n)]
    return (round(float(np.mean(v)), 3) if v else None), len(v)


def stroke_pr(faces, preds):
    y = np.concatenate([f["y"] for f in faces]); p = np.concatenate(preds)
    out = {}
    for c, name in enumerate(PARTS):
        tp = int(((p == c) & (y == c)).sum())
        out[name] = dict(n=int((y == c).sum()), n_pred=int((p == c).sum()), precision=round(tp / max(int((p == c).sum()), 1), 3),
                         recall=round(tp / max(int((y == c).sum()), 1), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h7_stroke_parts_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    faces = load_faces()
    fold, g2f = folds_of(faces)
    FA = [feats_A(f) for f in faces]; FB = [feats_B(f) for f in faces]

    # 道具確認 1: 分割
    ok1 = all(len({fold[i] for i in range(len(faces)) if faces[i]["group"] == g}) == 1 for g in g2f)
    # 道具確認 2: 正解を予測として入れる
    truth_pred = [f["y"].copy() for f in faces]
    pr_t = stroke_pr(faces, truth_pred)
    ok2 = all(v["precision"] == 1 and v["recall"] == 1 for v in pr_t.values() if v["n"])
    ceiling = {str(t): rate(group_eval(faces, truth_pred, t)[0])[0] for t in TAUS}
    # 道具確認 3: 学ぶ側の目印を顔の中でかき混ぜる
    rng = np.random.default_rng(20261012)
    shuf = {i: rng.permutation(f["y"]) for i, f in enumerate(faces)}
    pn = [p.argmax(1) for p in cv_probs(faces, fold, lambda i, k: FB[i], dev, labels=shuf, seed0=50)]
    neg = rate(group_eval(faces, pn, 0.04)[0])[0]
    chk = dict(n_faces=len(faces), n_strokes=int(sum(len(f["y"]) for f in faces)), fold_sizes=Counter(fold.tolist()),
               class_counts={c: int(sum((f["y"] == k).sum() for f in faces)) for k, c in enumerate(CLASSES)},
               folds_disjoint=bool(ok1), truth_pr_is_1=bool(ok2), grouping_ceiling_by_tau=ceiling, negative_recovered=neg)
    chk["passed"] = bool(ok1 and ok2 and neg <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False, default=lambda o: dict(o))
    print(json.dumps(chk, ensure_ascii=False, default=lambda o: dict(o))); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    probs = {"A": cv_probs(faces, fold, lambda i, k: FA[i], dev), "B": cv_probs(faces, fold, lambda i, k: FB[i], dev, seed0=10)}
    # C: 目の確率は、当てる側 = 外側の分割の B の予測、学ぶ側 = 学ぶ側の中だけでもう一度分割して作った B の予測
    eye = CLASSES.index("eye")
    inner = {}
    for k in sorted(set(fold.tolist())):
        tr = [i for i in range(len(faces)) if fold[i] != k]
        sub = [faces[i] for i in tr]; subfold, _ = folds_of(sub, 4)
        pp = cv_probs(sub, subfold, lambda j, kk, tr=tr: FB[tr[j]], dev, seed0=20 + k)
        inner[k] = {i: pp[j][:, eye] for j, i in enumerate(tr)}
    def featC(i, k):
        return feats_C(faces[i], probs["B"][i][:, eye] if fold[i] == k else inner[k][i])
    probs["C"] = cv_probs(faces, fold, featC, dev, seed0=30)

    S = dict(baseline_trackF={}, variants={})
    base = baseline(faces)
    def table(rows):
        t = {"all": rate(rows), "size 1-2": rate(rows, lambda c, n: n <= 2), "size 3+": rate(rows, lambda c, n: n >= 3)}
        for c, name in enumerate(PARTS):
            t[name] = rate(rows, lambda cc, n, c=c: cc == c)
            t[name + " size 1-2"] = rate(rows, lambda cc, n, c=c: cc == c and n <= 2)
        return {k: dict(recovered=v[0], n=v[1]) for k, v in t.items()}
    S["baseline_trackF"] = table(base)
    S["grouping_ceiling_truth_labels"] = {str(t): table(group_eval(faces, truth_pred, t)[0]) for t in TAUS}
    for name, pr in probs.items():
        preds = [p.argmax(1) for p in pr]
        v = dict(stroke=stroke_pr(faces, preds), group={})
        for t in TAUS:
            rows, n_pred, n_hit = group_eval(faces, preds, t)
            v["group"][str(t)] = dict(table=table(rows), n_pred_groups=n_pred, pred_group_precision=round(n_hit / max(n_pred, 1), 3))
        S["variants"][name] = v
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "strokes.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["order", "stroke", "fold", "truth", "gid", "pred_A", "pred_B", "pred_C"])
        for i, f in enumerate(faces):
            for j in range(len(f["y"])):
                wr.writerow([f["order"], int(f["ids"][j]), int(fold[i]), CLASSES[f["y"][j]], int(f["gid"][j])] +
                            [CLASSES[probs[n][i][j].argmax()] for n in "ABC"])

    print("\n線単位(適合率 / 再現率)")
    for name in "ABC":
        print(name, " ".join(f"{p}:{v['precision']:.2f}/{v['recall']:.2f}" for p, v in S["variants"][name]["stroke"].items()))
    print("\nグループの復元率(τ = 0.04)")
    keys = ["all", "size 1-2", "size 3+"] + PARTS + ["nose size 1-2", "mouth size 1-2", "brow size 1-2"]
    def line(t):
        return " ".join(f"{k}:{'-' if t[k]['recovered'] is None else format(t[k]['recovered'], '.2f')}({t[k]['n']})" for k in keys)
    print("元のまとまり ", line(S["baseline_trackF"]))
    for name in "ABC":
        g = S["variants"][name]["group"]["0.04"]
        print(name, "          ", line(g["table"]), f" 予測グループ {g['n_pred_groups']} 個のうち対応あり {g['pred_group_precision']:.2f}")
    print("正解の目印で  ", line(S["grouping_ceiling_truth_labels"]["0.04"]))
    for t in TAUS:
        print(f"τ={t}: " + " ".join(f"{n}:{S['variants'][n]['group'][str(t)]['table']['all']['recovered']:.2f}"
                                    f"(適合 {S['variants'][n]['group'][str(t)]['pred_group_precision']:.2f})" for n in "ABC")
              + f" 正解の目印で:{S['grouping_ceiling_truth_labels'][str(t)]['all']['recovered']:.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
