#!/usr/bin/env python
"""実インスタンス描画 (2026-09-23 事前登録の描画第2案)。

デコード原型のかわりに実クラスタのストローク(cluster_set の pts/width)を配置する。
B/C/D は語の出現ごとに実インスタンスをプールからサンプリング。

道具確認(実行順):
1. cluster_set からの正準順再構成が corpus トークンと一致するか(全5,345コマ)
2. 実インスタンスを元の位置に置いた描画(A列)が元のコマ線画と agreement/F1 で一致するか
   (→ cluster_set のパネルストロークカバー率の確認でもある)
合否: ユーザー目視で「判断できる」か。定量は併記のみ。
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TRACKF, CODEBOOK, CORPUS, N_WORDS, N_BINS, S_BINS, S_LO, S_HI,  # noqa: E402
                    load_all)
from train_codebook import load                                        # noqa: E402
from train_codebook2 import Codebook2                                  # noqa: E402
from train_set import SetAR                                            # noqa: E402
from train_2stage import WordAR, PosAR                                 # noqa: E402
from render_gen import sample_set, sample_words, sample_pos            # noqa: E402
from render_strokes import agreement                                   # noqa: E402

CLUSTERS = TRACKF / "results/cluster_set_20260919"
PACK = TRACKF / "results/panel_pack_20260919"
H_CANVAS = 220


def build_clusters(dev):
    """cluster_set を読み、語IDを付け、語プールとパネル別正準順リストを作る"""
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    cm = Codebook2(levels=tuple(cb["levels"]), coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    Ps, Ws, Ms, ws, rows = [], [], [], [], []
    with torch.no_grad():
        for split in ("train", "test"):
            P, W, M, rr = load(CLUSTERS, split, dev)
            wd = []
            for i in range(0, len(P), 4096):
                _q, w = cm.encode(P[i:i + 4096].float(), W[i:i + 4096].float(), M[i:i + 4096], 1)
                wd.append(w.cpu())
            Ps.append(P.cpu().numpy()); Ws.append(W.cpu().numpy())
            Ms.append(M.cpu().numpy()); ws.append(torch.cat(wd).numpy()); rows += list(rr)
    P = np.concatenate(Ps); W = np.concatenate(Ws); M = np.concatenate(Ms)
    words = np.concatenate(ws)
    scale = np.array([float(r["scale"]) for r in rows])
    cy = np.array([float(r["cy"]) for r in rows]); cx = np.array([float(r["cx"]) for r in rows])
    panel = np.array([int(r["panel"]) for r in rows])
    print(f"clusters {len(P)} | pts frame |max| = {np.abs(P[M]).max():.1f} (decode 原型は ±28)", flush=True)

    pools = defaultdict(list)
    for i, w in enumerate(words):
        pools[int(w)].append(i)
    by_panel = defaultdict(list)
    for i in range(len(P)):
        by_panel[int(panel[i])].append(i)
    panel_order = {}
    for pid, idxs in by_panel.items():
        idxs.sort(key=lambda i: (-(1.0 / scale[i]), cy[i], cx[i]))
        panel_order[pid] = idxs
    return P, W, M, words, scale, cy, cx, panel_order, pools


def bins_of(i, scale, cy, cx, H, Wd):
    ss = min(max(int((np.log2(scale[i]) - S_LO) / (S_HI - S_LO) * S_BINS), 0), S_BINS - 1)
    sy = min(int(cy[i] / max(H, 1) * N_BINS), N_BINS - 1)
    sx = min(int(cx[i] / max(Wd, 1) * N_BINS), N_BINS - 1)
    return ss, sy, sx


def verify_tokens(panel_order, words, scale, cy, cx, dims):
    """cluster_set 再構成の語列+ビンが corpus トークンと一致するか(全コマ)"""
    z = np.load(CORPUS)
    seq, pids = z["seq"], z["panels"]
    bad = 0
    for r in range(len(seq)):
        t = seq[r][seq[r] >= 0]
        cl = t[1:-1].reshape(-1, 4)
        H, Wd = dims[int(pids[r])]
        mine = panel_order[int(pids[r])]
        n = min(len(mine), len(cl))
        ok = all(words[mine[k]] == cl[k, 0]
                 and (bins_of(mine[k], scale, cy, cx, H, Wd) == (cl[k, 3] - 544, cl[k, 1] - 512, cl[k, 2] - 528))
                 for k in range(n))
        if not ok or len(mine) != len(cl):
            bad += 1
    print(f"token 一致: {len(seq) - bad}/{len(seq)} コマ (不一致 {bad})", flush=True)
    return bad


def draw_clusters(canvas, P, W, M, idxs, pos, zf, th_fix=None):
    """pos: [(cy, cx, scale_native)]。実クラスタを描く。座標は (x,y)=(col,row)"""
    for ci, (pcy, pcx, psc) in zip(idxs, pos):
        pts = np.asarray(P[ci], dtype=np.float64) / psc + np.array([pcy, pcx])
        m = np.asarray(M[ci])
        th = th_fix if th_fix else max(1, int(round(float(np.asarray(W[ci])[m].mean() if m.any() else 1.0) / psc * zf)))
        for s in range(pts.shape[0]):
            if not m[s]:
                continue
            xy = np.clip(np.round(pts[s][m[s]] * zf)[:, ::-1].astype(np.int32), 0,
                         [canvas.shape[1] - 1, canvas.shape[0] - 1])
            cv2.polylines(canvas, [xy], False, (40, 40, 40), th)
    return canvas


def bin_pos(py, px, sc, H, Wd):
    scale = 2.0 ** (S_LO + (sc + 0.5) / S_BINS * (S_HI - S_LO))
    return ((py + 0.5) / 16 * H, (px + 0.5) / 16 * Wd, scale)


def check_agreement(P, W, M, scale, cy, cx, panel_order, dims, test_panels, strokes, panels_csv):
    """A 描画(実インスタンスを元位置に) vs 元のコマ線画: agreement/F1"""
    res = {}
    for pid in test_panels:
        pr = panels_csv[int(pid)]
        rows_s = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
        pp = rows_s[:, :32].reshape(-1, 16, 2)
        ink = np.abs(pp).sum(-1) > 0
        H = max(int(np.ceil(pp[..., 1][ink].max())) + 2, 2) if ink.any() else 2
        Wd = max(int(np.ceil(pp[..., 0][ink].max())) + 2, 2) if ink.any() else 2
        orig = np.zeros((H, Wd), np.uint8)
        for s in range(pp.shape[0]):
            if not ink[s].any():
                continue
            xy = np.clip(np.round(pp[s][ink[s]]).astype(np.int32), 0, [Wd - 1, H - 1])
            cv2.polylines(orig, [xy], False, 255, 1)
        mine = np.zeros((H, Wd), np.uint8)
        idxs = panel_order[int(pid)]
        draw_clusters(mine, P, W, M, idxs,
                      [(cy[i], cx[i], scale[i]) for i in idxs], 1.0, th_fix=1)
        r = agreement(mine > 0, orig > 0, tol=2)
        res[int(pid)] = dict(recall=round(r["recall"], 4), precision=round(r["precision"], 4),
                             f1=round(r["f1"], 4))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_ckpt", default="results/gen_smoke_20260921/set/gen_best.pt")
    ap.add_argument("--s1_ckpt", default="results/gen_smoke_20260921/twostage/stage1/gen_best.pt")
    ap.add_argument("--s2_ckpt", default="results/gen_smoke_20260921/twostage/stage2/gen_best.pt")
    ap.add_argument("--out", default="results/real_render_20260923")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rng = np.random.default_rng(777)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    P, W, M, words, scale, cy, cx, panel_order, pools = build_clusters(dev)
    z = np.load(CORPUS)
    dims = z["dims"]
    bad = verify_tokens(panel_order, words, scale, cy, cx, dims)
    assert bad < 20, f"token 不一致 {bad} コマ — 再構成が corpus と違う"

    panels_csv = list(csv.DictReader(open(PACK / "panels.csv")))
    strokes = np.load(PACK / "strokes.npy", mmap_mode="r")
    d = load_all(0)
    panels, lab, tg, ite = d["panels"], d["lab"], d["tg"], d["ite"]

    # 12コマ選定(型が被らない、従来どおり)
    seen, chosen = set(), []
    for i in ite:
        t = int(lab[i])
        if t in seen:
            continue
        seen.add(t); chosen.append(i)
        if len(chosen) >= a.n:
            break
    chosen_pids = [int(z["panels"][i]) for i in chosen]

    agr = check_agreement(P, W, M, scale, cy, cx, panel_order, dims, chosen_pids, strokes, panels_csv)
    f1s = [v["f1"] for v in agr.values()]
    print(f"A vs 元線画 F1: mean {np.mean(f1s):.4f} min {np.min(f1s):.4f} "
          f"(recall {np.mean([v['recall'] for v in agr.values()]):.4f} = cluster_set のカバー率)", flush=True)
    json.dump(agr, open(out / "agreement.json", "w"), indent=1)

    # --- A/B/C/D モンタージュ(実インスタンス描画) ---
    ck = torch.load(a.set_ckpt, map_location=dev, weights_only=False)
    ms = SetAR().to(dev); ms.load_state_dict(ck["model"]); ms.eval()
    ck1 = torch.load(a.s1_ckpt, map_location=dev, weights_only=False)
    m1 = WordAR().to(dev); m1.load_state_dict(ck1["model"]); m1.eval()
    ck2 = torch.load(a.s2_ckpt, map_location=dev, weights_only=False)
    m2 = PosAR().to(dev); m2.load_state_dict(ck2["model"]); m2.eval()

    def cell(pid, idxs_pos, label):
        H, Wd = int(dims[pid][0]), int(dims[pid][1])
        zf = H_CANVAS / H
        canvas = np.full((H_CANVAS + 24, max(int(Wd * zf), 2), 3), 255, np.uint8)
        idxs, pos = zip(*idxs_pos)
        draw_clusters(canvas, P, W, M, list(idxs), list(pos), zf)
        cv2.putText(canvas, label, (4, H_CANVAS + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)
        return canvas

    rows, samples = [], []
    for i, pid in zip(chosen, chosen_pids):
        w, py, px, sc = panels[i]
        wt, pyt, pxt, sct = w.tolist(), py.tolist(), px.tolist(), sc.tolist()
        ctype = torch.tensor([lab[i]], device=dev)
        ctag = torch.tensor(tg[i:i + 1], device=dev).float()
        H, Wd = int(dims[pid][0]), int(dims[pid][1])
        # A: 実インスタンスを元位置に
        idxs_a = panel_order[pid]
        cell_a = cell(pid, [(ci, (cy[ci], cx[ci], scale[ci])) for ci in idxs_a],
                      f"#{i} ty{int(lab[i])} A:truth")
        # B: 真の語+生成配置 (実インスタンス)
        pys_b, pxs_b, scs_b = sample_pos(m2, wt, ctype, ctag, dev, rng)
        ib = [rng.choice(pools[wi]) for wi in wt]
        cell_b = cell(pid, [(ci, bin_pos(py2, px2, sc2, H, Wd)) for ci, py2, px2, sc2
                            in zip(ib, pys_b, pxs_b, scs_b)], f"#{i} B:trueW+genP")
        # C: 生成語+真の位置
        ws_c = sample_words(m1, ctype, ctag, dev, rng)
        nc = min(len(ws_c), len(wt))
        ic = [(k, rng.choice(pools[wi])) for k, wi in enumerate(ws_c[:nc]) if pools[wi]]
        cell_c = cell(pid, [(ci2, bin_pos(pyt[k], pxt[k], sct[k], H, Wd)) for k, ci2 in ic],
                      f"#{i} C:genW+trueP")
        # D: 生成語+生成配置
        fs_d = sample_set(ms, ctype, ctag, dev, rng)
        id_d = [(ci3, py3, px3, sc3) for ci3, py3, px3, sc3
                in zip([rng.choice(pools[wi]) if pools[wi] else -1 for wi in fs_d[0]], *fs_d[1:])
                if ci3 >= 0]
        cell_d = cell(pid, [(ci3, bin_pos(py3, px3, sc3, H, Wd)) for ci3, py3, px3, sc3 in id_d],
                      f"#{i} D:gen")
        samples.append((int(i), dict(type=int(lab[i]), A=wt, B=wt,
                                     C=[ws_c[:nc]], D=[list(map(list, fs_d))])))
        rows.append([cell_a, cell_b, cell_c, cell_d])
    cw = max(c.shape[1] for row in rows for c in row)
    rows = [np.hstack([np.pad(c, ((0, 0), (0, cw - c.shape[1]), (0, 0)), constant_values=255)
                       for c in row]) for row in rows]
    cv2.imwrite(str(out / "montage.png"), np.vstack(rows))
    json.dump(dict(samples), open(out / "real_samples.json", "w"))
    for r, i in enumerate(chosen):
        cells = [row_img[r * (H_CANVAS + 24):(r + 1) * (H_CANVAS + 24), c * cw:(c + 1) * cw]
                 for row_img in [np.vstack(rows)] for c in range(4)]
        cv2.imwrite(str(out / f"cells_{i}_ty{int(lab[i])}.png"),
                    np.vstack([np.pad(c, ((0, 6), (0, 0), (0, 0)), constant_values=200) for c in cells]))
    print("montage ->", out / "montage.png", " (A truth | B trueW+genP | C genW+trueP | D gen; 実インスタンス描画)")


if __name__ == "__main__":
    main()
