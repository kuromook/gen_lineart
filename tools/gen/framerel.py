#!/usr/bin/env python
"""配置: 顔枠を参照枠にした絶対 vs 相対(2026-10-09 事前登録)。

bits は使わない。点で予測して外れた距離(コマ短辺比)で採点する。
参照枠は Track H の顔検出枠(混ざった語ではない参照枠)。

座標(2026-10-09 訂正済みの読み。本tree の placement_gt.py は x/y の名前が入れ替わっている
ので座標部分は再利用しない): meta の cx = x・cy = y(列名どおり)、pts は (y, x) 順。
コマ寸法は corpus dims ではなく faces.csv の png_h / png_w を使う。

--check : 道具確認 K1〜K5 のみ(合格が本測定の前提)
(既定)  : 道具確認 → 合格なら本測定 → 図
"""
import argparse, csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CLUSTERS = TRACKF / "results/cluster_set_20260919"
PACK = TRACKF / "results/panel_pack_20260919"
FACES = Path("/home/sh1/deepl/lineart-face-words/results/h3_face_parts_20261004/faces.csv")
H_CHECK = Path("/home/sh1/deepl/lineart-face-words/results/h3_face_parts_20261004/check.json")
DATASET = Path("/home/sh1/deepl/lineart/dataset")

A_SIZE = 28.0
R_EDGES = [0.1, 0.25, 0.5]
R_NAMES = ["r<0.1", "0.1-0.25", "0.25-0.5", "r>=0.5"]
EAR = 163364                      # Track H が耳として確認した meta 行(既知の値のセル)
SEED = 20261009


# ---------------------------------------------------------------- データ
def load_meta():
    """cluster_set meta.csv を行番号順に読む(Track H encode_all と同じ順)。
    返る dict: panel, work, group(作品の group ハッシュ), scale, x, y, nstk"""
    rows = list(csv.DictReader(open(CLUSTERS / "meta.csv")))
    D = dict(
        panel=np.array([int(r["panel"]) for r in rows]),
        work=np.array([r["work"] for r in rows]),
        scale=np.array([float(r["scale"]) for r in rows]),
        x=np.array([float(r["cx"]) for r in rows]),      # 列名どおり: cx = x
        y=np.array([float(r["cy"]) for r in rows]),      # 列名どおり: cy = y
        nstk=np.array([int(r["n"]) for r in rows]),
    )
    sg = json.load(open(HERE / "series_groups.json"))
    w2g = {w: g for g, ws in sg["groups"].items() for w in ws}
    half = {g: 0 for g in sg["half_A"]} | {g: 1 for g in sg["half_B"]}
    D["series"] = np.array([w2g[w] for w in D["work"]])
    D["half"] = np.array([half[s] for s in D["series"]])
    return D, rows


def load_faces():
    """faces.csv -> (枠の辞書 panel -> [(k, x0, y0, x1, y1)], コマ寸法 panel -> (png_h, png_w))"""
    faces, dims = defaultdict(list), {}
    for t in csv.DictReader(open(FACES)):
        p = int(t["panel"])
        if t["png_h"]:
            dims[p] = (int(t["png_h"]), int(t["png_w"]))
        if int(t["box"]) > 0:
            faces[p].append((int(t["box"]), float(t["x0"]), float(t["y0"]),
                             float(t["x1"]), float(t["y1"])))
    return faces, dims


def assign_boxes(D, faces):
    """まとまりの中心が入る枠(複数なら面積最小)に割り当てる。Track H h3_face_parts.py と同式。
    返る: fbox(枠番号、無割当は -1)、u, v, r, box(x0,y0,x1,y1)"""
    N = len(D["panel"])
    fbox = np.full(N, -1)
    U = np.full(N, np.nan); V = np.full(N, np.nan); R = np.full(N, np.nan)
    BX = np.full((N, 4), np.nan)
    diam = 2 * A_SIZE / D["scale"]
    by_panel = defaultdict(list)
    for i in range(N):
        by_panel[int(D["panel"][i])].append(i)
    cx, cy = D["x"], D["y"]
    for pid, fl in faces.items():
        idx = np.array(by_panel.get(pid, []), int)
        if not len(idx):
            continue
        best = np.full(len(idx), np.inf)
        for (k, x0, y0, x1, y1) in fl:
            area = (x1 - x0) * (y1 - y0)
            ins = ((cx[idx] >= x0) & (cx[idx] <= x1) & (cy[idx] >= y0) & (cy[idx] <= y1)
                   & (area < best))
            j = idx[ins]
            fbox[j] = k
            U[j] = (cx[j] - x0) / (x1 - x0); V[j] = (cy[j] - y0) / (y1 - y0)
            R[j] = diam[j] / np.sqrt(area)
            BX[j] = (x0, y0, x1, y1)
            best[ins] = area
    return fbox, U, V, R, BX


def dedupe(D):
    """同一群内の重複コマ(クラスタ位置/8 と本数の一致がコマの過半)を除く。
    placement_gt.dedupe と同じ規則(丸めキーは x/y の入れ替えに対称)。返る keep マスク, 除去コマ数"""
    key_pan = defaultdict(set)
    size = Counter(D["panel"].tolist())
    pw = {}
    for i in range(len(D["panel"])):
        p = int(D["panel"][i]); pw[p] = (D["series"][i], D["work"][i])
        key_pan[(round(D["x"][i] / 8), round(D["y"][i] / 8), int(D["nstk"][i]))].add(p)
    share = Counter()
    for ps in key_pan.values():
        if 1 < len(ps) < 30:
            ps = sorted(ps)
            for a in range(len(ps)):
                for b in range(a + 1, len(ps)):
                    share[(ps[a], ps[b])] += 1
    drop = set()
    for (a, b), c in share.items():
        if pw[a][1] != pw[b][1] and pw[a][0] == pw[b][0] and c / min(size[a], size[b]) > 0.5:
            drop.add(a if pw[a][1] > pw[b][1] else b)
    return ~np.isin(D["panel"], list(drop)), len(drop)


def rbin_of(R):
    return np.searchsorted(R_EDGES, R, side="right")


# ---------------------------------------------------------------- K1
def k1(D, fbox, U, V, R):
    """既知の値のセル: Track H check.json の枠内まとまり数・r 区分の 4 数・耳個体の (u, v, r)"""
    ref = json.load(open(H_CHECK))
    inface = fbox > 0
    rb = rbin_of(R)
    got = dict(clusters=len(D["panel"]), clusters_in_face=int(inface.sum()),
               r_bin_counts={R_NAMES[k]: int((inface & (rb == k)).sum()) for k in range(4)},
               ear_in_face=bool(inface[EAR]),
               ear_uv=[round(float(U[EAR]), 3), round(float(V[EAR]), 3)],
               ear_r=round(float(R[EAR]), 3))
    want = {k: ref[k] for k in ("clusters", "clusters_in_face", "r_bin_counts",
                                "ear_in_face", "ear_uv", "ear_r")}
    ok = got == want
    return ok, dict(got=got, want=want)


# ---------------------------------------------------------------- 語
def encode_words(dev):
    """cluster_set 全行を meta.csv の行番号順に符号化(Track H encode_all と同じ順)"""
    import torch
    sys.path.insert(0, str(HERE.parent / "trackf"))
    sys.path.insert(0, str(HERE))
    from common import CODEBOOK                                        # noqa: E402
    from train_codebook2 import Codebook2                              # noqa: E402
    P = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    W = np.load(CLUSTERS / "width.npy", mmap_mode="r")
    M = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    cm = Codebook2(levels=tuple(cb["levels"]), coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    ws = []
    with torch.no_grad():
        for i in range(0, len(P), 4096):
            p = torch.from_numpy(np.asarray(P[i:i + 4096])).to(dev).float()
            w = torch.from_numpy(np.asarray(W[i:i + 4096])).to(dev).float()
            m = torch.from_numpy(np.asarray(M[i:i + 4096])).to(dev)
            ws.append(cm.encode(p, w, m, 1)[1].cpu().numpy())
    return np.concatenate(ws)


# ---------------------------------------------------------------- 外部の基準(線画 PNG)
def panel_png(prow):
    return DATASET / prow["source"] / "line" / prow["name"]


def _inst_points(P, M, meta_rows, i, mode="correct"):
    """まとまり i の点を画像の (x, y) で返す。pts は (y, x) 順、meta の cx = x・cy = y。

    mode は陰性対照の種類:
      correct   正しい読み
      swap      できあがった点の x と y を入れ替える(事前登録の陰性対照。
                = pts を (x, y) と読み、かつ meta の cy/cx を取り違えた場合に一致する)
      shift50   50 px ずらす(感度の確認。Track H が lesson 12 で使ったもの)
      localswap pts の軸順だけ入れ替え、中心は正しいまま(どの誤りにも対応しない混成。
                2026-10-09 の K2 不合格はこれを計算していたことが原因。記録のため残す)
    """
    r = meta_rows[i]
    sc = float(r["scale"])
    m = np.asarray(M[i])
    raw = np.asarray(P[i], np.float64)
    loc = raw if mode == "localswap" else raw[..., ::-1]      # (y, x) -> (x, y)
    pts = loc / sc + np.array([float(r["cx"]), float(r["cy"])])
    pts = np.concatenate([pts[s] for s in range(len(m)) if m[s]])
    if mode == "swap":
        pts = pts[:, ::-1]
    elif mode == "shift50":
        pts = pts + 50.0
    return pts


def on_line_frac(dt, pts, tol=3.0):
    x = np.round(pts[:, 0]).astype(int); y = np.round(pts[:, 1]).astype(int)
    inb = (x >= 0) & (y >= 0) & (x < dt.shape[1]) & (y < dt.shape[0])
    if not inb.any():
        return 0.0
    return float((dt[y[inb], x[inb]] <= tol).sum() / len(pts))


# ---------------------------------------------------------------- 予測子
NMIN = 20                                 # 唯一のツマミ: 語の最小件数


def _med(a):
    return np.array([np.median(a[:, 0]), np.median(a[:, 1])])


def fit_tables(D, keep, inface, U, V, rb, word, half_fit):
    """学習側(half_fit の群)の中央値表。絶対は枠内まとまりで当てる(絶対側を最も強くする)"""
    f = keep & (D["half"] == half_fit)
    fi = f & inface
    T = {}
    # 絶対(コマ内の比)
    ax = np.stack([D["xn"], D["yn"]], 1)
    T["A1_all"] = _med(ax[f])
    T["A1"] = _med(ax[fi])
    T["A2_all"] = {}; T["A2"] = {}
    for name, msk in (("A2_all", f), ("A2", fi)):
        for w in np.unique(word[msk]):
            s = ax[msk & (word == w)]
            if len(s) >= NMIN:
                T[name][int(w)] = _med(s)
    # 枠の中
    uv = np.stack([U, V], 1)
    T["F1"] = _med(uv[fi])
    T["F2"] = {int(k): _med(uv[fi & (rb == k)]) for k in range(4) if (fi & (rb == k)).sum() >= NMIN}
    T["F3"] = {}
    for w in np.unique(word[fi]):
        s = uv[fi & (word == w)]
        if len(s) >= NMIN:
            T["F3"][int(w)] = _med(s)
    T["F4"] = {}
    for w in np.unique(word[fi]):
        for k in range(4):
            s = uv[fi & (word == w) & (rb == k)]
            if len(s) >= NMIN:
                T["F4"][(int(w), k)] = _med(s)
    # C1: 枠そのものを学習側の中央値で置く(コマ寸法で正規化)
    bc = np.stack([(D["bx0"] + D["bx1"]) / 2 / D["W"], (D["by0"] + D["by1"]) / 2 / D["H"]], 1)
    bs = np.stack([(D["bx1"] - D["bx0"]) / D["W"], (D["by1"] - D["by0"]) / D["H"]], 1)
    T["C1_c"] = _med(bc[fi]); T["C1_s"] = _med(bs[fi])
    return T


def predict(D, idx, T, word, rb, U, V, box=None):
    """idx のまとまりに対する各予測子の (x, y)(コマの px)。box は (x0,y0,x1,y1) の上書き用"""
    W, H = D["W"][idx], D["H"][idx]
    if box is None:
        x0, y0, x1, y1 = D["bx0"][idx], D["by0"][idx], D["bx1"][idx], D["by1"][idx]
    else:
        x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    w, k = word[idx], rb[idx]
    out = {}
    out["A0"] = np.stack([W / 2, H / 2], 1)
    for name in ("A1", "A1_all"):
        m = T[name]
        out[name] = np.stack([m[0] * W, m[1] * H], 1)
    for name, fb in (("A2", "A1"), ("A2_all", "A1_all")):
        tab = T[name]; dflt = T[fb]
        m = np.array([tab.get(int(wi), dflt) for wi in w])
        out[name] = np.stack([m[:, 0] * W, m[:, 1] * H], 1)
        out[name + "_hit"] = np.array([int(wi) in tab for wi in w])
    out["F0"] = np.stack([x0 + bw / 2, y0 + bh / 2], 1)
    m = T["F1"]; out["F1"] = np.stack([x0 + m[0] * bw, y0 + m[1] * bh], 1)
    m2 = np.array([T["F2"].get(int(ki), T["F1"]) for ki in k])
    out["F2"] = np.stack([x0 + m2[:, 0] * bw, y0 + m2[:, 1] * bh], 1)
    m3 = np.array([T["F3"].get(int(wi), T["F1"]) for wi in w])
    out["F3"] = np.stack([x0 + m3[:, 0] * bw, y0 + m3[:, 1] * bh], 1)
    out["F3_hit"] = np.array([int(wi) in T["F3"] for wi in w])
    m4 = np.array([T["F4"].get((int(wi), int(ki)), T["F3"].get(int(wi), T["F1"]))
                   for wi, ki in zip(w, k)])
    out["F4"] = np.stack([x0 + m4[:, 0] * bw, y0 + m4[:, 1] * bh], 1)
    # C1: 枠の位置も学習側から(アンカーを与えない連鎖)
    cx0 = T["C1_c"][0] * W - T["C1_s"][0] * W / 2; cy0 = T["C1_c"][1] * H - T["C1_s"][1] * H / 2
    cbw = T["C1_s"][0] * W; cbh = T["C1_s"][1] * H
    out["C1"] = np.stack([cx0 + m3[:, 0] * cbw, cy0 + m3[:, 1] * cbh], 1)
    return out


MODELS = ["A0", "A1_all", "A1", "A2_all", "A2", "F0", "F1", "F2", "F3", "F4", "C1"]


def err(pred, D, idx):
    short = np.minimum(D["W"][idx], D["H"][idx])
    tru = np.stack([D["x"][idx], D["y"][idx]], 1)
    return np.hypot(pred[:, 0] - tru[:, 0], pred[:, 1] - tru[:, 1]) / short


# ---------------------------------------------------------------- ブートストラップ
def boot_pairs(E, panels, pairs, n=1000, seed=SEED):
    """コマ単位ブートストラップ。pairs = [(a, b), ...] について median e(a) - median e(b) の 95% 区間"""
    uniq, inv = np.unique(panels, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    acc = {p: np.empty(n) for p in pairs}
    for t in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[j] for j in pick])
        for (a, b) in pairs:
            acc[(a, b)][t] = np.median(E[a][sel]) - np.median(E[b][sel])
    return {f"{a}-{b}": [round(float(np.percentile(acc[(a, b)], 2.5)), 5),
                         round(float(np.percentile(acc[(a, b)], 97.5)), 5)] for (a, b) in pairs}
# ---------------------------------------------------------------- 本体
def build(dev="cuda"):
    D, rows = load_meta()
    faces, dims = load_faces()
    fbox, U, V, R, BX = assign_boxes(D, faces)
    pan = list(csv.DictReader(open(PACK / "panels.csv")))
    hh = np.array([dims[int(p)][0] for p in D["panel"]], float)
    ww = np.array([dims[int(p)][1] for p in D["panel"]], float)
    D["H"], D["W"] = hh, ww
    D["xn"], D["yn"] = D["x"] / ww, D["y"] / hh
    D["bx0"], D["by0"], D["bx1"], D["by1"] = BX[:, 0], BX[:, 1], BX[:, 2], BX[:, 3]
    return D, rows, pan, faces, dims, fbox, U, V, R, BX


def pick12(keep, inface, R, D, n=12):
    """fig2 と K2 に使う代表例(結果を見る前に固定)。0.1 <= r < 0.5、1 コマ 1 個まで"""
    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(keep & inface & (R >= 0.1) & (R < 0.5))
    rng.shuffle(pool)
    seen, out = set(), []
    for i in pool:
        p = int(D["panel"][i])
        if p in seen:
            continue
        seen.add(p); out.append(int(i))
        if len(out) == n:
            break
    return sorted(out)


MODES2 = ("correct", "swap", "shift50", "localswap")


def k2(rows, pan, D, picks, faces, dims):
    """外部の基準(線画 PNG とコマの寸法)との照合と陰性対照(lesson 12)"""
    import cv2
    P = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    M = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    acc = {k: [] for k in MODES2}
    for i in picks:
        pid = int(D["panel"][i])
        img = cv2.imread(str(panel_png(pan[pid])), cv2.IMREAD_GRAYSCALE)
        dt = cv2.distanceTransform((img >= 200).astype(np.uint8), cv2.DIST_L2, 3)
        for k in MODES2:
            acc[k].append(on_line_frac(dt, _inst_points(P, M, rows, i, k)))
    med = {k: round(float(np.median(acc[k])), 4) for k in MODES2}
    # K2e: 枠の向き。列名どおりならコマの中に収まり、入れ替えるとコマの外に出る
    rec = []
    for pid, fl in faces.items():
        h, w = dims[pid]
        for (k, x0, y0, x1, y1) in fl:
            rec.append((w, h, x0, y0, x1, y1))
    w, h, x0, y0, x1, y1 = np.array(rec, float).T
    t = 2.0
    out_named = float(((x0 < -t) | (y0 < -t) | (x1 > w + t) | (y1 > h + t)).mean())
    out_swap = float(((y0 < -t) | (x0 < -t) | (y1 > w + t) | (x1 > h + t)).mean())
    det = dict(png_on_line=med, png_on_line_min=round(float(np.min(acc["correct"])), 4),
               per_instance=[round(v, 4) for v in acc["correct"]],
               box_out_of_panel_as_named=round(out_named, 4),
               box_out_of_panel_swapped=round(out_swap, 4), n_boxes=len(rec))
    ok = (med["correct"] >= 0.8 and med["swap"] <= 0.2 and med["shift50"] <= 0.2
          and out_named <= 0.10 and out_swap >= 0.30)
    return ok, det


def swap_boxes(D, idx, seed=SEED, block=50):
    """K4: 面積の近い枠同士を無作為に入れ替える(評価側の枠だけ差し替える)"""
    rng = np.random.default_rng(seed + 1)
    key = np.stack([D["panel"][idx], D["bx0"][idx], D["by0"][idx]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    bx = np.stack([D["bx0"][idx], D["by0"][idx], D["bx1"][idx], D["by1"][idx]], 1)
    ub = np.zeros((len(uniq), 4))
    for j in range(len(uniq)):
        ub[j] = bx[inv == j][0]
    area = (ub[:, 2] - ub[:, 0]) * (ub[:, 3] - ub[:, 1])
    order = np.argsort(area)
    perm = order.copy()
    for s in range(0, len(order), block):
        blk = order[s:s + block]
        if len(blk) > 1:
            perm[s:s + len(blk)] = np.roll(blk, 1 + int(rng.integers(1, len(blk))) % max(len(blk) - 1, 1))
    newb = np.empty_like(ub)
    newb[order] = ub[perm]
    sb = newb[inv]
    return sb[:, 0], sb[:, 1], sb[:, 2], sb[:, 3]


def run_half(D, keep, inface, U, V, rb, word, half_fit, out, verbose=True):
    """学習 half_fit -> 評価 1-half_fit。返る: 結果 dict"""
    half_ev = 1 - half_fit
    T = fit_tables(D, keep, inface, U, V, rb, word, half_fit)
    idx = np.flatnonzero(keep & inface & (D["half"] == half_ev))
    pr = predict(D, idx, T, word, rb, U, V)
    E = {m: err(pr[m], D, idx) for m in MODELS}
    res = dict(
        fit_half="AB"[half_fit], eval_half="AB"[half_ev], n_eval=int(len(idx)),
        n_panels=int(len(np.unique(D["panel"][idx]))),
        median={m: round(float(np.median(E[m])), 5) for m in MODELS},
        mean={m: round(float(np.mean(E[m])), 5) for m in MODELS},
        word_table_hit=dict(A2=round(float(pr["A2_hit"].mean()), 4),
                            F3=round(float(pr["F3_hit"].mean()), 4)),
        n_words_F3=len(T["F3"]), n_words_A2=len(T["A2"]), n_cells_F4=len(T["F4"]),
    )
    pairs = [("F3", "F0"), ("F3", "A2"), ("F3", "A2_all"), ("F3", "F1"), ("F4", "F2"),
             ("A2", "A1"), ("C1", "A2"), ("F2", "F0"), ("F0", "A2")]
    res["ci95_median_diff"] = boot_pairs(E, D["panel"][idx], pairs)
    # r 区分別(記録のみ)
    res["by_rbin"] = {}
    for k in range(4):
        m = rb[idx] == k
        if m.sum() < 50:
            continue
        res["by_rbin"][R_NAMES[k]] = dict(n=int(m.sum()), **{
            mm: round(float(np.median(E[mm][m])), 5) for mm in ("A2", "F0", "F2", "F3", "F4")})
    # K4 陰性対照: 枠を入れ替える
    sb = swap_boxes(D, idx)
    prs = predict(D, idx, T, word, rb, U, V, box=sb)
    Es = {m: err(prs[m], D, idx) for m in ("F0", "F3")}
    res["K4_swapped"] = dict(
        median_F0=round(float(np.median(Es["F0"])), 5), median_F3=round(float(np.median(Es["F3"])), 5),
        ci95_F3_minus_F0=boot_pairs(Es, D["panel"][idx], [("F3", "F0")])["F3-F0"])
    if verbose:
        print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return res, T, idx, pr, E


def fig1(out, inface, keep, U, V, rb):
    import cv2
    G, S = 30, 330
    tiles = []
    for k in range(4):
        m = keep & inface & (rb == k)
        h, _, _ = np.histogram2d(V[m], U[m], bins=[G, G], range=[[0, 1], [0, 1]])
        v = h / max(h.max(), 1)
        img = (255 * (1 - v ** 0.5)).astype(np.uint8)
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, 0), (S - 1, S - 1), (0, 0, 255), 2)
        cv2.putText(img, f"{R_NAMES[k]} n={int(m.sum())}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)
    cv2.imwrite(str(out / "fig1_uv_density.png"), np.concatenate(tiles, 1))


def fig2(out, rows, pan, D, picks, T, word, rb, U, V):
    """コマ線画 PNG の上に 真(緑)・A2(青)・F3(橙)・枠(赤)。向きの誤りが目で分かる形"""
    import cv2
    P = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    M = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    CELL = 420
    cells = []
    for i in picks:
        pid = int(D["panel"][i])
        img = cv2.imread(str(panel_png(pan[pid])))
        f = CELL / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)))
        canvas = np.full((CELL, CELL, 3), 245, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        pr = predict(D, np.array([i]), T, word, rb, U, V)
        pts = _inst_points(P, M, rows, i, "correct") * f
        for (px, py) in pts.astype(int):
            cv2.circle(canvas, (px, py), 1, (0, 0, 220), -1)
        b = (np.array([D["bx0"][i], D["by0"][i], D["bx1"][i], D["by1"][i]]) * f).astype(int)
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (0, 0, 220), 2)
        for name, col, mk in (("A2", (220, 60, 0), "x"), ("F3", (0, 150, 240), "+")):
            q = (pr[name][0] * f).astype(int)
            cv2.drawMarker(canvas, (q[0], q[1]), col,
                           cv2.MARKER_TILTED_CROSS if mk == "x" else cv2.MARKER_CROSS, 22, 3)
        t = (np.array([D["x"][i], D["y"][i]]) * f).astype(int)
        cv2.circle(canvas, (t[0], t[1]), 9, (0, 170, 0), 3)
        cv2.putText(canvas, f"#{i} w{int(word[i])} r={R_NAMES[int(rb[i])]}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (CELL - 1, CELL - 1), (150, 150, 150), 1)
        cells.append(canvas)
    rows_ = [np.concatenate(cells[r * 4:(r + 1) * 4], 1) for r in range(3)]
    grid = np.concatenate(rows_, 0)
    leg = np.full((34, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(leg, "green circle = true / blue x = A2 (absolute, per-word) / "
                     "orange + = F3 (frame + word) / red box = face frame, red dots = the instance",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "fig2_predictions.png"), np.concatenate([leg, grid], 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/framerel_20261009")
    ap.add_argument("--check", action="store_true", help="道具確認のみ")
    a = ap.parse_args()
    out = HERE.parent.parent / a.out; out.mkdir(parents=True, exist_ok=True)

    D, rows, pan, faces, dims, fbox, U, V, R, BX = build()
    inface = fbox > 0
    rb = rbin_of(R)
    keep, ndrop = dedupe(D)
    chk = {}

    ok1, det1 = k1(D, fbox, U, V, R)
    chk["K1"] = dict(passed=ok1, **det1)
    print("K1(既知の値のセル):", "合格" if ok1 else "不合格", flush=True)
    if not ok1:
        json.dump(chk, open(out / "check.json", "w"), ensure_ascii=False, indent=1); return 1

    picks = pick12(keep, inface, R, D)
    ok2, det2 = k2(rows, pan, D, picks, faces, dims)
    chk["K2"] = dict(passed=ok2, picks=picks, **det2)
    print("K2(線画 PNG との照合 + 陰性対照):", "合格" if ok2 else "不合格",
          json.dumps(det2["png_on_line"]), "枠はみ出し",
          det2["box_out_of_panel_as_named"], "対", det2["box_out_of_panel_swapped"], flush=True)

    word = encode_words("cuda")
    # K5: 群の漏れ
    leak = {}
    for hf in (0, 1):
        sf = set(D["series"][keep & (D["half"] == hf)].tolist())
        se = set(D["series"][keep & (D["half"] == 1 - hf)].tolist())
        leak["AB"[hf]] = sorted(sf & se)
    ok5 = all(not v for v in leak.values())
    chk["K5"] = dict(passed=ok5, overlap=leak,
                     n_series_A=len(set(D["series"][D["half"] == 0].tolist())),
                     n_series_B=len(set(D["series"][D["half"] == 1].tolist())))
    print("K5(群の漏れ):", "合格" if ok5 else "不合格", flush=True)

    # K3: 採点の健全性
    idx0 = np.flatnonzero(keep & inface & (D["half"] == 1))
    tru = np.stack([D["x"][idx0], D["y"][idx0]], 1)
    e_id = err(tru, D, idx0)
    rng = np.random.default_rng(SEED)
    uni = np.stack([rng.random(len(idx0)) * D["W"][idx0], rng.random(len(idx0)) * D["H"][idx0]], 1)
    e_un = float(np.median(err(uni, D, idx0)))
    ok3 = bool(e_id.max() <= 1e-9 and 0.3 <= e_un <= 0.7)
    chk["K3"] = dict(passed=ok3, e_identity_max=float(e_id.max()), e_uniform_median=round(e_un, 4))
    print("K3(採点の健全性):", "合格" if ok3 else "不合格", "一様の中央値", round(e_un, 4), flush=True)

    chk["dedupe_panels_dropped"] = ndrop
    chk["n_kept_in_face"] = int((keep & inface).sum())
    passed_pre = ok1 and ok2 and ok3 and ok5
    json.dump(chk, open(out / "check.json", "w"), ensure_ascii=False, indent=1)
    print("道具確認(K1/K2/K3/K5):", "合格" if passed_pre else "不合格", flush=True)
    if not passed_pre:
        print("主測定に進まず停止。", flush=True); return 1
    if a.check:
        return 0

    results = {}
    for hf in (0, 1):
        res, T, idx, pr, E = run_half(D, keep, inface, U, V, rb, word, hf, out)
        results["AB"[hf] + "->" + "AB"[1 - hf]] = res
        if hf == 0:
            T0 = T
    k4ok = all(r["K4_swapped"]["ci95_F3_minus_F0"][0] <= 0 <= r["K4_swapped"]["ci95_F3_minus_F0"][1]
               for r in results.values())
    chk["K4"] = dict(passed=k4ok, detail={k: v["K4_swapped"] for k, v in results.items()})
    print("K4(枠入替の陰性対照):", "合格" if k4ok else "不合格", flush=True)
    json.dump(chk, open(out / "check.json", "w"), ensure_ascii=False, indent=1)
    json.dump(results, open(out / "results.json", "w"), ensure_ascii=False, indent=1)

    fig1(out, inface, keep, U, V, rb)
    fig2(out, rows, pan, D, picks, T0, word, rb, U, V)
    print("図:", out / "fig1_uv_density.png", out / "fig2_predictions.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
