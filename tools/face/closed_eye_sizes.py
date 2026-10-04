#!/usr/bin/env python
"""最初の一手 1(2026-09-25 事前登録): 閉じた目の候補 w342・w343・w344 を大きさ別に並べる。

大きさ = 直径/コマ短辺 = 2*28/scale / min(H, W)(Track G relpos.rel_size と同式)。
区分 <0.05 / 0.05-0.1 / 0.1-0.2 / 0.2-0.4 / >=0.4 から seed 固定の一様無作為で最大 10 個。
1 インスタンス = 近景(一辺 直径x2)+遠景(一辺 max(直径x6, 短辺x0.3)、コマ全体で頭打ち)。
背景は panel_pack の元のコマ線画(灰)、当該インスタンスは赤。

道具確認: (1) 件数が Track G word_reliability_v2.csv と一致 (2) インスタンス点が元の線画から
3px 以内にある割合の中央値 >= 0.8(3 語とも)。--check で道具確認だけ行う。
座標(2026-10-04 訂正): strokes/pts の並びは (y, x)(線画 PNG と照合して確認)。meta の cy = y、cx = x。
corpus dims は (W, H) の順で入っている。このファイルは読み込み時にすべて画像の (x, y)・(H, W) に直す。
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trackf"))
sys.path.insert(0, str(ROOT / "gen"))
from common import CODEBOOK, CORPUS                                    # noqa: E402
from train_codebook2 import Codebook2                                  # noqa: E402

TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CLUSTERS = TRACKF / "results/cluster_set_20260919"
PACK = TRACKF / "results/panel_pack_20260919"
REL_V2 = Path("/home/sh1/deepl/lineart-panel-generation/results/word_reliability_v2_20260924/word_reliability_v2.csv")
WORDS = (342, 343, 344)
EDGES = np.array([0.05, 0.1, 0.2, 0.4])
BIN_NAMES = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", ">=0.4"]
A_SIZE = 28.0
CELL = 240
PER_LINE = 5


def encode_all(dev):
    """cluster_set 全行(meta の行番号順)を符号化。返る: words, meta, pts, width, mask"""
    meta = list(csv.DictReader(open(CLUSTERS / "meta.csv")))
    P = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    W = np.load(CLUSTERS / "width.npy", mmap_mode="r")
    M = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    cm = Codebook2(levels=tuple(cb["levels"]), coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    ws = []
    with torch.no_grad():
        for i in range(0, len(meta), 4096):
            p = torch.from_numpy(np.asarray(P[i:i + 4096])).to(dev).float()
            w = torch.from_numpy(np.asarray(W[i:i + 4096])).to(dev).float()
            m = torch.from_numpy(np.asarray(M[i:i + 4096])).to(dev)
            ws.append(cm.encode(p, w, m, 1)[1].cpu().numpy())
    return np.concatenate(ws), meta, P, M


class Panels:
    """panel_pack の元のコマ線画(ポリライン)。返す座標は画像の (x, y)、dims は (H, W)"""

    def __init__(self):
        self.rows = list(csv.DictReader(open(PACK / "panels.csv")))
        self.strokes = np.load(PACK / "strokes.npy", mmap_mode="r")
        self.dims = np.load(CORPUS)["dims"][:, ::-1]      # corpus dims は (W, H) 順 → (H, W)
        self._dt = {}

    def lines(self, pid):
        r = self.rows[pid]
        s = np.asarray(self.strokes[int(r["start"]):int(r["start"]) + int(r["n"])])
        pp = s[:, :32].reshape(-1, 16, 2).astype(np.float64)[..., ::-1]   # (y, x) → (x, y)
        ink = np.abs(pp).sum(-1) > 0
        return [pp[k][ink[k]] for k in range(len(pp)) if ink[k].any()]

    def dist(self, pid):
        """元の線画(1px)までの距離変換(ネイティブ px)"""
        if pid not in self._dt:
            H, Wd = (int(v) for v in self.dims[pid])
            img = np.full((H + 4, Wd + 4), 255, np.uint8)
            for ln in self.lines(pid):
                cv2.polylines(img, [np.round(ln).astype(np.int32)], False, 0, 1)
            self._dt = {pid: cv2.distanceTransform(img, cv2.DIST_L2, 3)}
        return self._dt[pid]


def inst_points(P, M, meta, i):
    sc = float(meta[i]["scale"])
    xy = np.array([float(meta[i]["cx"]), float(meta[i]["cy"])])   # 画像の (x, y)
    m = np.asarray(M[i])
    pts = np.asarray(P[i], np.float64)[..., ::-1] / sc + xy       # pts は (y, x) 順
    return [pts[s] for s in range(len(m)) if m[s]]


def rel_size(meta, dims, i):
    H, Wd = dims[int(meta[i]["panel"])]
    return 2 * A_SIZE / float(meta[i]["scale"]) / min(H, Wd)


def on_line_frac(pan, P, M, meta, i, tol=3.0):
    dt = pan.dist(int(meta[i]["panel"]))
    pts = np.concatenate(inst_points(P, M, meta, i))
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, dt.shape[1] - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, dt.shape[0] - 1)
    return float((dt[yi, xi] <= tol).mean())


def crop_cell(pan, lines, inst, center, side, pid):
    """center (x, y) を中心に一辺 side(ネイティブ px)の窓を CELL px に描く"""
    f = CELL / side
    o = np.array(center) - side / 2
    img = np.full((CELL, CELL, 3), 255, np.uint8)
    H, Wd = (float(v) for v in pan.dims[pid])
    cv2.rectangle(img, tuple(np.round((np.array([0, 0]) - o) * f).astype(int)),
                  tuple(np.round((np.array([Wd, H]) - o) * f).astype(int)), (230, 200, 160), 1)
    for ln in lines:
        q = (ln - o) * f
        if (q.max(0) < -2).any() or (q.min(0) > CELL + 2).any():
            continue
        cv2.polylines(img, [np.round(q * 4).astype(np.int32)], False, (120, 120, 120), 1, cv2.LINE_AA, 2)
    for ln in inst:
        q = (ln - o) * f
        cv2.polylines(img, [np.round(q * 4).astype(np.int32)], False, (0, 0, 220), 2 if side < 400 else 1,
                      cv2.LINE_AA, 2)
    return img


def label_bar(w, text, h=22, color=(60, 60, 60)):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (3, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/closed_eye_candidates_20260925")
    ap.add_argument("--per_bin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--check", action="store_true", help="道具確認だけ行う")
    a = ap.parse_args()
    out = ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)

    words, meta, P, M = encode_all(dev)
    pan = Panels()
    rs = np.array([rel_size(meta, pan.dims, i) for i in range(len(meta))])
    ref = {int(r["word"]): int(r["count"]) for r in csv.DictReader(open(REL_V2))}

    # 抽出(区分ごとの無作為)は道具確認より先に固定する(結果を見て選び直さない)
    picks, counts = {}, []
    for w in WORDS:
        idx = np.flatnonzero(words == w)
        b = np.searchsorted(EDGES, rs[idx], side="right")
        for k in range(len(BIN_NAMES)):
            pool = idx[b == k]
            picks[(w, k)] = np.sort(rng.choice(pool, min(a.per_bin, len(pool)), replace=False)) if len(pool) else []
            counts.append(dict(word=w, bin=BIN_NAMES[k], n=len(pool), frac=round(len(pool) / len(idx), 4)))
    with open(out / "counts.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["word", "bin", "n", "frac"]); wr.writeheader(); wr.writerows(counts)

    # 道具確認
    chk = {}
    for w in WORDS:
        n = int((words == w).sum())
        sel = np.concatenate([np.asarray(picks[(w, k)], int) for k in range(len(BIN_NAMES))])
        fr = [on_line_frac(pan, P, M, meta, int(i)) for i in sel]
        chk[w] = dict(count=n, ref_count=ref[w], count_ok=n == ref[w],
                      on_line_median=round(float(np.median(fr)), 4), on_line_min=round(float(np.min(fr)), 4),
                      on_line_ok=bool(np.median(fr) >= 0.8), n_checked=len(fr),
                      rel_size_p10_p90=[round(float(v), 4) for v in np.percentile(rs[words == w], [10, 90])])
        print(w, chk[w], flush=True)
    ok = all(c["count_ok"] and c["on_line_ok"] for c in chk.values())
    json.dump(dict(check=chk, passed=ok), open(out / "check.json", "w"), indent=1)
    print("道具確認:", "合格" if ok else "不合格", flush=True)
    if a.check or not ok:
        return

    # 本番の図
    for w in WORDS:
        n_all = int((words == w).sum())
        rows_img = []
        for k in range(len(BIN_NAMES)):
            cells = []
            for i in picks[(w, k)]:
                i = int(i); pid = int(meta[i]["panel"])
                H, Wd = (float(v) for v in pan.dims[pid])
                inst = inst_points(P, M, meta, i)
                allp = np.concatenate(inst)
                c = (allp.min(0) + allp.max(0)) / 2
                d = rs[i] * min(H, Wd)
                lines = pan.lines(pid)
                near = crop_cell(pan, lines, inst, c, max(d * 2, 8.0), pid)
                far = crop_cell(pan, lines, inst, c, min(max(d * 6, 0.3 * min(H, Wd)), max(H, Wd)), pid)
                pair = np.hstack([near, np.full((CELL, 2, 3), 200, np.uint8), far])
                cells.append(np.vstack([pair, label_bar(pair.shape[1], f"r{i} p{pid} s={rs[i]:.3f}  {meta[i]['work'][:28]}")]))
            cw = 2 * CELL + 2
            while len(cells) < a.per_bin:
                cells.append(np.full((CELL + 22, cw, 3), 245, np.uint8))
            nb = next(c["n"] for c in counts if c["word"] == w and c["bin"] == BIN_NAMES[k])
            cells = [np.pad(c_, ((0, 6), (0, 10), (0, 0)), constant_values=255) for c_ in cells]
            row = np.vstack([np.hstack(cells[j:j + PER_LINE]) for j in range(0, len(cells), PER_LINE)])
            head = label_bar(row.shape[1], f"size {BIN_NAMES[k]}   n={nb} ({nb / n_all:.1%})", 28, (0, 0, 160))
            rows_img.append(np.vstack([head, row, np.full((8, row.shape[1], 3), 255, np.uint8)]))
        title = label_bar(rows_img[0].shape[1], f"w{w}  all={n_all}  cell: near(side=2d) | far(side=max(6d,0.3 short)); red=instance, gray=original panel lines", 30, (0, 0, 0))
        path = out / f"w{w}_by_size.png"
        cv2.imwrite(str(path), np.vstack([title] + rows_img))
        print("->", path, flush=True)


if __name__ == "__main__":
    main()
