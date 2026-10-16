#!/usr/bin/env python
"""IC-k7(2026-10-16 事前登録): pack の線番号と、私が再現した span の番号が一致するか。

IC-k5b は本数の一致までしか見ていない。作者のラベルは線番号で付いているので、
番号が対応しなければ K5・K2・K4 主脚は全部無意味になる。外れたら止める。

棒: 線の 99% 以上で「pack の 16 点 → 私の線 i の画素集合」の距離の中央値 <= 2.0px
NC-k7: 同じ判定を線 i+1(巡回)に当て、99% 以上で中央値 > 2.0px に落ちること
"""
import argparse, csv, json, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_F = Path("/home/sh1/deepl/lineart-stroke-grammar")
TRACK_H = Path("/home/sh1/deepl/lineart-face-words")
PACK = TRACK_F / "results/panel_pack_20260919"
DATASET = Path("/home/sh1/deepl/lineart/dataset")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TRACK_F / "tools/stroke"))

THR, GROW, MAX_BEND = 128, 200, 35.0
DIST_BAR, SHARE_BAR = 2.0, 0.99        # 旧形(画素集合までの距離)。記録として併記する
PT_BAR = 0.01                          # 修正 C1: 16 点の最大ずれ [px]。主の棒


def work(t):
    pid, source, name, start, n = t
    g = cv2.imread(str(DATASET / source / "line" / name), 0)
    if g is None:
        return dict(pid=pid, ok=False)
    from skimage.morphology import skeletonize
    from stroke_path import chain, resample, span_paths
    from train_infill import POINTS, to16
    import strokes_instrumented as SI
    ink = SI.ink_mask(g, thr=THR, grow=GROW)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    spans, sos, keep, info, _x = SI.strokes_from_skeleton(sk, dist * 2, max_bend=MAX_BEND)
    groups = {}
    for i in range(len(spans)):
        groups.setdefault(int(sos[i]), []).append(i)
    mine, mine16, nspan = [], [], []                 # export と同じ順序
    for k, mem in groups.items():
        if not keep[k]:
            continue
        path, _gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        mine.append(np.stack([rr, cc], 1).astype(np.float64))
        q = resample(path, 8.0, 8, 64)               # export と同一の手順
        mine16.append(np.asarray(to16(np.asarray(q, np.float32), len(q)), np.float64))
        nspan.append(len(mem))
    arr = np.load(PACK / "strokes.npy", mmap_mode="r")
    block = np.asarray(arr[start:start + n])
    pts = block[:, :32].reshape(n, 16, 2)            # pack は (y, x) のまま
    if len(mine) != n:
        return dict(pid=pid, ok=False, n_mine=len(mine), n_pack=n)
    med, med_nc, pt, pt_nc = [], [], [], []
    trees = [cKDTree(m) for m in mine]
    ns = np.asarray(nspan)
    for i in range(n):
        med.append(float(np.median(trees[i].query(pts[i])[0])))
        med_nc.append(float(np.median(trees[(i + 1) % n].query(pts[i])[0])))
        pt.append(float(np.abs(mine16[i] - pts[i]).max()))
        pt_nc.append(float(np.abs(mine16[(i + 1) % n] - pts[i]).max()))
    med, pt = np.asarray(med), np.asarray(pt)
    one = ns == 1
    return dict(pid=pid, ok=True, n=n,
                pt_hit=int((pt <= PT_BAR).sum()), pt_nc_hit=int((np.asarray(pt_nc) <= PT_BAR).sum()),
                pt_p50=float(np.median(pt)), pt_p99=float(np.percentile(pt, 99)),
                hit=int((med <= DIST_BAR).sum()),
                nc_hit=int(sum(1 for d in med_nc if d <= DIST_BAR)),
                hit_1span=int((med[one] <= DIST_BAR).sum()), n_1span=int(one.sum()),
                hit_multispan=int((med[~one] <= DIST_BAR).sum()), n_multispan=int((~one).sum()),
                med_p50=float(np.median(med)), med_p99=float(np.percentile(med, 99)),
                nc_p50=float(np.median(med_nc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/k4a_bridge_loss_20261016")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    recs = [r for r in G.S["faces"].values() if r["usage"] == "train" and r["status"] == "done" and "broken" in r]
    pids = sorted({r["panel"] for r in recs})
    if a.limit:
        pids = pids[:a.limit]
    P = G.S["panels"]
    tasks = [(p, P[p]["source"], P[p]["name"], int(P[p]["start"]), int(P[p]["n"])) for p in pids]
    print(f"コマ {len(tasks)} 枚", flush=True)
    rows, bad = [], []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            (rows if r["ok"] else bad).append(r)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    tot = sum(r["n"] for r in rows)
    g = lambda k: sum(r[k] for r in rows)
    share, nc_share = g("pt_hit") / tot, g("pt_nc_hit") / tot
    chk = dict(n_panels=len(rows), failed_panels=len(bad), n_strokes=tot,
               primary_C1_16pt=dict(bar_px=PT_BAR, match_share=round(share, 5), share_bar=SHARE_BAR,
                                    nc_match_share=round(nc_share, 5)),
               side_old_distance=dict(bar_px=DIST_BAR, match_share=round(g("hit") / tot, 5),
                                      nc_match_share=round(g("nc_hit") / tot, 5),
                                      one_span=dict(n=g("n_1span"),
                                                    share=round(g("hit_1span") / max(g("n_1span"), 1), 5)),
                                      multi_span=dict(n=g("n_multispan"),
                                                      share=round(g("hit_multispan") / max(g("n_multispan"), 1), 5)),
                                      note="pack の 16 点は隙間を跨ぐ鎖の弧長標本なので、"
                                           "隙間に落ちた点は骨格画素から必ず遠い。番号の誤りではない"),
               passed=bool(not bad and share >= SHARE_BAR and nc_share <= 1 - SHARE_BAR))
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    json.dump(chk, open(out / "ic_k7.json", "w"), indent=1, ensure_ascii=False)
    with open(out / "ic_k7_per_panel.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    print("IC-k7:", "合格" if chk["passed"] else "不合格 —— 番号を使う測定はしない")


if __name__ == "__main__":
    main()
