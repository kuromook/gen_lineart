#!/usr/bin/env python
"""実験 H3(2026-10-04 事前登録): 顔の枠の中のまとまりを、枠の中の位置で分けて並べる。

--detect : 全コマに顔検出(H2 と同じ設定)→ faces.csv
(既定)   : 道具確認 → 図 A(枠の中の位置の密度、相対的な大きさ r の区分ごと)→ 図 B(3x3 区画ごとの実例)

まとまりの中心(meta cx = x、cy = y)が入る顔の枠(複数なら面積最小)に割り当てる。
(u, v) = 枠の左上 (0,0)・右下 (1,1)。r = 直径(2*28/scale)/ 枠の一辺(sqrt 面積)。
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_eye_sizes as C                                           # noqa: E402
import h2_face_detect as H2                                            # noqa: E402

R_EDGES = [0.1, 0.25, 0.5]
R_NAMES = ["r<0.1", "0.1-0.25", "0.25-0.5", "r>=0.5"]
R_SHOW = (0.1, 0.5)
ZONE_X = ["left", "center", "right"]
ZONE_Y = ["top", "middle", "bottom"]
CELL = 200
PER_ZONE = 12
SEED = 20261004
EAR = 163364


def detect_all(out):
    rows = list(csv.DictReader(open(C.PACK / "panels.csv")))
    det = H2.FaceDetector()
    table, missing = [], 0
    for pid, r in enumerate(rows):
        img = cv2.imread(str(H2.panel_png(r)))
        if img is None:
            missing += 1
            table.append(dict(panel=pid, box=-1, conf="", x0="", y0="", x1="", y1="", png_h="", png_w=""))
            continue
        boxes = det(img)
        for k, (x0, y0, x1, y1, c) in enumerate(boxes, 1):
            table.append(dict(panel=pid, box=k, conf=round(c, 4), x0=round(x0, 1), y0=round(y0, 1),
                              x1=round(x1, 1), y1=round(y1, 1), png_h=img.shape[0], png_w=img.shape[1]))
        if not boxes:
            table.append(dict(panel=pid, box=0, conf="", x0="", y0="", x1="", y1="",
                              png_h=img.shape[0], png_w=img.shape[1]))
        if pid % 500 == 0:
            print(f"detect {pid}/{len(rows)}", flush=True)
    with open(out / "faces.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    print(f"faces.csv: コマ {len(rows)}  PNG なし {missing}  枠 {sum(1 for t in table if int(t['box']) > 0)}", flush=True)


def load_faces(out):
    faces = defaultdict(list)
    n_png_missing = 0
    for t in csv.DictReader(open(out / "faces.csv")):
        if int(t["box"]) > 0:
            faces[int(t["panel"])].append((int(t["box"]), float(t["x0"]), float(t["y0"]), float(t["x1"]),
                                           float(t["y1"]), float(t["conf"])))
        n_png_missing += int(t["box"]) < 0
    return faces, n_png_missing


def png_on_line(pan_rows, pts, pid, cache, tol=3.0):
    """点が線画 PNG のインク(<200)から tol px 以内にある割合(外部の基準)"""
    if pid not in cache:
        cache.clear()
        img = cv2.imread(str(H2.panel_png(pan_rows[pid])), cv2.IMREAD_GRAYSCALE)
        cache[pid] = cv2.distanceTransform((img >= 200).astype(np.uint8), cv2.DIST_L2, 3)
    dt = cache[pid]
    x = np.round(pts[:, 0]).astype(int); y = np.round(pts[:, 1]).astype(int)
    inb = (x >= 0) & (y >= 0) & (x < dt.shape[1]) & (y < dt.shape[0])
    return float((dt[y[inb], x[inb]] <= tol).sum() / len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h3_face_parts_20261004")
    ap.add_argument("--detect", action="store_true")
    a = ap.parse_args()
    out = C.ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    if a.detect:
        return detect_all(out)

    import torch
    faces, n_missing = load_faces(out)
    words, meta, P, M = C.encode_all(torch.device("cuda"))
    pan = C.Panels()
    N = len(meta)
    panel = np.array([int(m["panel"]) for m in meta])
    cx = np.array([float(m["cx"]) for m in meta]); cy = np.array([float(m["cy"]) for m in meta])
    diam = 2 * C.A_SIZE / np.array([float(m["scale"]) for m in meta])

    # 割り当て
    fbox = np.full(N, -1); U = np.full(N, np.nan); V = np.full(N, np.nan); R = np.full(N, np.nan)
    box_of = {}
    by_panel = defaultdict(list)
    for i in range(N):
        by_panel[panel[i]].append(i)
    for pid, fl in faces.items():
        idx = np.array(by_panel.get(pid, []), int)
        if not len(idx):
            continue
        best = np.full(len(idx), np.inf)
        for (k, x0, y0, x1, y1, _c) in fl:
            area = (x1 - x0) * (y1 - y0)
            ins = (cx[idx] >= x0) & (cx[idx] <= x1) & (cy[idx] >= y0) & (cy[idx] <= y1) & (area < best)
            j = idx[ins]
            fbox[j] = k; U[j] = (cx[j] - x0) / (x1 - x0); V[j] = (cy[j] - y0) / (y1 - y0)
            R[j] = diam[j] / np.sqrt(area); best[ins] = area
            box_of[(pid, k)] = (x0, y0, x1, y1)
    inface = fbox > 0
    rbin = np.searchsorted(R_EDGES, R, side="right")
    zone = (np.clip(np.floor(np.nan_to_num(V) * 3), 0, 2) * 3
            + np.clip(np.floor(np.nan_to_num(U) * 3), 0, 2)).astype(int)

    # 図 B の抽出(道具確認より先に固定)
    rng = np.random.default_rng(SEED)
    show = inface & (R >= R_SHOW[0]) & (R < R_SHOW[1])
    picks = {}
    for z in range(9):
        pool = np.flatnonzero(show & (zone == z))
        picks[z] = np.sort(rng.choice(pool, min(PER_ZONE, len(pool)), replace=False))

    # 道具確認
    smoke = list(csv.DictReader(open(C.ROOT.parent / "results/h2_face_detect_smoke_20261004/detections.csv")))
    agree = True
    for pid in sorted({int(t["panel"]) for t in smoke}):
        ref = sorted((round(float(t["x0"])), round(float(t["y0"])), round(float(t["x1"])), round(float(t["y1"])))
                     for t in smoke if int(t["panel"]) == pid and int(t["box"]) > 0)
        mine = sorted((round(b[1]), round(b[2]), round(b[3]), round(b[4])) for b in faces.get(pid, []))
        if len(ref) != len(mine) or any(max(abs(p - q) for p, q in zip(r_, m_)) > 1 for r_, m_ in zip(ref, mine)):
            agree = False; print("H2 と不一致: panel", pid, ref, mine, flush=True)
    ear_ok = bool(inface[EAR] and (U[EAR] < 1 / 3 or U[EAR] > 2 / 3))
    cache, fr = {}, []
    for z in range(9):
        for i in picks[z]:
            fr.append(png_on_line(pan.rows, np.concatenate(C.inst_points(P, M, meta, int(i))), int(panel[i]), cache))
    n_panels = len(pan.rows)
    chk = dict(h2_agree=agree, ear_in_face=bool(inface[EAR]), ear_uv=[round(float(U[EAR]), 3), round(float(V[EAR]), 3)],
               ear_r=round(float(R[EAR]), 3), ear_ok=ear_ok,
               png_on_line_median=round(float(np.median(fr)), 4), png_on_line_min=round(float(np.min(fr)), 4),
               png_ok=bool(np.median(fr) >= 0.8), png_missing=n_missing,
               panels=n_panels, panels_with_face=len(faces), boxes=sum(len(v) for v in faces.values()),
               clusters=N, clusters_in_face=int(inface.sum()),
               r_bin_counts={R_NAMES[k]: int((inface & (rbin == k)).sum()) for k in range(4)},
               zone_counts_shown={f"{ZONE_Y[z // 3]}-{ZONE_X[z % 3]}": int((show & (zone == z)).sum()) for z in range(9)})
    chk["passed"] = bool(agree and ear_ok and chk["png_ok"])
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False), flush=True)
    print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    # 図 A: (u, v) の密度(r の区分ごと)
    G, S = 30, 330
    tiles = []
    for k in range(4):
        sel = inface & (rbin == k)
        h, _, _ = np.histogram2d(V[sel], U[sel], bins=G, range=[[0, 1], [0, 1]])
        g = (255 - 215 * h / max(h.max(), 1)).astype(np.uint8)
        im = cv2.resize(g, (S, S), interpolation=cv2.INTER_NEAREST)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for t in (1, 2):
            cv2.line(im, (S * t // 3, 0), (S * t // 3, S), (200, 170, 120), 1)
            cv2.line(im, (0, S * t // 3), (S, S * t // 3), (200, 170, 120), 1)
        cv2.rectangle(im, (0, 0), (S - 1, S - 1), (90, 90, 90), 1)
        tiles.append(cv2.copyMakeBorder(np.vstack([C.label_bar(S, f"{R_NAMES[k]}  n={int(sel.sum())}", 24), im]),
                                        6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    fa = np.hstack(tiles)
    fa = np.vstack([C.label_bar(fa.shape[1], "position of cluster centers inside the face box (dark = many); "
                                             "r = cluster diameter / face box side", 26, (0, 0, 0)), fa])
    cv2.imwrite(str(out / "figA_density.png"), fa)
    print("->", out / "figA_density.png", flush=True)

    # 図 B: 区画ごとの実例
    C.CELL = CELL
    parts, table, no = [], [], 0
    for z in range(9):
        sets = []
        for i in picks[z]:
            i = int(i); no += 1; pid = int(panel[i])
            x0, y0, x1, y1 = box_of[(pid, int(fbox[i]))]
            side = 1.4 * max(x1 - x0, y1 - y0)
            ctr = ((x0 + x1) / 2, (y0 + y1) / 2)
            im = C.crop_cell(pan, pan.lines(pid), C.inst_points(P, M, meta, i), ctr, side, pid)
            f = CELL / side; o = np.array(ctr) - side / 2
            cv2.rectangle(im, tuple(np.round((np.array([x0, y0]) - o) * f).astype(int)),
                          tuple(np.round((np.array([x1, y1]) - o) * f).astype(int)), (220, 160, 60), 1)
            head = np.full((20, CELL, 3), 235, np.uint8)
            cv2.putText(head, f"[{no}] r{i} w{words[i]} r={R[i]:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1, cv2.LINE_AA)
            box = cv2.copyMakeBorder(np.vstack([head, im]), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(90, 90, 90))
            sets.append(cv2.copyMakeBorder(box, 3, 6, 3, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
            table.append(dict(no=no, zone=f"{ZONE_Y[z // 3]}-{ZONE_X[z % 3]}", row=i, word=int(words[i]),
                              r=round(float(R[i]), 3), u=round(float(U[i]), 3), v=round(float(V[i]), 3),
                              panel=pid, box=int(fbox[i]), work=meta[i]["work"]))
        sets += [np.full_like(sets[0], 255)] * (PER_ZONE - len(sets))
        row = np.hstack(sets)
        nz = int((show & (zone == z)).sum())
        parts += [C.label_bar(row.shape[1], f"zone {ZONE_Y[z // 3]}-{ZONE_X[z % 3]}   n={nz}   "
                                            f"(r {R_SHOW[0]}-{R_SHOW[1]}; blue box = detected face, red = cluster)",
                              28, (0, 0, 160)), row]
    cv2.imwrite(str(out / "figB_zones.png"), np.vstack(parts))
    with open(out / "figB_instances.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    print("->", out / "figB_zones.png", flush=True)


if __name__ == "__main__":
    main()
