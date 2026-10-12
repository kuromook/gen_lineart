#!/usr/bin/env python
"""実験 H9a(2026-10-12 事前登録): 既製の頭の検出器が線画で使えるかの小さな試し。学習用の顔のコマだけ。

出力: heads.csv、summary.json、montage.png
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h2_face_detect as H2                                            # noqa: E402

REPO, MODEL = "deepghs/anime_head_detection", "head_detect_v2.0_s"
MAX_ORDER = 104


class HeadDetector(H2.FaceDetector):
    def __init__(self):
        self.sess = ort.InferenceSession(hf_hub_download(REPO, f"{MODEL}/model.onnx"),
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name


def match(face, heads):
    x0, y0, x1, y1 = face
    cx, cy, area = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * (y1 - y0)
    best = None
    for h in heads:
        iw = max(0, min(x1, h[2]) - max(x0, h[0])); ih = max(0, min(y1, h[3]) - max(y0, h[1]))
        if h[0] <= cx <= h[2] and h[1] <= cy <= h[3] and iw * ih >= 0.8 * area:
            if best is None or (h[2] - h[0]) * (h[3] - h[1]) < (best[2] - best[0]) * (best[3] - best[1]):
                best = h
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h9a_head_detect_20261012")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    try:
        thr = json.load(open(hf_hub_download(REPO, f"{MODEL}/threshold.json")))["threshold"]
    except Exception:
        thr = 0.3
    H2.CONF = float(thr)
    det = HeadDetector()
    blank = det(np.full((900, 700, 3), 255, np.uint8))
    print("閾値", thr, "真っ白な画像の検出数", len(blank))
    if blank:
        print("道具確認: 不合格"); return
    G.load_state(G.LAB / "face_groups.jsonl")
    recs = [r for o, r in sorted(G.S["faces"].items()) if o <= MAX_ORDER and r["usage"] == "train" and r["status"] == "done"]
    rows, cache = [], {}
    for r in recs:
        q = G.S["queue"][r["order"]]; pid = int(q["panel"])
        if pid not in cache:
            cache[pid] = det(G.panel_img(pid))
        face = tuple(float(q[k]) for k in ("x0", "y0", "x1", "y1"))
        h = match(face, cache[pid])
        rows.append(dict(order=r["order"], panel=pid, fx0=face[0], fy0=face[1], fx1=face[2], fy1=face[3], n_heads=len(cache[pid]),
                         hx0=h[0] if h else "", hy0=h[1] if h else "", hx1=h[2] if h else "", hy1=h[3] if h else "", hconf=round(h[4], 3) if h else ""))
    with open(out / "heads.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    hit = [t for t in rows if t["hx0"] != ""]
    ratio = [np.sqrt((t["hx1"] - t["hx0"]) * (t["hy1"] - t["hy0"]) / ((t["fx1"] - t["fx0"]) * (t["fy1"] - t["fy0"]))) for t in hit]
    S = dict(threshold=thr, n_faces=len(rows), matched=len(hit), matched_rate=round(len(hit) / len(rows), 3),
             side_ratio_p10_p50_p90=[round(float(v), 2) for v in np.percentile(ratio, [10, 50, 90])] if ratio else None)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(S, ensure_ascii=False))

    rng = np.random.default_rng(20261012)
    pick = sorted(rng.choice(len(rows), min(24, len(rows)), replace=False).tolist())
    CELL, tiles = 300, []
    for k, j in enumerate(pick):
        t = rows[j]; img = G.panel_img(t["panel"]).copy()
        xs = [t["fx0"], t["fx1"]] + ([t["hx0"], t["hx1"]] if t["hx0"] != "" else []); ys = [t["fy0"], t["fy1"]] + ([t["hy0"], t["hy1"]] if t["hy0"] != "" else [])
        side = 1.5 * max(max(xs) - min(xs), max(ys) - min(ys)); cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        f = CELL / side
        A = np.array([[f, 0, CELL / 2 - cx * f], [0, f, CELL / 2 - cy * f]])
        cell = cv2.warpAffine(img, A, (CELL, CELL), flags=cv2.INTER_AREA, borderValue=(236, 236, 236))
        def box(b, col, th):
            p = (np.array([[b[0], b[1]], [b[2], b[3]]]) @ A[:, :2].T + A[:, 2]).round().astype(int)
            cv2.rectangle(cell, tuple(p[0]), tuple(p[1]), col, th, cv2.LINE_AA)
        box((t["fx0"], t["fy0"], t["fx1"], t["fy1"]), (222, 141, 60), 2)
        if t["hx0"] != "":
            box((t["hx0"], t["hy0"], t["hx1"], t["hy1"]), (30, 130, 235), 2)
        head = np.full((24, CELL, 3), 255, np.uint8)
        cv2.putText(head, f"#{k + 1}  face {t['order'] + 1}  p{t['panel']}" + ("" if t["hx0"] != "" else "  NO HEAD"), (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(np.vstack([head, cell]), 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    g = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    title = np.full((30, g.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "blue = face box (face detector), orange = head box (head detector). 24 random train faces", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "montage.png"), np.vstack([title, g])); print("->", out / "montage.png")


if __name__ == "__main__":
    main()
