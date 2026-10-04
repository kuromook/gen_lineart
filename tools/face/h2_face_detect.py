#!/usr/bin/env python
"""実験 H2(2026-10-04 事前登録): 既製の顔検出器がコマの線画で使えるかの小規模な試し。

検出器: deepghs/anime_face_detection face_detect_v1.4_s(YOLOv8s ONNX)。閾値 0.307(配布元既定)、
NMS IoU 0.7、入力長辺 640(レターボックス)。入力はコマの線画 PNG(正立)。
対象: seed 20261004 の無作為 40 コマ + 陽性対照 p1874・p4549。
道具確認: 真っ白な画像で検出 0 個(PNG とストロークの向きの照合は work_log 2026-10-04 で済み)。
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_eye_sizes as C                                           # noqa: E402

DATASET = Path("/home/sh1/deepl/lineart/dataset")
REPO, MODEL = "deepghs/anime_face_detection", "face_detect_v1.4_s"
CONF, IOU, SIZE = 0.307, 0.7, 640
CONTROLS = [1874, 4549]
CELL = 400
PER_LINE = 6


class FaceDetector:
    def __init__(self):
        path = hf_hub_download(REPO, f"{MODEL}/model.onnx")
        self.sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, bgr):
        """返る: [(x0, y0, x1, y1, conf)](元画像の座標)"""
        h, w = bgr.shape[:2]
        r = SIZE / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]                 # (4 + nc, N)
        out = out.T
        conf = out[:, 4:].max(1)
        keep = conf >= CONF
        b, conf = out[keep, :4], conf[keep]
        if not len(b):
            return []
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        idx = cv2.dnn.NMSBoxes([[float(v) for v in (q[0], q[1], q[2] - q[0], q[3] - q[1])] for q in xyxy],
                               [float(c) for c in conf], CONF, IOU)
        res = []
        for k in np.array(idx).reshape(-1):
            q = (xyxy[k] - np.array([left, top, left, top])) / r
            res.append((float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(conf[k])))
        return res


def panel_png(row):
    return DATASET / row["source"] / "line" / row["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h2_face_detect_smoke_20261004")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()
    out = C.ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(C.PACK / "panels.csv")))
    rng = np.random.default_rng(a.seed)
    pids = CONTROLS + sorted(rng.choice(len(rows), a.n, replace=False).tolist())

    det = FaceDetector()
    n_white = len(det(np.full((900, 700, 3), 255, np.uint8)))
    json.dump(dict(white_image_detections=n_white, passed=n_white == 0, model=f"{REPO}/{MODEL}",
                   conf=CONF, iou=IOU, size=SIZE, providers=det.sess.get_providers()),
              open(out / "check.json", "w"), indent=1)
    print("道具確認(白画像の検出数):", n_white, "合格" if n_white == 0 else "不合格", det.sess.get_providers(), flush=True)
    if n_white:
        return

    cells, table = [], []
    for no, pid in enumerate(pids, 1):
        img = cv2.imread(str(panel_png(rows[pid])))
        boxes = det(img)
        f = CELL / max(img.shape[:2])
        sm = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cell = np.full((CELL, CELL, 3), 255, np.uint8)
        oy, ox = (CELL - sm.shape[0]) // 2, (CELL - sm.shape[1]) // 2
        cell[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] = sm
        cv2.rectangle(cell, (ox, oy), (ox + sm.shape[1] - 1, oy + sm.shape[0] - 1), (230, 200, 160), 1)
        for k, (x0, y0, x1, y1, c) in enumerate(boxes, 1):
            p0 = (int(x0 * f) + ox, int(y0 * f) + oy); p1 = (int(x1 * f) + ox, int(y1 * f) + oy)
            cv2.rectangle(cell, p0, p1, (0, 0, 230), 2)
            cv2.putText(cell, f"{c:.2f}", (p0[0] + 2, max(p0[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 230), 1, cv2.LINE_AA)
            table.append(dict(no=no, panel=pid, box=k, conf=round(c, 3), x0=round(x0), y0=round(y0),
                              x1=round(x1), y1=round(y1), name=rows[pid]["name"]))
        if not boxes:
            table.append(dict(no=no, panel=pid, box=0, conf="", x0="", y0="", x1="", y1="", name=rows[pid]["name"]))
        head = np.full((22, CELL, 3), 235, np.uint8)
        tag = " (control)" if pid in CONTROLS else ""
        cv2.putText(head, f"[{no}] p{pid}{tag}  faces={len(boxes)}  {rows[pid]['name'][11:38]}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        box = cv2.copyMakeBorder(np.vstack([head, cell]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(90, 90, 90))
        cells.append(cv2.copyMakeBorder(box, 5, 8, 5, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    cells += [np.full_like(cells[0], 255)] * (-len(cells) % PER_LINE)
    m = np.vstack([np.hstack(cells[j:j + PER_LINE]) for j in range(0, len(cells), PER_LINE)])
    cv2.imwrite(str(out / "montage.png"), m)
    with open(out / "detections.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    nb = sum(1 for t in table if t["box"])
    print(f"コマ {len(pids)}  検出枠 {nb}  枠なしのコマ {sum(1 for t in table if not t['box'])}", flush=True)
    print("->", out / "montage.png", flush=True)


if __name__ == "__main__":
    main()
