"""Stage 1 of the composition-scale angle (2026-11-04): how big are faces, at
tile pixel scale, on each side? No profile axis and no AUC is computed here.

Detector: deepghs/anime_face_detection face_detect_v1.4_s (YOLOv8s ONNX),
distributed threshold 0.307, NMS IoU 0.7, 640 letterbox -- the settings the
face-words track validated on the author's panels (7 missed of 46 faces, no
false positive), knobs untouched. Wrapper copied from
../lineart-face-words/tools/face/h2_face_detect.py.

The detector runs on the WHOLE source picture (author: the koma sub-region
at native scale the tile was cut from; external: the 1024 generated image),
never on the 480 tile -- a close-up tile cuts faces and the detector would
miss them. Tiles are 1:1 crops of those sources on both sides, so a box
height in source pixels is a height in tile pixels.

Per tile: the faces whose box overlaps the tile window by >= 50% of the box
area; the tile's face height is the largest such box height. Tiles with no
such face are counted, never dropped silently.

The column is a guess by an external detector (census owns manuscript
facts), so it is named face_px_est.
"""

import argparse
import collections
import csv
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

R = Path("/home/sh1/deepl/lineart")
REPO, MODEL = "deepghs/anime_face_detection", "face_detect_v1.4_s"
CONF, IOU, SIZE = 0.307, 0.7, 640
TILE = 480
A1_WORKS = ("ako5ver2", "fitness", "gakuen", "hamlabi", "housei")


class FaceDetector:
    def __init__(self):
        path = hf_hub_download(REPO, f"{MODEL}/model.onnx")
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, gray):
        bgr = np.repeat(gray[:, :, None], 3, 2)
        h, w = bgr.shape[:2]
        r = SIZE / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0].T
        conf = out[:, 4:].max(1)
        keep = conf >= CONF
        b, conf = out[keep, :4], conf[keep]
        if not len(b):
            return []
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        idx = cv2.dnn.NMSBoxes([[float(v) for v in (q[0], q[1], q[2] - q[0], q[3] - q[1])] for q in xyxy],
                               [float(c) for c in conf], CONF, IOU)
        res = []
        for k in np.array(idx).reshape(-1):
            q = (xyxy[k] - np.array([left, top, left, top])) / r
            res.append([float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(conf[k])])
        return res


def tiles_a1():
    """tile stem -> (source path, x, y). Name: <work>koma_<idx>_<x>_<y>."""
    srcs = {}
    for w in A1_WORKS:
        d = glob.glob(str(R / f"dataset/regions_{w}_koma_subregions_*_masked"))[0]
        for r in csv.DictReader(open(Path(d) / "manifest.csv")):
            srcs[(w, r["masked_name"].split("_")[1])] = r["masked_line_path"]
    out = {}
    for p in sorted(glob.glob("data/external_standard/norm_author/*.png")):
        stem = Path(p).stem
        head, idx, x, y = stem.split("_")
        out[stem] = (srcs[(head[:-len("koma")], idx)], int(x), int(y))
    return out


def tiles_a2():
    d = R / "dataset/regions_clip_pairs_v3_koma_subregions_20260826_masked"
    srcs = {r["masked_name"].split("_")[1]: r["masked_line_path"] for r in csv.DictReader(open(d / "manifest.csv"))}
    out = {}
    for p in sorted(glob.glob("data/author_digital_a2/norm/*.png")):
        stem = Path(p).stem
        idx, x, y = stem.rsplit("_", 3)[1:]
        out[stem] = (srcs[idx], int(x), int(y))
    return out


def tiles_ext(norm, gen):
    out = {}
    for p in sorted(glob.glob(f"data/external_standard/{norm}/*.png")):
        stem = Path(p).stem
        base, wy, x = stem.rsplit("_", 2)
        out[stem] = (f"data/external_standard/{gen}/{base}.png", int(x), int(wy[1:]))
    return out


# The crop check reads the tile BEFORE the white point (external papers are
# 149-216 and the normalized tile is not a pixel copy of the source).
POOLS = {
    "a1": (tiles_a1, "data/external_standard/norm_author/{}.png"),
    "a2": (tiles_a2, "data/author_digital_a2/tiles/{}.jpg"),
    "e6": (lambda: tiles_ext("norm_animagine_xl_31_seed2", "gen_animagine_xl_31_seed2"),
           "data/external_standard/tiles_animagine_xl_31_seed2/{}.jpg"),
    "i6": (lambda: tiles_ext("norm_illustrious_xl_v01_seed2", "gen_illustrious_xl_v01_seed2"),
           "data/external_standard/tiles_illustrious_xl_v01_seed2/{}.jpg"),
}


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/face_scale_20261104")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = FaceDetector()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    white = len(det(np.full((900, 700), 255, np.uint8)))
    p(f"# 構図のスケール 第1段(顔の大きさ)。検出器 {REPO}/{MODEL} 閾値 {CONF}")
    p(f"道具確認: 白画像の検出 {white}({'合格' if white == 0 else '不合格'})")
    rows = []
    for pool, (fn, norm_dir) in POOLS.items():
        tiles = fn()
        # IC: the tile is the crop it claims to be (first 20 tiles).
        errs = []
        for stem, (src, x, y) in list(tiles.items())[:20]:
            g = load_gray(src).astype(int)
            t = load_gray(norm_dir.format(stem)).astype(int)
            c = g[y:y + TILE, x:x + TILE]
            errs.append(np.abs(c - t).mean() if c.shape == t.shape else 999)
        cache = out / f"boxes_{pool}.json"
        if cache.exists():
            boxes = json.load(open(cache))
        else:
            boxes = {src: det(load_gray(src)) for src in sorted({v[0] for v in tiles.values()})}
            json.dump(boxes, open(cache, "w"))
        heights_src = [b[3] - b[1] for bs in boxes.values() for b in bs]
        n_face_src = sum(1 for bs in boxes.values() if bs)
        tile_h, src_h = [], []
        for stem, (src, x, y) in tiles.items():
            best = 0.0
            for b in boxes[src]:
                ix = max(0.0, min(b[2], x + TILE) - max(b[0], x))
                iy = max(0.0, min(b[3], y + TILE) - max(b[1], y))
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > 0 and ix * iy / area >= 0.5:
                    best = max(best, b[3] - b[1])
            tile_h.append(best)
            hs = [b[3] - b[1] for b in boxes[src]]
            src_h.append(float(np.median(hs)) if hs else 0.0)
            rows.append({"pool": pool, "tile": stem, "source": src, "face_px_est": best if best else "",
                         "source_face_px_est": src_h[-1] if hs else ""})
        th = np.array(tile_h)
        q = lambda a: " / ".join(f"{v:.0f}" for v in np.percentile(a, [25, 50, 75])) if len(a) else "—"
        p(f"\n## {pool}: タイル {len(tiles)} / 元の絵 {len(boxes)}")
        p(f"  IC 切り出しの照合(先頭20枚の平均画素差の最大): {max(errs):.2f}")
        p(f"  顔のある元の絵 {n_face_src}/{len(boxes)}  箱 {len(heights_src)}  箱の高さ px 四分位 {q(heights_src)}")
        sh = np.array(src_h)
        p(f"  元の絵に顔があるタイル {int((sh > 0).sum())}/{len(sh)} ({(sh > 0).mean():.1%})  その元の絵の顔の高さ(中央値)px 四分位 {q(sh[sh > 0])}")
        p(f"  顔が窓に入るタイル {int((th > 0).sum())}/{len(th)} ({(th > 0).mean():.1%})  そのタイルの顔の高さ px 四分位 {q(th[th > 0])}")
    with open(out / "face_px_est.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["pool", "tile", "source", "face_px_est", "source_face_px_est"])
        wr.writeheader()
        wr.writerows(rows)
    (out / "face_scale_stage1.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
