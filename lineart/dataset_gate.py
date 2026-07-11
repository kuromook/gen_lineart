"""Train and run a lightweight dataset-style gate for rough images.

The first gate is intentionally simple: handcrafted image statistics plus a
nearest-centroid classifier. It routes dataset/source style separately from
future content categories such as bust, fullbody, scenery, or objects.
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 480
FEATURE_SIZE = 160
DEFAULT_ROUGH_DIR = "dataset/pairs_480/train/rough"
DEFAULT_MODEL_OUT = "results/dataset_gate_centroids.json"
DEFAULT_REPORT_OUT = "results/dataset_gate_report.json"
DEFAULT_LABELS = ("ako5", "housei", "kurip", "lineart")


def label_from_name(name):
    if name.startswith("ako5_"):
        return "ako5"
    if name.startswith("housei_"):
        return "housei"
    if name.startswith("kurip_"):
        return "kurip"
    if name.startswith("lineart_"):
        return "lineart"
    return None


def load_gray(path, autocontrast=False, size=IMAGE_SIZE):
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return np.asarray(image, dtype=np.uint8)


def safe_entropy(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    prob = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum() / math.log(bins))


def image_features(gray):
    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    edges = cv2.Canny(blur, 45, 135) > 0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    valid = grad > 0.02
    if valid.any():
        angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
        orient_hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=grad[valid])
        orient_hist = orient_hist.astype(np.float64) / max(float(orient_hist.sum()), 1.0)
    else:
        orient_hist = np.zeros(12, dtype=np.float64)

    spectrum = np.fft.fftshift(np.fft.fft2(gray_f - gray_f.mean()))
    power = np.abs(spectrum) ** 2
    h, w = power.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    max_radius = math.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)
    low = power[radius < max_radius * 0.10].sum()
    mid = power[(radius >= max_radius * 0.10) & (radius < max_radius * 0.25)].sum()
    high = power[radius >= max_radius * 0.25].sum()
    total_power = max(float(low + mid + high), 1e-9)

    dark = gray < 128
    midtone = (gray >= 128) & (gray < 240)
    features = [
        float(gray_f.mean()),
        float(gray_f.std()),
        float(np.percentile(gray_f, 1)),
        float(np.percentile(gray_f, 5)),
        float(np.percentile(gray_f, 50)),
        float(np.percentile(gray_f, 95)),
        float(dark.mean()),
        float(midtone.mean()),
        float(edges.mean()),
        float(grad.mean()),
        float(np.percentile(grad, 95)),
        safe_entropy(gray.reshape(-1), 32, (0, 256)),
        float(low / total_power),
        float(mid / total_power),
        float(high / total_power),
    ]
    features.extend(orient_hist.tolist())
    return np.asarray(features, dtype=np.float64)


def iter_labeled_paths(rough_dir, labels, max_per_label=None):
    grouped = {label: [] for label in labels}
    for path in sorted(Path(rough_dir).glob("*.jpg")):
        label = label_from_name(path.name)
        if label in grouped:
            grouped[label].append(path)
    for label, paths in grouped.items():
        if max_per_label:
            paths = paths[:max_per_label]
        for path in paths:
            yield label, path


def train_gate(args):
    labels = list(args.labels)
    rows = []
    for label, path in iter_labeled_paths(args.rough_dir, labels, args.max_per_label):
        rows.append((label, str(path), image_features(load_gray(path, args.autocontrast, FEATURE_SIZE))))
    if not rows:
        raise RuntimeError("No labeled rough images found")

    x = np.stack([row[2] for row in rows])
    y = np.asarray([labels.index(row[0]) for row in rows], dtype=np.int64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std

    centroids = []
    counts = {}
    for index, label in enumerate(labels):
        subset = z[y == index]
        counts[label] = int(len(subset))
        if len(subset) == 0:
            raise RuntimeError(f"No samples for label: {label}")
        centroids.append(subset.mean(axis=0))
    centroids = np.stack(centroids)

    distances = ((z[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    pred = distances.argmin(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_index, pred_index in zip(y, pred):
        confusion[true_index, pred_index] += 1
    accuracy = float((pred == y).mean())

    model = {
        "labels": labels,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "centroids": centroids.tolist(),
        "autocontrast": bool(args.autocontrast),
        "counts": counts,
    }
    report = {
        "accuracy": accuracy,
        "counts": counts,
        "labels": labels,
        "confusion": confusion.tolist(),
        "note": "Training-set accuracy for a lightweight dataset/source-style gate.",
    }
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    Path(args.model_out).write_text(json.dumps(model, indent=2) + "\n")
    Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"samples={len(rows)} accuracy={accuracy:.4f}")
    print(f"saved model: {args.model_out}")
    print(f"saved report: {args.report_out}")


def predict(args):
    model = json.loads(Path(args.model).read_text())
    labels = model["labels"]
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    centroids = np.asarray(model["centroids"], dtype=np.float64)
    features = image_features(load_gray(args.input, bool(model.get("autocontrast", False)), FEATURE_SIZE))
    z = (features - mean) / std
    distances = ((centroids - z[None, :]) ** 2).mean(axis=1)
    order = np.argsort(distances)
    top = [{"label": labels[i], "distance": float(distances[i])} for i in order[: args.top_k]]
    print(json.dumps({"input": args.input, "predicted_label": top[0]["label"], "top": top}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--rough-dir", default=DEFAULT_ROUGH_DIR)
    train_parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    train_parser.add_argument("--max-per-label", type=int, default=1000)
    train_parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT)
    train_parser.add_argument("--report-out", default=DEFAULT_REPORT_OUT)
    train_parser.add_argument("--autocontrast", action="store_true")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", default=DEFAULT_MODEL_OUT)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    if args.command == "train":
        train_gate(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
