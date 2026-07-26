"""Find partially corresponding regions among all ako5 sketch/line pages.

Each page pair may yield multiple local similarity transforms. SIFT matches are
peeled into RANSAC-supported transform clusters, then each cluster is evaluated
over a padded region instead of only at its matched keypoints. The output is a
ranked candidate list for visual review; it does not modify the training set.
"""
import argparse
import csv
import io
import json
import math
import os
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage


ZIP_PATH = os.path.expanduser("~/dataset_ako5.zip")
CSV_OUT = "results/ako5_region_matches.csv"
JSON_OUT = "results/ako5_region_matches.json"
QC_OUT = "results/ako5_region_match_qc.png"


def page_id(filename):
    return filename.split("_")[1]


def load_gray(zf, filename, max_dim):
    raw = zf.read(f"dataset_ako5/{filename}")
    image = Image.open(io.BytesIO(raw)).convert("L")
    resize_scale = min(1.0, max_dim / max(image.size))
    size = tuple(max(1, round(v * resize_scale)) for v in image.size)
    gray = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
    return gray, resize_scale


def available_manifest_entries(zf, manifest):
    names = set(zf.namelist())
    rows = []
    missing = 0
    for entry in manifest:
        sketch_path = f"dataset_ako5/{entry['sketch']}"
        line_path = f"dataset_ako5/{entry['line']}"
        if sketch_path in names and line_path in names:
            rows.append(entry)
        else:
            missing += 1
    if missing:
        print(f"skip missing manifest rows: {missing}", flush=True)
    return rows


def prepare(gray, sift):
    ac = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=0))
    blur = cv2.GaussianBlur(ac, (0, 0), 1.0)
    edges = cv2.Canny(blur, 45, 135) > 0
    feature = cv2.GaussianBlur(edges.astype(np.uint8) * 255, (0, 0), 1.5)
    keypoints, descriptors = sift.detectAndCompute(feature, None)
    return {
        "gray": ac,
        "edges": edges,
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def ratio_matches(a, b, matcher, ratio):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return []
    return [m for m, n in matcher.knnMatch(a, b, k=2) if m.distance < ratio * n.distance]


def transform_properties(matrix):
    a, b = matrix[0, 0], matrix[0, 1]
    return float(math.hypot(a, b)), float(math.degrees(math.atan2(-b, a)))


def full_resolution_geometry(matrix, bbox, sketch_scale, line_scale):
    full_matrix = matrix.copy()
    full_matrix[:, :2] *= sketch_scale / line_scale
    full_matrix[:, 2] /= line_scale
    full_bbox = tuple(int(round(value / line_scale)) for value in bbox)
    return full_matrix.tolist(), full_bbox


def estimate_models(sketch, line, matcher, args):
    matches = ratio_matches(sketch["descriptors"], line["descriptors"], matcher, args.ratio)
    if len(matches) < args.min_matches:
        return []

    src_all = np.float32([sketch["keypoints"][m.queryIdx].pt for m in matches])
    dst_all = np.float32([line["keypoints"][m.trainIdx].pt for m in matches])
    remaining = np.arange(len(matches))
    models = []

    for _ in range(args.max_models):
        if len(remaining) < args.min_matches:
            break
        matrix, mask = cv2.estimateAffinePartial2D(
            src_all[remaining], dst_all[remaining], method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_px, maxIters=5000,
            confidence=0.999, refineIters=30,
        )
        if matrix is None or mask is None:
            break
        local_inliers = mask.ravel().astype(bool)
        inlier_indices = remaining[local_inliers]
        if len(inlier_indices) < args.min_inliers:
            break

        scale, angle = transform_properties(matrix)
        if args.scale_min <= scale <= args.scale_max and abs(angle) <= args.angle_max:
            models.append({
                "matrix": matrix,
                "match_count": len(matches),
                "inlier_count": len(inlier_indices),
                "src_inliers": src_all[inlier_indices],
                "dst_inliers": dst_all[inlier_indices],
                "scale": scale,
                "angle": angle,
            })

        # Remove this consensus set. Further iterations can discover a different
        # local transform elsewhere in the same page pair.
        remaining = remaining[~local_inliers]
    return models


def bbox_from_points(points, shape, pad, min_side):
    h, w = shape
    x0, y0 = np.floor(points.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(points.max(axis=0) + pad).astype(int)
    cx, cy = points.mean(axis=0)
    x0, x1 = min(x0, int(cx - min_side / 2)), max(x1, int(cx + min_side / 2))
    y0, y1 = min(y0, int(cy - min_side / 2)), max(y1, int(cy + min_side / 2))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def local_bboxes(points, shape, sides, min_inliers):
    """Generate fixed-size local windows supported by several inlier points."""
    h, w = shape
    boxes = []
    for side in sides:
        # Quantized centers avoid evaluating nearly identical windows around
        # every keypoint while still covering separate characters/panels.
        step = max(side // 3, 1)
        centers = {(round(x / step) * step, round(y / step) * step) for x, y in points}
        for cx, cy in centers:
            x0 = int(np.clip(cx - side / 2, 0, max(w - side, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(h - side, 0)))
            x1, y1 = min(w, x0 + side), min(h, y0 + side)
            inside = (
                (points[:, 0] >= x0) & (points[:, 0] < x1)
                & (points[:, 1] >= y0) & (points[:, 1] < y1)
            )
            if inside.sum() >= min_inliers:
                boxes.append(((x0, y0, x1, y1), inside))
    return boxes


def spatial_coverage(points, bbox, grid=4):
    x0, y0, x1, y1 = bbox
    if len(points) == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    gx = np.clip(((points[:, 0] - x0) / (x1 - x0) * grid).astype(int), 0, grid - 1)
    gy = np.clip(((points[:, 1] - y0) / (y1 - y0) * grid).astype(int), 0, grid - 1)
    return len(set(zip(gy.tolist(), gx.tolist()))) / (grid * grid)


def orientation_entropy(edges):
    if edges.sum() < 20:
        return 0.0
    image = edges.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    valid = magnitude > 0.25
    if valid.sum() < 20:
        return 0.0
    angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi), weights=magnitude[valid])
    probs = hist / max(hist.sum(), 1e-9)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / np.log(12))


def evaluate_model(sketch, line, model, args, bbox=None):
    h, w = line["edges"].shape
    if bbox is None:
        bbox = bbox_from_points(model["dst_inliers"], (h, w), args.region_pad, args.min_region_side)
    x0, y0, x1, y1 = bbox
    if x1 - x0 < args.min_region_side or y1 - y0 < args.min_region_side:
        return None

    warped = cv2.warpAffine(
        sketch["edges"].astype(np.uint8), model["matrix"], (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(bool)
    support = cv2.warpAffine(
        np.ones(sketch["edges"].shape, np.uint8), model["matrix"], (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(bool)
    rough_edge = warped[y0:y1, x0:x1]
    line_edge = line["edges"][y0:y1, x0:x1] & support[y0:y1, x0:x1]
    if rough_edge.sum() < args.min_edge_pixels or line_edge.sum() < args.min_edge_pixels:
        return None

    d_to_rough = ndimage.distance_transform_edt(~rough_edge)
    d_to_line = ndimage.distance_transform_edt(~line_edge)
    line_dist = d_to_rough[line_edge]
    rough_dist = d_to_line[rough_edge]
    precision = float((rough_dist <= args.close_px).mean())
    recall = float((line_dist <= args.close_px).mean())
    edge_f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    chamfer = float((np.minimum(line_dist, args.truncate_px).mean()
                     + np.minimum(rough_dist, args.truncate_px).mean()) / 2)
    coverage = spatial_coverage(model["dst_inliers"], bbox)
    entropy = min(orientation_entropy(rough_edge), orientation_entropy(line_edge))
    area_ratio = ((x1 - x0) * (y1 - y0)) / (h * w)

    # Higher is better. Coverage and entropy suppress accidental matches on one
    # isolated curve or panel border while F1/chamfer measure local agreement.
    score = (
        5.0 * edge_f1
        - 0.20 * chamfer
        + 1.5 * coverage
        + 0.8 * entropy
        + 0.12 * math.log1p(model["inlier_count"])
    )
    return {
        **model,
        "bbox": bbox,
        "coverage": coverage,
        "orientation_entropy": entropy,
        "area_ratio": area_ratio,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": edge_f1,
        "chamfer": chamfer,
        "score": score,
        "warped_edges": warped,
    }


def overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    intersection = iw * ih
    return intersection / max(min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)), 1)


def suppress_duplicates(rows, threshold=0.65):
    kept = []
    for row in sorted(rows, key=lambda item: item["score"], reverse=True):
        duplicate = any(
            row["sketch_page"] == other["sketch_page"]
            and row["line_page"] == other["line_page"]
            and overlap_ratio(row["bbox"], other["bbox"]) >= threshold
            and abs(row["scale"] - other["scale"]) < 0.03
            and abs(row["angle"] - other["angle"]) < 1.0
            for other in kept
        )
        if not duplicate:
            kept.append(row)
    return kept


def serializable(row):
    out = {k: v for k, v in row.items()
           if k not in {"matrix", "src_inliers", "dst_inliers", "warped_edges"}}
    for key, value in list(out.items()):
        if isinstance(value, np.generic):
            out[key] = value.item()
    out["bbox"] = [int(value) for value in row["bbox"]]
    out["matrix"] = row["matrix"].tolist()
    return out


def make_qc(rows, sketches, lines, path, count):
    picks = rows[:count]
    if not picks:
        return
    thumb, label_h = 260, 32
    canvas = Image.new("RGB", (thumb * 3, (thumb + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, row in enumerate(picks):
        y = index * (thumb + label_h)
        line = lines[row["line_page"]]
        sketch = sketches[row["sketch_page"]]
        h, w = line["gray"].shape
        aligned = cv2.warpAffine(sketch["gray"], row["matrix"], (w, h), borderValue=255)
        x0, y0, x1, y1 = row["bbox"]
        overlay = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
        overlay[row["warped_edges"][y0:y1, x0:x1]] = (255, 60, 60)
        overlay[line["edges"][y0:y1, x0:x1]] = (50, 80, 255)
        images = (aligned[y0:y1, x0:x1], line["gray"][y0:y1, x0:x1], overlay)
        for column, image in enumerate(images):
            pil = Image.fromarray(image).convert("RGB")
            pil.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(pil, (column * thumb, y))
        text = (
            f'{row["sketch_page"]}->{row["line_page"]} score={row["score"]:.2f} '
            f'F1={row["edge_f1"]:.2f} cham={row["chamfer"]:.1f} '
            f'in={row["inlier_count"]} cov={row["coverage"]:.2f} '
            f's={row["scale"]:.3f} a={row["angle"]:.1f}'
        )
        draw.text((3, y + thumb + 2), text, fill="black", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    parser.add_argument("--max-dim", type=int, default=1200)
    parser.add_argument("--sketch-limit", type=int, default=0)
    parser.add_argument("--line-limit", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument("--ransac-px", type=float, default=5.0)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.35)
    parser.add_argument("--angle-max", type=float, default=8.0)
    parser.add_argument("--region-pad", type=int, default=80)
    parser.add_argument("--min-region-side", type=int, default=180)
    parser.add_argument("--region-sides", default="240,360",
                        help="comma-separated local evaluation window sizes at max-dim resolution")
    parser.add_argument("--min-region-inliers", type=int, default=5)
    parser.add_argument("--min-edge-pixels", type=int, default=120)
    parser.add_argument("--close-px", type=float, default=4.0)
    parser.add_argument("--truncate-px", type=float, default=20.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--qc-count", type=int, default=40)
    parser.add_argument("--csv-out", default=CSV_OUT)
    parser.add_argument("--json-out", default=JSON_OUT)
    parser.add_argument("--qc-out", default=QC_OUT)
    args = parser.parse_args()
    region_sides = [int(value) for value in args.region_sides.split(",") if value.strip()]

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015, edgeThreshold=15)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    sketches, lines = {}, {}
    with zipfile.ZipFile(args.zip_path) as zf:
        manifest = json.loads(zf.read("dataset_ako5/manifest.json"))
        manifest = available_manifest_entries(zf, manifest)
        sketch_files = {page_id(entry["sketch"]): entry["sketch"] for entry in manifest}
        line_files = {page_id(entry["line"]): entry["line"] for entry in manifest}
        selected_sketches = sorted(sketch_files)[:args.sketch_limit or None]
        selected_lines = sorted(line_files)[:args.line_limit or None]
        for sid in selected_sketches:
            if sid not in sketches:
                gray, resize_scale = load_gray(zf, sketch_files[sid], args.max_dim)
                sketches[sid] = prepare(gray, sift)
                sketches[sid]["resize_scale"] = resize_scale
        for lid in selected_lines:
            if lid not in lines:
                gray, resize_scale = load_gray(zf, line_files[lid], args.max_dim)
                lines[lid] = prepare(gray, sift)
                lines[lid]["resize_scale"] = resize_scale

    sketch_ids = sorted(sketches)
    line_ids = sorted(lines)
    rows = []
    total_pairs = len(sketch_ids) * len(line_ids)
    done = 0
    for sid in sketch_ids:
        for lid in line_ids:
            done += 1
            models = estimate_models(sketches[sid], lines[lid], matcher, args)
            for model_index, model in enumerate(models, 1):
                boxes = local_bboxes(
                    model["dst_inliers"], lines[lid]["edges"].shape,
                    region_sides, args.min_region_inliers,
                )
                for region_index, (bbox, inside) in enumerate(boxes, 1):
                    local_model = {
                        **model,
                        "inlier_count": int(inside.sum()),
                        "src_inliers": model["src_inliers"][inside],
                        "dst_inliers": model["dst_inliers"][inside],
                    }
                    row = evaluate_model(sketches[sid], lines[lid], local_model, args, bbox)
                    if row is not None and row["score"] >= args.min_score:
                        full_matrix, full_bbox = full_resolution_geometry(
                            row["matrix"], row["bbox"],
                            sketches[sid]["resize_scale"], lines[lid]["resize_scale"],
                        )
                        row.update({
                            "sketch_page": sid,
                            "line_page": lid,
                            "model_index": model_index,
                            "region_index": region_index,
                            "full_matrix": full_matrix,
                            "bbox_full": full_bbox,
                            "decision": "",
                            "notes": "",
                        })
                        rows.append(row)
        print(f"[{done:04d}/{total_pairs:04d}] {sid}: candidates={len(rows)}", flush=True)

    rows = suppress_duplicates(rows)
    rows.sort(key=lambda item: item["score"], reverse=True)
    rows = rows[:args.top_k]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "rank", "sketch_page", "line_page", "model_index", "region_index", "score", "edge_f1",
        "edge_precision", "edge_recall", "chamfer", "inlier_count", "match_count",
        "coverage", "orientation_entropy", "area_ratio", "scale", "angle", "bbox",
        "bbox_full", "full_matrix", "decision", "notes",
    ]
    with open(args.csv_out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(args.json_out, "w") as file:
        json.dump([serializable(row) for row in rows], file, indent=2)
    make_qc(rows, sketches, lines, args.qc_out, args.qc_count)
    print(f"Wrote {len(rows)} candidates: {args.csv_out}, {args.json_out}, {args.qc_out}")


if __name__ == "__main__":
    main()
