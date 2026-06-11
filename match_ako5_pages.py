"""Rank candidate ako5 sketch/line page pairs and estimate similarity alignment.

The manifest pairing is deliberately not trusted. Every sketch is matched against
every line page using SIFT + RANSAC, then the strongest candidates are reranked
with edge-distance statistics after alignment.

Outputs:
  results/ako5_page_matches.csv   one row per retained candidate
  results/ako5_page_matches.json  best candidates grouped by sketch page
  results/ako5_page_match_qc.png  best-match alignment montage
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
CSV_OUT = "results/ako5_page_matches.csv"
JSON_OUT = "results/ako5_page_matches.json"
QC_OUT = "results/ako5_page_match_qc.png"


def page_id(filename):
    return filename.split("_")[1]


def load_gray(zf, filename, max_dim):
    raw = zf.read(f"dataset_ako5/{filename}")
    img = Image.open(io.BytesIO(raw)).convert("L")
    scale = max_dim / max(img.size)
    size = tuple(max(1, round(v * scale)) for v in img.size)
    return np.asarray(img.resize(size, Image.Resampling.LANCZOS)), scale


def prepare(gray, sift):
    """Return contrast-normalized image, edge map, keypoints and descriptors."""
    ac = np.asarray(ImageOps.autocontrast(Image.fromarray(gray), cutoff=0))
    blur = cv2.GaussianBlur(ac, (0, 0), 1.0)
    edges = cv2.Canny(blur, 40, 120)
    # Blurred edges reduce the appearance gap between faint sketch and clean ink.
    feature = cv2.GaussianBlur(edges, (0, 0), 2.0)
    keypoints, desc = sift.detectAndCompute(feature, None)
    return {
        "gray": ac, "edges": edges > 0, "feature": feature.astype(np.float32) / 255.0,
        "keypoints": keypoints, "desc": desc,
    }


def grid_candidate(sketch, line, scales, angles):
    """Exhaustive scale/angle search; phase correlation supplies translation."""
    h, w = line["feature"].shape
    src = cv2.resize(sketch["feature"], (w, h), interpolation=cv2.INTER_AREA)
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    best = None
    for scale in scales:
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            warped = cv2.warpAffine(src, matrix, (w, h), borderValue=0)
            shift, response = cv2.phaseCorrelate(warped, line["feature"], window)
            matrix[:, 2] += shift
            if best is None or response > best["rank_score"]:
                best = {
                    "matrix": matrix, "matches": 0, "inliers": 0,
                    "inlier_ratio": 0.0, "coverage": 0.0,
                    "scale": float(scale), "angle": float(angle),
                    "rank_score": float(response), "method": "grid",
                }
    return best


def ratio_matches(desc_a, desc_b, matcher, ratio):
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    return [a for a, b in matcher.knnMatch(desc_a, desc_b, k=2)
            if a.distance < ratio * b.distance]


def spatial_coverage(points, shape, grid=4):
    if len(points) == 0:
        return 0.0
    h, w = shape
    x = np.clip((points[:, 0] / max(w, 1) * grid).astype(int), 0, grid - 1)
    y = np.clip((points[:, 1] / max(h, 1) * grid).astype(int), 0, grid - 1)
    return len(set(zip(y.tolist(), x.tolist()))) / (grid * grid)


def estimate_candidate(sketch, line, matcher, ratio, ransac_px, min_inliers,
                       scale_min, scale_max, angle_max):
    matches = ratio_matches(sketch["desc"], line["desc"], matcher, ratio)
    if len(matches) < 4:
        return None
    src = np.float32([sketch["keypoints"][m.queryIdx].pt for m in matches])
    dst = np.float32([line["keypoints"][m.trainIdx].pt for m in matches])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
        maxIters=3000, confidence=0.995, refineIters=20,
    )
    if matrix is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        return None
    a, b = matrix[0, 0], matrix[0, 1]
    scale = float(math.hypot(a, b))
    angle = float(math.degrees(math.atan2(-b, a)))
    if not scale_min <= scale <= scale_max or abs(angle) > angle_max:
        return None
    coverage = spatial_coverage(dst[inlier_mask], line["edges"].shape)
    inlier_ratio = inliers / len(matches)
    # Stage-one score rewards numerous, consistent, spatially distributed matches.
    rank_score = inliers * inlier_ratio * (0.25 + coverage)
    return {
        "matrix": matrix, "matches": len(matches), "inliers": inliers,
        "inlier_ratio": inlier_ratio, "coverage": coverage,
        "scale": scale, "angle": angle, "rank_score": rank_score, "method": "sift",
    }


def edge_statistics(sketch, line, candidate):
    h, w = line["edges"].shape
    matrix = candidate["matrix"]
    warped = cv2.warpAffine(
        sketch["edges"].astype(np.uint8), matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(bool)
    support = cv2.warpAffine(
        np.ones(sketch["edges"].shape, np.uint8), matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(bool)
    line_edge = line["edges"] & support
    if warped.sum() < 10 or line_edge.sum() < 10:
        return None
    d_to_sketch = ndimage.distance_transform_edt(~warped)
    d_to_line = ndimage.distance_transform_edt(~line["edges"])
    l2s = d_to_sketch[line_edge]
    s2l = d_to_line[warped]
    l2s_median = float(np.median(l2s))
    l2s_p90 = float(np.percentile(l2s, 90))
    s2l_median = float(np.median(s2l))
    overlap = float(line_edge.sum() / max(line["edges"].sum(), 1))
    close_ratio = float((l2s <= 4).mean())
    # Lower is better. Penalize one-sided/cropped matches and poor edge overlap.
    edge_score = (l2s_median + 0.3 * s2l_median + 0.15 * l2s_p90
                  + 8.0 * (1.0 - close_ratio) + 3.0 * (1.0 - overlap))
    return {
        "line_to_sketch_median": l2s_median,
        "line_to_sketch_p90": l2s_p90,
        "sketch_to_line_median": s2l_median,
        "overlap_ratio": overlap, "close_ratio": close_ratio,
        "edge_score": edge_score, "warped_edges": warped,
    }


def selection_score(candidate):
    """Combined score for ranking; lower is better."""
    if candidate["method"] == "sift":
        evidence_bonus = 0.30 * candidate["inliers"] + candidate["coverage"]
    else:
        evidence_bonus = 5.0 * candidate["rank_score"]
    return candidate["edge_score"] - evidence_bonus


def serializable(row):
    out = {k: v for k, v in row.items() if k not in {"matrix", "warped_edges", "full_matrix"}}
    out["full_matrix"] = row["full_matrix"].tolist()
    return out


def make_qc(best_rows, sketches, lines, path, max_rows=16):
    picks = best_rows[:max_rows]
    if not picks:
        return
    thumb_w, thumb_h, label_h = 180, 255, 32
    canvas = Image.new("RGB", (thumb_w * 3, (thumb_h + label_h) * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, row in enumerate(picks):
        y = i * (thumb_h + label_h)
        sketch = sketches[row["sketch_page"]]
        line = lines[row["line_page"]]
        h, w = line["gray"].shape
        aligned = cv2.warpAffine(sketch["gray"], row["matrix"], (w, h), borderValue=255)
        overlay = np.full((h, w, 3), 255, np.uint8)
        overlay[row["warped_edges"]] = (255, 40, 40)
        overlay[line["edges"]] = (40, 80, 255)
        for j, image in enumerate((aligned, line["gray"], overlay)):
            pil = Image.fromarray(image).convert("RGB")
            pil.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(pil, (j * thumb_w, y))
        text = (f'{row["sketch_page"]}->{row["line_page"]} rank={row["candidate_rank"]} '
                f'edge={row["edge_score"]:.1f} in={row["inliers"]}/{row["matches"]} '
                f's={row["scale"]:.3f} a={row["angle"]:.2f} margin={row["margin"]:.1f}')
        draw.text((3, y + thumb_h + 2), text, fill="black", font=font)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP_PATH, dest="zip_path")
    ap.add_argument("--max-dim", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0, help="process first N sketches (0=all)")
    ap.add_argument("--shortlist", type=int, default=0,
                    help="coarse candidates reranked per sketch (0=all; highest recall)")
    ap.add_argument("--top-k", type=int, default=10, help="unique line-page candidates written per sketch")
    ap.add_argument("--ratio", type=float, default=0.78, help="SIFT Lowe ratio")
    ap.add_argument("--ransac-px", type=float, default=5.0)
    ap.add_argument("--min-inliers", type=int, default=6)
    ap.add_argument("--scale-min", type=float, default=0.80)
    ap.add_argument("--scale-max", type=float, default=1.25)
    ap.add_argument("--angle-max", type=float, default=5.0)
    ap.add_argument("--grid-fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="also search every pair by scale/angle grid + phase correlation")
    ap.add_argument("--grid-scale-step", type=float, default=0.10)
    ap.add_argument("--grid-angle-step", type=float, default=2.0)
    ap.add_argument("--csv-out", default=CSV_OUT)
    ap.add_argument("--json-out", default=JSON_OUT)
    ap.add_argument("--qc-out", default=QC_OUT)
    args = ap.parse_args()

    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.02, edgeThreshold=15)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    sketches, lines = {}, {}
    with zipfile.ZipFile(args.zip_path) as zf:
        manifest = json.loads(zf.read("dataset_ako5/manifest.json"))
        for entry in manifest:
            sid, lid = page_id(entry["sketch"]), page_id(entry["line"])
            if sid not in sketches:
                gray, scale = load_gray(zf, entry["sketch"], args.max_dim)
                sketches[sid] = prepare(gray, sift)
                sketches[sid]["resize_scale"] = scale
            if lid not in lines:
                gray, scale = load_gray(zf, entry["line"], args.max_dim)
                lines[lid] = prepare(gray, sift)
                lines[lid]["resize_scale"] = scale

    sketch_ids = sorted(sketches)
    if args.limit:
        sketch_ids = sketch_ids[:args.limit]
    rows = []
    grouped = {}
    scales = np.arange(args.scale_min, args.scale_max + 1e-6, args.grid_scale_step)
    angles = np.arange(-args.angle_max, args.angle_max + 1e-6, args.grid_angle_step)
    for index, sid in enumerate(sketch_ids, 1):
        stage_one = []
        for lid, line in lines.items():
            candidate = estimate_candidate(
                sketches[sid], line, matcher, args.ratio, args.ransac_px,
                args.min_inliers, args.scale_min, args.scale_max, args.angle_max,
            )
            if candidate is not None:
                candidate.update({"sketch_page": sid, "line_page": lid})
                stage_one.append(candidate)
            if args.grid_fallback:
                fallback = grid_candidate(sketches[sid], line, scales, angles)
                fallback.update({"sketch_page": sid, "line_page": lid})
                stage_one.append(fallback)
        stage_one.sort(key=lambda x: x["rank_score"], reverse=True)
        reranked = []
        shortlist = stage_one[:args.shortlist] if args.shortlist else stage_one
        for candidate in shortlist:
            stats = edge_statistics(sketches[sid], lines[candidate["line_page"]], candidate)
            if stats is not None:
                candidate.update(stats)
                candidate["selection_score"] = selection_score(candidate)
                full_matrix = candidate["matrix"].copy()
                full_matrix[:, :2] *= (sketches[sid]["resize_scale"]
                                       / lines[candidate["line_page"]]["resize_scale"])
                full_matrix[:, 2] /= lines[candidate["line_page"]]["resize_scale"]
                candidate["full_matrix"] = full_matrix
                candidate["tx_full"] = float(full_matrix[0, 2])
                candidate["ty_full"] = float(full_matrix[1, 2])
                reranked.append(candidate)
        reranked.sort(key=lambda x: x["selection_score"])
        # Keep the strongest transform for each line page so top-k means top-k pages.
        unique = {}
        for candidate in reranked:
            unique.setdefault(candidate["line_page"], candidate)
        reranked = list(unique.values())
        margin = (reranked[1]["selection_score"] - reranked[0]["selection_score"]
                  if len(reranked) > 1 else 0.0)
        for rank, row in enumerate(reranked[:args.top_k], 1):
            row["candidate_rank"] = rank
            row["margin"] = margin if rank == 1 else 0.0
            rows.append(row)
        grouped[sid] = [serializable(row) for row in reranked[:args.top_k]]
        best = reranked[0] if reranked else None
        result = (f'{best["line_page"]} edge={best["edge_score"]:.1f} '
                  f'inliers={best["inliers"]}' if best else "no candidate")
        print(f"[{index:02d}/{len(sketch_ids):02d}] {sid}: {result}")

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    fields = [
        "sketch_page", "line_page", "candidate_rank", "selection_score", "edge_score", "margin",
        "line_to_sketch_median", "line_to_sketch_p90", "sketch_to_line_median",
        "overlap_ratio", "close_ratio", "matches", "inliers", "inlier_ratio",
        "coverage", "rank_score", "scale", "angle", "tx_full", "ty_full", "method",
    ]
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(args.json_out, "w") as f:
        json.dump(grouped, f, indent=2)
    best_rows = [row for row in rows if row["candidate_rank"] == 1]
    make_qc(best_rows, sketches, lines, args.qc_out)
    print(f"Wrote {len(rows)} candidates: {args.csv_out}, {args.json_out}, {args.qc_out}")


if __name__ == "__main__":
    main()
