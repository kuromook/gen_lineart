"""Diagnostic-only eval for the controlnet_conditioning_scale sweep on this
track's 5-sample diag set. Adapted from ../lineart/tools/evaluation/
condition_roundtrip_fidelity.py + evaluate_fixed_outputs.py, but pointed at
this track's flat data/ layout (data/diag_rough_raw/, data/diag_gt_line_*.jpg)
instead of dataset/pairs_480/{split}/... since this track doesn't mirror
that directory structure. Reuses the shared repo's edge_map/bipartite_match_f1
so numbers stay comparable to prior track results.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
ROUGH_DIR = TRACK / "data/diag_rough_raw"
GT_PREFIX = TRACK / "data"
SWEEP_DIR = TRACK / "results/cond_scale_sweep"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0

SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
SCALES = ["0.50", "0.75", "1.00", "1.25", "1.50", "1.75", "2.00"]


def load_rgb(path):
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def load_gray_array(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def get_detector():
    from controlnet_aux import LineartAnimeDetector
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")


def preprocess_condition(detector, rgb_image):
    out = detector(rgb_image, image_resolution=IMAGE_SIZE).resize((IMAGE_SIZE, IMAGE_SIZE))
    gray = np.asarray(out.convert("L"))
    return gray, gray > 32


def main():
    detector = get_detector()
    rows = []
    for base in SAMPLES:
        rough_path = ROUGH_DIR / f"{base}.jpg"
        orig_gray, orig_edge = preprocess_condition(detector, load_rgb(rough_path))
        gt_edge = edge_map(load_gray_array(GT_PREFIX / f"diag_gt_line_{base}.jpg"))
        for scale in SCALES:
            out_path = SWEEP_DIR / f"cs{scale}" / f"{base}_out.png"
            if not out_path.exists():
                continue
            recov_gray, recov_edge = preprocess_condition(detector, load_rgb(out_path))
            roundtrip_ssim = float(structural_similarity(orig_gray, recov_gray, data_range=255))
            roundtrip_bsds_f1, _, _ = bipartite_match_f1(recov_edge, orig_edge, BSDS_TOLERANCE_PX)
            out_edge = edge_map(load_gray_array(out_path))
            gt_bsds_f1, gt_p, gt_r = bipartite_match_f1(out_edge, gt_edge, BSDS_TOLERANCE_PX)
            rows.append({
                "scale": scale, "sample": base,
                "roundtrip_ssim": round(roundtrip_ssim, 4),
                "roundtrip_bsds_f1": round(roundtrip_bsds_f1, 4),
                "gt_bsds_f1": round(gt_bsds_f1, 4),
                "gt_bsds_precision": round(gt_p, 4),
                "gt_bsds_recall": round(gt_r, 4),
                "ink_ratio": round(out_edge.sum() / max(gt_edge.sum(), 1), 3),
            })

    out_csv = SWEEP_DIR / "sweep_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'scale':>6} {'roundtrip_ssim':>15} {'roundtrip_bsds_f1':>18} {'gt_bsds_f1':>11} {'ink_ratio':>10}")
    for scale in SCALES:
        sub = [r for r in rows if r["scale"] == scale]
        if not sub:
            continue
        ssim = np.mean([r["roundtrip_ssim"] for r in sub])
        rbf1 = np.mean([r["roundtrip_bsds_f1"] for r in sub])
        gbf1 = np.mean([r["gt_bsds_f1"] for r in sub])
        ink = np.mean([r["ink_ratio"] for r in sub])
        print(f"{scale:>6} {ssim:15.4f} {rbf1:18.4f} {gbf1:11.4f} {ink:10.3f}")
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
