"""Break gt_bsds_f1 into precision/recall, and add a raw dark-area coverage
metric, to check the user's visual impression that coarse_trained wastes
less area/ink on content-independent hatching than anime_trained, even
though the composite f1 metric doesn't clearly show it.

- gt_bsds_precision: of everything the model drew, what fraction matches
  real GT content within tolerance. Low precision = lots of spurious
  hallucinated ink not explained by GT -- exactly the "meaningless
  line-filling" the user is describing at the edge-pixel level.
- raw_area_ink_fraction: fraction of the 480x480 canvas that is "dark"
  under a plain grayscale threshold (not edge-detected) -- captures area
  actually blackened/filled by dense hatching, which reads differently
  from thin GT lines even at matched edge-pixel counts.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
DARK_THRESHOLD = 200

MODELS = {
    "anime_trained": TRACK / "results/preprocessed_cond_check/cs1.00",
    "coarse_trained": TRACK / "results/controlnet_lora_coarse_20260827_eval",
}


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def main():
    print(f"{'model':16} {'sample':16} {'f1':>7} {'precision':>10} {'recall':>7} {'area_ink%':>10} {'gt_area_ink%':>13}")
    means = {label: {"f1": [], "precision": [], "recall": [], "area": []} for label in MODELS}
    for label, out_dir in MODELS.items():
        for s in SAMPLES:
            gt_gray = load_gray(TRACK / f"data/diag_gt_line_{s}.jpg")
            gt_edge = edge_map(gt_gray)
            out_path = out_dir / f"{s}_out.png"
            out_gray = load_gray(out_path)
            out_edge = edge_map(out_gray)
            f1, precision, recall = bipartite_match_f1(out_edge, gt_edge, BSDS_TOLERANCE_PX)
            area_ink = (out_gray < DARK_THRESHOLD).mean() * 100
            gt_area_ink = (gt_gray < DARK_THRESHOLD).mean() * 100
            means[label]["f1"].append(f1)
            means[label]["precision"].append(precision)
            means[label]["recall"].append(recall)
            means[label]["area"].append(area_ink)
            print(f"{label:16} {s:16} {f1:7.4f} {precision:10.4f} {recall:7.4f} {area_ink:10.2f} {gt_area_ink:13.2f}")
    print()
    for label, vals in means.items():
        print(
            f"{label:16} MEAN f1={np.mean(vals['f1']):.4f} precision={np.mean(vals['precision']):.4f} "
            f"recall={np.mean(vals['recall']):.4f} area_ink%={np.mean(vals['area']):.2f}"
        )


if __name__ == "__main__":
    main()
