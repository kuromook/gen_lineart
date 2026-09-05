"""Seven-way comparison, extending results/six_model_comparison/eval_compare.py
with the LoRA rank32 retrain of the manga_line candidate (hypothesis #2,
inbox/initial_notice.md 2026-09-02): same base_ckpt/ControlNet-init/
conditioning/epochs/lr as manga_trained (controlnet_lora_manga_20260827,
rank16), only rank16->rank32, to test whether LoRA capacity was limiting
clean-line reproduction and causing the cross-hatch fallback.
- manga_rank32_trained: controlnet_lora_manga_rank32_20260902
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
    "sd15_lineart_trained": TRACK / "results/controlnet_lora_lineartsd15_20260827_eval",
    "manga_trained": TRACK / "results/controlnet_lora_manga_20260827_eval",
    "manga_rank32_trained": TRACK / "results/controlnet_lora_manga_rank32_20260902_eval",
    "sdxl_trained": TRACK / "results/controlnet_lora_sdxl_20260829_eval",
    "sdxl_manga_trained": TRACK / "results/controlnet_lora_sdxl_manga_20260830_eval",
    "manga_epoch2_trained": TRACK / "results/controlnet_lora_manga_epoch2_20260903_eval",
}


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def main():
    print(f"{'model':22} {'sample':16} {'f1':>7} {'precision':>10} {'recall':>7} {'area_ink%':>10}")
    rows = []
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
            means[label]["f1"].append(f1)
            means[label]["precision"].append(precision)
            means[label]["recall"].append(recall)
            means[label]["area"].append(area_ink)
            rows.append({"model": label, "sample": s, "f1": f1, "precision": precision, "recall": recall, "area_ink_pct": area_ink})
            print(f"{label:22} {s:16} {f1:7.4f} {precision:10.4f} {recall:7.4f} {area_ink:10.2f}")
    print()
    print(f"{'model':22} {'MEAN f1':>8} {'precision':>10} {'recall':>8} {'area_ink%':>10}")
    for label, vals in means.items():
        print(
            f"{label:22} {np.mean(vals['f1']):8.4f} {np.mean(vals['precision']):10.4f} "
            f"{np.mean(vals['recall']):8.4f} {np.mean(vals['area']):10.2f}"
        )

    import csv
    with open(TRACK / "results/eight_model_comparison/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved: {TRACK / 'results/eight_model_comparison/metrics.csv'}")


if __name__ == "__main__":
    main()
