"""Hatch-fill-escape metric (orientation_entropy axis), extending
results/four_model_comparison/hatch_score.py with the SDXL migration
candidate. See that file's docstring for the full rationale: this is
orthogonal to gt_bsds_f1/precision/recall and is the axis that actually
tracks "escape hatch-fill hallucination", the track's current goal.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map, orientation_entropy  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
IMAGE_SIZE = 480

MODELS = {
    "anime_trained": TRACK / "results/preprocessed_cond_check/cs1.00",
    "coarse_trained": TRACK / "results/controlnet_lora_coarse_20260827_eval",
    "sd15_lineart_trained": TRACK / "results/controlnet_lora_lineartsd15_20260827_eval",
    "manga_trained": TRACK / "results/controlnet_lora_manga_20260827_eval",
    "sdxl_trained": TRACK / "results/controlnet_lora_sdxl_20260829_eval",
}


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def main():
    print(f"{'sample':16} {'GT':>8}", end="")
    for label in MODELS:
        print(f" {label:>22}", end="")
    print()

    gt_entropies = []
    model_entropies = {label: [] for label in MODELS}
    for s in SAMPLES:
        gt_gray = load_gray(TRACK / f"data/diag_gt_line_{s}.jpg")
        gt_ent = orientation_entropy(edge_map(gt_gray))
        gt_entropies.append(gt_ent)
        print(f"{s:16} {gt_ent:8.4f}", end="")
        for label, out_dir in MODELS.items():
            out_gray = load_gray(out_dir / f"{s}_out.png")
            ent = orientation_entropy(edge_map(out_gray))
            model_entropies[label].append(ent)
            print(f" {ent:22.4f}", end="")
        print()

    print()
    print(f"{'MEAN':16} {np.mean(gt_entropies):8.4f}", end="")
    for label in MODELS:
        print(f" {np.mean(model_entropies[label]):22.4f}", end="")
    print()
    print("\n(0=maximally hatch-like/single dominant angle, 1=fully uniform across all angles; GT row is the reference 'real linework' range)")


if __name__ == "__main__":
    main()
