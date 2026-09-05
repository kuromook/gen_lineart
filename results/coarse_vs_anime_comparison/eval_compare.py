"""Final comparison: lineart_anime-trained LoRA (controlnet_lora_realpairs_20260824,
evaluated with matching lineart_anime-preprocessed conditioning) vs
lineart_coarse-trained LoRA (controlnet_lora_coarse_20260827, evaluated with
matching lineart_coarse-preprocessed conditioning), both against GT on diag5."""

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

MODELS = {
    "anime_trained (preproc cond)": TRACK / "results/preprocessed_cond_check/cs1.00",
    "coarse_trained (preproc cond)": TRACK / "results/controlnet_lora_coarse_20260827_eval",
}


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def main():
    print(f"{'model':32} {'sample':16} {'gt_bsds_f1':>11} {'ink_ratio':>10}")
    means = {label: [] for label in MODELS}
    for label, out_dir in MODELS.items():
        for s in SAMPLES:
            gt_edge = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
            out_path = out_dir / f"{s}_out.png"
            out_edge = edge_map(load_gray(out_path))
            f1, _, _ = bipartite_match_f1(out_edge, gt_edge, BSDS_TOLERANCE_PX)
            ink_ratio = out_edge.sum() / max(gt_edge.sum(), 1)
            means[label].append(f1)
            print(f"{label:32} {s:16} {f1:11.4f} {ink_ratio:10.3f}")
    print()
    for label, vals in means.items():
        print(f"{label:32} MEAN gt_bsds_f1={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
