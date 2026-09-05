"""Score each candidate-word variant's orientation_entropy against the
zero-word baseline (results/controlnet_lora_manga_nomangaword_20260905_eval/,
mean 0.7860) and GT (0.7117). See generate_sweep.py's docstring for the
generation setup.
"""

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map, orientation_entropy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_tags import CANDIDATE_TAGS  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT_ROOT = TRACK / "results/word_ablation_20260905/outputs"
BASELINE_DIR = TRACK / "results/controlnet_lora_manga_nomangaword_20260905_eval"
IMAGE_SIZE = 480


def safe_dirname(tag):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", tag)


def load_gray(p):
    return np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def mean_entropy(out_dir):
    vals = []
    for s in SAMPLES:
        gray = load_gray(out_dir / f"{s}_out.png")
        vals.append(orientation_entropy(edge_map(gray)))
    return float(np.mean(vals)), vals


def main():
    gt_entropies = []
    for s in SAMPLES:
        gt_gray = load_gray(TRACK / f"data/diag_gt_line_{s}.jpg")
        gt_entropies.append(orientation_entropy(edge_map(gt_gray)))
    gt_mean = float(np.mean(gt_entropies))

    baseline_mean, _ = mean_entropy(BASELINE_DIR)

    rows = []
    for tag in CANDIDATE_TAGS:
        out_dir = OUT_ROOT / safe_dirname(tag)
        mean_ent, vals = mean_entropy(out_dir)
        rows.append((tag, mean_ent, mean_ent - baseline_mean))

    rows.sort(key=lambda r: -r[1])

    import csv
    with open(TRACK / "results/word_ablation_20260905/scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "mean_orientation_entropy", "delta_vs_baseline"])
        for tag, ent, delta in rows:
            w.writerow([tag, f"{ent:.4f}", f"{delta:+.4f}"])

    print(f"GT reference:                          {gt_mean:.4f}")
    print(f"baseline (no word added, 'manga panel'-free): {baseline_mean:.4f}")
    print()
    print(f"{'tag':22} {'mean_entropy':>14} {'delta_vs_baseline':>18}")
    for tag, ent, delta in rows:
        flag = "  <-- MOST hatch-like (max entropy)" if (tag, ent, delta) == rows[0] else ""
        print(f"{tag:22} {ent:14.4f} {delta:+18.4f}{flag}")
    print()
    print("(sorted descending by mean_entropy: top = most hatch-like/uniform-angle,")
    print(" bottom = closest to GT's single-dominant-angle clean strokes)")
    print(f"\nsaved: {TRACK / 'results/word_ablation_20260905/scores.csv'}")


if __name__ == "__main__":
    main()
