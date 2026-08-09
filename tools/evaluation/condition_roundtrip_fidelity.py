"""Conditioning-roundtrip fidelity: does a ControlNet output actually follow
its own input conditioning, independent of how close it lands to GT?

Chamfer-to-GT (`evaluate_fixed_outputs.py`) repeatedly failed to separate a
known-bad hallucination case (own domain-LoRA overriding ControlNet
conditioning, producing a content-independent repeating hatch texture) from
a known-good structural-correspondence case, because both outputs are dense
enough that nearest-edge distances saturate regardless of whether the
output's structure actually traces the input (see `doc/work_log.md`
2026-08-09 "Workstream C" entry).

This metric sidesteps GT entirely: re-run the same `lineart_anime`
preprocessor used to build the ControlNet conditioning input on the model's
*output*, and compare that recovered conditioning map to the original
conditioning map (both derived from the rough tile). A model faithfully
following its conditioning should recover something close to the original
conditioning; a model that ignored conditioning and hallucinated its own
content should not, regardless of output density or GT similarity.

**2026-08-09 addition**: a same-day literature check
(`doc/eval_metric_literature_survey_20260809.md`) found ControlNet++
(Zhang et al. follow-up, ECCV 2024) independently formalizes this exact
roundtrip idea as a "controllability" consistency reward, and specifies
**SSIM** as its comparison function for edge/line-art conditions
specifically -- not Chamfer. Our roundtrip framework converged on the same
design independently, but the original Chamfer-based comparison (`
roundtrip_chamfer` below) is still the same point-matching primitive that
failed for `evaluate_fixed_outputs.py`; it likely scored 5/6 against the
hand ranking because self-consistent conditioning maps are closer in
density than model-output-vs-hand-drawn-GT, not because the mechanism
itself got more robust. Added `roundtrip_ssim` (`skimage.metrics.
structural_similarity` on the raw grayscale conditioning maps, before
binarization) and `roundtrip_bsds_f1` (one-to-one bipartite-matched F1,
`tile_region_manifest_480.bipartite_match_f1`, on the binarized maps) as
of-record comparisons -- validate all three against a hand ranking before
picking one as primary for a given use.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pair_extraction"))
from tile_region_manifest_480 import edge_map, chamfer, bipartite_match_f1  # noqa: E402

IMAGE_SIZE = 480
TRUNCATE_PX = 8.0
BSDS_TOLERANCE_PX = 2.0


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def load_rgb(path):
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def load_gray_array(path):
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(img)


def get_detector():
    from controlnet_aux import LineartAnimeDetector
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")


def preprocess_condition(detector, rgb_image):
    """Run the lineart_anime preprocessor. Returns (grayscale, binary_edge):
    grayscale is the raw 0-255 conditioning map (for SSIM, which wants
    continuous tone -- gradients/antialiasing carry real information a
    binary mask throws away); binary_edge is the thresholded boolean mask
    (white line on black), same convention as `tile_region_manifest_480.
    edge_map`, for chamfer/bipartite-match comparisons.
    """
    out = detector(rgb_image, image_resolution=IMAGE_SIZE)
    out = out.resize((IMAGE_SIZE, IMAGE_SIZE))
    gray = np.asarray(out.convert("L"))
    return gray, gray > 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model", action="append", required=True, help="LABEL=DIR")
    parser.add_argument("--truncate", type=float, default=TRUNCATE_PX)
    parser.add_argument("--output-csv", default="results/condition_roundtrip_fidelity.csv")
    args = parser.parse_args()

    names = [n.strip() for n in open(args.sample_list) if n.strip()]
    bases = [normalize_name(n) for n in names]
    models = []
    for value in args.model:
        label, directory = value.split("=", 1) if "=" in value else (value, f"results/{value}")
        models.append((label, Path(directory)))

    detector = get_detector()

    rows = []
    for base in bases:
        rough_path = f"dataset/pairs_480/{args.split}/rough/{base}.jpg"
        orig_gray, orig_edge = preprocess_condition(detector, load_rgb(rough_path))
        for label, directory in models:
            out_path = directory / f"{base}_out.png"
            if not out_path.exists():
                continue
            recov_gray, recov_edge = preprocess_condition(detector, load_rgb(out_path))
            roundtrip_chamfer = chamfer(recov_edge, orig_edge, truncate=args.truncate)
            roundtrip_ssim = float(structural_similarity(orig_gray, recov_gray, data_range=255))
            roundtrip_bsds_f1, _, _ = bipartite_match_f1(recov_edge, orig_edge, BSDS_TOLERANCE_PX)
            gt_edge = edge_map(load_gray_array(f"dataset/pairs_480/{args.split}/line/{base}.jpg"))
            gt_chamfer = chamfer(edge_map(load_gray_array(out_path)), gt_edge, truncate=args.truncate)
            rows.append({
                "sample": base, "model": label,
                "roundtrip_chamfer": round(roundtrip_chamfer, 4),
                "roundtrip_ssim": round(roundtrip_ssim, 4),
                "roundtrip_bsds_f1": round(roundtrip_bsds_f1, 4),
                "gt_chamfer": round(gt_chamfer, 4),
                "orig_condition_ink": int(orig_edge.sum()),
                "recovered_condition_ink": int(recov_edge.sum()),
            })

    fields = list(rows[0].keys())
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'model':30s} {'roundtrip_chamfer':>18s} {'roundtrip_ssim':>15s} {'roundtrip_bsds_f1':>18s} {'gt_chamfer':>12s}")
    for label, _ in models:
        model_rows = [r for r in rows if r["model"] == label]
        if not model_rows:
            continue
        rt = np.mean([r["roundtrip_chamfer"] for r in model_rows])
        ssim_mean = np.mean([r["roundtrip_ssim"] for r in model_rows])
        bsds_mean = np.mean([r["roundtrip_bsds_f1"] for r in model_rows])
        gc = np.mean([r["gt_chamfer"] for r in model_rows])
        print(f"{label:30s} {rt:18.4f} {ssim_mean:15.4f} {bsds_mean:18.4f} {gc:12.4f}")
    print(f"\nsaved: {args.output_csv}")


if __name__ == "__main__":
    main()
