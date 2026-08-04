"""Quantify stroke continuity/fragmentation ("wobble") in line-art outputs,
independent of GT coordinate alignment. Complements
evaluate_fixed_outputs.py's F1/chamfer/ink_ratio (alignment- and
darkness-focused) with metrics that specifically capture whether ink
forms long continuous strokes or many short disconnected fragments.

Motivation (doc/work_log.md 2026-08-01, single-stage direct-regression
"wobble" investigation): the existing metrics can't distinguish "more
training made the output more *confident* (darker/more ink)" from "more
training made the output more *stable* (fragments merging into
continuous strokes)" -- these are separate claims that happened to move
together so far, but aren't guaranteed to.

Metrics (all computed on the prediction alone, thresholded at 128 like
the other eval tools; also computed on GT for reference):

- component_count / mean_component_len / median_component_len /
  long_component_ratio: skeletonize the ink mask, take connected
  components (a continuous stroke skeletonizes to ~1 component spanning
  its length; a wobbly/fragmented stroke breaks into many short
  components). long_component_ratio is the fraction of skeleton pixels
  belonging to components at least LONG_COMPONENT_PX long.
- orientation_entropy: reused from
  tools/pair_extraction/tile_region_manifest_480.py -- entropy of local
  edge-direction histogram; low = locally coherent/directional strokes,
  high = chaotic/noisy edges.
- long_line_ratio: reused from the same module (Hough-line-based); a
  smaller --min-length-fraction than that module's panel-border default,
  since character strokes are much shorter than panel borders.
- endpoint_count / endpoints_per_1k_ink_px (added 2026-08-04, clDice
  probe follow-up): count of skeleton pixels with exactly one 8-connected
  skeleton neighbor. A single continuous stroke has exactly 2 endpoints
  (its ends); every break adds 2 more. Complements long_component_ratio,
  which is length-weighted and can stay high even as many short breaks
  accumulate along an otherwise-long path -- endpoint density is a count
  of breaks, not their share of total ink.
- topo_precision / topo_sensitivity / hard_cldice (added 2026-08-04):
  the clDice paper's own topology metric (Shit et al.,
  https://arxiv.org/abs/2003.07311), computed on hard-thresholded masks
  rather than the soft/differentiable version used as a training loss in
  lineart/losses.py::soft_cldice_loss. topo_precision = fraction of the
  prediction's skeleton that lands on GT ink; topo_sensitivity =
  fraction of GT's skeleton that lands on prediction ink; hard_cldice is
  their harmonic mean. Unlike the other metrics here, these ARE paired
  against GT (not GT-alignment-independent) -- added specifically to
  check whether a soft-clDice-trained model's actual hard-threshold
  topology overlap with GT improved, or whether the soft loss was
  satisfied some other way (e.g. widening strokes) that a hard threshold
  doesn't reward.
- rough_topo_precision / rough_topo_sensitivity / rough_hard_cldice
  (added 2026-08-04, second follow-up): same clDice-style computation as
  above, but paired against the *rough* input instead of GT. Motivation
  (doc/architecture_decisions.md 2026-08-02 binarization/rough-fidelity
  entry): the earlier tolerance-based evaluate_rough_fidelity.py
  (Canny-edge distance-tolerance matching, from
  score_pair_agreement.py) failed to detect a real degradation the user
  spotted visually -- confident/binarized later-epoch output traces a
  different, merely "plausible-looking" stroke path instead of the
  rough's actual specific path, but tolerance-based edge matching can't
  tell "near the rough's edge-dense region" from "tracing the same
  curve," so its score stays flat instead of catching the divergence.
  clDice's skeleton-vs-mask formulation is curve-identity-aware (a
  skeleton pixel either lands inside the reference stroke's own local
  width or it doesn't), so it should be more sensitive to exactly this
  failure mode. Since the rough is a lighter/softer pencil sketch, not a
  hard-thresholdable ink mask, rough structure is extracted with Canny
  edges (same 45/135 thresholds validated in score_pair_agreement.py)
  instead of the gray<128 threshold used for line art; the "rough ink"
  side of the pairing is a small (3px, matching that module's tolerance)
  dilation of those edges, standing in for the stroke's own local width.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pair_extraction"))

import cv2
import numpy as np

from tile_region_manifest_480 import edge_map, long_line_ratio, orientation_entropy
from score_pair_agreement import edges as canny_edges


IMAGE_SIZE = 480
THRESHOLD = 128
LONG_COMPONENT_PX = 20
ROUGH_EDGE_TOLERANCE_PX = 3
DEFAULT_SAMPLES = [
    "housei_002_06_15",
    "housei_002_07_12",
    "housei_002_19_12",
    "lineart_004_002",
    "lineart_004_004",
    "lineart_004_006",
    "lineart_004_008",
    "lineart_004_010",
]


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def dataset_path(name, split):
    base = normalize_name(name)
    if split == "auto":
        split = "train" if base.startswith("housei") else "test"
    return f"dataset/pairs_480/{split}/line/{base}.jpg"


def rough_path(name, split):
    base = normalize_name(name)
    if split == "auto":
        split = "train" if base.startswith("housei") else "test"
    return f"dataset/pairs_480/{split}/rough/{base}.jpg"


def load_ink_and_gray(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    ink = gray < THRESHOLD
    return ink, gray


def skeletonize(mask):
    """Morphological thinning (same approach as
    scripts/train_i2i_survey.py::skeletonize_ink)."""
    m = mask.astype(np.uint8) * 255
    skel = np.zeros_like(m)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(m) > 0:
        eroded = cv2.erode(m, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(m, opened))
        m = eroded
    return skel > 0


def skeleton_endpoints(skeleton):
    """Count skeleton pixels with exactly one 8-connected skeleton neighbor."""
    skel_u8 = skeleton.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    tagged = cv2.filter2D(skel_u8, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    # center pixel contributes 10; tagged == 11 means center is skeleton
    # (10) plus exactly one skeleton neighbor (1).
    return int(np.count_nonzero(tagged == 11))


def hard_cldice_metrics(pred_skel, pred_ink, gt_skel, gt_ink, smooth=1e-6):
    """Hard-threshold analog of lineart.losses.soft_cldice_loss's topology
    precision/sensitivity (Shit et al., https://arxiv.org/abs/2003.07311)."""
    pred_skel_px = float(pred_skel.sum())
    gt_skel_px = float(gt_skel.sum())
    t_prec = float((pred_skel & gt_ink).sum()) / (pred_skel_px + smooth)
    t_sens = float((gt_skel & pred_ink).sum()) / (gt_skel_px + smooth)
    hard_cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + smooth)
    return {
        "topo_precision": t_prec,
        "topo_sensitivity": t_sens,
        "hard_cldice": hard_cldice,
    }


def load_rough_bundle(path):
    """Canny-edge structure of the rough input, standing in for a skeleton
    (edge) and its own local stroke width (a small tolerance dilation of
    that edge) -- see module docstring for why Canny instead of a gray
    threshold."""
    rough = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if rough is None:
        raise FileNotFoundError(path)
    rough = cv2.resize(rough, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    edge = canny_edges(rough, 45, 135)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ROUGH_EDGE_TOLERANCE_PX + 1, 2 * ROUGH_EDGE_TOLERANCE_PX + 1)
    )
    edge_dilated = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    return {"edge": edge, "edge_dilated": edge_dilated}


def rough_fidelity_metrics(pred_skel, pred_ink, rough_bundle):
    result = hard_cldice_metrics(pred_skel, pred_ink, rough_bundle["edge"], rough_bundle["edge_dilated"])
    return {f"rough_{key}": value for key, value in result.items()}


def component_stats(skeleton, long_threshold_px=LONG_COMPONENT_PX):
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), connectivity=8
    )
    lengths = stats[1:, cv2.CC_STAT_AREA]  # skeleton is ~1px wide, so area ~= path length
    if len(lengths) == 0:
        return {
            "component_count": 0,
            "mean_component_len": 0.0,
            "median_component_len": 0.0,
            "long_component_ratio": 0.0,
        }
    total = float(lengths.sum())
    long_ratio = float(lengths[lengths >= long_threshold_px].sum()) / max(total, 1.0)
    return {
        "component_count": int(len(lengths)),
        "mean_component_len": float(lengths.mean()),
        "median_component_len": float(np.median(lengths)),
        "long_component_ratio": long_ratio,
    }


EMPTY_METRICS = {
    "orientation_entropy": 0.0,
    "long_line_ratio": 0.0,
    "component_count": 0,
    "mean_component_len": 0.0,
    "median_component_len": 0.0,
    "long_component_ratio": 0.0,
    "components_per_1k_ink_px": 0.0,
    "endpoint_count": 0,
    "endpoints_per_1k_ink_px": 0.0,
}


def load_bundle(path):
    """Load ink mask + skeleton once, shared by both the standalone
    (GT-alignment-independent) metrics and the paired hard-clDice metrics."""
    ink, gray = load_ink_and_gray(path)
    ink_px = int(ink.sum())
    skel = skeletonize(ink) if ink_px >= 20 else np.zeros_like(ink)
    return {"ink": ink, "gray": gray, "ink_px": ink_px, "skel": skel}


def stability_metrics(bundle, min_length_fraction):
    ink, gray, ink_px, skel = bundle["ink"], bundle["gray"], bundle["ink_px"], bundle["skel"]
    if ink_px < 20:
        return {"ink_pixels": ink_px, **EMPTY_METRICS}
    edges = edge_map(gray)
    comp = component_stats(skel)
    endpoint_count = skeleton_endpoints(skel)
    return {
        "ink_pixels": ink_px,
        "orientation_entropy": orientation_entropy(edges),
        "long_line_ratio": long_line_ratio(ink, IMAGE_SIZE, min_length_fraction),
        **comp,
        "components_per_1k_ink_px": (comp["component_count"] / max(ink_px, 1)) * 1000.0,
        "endpoint_count": endpoint_count,
        "endpoints_per_1k_ink_px": (endpoint_count / max(ink_px, 1)) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sample-list", default=None)
    parser.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    parser.add_argument("--output-csv", default="results/stroke_stability_metrics.csv")
    parser.add_argument(
        "--min-length-fraction",
        type=float,
        default=0.08,
        help="Hough long_line_ratio's min segment length as a fraction of tile size "
        "(0.08 ~= 38px for a 480 tile -- tuned for character strokes, not panel borders)",
    )
    args = parser.parse_args()

    samples = read_sample_list(args.sample_list) if args.sample_list else DEFAULT_SAMPLES

    rows = []
    for name in samples:
        base = normalize_name(name)
        gt_bundle = load_bundle(dataset_path(name, args.split))
        rough_bundle = load_rough_bundle(rough_path(name, args.split))
        gt_row = {
            "sample": base,
            "model": "GT",
            **stability_metrics(gt_bundle, args.min_length_fraction),
            "topo_precision": 1.0,
            "topo_sensitivity": 1.0,
            "hard_cldice": 1.0,
            **rough_fidelity_metrics(gt_bundle["skel"], gt_bundle["ink"], rough_bundle),
        }
        rows.append(gt_row)
        for model in args.models:
            pred_path = f"results/{model}/{base}_out.png"
            pred_bundle = load_bundle(pred_path)
            row = {
                "sample": base,
                "model": model,
                **stability_metrics(pred_bundle, args.min_length_fraction),
                **hard_cldice_metrics(
                    pred_bundle["skel"], pred_bundle["ink"],
                    gt_bundle["skel"], gt_bundle["ink"],
                ),
                **rough_fidelity_metrics(pred_bundle["skel"], pred_bundle["ink"], rough_bundle),
            }
            rows.append(row)

    fields = list(rows[0])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "component_count",
        "long_component_ratio",
        "components_per_1k_ink_px",
        "endpoint_count",
        "endpoints_per_1k_ink_px",
        "topo_precision",
        "topo_sensitivity",
        "hard_cldice",
        "rough_topo_precision",
        "rough_topo_sensitivity",
        "rough_hard_cldice",
    ]
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    header = "model".ljust(28) + "".join(k[:14].rjust(16) for k in metric_keys)
    print(header)
    for model in ["GT"] + [m for m in args.models]:
        model_rows = by_model[model]
        means = {k: np.mean([r[k] for r in model_rows]) for k in metric_keys}
        line = model.ljust(28) + "".join(f"{means[k]:16.3f}" for k in metric_keys)
        print(line)

    print(f"\nsaved: {args.output_csv}")


if __name__ == "__main__":
    main()
