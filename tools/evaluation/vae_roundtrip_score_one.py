"""Score exactly one (gt, recon) pair and print the result as one JSON line.

Deliberately a single-pair CLI, not a library call -- `vae_roundtrip_score.py`
dispatches this as a subprocess per tile (via `subprocess.run(timeout=...)`)
because `tile_region_manifest_480.bipartite_match_f1`'s scipy
`maximum_bipartite_matching` was found (this track, 2026-09-13) to blow up to
several hundred seconds -- and in at least one observed case, 12+ minutes --
for specific edge-point configurations (not a fixed property of image
density; two visually unremarkable ~9,000-edge-point tiles reproduced it).
Running each tile as its own subprocess is the only way to enforce a hard
per-tile wall-clock cap without touching that shared, validated metric
function's own implementation.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pair_extraction"))
from measure_lineart_profile import load_gray, profile_metrics  # noqa: E402
from tile_region_manifest_480 import bipartite_match_f1, chamfer, edge_map  # noqa: E402

TOLERANCE_PX = 2.0
TRUNCATE_PX = 8.0
PROFILE_KEYS = ("near_white_frac", "midtone_frac", "bg_mode", "fill_ratio", "ink_ratio", "line_width_p50")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    recon_gray = load_gray(args.recon)
    gt_gray = load_gray(args.gt)
    recon_edge = edge_map(recon_gray)
    gt_edge = edge_map(gt_gray)
    bsds_f1, bsds_precision, bsds_recall = bipartite_match_f1(recon_edge, gt_edge, TOLERANCE_PX)
    chamfer_px = chamfer(recon_edge, gt_edge, TRUNCATE_PX)
    recon_profile = profile_metrics(args.recon)
    gt_profile = profile_metrics(args.gt)

    result = {
        "bsds_f1": bsds_f1,
        "bsds_precision": bsds_precision,
        "bsds_recall": bsds_recall,
        "chamfer_px": chamfer_px,
    }
    for key in PROFILE_KEYS:
        result[f"recon_{key}"] = recon_profile[key]
        result[f"gt_{key}"] = gt_profile[key]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
