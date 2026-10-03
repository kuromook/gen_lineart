"""Score one tile of one arm (any arm, any track). Run as its own process by
measure_f1_signal.py under a hard timeout.

Copied into the shared tools 2026-10-03 from Track I's
`lineart-image-prompt/experiments/score_ipadapter_probe_one_20260930.py`
(unchanged apart from this header) to complete that script's port --
`measure_f1_signal.py` was ported into `tools/evaluation/` on 2026-10-03 but
its `WORKER` constant still pointed at this file's old Track-I-only path,
which no longer resolved once the orchestrator moved. Found while
Track C tried to rescore its own outputs per the new signal protocol.

The reason for the subprocess-per-tile shape: `bipartite_match_f1` can take
tens of seconds to 12+ minutes on particular edge-point configurations, and it
is not predictable from image density (doc/CURRENT.md, Known Tool Traps). A
serial run therefore has an unbounded worst case. Same mitigation as
`vae_roundtrip_score.py` in the pair-signal track.

Prints one JSON object on stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import bipartite_match_f1, edge_map  # noqa: E402

IMAGE_SIZE = 480
TOL = 2.0


def load_gray(path, invert=False):
    gray = np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
    return 255 - gray if invert else gray


def main():
    img_path, gt_path, cond_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # `--invert-pred` is for scoring a conditioning map as if it were an output:
    # conditioning is white-on-black, everything else here is black-on-white.
    invert_pred = len(sys.argv) > 4 and sys.argv[4] == "--invert-pred"

    pred = load_gray(img_path, invert=invert_pred)
    gt = load_gray(gt_path)
    cond = load_gray(cond_path, invert=True)  # to GT polarity

    pred_edge = edge_map(pred)
    f1, precision, recall = bipartite_match_f1(pred_edge, edge_map(gt), TOL)
    # How much of the output is just the conditioning map restated. Track D's
    # `vs_condition`: high means the output has not moved away from its input.
    vs_cond_f1, _, _ = bipartite_match_f1(pred_edge, edge_map(cond), TOL)

    row = {
        "gt_bsds_f1": f1,
        "precision": precision,
        "recall": recall,
        "vs_condition_f1": vs_cond_f1,
    }
    # profile_metrics reads from disk and expects GT polarity, so a conditioning
    # map has to be written out inverted first; the dispatcher does that.
    for key, value in profile_metrics(img_path).items():
        if key in ("ink_ratio", "fill_ratio", "line_width_p50", "near_white_frac",
                   "midtone_frac", "bg_mode"):
            row[key] = value
    print(json.dumps(row))


if __name__ == "__main__":
    main()
