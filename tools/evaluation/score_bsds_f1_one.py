"""Score one tile's `bipartite_match_f1` and print it as JSON.

Exists so `bsds_guard.match_f1` can impose a hard timeout on a metric whose
worst case is unbounded -- see that module. Takes an .npz holding `pred` and
`gt` boolean edge maps plus the tolerance in pixels, and prints one JSON
object on the last stdout line.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pair_extraction"))
from tile_region_manifest_480 import bipartite_match_f1  # noqa: E402


def main(payload_path, tolerance_px):
    arrays = np.load(payload_path)
    f1, precision, recall = bipartite_match_f1(
        arrays["pred"], arrays["gt"], float(tolerance_px)
    )
    print(json.dumps({"f1": f1, "precision": precision, "recall": recall}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
