"""`bipartite_match_f1` behind a hard per-tile timeout.

The metric itself is fine and is not touched here -- it is shared and
validated, and comparisons depend on it. What is not fine is its tail:
scipy's `maximum_bipartite_matching` approaches its worst case on particular
edge-point configurations, so a single tile can take tens of seconds to 12+
minutes, unpredictably from density (98 of 584 pairs timed out in one run).
A serial batch therefore has an unbounded worst case, and a batch that looks
hung is usually one slow tile.

`../lineart-pair-signal` worked around it per tile, in its own
`vae_roundtrip_score.py`, and that mitigation never reached shared tooling --
recorded as an open bug in doc/CURRENT.md. This is the shared form: one call
that returns `None` instead of blocking, so a caller records a miss and keeps
going.

The subprocess costs an interpreter start (roughly 0.3-1s per tile), so a
caller that has never hit the tail can pass `timeout_s=0` for the old
in-process behaviour. The numbers are identical either way -- the same
function on the same arrays -- which `self_check()` below verifies.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "score_bsds_f1_one.py"
PYTHON = Path(sys.executable)
DEFAULT_TIMEOUT_S = 300.0


def _in_process(pred_edge, gt_edge, tolerance_px):
    sys.path.insert(0, str(HERE.parent / "pair_extraction"))
    from tile_region_manifest_480 import bipartite_match_f1

    return bipartite_match_f1(pred_edge, gt_edge, tolerance_px)


def match_f1(pred_edge, gt_edge, tolerance_px, timeout_s=DEFAULT_TIMEOUT_S):
    """(f1, precision, recall), or None if the tile exceeded `timeout_s`.

    `timeout_s <= 0` runs in-process with no cap, which is what every caller
    did before this existed.
    """
    if timeout_s is None or timeout_s <= 0:
        return _in_process(pred_edge, gt_edge, tolerance_px)

    with tempfile.TemporaryDirectory(prefix="bsds_guard_") as work:
        payload = Path(work) / "arrays.npz"
        np.savez_compressed(
            payload,
            pred=np.asarray(pred_edge, dtype=bool),
            gt=np.asarray(gt_edge, dtype=bool),
        )
        try:
            proc = subprocess.run(
                [str(PYTHON), str(WORKER), str(payload), str(tolerance_px)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            raise RuntimeError(
                f"bsds worker failed ({proc.returncode}): {proc.stderr.strip()[-500:]}"
            )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return result["f1"], result["precision"], result["recall"]


def self_check(seed=20261009, size=64, points=400):
    """Guarded and in-process agree on a random case, and a tiny timeout is
    reported as a miss rather than as a score. Run as
    `python tools/evaluation/bsds_guard.py`."""
    rng = np.random.default_rng(seed)
    pred = np.zeros((size, size), dtype=bool)
    gt = np.zeros((size, size), dtype=bool)
    idx = rng.integers(0, size, size=(points, 2))
    pred[idx[:, 0], idx[:, 1]] = True
    idx = rng.integers(0, size, size=(points, 2))
    gt[idx[:, 0], idx[:, 1]] = True

    direct = _in_process(pred, gt, 2.0)
    guarded = match_f1(pred, gt, 2.0, timeout_s=DEFAULT_TIMEOUT_S)
    agree = all(abs(a - b) < 1e-12 for a, b in zip(direct, guarded))
    missed = match_f1(pred, gt, 2.0, timeout_s=0.001)
    print(f"in-process f1={direct[0]:.6f}  guarded f1={guarded[0]:.6f}  agree={agree}")
    print(f"timeout 0.001s returns {missed!r} (None means the cap works)")
    return agree and missed is None


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)
