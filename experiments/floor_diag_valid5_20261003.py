"""The floor on a five-tile set: hold the GT fixed, vary the prediction.

DIRECTION MATTERS, and the first version of this had it backwards. Scoring one
output against many other GTs averages over how easy each of those GTs is to
hit, so a tile whose own GT is harder than average comes out *below* its own
floor -- the degenerate arm, whose signal is zero by construction, read -0.0645
that way. The null has to keep the target fixed and randomise the prediction:
for tile i, score the SAME ARM's outputs for other tiles against GT_i. That is
the permutation null for "could an output unrelated to this input have scored
this well against THIS GT", and it holds tile difficulty constant.

On 192 tiles the two directions happen to agree (the degenerate arm reads
-0.0001 either way) because both average over the same pool. On five tiles they
do not, and five tiles is where this project's early verdicts were decided.

On 192 tiles one mismatched partner per tile is enough: the pairing is a
permutation, so the 192 draws average out. On `diag_valid5` it is not -- five
draws leave the estimate swinging by more than the quantity being measured (the
degenerate arm, whose signal is zero by construction, came out at -0.0725 with
one partner each). So every tile here is scored against MANY different-source
GTs and averaged, which is the same estimator with its variance brought down.

This matters because diag_valid5 is where the eleven-model table, the cs sweep
and both ControlNet tracks' early verdicts were decided.
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FOUNDATION = Path("/home/sh1/deepl/lineart")
TRACK = Path(__file__).resolve().parents[1]
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
WORKER = TRACK / "experiments/score_ipadapter_probe_one_20260930.py"
PYTHON = FOUNDATION / "venv/bin/python"
DIAG = Path("/home/sh1/deepl/lineart-controlnet-realpairs/data/diag_valid5.txt")
ALL = FOUNDATION / "dataset/pairs_480/holdout_lineart_family.txt"


def source_of(t):
    p = t.split("_")
    return p[1] if len(p) > 1 else t


def score(img, gt, cond, invert=False):
    cmd = [str(PYTHON), str(WORKER), str(img), str(gt), str(cond)]
    if invert:
        cmd.append("--invert-pred")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    diag = [l.strip() for l in open(DIAG) if l.strip()]
    pool = [l.strip() for l in open(ALL) if l.strip()]
    SDXL = Path(
        "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/outputs"
    )
    arms = {
        "degenerate cs0.0": (TRACK / "results/cs_sweep_matched_20261002/cs00.00", False),
        "conditioning map": (None, True),
        "anime cs2.5 (void probe)": (TRACK / "results/cs_sweep_20261002/cs02.50", False),
        "matched cs2.5": (TRACK / "results/cs_sweep_matched_20261002/cs02.50", False),
        # The figures the foundation asked about: these are the actual outputs
        # the eleven-model-era verdicts were read from, so their floor is
        # measured from their own density rather than borrowed.
        "SDXL bare cs2.0": (SDXL / "bare_cs2.0", False),
        "SDXL bare cs2.5": (SDXL / "bare_cs2.5", False),
        "SDXL bare cs3.0": (SDXL / "bare_cs3.0", False),
        "SDXL fine-tune cs2.0": (SDXL / "ft_cs2.0", False),
    }
    print(f"diag_valid5 = {len(diag)} tiles. Floor = this arm's outputs for OTHER "
          f"tiles (different source image) scored against THIS tile's GT.\n")
    for label, (d, invert) in arms.items():
        rows = []
        for tile in diag:
            img = (COND_DIR / tile) if d is None else d / f"{Path(tile).stem}_out.png"
            if not Path(img).exists():
                continue
            # Fix GT_i; swap in the same arm's output for every other source.
            partners = [p for p in pool if source_of(p) != source_of(tile)]
            jobs = []
            for p in partners:
                other = (COND_DIR / p) if d is None else d / f"{Path(p).stem}_out.png"
                if Path(other).exists():
                    jobs.append((other, GT_DIR / tile, COND_DIR / tile))
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(score, a, b, c, invert) for a, b, c in jobs]
                vals = [f.result()["gt_bsds_f1"] for f in futs if f.result() is not None]
            true = score(img, GT_DIR / tile, COND_DIR / tile, invert)["gt_bsds_f1"]
            rows.append((tile, true, float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)))
            print(f"  {tile:22s} true={true:.4f}  floor={np.mean(vals):.4f} "
                  f"(sd {np.std(vals, ddof=1):.4f}, n={len(vals)})  signal={true-np.mean(vals):+.4f}",
                  flush=True)
        t = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows])
        print(f"  --> {label}: true={t.mean():.4f}  FLOOR={f.mean():.4f}  "
              f"signal={(t-f).mean():+.4f}  (tile-to-tile SE of floor {f.std(ddof=1)/np.sqrt(len(f)):.4f})\n",
              flush=True)


if __name__ == "__main__":
    main()
