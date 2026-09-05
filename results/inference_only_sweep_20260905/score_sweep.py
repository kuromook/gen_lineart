"""Score every variant on the corrected axis set. As of 2026-09-05,
orientation_entropy alone is known to be insufficient: it cannot tell a
cross-hatch mesh apart from the smoothly-curved boundary of a flat solid
fill (found via the text_focus case in the 63-tag sweep; see
inbox/initial_notice.md's 運用ルール). So line_width_p50 (GT ~3.72;
much thicker = escaped into solid fill rather than strokes) and ink_ratio
(GT ~0.035; the trained models all sit ~10x above it) are reported
alongside, all from measure_lineart_profile.py's profile_metrics().

Ranked by distance from GT on ink_ratio, since that is the axis where the
hallucination is largest and least ambiguous.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import VARIANTS, VARIANTS_ROUND2, VARIANTS_ROUND3  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT_ROOT = TRACK / "results/inference_only_sweep_20260905/outputs"
AXES = ["orientation_entropy", "line_width_p50", "ink_ratio", "components_per_1k_ink_px"]
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0


def safe_dirname(label):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", label)


def load_gray_array(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def mean_profile(path_tpl, with_f1=False):
    vals = {a: [] for a in AXES}
    f1s, precisions, recalls = [], [], []
    for s in SAMPLES:
        p = Path(str(path_tpl).format(s=s))
        m = profile_metrics(p)
        for a in AXES:
            vals[a].append(m[a])
        if with_f1:
            gt_edge = edge_map(load_gray_array(TRACK / f"data/diag_gt_line_{s}.jpg"))
            f1, precision, recall = bipartite_match_f1(
                edge_map(load_gray_array(p)), gt_edge, BSDS_TOLERANCE_PX
            )
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
    out = {a: float(np.mean(vals[a])) for a in AXES}
    if with_f1:
        out["gt_bsds_f1"] = float(np.mean(f1s))
        out["precision"] = float(np.mean(precisions))
        out["recall"] = float(np.mean(recalls))
    return out


def main():
    gt = mean_profile(TRACK / "data" / "diag_gt_line_{s}.jpg")
    gt.update({"gt_bsds_f1": float("nan"), "precision": float("nan"), "recall": float("nan")})

    cols = AXES + ["gt_bsds_f1", "precision", "recall"]
    rows = []
    for v in VARIANTS + VARIANTS_ROUND2 + VARIANTS_ROUND3:
        prof = mean_profile(OUT_ROOT / safe_dirname(v["label"]) / "{s}_out.png", with_f1=True)
        rows.append({"label": v["label"], **prof})

    rows.sort(key=lambda r: abs(r["ink_ratio"] - gt["ink_ratio"]))

    with open(TRACK / "results/inference_only_sweep_20260905/scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label"] + cols)
        w.writeheader()
        w.writerow({"label": "GT(reference)", **{a: round(gt[a], 4) for a in cols}})
        for r in rows:
            w.writerow({"label": r["label"], **{a: round(r[a], 4) for a in cols}})

    hdr = f"{'variant':22}" + "".join(f"{a[:16]:>18}" for a in cols)
    print(hdr)
    print(f"{'GT(reference)':22}" + "".join(f"{gt[a]:18.4f}" for a in cols))
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:22}" + "".join(f"{r[a]:18.4f}" for a in cols))
    print()
    print("(sorted by |ink_ratio - GT|, closest first. line_width_p50 much above GT's")
    print(" ~3.7 means the variant escaped into solid fill, not cleaner strokes.)")
    print(f"\nsaved: {TRACK / 'results/inference_only_sweep_20260905/scores.csv'}")


if __name__ == "__main__":
    main()
