"""Score every (model, resolution, cs) cell from
experiments/run_resolution_sweep_20260906.sh.

Axes follow the track's 運用ルール (doc/initial_notice.md): orientation_entropy
alone cannot separate a hatch mesh from the smooth boundary of a solid fill,
so line_width_p50 (GT ~3.72) and ink_ratio (GT ~0.0353) are always reported
beside it. This matters more here than anywhere else -- sdxl_trained's
headline 0.0945 came with line_width_p50 20.64, i.e. it was scoring a solid
fill, not lines.

The *_nooffload cells are both the control for applying --cpu-offload
uniformly and the anchor against the historical 11-model table: if they do
not reproduce 0.0945 / 0.1226, the model->checkpoint/conditioning mapping is
wrong and nothing else in the table should be trusted.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# GT line art is bimodal: white paper plus black ink. An output can score well
# on gt_bsds_f1 while having no paper at all -- anime_base at 512/cs1.0 scores
# the sweep's best f1 (0.2263) with a background mode of 164 and only 3.1% of
# pixels near white, i.e. a grey shaded image whose thin strokes happen to land
# near GT strokes (user observation, 2026-09-06). These axes make that visible
# in the table instead of only in the montage.
NEAR_WHITE = 224
MIDTONE_LO, MIDTONE_HI = 64, 192

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT_ROOT = TRACK / "results/resolution_sweep_20260906/outputs"
SCORES_CSV = TRACK / "results/resolution_sweep_20260906/scores.csv"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
PROFILE_AXES = ["orientation_entropy", "line_width_p50", "ink_ratio", "components_per_1k_ink_px"]
PAPER_AXES = ["bg_mode", "near_white_frac", "midtone_frac"]
CELL_RE = re.compile(r"^res(\d+)_cs([\d.]+)$")

# gt_bsds_f1 at 512/cs1.0 from the 11-model table in
# doc/track_controlnet_realpairs_work_log.md, for the anchor check.
HISTORICAL_F1 = {"anime_lora_nooffload": 0.0945, "manga_lora_nooffload": 0.1226}


def load_gray_array(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def paper_metrics(gray):
    """Is there white paper under the ink, or is the page a grey wash?"""
    return {
        "bg_mode": float(np.bincount(gray.astype(np.uint8).ravel(), minlength=256).argmax()),
        "near_white_frac": float((gray >= NEAR_WHITE).mean()),
        "midtone_frac": float(((gray > MIDTONE_LO) & (gray < MIDTONE_HI)).mean()),
    }


def score_dir(out_dir):
    vals = {a: [] for a in PROFILE_AXES + PAPER_AXES}
    f1s, precisions, recalls = [], [], []
    for s in SAMPLES:
        p = out_dir / f"{s}_out.png"
        m = profile_metrics(p)
        for a in PROFILE_AXES:
            vals[a].append(m[a])
        gray = load_gray_array(p)
        for a, v in paper_metrics(gray).items():
            vals[a].append(v)
        gt_edge = edge_map(load_gray_array(TRACK / f"data/diag_gt_line_{s}.jpg"))
        f1, precision, recall = bipartite_match_f1(
            edge_map(gray), gt_edge, BSDS_TOLERANCE_PX
        )
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    out = {a: float(np.mean(vals[a])) for a in PROFILE_AXES + PAPER_AXES}
    out["gt_bsds_f1"] = float(np.mean(f1s))
    out["precision"] = float(np.mean(precisions))
    out["recall"] = float(np.mean(recalls))
    return out


def gt_reference():
    vals = {a: [] for a in PROFILE_AXES + PAPER_AXES}
    for s in SAMPLES:
        path = TRACK / f"data/diag_gt_line_{s}.jpg"
        m = profile_metrics(path)
        for a in PROFILE_AXES:
            vals[a].append(m[a])
        for a, v in paper_metrics(load_gray_array(path)).items():
            vals[a].append(v)
    return {a: float(np.mean(vals[a])) for a in PROFILE_AXES + PAPER_AXES}


def main():
    rows = []
    for model_dir in sorted(d for d in OUT_ROOT.iterdir() if d.is_dir()):
        for cell in sorted(model_dir.iterdir()):
            match = CELL_RE.match(cell.name)
            if not match or not (cell / ".complete").exists():
                if match:
                    print(f"(skipping incomplete cell: {model_dir.name} {cell.name})", file=sys.stderr)
                continue
            rows.append(
                {
                    "model": model_dir.name,
                    "resolution": int(match.group(1)),
                    "cs": float(match.group(2)),
                    **score_dir(cell),
                }
            )
    if not rows:
        print("no completed cells found", file=sys.stderr)
        return

    gt = gt_reference()
    cols = ["gt_bsds_f1", "precision", "recall"] + PROFILE_AXES + PAPER_AXES
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "resolution", "cs"] + cols)
        w.writeheader()
        w.writerow(
            {"model": "GT(reference)", "resolution": "", "cs": "",
             **{a: round(gt[a], 4) for a in PROFILE_AXES + PAPER_AXES}}
        )
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    print("=== anchor check: 512/cs1.0 without --cpu-offload vs the 11-model table ===")
    worst = 0.0
    for r in rows:
        hist = HISTORICAL_F1.get(r["model"])
        if hist is None or r["resolution"] != 512 or r["cs"] != 1.0:
            continue
        delta = r["gt_bsds_f1"] - hist
        worst = max(worst, abs(delta))
        print(f"{r['model']:24}{r['gt_bsds_f1']:9.4f}  historical {hist:.4f}  delta {delta:+.4f}")
    print(f"max |delta| = {worst:.4f} -> "
          f"{'OK' if worst < 0.005 else 'MISMATCH -- fix the mapping before reading anything below'}")

    print("\n=== offload control: 512/cs1.0 with vs without --cpu-offload ===")
    for base in ("anime_lora", "manga_lora"):
        pair = {}
        for r in rows:
            if r["resolution"] == 512 and r["cs"] == 1.0:
                if r["model"] == base:
                    pair["offload"] = r["gt_bsds_f1"]
                elif r["model"] == f"{base}_nooffload":
                    pair["plain"] = r["gt_bsds_f1"]
        if len(pair) == 2:
            print(f"{base:24}offload {pair['offload']:.4f}  plain {pair['plain']:.4f}  "
                  f"delta {pair['offload'] - pair['plain']:+.4f}")

    resolutions = sorted({r["resolution"] for r in rows if not r["model"].endswith("_nooffload")})
    scales = sorted({r["cs"] for r in rows if not r["model"].endswith("_nooffload")})
    print(f"\n=== gt_bsds_f1 by resolution x cs (GT ink_ratio {gt['ink_ratio']:.4f}, "
          f"line_width_p50 {gt['line_width_p50']:.2f}) ===")
    for model in sorted({r["model"] for r in rows if not r["model"].endswith("_nooffload")}):
        print(f"\n{model}")
        print(f"{'':10}" + "".join(f"{f'cs{c}':>10}" for c in scales))
        for res in resolutions:
            line = f"{f'res{res}':10}"
            for cs in scales:
                cell = [r for r in rows if r["model"] == model and r["resolution"] == res and r["cs"] == cs]
                line += f"{cell[0]['gt_bsds_f1']:10.4f}" if cell else f"{'-':>10}"
            print(line)

    print("\n=== each model at its best cell (full axes) ===")
    print("note: gt_bsds_f1 rewards strokes landing near GT strokes and says nothing")
    print("      about whether there is white paper under them -- read it with")
    print("      near_white/midtone, which separate line art from a grey wash.")
    print(f"{'model':24}{'res':>6}{'cs':>6}{'f1':>9}{'ink_ratio':>11}{'line_w_p50':>12}"
          f"{'orient_ent':>12}{'bg_mode':>9}{'near_wht':>10}{'midtone':>9}")
    best_rows = []
    for model in sorted({r["model"] for r in rows}):
        cells = [r for r in rows if r["model"] == model]
        best_rows.append(max(cells, key=lambda r: r["gt_bsds_f1"]))
    for r in sorted(best_rows, key=lambda r: -r["gt_bsds_f1"]):
        print(f"{r['model']:24}{r['resolution']:6d}{r['cs']:6.1f}{r['gt_bsds_f1']:9.4f}"
              f"{r['ink_ratio']:11.4f}{r['line_width_p50']:12.2f}{r['orientation_entropy']:12.4f}"
              f"{r['bg_mode']:9.0f}{r['near_white_frac']*100:9.1f}%{r['midtone_frac']*100:8.1f}%")
    print(f"{'GT(reference)':24}{'':6}{'':6}{'':9}{gt['ink_ratio']:11.4f}"
          f"{gt['line_width_p50']:12.2f}{gt['orientation_entropy']:12.4f}"
          f"{gt['bg_mode']:9.0f}{gt['near_white_frac']*100:9.1f}%{gt['midtone_frac']*100:8.1f}%")

    print(f"\nsaved: {SCORES_CSV}")


if __name__ == "__main__":
    main()
