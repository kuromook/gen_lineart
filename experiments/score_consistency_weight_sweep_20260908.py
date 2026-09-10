"""Score the consistency_weight sweep (companion to
run_consistency_weight_sweep_20260908.sh): 5 weights
(0.02/0.05/0.1-anchor/0.2/0.5) x 3 controlnet_conditioning_scale values
(1.0/2.5/3.5), consistency_max_timestep held fixed at 200 throughout (one
variable isolated at a time, per this project's isolation-experiment
methodology).

Reports gt_bsds_f1 alongside the paper-profile axes
(bg_mode/near_white_frac/midtone_frac) added to
measure_lineart_profile.profile_metrics() on 2026-09-06, plus
line_width_p50/ink_ratio -- per this track's own operating rules, never read
gt_bsds_f1 or near_white_frac alone (see doc/initial_notice.md).
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
ROOT = TRACK / "results/consistency_weight_sweep_20260908"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
WEIGHTS = ["0.02", "0.05", "0.1", "0.2", "0.5"]  # 0.1 is the pre-existing anchor point
GRID_SAMPLE = "lineart_008_014"


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def eval_dir_for(weight):
    tag = f"controlnet_lora_manga_consistency_w{weight}_20260908"
    return TRACK / f"results/{tag}_eval"


def score_cell(cell):
    acc = {}
    for s in SAMPLES:
        p = cell / f"{s}_out.png"
        if not p.exists():
            return None
        gray = load_gray(p)
        m = dict(profile_metrics(p))
        gt_edge = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
        m["gt_bsds_f1"] = bipartite_match_f1(edge_map(gray), gt_edge, BSDS_TOLERANCE_PX)[0]
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def gt_reference():
    acc = {}
    for s in SAMPLES:
        path = TRACK / f"data/diag_gt_line_{s}.jpg"
        for k, v in profile_metrics(path).items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def montage(cs_values):
    cell_px, label_h = 200, 24
    font = ImageFont.load_default()
    cols = ["cond", "GT"] + [f"cs{c}" for c in cs_values]
    sheet = Image.new("L", (len(cols) * cell_px, len(WEIGHTS) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    cond_path = TRACK / f"data/diag_rough_manga_line/{GRID_SAMPLE}.jpg"
    gt_path = TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"
    for r, weight in enumerate(WEIGHTS):
        y0 = label_h + r * (cell_px + label_h)
        anchor_note = " (anchor, backfilled)" if weight == "0.1" else ""
        draw.text((4, y0 + 6), f"weight={weight}{anchor_note}", fill=0, font=font)
        paths = [cond_path, gt_path]
        d = eval_dir_for(weight)
        paths += [d / f"cs{cs}" / f"{GRID_SAMPLE}_out.png" for cs in cs_values]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = ROOT / "montage_consistency_weight_sweep.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cs_values = ["1.0", "2.5", "3.5"]
    rows = []
    for weight in WEIGHTS:
        d = eval_dir_for(weight)
        for cs in cs_values:
            cell = d / f"cs{cs}"
            if not cell.is_dir():
                continue
            scored = score_cell(cell)
            if scored is None:
                continue
            rows.append({"consistency_weight": weight, "cs": cs, **scored})

    if not rows:
        print("no completed cells yet", file=sys.stderr)
        return

    gt = gt_reference()
    csv_path = ROOT / "scores.csv"
    fields = ["consistency_weight", "cs"] + [
        k for k in rows[0] if k not in ("consistency_weight", "cs")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    for axis, label, fmt in [
        ("gt_bsds_f1", "gt_bsds_f1", "{:8.4f}"),
        ("near_white_frac", "near-white fraction (GT %.3f)" % gt["near_white_frac"], "{:8.3f}"),
        ("bg_mode", "background mode (GT %.0f)" % gt["bg_mode"], "{:8.0f}"),
        ("midtone_frac", "midtone fraction (GT %.3f)" % gt["midtone_frac"], "{:8.3f}"),
        ("line_width_p50", "line_width_p50 (GT %.2f)" % gt["line_width_p50"], "{:8.2f}"),
        ("ink_ratio", "ink_ratio (GT %.4f)" % gt["ink_ratio"], "{:8.4f}"),
    ]:
        print(f"\n=== {label} ===")
        print(f"    {'':10}" + "".join(f"{('cs' + c):>12}" for c in cs_values))
        for weight in WEIGHTS:
            line = f"    w={weight:<8}"
            for cs in cs_values:
                hit = [r for r in rows if r["consistency_weight"] == weight and r["cs"] == cs]
                line += f"{fmt.format(hit[0][axis]):>12}" if hit else f"{'-':>12}"
            print(line)

    print("\nweight=0.1 is the pre-existing checkpoint, backfilled with inference-only")
    print("eval at each cs (not retrained) -- see run_consistency_weight_sweep_20260908.sh.")
    print(f"\nsaved: {csv_path}")
    montage(cs_values)


if __name__ == "__main__":
    main()
