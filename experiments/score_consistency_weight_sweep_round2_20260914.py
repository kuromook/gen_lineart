"""Score round 2 of the consistency_weight sweep, merged with round 1's
already-scored points into one combined table/montage. Round 1:
0.02/0.05/0.1/0.2/0.5 (eval dirs suffixed _20260908_eval). Round 2:
0.15/0.25/0.3/0.4 (eval dirs suffixed _20260914_eval). See
run_consistency_weight_sweep_round2_20260914.sh and doc/work_log.md
"2026-09-08/10 (Track A)" for why this round fills the 0.2-0.5 gap rather
than narrowing tightly around 0.2 (round 1's weight=0.5 point only dipped
slightly below weight=0.2 on near_white_frac, suggesting the true peak may
sit between them, not at 0.2 itself).

Updated 2026-09-11 per the shared foundation's contamination-check notices
(`../lineart/inbox/note_contamination_check_both_tracks_20260911.md`,
`note_sdxl_correction_preprocessor_20260911.md`): a bare gt_bsds_f1 table is
not enough to tell whether a model is contributing anything, because a
model that barely changes its conditioning image can score deceptively well
just by copying it. Three additions, all inference-free (computed from
images already on disk):

1. A **preprocessor-alone baseline row** (`condition_only`): the manga_line
   conditioning image itself, scored directly against GT with no model in
   the loop at all. Every weight's cs row must be read against this floor,
   not in isolation -- round 1's own new-best point (w=0.2, cs2.5, f1
   0.2354) is *below* this baseline (manga_line alone scores 0.2566 on this
   same 5-tile set per the shared foundation's cross-track comparison).
2. A **vs_condition_f1 column** per weight x cs cell: bipartite_match_f1
   between the model's output and the conditioning image itself (not GT).
   This is the axis that actually distinguishes "genuinely generating
   toward GT" from "copying the input": Track B's SDXL models had
   vs_condition_f1 ~0.88 (near-copy) and f1 tracked that copying almost
   exactly. Track A's own weight sweep moves the *opposite* direction
   (vs_condition_f1 drops as consistency_weight rises while gt_bsds_f1
   rises) -- that divergence is the actual evidence the loss term is doing
   something, and must be reported alongside every f1 number from now on,
   not just the winning cell.
3. A **selection-ceiling oracle** (`condition_only_oracle`): what f1 would
   be if every false-positive stroke in the conditioning image were removed
   (perfect pruning, no drawing added). Derived directly from the
   conditioning image's own precision/recall against GT
   (bipartite_match_f1's second and third return values): pruning all false
   positives drives precision to 1.0 while recall is unchanged (recall
   only depends on which GT edges are already covered, not on how many
   extra false positives exist), so oracle_f1 = 2*recall / (1+recall). This
   answers "how much headroom is actually reachable by selection alone,
   before generation would need to add anything" -- the same method the
   shared foundation used to find its own pool's ceiling at 0.74 (vs. that
   pool's ~0.30 model ceiling). Only computed on the 5-tile diagnostic set
   here (same n as everything else in this file) -- the fuller
   292-tile holdout version is already recorded as a later step in
   doc/initial_notice.md, not duplicated here.

Same metric surface as round 1's scorer otherwise: profile_metrics()
(includes the paper_profile bg_mode/near_white_frac/midtone_frac axes) +
gt_bsds_f1.
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
ROOT = TRACK / "results/consistency_weight_sweep_round2_20260914"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
CONDITION_DIR = TRACK / "data/diag_rough_manga_line"
# weight -> which round's eval-dir suffix it lives under
WEIGHT_SUFFIX = {
    "0.02": "20260908", "0.05": "20260908", "0.1": "20260908",
    "0.15": "20260914", "0.2": "20260908", "0.25": "20260914",
    "0.3": "20260914", "0.4": "20260914", "0.5": "20260908",
}
WEIGHTS = sorted(WEIGHT_SUFFIX, key=float)
GRID_SAMPLE = "lineart_008_014"


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def eval_dir_for(weight):
    suffix = WEIGHT_SUFFIX[weight]
    tag = f"controlnet_lora_manga_consistency_w{weight}_{suffix}"
    return TRACK / f"results/{tag}_eval"


def score_cell(cell):
    """Per-sample mean of profile_metrics + gt_bsds_f1 + vs_condition_f1."""
    acc = {}
    for s in SAMPLES:
        p = cell / f"{s}_out.png"
        if not p.exists():
            return None
        gray = load_gray(p)
        pred_edge = edge_map(gray)
        m = dict(profile_metrics(p))
        gt_edge = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
        m["gt_bsds_f1"] = bipartite_match_f1(pred_edge, gt_edge, BSDS_TOLERANCE_PX)[0]
        cond_edge = edge_map(load_gray(CONDITION_DIR / f"{s}.jpg"))
        m["vs_condition_f1"] = bipartite_match_f1(pred_edge, cond_edge, BSDS_TOLERANCE_PX)[0]
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


def condition_baseline():
    """The manga_line conditioning image itself, scored against GT with no
    model involved -- the floor every weight x cs cell must be read against
    -- plus the selection-ceiling oracle derived from the same
    precision/recall (see module docstring point 3)."""
    acc = {}
    precisions, recalls = [], []
    for s in SAMPLES:
        cond_path = CONDITION_DIR / f"{s}.jpg"
        cond_gray = load_gray(cond_path)
        cond_edge = edge_map(cond_gray)
        gt_edge = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
        f1, precision, recall = bipartite_match_f1(cond_edge, gt_edge, BSDS_TOLERANCE_PX)
        m = dict(profile_metrics(cond_path))
        m["gt_bsds_f1"] = f1
        m["vs_condition_f1"] = 1.0  # the condition image compared to itself
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
        precisions.append(precision)
        recalls.append(recall)
    row = {k: float(np.mean(v)) for k, v in acc.items()}
    mean_recall = float(np.mean(recalls))
    row["oracle_f1"] = 2 * mean_recall / (1.0 + mean_recall)
    row["oracle_recall"] = mean_recall
    row["mean_precision"] = float(np.mean(precisions))
    return row


def montage(cs_values):
    cell_px, label_h = 200, 24
    font = ImageFont.load_default()
    cols = ["cond", "GT"] + [f"cs{c}" for c in cs_values]
    n_rows = len(WEIGHTS) + 1  # +1 for the condition-only baseline row
    sheet = Image.new("L", (len(cols) * cell_px, n_rows * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    cond_path = CONDITION_DIR / f"{GRID_SAMPLE}.jpg"
    gt_path = TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"

    y0 = label_h
    draw.text((4, y0 + 6), "condition_only (no model)", fill=0, font=font)
    for c, p in enumerate([cond_path, gt_path, cond_path, cond_path, cond_path][:len(cols)]):
        if Path(p).exists():
            sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                        (c * cell_px, y0 + label_h))

    for r, weight in enumerate(WEIGHTS, start=1):
        y0 = label_h + r * (cell_px + label_h)
        round_note = " (round1)" if WEIGHT_SUFFIX[weight] == "20260908" else " (round2)"
        draw.text((4, y0 + 6), f"weight={weight}{round_note}", fill=0, font=font)
        paths = [cond_path, gt_path]
        d = eval_dir_for(weight)
        paths += [d / f"cs{cs}" / f"{GRID_SAMPLE}_out.png" for cs in cs_values]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = ROOT / "montage_consistency_weight_sweep_round2.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cs_values = ["1.0", "2.5", "3.5"]

    baseline = condition_baseline()
    print("=== condition_only baseline (manga_line preprocessor, no model) ===")
    print(f"  gt_bsds_f1        = {baseline['gt_bsds_f1']:.4f}  <- every weight x cs cell below must beat this")
    print(f"  oracle_f1         = {baseline['oracle_f1']:.4f}  (selection ceiling: prune all false positives,"
          f" recall unchanged at {baseline['oracle_recall']:.4f})")
    print(f"  precision/recall  = {baseline['mean_precision']:.4f} / {baseline['oracle_recall']:.4f}")

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
        print("no completed cells yet (round 2 has not run)", file=sys.stderr)
        return

    gt = gt_reference()
    csv_path = ROOT / "scores.csv"
    fields = ["consistency_weight", "cs"] + [
        k for k in rows[0] if k not in ("consistency_weight", "cs")]
    baseline_row = {"consistency_weight": "condition_only", "cs": "-", **baseline}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields + list(baseline_row))))
        w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in baseline_row.items()})
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    for axis, label, fmt in [
        ("gt_bsds_f1", "gt_bsds_f1 (condition_only baseline %.4f, oracle ceiling %.4f)"
         % (baseline["gt_bsds_f1"], baseline["oracle_f1"]), "{:8.4f}"),
        ("vs_condition_f1", "vs_condition_f1 (1.0 = pure copy of input; lower = genuinely different)", "{:8.4f}"),
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

    beats_baseline = [r for r in rows if r["gt_bsds_f1"] > baseline["gt_bsds_f1"]]
    print(f"\ncells beating the condition_only baseline ({baseline['gt_bsds_f1']:.4f}): "
          f"{len(beats_baseline)}/{len(rows)}")
    for r in sorted(beats_baseline, key=lambda r: -r["gt_bsds_f1"]):
        print(f"  w={r['consistency_weight']} cs={r['cs']}: f1={r['gt_bsds_f1']:.4f}, "
              f"vs_condition_f1={r['vs_condition_f1']:.4f}")

    print("\nweight=0.02/0.05/0.1/0.2/0.5 are round 1 (2026-09-08), reused as-is.")
    print("weight=0.15/0.25/0.3/0.4 are round 2 (2026-09-14).")
    print(f"\nsaved: {csv_path}")
    montage(cs_values)


if __name__ == "__main__":
    main()
