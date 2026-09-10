"""Score the SD1.5 2x3 ControlNet x conditioning cross (companion to
run_grey_source_cross_sd15_20260908.sh), printed as a matrix on the
paper-white axis so the question it was built for -- does the grey wash
follow the ControlNet or the conditioning preprocessor? -- is answered by
reading a row against a column. Adapted from Track B's
../lineart-controlnet-sdxl-fidelity/experiments/score_grey_source_cross_20260906.py;
simpler here because `profile_metrics()` already includes the
`bg_mode`/`near_white_frac`/`midtone_frac` paper-profile axes directly (added
to the shared module 2026-09-06), so no separate `paper_metrics()` import is
needed.

Also writes a montage: this axis is about tone, and a number like
near_white_frac is exactly the kind of summary that can agree with a hand
ranking for the wrong reason, so the sheet goes next to the table.
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
ROOT = TRACK / "results/grey_source_cross_sd15_20260908"
OUT_ROOT = ROOT / "outputs"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
CONTROLNETS = ["cnLineart", "cnAnime"]
CONDITIONS = ["condAnime", "condCoarse", "condManga"]
COND_DIR = {
    "condAnime": "data/diag_rough_lineart_anime",
    "condCoarse": "data/diag_rough_lineart_coarse",
    "condManga": "data/diag_rough_manga_line",
}
GRID_SAMPLE = "lineart_008_014"


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def score_cell(cell):
    acc = {}
    for s in SAMPLES:
        p = cell / f"{s}_out.png"
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
    rows = [(cn, cond) for cn in CONTROLNETS for cond in CONDITIONS]
    cols = ["cond", "GT"] + [f"cs{c}" for c in cs_values]
    sheet = Image.new("L", (len(cols) * cell_px, len(rows) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    for r, (cn, cond) in enumerate(rows):
        y0 = label_h + r * (cell_px + label_h)
        draw.text((4, y0 + 6), f"{cn} x {cond}", fill=0, font=font)
        paths = [TRACK / COND_DIR[cond] / f"{GRID_SAMPLE}.jpg",
                 TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"]
        paths += [OUT_ROOT / f"{cn}_{cond}" / f"cs{cs}" / f"{GRID_SAMPLE}_out.png"
                  for cs in cs_values]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = ROOT / "montage_grey_source_sd15.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    rows = []
    for cn in CONTROLNETS:
        for cond in CONDITIONS:
            d = OUT_ROOT / f"{cn}_{cond}"
            if not d.is_dir():
                continue
            for cell in sorted(d.iterdir()):
                if not (cell / ".complete").exists():
                    continue
                cs = cell.name[2:]  # "cs1.0" -> "1.0"
                rows.append({"controlnet": cn, "conditioning": cond, "cs": cs,
                             **score_cell(cell)})
    if not rows:
        print("no completed cells", file=sys.stderr)
        return

    gt = gt_reference()
    csv_path = ROOT / "scores.csv"
    fields = ["controlnet", "conditioning", "cs"] + [
        k for k in rows[0] if k not in ("controlnet", "conditioning", "cs")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    cs_values = sorted({r["cs"] for r in rows}, key=float)
    for axis, label, fmt in [
        ("near_white_frac", "near-white fraction (GT %.3f)" % gt["near_white_frac"], "{:8.3f}"),
        ("bg_mode", "background mode (GT %.0f)" % gt["bg_mode"], "{:8.0f}"),
        ("midtone_frac", "midtone fraction (GT %.3f)" % gt["midtone_frac"], "{:8.3f}"),
        ("gt_bsds_f1", "gt_bsds_f1", "{:8.4f}"),
        ("line_width_p50", "line_width_p50 (GT %.2f)" % gt["line_width_p50"], "{:8.2f}"),
        ("ink_ratio", "ink_ratio (GT %.4f)" % gt["ink_ratio"], "{:8.4f}"),
    ]:
        print(f"\n=== {label} ===")
        for cs in cs_values:
            print(f"  cs{cs}")
            print(f"    {'':16}" + "".join(f"{c:>12}" for c in CONDITIONS))
            for cn in CONTROLNETS:
                line = f"    {cn:16}"
                for cond in CONDITIONS:
                    hit = [r for r in rows if r["controlnet"] == cn
                           and r["conditioning"] == cond and r["cs"] == cs]
                    line += f"{fmt.format(hit[0][axis]):>12}" if hit else f"{'-':>12}"
                print(line)

    print("\nReading: if a row is uniform and the rows differ, the effect belongs to the")
    print("ControlNet; if a column is uniform and the columns differ, it belongs to the")
    print("conditioning preprocessor. SD1.5 has no manga_line ControlNet, so this is a")
    print("2x3 grid (2 ControlNets x 3 preprocessors), not a literal 2x2.")
    print(f"\nsaved: {csv_path}")
    montage(cs_values)


if __name__ == "__main__":
    main()
