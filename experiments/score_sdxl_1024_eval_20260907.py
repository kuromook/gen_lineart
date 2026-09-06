"""Score the 1024 fine-tune against the bare ControlNet it has to beat, and
build the montage beside the table.

Usage: score_sdxl_1024_eval_20260907.py <eval-dir>

The eval dir holds cs<N>/ (fine-tuned) and base_cs<N>/ (bare) pairs written
by run_controlnet_lora_sdxl_1024_20260907.sh.

The verdict this prints is deliberately narrow: does the fine-tune beat the
bare ControlNet at each one's own best cs? As of 2026-09-06 the bare
lineart_anime ControlNet at 1024 reaches gt_bsds_f1 0.2582 at cs2.5 -- above
every model in the predecessor track's 11-model table -- while both
512-trained LoRAs came in below it at every resolution and scale tested. If
the 1024 fine-tune cannot clear that bar, the finding is that this dataset
and recipe do not improve on the bare ControlNet, which is a result worth
having rather than a failed run.

Read f1 with the paper axes, never alone: in this study a grey wash scored
the best f1 in the sweep (anime_base 512/cs1.0, f1 0.2263 at 3.1% near
white), and a nearly blank page scored 0.2121 (manga_base 1024/cs1.0, 87.8%
near white with no sword visible in the montage).
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402
from score_resolution_sweep_20260906 import paper_metrics  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
GRID_SAMPLE = "lineart_008_014"


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def score_cell(cell):
    acc = {}
    for s in SAMPLES:
        p = cell / f"{s}_out.png"
        gray = load_gray(p)
        m = dict(profile_metrics(p), **paper_metrics(gray))
        gt = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
        m["gt_bsds_f1"] = bipartite_match_f1(edge_map(gray), gt, BSDS_TOLERANCE_PX)[0]
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def gt_reference():
    acc = {}
    for s in SAMPLES:
        path = TRACK / f"data/diag_gt_line_{s}.jpg"
        for k, v in dict(profile_metrics(path), **paper_metrics(load_gray(path))).items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def montage(root, scales, rows_out):
    cell_px, label_h = 200, 22
    font = ImageFont.load_default()
    cols = ["GT"] + [f"ft cs{c}" for c in scales] + [f"bare cs{c}" for c in scales]
    sheet = Image.new("L", (len(cols) * cell_px, len(SAMPLES) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 5), label, fill=0, font=font)
    for r, s in enumerate(SAMPLES):
        y = label_h + r * (cell_px + label_h)
        draw.text((4, y + 5), s, fill=0, font=font)
        paths = [TRACK / f"data/diag_gt_line_{s}.jpg"]
        paths += [root / f"cs{c}" / f"{s}_out.png" for c in scales]
        paths += [root / f"base_cs{c}" / f"{s}_out.png" for c in scales]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y + label_h))
    out = root / "montage_ft_vs_bare.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    scales = sorted(
        {d.name.split("cs")[1] for d in root.iterdir()
         if d.is_dir() and (d / ".complete").exists()},
        key=float,
    )
    if not scales:
        print(f"no completed cells under {root}", file=sys.stderr)
        return

    rows = []
    for cs in scales:
        for kind, prefix in (("finetuned", "cs"), ("bare", "base_cs")):
            cell = root / f"{prefix}{cs}"
            if (cell / ".complete").exists():
                rows.append({"variant": kind, "cs": float(cs), **score_cell(cell)})

    gt = gt_reference()
    csv_path = root / "scores.csv"
    fields = ["variant", "cs"] + [k for k in rows[0] if k not in ("variant", "cs")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'variant':11}{'cs':>6}{'f1':>9}{'ink_ratio':>11}{'line_w_p50':>12}"
          f"{'near_wht':>10}{'midtone':>9}{'orient_ent':>12}")
    for r in sorted(rows, key=lambda r: (r["variant"], r["cs"])):
        print(f"{r['variant']:11}{r['cs']:6.1f}{r['gt_bsds_f1']:9.4f}{r['ink_ratio']:11.4f}"
              f"{r['line_width_p50']:12.2f}{r['near_white_frac']*100:9.1f}%"
              f"{r['midtone_frac']*100:8.1f}%{r['orientation_entropy']:12.4f}")
    print(f"{'GT':11}{'':6}{'':9}{gt['ink_ratio']:11.4f}{gt['line_width_p50']:12.2f}"
          f"{gt['near_white_frac']*100:9.1f}%{gt['midtone_frac']*100:8.1f}%"
          f"{gt['orientation_entropy']:12.4f}")

    best = {}
    for kind in ("finetuned", "bare"):
        cells = [r for r in rows if r["variant"] == kind]
        if cells:
            best[kind] = max(cells, key=lambda r: r["gt_bsds_f1"])
    print("\n=== verdict: fine-tune vs bare ControlNet, each at its own best cs ===")
    for kind, r in best.items():
        print(f"  {kind:10} cs{r['cs']:<4} f1 {r['gt_bsds_f1']:.4f}  "
              f"near_white {r['near_white_frac']*100:.1f}%  line_w {r['line_width_p50']:.2f}")
    if len(best) == 2:
        delta = best["finetuned"]["gt_bsds_f1"] - best["bare"]["gt_bsds_f1"]
        print(f"  delta (fine-tuned - bare) = {delta:+.4f} -> "
              f"{'fine-tuning helps' if delta > 0 else 'fine-tuning does NOT beat the bare ControlNet'}")
    print("  (confirm against the montage before acting on this -- f1 alone cannot")
    print("   tell a grey wash or a blank page from line art.)")
    print(f"\nsaved: {csv_path}")
    montage(root, scales, rows)


if __name__ == "__main__":
    main()
