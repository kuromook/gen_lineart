"""Two views of the cs re-evaluation:

- montage_by_scale.png: model x scale grid on one sample. Uses
  lineart_008_014 (the sword tile), where the cross-hatch fill is the most
  legible of the five diag samples -- it is the tile whose background went
  from a dense hatch mesh at cs1.0 to clean white at cs2.5 in the
  2026-09-05 sweep.
- montage_best.png: every model at its own best-f1 scale, across all five
  samples -- i.e. the ten/eleven-model comparison redone at each model's
  useful operating point instead of the historical flat cs=1.0.
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT_ROOT = TRACK / "results/cs_reeval_20260906/outputs"
SCORES = TRACK / "results/cs_reeval_20260906/scores.csv"
GRID_SAMPLE = "lineart_008_014"

CELL = 170
LABEL_H = 22
font = ImageFont.load_default()


def build(cols, rows, cell_for, out_path, col_labels, row_labels):
    sheet = Image.new("L", (len(cols) * CELL, len(rows) * (CELL + LABEL_H) + LABEL_H), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(col_labels):
        draw.text((c * CELL + 4, 4), label, fill=0, font=font)
    for r, row_label in enumerate(row_labels):
        y0 = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y0), row_label, fill=0, font=font)
        for c in range(len(cols)):
            p = cell_for(rows[r], cols[c])
            if p is None or not p.exists():
                continue
            sheet.paste(Image.open(p).convert("L").resize((CELL, CELL)), (c * CELL, y0 + LABEL_H))
    sheet.save(out_path)
    print(f"saved: {out_path}")


def main():
    if not SCORES.exists():
        print(f"missing {SCORES} -- run score_reeval.py first", file=sys.stderr)
        return
    rows_csv = [r for r in csv.DictReader(open(SCORES)) if r["model"] != "GT(reference)"]
    models = sorted({r["model"] for r in rows_csv})
    scales = sorted({float(r["cs"]) for r in rows_csv})

    best_cs = {}
    for m in models:
        cells = [r for r in rows_csv if r["model"] == m]
        best_cs[m] = float(max(cells, key=lambda r: float(r["gt_bsds_f1"]))["cs"])

    # model x scale on one sample, with GT as the leading column
    def cell_grid(model, cs):
        if cs == "GT":
            return TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"
        return OUT_ROOT / model / f"cs{cs}" / f"{GRID_SAMPLE}_out.png"

    build(
        cols=["GT"] + [f"{c}" for c in scales],
        rows=models,
        cell_for=cell_grid,
        out_path=TRACK / "results/cs_reeval_20260906/montage_by_scale.png",
        col_labels=["GT"] + [f"cs{c}" for c in scales],
        row_labels=models,
    )

    # every model at its best scale, all samples
    def cell_best(sample, model):
        if model == "GT":
            return TRACK / f"data/diag_gt_line_{sample}.jpg"
        return OUT_ROOT / model / f"cs{best_cs[model]}" / f"{sample}_out.png"

    build(
        cols=["GT"] + models,
        rows=SAMPLES,
        cell_for=cell_best,
        out_path=TRACK / "results/cs_reeval_20260906/montage_best.png",
        col_labels=["GT"] + [f"{m[:14]} cs{best_cs[m]}" for m in models],
        row_labels=SAMPLES,
    )


if __name__ == "__main__":
    main()
