"""Visual review sheets for the resolution sweep. Numbers alone cannot tell
a solid fill from lines here (that is the whole reason this track's 運用ルール
requires a montage beside every metric), and the two known SDXL failure
modes -- a black blob at 512 and a scribble texture at 1024 -- can score
similarly while looking nothing alike.

- montage_by_res.png: one sample, every (model, cs) row against the three
  resolutions, with the conditioning input and GT as leading columns. This is
  the sheet that answers "does 1024 change the failure mode?".
- montage_best.png: every model at its own best-f1 cell, across all five
  samples.
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
ROOT = TRACK / "results/resolution_sweep_20260906"
OUT_ROOT = ROOT / "outputs"
SCORES = ROOT / "scores.csv"
# The sword tile: the most legible of the five diag samples for telling a
# hatch/scribble background apart from clean white (used the same way by the
# previous track's cs re-evaluation).
GRID_SAMPLE = "lineart_008_014"
COND_DIR = {"anime": "data/diag_rough_lineart_coarse", "manga": "data/diag_rough_manga_line"}

CELL = 190
LABEL_H = 24
font = ImageFont.load_default()


def build(rows, cols, cell_for, out_path, row_labels, col_labels):
    sheet = Image.new("L", (len(cols) * CELL, len(rows) * (CELL + LABEL_H) + LABEL_H), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(col_labels):
        draw.text((c * CELL + 4, 6), label, fill=0, font=font)
    for r, row_label in enumerate(row_labels):
        y0 = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y0 + 6), row_label, fill=0, font=font)
        for c in range(len(cols)):
            p = cell_for(rows[r], cols[c])
            if p is None or not Path(p).exists():
                continue
            sheet.paste(Image.open(p).convert("L").resize((CELL, CELL)), (c * CELL, y0 + LABEL_H))
    sheet.save(out_path)
    print(f"saved: {out_path}")


def main():
    if not SCORES.exists():
        print(f"missing {SCORES} -- run score_resolution_sweep_20260906.py first", file=sys.stderr)
        return
    rows_csv = [
        r for r in csv.DictReader(open(SCORES))
        if r["model"] != "GT(reference)" and not r["model"].endswith("_nooffload")
    ]
    models = sorted({r["model"] for r in rows_csv})
    resolutions = sorted({int(r["resolution"]) for r in rows_csv})
    scales = sorted({float(r["cs"]) for r in rows_csv})

    def cond_for(model, sample):
        family = "manga" if model.startswith("manga") else "anime"
        return TRACK / COND_DIR[family] / f"{sample}.jpg"

    # (model, cs) x resolution, on one sample
    grid_rows = [(m, cs) for m in models for cs in scales]

    def cell_grid(row, col):
        model, cs = row
        if col == "cond":
            return cond_for(model, GRID_SAMPLE)
        if col == "GT":
            return TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"
        return OUT_ROOT / model / f"res{col}_cs{cs}" / f"{GRID_SAMPLE}_out.png"

    build(
        rows=grid_rows,
        cols=["cond", "GT"] + resolutions,
        cell_for=cell_grid,
        out_path=ROOT / "montage_by_res.png",
        row_labels=[f"{m} cs{cs}" for m, cs in grid_rows],
        col_labels=["cond", "GT"] + [f"res{r}" for r in resolutions],
    )

    # every model at its best cell, all samples
    best = {}
    for m in models:
        cells = [r for r in rows_csv if r["model"] == m]
        top = max(cells, key=lambda r: float(r["gt_bsds_f1"]))
        best[m] = (int(top["resolution"]), float(top["cs"]))

    def cell_best(sample, model):
        if model == "GT":
            return TRACK / f"data/diag_gt_line_{sample}.jpg"
        res, cs = best[model]
        return OUT_ROOT / model / f"res{res}_cs{cs}" / f"{sample}_out.png"

    build(
        rows=SAMPLES,
        cols=["GT"] + models,
        cell_for=cell_best,
        out_path=ROOT / "montage_best.png",
        row_labels=SAMPLES,
        col_labels=["GT"] + [f"{m} r{best[m][0]} cs{best[m][1]}" for m in models],
    )


if __name__ == "__main__":
    main()
