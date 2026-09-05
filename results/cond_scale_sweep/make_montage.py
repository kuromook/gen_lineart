"""Contact sheet for the controlnet_conditioning_scale sweep: one row per
diag5 sample, columns = [rough input, GT line, cs0.50 .. cs2.00]."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
ROUGH_DIR = TRACK / "data/diag_rough_raw"
SWEEP_DIR = TRACK / "results/cond_scale_sweep"
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
SCALES = ["0.50", "0.75", "1.00", "1.25", "1.50", "1.75", "2.00"]

CELL = 200
LABEL_H = 24
COLS = [("rough", None)] + [("GT", None)] + [(f"cs{s}", s) for s in SCALES]

font = ImageFont.load_default()


def cell_image(sample, col_kind, scale):
    if col_kind == "rough":
        p = ROUGH_DIR / f"{sample}.jpg"
    elif col_kind == "GT":
        p = TRACK / "data" / f"diag_gt_line_{sample}.jpg"
    else:
        p = SWEEP_DIR / f"cs{scale}" / f"{sample}_out.png"
    img = Image.open(p).convert("L").resize((CELL, CELL))
    return img


def main():
    n_rows, n_cols = len(SAMPLES), len(COLS)
    sheet = Image.new("L", (n_cols * CELL, n_rows * (CELL + LABEL_H) + LABEL_H), 255)
    draw = ImageDraw.Draw(sheet)

    for c, (label, scale) in enumerate(COLS):
        draw.text((c * CELL + 4, 4), label, fill=0, font=font)

    for r, sample in enumerate(SAMPLES):
        y0 = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y0), sample, fill=0, font=font)
        for c, (label, scale) in enumerate(COLS):
            col_kind = "rough" if label == "rough" else ("GT" if label == "GT" else "out")
            img = cell_image(sample, col_kind, scale)
            sheet.paste(img, (c * CELL, y0 + LABEL_H))

    out_path = SWEEP_DIR / "montage.png"
    sheet.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
