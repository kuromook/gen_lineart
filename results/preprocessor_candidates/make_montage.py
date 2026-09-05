from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT_DIR = TRACK / "results/preprocessor_candidates"

CELL = 200
LABEL_H = 24
COLS = [
    ("raw rough", TRACK / "data/diag_rough_raw" / "{s}.jpg"),
    ("GT", TRACK / "data" / "diag_gt_line_{s}.jpg"),
    ("lineart_anime", OUT_DIR / "lineart_anime" / "{s}.jpg"),
    ("lineart_coarse", OUT_DIR / "lineart_coarse" / "{s}.jpg"),
    ("lineart_fine", OUT_DIR / "lineart_fine" / "{s}.jpg"),
    ("lineart_standard", OUT_DIR / "lineart_standard" / "{s}.jpg"),
]

font = ImageFont.load_default()


def main():
    n_rows, n_cols = len(SAMPLES), len(COLS)
    sheet = Image.new("L", (n_cols * CELL, n_rows * (CELL + LABEL_H) + LABEL_H), 255)
    draw = ImageDraw.Draw(sheet)
    for c, (label, _) in enumerate(COLS):
        draw.text((c * CELL + 4, 4), label, fill=0, font=font)
    for r, sample in enumerate(SAMPLES):
        y0 = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y0), sample, fill=0, font=font)
        for c, (label, path_tpl) in enumerate(COLS):
            p = Path(str(path_tpl).format(s=sample))
            img = Image.open(p).convert("L").resize((CELL, CELL))
            sheet.paste(img, (c * CELL, y0 + LABEL_H))
    out_path = OUT_DIR / "montage.png"
    sheet.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
