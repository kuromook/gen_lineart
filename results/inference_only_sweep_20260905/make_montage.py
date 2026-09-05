"""Visual check for the inference-only sweep. Columns picked to span the
result: GT, baseline, the conditioning-scale ladder (the axis that moved
ink_ratio/components_per_1k_ink_px toward GT), the best CFG variant, one
style prompt, and the two negative-prompt variants that scored worst --
neg_hatching in particular is the cautionary case (line_width_p50 14.85,
i.e. "no hatching" pushed it into solid black fill instead).
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
OUT = TRACK / "results/inference_only_sweep_20260905/outputs"

CELL = 190
LABEL_H = 24
COLS = [
    ("GT", TRACK / "data" / "diag_gt_line_{s}.jpg"),
    ("baseline cs1.0", OUT / "baseline" / "{s}_out.png"),
    ("cs1.2", OUT / "cs1.2" / "{s}_out.png"),
    ("cs1.5", OUT / "cs1.5" / "{s}_out.png"),
    ("cs2.0 (best)", OUT / "cs2.0" / "{s}_out.png"),
    ("cfg10.0", OUT / "cfg10.0" / "{s}_out.png"),
    ("style_coloringbook", OUT / "style_coloringbook" / "{s}_out.png"),
    ("neg_combined", OUT / "neg_combined" / "{s}_out.png"),
    ("neg_hatching (worst)", OUT / "neg_hatching" / "{s}_out.png"),
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
    out_path = TRACK / "results/inference_only_sweep_20260905/montage.png"
    sheet.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
