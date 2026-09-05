"""Visual check for the 2026-09-05 caption ablation: does dropping "manga
panel" from the *inference-time* caption (holding the manga_trained
checkpoint's weights fixed) reduce cross-hatch hallucination? See
inbox/initial_notice.md's "2026-09-05" note for the full context and the
numeric orientation_entropy result -- this only builds the paired visual.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]

CELL = 220
LABEL_H = 24
COLS = [
    ("GT", TRACK / "data" / "diag_gt_line_{s}.jpg"),
    ("manga panel (orig)", TRACK / "results/controlnet_lora_manga_20260827_eval" / "{s}_out.png"),
    ("no manga word", TRACK / "results/caption_ablation_20260905/no_manga_word" / "{s}_out.png"),
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
    out_path = TRACK / "results/caption_ablation_20260905/montage.png"
    sheet.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
