from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]

CELL = 190
LABEL_H = 24
COLS = [
    ("raw rough", TRACK / "data/diag_rough_raw" / "{s}.jpg"),
    ("GT", TRACK / "data" / "diag_gt_line_{s}.jpg"),
    ("anime", TRACK / "results/preprocessed_cond_check/cs1.00" / "{s}_out.png"),
    ("coarse", TRACK / "results/controlnet_lora_coarse_20260827_eval" / "{s}_out.png"),
    ("sd15_lineart", TRACK / "results/controlnet_lora_lineartsd15_20260827_eval" / "{s}_out.png"),
    ("manga_line", TRACK / "results/controlnet_lora_manga_20260827_eval" / "{s}_out.png"),
    ("manga_r32", TRACK / "results/controlnet_lora_manga_rank32_20260902_eval" / "{s}_out.png"),
    ("sdxl", TRACK / "results/controlnet_lora_sdxl_20260829_eval" / "{s}_out.png"),
    ("sdxl_manga", TRACK / "results/controlnet_lora_sdxl_manga_20260830_eval" / "{s}_out.png"),
    ("manga_ep2", TRACK / "results/controlnet_lora_manga_epoch2_20260903_eval" / "{s}_out.png"),
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
    out_path = TRACK / "results/eight_model_comparison/montage.png"
    sheet.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
