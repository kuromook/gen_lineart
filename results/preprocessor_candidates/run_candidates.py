"""Compare candidate rough->condition preprocessors for how much of the raw
rough sketch's content they retain, as a first step toward finding a
conversion method better suited to genuine rough/construction-line pencil
sketches than `lineart_anime` (which is trained to extract lines from
finished/colored anime art, not to interpret rough sketches -- see
inbox/initial_notice.md 2026-08-26 entry)."""

import numpy as np
from pathlib import Path
from PIL import Image

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
ROUGH_DIR = TRACK / "data/diag_rough_raw"
OUT_DIR = TRACK / "results/preprocessor_candidates"
SIZE = 480


def raw_ink_frac(arr):
    return (arr < 200).mean()


def load_rgb(p):
    return Image.open(p).convert("RGB").resize((SIZE, SIZE))


def main():
    from controlnet_aux import LineartAnimeDetector, LineartDetector, LineartStandardDetector

    anime_det = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    line_det = LineartDetector.from_pretrained("lllyasviel/Annotators")
    std_det = LineartStandardDetector()

    candidates = {
        "lineart_anime": lambda rgb: anime_det(rgb, image_resolution=SIZE),
        "lineart_coarse": lambda rgb: line_det(rgb, coarse=True, detect_resolution=SIZE, image_resolution=SIZE),
        "lineart_fine": lambda rgb: line_det(rgb, coarse=False, detect_resolution=SIZE, image_resolution=SIZE),
        "lineart_standard": lambda rgb: std_det(rgb, detect_resolution=SIZE),
    }

    print(f"{'sample':16} {'raw ink%':>9}", end="")
    for name in candidates:
        print(f" {name+' ink%':>18} {name+' ret%':>14}", end="")
    print()

    rows = []
    for s in SAMPLES:
        rough_path = ROUGH_DIR / f"{s}.jpg"
        rgb = load_rgb(rough_path)
        raw_arr = np.asarray(rgb.convert("L"))
        raw_frac = raw_ink_frac(raw_arr)
        row = {"sample": s, "raw_ink_pct": raw_frac * 100}
        print(f"{s:16} {raw_frac*100:9.2f}", end="")
        for name, fn in candidates.items():
            out = fn(rgb).resize((SIZE, SIZE)).convert("L")
            out_dir = OUT_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out.save(out_dir / f"{s}.jpg")
            out_arr = np.asarray(out)
            ink_frac = (out_arr > 32).mean()
            ret = ink_frac / raw_frac * 100 if raw_frac > 0 else 0.0
            row[f"{name}_ink_pct"] = ink_frac * 100
            row[f"{name}_retention_pct"] = ret
            print(f" {ink_frac*100:18.2f} {ret:14.1f}", end="")
        print()
        rows.append(row)

    print("\nMEAN:")
    for name in candidates:
        mean_ret = np.mean([r[f"{name}_retention_pct"] for r in rows])
        print(f"  {name}: retention={mean_ret:.1f}%")


if __name__ == "__main__":
    main()
