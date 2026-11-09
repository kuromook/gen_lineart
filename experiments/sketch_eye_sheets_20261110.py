"""Screening sheets for the eye check of v4 sketches (2026-11-10, author: 下絵のクレンジング).
Each sheet: 4 pages, each as [sketch | line] at reduced scale, labelled with an index, key and sketch layer name.
Screening only: any page whose content is in doubt goes to the author as native crops (discipline 6).
Writes to results/sketch_eye_check_20261110/sheets/ (ignored; contains manuscript pixels)."""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "tools/evaluation")
from pair_feature_tiles import load_jobs

H = 640
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--out", default="results/sketch_eye_check_20261110/sheets")
    a = ap.parse_args()
    qc = {f"{x['work_id']}|{x['slug']}|{x['page']}": x for x in csv.DictReader(open("data/clip_pairs_v4/clip_pairs/clip_pairs_qc_v4.csv", encoding="utf-8"))}
    jobs = load_jobs("data/clip_pairs_v4", a.only)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = open(out.parent / "sheet_index.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(idx); w.writerow(["sheet", "slot", "n", "key", "sketch_name", "aligned"])
    for s0 in range(0, len(jobs), 4):
        tiles = []
        for slot, (key, sp, lp) in enumerate(jobs[s0:s0 + 4]):
            n = s0 + slot
            ims = [Image.open(p).convert("L") for p in (sp, lp)]
            ims = [im.resize((round(im.width * H / im.height), H), Image.LANCZOS) for im in ims]
            pw = sum(im.width for im in ims) + 10
            t = Image.new("L", (pw, H + 34), 255)
            t.paste(ims[0], (0, 34)); t.paste(ims[1], (ims[0].width + 10, 34))
            d = ImageDraw.Draw(t); x = qc[key]
            d.text((2, 4), f"#{n} {key.split('|')[0]} {key.split('|')[2]} | 下絵層: {x['sketch_name'][:24]} | aligned={x['aligned']}", font=FONT, fill=0)
            d.line([(ims[0].width + 5, 34), (ims[0].width + 5, H + 34)], fill=0, width=2)
            tiles.append(t); w.writerow([s0 // 4, slot, n, key, x["sketch_name"], x["aligned"]])
        tw = max(t.width for t in tiles)
        sheet = Image.new("L", (tw * 2 + 20, (H + 34) * 2 + 20), 128)
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * (tw + 20), (i // 2) * (H + 54)))
        sheet.save(out / f"sheet_{s0 // 4:04d}.jpg", quality=88)
    print(f"{len(jobs)} pages -> {(len(jobs) + 3) // 4} sheets in {out}")

if __name__ == "__main__":
    main()
