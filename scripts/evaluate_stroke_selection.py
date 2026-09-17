"""Track C step 2: evaluate the trained keep/drop classifier on the group-A
holdout (192 lineart_family tiles), against the preprocessor baseline and
the pixel-level deletion oracle ceiling, using the same metric
(`bipartite_match_f1`) and the same conditioning batch throughout so all
three numbers are directly comparable.

Baseline/ceiling per-tile values are read from
`results/oracle_visual_check_20260913/scores_per_tile.csv` rather than
recomputed -- it was built from the exact same conditioning directory this
script uses.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")

from lineart.model_zoo import load_generator_checkpoint
from tile_region_manifest_480 import edge_map, bipartite_match_f1
from train_stroke_selection import load_gray01, to_keep_mask  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-stroke-selection")
COND_DIR = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
GT_ROOT = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
HOLDOUT_LIST = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
ORACLE_CSV = TRACK / "results/oracle_visual_check_20260913/scores_per_tile.csv"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0


def gt_path(tile):
    for split in ("train", "test"):
        p = GT_ROOT / split / "line" / tile
        if p.exists():
            return p
    raise FileNotFoundError(tile)


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(TRACK / "checkpoints/stroke_selection_full_20260917/final.pth"))
    p.add_argument("--out-dir", default=str(TRACK / "results/stroke_selection_eval_20260917"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_name = load_generator_checkpoint(args.checkpoint, device)
    model.eval()
    print(f"loaded {args.checkpoint} (model_name={model_name})", flush=True)

    ceiling = {}
    with open(ORACLE_CSV) as f:
        for row in csv.DictReader(f):
            ceiling[row["tile"]] = row

    tiles = [l.strip() for l in open(HOLDOUT_LIST) if l.strip()]

    rows = []
    with torch.no_grad():
        for i, tile in enumerate(tiles):
            cond_path = COND_DIR / tile
            if not cond_path.exists() or tile not in ceiling:
                print(f"SKIP (missing conditioning or ceiling row): {tile}", file=sys.stderr)
                continue
            cond01 = load_gray01(cond_path)
            cond_gray = (cond01 * 255).astype(np.uint8)
            cond_edge = edge_map(cond_gray)
            gt_edge = edge_map(load_gray(gt_path(tile)))

            cond_t = torch.from_numpy(cond01).unsqueeze(0).unsqueeze(0).to(device)
            cond_edge_t = torch.from_numpy(cond_edge.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(cond_t)
            pred_mask = to_keep_mask(cond_edge_t, logit)[0, 0].cpu().numpy() > 0.5

            model_f1, model_p, model_r = bipartite_match_f1(pred_mask, gt_edge, BSDS_TOLERANCE_PX)
            c = ceiling[tile]
            rows.append({
                "tile": tile,
                "base_f1": float(c["base_f1"]), "base_precision": float(c["base_precision"]), "base_recall": float(c["base_recall"]),
                "model_f1": model_f1, "model_precision": model_p, "model_recall": model_r,
                "oracle_f1": float(c["oracle_f1"]), "oracle_precision": float(c["oracle_precision"]), "oracle_recall": float(c["oracle_recall"]),
                "pred_mask": pred_mask, "cond_edge": cond_edge, "cond01": cond01,
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tiles)}", flush=True)

    with open(out_dir / "scores_per_tile.csv", "w", newline="") as f:
        fields = ["tile", "base_f1", "base_precision", "base_recall",
                   "model_f1", "model_precision", "model_recall",
                   "oracle_f1", "oracle_precision", "oracle_recall"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"\nn={len(rows)} tiles (group A / lineart_family)")
    print(f"{'construction':32}{'f1':>8}{'precision':>11}{'recall':>9}")
    print(f"{'preprocessor baseline':32}{mean('base_f1'):8.4f}{mean('base_precision'):11.4f}{mean('base_recall'):9.4f}")
    print(f"{'trained classifier (this run)':32}{mean('model_f1'):8.4f}{mean('model_precision'):11.4f}{mean('model_recall'):9.4f}")
    print(f"{'pixel-level oracle (ceiling)':32}{mean('oracle_f1'):8.4f}{mean('oracle_precision'):11.4f}{mean('oracle_recall'):9.4f}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["construction", "f1", "precision", "recall"])
        w.writerow(["preprocessor_baseline", round(mean("base_f1"), 4), round(mean("base_precision"), 4), round(mean("base_recall"), 4)])
        w.writerow(["trained_classifier", round(mean("model_f1"), 4), round(mean("model_precision"), 4), round(mean("model_recall"), 4)])
        w.writerow(["oracle_ceiling", round(mean("oracle_f1"), 4), round(mean("oracle_precision"), 4), round(mean("oracle_recall"), 4)])

    montage(rows, out_dir)


def montage(rows, out_dir):
    ranked = sorted(rows, key=lambda r: r["model_f1"])
    n = len(ranked)
    picks = [ranked[0], ranked[n // 4], ranked[n // 2], ranked[3 * n // 4], ranked[-1],
             ranked[n // 2 - 10], ranked[n // 2 + 10]]

    cell, lab_h = 190, 34
    font = ImageFont.load_default()
    cols = ["condition", "model output", "GT"]
    sheet = Image.new("L", (len(cols) * cell, len(picks) * (cell + lab_h) + lab_h), 255)
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((c * cell + 4, 8), name, fill=0, font=font)
    for row_idx, r in enumerate(picks):
        tile = r["tile"]
        y = lab_h + row_idx * (cell + lab_h)
        d.text((4, y + 6), f"{tile[:-4]} base={r['base_f1']:.3f} model={r['model_f1']:.3f} "
                            f"oracle={r['oracle_f1']:.3f} r={r['model_recall']:.3f}", fill=0, font=font)
        cond_img = Image.fromarray((r["cond01"] * 255).astype(np.uint8))
        cond_img = Image.eval(cond_img, lambda p: 255 - p)
        model_img = Image.fromarray(np.where(r["pred_mask"], 0, 255).astype(np.uint8))
        gt_img = Image.open(gt_path(tile)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        for c, img in enumerate([cond_img, model_img, gt_img]):
            sheet.paste(img.resize((cell, cell)), (c * cell, y + lab_h))
    out_path = out_dir / "montage.png"
    sheet.save(out_path)
    print(f"\nmontage: {out_path}")


if __name__ == "__main__":
    main()
