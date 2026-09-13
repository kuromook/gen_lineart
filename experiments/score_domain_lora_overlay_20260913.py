"""Score task 4 of the LoRA style-contribution investigation: does an
EXISTING UNet-side style LoRA (domain_lora_line_sd15base_sksv2_20260807,
which touches the frozen SD1.5 UNet's attention layers -- unlike the
ControlNet-only consistency LoRA) actually shift CLIP style-similarity to
the GT corpus when stacked on top of this track's current best ControlNet
checkpoint (w=0.4, cs=2.5)? See run_domain_lora_overlay_20260913.sh for the
generation step and doc/work_log.md "2026-09-13" for the full write-up.

5 cells:
- baseline: no style LoRA at all (lora-scale effectively 0) -- reused
  directly from the existing round-2 w=0.4/cs2.5 eval output, not
  regenerated.
- scale07_notrigger / scale07_trigger: lora-scale=0.7, caption without/with
  the "sks style," trigger phrase the style LoRA was trained with.
- scale14_notrigger / scale14_trigger: lora-scale=1.4 (the adopted scale
  per doc/diffusion_fidelity_budget_policy.md), same two caption variants.

Same metric surface as the consistency-weight sweep scorers (profile_metrics
+ gt_bsds_f1 + vs_condition_f1) plus task 3's CLIP style-similarity to the
GT corpus (same 80-image seed=0 sample from data/line/, same
laion/CLIP-ViT-B-32-laion2B-s34B-b79K checkpoint), so all three of this
track's LoRA-diagnostic numbers stay directly comparable.
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
CONDITION_DIR = TRACK / "data/diag_rough_manga_line"
ROOT = TRACK / "results/domain_lora_overlay_20260913"
GRID_SAMPLE = "lineart_008_014"

# label -> (output dir, lora_scale, has_trigger) -- baseline reuses the
# existing round-2 w=0.4/cs2.5 eval output rather than regenerating it.
CELLS = {
    "baseline_no_style_lora": (TRACK / "results/controlnet_lora_manga_consistency_w0.4_20260914_eval/cs2.5", 0.0, False),
    "scale07_notrigger": (ROOT / "outputs/scale07_notrigger", 0.7, False),
    "scale07_trigger": (ROOT / "outputs/scale07_trigger", 0.7, True),
    "scale14_notrigger": (ROOT / "outputs/scale14_notrigger", 1.4, False),
    "scale14_trigger": (ROOT / "outputs/scale14_trigger", 1.4, True),
}


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def score_cell(cell_dir):
    acc = {}
    for s in SAMPLES:
        p = cell_dir / f"{s}_out.png"
        if not p.exists():
            return None
        gray = load_gray(p)
        pred_edge = edge_map(gray)
        m = dict(profile_metrics(p))
        gt_edge = edge_map(load_gray(TRACK / f"data/diag_gt_line_{s}.jpg"))
        m["gt_bsds_f1"] = bipartite_match_f1(pred_edge, gt_edge, BSDS_TOLERANCE_PX)[0]
        cond_edge = edge_map(load_gray(CONDITION_DIR / f"{s}.jpg"))
        m["vs_condition_f1"] = bipartite_match_f1(pred_edge, cond_edge, BSDS_TOLERANCE_PX)[0]
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def load_clip():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import CLIPModel, CLIPProcessor
    model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def clip_embed(model, processor, paths):
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def montage():
    cell_px, label_h = 200, 24
    font = ImageFont.load_default()
    cols = ["cond", "GT", "output"]
    labels = list(CELLS)
    sheet = Image.new("L", (len(cols) * cell_px, len(labels) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    cond_path = CONDITION_DIR / f"{GRID_SAMPLE}.jpg"
    gt_path = TRACK / f"data/diag_gt_line_{GRID_SAMPLE}.jpg"
    for r, label in enumerate(labels):
        y0 = label_h + r * (cell_px + label_h)
        cell_dir, scale, trigger = CELLS[label]
        draw.text((4, y0 + 6), f"{label} (scale={scale}, trigger={trigger})", fill=0, font=font)
        paths = [cond_path, gt_path, cell_dir / f"{GRID_SAMPLE}_out.png"]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = ROOT / "montage_domain_lora_overlay.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, (cell_dir, scale, trigger) in CELLS.items():
        scored = score_cell(cell_dir)
        if scored is None:
            print(f"  (missing outputs for {label} at {cell_dir}, skipping)")
            continue
        rows.append({"label": label, "lora_scale": scale, "trigger": trigger, **scored})

    if not rows:
        print("no cells scored -- did the generation step run?", file=sys.stderr)
        return

    # CLIP style-similarity to the same GT-corpus sample task 3 used
    try:
        model, processor = load_clip()
        all_line = sorted((TRACK / "data/line").glob("*.jpg"))
        corpus_sample = random.Random(0).sample(all_line, min(80, len(all_line)))
        corpus_feats = clip_embed(model, processor, corpus_sample)
        for r in rows:
            cell_dir, _, _ = CELLS[r["label"]]
            paths = [cell_dir / f"{s}_out.png" for s in SAMPLES]
            out_feats = clip_embed(model, processor, paths)
            r["clip_sim_to_gt_corpus"] = float((out_feats @ corpus_feats.T).mean())
    except Exception as e:
        print(f"  CLIP unavailable ({e}), skipping style-similarity column")

    csv_path = ROOT / "scores.csv"
    fields = list(rows[0])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\n{'label':<24} {'gt_bsds_f1':>11} {'vs_cond_f1':>11} {'near_white':>11} "
          f"{'ink_ratio':>10} {'line_w_p50':>11} {'clip_sim':>9}")
    for r in rows:
        print(f"{r['label']:<24} {r['gt_bsds_f1']:>11.4f} {r['vs_condition_f1']:>11.4f} "
              f"{r['near_white_frac']:>11.4f} {r['ink_ratio']:>10.4f} "
              f"{r['line_width_p50']:>11.2f} {r.get('clip_sim_to_gt_corpus', float('nan')):>9.4f}")

    clip_vals = [r.get("clip_sim_to_gt_corpus") for r in rows if "clip_sim_to_gt_corpus" in r]
    if len(clip_vals) > 1:
        print(f"\nclip_sim_to_gt_corpus range across the 5 cells: {max(clip_vals) - min(clip_vals):.4f} "
              f"(baseline={rows[0].get('clip_sim_to_gt_corpus'):.4f})")
        print("(compare against task 3's 0.0172 range across the whole 9-weight consistency sweep --")
        print(" a range much larger than that here would mean the style LoRA IS moving style,")
        print(" unlike the ControlNet-only consistency LoRA.)")

    print(f"\nsaved: {csv_path}")
    montage()


if __name__ == "__main__":
    main()
