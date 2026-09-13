"""Score the LoRA style-contribution diagnostic: does the base SD1.5 prior
dominate over the rough conditioning at high consistency_weight? See
doc/work_log.md "2026-09-13" for the full write-up and
run_lora_style_diagnostics_20260913.sh for task 1's generation step.

Architecture fact motivating this (confirmed in scripts/train_controlnet_consistency.py):
the LoRA is attached only to ControlNet's attention projections
(to_k/to_q/to_v/to_out.0, rank16); the base UNet is fully frozen, plain
v1-5-pruned-emaonly.safetensors (no anime merge). There is no mechanism in
this pipeline for the LoRA to learn "style" -- it can only reshape how
strongly/where the rough sketch's structure is injected into an otherwise
untouched, non-domain-specific UNet.

Three tasks, all inference-free except task 1's generation step (already run
by the shell script):

1. Prior-only ablation: for weight in {0.02, 0.2, 0.4, 0.5} at cs=2.5,
   compare the real-rough output against outputs generated (same seed) from
   blank-white / blank-black / fixed-noise conditioning instead. If the
   real-rough and null-condition outputs converge (edge-similarity rises)
   as weight increases, that is direct evidence the base prior is
   increasingly winning over the conditioning signal.
2. Cross-sample diversity: for each weight (all 9 from both sweep rounds),
   compare the diagnostic 5 samples' pairwise output similarity (all 10
   pairs) against their pairwise conditioning-image similarity. If outputs
   are much more similar to each other than the inputs are, that is a
   mode-collapse signature independent of task 1.
3. CLIP style-similarity to the GT corpus: does cosine similarity to real
   line-art (data/line/, sampled) move at all across the weight sweep? If
   it stays flat, that is direct evidence the LoRA (ControlNet-only) is not
   moving the output's *style* toward the target corpus, consistent with
   the architecture fact above.
"""

import csv
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
ROOT = TRACK / "results/lora_style_diagnostics_20260913"
CS = "2.5"

# all 9 sweep weights and which round's eval-dir suffix they live under
WEIGHT_SUFFIX = {
    "0.02": "20260908", "0.05": "20260908", "0.1": "20260908",
    "0.15": "20260914", "0.2": "20260908", "0.25": "20260914",
    "0.3": "20260914", "0.4": "20260914", "0.5": "20260908",
}
TASK1_WEIGHTS = ["0.02", "0.2", "0.4", "0.5"]


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def real_eval_dir(weight):
    suffix = WEIGHT_SUFFIX[weight]
    tag = f"controlnet_lora_manga_consistency_w{weight}_{suffix}"
    return TRACK / f"results/{tag}_eval/cs{CS}"


def sim_f1(path_a, path_b):
    ea = edge_map(load_gray(path_a))
    eb = edge_map(load_gray(path_b))
    return bipartite_match_f1(ea, eb, BSDS_TOLERANCE_PX)[0]


# --- Task 1: prior-only ablation ---

def task1():
    print("\n############ Task 1: prior-only ablation (real rough vs blank/noise) ############")
    rows = []
    for w in TASK1_WEIGHTS:
        real_dir = real_eval_dir(w)
        for cond in ["null_white", "null_black", "null_noise"]:
            cond_dir = ROOT / f"outputs/w{w}/{cond}"
            if not cond_dir.is_dir():
                print(f"  (missing outputs for w={w} {cond}, skipping)")
                continue
            sims = []
            for s in SAMPLES:
                real_p = real_dir / f"{s}_out.png"
                cond_p = cond_dir / f"{s}_out.png"
                if not (real_p.exists() and cond_p.exists()):
                    continue
                sims.append(sim_f1(real_p, cond_p))
            if sims:
                rows.append({"weight": w, "condition": cond, "mean_sim_to_real_rough": float(np.mean(sims))})

    if not rows:
        print("  no task-1 outputs found -- did the generation step run?")
        return rows

    print(f"\n{'weight':>8} {'null_white':>12} {'null_black':>12} {'null_noise':>12}")
    for w in TASK1_WEIGHTS:
        line = f"{w:>8}"
        for cond in ["null_white", "null_black", "null_noise"]:
            hit = [r for r in rows if r["weight"] == w and r["condition"] == cond]
            line += f"{hit[0]['mean_sim_to_real_rough']:>12.4f}" if hit else f"{'-':>12}"
        print(line)
    print("\n(similarity = edge-map bipartite-match F1 between the real-rough output and the")
    print(" null-condition output, same seed, same weight/cs -- higher means the output ignores")
    print(" what it was actually conditioned on. Rising across the weight column would support")
    print(" 'the base prior increasingly dominates at high weight'.)")
    return rows


# --- Task 2: cross-sample diversity ---

def task2():
    print("\n############ Task 2: output diversity vs input diversity ############")
    cond_dir = TRACK / "data/diag_rough_manga_line"
    cond_pairs = list(combinations(SAMPLES, 2))
    cond_sims = [sim_f1(cond_dir / f"{a}.jpg", cond_dir / f"{b}.jpg") for a, b in cond_pairs]
    mean_cond_sim = float(np.mean(cond_sims))
    print(f"condition-image pairwise similarity (10 pairs): mean={mean_cond_sim:.4f}")

    rows = []
    for w in sorted(WEIGHT_SUFFIX, key=float):
        d = real_eval_dir(w)
        paths = {s: d / f"{s}_out.png" for s in SAMPLES}
        if not all(p.exists() for p in paths.values()):
            continue
        sims = [sim_f1(paths[a], paths[b]) for a, b in cond_pairs]
        mean_out_sim = float(np.mean(sims))
        rows.append({"weight": w, "mean_output_pairwise_sim": mean_out_sim,
                     "mean_condition_pairwise_sim": mean_cond_sim,
                     "output_minus_condition_sim": mean_out_sim - mean_cond_sim})

    print(f"\n{'weight':>8} {'output_pairwise':>16} {'condition_pairwise':>19} {'delta':>10}")
    for r in rows:
        print(f"{r['weight']:>8} {r['mean_output_pairwise_sim']:>16.4f} "
              f"{r['mean_condition_pairwise_sim']:>19.4f} {r['output_minus_condition_sim']:>10.4f}")
    print("\n(delta > 0 means the 5 outputs resemble each other MORE than the 5 inputs resemble")
    print(" each other -- a mode-collapse signature independent of task 1.)")
    return rows


# --- Task 3: CLIP style-similarity to GT corpus ---

def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
    # transformers 5.x's CLIPModel.get_image_features returns a
    # BaseModelOutputWithPooling instead of a plain tensor; the pooled
    # image embedding (pre-normalization) is .pooler_output here.
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def task3():
    print("\n############ Task 3: CLIP style-similarity to GT corpus ############")
    try:
        model, processor = load_clip()
    except Exception as e:
        print(f"  CLIP unavailable ({e}), skipping task 3")
        return []

    all_line = sorted((TRACK / "data/line").glob("*.jpg"))
    rng = random.Random(0)
    corpus_sample = rng.sample(all_line, min(80, len(all_line)))
    corpus_feats = clip_embed(model, processor, corpus_sample)
    print(f"corpus: {len(corpus_sample)} images sampled from data/line/ (seed=0)")

    rows = []
    for w in sorted(WEIGHT_SUFFIX, key=float):
        d = real_eval_dir(w)
        paths = [d / f"{s}_out.png" for s in SAMPLES]
        if not all(p.exists() for p in paths):
            continue
        out_feats = clip_embed(model, processor, paths)
        sim_matrix = out_feats @ corpus_feats.T  # (5, n_corpus)
        mean_sim = float(sim_matrix.mean())
        rows.append({"weight": w, "clip_sim_to_gt_corpus": mean_sim})

    print(f"\n{'weight':>8} {'clip_sim_to_gt_corpus':>22}")
    for r in rows:
        print(f"{r['weight']:>8} {r['clip_sim_to_gt_corpus']:>22.4f}")
    if rows:
        vals = [r["clip_sim_to_gt_corpus"] for r in rows]
        print(f"\nrange across weights: {max(vals) - min(vals):.4f} "
              f"(min={min(vals):.4f}, max={max(vals):.4f})")
        print("(a small range relative to typical CLIP similarity spread supports 'style")
        print(" similarity to the GT corpus barely moves with consistency_weight'.)")
    return rows


def montage_task1():
    if not TASK1_WEIGHTS:
        return
    cell_px, label_h = 200, 24
    font = ImageFont.load_default()
    cols = ["real_rough", "null_white", "null_black", "null_noise"]
    sheet = Image.new("L", (len(cols) * cell_px, len(TASK1_WEIGHTS) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    grid_sample = "lineart_008_014"
    for r, w in enumerate(TASK1_WEIGHTS):
        y0 = label_h + r * (cell_px + label_h)
        draw.text((4, y0 + 6), f"weight={w}", fill=0, font=font)
        paths = [real_eval_dir(w) / f"{grid_sample}_out.png"]
        paths += [ROOT / f"outputs/w{w}/{c}/{grid_sample}_out.png" for c in cols[1:]]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = ROOT / "montage_prior_only_ablation.png"
    sheet.save(out)
    print(f"\nmontage: {out}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    t1 = task1()
    t2 = task2()
    t3 = task3()

    csv_path = ROOT / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("# task1: prior-only ablation\n")
        if t1:
            w = csv.DictWriter(f, fieldnames=list(t1[0]))
            w.writeheader()
            for r in t1:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        f.write("\n# task2: cross-sample diversity\n")
        if t2:
            w = csv.DictWriter(f, fieldnames=list(t2[0]))
            w.writeheader()
            for r in t2:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        f.write("\n# task3: CLIP style-similarity to GT corpus\n")
        if t3:
            w = csv.DictWriter(f, fieldnames=list(t3[0]))
            w.writeheader()
            for r in t3:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nsaved: {csv_path}")

    montage_task1()


if __name__ == "__main__":
    main()
