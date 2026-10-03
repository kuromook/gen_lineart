"""Inference-only IP-Adapter probe: does an image-prompt channel change the
spatial failure lesson 8 describes?

Question (doc/CURRENT.md "Approaches Never Taken Up", 2026-09-30): the survey
predicted IP-Adapter would not help, because lesson 8's failure is *spatial*
(where the conditioning map lacks a GT stroke the model draws it 9.3% of the
time) while an IP-Adapter embedding is pooled and spatially unlocalised. It
further predicted it might actively override the conditioning, by analogy with
2026-08-08, where a strong global style prior stacked on this same ControlNet
overrode it. This script runs the arms; nothing is trained.

Arms, all sharing one conditioning map, caption, scheduler, step count and seed
so that the image-prompt channel is the only difference:

  baseline        no IP-Adapter at all -- the anchor. Must reproduce the known
                  number for this config or nothing else here is readable.
  gt_same_sX      the tile's own GT line art as image prompt. This LEAKS: like
                  the delete oracle it bounds the mechanism, it does not show
                  an achievable score. Run to see the ceiling, never quoted as
                  a result.
  gt_other_sX     a line-art tile from the housei holdout group as the prompt.
                  Disjoint source, so it cannot leak this tile's content.
                  CAVEAT measured after the fact: housei is a different
                  drawing style as well as a different image (GT fill_ratio
                  0.124 vs lineart_family's 0.042, line width 4.10 vs 2.26),
                  so this arm moves two variables at once.
  gt_otherfam_sX  a line-art tile from a DIFFERENT lineart_family source image.
                  Same style and sub-task, only the content differs -- this is
                  the clean version of the use case, and the arm to read for
                  "style from an exemplar, spatial content from the map".
                  Added once the housei confound above was measured.

All references are fixed per tile by index, so scales differ only in scale.

Two `set_ip_adapter_scale` values per reference type, because scale is exactly
the lever that decided the analogous 2026-08-08 question: a weak global prior
was harmless and a strong one took over.

This lives in Track I (`lineart-image-prompt`). Outputs go under this tree's
`results/`, never the common foundation's -- the track pattern's core rule,
which this probe broke once and was moved for. The dataset, shared evaluation
tools and venv are still read from the foundation, which is normal.

Polarity, which this project has got wrong before: the conditioning maps on
disk are white-lines-on-black (mean ~17, 95% of pixels dark), which is what
ControlNet wants, so they are fed through unchanged. GT and the pipeline's
output are black-on-white. The image prompt is given in GT polarity, because
that is the look being transferred.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/sh1/deepl/lineart")  # shared tools live in the foundation

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

IMAGE_SIZE = 480
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
# Disjoint reference pool: the other holdout group. Different source entirely,
# so a reference from here cannot carry the evaluated tile's content.
REF_LIST = SHARED / "holdout_housei_100.txt"


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def gt_tile_path(tile):
    """GT for a tile, resolved by looking for the file rather than by a name
    prefix test -- evaluate_fixed_outputs.py --split auto gets this wrong for
    168 of these 192 tiles (see doc/CURRENT.md, Known Tool Traps)."""
    direct = GT_DIR / tile
    if direct.exists():
        return direct
    for split in ("train", "test"):
        candidate = SHARED / split / "line" / tile
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no GT for {tile}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default=str(SHARED / "holdout_lineart_family.txt"))
    parser.add_argument("--limit", type=int, default=0, help="first N tiles only (0 = all); smoke tests")
    parser.add_argument(
        "--controlnet-dir",
        default=os.path.expanduser("~/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime"),
    )
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"),
    )
    parser.add_argument("--caption", default="monochrome line art, manga panel, black and white")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ip-scales", default="0.4,0.8")
    # Added 2026-10-02 after checking this track's usage against the official
    # diffusers guide and the h94 model card. Both defaults keep the
    # 2026-09-30 behaviour, so earlier runs stay reproducible.
    parser.add_argument(
        "--ip-weight-name",
        default="ip-adapter_sd15.safetensors",
        help="`ip-adapter_sd15` conditions on the GLOBAL (pooled) CLIP embedding -- "
             "the model card's own wording, and the premise this track's prediction "
             "rested on. `ip-adapter-plus_sd15` conditions on PATCH embeddings and is "
             "the variant that could carry stroke placement, so a null from the pooled "
             "one does not cover it.",
    )
    parser.add_argument(
        "--ip-scale-mode",
        default="uniform",
        choices=("uniform", "style_only", "style_layout"),
        help="where the adapter is injected. `uniform` is one float on every block, "
             "which is what 2026-09-30 did and what the docs warn 'focuses more on the "
             "image prompt'. The others are InstantStyle (arXiv:2404.02733) as diffusers "
             "implements it: up block_0 is the style block, down block_2 the layout "
             "block. Injecting only into the style block is the documented way to move "
             "tone without content leaking in -- and content leaking in is exactly what "
             "neither_ink 0.195 -> 0.320 looked like.",
    )
    parser.add_argument("--out-root", default="results/ipadapter_probe_20260930")
    parser.add_argument(
        "--arms",
        default="baseline,gt_same,gt_other,gt_otherfam",
        help="comma-separated: baseline, gt_same, gt_other, gt_otherfam",
    )
    return parser.parse_args()


def load_pipe(args):
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        args.base_ckpt, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def ip_scale_config(mode, scale):
    """A float for every block, or InstantStyle's per-block dictionary.

    THE BLOCK INDICES ARE ARCHITECTURE-SPECIFIC AND THESE ARE SD1.5's. The
    diffusers guide's example, `{"up": {"block_0": [0.0, 1.0, 0.0]}}`, is
    written against an SDXL pipeline, whose `up_blocks[0]` is a
    CrossAttnUpBlock2D. SD1.5's `up_blocks[0]` is a plain UpBlock2D with **zero**
    attention layers, so that dictionary names nothing, every IP-Adapter
    processor keeps the 0.0 default, and the adapter is silently off -- it was
    run that way once on 2026-10-02 and produced output byte-identical to the
    baseline across all 8 arms. InstantStyle's own `infer_style_sd15.py` uses
    `target_blocks=["up_blocks.1"]` for style and
    `["down_blocks.2", "mid_block", "up_blocks.1"]` for style+layout; SD1.5's
    `up_blocks[1]` is the first attention-bearing up block, the analogue of
    SDXL's `up_blocks[0]`, and carries 3 attention layers.

    Blocks left out of the dictionary are set to 0, i.e. the adapter is off
    there -- which is the point of the mode, and also why a wrong index fails
    silently rather than raising. Check the output differs from the baseline.
    """
    if mode == "uniform":
        return scale
    if mode == "style_only":
        return {"up": {"block_1": [scale, scale, scale]}}
    if mode == "style_layout":
        return {"down": {"block_2": [scale, scale]}, "up": {"block_1": [scale, scale, scale]}}
    raise ValueError(mode)


def run_arm(pipe, args, tiles, refs, arm, ip_scale, out_dir):
    """One arm. `refs` is None for the baseline, else tile -> reference path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda")
    done = 0
    for i, tile in enumerate(tiles):
        out_path = out_dir / f"{Path(tile).stem}_out.png"
        if out_path.exists():
            done += 1
            continue
        cond = Image.open(COND_DIR / tile).convert("RGB").resize(
            (args.resolution, args.resolution), Image.BILINEAR
        )
        kwargs = {}
        if refs is not None:
            ref = Image.open(refs[tile]).convert("RGB").resize(
                (args.resolution, args.resolution), Image.BILINEAR
            )
            kwargs["ip_adapter_image"] = ref
        # Seed depends only on tile index, so every arm sees identical noise.
        generator.manual_seed(args.seed + i)
        result = pipe(
            prompt=args.caption,
            image=cond,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
            **kwargs,
        ).images[0]
        result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(out_path)
        done += 1
        if done % 20 == 0 or done == len(tiles):
            print(f"[{arm} s{ip_scale}] {done}/{len(tiles)}", flush=True)
    return done


def main():
    args = parse_args()
    tiles = read_list(args.sample_list)
    if args.limit:
        tiles = tiles[: args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ip_scales = [float(s) for s in args.ip_scales.split(",") if s.strip()]
    out_root = Path(args.out_root)

    ref_pool = read_list(REF_LIST)
    # Fixed pairing by index: tile k always gets reference k % len(pool), so the
    # gt_other arms differ from each other only in scale.
    refs_other = {t: gt_tile_path(ref_pool[i % len(ref_pool)]) for i, t in enumerate(tiles)}
    refs_same = {t: gt_tile_path(t) for t in tiles}
    # Same-pool reference from a different SOURCE IMAGE: tiles are
    # lineart_<image>_<tile>.jpg, so stepping to the next distinct <image>
    # keeps the style fixed and removes only the content overlap.
    def source_of(tile):
        return tile.split("_")[1] if "_" in tile else tile
    by_source = {}
    for t in tiles:
        by_source.setdefault(source_of(t), []).append(t)
    sources = sorted(by_source)
    refs_otherfam = {}
    for t in tiles:
        here = source_of(t)
        other = sources[(sources.index(here) + 1) % len(sources)]
        if other == here:  # single-source degenerate case
            refs_otherfam[t] = gt_tile_path(t)
        else:
            refs_otherfam[t] = gt_tile_path(by_source[other][0])

    pipe = load_pipe(args)
    ip_loaded = False

    for arm in arms:
        if arm == "baseline":
            print(f"[probe] arm baseline, {len(tiles)} tiles", flush=True)
            run_arm(pipe, args, tiles, None, "baseline", 0.0, out_root / "baseline")
            continue
        if not ip_loaded:
            pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name
            )
            ip_loaded = True
        refs = {"gt_same": refs_same, "gt_other": refs_other,
                "gt_otherfam": refs_otherfam}[arm]
        for scale in ip_scales:
            pipe.set_ip_adapter_scale(ip_scale_config(args.ip_scale_mode, scale))
            tag = f"{arm}_s{scale}"
            print(f"[probe] arm {tag}, {len(tiles)} tiles", flush=True)
            run_arm(pipe, args, tiles, refs, arm, scale, out_root / tag)

    print(f"[probe] weights={args.ip_weight_name} scale_mode={args.ip_scale_mode}", flush=True)
    print(f"[probe] done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
