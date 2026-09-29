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
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors"
            )
            ip_loaded = True
        refs = {"gt_same": refs_same, "gt_other": refs_other,
                "gt_otherfam": refs_otherfam}[arm]
        for scale in ip_scales:
            pipe.set_ip_adapter_scale(scale)
            tag = f"{arm}_s{scale}"
            print(f"[probe] arm {tag}, {len(tiles)} tiles", flush=True)
            run_arm(pipe, args, tiles, refs, arm, scale, out_root / tag)

    print(f"[probe] done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
