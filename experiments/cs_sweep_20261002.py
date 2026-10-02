"""Conditioning-scale sweep for *this* track's configuration, baseline only.

Why this exists: the 2026-09-30 probe fixed `cs=2.5` as "the established best"
and every one of its eight arms landed in the hatch-dominated regime
(ink_ratio 0.442 against the conditioning map's 0.054), where lesson 1 says
model differences are crushed and nothing is readable. 2.5 came from Track B's
SDXL stock ControlNet and from Track A's *own consistency LoRA* stack. **No
track has ever established an optimal cs for SD1.5 + public
`control_v11p_sd15s2_lineart_anime` + no LoRA**, so this sweep establishes it
here rather than borrowing it.

Both ends are swept on purpose. Low cs is expected to be hatch-dominated (the
model free-runs and fills the tile with grey); high cs is expected to thin the
strokes until they disappear (`controlnet-realpairs` saw a sword vanish at its
own cs 5.0 -- a different configuration, quoted only as the shape of the
failure, not as a value to reuse).

Everything except cs is byte-identical to the 2026-09-30 probe: same
conditioning maps, caption, scheduler, step count, guidance scale, resolution
and per-tile seed. No IP-Adapter is loaded at all, so this measures the
ControlNet configuration alone.

Outputs: `<out-root>/cs<value>/<tile>_out.png`, one directory per cs so that
`score_ipadapter_probe_20260930.py --probe-root <out-root>` reads each as an
arm and adds the conditioning map's own 0.3164 as the row everything is read
against (lesson 6).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/sh1/deepl/lineart")

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

IMAGE_SIZE = 480
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default=str(SHARED / "holdout_lineart_family.txt"))
    parser.add_argument("--limit", type=int, default=0, help="first N tiles only (0 = all); smoke tests")
    parser.add_argument("--stride", type=int, default=1, help="take every Nth tile; spreads a smoke test over source images")
    parser.add_argument(
        "--controlnet-dir",
        default=os.path.expanduser("~/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime"),
    )
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"),
    )
    parser.add_argument("--caption", default="monochrome line art, manga panel, black and white")
    # Both default to the 2026-09-30 probe's behaviour so the sweep stays a
    # one-variable study. They exist because the smoke run showed every cs from
    # 0.5 to 8.0 sitting in the hatch-dominated regime, which makes "which cs"
    # unanswerable until something gets the configuration out of that regime;
    # they are the two cheapest candidates and are tested one at a time.
    parser.add_argument("--negative-prompt", default="", help="SD negative prompt; the 2026-09-30 probe had none")
    parser.add_argument(
        "--cond-dir",
        default=str(COND_DIR),
        help="where the control images come from. Pointing this at the GT line "
             "art (with --invert-cond) is the oracle test: if ControlNet is "
             "being used correctly, a clean line-art control must come back as "
             "clean line art. If that fails, the preprocessor is not the issue.",
    )
    parser.add_argument(
        "--cond-shift",
        type=int,
        default=0,
        help="use tile (i+N)'s control image for tile i. Nonzero deliberately "
             "mismatches structure from target: if the output follows the "
             "shifted map, the map is genuinely steering; if it degrades the "
             "same way regardless, cs is just destroying the image.",
    )
    parser.add_argument(
        "--invert-cond",
        action="store_true",
        help="feed the conditioning map inverted. Stored maps are the raw "
             "LineartDetector return (white lines on black, mean ~17), which is "
             "what diffusers' documented usage feeds, so this is OFF by default "
             "and exists only to test that convention rather than assume it.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cs-values", default="0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0")
    parser.add_argument("--out-root", default="results/cs_sweep_20261002")
    return parser.parse_args()


def cs_tag(value):
    """Directory name for a cs value. Zero-padded so a plain sort is numeric."""
    return f"cs{value:05.2f}"


def load_pipe(args):
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        args.base_ckpt, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_conditioning(args, tile):
    cond = Image.open(Path(args.cond_dir) / tile).convert("RGB")
    if args.invert_cond:
        cond = Image.eval(cond, lambda p: 255 - p)
    return cond.resize((args.resolution, args.resolution), Image.BILINEAR)


def run_cs(pipe, args, tiles, cs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda")
    done = 0
    for i, tile in enumerate(tiles):
        out_path = out_dir / f"{Path(tile).stem}_out.png"
        if out_path.exists():
            done += 1
            continue
        cond = load_conditioning(args, tiles[(i + args.cond_shift) % len(tiles)])
        # Seed depends only on tile index, so every cs sees identical noise.
        generator.manual_seed(args.seed + i)
        result = pipe(
            prompt=args.caption,
            negative_prompt=args.negative_prompt or None,
            image=cond,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=cs,
            generator=generator,
        ).images[0]
        result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(out_path)
        done += 1
        if done % 24 == 0 or done == len(tiles):
            print(f"[cs {cs}] {done}/{len(tiles)}", flush=True)
    return done


def main():
    args = parse_args()
    tiles = read_list(args.sample_list)
    if args.stride > 1:
        tiles = tiles[:: args.stride]
    if args.limit:
        tiles = tiles[: args.limit]
    cs_values = [float(v) for v in args.cs_values.split(",") if v.strip()]
    out_root = Path(args.out_root)

    pipe = load_pipe(args)
    print(f"[sweep] {len(tiles)} tiles x {len(cs_values)} cs values -> {out_root}", flush=True)
    print(f"[sweep] guidance={args.guidance_scale} invert_cond={args.invert_cond} "
          f"negative={args.negative_prompt!r}", flush=True)
    print(f"[sweep] cond_dir={args.cond_dir} cond_shift={args.cond_shift} "
          f"controlnet={args.controlnet_dir}", flush=True)
    for cs in cs_values:
        print(f"[sweep] cs={cs}", flush=True)
        run_cs(pipe, args, tiles, cs, out_root / cs_tag(cs))
    print(f"[sweep] done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
