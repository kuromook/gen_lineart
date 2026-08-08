"""SDEdit-style rough->line conversion: start the reverse diffusion process
from a partially-noised rough tile (not pure noise, not a separate trained
conditioning network) and denoise with the line-domain LoRA toward the
line-art domain. No training required -- reuses the already-adopted
line-domain LoRA (checkpoints/domain_lora_line_sd15base_sksv2_20260807).

Motivation (doc/work_log.md, `diffusion` branch, 2026-08-08): the ControlNet
conditional-conversion approach (`diffusion-controlnet` branch, Direction 4)
hallucinates content unrelated to the input rough tile, confirmed via two
diagnostics (2026-08-08) to be independent of pair alignment quality and of
inference-time controlnet-conditioning-scale -- pointing at a structural
limitation of that conditioning mechanism rather than something fixable
cheaply. SDEdit sidesteps the separate conditioning branch entirely: the
input image's own (partially-noised) content IS the starting point, so
structure preservation is a direct consequence of the sampling procedure,
not something a trained network has to learn to attend to.

Key trade-off (expect to need a sweep, same shape as every other knob in
this project's isolation chain): `--strength` controls how much noise is
added to the rough tile before denoising. Low strength keeps the input's
structure but converts weakly toward line-art style; high strength
converts more fully but starts to lose correspondence to the specific
input (approaching unconditional generation). No single value is assumed
correct here -- sweep and evaluate both chamfer-to-GT (structure fidelity)
and visual style/content plausibility, mirroring how every other scale
knob in this branch's isolation chain was evaluated.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image

IMAGE_SIZE = 480


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument(
        "--lora-dir", default="checkpoints/domain_lora_line_sd15base_sksv2_20260807/final"
    )
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"),
    )
    parser.add_argument(
        "--caption", default="sks style, monochrome line art, manga panel, black and white"
    )
    parser.add_argument("--negative-caption", default="")
    parser.add_argument("--lora-scale", type=float, default=1.4)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default=None, help="defaults to results/{tag}")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        args.base_ckpt, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.load_lora_weights(args.lora_dir)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    samples = read_sample_list(args.sample_list)
    generator = torch.Generator(device="cuda")

    for i, name in enumerate(samples):
        base = normalize_name(name)
        rough_path = Path(args.rough_dir) / f"{base}.jpg"
        rough = Image.open(rough_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.BILINEAR
        )
        generator.manual_seed(args.seed + i)
        result = pipe(
            prompt=args.caption,
            negative_prompt=args.negative_caption or None,
            image=rough,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale},
        ).images[0]

        out = result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        out_path = output_dir / f"{base}_out.png"
        out.save(out_path)
        print(f"[{i + 1}/{len(samples)}] {rough_path} -> {out_path}")

    print(f"[sdedit_line_from_rough] done, {len(samples)} samples written to {output_dir}")


if __name__ == "__main__":
    main()
