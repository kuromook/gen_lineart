"""Sample a trained domain LoRA (scripts/train_domain_lora.py) with plain
txt2img generation (no conditioning image) and build a contact sheet, to
visually judge domain generation quality -- the actual deliverable of
this branch's current direction (see doc/work_log.md, `diffusion` branch,
2026-08-04/05: set the rough->line conversion task aside, check rough-
domain and line-domain generation quality individually first).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-dir", required=True, help="a train_domain_lora.py output dir's final/ or step_N/")
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors"),
    )
    parser.add_argument("--caption", required=True, help="must match the trigger caption used at training time")
    parser.add_argument("--negative-caption", default="")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default=None, help="defaults to results/{tag}")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_single_file(
        args.base_ckpt, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.load_lora_weights(args.lora_dir)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda")
    images = []
    for i in range(args.num_samples):
        generator.manual_seed(args.seed + i)
        result = pipe(
            prompt=args.caption,
            negative_prompt=args.negative_caption or None,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        out_path = output_dir / f"sample_{i:03d}.png"
        result.save(out_path)
        images.append(result)
        print(f"[{i + 1}/{args.num_samples}] -> {out_path}")

    cell, cols = args.cell, args.cols
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols, cell * rows), "white")
    for i, image in enumerate(images):
        thumb = image.resize((cell, cell))
        row, col = divmod(i, cols)
        canvas.paste(thumb, (col * cell, row * cell))
    contact_sheet_path = output_dir / f"contact_sheet_{args.tag}.png"
    canvas.save(contact_sheet_path)
    print(f"[sample_domain_lora] done, {len(images)} samples + {contact_sheet_path}")


if __name__ == "__main__":
    main()
