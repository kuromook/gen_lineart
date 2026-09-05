"""Inference counterpart to scripts/train_controlnet_sdxl.py (candidate #1,
SDXL migration). Mirrors ../lineart/scripts/infer_controlnet.py's CLI/output
conventions (results/{tag}/{base}_out.png) so the existing evaluation
tooling (evaluate_fixed_outputs.py, this track's four_model_comparison
scripts) can be reused as-is.
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

IMAGE_SIZE = 480


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="data/diag_valid5.txt")
    parser.add_argument("--rough-dir", default="data/diag_rough_lineart_coarse")
    parser.add_argument("--controlnet-dir", required=True)
    parser.add_argument(
        "--controlnet-lora-dir",
        default=None,
        help="optional LoRA adapter dir (from train_controlnet_sdxl.py) to load on top of --controlnet-dir",
    )
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--tag", default="controlnet_sdxl")
    parser.add_argument("--caption", default="monochrome line art, manga panel, black and white")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="defaults to results/{tag}")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="enable_model_cpu_offload(): keep only the module in use on the GPU. "
        "Required at --resolution 1024 on 12GB -- the fully-resident fp16 pipeline "
        "(UNet 2.6B + SDXL ControlNet 1.25B + two text encoders + VAE) peaks at 11.1GB "
        "and OOMs before the first UNet block (verified 2026-09-06).",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="enable_vae_tiling(): decode the 128x128 latent in tiles (the fp16 SDXL VAE "
        "decode at 1024 is a second, separate memory spike after the denoise loop).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    except OSError:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16, variant="fp16")
    if args.controlnet_lora_dir:
        controlnet.load_lora_adapter(
            args.controlnet_lora_dir, weight_name="pytorch_lora_weights.safetensors", prefix=None
        )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_ckpt, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if args.vae_tiling:
        pipe.enable_vae_tiling()
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
            image=rough,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

        out = result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        out_path = output_dir / f"{base}_out.png"
        out.save(out_path)
        print(f"[{i + 1}/{len(samples)}] {rough_path} -> {out_path}")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[infer_controlnet_sdxl] done, {len(samples)} samples written to {output_dir} "
        f"(resolution={args.resolution}, cs={args.controlnet_conditioning_scale}, "
        f"peak_vram={peak:.2f}GiB)"
    )


if __name__ == "__main__":
    main()
