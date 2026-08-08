"""Sample line-art from rough tiles using a trained Direction 4 ControlNet
checkpoint, writing outputs in the `results/{tag}/{base}_out.png` convention
so the existing `tools/evaluation/evaluate_fixed_outputs.py` and
`tools/compare/make_multi_model_eval_compare.py` tooling can be reused
as-is for comparison against the CNN/GAN refiner baselines.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image


IMAGE_SIZE = 480


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", default="dataset/pairs_480/eval_clean_lineart004_8.txt")
    parser.add_argument("--rough-dir", default="dataset/pairs_480/test/rough")
    parser.add_argument(
        "--controlnet-dir", default="checkpoints/controlnet_koma_direction4_20260731/final"
    )
    parser.add_argument(
        "--controlnet-lora-dir",
        default=None,
        help="optional LoRA adapter dir (from train_controlnet.py --controlnet-lora-rank) to load "
        "on top of --controlnet-dir's base weights",
    )
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors"),
    )
    parser.add_argument("--tag", default="controlnet_koma_direction4_20260731")
    parser.add_argument(
        "--caption",
        default="monochrome line art, clean linework, manga panel, black and white",
        help="must match the fixed caption used at training time",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument(
        "--lora-dir",
        default=None,
        help="optional domain LoRA to load on top of the base UNet (e.g. checkpoints/domain_lora_line_sd15base_sksv2_20260807/final)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="defaults to results/{tag}")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or f"results/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16)
    if args.controlnet_lora_dir:
        controlnet.load_lora_adapter(
            args.controlnet_lora_dir, weight_name="pytorch_lora_weights.safetensors", prefix=None
        )
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        args.base_ckpt, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    if args.lora_dir:
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
            image=rough,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale} if args.lora_dir else None,
        ).images[0]

        out = result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        out_path = output_dir / f"{base}_out.png"
        out.save(out_path)
        print(f"[{i + 1}/{len(samples)}] {rough_path} -> {out_path}")

    print(f"[infer_controlnet] done, {len(samples)} samples written to {output_dir}")


if __name__ == "__main__":
    main()
