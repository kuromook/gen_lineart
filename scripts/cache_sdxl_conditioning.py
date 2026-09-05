"""Pre-compute the frozen halves of SDXL ControlNet training so they never
need to live on the GPU during the training loop.

Why (2026-09-06, this track): a naive `--resolution 1024` run of
`train_controlnet_sdxl.py` OOMs on a 12GB card *inside the fp32 VAE
encoder*, before the UNet is ever reached. The training loop keeps four
frozen modules resident purely to recompute the same values every epoch:

    UNet fp16          ~5.2GB   (needed -- gradients flow through it)
    ControlNet fp16    ~2.5GB   (needed -- it is what we train)
    VAE fp32           ~1.3GB   (frozen, and the encode spikes at 1024)
    text encoders fp16 ~1.6GB   (frozen, and captions are fixed per tile)

The bottom two produce a deterministic function of (tile, caption), so they
can be computed once here and read from disk. That frees ~2.9GB of resident
weights plus the 1024x1024 fp32 VAE encode spike, which is the difference
between OOM and fitting.

Latents are stored as the *distribution* (mean and std), not a single
sample, so the training loop still draws a fresh `latent_dist.sample()`
each epoch exactly as the uncached path does -- caching costs no
stochasticity.

Output layout: <out-dir>/<name>.npz per tile, with keys
    latent_mean  (4, R/8, R/8) fp16
    latent_std   (4, R/8, R/8) fp16
    prompt_embeds (77, 2048)   fp16
    pooled_embeds (1280,)      fp16
At R=1024 that is ~580KB/tile, ~4.9GB for the 8,467-tile train list.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", default="data/train_list.txt")
    parser.add_argument("--line-dir", default="data/line")
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion-XL/animagine-xl-3.1"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--caption", default="monochrome line art, clean linework, manga panel")
    parser.add_argument("--caption-csv", default=None, help="CSV with columns name,caption")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute tiles that already have an .npz"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    names = [line.strip() for line in open(args.file_list) if line.strip()]
    caption_by_name = {}
    if args.caption_csv:
        with open(args.caption_csv) as f:
            for row in csv.DictReader(f):
                caption_by_name[row["name"]] = row["caption"]
        print(f"[cache_sdxl] loaded {len(caption_by_name)} per-tile captions from {args.caption_csv}")

    todo = names if args.overwrite else [n for n in names if not (out_dir / f"{n}.npz").exists()]
    print(f"[cache_sdxl] {len(names)} tiles, {len(todo)} to compute, resolution={args.resolution}")
    if not todo:
        return

    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_ckpt, torch_dtype=torch.float32)
    # Same dtype policy as train_controlnet_sdxl.py: the stock SDXL VAE is
    # numerically unstable in fp16, so it encodes in fp32; the text encoders
    # are run in fp16, which is what the training loop consumed.
    vae = pipe.vae.to(device, dtype=torch.float32).eval()
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [
        pipe.text_encoder.to(device, dtype=torch.float16).eval(),
        pipe.text_encoder_2.to(device, dtype=torch.float16).eval(),
    ]
    del pipe.unet
    torch.cuda.empty_cache()

    # Identical to ControlNetSDXLTileDataset.target_transform.
    target_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    @torch.no_grad()
    def encode_prompt(captions):
        embeds, pooled = [], None
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            input_ids = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = text_encoder(input_ids, output_hidden_states=True)
            pooled = out[0]
            embeds.append(out.hidden_states[-2])
        return torch.cat(embeds, dim=-1), pooled

    start = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        images = torch.stack(
            [
                target_transform(Image.open(os.path.join(args.line_dir, n)).convert("RGB"))
                for n in batch
            ]
        ).to(device, dtype=torch.float32)
        captions = [caption_by_name.get(n, args.caption) for n in batch]

        with torch.no_grad():
            dist = vae.encode(images).latent_dist
            # Fold in the scaling factor here so the training loop reads a
            # latent it can use directly, matching the uncached path's
            # `latents * vae.config.scaling_factor`. Scaling is linear, so it
            # applies to mean and std alike.
            scale = vae.config.scaling_factor
            mean = (dist.mean * scale).cpu().numpy().astype(np.float16)
            std = (dist.std * scale).cpu().numpy().astype(np.float16)
            prompt_embeds, pooled_embeds = encode_prompt(captions)
            prompt_embeds = prompt_embeds.cpu().numpy().astype(np.float16)
            pooled_embeds = pooled_embeds.cpu().numpy().astype(np.float16)

        for j, name in enumerate(batch):
            np.savez(
                out_dir / f"{name}.npz",
                latent_mean=mean[j],
                latent_std=std[j],
                prompt_embeds=prompt_embeds[j],
                pooled_embeds=pooled_embeds[j],
            )

        done = i + len(batch)
        if done % (args.batch_size * 50) == 0 or done == len(todo):
            elapsed = time.time() - start
            rate = done / elapsed
            print(
                f"[cache_sdxl] {done}/{len(todo)} ({rate:.1f} tiles/s, "
                f"eta {(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min)",
                flush=True,
            )

    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 2**20
    print(f"[cache_sdxl] done, {out_dir} holds {total_mb:.0f} MiB")


if __name__ == "__main__":
    main()
