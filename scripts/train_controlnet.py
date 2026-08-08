"""Direction 4: fine-tune a ControlNet adapter on an SD1.5-family checkpoint
to condition on rough tiles and denoise toward line-art tiles.

Reference data format:
  --file-list   dataset/pairs_480/valid_train_combined_koma_20260729.txt (1489 tiles)
  --rough-dir   dataset/pairs_480/train/rough            (conditioning image)
  --line-dir    dataset/pairs_480/train/line_combined_koma_20260729  (target image)

By default there is no per-tile caption, so a single fixed caption is used
for every example (--caption). Optionally pass --caption-csv (columns
name,caption -- see scripts/tag_wd14.py, which auto-tags tiles with a
WD14-style anime tagger) to use a real per-tile caption instead, falling
back to --caption for any tile missing from the CSV. The base SD
checkpoint, text encoder, and VAE are frozen; only the ControlNet adapter
is trained (~361M params on top of the AOM3A1B_orangemixs UNet), following
diffusers' train_controlnet.py pattern adapted for a single 12GB GPU
(fp16, gradient checkpointing, grad accum).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ControlNetTileDataset(Dataset):
    def __init__(self, file_list, rough_dir, line_dir, resolution, input_ids_by_name):
        """input_ids_by_name: dict mapping tile name -> tokenized input_ids tensor.
        Pass a defaultdict-like object (e.g. build_input_ids_by_name below) so every
        tile name resolves to something, including a fixed-caption fallback."""
        with open(file_list) as f:
            self.files = [line.strip() for line in f if line.strip()]
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.input_ids_by_name = input_ids_by_name
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        rough = Image.open(os.path.join(self.rough_dir, name)).convert("RGB")
        line = Image.open(os.path.join(self.line_dir, name)).convert("RGB")
        return {
            "pixel_values": self.target_transform(line),
            "conditioning_pixel_values": self.cond_transform(rough),
            "input_ids": self.input_ids_by_name[name],
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", default="dataset/pairs_480/valid_train_combined_koma_20260729.txt")
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--line-dir", default="dataset/pairs_480/train/line_combined_koma_20260729")
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors"),
    )
    parser.add_argument("--output-dir", default="checkpoints/controlnet_koma_direction4")
    parser.add_argument(
        "--controlnet-init",
        default=None,
        help="path to a pretrained ControlNetModel dir to fine-tune from (e.g. a downloaded "
        "public checkpoint), instead of ControlNetModel.from_unet(unet) (fresh copy of the "
        "base UNet's encoder, i.e. training from scratch)",
    )
    parser.add_argument(
        "--controlnet-lora-rank",
        type=int,
        default=None,
        help="if set, freeze --controlnet-init and train only a LoRA adapter of this rank on its "
        "attention layers, instead of full-parameter fine-tuning. Intended for continuing an "
        "already-trained (warm-started) ControlNet checkpoint, where the zero-convolution "
        "safety net that protects from-scratch training no longer applies -- a small, bounded "
        "LoRA delta is a safer perturbation than an unconstrained full fine-tune in that case. "
        "See doc/work_log.md 2026-08-08 entry.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--caption",
        default="monochrome line art, clean linework, manga panel, black and white",
        help="fixed fallback caption, used for every tile unless --caption-csv is given "
        "(and for any tile missing from it)",
    )
    parser.add_argument(
        "--caption-csv",
        default=None,
        help="optional CSV with columns name,caption for real per-tile captions "
        "(see scripts/tag_wd14.py); falls back to --caption for missing tiles",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None, help="overrides --epochs if set (smoke tests)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="interval for the (overwritten, ~4.3GB) crash-resume state -- cheap, can be frequent",
    )
    parser.add_argument(
        "--eval-snapshot-steps",
        type=int,
        default=0,
        help="interval for accumulating weights-only (~1.4GB each) step_N/ eval snapshots; "
        "0 disables (default) since these are NOT overwritten and accumulate on disk",
    )
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="output-dir of a previous run to resume from (reads <dir>/resume_state); "
        "if 'latest', resumes from --output-dir itself",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    )

    pipe = StableDiffusionPipeline.from_single_file(args.base_ckpt, torch_dtype=torch.float32)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe

    if args.controlnet_init:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_init, torch_dtype=torch.float32)
    else:
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.controlnet_lora_rank:
        controlnet.requires_grad_(False)
        lora_config = LoraConfig(
            r=args.controlnet_lora_rank,
            lora_alpha=args.controlnet_lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        controlnet.add_adapter(lora_config)
        n_trainable = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        print(f"[train_controlnet] LoRA mode: rank={args.controlnet_lora_rank}, {n_trainable:,} trainable params (base frozen)")
    controlnet.train()

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    def tokenize(caption):
        return tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    caption_cache = {}

    def cached_tokenize(caption):
        if caption not in caption_cache:
            caption_cache[caption] = tokenize(caption)
        return caption_cache[caption]

    fallback_input_ids = cached_tokenize(args.caption)

    class InputIdsByName(dict):
        def __missing__(self, key):
            return fallback_input_ids

    input_ids_by_name = InputIdsByName()
    if args.caption_csv:
        with open(args.caption_csv) as f:
            for row in csv.DictReader(f):
                input_ids_by_name[row["name"]] = cached_tokenize(row["caption"])
        print(f"[train_controlnet] loaded {len(input_ids_by_name)} per-tile captions from {args.caption_csv}")

    dataset = ControlNetTileDataset(args.file_list, args.rough_dir, args.line_dir, args.resolution, input_ids_by_name)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    trainable_params = [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    steps_per_epoch = max(1, len(dataloader) // args.grad_accum)
    max_train_steps = args.max_train_steps or steps_per_epoch * args.epochs

    controlnet, optimizer, dataloader = accelerator.prepare(controlnet, optimizer, dataloader)

    global_step = 0
    is_latest = args.resume_from_checkpoint == "latest"
    resume_dir = args.output_dir if is_latest else args.resume_from_checkpoint
    resume_state_path = os.path.join(resume_dir, "resume_state") if resume_dir else None
    if resume_state_path and os.path.isdir(resume_state_path):
        accelerator.load_state(resume_state_path)
        with open(os.path.join(resume_state_path, "trainer_state.json")) as f:
            global_step = json.load(f)["global_step"]
        print(f"[train_controlnet] resumed from {resume_state_path} at step {global_step}")
    elif resume_dir and not is_latest:
        raise FileNotFoundError(f"--resume-from-checkpoint given but no state at {resume_state_path}")
    elif resume_dir:
        print(f"[train_controlnet] --resume-from-checkpoint latest: no prior state at {resume_state_path}, starting fresh")

    def save_resume_state(step):
        state_path = os.path.join(args.output_dir, "resume_state")
        accelerator.save_state(state_path)
        with open(os.path.join(state_path, "trainer_state.json"), "w") as f:
            json.dump({"global_step": step}, f)
        print(f"saved {state_path} (resume, overwritten)")

    def save_eval_snapshot(step):
        light_path = os.path.join(args.output_dir, f"step_{step}")
        unwrapped = accelerator.unwrap_model(controlnet)
        if args.controlnet_lora_rank:
            unwrapped.save_lora_adapter(light_path)
        else:
            unwrapped.save_pretrained(light_path)
        print(f"saved {light_path} (weights-only eval snapshot)")

    start_time = time.time()
    print(
        f"[train_controlnet] {len(dataset)} tiles, {steps_per_epoch} steps/epoch, "
        f"target max_train_steps={max_train_steps}, starting from step {global_step}, "
        f"device={accelerator.device}, mixed_precision={args.mixed_precision}"
    )
    if global_step >= max_train_steps:
        print("[train_controlnet] already at/past max_train_steps, nothing to do")
        return
    initial_step = global_step

    done = False
    while not done:
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                input_ids = batch["input_ids"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0].to(weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(weight_dtype),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"unsupported prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                steps_done_this_run = global_step - initial_step
                if global_step % args.log_steps == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"step {global_step}/{max_train_steps} loss={loss.item():.4f} "
                        f"elapsed={elapsed:.0f}s ({elapsed / max(steps_done_this_run, 1):.2f}s/step)"
                    )
                if global_step % args.save_steps == 0 or global_step >= max_train_steps:
                    save_resume_state(global_step)
                if args.eval_snapshot_steps and (
                    global_step % args.eval_snapshot_steps == 0 or global_step >= max_train_steps
                ):
                    save_eval_snapshot(global_step)
                if global_step >= max_train_steps:
                    done = True
                    break

    final_path = os.path.join(args.output_dir, "final")
    unwrapped_final = accelerator.unwrap_model(controlnet)
    if args.controlnet_lora_rank:
        unwrapped_final.save_lora_adapter(final_path)
    else:
        unwrapped_final.save_pretrained(final_path)
    print(f"[train_controlnet] done, saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
