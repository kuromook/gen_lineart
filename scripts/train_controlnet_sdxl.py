"""Candidate #1 from the 2026-08-27 alternative-model survey
(inbox/initial_notice.md): SDXL migration. Fine-tune a LoRA adapter on an
SDXL ControlNet, mirroring ../lineart/scripts/train_controlnet.py's design
(same dataset/CLI conventions, same LoRA-on-frozen-ControlNet approach) but
adapted for SDXL's two text encoders + add_time_ids conditioning, per
diffusers' examples/controlnet/train_controlnet_sdxl.py reference (fetched
2026-08-29; this repo has no local copy of that example script, so this is
a from-scratch adaptation, not a vendored copy).

Base checkpoint: cagliostrolab/animagine-xl-3.1 (anime-tuned SDXL, full
diffusers-format subfolders -- loaded via from_pretrained, no from_single_file
needed). ControlNet init: Eugeoter/noob-sdxl-controlnet-lineart_anime
(already in diffusers ControlNetModel format; trained against a different
base -- Laxhar/sdxl_noob, a messy non-diffusers training-checkpoint dump
not worth chasing -- but SDXL ControlNets are architecturally portable
across same-family anime checkpoints in practice, same tradeoff already
validated for the SD1.5 candidate #2 swap).

VAE kept in fp32 during encode (common SDXL practice -- the stock SDXL VAE
is numerically unstable in fp16) regardless of --mixed-precision.
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
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ControlNetSDXLTileDataset(Dataset):
    def __init__(self, file_list, rough_dir, line_dir, resolution, caption_by_name, fallback_caption):
        with open(file_list) as f:
            self.files = [line.strip() for line in f if line.strip()]
        self.rough_dir = rough_dir
        self.line_dir = line_dir
        self.caption_by_name = caption_by_name
        self.fallback_caption = fallback_caption
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
            "caption": self.caption_by_name.get(name, self.fallback_caption),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", default="data/train_list.txt")
    parser.add_argument("--rough-dir", default="data/rough_lineart_coarse")
    parser.add_argument("--line-dir", default="data/line")
    parser.add_argument("--base-ckpt", default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion-XL/animagine-xl-3.1"))
    parser.add_argument("--output-dir", default="checkpoints/controlnet_lora_sdxl")
    parser.add_argument("--controlnet-init", default=os.path.expanduser("~/disk/checkpoint/ControlNet/noob-sdxl-controlnet-lineart_anime"))
    parser.add_argument("--controlnet-lora-rank", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--caption", default="monochrome line art, clean linework, manga panel")
    parser.add_argument("--caption-csv", default=None, help="CSV with columns name,caption")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None, help="overrides --epochs if set (smoke tests)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
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

    pipe = StableDiffusionXLPipeline.from_pretrained(args.base_ckpt, torch_dtype=torch.float32)
    tokenizer_one, tokenizer_two = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_one, text_encoder_two = pipe.text_encoder, pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe

    # SDXL's ControlNet mirrors SDXL's much larger UNet (~1.25B params, ~5GB
    # fp32) -- unlike SD1.5's ~361M-param ControlNet, keeping it resident in
    # fp32 alone blows the 12GB budget before a single forward pass. Load the
    # frozen base in fp16 and upcast only the (small) trainable LoRA delta
    # back to fp32 after add_adapter, matching diffusers' standard
    # cast_training_params pattern for LoRA fine-tuning under fp16 autocast.
    try:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_init, torch_dtype=weight_dtype)
    except OSError:
        # noob-sdxl-controlnet-lineart_anime only ships the fp16-variant file
        controlnet = ControlNetModel.from_pretrained(args.controlnet_init, torch_dtype=weight_dtype, variant="fp16")

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    controlnet.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.controlnet_lora_rank,
        lora_alpha=args.controlnet_lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    controlnet.add_adapter(lora_config)
    if weight_dtype != torch.float32:
        for param in controlnet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
    n_trainable = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"[train_controlnet_sdxl] LoRA mode: rank={args.controlnet_lora_rank}, {n_trainable:,} trainable params (base frozen, fp16; LoRA delta fp32)")
    controlnet.train()

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    # SDXL's stock VAE is numerically unstable in fp16 -- keep it fp32 regardless
    # of --mixed-precision (standard SDXL training practice).
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    caption_by_name = {}
    if args.caption_csv:
        with open(args.caption_csv) as f:
            for row in csv.DictReader(f):
                caption_by_name[row["name"]] = row["caption"]
        print(f"[train_controlnet_sdxl] loaded {len(caption_by_name)} per-tile captions from {args.caption_csv}")

    def encode_prompt(captions):
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        for tokenizer, text_encoder in [(tokenizer_one, text_encoder_one), (tokenizer_two, text_encoder_two)]:
            input_ids = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)
            out = text_encoder(input_ids, output_hidden_states=True)
            pooled_prompt_embeds = out[0]
            prompt_embeds_list.append(out.hidden_states[-2])
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds.to(weight_dtype), pooled_prompt_embeds.to(weight_dtype)

    dataset = ControlNetSDXLTileDataset(
        args.file_list, args.rough_dir, args.line_dir, args.resolution, caption_by_name, args.caption
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
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
        print(f"[train_controlnet_sdxl] resumed from {resume_state_path} at step {global_step}")
    elif resume_dir and not is_latest:
        raise FileNotFoundError(f"--resume-from-checkpoint given but no state at {resume_state_path}")
    elif resume_dir:
        print(f"[train_controlnet_sdxl] --resume-from-checkpoint latest: no prior state at {resume_state_path}, starting fresh")

    def save_resume_state(step):
        state_path = os.path.join(args.output_dir, "resume_state")
        accelerator.save_state(state_path)
        with open(os.path.join(state_path, "trainer_state.json"), "w") as f:
            json.dump({"global_step": step}, f)
        print(f"saved {state_path} (resume, overwritten)")

    start_time = time.time()
    print(
        f"[train_controlnet_sdxl] {len(dataset)} tiles, {steps_per_epoch} steps/epoch, "
        f"target max_train_steps={max_train_steps}, starting from step {global_step}, "
        f"device={accelerator.device}, mixed_precision={args.mixed_precision}, resolution={args.resolution}"
    )
    if global_step >= max_train_steps:
        print("[train_controlnet_sdxl] already at/past max_train_steps, nothing to do")
        return
    initial_step = global_step

    done = False
    while not done:
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                bsz = pixel_values.shape[0]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents * vae.config.scaling_factor).to(weight_dtype)
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(list(batch["caption"]))

                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                    dtype=weight_dtype, device=accelerator.device,
                ).repeat(bsz, 1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=conditioning_pixel_values,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
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
                if global_step >= max_train_steps:
                    done = True
                    break

    final_path = os.path.join(args.output_dir, "final")
    unwrapped_final = accelerator.unwrap_model(controlnet)
    unwrapped_final.save_lora_adapter(final_path)
    print(f"[train_controlnet_sdxl] done, saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
