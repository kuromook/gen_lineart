"""Direction 4 follow-up: unconditional (no pairing, no conditioning image)
LoRA fine-tune of an SD1.5-family checkpoint on a single image domain --
either "rough" or "line", trained separately (invoke this script twice
with different --image-dirs/--caption/--output-dir).

Motivation (doc/work_log.md, `diffusion` branch, 2026-08-04/05): the
ControlNet conditional rough->line translation attempt (`diffusion-
controlnet` branch) hallucinates plausible-but-rough-unrelated content,
and 10x more training steps didn't fix it. Per user direction, set that
conversion task aside and first check whether this base model can even
be adapted to genuinely represent the rough domain and the line-art
domain *individually* -- unconditional txt2img generation quality from a
trigger caption, no conditioning image, no pairing needed. Reuses
scripts/train_controlnet.py's model-loading/Accelerator/resume-state
conventions, with the ControlNet network and paired dataset removed.

Same base checkpoint as the ControlNet work (AOM3A1B_orangemixs, frozen);
only a LoRA adapter injected into the UNet's attention layers (peft) is
trained, following diffusers' own train_text_to_image_lora.py pattern.
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
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class DomainImageDataset(Dataset):
    """Pools every image under one or more directories, no pairing. Each
    sample gets the same fixed caption unless --caption-csv overrides it
    by filename stem (mirrors train_controlnet.py's per-tile caption
    fallback, kept for future per-tile-tag experiments)."""

    def __init__(self, image_dirs, resolution, fallback_input_ids, input_ids_by_stem):
        self.paths = []
        for image_dir in image_dirs:
            for ext in IMAGE_EXTENSIONS:
                self.paths.extend(sorted(Path(image_dir).glob(f"*{ext}")))
        if not self.paths:
            raise SystemExit(f"no images found under {image_dirs}")
        self.fallback_input_ids = fallback_input_ids
        self.input_ids_by_stem = input_ids_by_stem
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        input_ids = self.input_ids_by_stem.get(path.stem, self.fallback_input_ids)
        return {"pixel_values": self.transform(image), "input_ids": input_ids}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dirs", nargs="+", required=True, help="one or more directories pooled together")
    parser.add_argument(
        "--base-ckpt",
        default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caption", required=True, help="fixed trigger caption for every image in this domain")
    parser.add_argument(
        "--caption-csv",
        default=None,
        help="optional CSV with columns name,caption (name = filename stem) overriding --caption per image",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--max-train-steps", type=int, default=None, help="overrides --epochs if set (smoke tests)")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA typically wants a higher LR than full fine-tune")
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--eval-snapshot-steps",
        type=int,
        default=0,
        help="interval for accumulating weights-only LoRA-only eval snapshots (small, a few MB/each); 0 disables",
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

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    unet.train()
    unet.enable_gradient_checkpointing()

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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

    input_ids_by_stem = {}
    if args.caption_csv:
        with open(args.caption_csv) as f:
            for row in csv.DictReader(f):
                input_ids_by_stem[row["name"]] = cached_tokenize(row["caption"])
        print(f"[train_domain_lora] loaded {len(input_ids_by_stem)} per-image captions from {args.caption_csv}")

    dataset = DomainImageDataset(args.image_dirs, args.resolution, fallback_input_ids, input_ids_by_stem)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[train_domain_lora] {len(dataset)} images, {sum(p.numel() for p in lora_params):,} trainable LoRA params")
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    steps_per_epoch = max(1, len(dataloader) // args.grad_accum)
    max_train_steps = args.max_train_steps or int(steps_per_epoch * args.epochs)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    global_step = 0
    is_latest = args.resume_from_checkpoint == "latest"
    resume_dir = args.output_dir if is_latest else args.resume_from_checkpoint
    resume_state_path = os.path.join(resume_dir, "resume_state") if resume_dir else None
    if resume_state_path and os.path.isdir(resume_state_path):
        accelerator.load_state(resume_state_path)
        with open(os.path.join(resume_state_path, "trainer_state.json")) as f:
            global_step = json.load(f)["global_step"]
        print(f"[train_domain_lora] resumed from {resume_state_path} at step {global_step}")
    elif resume_dir and not is_latest:
        raise FileNotFoundError(f"--resume-from-checkpoint given but no state at {resume_state_path}")
    elif resume_dir:
        print(f"[train_domain_lora] --resume-from-checkpoint latest: no prior state at {resume_state_path}, starting fresh")

    def save_resume_state(step):
        state_path = os.path.join(args.output_dir, "resume_state")
        accelerator.save_state(state_path)
        with open(os.path.join(state_path, "trainer_state.json"), "w") as f:
            json.dump({"global_step": step}, f)
        print(f"saved {state_path} (resume, overwritten)")

    def lora_state_dict_for(unwrapped_unet):
        return convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

    def save_eval_snapshot(step):
        light_path = os.path.join(args.output_dir, f"step_{step}")
        unwrapped = accelerator.unwrap_model(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=light_path, unet_lora_layers=lora_state_dict_for(unwrapped), safe_serialization=True
        )
        print(f"saved {light_path} (LoRA-only eval snapshot)")

    start_time = time.time()
    print(
        f"[train_domain_lora] {len(dataset)} images, {steps_per_epoch} steps/epoch, "
        f"target max_train_steps={max_train_steps}, starting from step {global_step}, "
        f"device={accelerator.device}, mixed_precision={args.mixed_precision}, lora_rank={args.lora_rank}"
    )
    if global_step >= max_train_steps:
        print("[train_domain_lora] already at/past max_train_steps, nothing to do")
        return
    initial_step = global_step

    done = False
    while not done:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
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

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
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
                    accelerator.clip_grad_norm_(lora_params, 1.0)
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
    unwrapped = accelerator.unwrap_model(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=final_path, unet_lora_layers=lora_state_dict_for(unwrapped), safe_serialization=True
    )
    print(f"[train_domain_lora] done, saved final LoRA to {final_path}")


if __name__ == "__main__":
    main()
