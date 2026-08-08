"""Plug-and-Play-style (self-attention feature injection) rough->line
conversion: DDIM-invert the rough tile to get its exact noise trajectory
and self-attention key/value maps, then generate the line-art version
starting from that inverted latent, substituting the source's self-
attention key/value into the early (high-noise, layout-determining)
denoising steps. No training required -- reuses the already-adopted
line-domain LoRA.

Motivation (doc/work_log.md, `diffusion` branch, 2026-08-08): plain
SDEdit (scripts/sdedit_line_from_rough.py, strength-based noise
blending) did not clearly beat ControlNet on structure fidelity -- even
at low strength, output sometimes drifted to unrelated content (faces),
and chamfer-to-GT got worse as strength increased. Two things plain
SDEdit throws away that this script keeps: (1) the exact DDIM-inverted
starting latent instead of a strength-scaled random-noise blend (a
lossy, non-invertible approximation), and (2) the source image's own
self-attention layout (via key/value injection into attn1 layers), which
directly constrains *where* the model attends spatially during the
layout-determining early steps, independent of how strongly the prompt/
LoRA pulls toward a different subject.

Design notes:
- Self-attention only (`attn1`), not cross-attention (`attn2`, which
  carries the text/LoRA-style signal we want to keep steering generation
  toward the line-art domain).
- Key/value substituted (query kept from the generation pass) -- this is
  the "mutual self-attention control" formulation (source K/V, target Q):
  the target's queries end up attending to the source's spatial content,
  which is what constrains layout, while the target's own query
  projections (shaped by the evolving generation + LoRA) still determine
  *what* each position is asking about.
- `--injection-fraction` (default 0.6): only the first N% of denoising
  steps (highest noise, coarse layout) use injected key/value; later
  steps fall back to normal self-attention so fine style/rendering is
  free to follow the LoRA/prompt rather than being locked to the source's
  own (rough-sketch) texture.
- Classifier-free guidance is intentionally disabled (guidance_scale
  fixed at 1.0, no uncond branch) to keep the injection cache's batch
  dimension trivially aligned between inversion and generation -- adding
  CFG support would require splitting cond/uncond batches and only
  injecting into the cond half, deferred as a possible follow-up if this
  cheap version shows a real effect worth refining.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 480


class PNPAttnProcessor:
    """Drop-in replacement for AttnProcessor2_0 (self-attention only) that
    can save key/value into a shared cache keyed by (layer_name, timestep)
    during a DDIM-inversion pass, and substitute cached key/value back in
    during a generation pass, up to an injection-fraction step threshold.
    """

    def __init__(self, layer_name, cache):
        self.layer_name = layer_name
        self.cache = cache
        self.mode = "none"  # "save" | "inject" | "none"
        self.current_t = None
        self.current_step_frac = 0.0
        self.injection_fraction = 0.6

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        assert encoder_hidden_states is None, "PNPAttnProcessor is for self-attention (attn1) only"
        encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.mode == "save" and self.current_t is not None:
            self.cache[(self.layer_name, self.current_t)] = (key.detach(), value.detach())
        elif self.mode == "inject" and self.current_step_frac <= self.injection_fraction:
            cached = self.cache.get((self.layer_name, self.current_t))
            if cached is not None:
                key, value = cached

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def encode_prompt(tokenizer, text_encoder, prompt, device):
    ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    with torch.no_grad():
        return text_encoder(ids.to(device))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--rough-dir", default="dataset/pairs_480/train/rough")
    parser.add_argument("--lora-dir", default="checkpoints/domain_lora_line_sd15base_sksv2_20260807/final")
    parser.add_argument("--base-ckpt", default=os.path.expanduser("~/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"))
    parser.add_argument("--caption", default="sks style, monochrome line art, manga panel, black and white")
    parser.add_argument("--inversion-caption", default="", help="prompt used during DDIM inversion; empty is standard practice for faithful inversion")
    parser.add_argument("--lora-scale", type=float, default=1.4)
    parser.add_argument("--injection-fraction", type=float, default=0.6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"results/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.float16
    pipe = StableDiffusionPipeline.from_single_file(args.base_ckpt, torch_dtype=dtype, safety_checker=None)
    pipe.load_lora_weights(args.lora_dir)
    unet, vae, tokenizer, text_encoder = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    cache = {}
    processors = {}
    for name in unet.attn_processors:
        if "attn1" in name:
            processors[name] = PNPAttnProcessor(name, cache)
        else:
            processors[name] = AttnProcessor2_0()
    # unet.set_attn_processor() pops entries out of the dict it's given
    # (diffusers internally does processor.pop(...) per submodule), so the
    # `processors` dict is empty after this call -- keep pnp_processors as
    # a separate reference to the actual PNPAttnProcessor instances now
    # attached to the UNet, for setting mode/current_t later.
    pnp_processors = [p for p in processors.values() if isinstance(p, PNPAttnProcessor)]
    unet.set_attn_processor(processors)
    assert len(pnp_processors) > 0, "no self-attention (attn1) layers found to hook"

    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    gen_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv_scheduler.set_timesteps(args.num_inference_steps, device=device)
    gen_scheduler.set_timesteps(args.num_inference_steps, device=device)

    image_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    inv_embeds = encode_prompt(tokenizer, text_encoder, args.inversion_caption, device).to(dtype)
    gen_embeds = encode_prompt(tokenizer, text_encoder, args.caption, device).to(dtype)
    cross_attention_kwargs = {"scale": args.lora_scale}

    samples = read_sample_list(args.sample_list)
    for i, name in enumerate(samples):
        base = normalize_name(name)
        rough_path = Path(args.rough_dir) / f"{base}.jpg"
        rough = Image.open(rough_path).convert("RGB")
        pixel_values = image_transform(rough).unsqueeze(0).to(device, dtype)

        with torch.no_grad():
            latent = vae.encode(pixel_values).latent_dist.mean * vae.config.scaling_factor

            # --- inversion pass: fill the self-attention cache, get final noised latent ---
            for p in pnp_processors:
                p.mode = "save"
            inv_latent = latent.clone()
            num_steps = len(inv_scheduler.timesteps)
            for step_index, t in enumerate(inv_scheduler.timesteps):
                for p in pnp_processors:
                    p.current_t = int(t)
                noise_pred = unet(inv_latent, t, encoder_hidden_states=inv_embeds, cross_attention_kwargs=cross_attention_kwargs).sample
                inv_latent = inv_scheduler.step(noise_pred, t, inv_latent).prev_sample

            # --- generation pass: start from the inverted latent, inject cached K/V early ---
            for p in pnp_processors:
                p.mode = "inject"
                p.injection_fraction = args.injection_fraction
            gen_latent = inv_latent
            for step_index, t in enumerate(gen_scheduler.timesteps):
                frac = step_index / max(num_steps - 1, 1)
                for p in pnp_processors:
                    p.current_t = int(t)
                    p.current_step_frac = frac
                noise_pred = unet(gen_latent, t, encoder_hidden_states=gen_embeds, cross_attention_kwargs=cross_attention_kwargs).sample
                gen_latent = gen_scheduler.step(noise_pred, t, gen_latent).prev_sample

            image = vae.decode(gen_latent / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
        out = Image.fromarray(image).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        out_path = output_dir / f"{base}_out.png"
        out.save(out_path)
        cache.clear()
        print(f"[{i + 1}/{len(samples)}] {rough_path} -> {out_path}")

    print(f"[pnp_line_from_rough] done, {len(samples)} samples written to {output_dir}")


if __name__ == "__main__":
    main()
