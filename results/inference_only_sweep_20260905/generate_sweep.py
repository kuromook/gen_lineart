"""Generate diag5 outputs for every config in variants.py, loading the
pipeline once (same pattern as results/word_ablation_20260905/
generate_sweep.py, which kept 63 variants to ~32 min by avoiding a
per-variant model load).

Host checkpoint: controlnet_lora_manga_consistency_20260904 -- the best
combined profile on hand (gt_bsds_f1 0.1411, tied best; orientation_entropy
0.7758, within noise of the best 0.7728). Its training captions all
contained "manga panel", so BASE_CAPTION in variants.py keeps that phrase:
the negative/CFG/conditioning-scale axes are then measured against the
caption distribution this checkpoint actually saw, and only the 1d style
variants deliberately depart from it.

All other generation settings mirror /home/sh1/deepl/lineart/scripts/
infer_controlnet.py exactly (UniPCMultistepScheduler, 512 resolution,
30 steps, seed = 0 + sample index, 480px grayscale output) so these
outputs stay directly comparable to every other eval dir in this track.
"""

import re
import sys
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import VARIANTS  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
ROUGH_DIR = TRACK / "data/diag_rough_manga_line"
OUT_ROOT = TRACK / "results/inference_only_sweep_20260905/outputs"

CONTROLNET_INIT = Path.home() / "disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime"
CONTROLNET_LORA_DIR = TRACK / "checkpoints/controlnet_lora_manga_consistency_20260904/final"
BASE_CKPT = Path.home() / "disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"

IMAGE_SIZE = 480
RESOLUTION = 512
NUM_INFERENCE_STEPS = 30
SEED = 0


def safe_dirname(label):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", label)


def main():
    print(f"[sweep] loading pipeline (controlnet_lora={CONTROLNET_LORA_DIR})", flush=True)
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_INIT, torch_dtype=torch.float16)
    controlnet.load_lora_adapter(
        str(CONTROLNET_LORA_DIR), weight_name="pytorch_lora_weights.safetensors", prefix=None
    )
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        str(BASE_CKPT), controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda")

    roughs = {}
    for s in SAMPLES:
        p = ROUGH_DIR / f"{s}.jpg"
        roughs[s] = Image.open(p).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.BILINEAR)

    print(f"[sweep] {len(VARIANTS)} variants x {len(SAMPLES)} samples", flush=True)
    for vi, v in enumerate(VARIANTS):
        out_dir = OUT_ROOT / safe_dirname(v["label"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(SAMPLES):
            generator.manual_seed(SEED + i)
            result = pipe(
                prompt=v["caption"],
                negative_prompt=v["negative_prompt"] or None,
                image=roughs[s],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=v["guidance_scale"],
                controlnet_conditioning_scale=v["controlnet_conditioning_scale"],
                generator=generator,
            ).images[0]
            out = result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            out.save(out_dir / f"{s}_out.png")
        print(
            f"[{vi + 1}/{len(VARIANTS)}] {v['label']}: cfg={v['guidance_scale']} "
            f"cs={v['controlnet_conditioning_scale']} neg={v['negative_prompt']!r} "
            f"caption={v['caption']!r} -> {out_dir}",
            flush=True,
        )

    print(f"[sweep] done, {len(VARIANTS)} variants written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
