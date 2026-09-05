"""Exhaustive caption-word ablation sweep (2026-09-05). For each candidate
tag in candidate_tags.py, generate the 5 diag5 samples with that single
word PREPENDED to the plain baseline caption ("monochrome line art, clean
linework, black and white" -- the manga_nomangaword_trained checkpoint's
own eval caption, matching how a per-tile tag would precede the fixed
suffix at training time), holding the checkpoint fixed
(checkpoints/controlnet_lora_manga_nomangaword_20260905/final -- the most
recently trained, least keyword-contaminated checkpoint on hand). The
all-zero-added-words baseline is already
results/controlnet_lora_manga_nomangaword_20260905_eval/ (orientation_entropy
mean 0.7860) and is not regenerated here.

Loads the pipeline once and loops over words to avoid ~10s of per-call
model-loading overhead x 63 words. Mirrors
/home/sh1/deepl/lineart/scripts/infer_controlnet.py's generation settings
exactly (same scheduler, resolution, guidance_scale, controlnet_conditioning_scale,
seed convention) so outputs are directly comparable to every other eval
dir in this track.
"""

import re
import sys
from pathlib import Path

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_tags import CANDIDATE_TAGS  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]
ROUGH_DIR = TRACK / "data/diag_rough_manga_line"
OUT_ROOT = TRACK / "results/word_ablation_20260905/outputs"
BASE_CAPTION = "monochrome line art, clean linework, black and white"

CONTROLNET_INIT = Path.home() / "disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime"
CONTROLNET_LORA_DIR = TRACK / "checkpoints/controlnet_lora_manga_nomangaword_20260905/final"
BASE_CKPT = Path.home() / "disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"

IMAGE_SIZE = 480
RESOLUTION = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.0
CONTROLNET_CONDITIONING_SCALE = 1.0
SEED = 0


def safe_dirname(tag):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", tag)


def main():
    print(f"[word_ablation] loading pipeline (controlnet_lora={CONTROLNET_LORA_DIR})", flush=True)
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

    print(f"[word_ablation] {len(CANDIDATE_TAGS)} candidate tags x {len(SAMPLES)} samples", flush=True)
    for wi, tag in enumerate(CANDIDATE_TAGS):
        caption = f"{tag}, {BASE_CAPTION}"
        out_dir = OUT_ROOT / safe_dirname(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(SAMPLES):
            generator.manual_seed(SEED + i)
            result = pipe(
                prompt=caption,
                image=roughs[s],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                generator=generator,
            ).images[0]
            out = result.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            out.save(out_dir / f"{s}_out.png")
        print(f"[{wi + 1}/{len(CANDIDATE_TAGS)}] tag={tag!r} caption={caption!r} -> {out_dir}", flush=True)

    print(f"[word_ablation] done, {len(CANDIDATE_TAGS)} tags written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
