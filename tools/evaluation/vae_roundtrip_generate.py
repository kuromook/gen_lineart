"""Track D, hypothesis 1 (generation phase): round-trip GT line-art tiles
through a VAE's encode/decode, ceiling-measurement style.

Latent diffusion never sees pixels directly -- every target it is trained
against, and every output it produces, passes through a VAE encode/decode
roundtrip. GT line art (1-3px black strokes on white paper) is exactly the
kind of high-frequency, low-area signal a VAE's 8x spatial compression is
worst at preserving, and this has never been measured in this project: every
prior fidelity investigation (`../lineart-controlnet-sd15-refine`,
`../lineart-controlnet-sdxl-fidelity`) measured *trained-model* output
against GT, which conflates "the model didn't learn this" with "the VAE
can't represent this even in principle." See `doc/initial_notice.md`.

This script only generates roundtrip images and a manifest -- scoring against
GT is a **separate script**, `vae_roundtrip_score.py`, run as a separate
process afterwards. That split is deliberate, not stylistic: mixing PyTorch
CUDA calls and OpenCV-based scoring (`measure_lineart_profile.py`,
`tile_region_manifest_480.py`) in one process was observed (this track,
2026-09-13) to cause reproducible multi-minute-to-25+-minute stalls on
specific tiles -- an image would score in 0.03s run standalone moments
later, but hang at the same point inside a process that had also been
calling `vae.encode`/`decode`. `cv2.setNumThreads(0)` did not fix it. Rather
than chase the exact interaction further, generation (all CUDA) and scoring
(all CPU/OpenCV) were split into separate OS processes with no overlap in
their timelines, which structurally rules out whatever the interaction was.

Method: `vae.encode(...).latent_dist.mean` (the deterministic posterior
mean, not a stochastic `.sample()` draw -- this is a ceiling measurement, so
the best-case reconstruction is the right quantity) then `vae.decode(...)`.
Runs SD1.5 (`AOM3A1B_orangemixs`, the base checkpoint every SD1.5 run in
this project actually trained against) and SDXL (`animagine-xl-3.1`, ditto
for the SDXL track) one at a time via `AutoencoderKL.from_single_file` --
loading only the VAE component, not the full UNet/text-encoder pipeline.
Both run in fp32 deliberately, even though this project's inference scripts
normally run SD1.5 in fp16: mixing fp16 rounding into a ceiling measurement
would conflate "the VAE architecture can't represent this" with "fp16 lost
it", and the stock SDXL VAE is documented in this project's own
`train_controlnet_sdxl.py` as numerically unstable in fp16 (hence that
script forces fp32 for its VAE regardless of mixed-precision mode) -- fp32
for both sides keeps the comparison apples-to-apples and reports each
model's more favorable (ceiling) number.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

IMAGE_SIZE = 480

MODEL_DEFAULTS = {
    "sd15": {
        "ckpt": str(Path.home() / "disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors"),
        "resolution": 512,
    },
    "sdxl": {
        "ckpt": str(Path.home() / "disk/checkpoint/Stable-diffusion/animagine-xl-3.1.safetensors"),
        "resolution": 1024,
    },
}

# lineart_family GT lives under train/line, housei under test/line -- verified
# directly against the holdout lists on disk. This is NOT the same
# housei-implies-train guess `evaluate_fixed_outputs.py::dataset_path` makes
# for its own (differently named) sample list; do not copy that logic here.
DEFAULT_POOLS = {
    "lineart_family": ("holdout_lineart_family.txt", "train"),
    "housei": ("holdout_housei_100.txt", "test"),
}


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def read_names(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def resolve_list_path(gt_root, spec):
    path = Path(spec)
    if path.exists():
        return path
    return Path(gt_root) / spec


def resolve_gt_path(gt_root, split_hint, name):
    """Holdout lists are not purely one-split-per-pool: most `lineart_family`
    tiles live under train/line, but 24 of the 192 (the `lineart_004_*`
    series) turned out to live under test/line instead -- found only by the
    first full generation run skipping them as missing (2026-09-13). Try the
    hinted split first (fast path for the common case), then the other."""
    primary = Path(gt_root) / split_hint / "line" / name
    if primary.exists():
        return primary
    other = "test" if split_hint == "train" else "train"
    fallback = Path(gt_root) / other / "line" / name
    return fallback if fallback.exists() else primary


def parse_pool_args(pool_args, gt_root):
    """Returns {label: (list_path, split)}. `--pool` values are
    `label=list_path[:split]`; split defaults to 'train' if omitted."""
    if not pool_args:
        return {
            label: (resolve_list_path(gt_root, list_name), split)
            for label, (list_name, split) in DEFAULT_POOLS.items()
        }
    pools = {}
    for spec in pool_args:
        label, _, rest = spec.partition("=")
        list_part, _, split = rest.partition(":")
        pools[label] = (resolve_list_path(gt_root, list_part), split or "train")
    return pools


def load_vae(ckpt_path, device, dtype, tiling):
    vae = AutoencoderKL.from_single_file(ckpt_path, torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    if tiling:
        vae.enable_slicing()
        vae.enable_tiling()
    return vae


def build_transform(resolution):
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def roundtrip(vae, transform, image_path, device, dtype):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        recon = vae.decode(latent / vae.config.scaling_factor).sample
    recon = (recon / 2 + 0.5).clamp(0, 1)
    arr = (recon[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def run_model(model, args, pools, manifest_rows):
    cfg = MODEL_DEFAULTS[model]
    ckpt = getattr(args, f"{model}_ckpt") or cfg["ckpt"]
    resolution = getattr(args, f"{model}_resolution") or cfg["resolution"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"[{model}] loading VAE from {ckpt} (resolution={resolution}, dtype=fp32, device={device})", file=sys.stderr, flush=True)
    t_load = time.time()
    vae = load_vae(ckpt, device, dtype, args.vae_tiling)
    transform = build_transform(resolution)
    print(f"[{model}] VAE loaded in {time.time() - t_load:.1f}s", file=sys.stderr, flush=True)

    for pool_label, (list_path, split) in pools.items():
        names = read_names(list_path)
        if args.limit:
            names = names[: args.limit]
        out_dir = Path(args.output_dir) / "roundtrip" / model / pool_label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{model}/{pool_label}] {len(names)} tiles", file=sys.stderr, flush=True)
        for i, name in enumerate(names):
            t0 = time.time()
            base = normalize_name(name)
            gt_path = resolve_gt_path(args.gt_root, split, name)
            if not gt_path.exists():
                print(f"  skip (missing GT): {gt_path}", file=sys.stderr, flush=True)
                continue
            recon_img = roundtrip(vae, transform, gt_path, device, dtype)
            out_path = out_dir / f"{base}_out.png"
            recon_img.save(out_path)
            manifest_rows.append(
                {"model": model, "pool": pool_label, "sample": base, "gt_path": str(gt_path), "recon_path": str(out_path)}
            )
            print(f"  [{i + 1}/{len(names)}] {base} ({time.time() - t0:.2f}s)", file=sys.stderr, flush=True)

    del vae
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["sd15", "sdxl", "both"], default="both")
    parser.add_argument("--sd15-ckpt", default=None)
    parser.add_argument("--sdxl-ckpt", default=None)
    parser.add_argument("--sd15-resolution", type=int, default=None)
    parser.add_argument("--sdxl-resolution", type=int, default=None)
    parser.add_argument("--gt-root", default="../lineart/dataset/pairs_480")
    parser.add_argument(
        "--pool", action="append", default=None,
        help="label=list_path[:split] (split is 'train' or 'test', default 'train'); repeatable. "
             "Default: lineart_family (holdout_lineart_family.txt:train), housei (holdout_housei_100.txt:test)",
    )
    parser.add_argument("--limit", type=int, default=0, help="process only the first N names per pool (smoke test)")
    parser.add_argument("--output-dir", default="results/vae_roundtrip_20260913")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument(
        "--vae-tiling", action="store_true",
        help="enable vae.enable_slicing()/enable_tiling() (OOM fallback only -- can introduce seam "
             "artifacts, so it is off by default and should not be used unless generation OOMs)",
    )
    args = parser.parse_args()

    pools = parse_pool_args(args.pool, args.gt_root)
    models = ["sd15", "sdxl"] if args.model == "both" else [args.model]

    manifest_rows = []
    for model in models:
        run_model(model, args, pools, manifest_rows)

    manifest_csv = args.manifest_csv or str(Path(args.output_dir) / "manifest.csv")
    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "pool", "sample", "gt_path", "recon_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{len(manifest_rows)} tiles generated, manifest saved: {manifest_csv}")


if __name__ == "__main__":
    main()
