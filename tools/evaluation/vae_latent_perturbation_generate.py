"""Track D follow-up to hypothesis 1: how fragile is the VAE decoder to
latent error, and does training-time encoding differ from the ceiling run?

Hypothesis 1 (`vae_roundtrip_generate.py`) showed encode->decode keeps thin
line art (gt_bsds_f1 0.96-0.98 on lineart_family) -- but only under the most
favourable conditions: the encoder's own posterior *mean*, in fp32, fed
straight back to the decoder. Two things it did not cover:

1. **Training-time encoding.** `train_controlnet_consistency.py` encodes GT
   with `.latent_dist.sample()` with the VAE in fp16 (SD1.5).
   `train_controlnet_sdxl.py` samples too, but forces the VAE to fp32. This
   script's per-model default conditions reproduce exactly those.
2. **Decoder sensitivity.** A trained UNet never hands the decoder the
   encoder's exact latent; it hands it something close. If a small latent
   error already breaks 1-3px strokes, a model can drive latent-space loss
   down without output line art improving -- the pattern Track B reported
   and hypothesis 2 is measuring directly.

Method: encode each GT tile once per condition, add isotropic Gaussian noise
of standard deviation sigma **in the scaled latent space**
(`latent * vae.config.scaling_factor`, the space the UNet and its epsilon-MSE
operate in), decode, save. One fixed noise draw per tile is scaled across
all sigmas, so each tile's curve varies only in magnitude. Score with
`vae_roundtrip_score.py`; the manifest's `model` column is set to
`<model>|<condition>|s<sigma>` so its per-(model, pool) summary becomes one
row per condition and sigma.

Reading sigma against training loss: the one-step estimate used by the
consistency loss is x0_hat = (z_t - sqrt(1-abar_t) eps_hat) / sqrt(abar_t),
so its per-element error std is sqrt((1-abar_t)/abar_t * eps_MSE). With the
SD scheduler (scaled_linear, beta 0.00085..0.012, 1000 steps) that factor is
small at low t and grows quickly, which is why the same eps_MSE means very
different pixel damage depending on t.

Run after the GPU is free (the hypothesis-2 probe owns it until it finishes):
    ./venv/bin/python tools/evaluation/vae_latent_perturbation_generate.py \\
        --model both --limit 48 --output-dir results/vae_latent_perturbation_<date>
    ./venv/bin/python tools/evaluation/vae_roundtrip_score.py \\
        --manifest-csv results/vae_latent_perturbation_<date>/manifest.csv \\
        --workers 7 --timeout 60
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from vae_roundtrip_generate import (
    IMAGE_SIZE,
    MODEL_DEFAULTS,
    build_transform,
    normalize_name,
    parse_pool_args,
    read_names,
    resolve_gt_path,
)
from diffusers import AutoencoderKL

# What each training script actually does when it encodes GT.
DEFAULT_CONDITIONS = {
    "sd15": ["mean-fp32", "sample-fp16"],
    "sdxl": ["mean-fp32", "sample-fp32"],
}
DEFAULT_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def parse_condition(cond):
    mode, _, dtype = cond.partition("-")
    if mode not in ("mean", "sample") or dtype not in DTYPES:
        raise SystemExit(f"condition must be mean|sample-fp32|fp16, got {cond!r}")
    return mode, DTYPES[dtype]


def to_image(decoded):
    image = (decoded / 2 + 0.5).clamp(0, 1)
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def run_model(model, args, pools, manifest_rows):
    cfg = MODEL_DEFAULTS[model]
    ckpt = getattr(args, f"{model}_ckpt") or cfg["ckpt"]
    resolution = getattr(args, f"{model}_resolution") or cfg["resolution"]
    conditions = args.conditions or DEFAULT_CONDITIONS[model]
    transform = build_transform(resolution)

    for cond in conditions:
        mode, dtype = parse_condition(cond)
        print(f"[{model}/{cond}] loading VAE from {ckpt} (resolution={resolution}, device={args.device})", file=sys.stderr, flush=True)
        vae = AutoencoderKL.from_single_file(ckpt, torch_dtype=dtype).to(args.device).eval()
        sf = vae.config.scaling_factor

        for pool_label, (list_path, split) in pools.items():
            names = read_names(list_path)
            if args.limit:
                names = names[: args.limit]
            print(f"[{model}/{cond}/{pool_label}] {len(names)} tiles x {len(args.sigmas)} sigmas", file=sys.stderr, flush=True)
            for i, name in enumerate(names):
                t0 = time.time()
                base = normalize_name(name)
                gt_path = resolve_gt_path(args.gt_root, split, name)
                if not gt_path.exists():
                    print(f"  skip (missing GT): {gt_path}", file=sys.stderr, flush=True)
                    continue
                x = transform(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(args.device, dtype)
                # Separate generators so a tile's perturbation direction is identical
                # across conditions; sharing one let `sample` consume draws first.
                dir_gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
                eps_gen = torch.Generator(device="cpu").manual_seed(args.seed + i + 1_000_003)
                with torch.no_grad():
                    dist = vae.encode(x).latent_dist
                    if mode == "mean":
                        z = dist.mean
                    else:
                        eps = torch.randn(dist.mean.shape, generator=eps_gen, dtype=torch.float32).to(args.device, dist.mean.dtype)
                        z = dist.mean + dist.std * eps
                    z = z.float() * sf
                    direction = torch.randn(z.shape, generator=dir_gen, dtype=torch.float32).to(z.device)
                    latent_std = float(z.std())
                    for sigma in args.sigmas:
                        noisy = (z + sigma * direction) / sf
                        decoded = vae.decode(noisy.to(dtype)).sample.float()
                        if not torch.isfinite(decoded).all():
                            print(f"  NON-FINITE decode: {model}/{cond}/s{sigma}/{base} -- not written", file=sys.stderr, flush=True)
                            continue
                        label = f"{model}|{cond}|s{sigma:g}"
                        out_dir = Path(args.output_dir) / "decoded" / model / cond / f"s{sigma:g}" / pool_label
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{base}_out.png"
                        to_image(decoded).save(out_path)
                        manifest_rows.append({
                            "model": label, "pool": pool_label, "sample": base,
                            "gt_path": str(gt_path), "recon_path": str(out_path),
                            "base_model": model, "condition": cond, "sigma": sigma, "latent_std": latent_std,
                        })
                print(f"  [{i + 1}/{len(names)}] {base} latent_std={latent_std:.3f} ({time.time() - t0:.2f}s)", file=sys.stderr, flush=True)

        del vae
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["sd15", "sdxl", "both"], default="both")
    parser.add_argument("--sd15-ckpt", default=None)
    parser.add_argument("--sdxl-ckpt", default=None)
    parser.add_argument("--sd15-resolution", type=int, default=None)
    parser.add_argument("--sdxl-resolution", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="mean|sample-fp32|fp16; default per model mirrors its training script "
                             "(sd15: mean-fp32 sample-fp16, sdxl: mean-fp32 sample-fp32)")
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS,
                        help="noise std in the scaled latent space (latent * scaling_factor)")
    parser.add_argument("--gt-root", default="../lineart/dataset/pairs_480")
    parser.add_argument("--pool", action="append", default=None,
                        help="label=list_path[:split]; default lineart_family + housei holdouts")
    parser.add_argument("--limit", type=int, default=0, help="first N names per pool (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/vae_latent_perturbation")
    parser.add_argument("--manifest-csv", default=None)
    args = parser.parse_args()

    pools = parse_pool_args(args.pool, args.gt_root)
    models = ["sd15", "sdxl"] if args.model == "both" else [args.model]

    manifest_rows = []
    for model in models:
        run_model(model, args, pools, manifest_rows)

    manifest_csv = args.manifest_csv or str(Path(args.output_dir) / "manifest.csv")
    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "pool", "sample", "gt_path", "recon_path", "base_model", "condition", "sigma", "latent_std"]
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n{len(manifest_rows)} decoded images, manifest saved: {manifest_csv}")


if __name__ == "__main__":
    main()
