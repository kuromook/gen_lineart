"""Track D, hypothesis 2 follow-up: is the training objective itself decoupled
from line-art quality, or was the logged loss just too noisy to show it?

The instrumented probe (`score_loss_quality_snapshots.py`,
`doc/work_log.md` 2026-09-15) found the *logged* loss flat within its own
noise while gt_bsds_f1 / fill / continuity moved significantly. That log is
one batch per 50 steps at a random timestep with fresh noise, so it cannot
separate "the objective does not track quality" from "the estimate is too
noisy to see it".

This script removes both noise sources. Every snapshot is evaluated on the
same held-out tiles, at the same fixed grid of timesteps, with the same
noise tensor per (tile, timestep) and the same GT latent (posterior mean).
The only thing that changes between snapshots is the ControlNet LoRA, so
differences in loss are differences in the model.

It reproduces the training objective of
`../lineart-controlnet-sd15-refine/scripts/train_controlnet_consistency.py`:
epsilon-MSE in latent space for every timestep, and for timesteps below
`--consistency-max-timestep` the Sobel edge-magnitude L1 between the VAE
decode of the one-step x0 estimate and the GT pixels. Same base checkpoint,
same ControlNet init, same caption the training run fell back to (holdout
tiles are not in `captions.csv`), same conditioning transform. Differences
from training, deliberate: posterior mean instead of `.sample()` (the SD1.5
posterior std is so small the two decode within one grey level -- measured
2026-09-14) and no dropout-like randomness anywhere.

Two analyses (`--analyze-only` reruns them from the saved per-row CSV):

1. Snapshot level (n = 11): Pearson/Spearman between each loss summary
   (uniform-t epsilon MSE, low/mid/high-t bands, consistency L1, and the
   composite the optimiser actually saw) and each quality axis
   (gt_bsds_f1, precision, recall, near_white, fill_ratio, line width).
   Tile-bootstrap standard errors show whether snapshot differences clear
   the estimate's own noise.
2. Tile level: for each consecutive snapshot pair, each tile's change in
   loss against its change in gt_bsds_f1, pooled over ~1,900 pairs. Far more
   power than 11 points; a real coupling should show here if anywhere.

Run (GPU): ./venv/bin/python tools/evaluation/fixed_t_validation_loss.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

TRACK_A = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine")
PROBE_TAG = "controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914"
FALLBACK_CAPTION = "monochrome line art, clean linework, manga panel, black and white"
ROW_FIELDS = ["label", "step", "sample", "t", "eps_mse", "consistency_l1"]


def default_timesteps():
    return [25 + 50 * k for k in range(20)]


def discover_snapshots(ckpt_root, include_base, only_labels):
    snaps = []
    if include_base:
        snaps.append(("base", 0, None))
    for d in Path(ckpt_root).iterdir():
        if d.is_dir() and d.name.startswith("step_") and (d / "pytorch_lora_weights.safetensors").exists():
            snaps.append((d.name, int(d.name.split("_")[1]), d))
    # `final` is byte-for-byte the same tensors as the last step_N snapshot
    # (verified 2026-09-14), so it is never evaluated separately.
    snaps.sort(key=lambda s: s[1])
    if only_labels:
        snaps = [s for s in snaps if s[0] in only_labels]
    return snaps


def sobel_edge_magnitude(x, sobel_x, sobel_y):
    import torch
    import torch.nn.functional as F

    gray = x.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def build_cache(args, device):
    import torch
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from PIL import Image
    from torchvision import transforms

    fp16 = torch.float16
    pipe = StableDiffusionPipeline.from_single_file(args.base_ckpt, torch_dtype=fp16, safety_checker=None)
    tokenizer, text_encoder, vae, unet = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe
    if scheduler.config.prediction_type != "epsilon":
        raise SystemExit(f"expected epsilon prediction, got {scheduler.config.prediction_type}")
    for m in (text_encoder, vae, unet):
        m.to(device).eval().requires_grad_(False)

    names = [line.strip() for line in open(args.sample_list) if line.strip()]
    if args.limit:
        names = names[: args.limit]

    target_tf = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    cond_tf = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        ids = tokenizer(args.caption, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        text_embed = text_encoder(ids)[0].to(fp16)

        latents, gt_pixels, conds = [], [], []
        for n in names:
            gt = target_tf(Image.open(Path(args.gt_dir) / n).convert("RGB")).unsqueeze(0)
            cond = cond_tf(Image.open(Path(args.rough_dir) / n).convert("RGB")).unsqueeze(0)
            z = vae.encode(gt.to(device, fp16)).latent_dist.mean * vae.config.scaling_factor
            latents.append(z.to("cpu", fp16))
            gt_pixels.append(gt.to(fp16))
            conds.append(cond.to(fp16))
    latents = torch.cat(latents)
    gt_pixels = torch.cat(gt_pixels)
    conds = torch.cat(conds)

    noises = torch.empty((len(names), len(args.timesteps)) + tuple(latents.shape[1:]), dtype=fp16)
    for i in range(len(names)):
        for j in range(len(args.timesteps)):
            g = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + i * 1000 + j)
            noises[i, j] = torch.randn(latents.shape[1:], generator=g, dtype=torch.float32).to(fp16)

    return {
        "names": names, "text_embed": text_embed, "latents": latents, "gt_pixels": gt_pixels,
        "conds": conds, "noises": noises, "vae": vae, "unet": unet, "scheduler": scheduler,
    }


def load_controlnet(args, lora_dir, device):
    import torch
    from diffusers import ControlNetModel

    controlnet = ControlNetModel.from_pretrained(args.controlnet_init, torch_dtype=torch.float32)
    n_lora = 0
    if lora_dir is not None:
        controlnet.load_lora_adapter(str(lora_dir), weight_name="pytorch_lora_weights.safetensors", prefix=None)
        n_lora = sum(p.numel() for name, p in controlnet.named_parameters() if "lora_" in name)
        if n_lora == 0:
            raise SystemExit(f"LoRA from {lora_dir} attached no parameters")
    return controlnet.to(device).eval().requires_grad_(False), n_lora


def evaluate_snapshot(args, cache, controlnet, device):
    import torch

    fp16 = torch.float16
    vae, unet, scheduler = cache["vae"], cache["unet"], cache["scheduler"]
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sf = vae.config.scaling_factor
    n = len(cache["names"])
    out = []

    for j, t in enumerate(args.timesteps):
        for start in range(0, n, args.batch_size):
            sl = slice(start, min(start + args.batch_size, n))
            b = sl.stop - sl.start
            z0 = cache["latents"][sl].to(device).float()
            noise = cache["noises"][sl, j].to(device).float()
            tt = torch.full((b,), t, device=device, dtype=torch.long)
            noisy = scheduler.add_noise(z0, noise, tt).to(fp16)
            cond = cache["conds"][sl].to(device, fp16)
            ehs = cache["text_embed"].expand(b, -1, -1)
            with torch.no_grad(), torch.autocast("cuda", dtype=fp16, enabled=device.startswith("cuda")):
                down, mid = controlnet(noisy, tt, encoder_hidden_states=ehs, controlnet_cond=cond, return_dict=False)
                pred = unet(
                    noisy, tt, encoder_hidden_states=ehs,
                    down_block_additional_residuals=[s.to(fp16) for s in down],
                    mid_block_additional_residual=mid.to(fp16), return_dict=False,
                )[0]
            pred = pred.float()
            eps = ((pred - noise) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

            cons = [None] * b
            if t < args.consistency_max_timestep:
                ab = alphas_cumprod[t].float()
                x0 = (noisy.float() - (1 - ab).sqrt() * pred) / ab.sqrt()
                vals = []
                with torch.no_grad():
                    for k in range(0, b, args.decode_batch):
                        dec = vae.decode((x0[k:k + args.decode_batch] / sf).to(fp16), return_dict=False)[0].float().clamp(-1, 1)
                        gt = cache["gt_pixels"][sl][k:k + args.decode_batch].to(device).float()
                        diff = (sobel_edge_magnitude(dec, sobel_x, sobel_y) - sobel_edge_magnitude(gt, sobel_x, sobel_y)).abs()
                        vals.append(diff.mean(dim=(1, 2, 3)).cpu().numpy())
                cons = list(np.concatenate(vals))

            for k in range(b):
                out.append((cache["names"][sl.start + k], t, float(eps[k]), None if cons[k] is None else float(cons[k])))
    return out


def run(args):
    import torch

    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "fixed_t_losses.csv"
    done_labels = set()
    if rows_csv.exists():
        with open(rows_csv) as f:
            done_labels = {r["label"] for r in csv.DictReader(f)}

    snapshots = discover_snapshots(args.ckpt_root, args.include_base, args.labels)
    todo = [s for s in snapshots if s[0] not in done_labels]
    print(f"snapshots: {[s[0] for s in snapshots]}; already done: {sorted(done_labels)}", file=sys.stderr, flush=True)
    if not todo:
        return

    t0 = time.time()
    cache = build_cache(args, device)
    print(f"cache: {len(cache['names'])} tiles x {len(args.timesteps)} timesteps ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    new_file = not rows_csv.exists()
    with open(rows_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        for label, step, lora_dir in todo:
            ts = time.time()
            controlnet, n_lora = load_controlnet(args, lora_dir, device)
            rows = evaluate_snapshot(args, cache, controlnet, device)
            for name, t, eps, cons in rows:
                writer.writerow({"label": label, "step": step, "sample": name, "t": t,
                                 "eps_mse": f"{eps:.6f}", "consistency_l1": "" if cons is None else f"{cons:.6f}"})
            f.flush()
            mean_eps = float(np.mean([r[2] for r in rows]))
            print(f"[{label}] lora_params={n_lora:,} rows={len(rows)} mean eps_mse={mean_eps:.5f} ({time.time() - ts:.0f}s)",
                  file=sys.stderr, flush=True)
            del controlnet
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def load_quality(results_root):
    """Per-snapshot, per-tile quality from the probe's per_tile CSVs."""
    quality = {}
    per_tile = Path(results_root) / "per_tile"
    if not per_tile.is_dir():
        return quality
    for p in per_tile.glob("step_*.csv"):
        with open(p) as f:
            quality[p.stem] = {r["sample"]: r for r in csv.DictReader(f) if r["timed_out"] == "False"}
    return quality


def analyze(args):
    from scipy import stats

    rows_csv = Path(args.output_dir) / "fixed_t_losses.csv"
    by_label = {}
    with open(rows_csv) as f:
        for r in csv.DictReader(f):
            by_label.setdefault(r["label"], []).append(r)
    labels = sorted(by_label, key=lambda l: int(by_label[l][0]["step"]))
    rng = np.random.default_rng(0)
    quality = load_quality(args.results_root)

    tile_eps, tile_cons, summary = {}, {}, []
    for label in labels:
        rows = by_label[label]
        names = sorted({r["sample"] for r in rows})
        eps_by_tile = {n: [] for n in names}
        cons_by_tile = {n: [] for n in names}
        bands = {"eps_low": [], "eps_mid": [], "eps_high": []}
        for r in rows:
            t, e = int(r["t"]), float(r["eps_mse"])
            eps_by_tile[r["sample"]].append(e)
            bands["eps_low" if t < 200 else "eps_mid" if t < 600 else "eps_high"].append(e)
            if r["consistency_l1"] != "":
                cons_by_tile[r["sample"]].append(float(r["consistency_l1"]))
        te = np.array([np.mean(eps_by_tile[n]) for n in names])
        tc = np.array([np.mean(cons_by_tile[n]) for n in names]) if any(cons_by_tile.values()) else np.full(len(names), np.nan)
        tile_eps[label] = dict(zip(names, te))
        tile_cons[label] = dict(zip(names, tc))
        boot = [te[rng.integers(0, len(te), len(te))].mean() for _ in range(2000)]
        p_any = 1 - (1 - args.consistency_max_timestep / 1000.0) ** args.train_batch_size
        row = {
            "label": label, "step": int(rows[0]["step"]), "n_tiles": len(names),
            "eps_uniform": float(te.mean()), "eps_uniform_se": float(np.std(boot)),
            **{k: float(np.mean(v)) if v else float("nan") for k, v in bands.items()},
            "consistency_l1": float(np.nanmean(tc)),
        }
        row["composite"] = row["eps_uniform"] + args.consistency_weight * p_any * row["consistency_l1"]
        q = quality.get(label)
        if q:
            common = [n for n in names if n in q]
            for key, col in (("gt_bsds_f1", "bsds_f1"), ("precision", "bsds_precision"), ("recall", "bsds_recall"),
                             ("near_white", "recon_near_white_frac"), ("fill_ratio", "recon_fill_ratio"),
                             ("line_width_p50", "recon_line_width_p50")):
                row[key] = float(np.mean([float(q[n][col]) for n in common]))
        summary.append(row)

    prev = None
    print("\nlabel        step   eps_uniform (+-SE)   d_vs_prev (+-SE)   eps_low  eps_mid  eps_high  consist   f1")
    for row in summary:
        d_txt = ""
        if prev is not None:
            names = sorted(set(tile_eps[row["label"]]) & set(tile_eps[prev["label"]]))
            d = np.array([tile_eps[row["label"]][n] - tile_eps[prev["label"]][n] for n in names])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            row["d_eps_vs_prev"], row["d_eps_vs_prev_se"] = float(d.mean()), float(np.std(boot))
            d_txt = f"{d.mean():+.5f} ({np.std(boot):.5f})"
        f1 = row.get("gt_bsds_f1")
        print(f"{row['label']:<11}{row['step']:>6}   {row['eps_uniform']:.5f} ({row['eps_uniform_se']:.5f})   {d_txt:<19}"
              f"{row['eps_low']:>8.5f}{row['eps_mid']:>9.5f}{row['eps_high']:>10.5f}{row['consistency_l1']:>9.5f}"
              f"{'' if f1 is None else f'{f1:>7.4f}'}")
        prev = row

    fields = sorted({k for r in summary for k in r}, key=lambda k: (k not in ("label", "step"), k))
    with open(Path(args.output_dir) / "fixed_t_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    scored = [r for r in summary if "gt_bsds_f1" in r]
    corr_rows = []
    if len(scored) >= 3:
        print(f"\nsnapshot-level correlations (n={len(scored)})")
        for xk in ("eps_uniform", "eps_low", "eps_mid", "eps_high", "consistency_l1", "composite"):
            for yk in ("gt_bsds_f1", "precision", "recall", "near_white", "fill_ratio", "line_width_p50"):
                x = np.array([r[xk] for r in scored]); y = np.array([r[yk] for r in scored])
                pr, pp = stats.pearsonr(x, y); sr, sp = stats.spearmanr(x, y)
                corr_rows.append({"level": "snapshot", "x": xk, "y": yk, "n": len(scored),
                                  "pearson": pr, "pearson_p": pp, "spearman": sr, "spearman_p": sp})
                print(f"  {xk:<15} vs {yk:<15} pearson {pr:+.3f} (p={pp:.2g})  spearman {sr:+.3f} (p={sp:.2g})")

    pooled = {"eps": ([], []), "consistency": ([], [])}
    per_pair = []
    for a, b in zip(scored, scored[1:]):
        qa, qb = quality[a["label"]], quality[b["label"]]
        names = sorted(set(tile_eps[a["label"]]) & set(tile_eps[b["label"]]) & set(qa) & set(qb))
        df1 = np.array([float(qb[n]["bsds_f1"]) - float(qa[n]["bsds_f1"]) for n in names])
        de = np.array([tile_eps[b["label"]][n] - tile_eps[a["label"]][n] for n in names])
        dc = np.array([tile_cons[b["label"]][n] - tile_cons[a["label"]][n] for n in names])
        pooled["eps"][0].extend(de); pooled["eps"][1].extend(df1)
        pooled["consistency"][0].extend(dc); pooled["consistency"][1].extend(df1)
        per_pair.append((f"{a['label']}->{b['label']}", stats.spearmanr(de, df1)[0], stats.spearmanr(dc, df1)[0], len(names)))
    if per_pair:
        print("\ntile-level: change in loss vs change in gt_bsds_f1, consecutive snapshots")
        for pair, re_, rc, nn in per_pair:
            print(f"  {pair:<24} n={nn}  spearman(d_eps, d_f1) {re_:+.3f}  spearman(d_consistency, d_f1) {rc:+.3f}")
        for key, (x, y) in pooled.items():
            sr, sp = stats.spearmanr(x, y)
            corr_rows.append({"level": "tile_delta_pooled", "x": f"d_{key}", "y": "d_gt_bsds_f1", "n": len(x),
                              "pearson": stats.pearsonr(x, y)[0], "pearson_p": stats.pearsonr(x, y)[1],
                              "spearman": sr, "spearman_p": sp})
            print(f"  pooled d_{key:<12} vs d_f1  n={len(x)}  spearman {sr:+.3f} (p={sp:.2g})")

    if corr_rows:
        with open(Path(args.output_dir) / "fixed_t_correlations.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
            w.writeheader()
            w.writerows(corr_rows)
    print(f"\nsaved: {Path(args.output_dir) / 'fixed_t_summary.csv'}, {Path(args.output_dir) / 'fixed_t_correlations.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-root", default=str(TRACK_A / "checkpoints" / PROBE_TAG))
    parser.add_argument("--results-root", default=str(TRACK_A / "results" / PROBE_TAG))
    parser.add_argument("--sample-list", default=str(TRACK_A / "data/holdout_lineart_family.txt"))
    parser.add_argument("--gt-dir", default=str(TRACK_A / "data/holdout_lineart_family_gt_line"))
    parser.add_argument("--rough-dir", default=str(TRACK_A / "data/holdout_lineart_family_rough_manga_line"))
    parser.add_argument("--base-ckpt", default=str(Path.home() / "disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors"))
    parser.add_argument("--controlnet-init", default=str(Path.home() / "disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime"))
    parser.add_argument("--caption", default=FALLBACK_CAPTION)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--timesteps", nargs="+", type=int, default=default_timesteps())
    parser.add_argument("--labels", nargs="+", default=None, help="evaluate only these snapshot labels")
    parser.add_argument("--include-base", action="store_true", help="also evaluate the ControlNet init with no LoRA (step 0)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--consistency-max-timestep", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=2, help="for the composite: P(a batch contains t<max) = 1-(1-p)^bs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/fixed_t_validation_20260915")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
