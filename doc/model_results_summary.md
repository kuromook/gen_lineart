# Model Results Summary

Updated: 2026-07-25 JST

This file summarizes current model-family conclusions so old run-by-run survey
history does not need to remain in active `doc/work_log.md`.

Metrics remain supporting evidence. Montage inspection is still the adoption
gate.

## Global Rules

- Do not use leak-era `shape1` metrics as adoption targets.
- Use clean eval and current reviewed datasets only.
- Treat high F1 from extra ink, halo, or gray texture as suspect until visual
  review confirms it is useful line pickup.

## Current Candidate Families

| family / model | status | main value | main risk |
|---|---|---|---|
| `lucy_mild_aux_msgan` | current safe candidate | balanced halo mitigation and line pickup | still not final line art |
| `lucy_thin_aux_msgan` | high-recall candidate | best recall-oriented Lucy variant | artifact / faint texture scrutiny required |
| `dog_aux_msgan` | controlled white/hint candidate | low ink, good chamfer tradeoff | can lose continuity |
| `flowdog_aux_msgan` | expert candidate | high recall / strong pickup | high ink, likely halo/extra pickup |
| cleanup + MSGAN/FM | promising family | balanced metrics, less emboss than first cleanup | pencil/texture residue remains |
| conservative 2ch U-Net refiner | reference | technically stable refiner path | soft atari-copy behavior |
| aggressive 2ch U-Net refiner | possible expert | recall / black-growth | too much ink globally |
| router/MoE oracle | upper-bound probe | expert choice improves eval upper bound | oracle only, not a deployed router |
| line-field initial refiner | research direction | runs end to end with center/offset heads | overproduces ink and blurs/darkens |

## Clean BCE Baseline

Current historical clean references:

| model | F1@2px | chamfer | ink_ratio | status |
|---|---:|---:|---:|---|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | clean baseline reference |
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | mild clean-baseline variant |

Interpretation:

- Clean BCE baselines are useful reference points, not current model targets.
- Milddup800 is a small numeric improvement but not a qualitative breakthrough.

## ResNet-GAN Atari / 2ch Refiner Direction

Useful ResNet-GAN atari checkpoint:

- `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`

Interpretation:

- ResNet-GAN is useful as a structure/atari generator.
- It is not a final line model.
- Static binary/ink sweeps did not produce a clear final-line solution.

2ch refiner reference checkpoints:

- conservative:
  - `checkpoints/line_refiner_tight_e2_refiner_unet_bin12_ink14/best.pth`
- aggressive:
  - `checkpoints/line_refiner_tight_e2_refiner_unet_bin20_ink16/best.pth`

Interpretation:

- `bin12_ink14` is the conservative refiner reference.
- `bin20_ink16` is an aggressive recall/black-growth expert candidate.
- Static tightening and staged aux-dropout did not solve soft atari-copy.

## Cleanup / MSGAN / Structure Surveys

Best balanced cleanup reference:

- `checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth`

Key comparison:

| model | F1@2px | chamfer | ink_ratio | interpretation |
|---|---:|---:|---:|---|
| `line_refiner_cleanup_e2_cleanup_skel06` | 0.4033 | 5.345 | 1.385 | strong but emboss-like artifact |
| `line_refiner_msgan_e2_cleanup_msgan_fm` | 0.3977 | 5.629 | 1.333 | more balanced, still pencil/texture residue |
| `line_refiner_width_e2_cleanup_width12` | 0.4028 | 5.325 | 1.345 | good ink, artifact remains |
| `line_refiner_structure_e2_cleanup_struct08_msgan` | 0.3982 | 5.664 | 1.400 | balanced, no clear jump over MSGAN/FM |

Interpretation:

- Cleanup architecture is promising.
- The first cleanup form can create white/emboss-like halos.
- MSGAN/FM improves balance but does not fully remove pencil/texture residue.

## Halo / Lucy Hint Direction

Current important candidates:

| model | F1@2px | chamfer | ink_ratio | interpretation |
|---|---:|---:|---:|---|
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | controlled white/hint candidate |
| `halo_filter_flowmask_e2_lucy_aux_msgan` | 0.3932 | 4.901 | 1.110 | balanced deconvolution-style candidate |
| `lucy_mask_deep_e2_lucy_mild_aux_msgan` | 0.4021 | 4.645 | 1.158 | current safer Lucy candidate |
| `lucy_mask_deep_e2_lucy_thin_aux_msgan` | 0.4095 | 4.536 | 1.252 | high-recall candidate needing artifact scrutiny |
| `halo_filter_flowmask_e2_flowdog_aux_msgan` | 0.4046 | 4.697 | 1.550 | high-recall/high-ink expert candidate |

Interpretation:

- `lucy_mild_aux_msgan` is the current best controlled Lucy candidate.
- `lucy_thin_aux_msgan` is stronger numerically but may pick up faint/halo-like
  texture.
- `flowdog_aux_msgan` is not a halo-suppression solution; keep as an expert
  candidate at most.

## Router / MoE Oracle

Oracle result:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `router_linefield_overnight_e2_oracle` | 0.4317 | 4.598 | 1.582 |

Oracle selected experts on the 8 clean eval tiles:

- `lucy_mild`: 3
- `lucy_thin`: 2
- `flowdog`: 1
- `bin20`: 1
- `bin12`: 1

Interpretation:

- Expert choice is useful as an upper-bound probe.
- This does not prove a trained router is ready.
- Future router work should use current expert candidates and broader reviewed
  samples, not only the 8 clean eval tiles.

## Line-Field Direction

Initial line-field result:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `linefield_initial_e2` | 0.4061 | 5.481 | 2.843 |

Artifacts:

- checkpoint:
  - `checkpoints/linefield_initial_e2/best.pth`
- montage:
  - `results/compare_linefield_initial_e2.png`
- note:
  - per-sample output/debug images under `results/` were removed during results
    image cleanup; keep the checkpoint, montage, and metrics as durable
    references

Interpretation:

- The line-field formulation runs end to end.
- The first version overproduces ink and looks too blurred/dark.
- Continue only with stronger ink/width constraints or revised reconstruction.

## Current Recommendation

Near-term model work should wait until the next reviewed dataset candidate is
stable, unless the task is specifically model-side.

If model work resumes first:

1. Use `lucy_mild_aux_msgan` as the safer current candidate.
2. Keep `lucy_thin_aux_msgan` and `flowdog_aux_msgan` as expert candidates.
3. Treat cleanup + MSGAN/FM as promising but artifact-prone.
4. Do not deepen line-field without fixing ink/width behavior.
