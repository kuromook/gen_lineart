# Work Log - Clean Rebuild

## 2026-07-18

### Reset Context After Leakage

The previous project history contained substantial leakage-driven assumptions.
Treat pre-clean-eval model scores and conclusions as historical only.

Current rule:

- Do not use old `shape1` metrics as adoption targets.
- Use leaky `shape1` only as a visual failure reference.
- Prefer clean lineart004 evaluation and integrity-audited training lists.
- Keep MoE/specialist work paused until a clean base model behaves reasonably.

Historical Markdown logs were moved out of the active `doc/*.md` set:

- `doc/archive/work_log_pre_clean_eval_leak_history.txt`
- `doc/archive/moe_plan_pre_clean_eval_leak_history.txt`
- `doc/archive/autoloop_handoff_shape1_base_clean_unique_bce.txt`
- `doc/archive/MEMORY_pre_clean_eval_leak_history.txt`
- `doc/archive/NOTES_early_history.txt`

Archive rule: do not read files under `archive/` directories unless the user
explicitly asks for archived history or audit material.

### Clean BCE Baseline Comparison

Completed comparison:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_base_clean_unique_bce` | 0.2786 | 7.718 | 1.600 | 0.2661 | 0.2952 |

Conclusion: `shape1_base_clean_unique_bce` is worse than
`shape1_clean_split_bce_lineart004`.

Artifacts:

- `results/compare_clean_baselines_lineart004.png`
- `results/fixed_output_metrics_base_clean_lineart004_compare.csv`

### Moredupes Epoch020 Check

The `shape1_std15_clean_split_moredupes_bce` run is large:

- train list: `dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt`
- rows: 6,598
- eval list: `dataset/pairs_480/eval_fixed_clean_lineart004.txt`
- full run: 200 epochs, too slow for a first signal

Added and launched an epoch020 watcher:

- script: `experiments/watch_moredupes_epoch020_eval.sh`
- unit: `lineart-watch-moredupes-epoch020-eval.service`

The watcher waits for:

- `checkpoints/shape1_std15_clean_split_moredupes_bce/epoch020.pth`

Then it stops:

- `lineart-exp1-moredupes-bce.service`

And writes:

- `results/shape1_std15_clean_split_moredupes_bce_epoch020/`
- `results/fixed_output_metrics_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.csv`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `logs/shape1_std15_clean_split_moredupes_bce_epoch020.done`

### Results Cleanup

Moved old root one-off files from `results/` into:

- `results/archive/legacy_root_oneoffs/`

Moved old leaky/pre-clean-eval model outputs, comparisons, and metrics into:

- `results/archive/leaky_pre_clean_eval/`

Updated:

- `config/results_manifest.json`
- `results/CURRENT.md`
- `results/README.md`

### Extraction Rules Restored

Added `doc/EXTRACTION_RULES.md` so raw manuscript extraction rules stay visible
after archiving leak-era logs.

Key preserved rules:

- diagnose alignment before same-coordinate tiling
- dry-run plus CSV/QC before `--save`
- use region matching when correspondence is uncertain
- run crop-scale diagnostics when source scale/context changes
- audit train/eval leakage and missing files before training

### Moredupes Epoch020 Result

The scratch `shape1_std15_clean_split_moredupes_bce` run was stopped at
epoch020 and evaluated on clean lineart004-only samples.

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_std15_clean_split_moredupes_bce_epoch020` | 0.1853 | 8.696 | 0.301 | 0.3857 | 0.1257 |

Conclusion: scratch training on the large duplicated list is not a useful
overnight direction. It under-produces ink and is worse than the clean baseline.

Artifacts:

- `results/compare_exp1_moredupes_epoch020_clean_lineart004.png`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_clean_lineart004_compare.csv`

## 2026-07-19

### Mild Duplicate Low-LR Fine-Tune

Started one conservative overnight run:

- experiment: `shape1_clean_split_bce_milddup800_ft10_lr1e5`
- unit: `lineart-milddup800-ft.service`
- log: `logs/train_shape1_clean_split_bce_milddup800_ft10_lr1e5.service.log`
- resume: `checkpoints/shape1_clean_split_bce/best.pth`
- train list: `dataset/pairs_480/valid_train_milddup800_clean.txt`
- eval list: `dataset/pairs_480/eval_clean_lineart004_8.txt`
- epochs: 10
- lr: `1e-5`
- loss weights: BCE only, `pos_weight=5.0`

The train list is generated from:

- seed: `dataset/pairs_480/valid_train_base_clean_unique.txt` (469 rows)
- pool: `dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt` (6,598 rows)
- target rows: 800
- max per canonical tile: 2
- exact duplicates: allowed intentionally for this experiment

Dry-run/build verification:

- rows: 800
- canonicals: 633
- duplicate canonicals: 167

Committed and pushed:

- `2eb995c Add mild duplicate fine-tune experiment`

Expected outputs when complete:

- `logs/shape1_clean_split_bce_milddup800_ft10_lr1e5.done`
- `results/fixed_output_metrics_shape1_clean_split_bce_milddup800_ft10_lr1e5_clean_lineart004.csv`
- `results/fixed_output_metrics_shape1_clean_split_bce_milddup800_ft10_lr1e5_compare.csv`
- `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5_clean_lineart004.png`
- `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5_vs_clean_split_bce.png`

### Next Actions

1. Check whether `lineart-milddup800-ft.service` completed.
2. Inspect the compare montage against `shape1_clean_split_bce_lineart004`.
3. Prefer the milddup fine-tune only if it improves F1/chamfer without obvious
   visual degradation or over-thick ink.
4. If it fails, keep `shape1_clean_split_bce_lineart004` as the current clean
   baseline and pivot to better extraction/data rules rather than larger
   duplicated training.

### Milddup800 Result And Model Survey

The mild duplicate fine-tune completed.

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |

Interpretation: milddup800 is a small numeric improvement and does not show
obvious over-thick ink, but the visual failure mode remains close to the clean
BCE baseline. Treat it as a mild clean-baseline variant, not a qualitative
breakthrough.

Current modeling decision:

- Do not treat deterministic xy-coordinate agreement as the goal.
- Rebuild model exploration around line-art feel, line selectivity, and rough
  atmosphere preservation.
- BCE and U-Net are not rejected outright; control their influence so they do
  not over-enforce pixel matching or rough-noise copying.
- GAN-style objectives are now in scope as candidates for line-art realism.
- Continue to use F1/chamfer as reference metrics, but make montage inspection
  central for adoption decisions.

Added survey implementation:

- `lineart/model_zoo.py`
  - `ScaledSkipUNet`
  - `ResnetGenerator`
  - `PatchDiscriminator`
  - generator checkpoint loading helpers
- `scripts/train_i2i_survey.py`
  - common short-run training entry for reconstruction and optional PatchGAN
  - supports `unet`, `unet_skip25`, `unet_skip50`, `resnet`
  - default workers set to `0` because sandboxed multiprocessing hit
    `PermissionError: Operation not permitted`
- `scripts/inference_i2i.py`
  - inference for survey checkpoint format
- `experiments/run_model_survey.sh`
  - trains and evaluates:
    - `unet_skip50`
    - `unet_gan`
    - `resnet`
    - `resnet_gan`
  - builds one shared montage with clean BCE baseline and milddup800

## 2026-07-20

### Worktree Operation Policy

Recorded the multi-worktree operating policy in `doc/worktree_policy.md`.

Current branch roles:

- main worktree `/home/sh1/deepl/lineart` on `MoE`: integration, shared docs,
  and result summaries
- `/home/sh1/deepl/lineart-halo-loss` on `halo-loss`: halo/faint-gray focused
  loss experiments
- `/home/sh1/deepl/lineart-router-moe` on `router-moe`: router/MoE feature
  engineering and routed inference

Schedule policy:

- keep weekday daytime work to implementation, analysis, short smoke tests, and
  documentation
- treat router/MoE as night-batch work because it can run long
- schedule heavy router/MoE runs primarily for Thursday, Friday, and Saturday
  nights
- send only the final completion notification for sleep-time chained runs

### Halo Loss Result And Current Priority

Completed `halo_loss_e2` from the `halo-loss` worktree.

Artifacts:

- `results/compare_halo_loss_e2.png`
- `results/fixed_output_metrics_halo_loss_e2_compare.csv`
- `results/halo_metrics_halo_loss_e2_compare.csv`
- `logs/halo_loss_e2.done`

Key results:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `lucy_thin` | 0.4095 | 4.536 | 1.252 |
| `agreement_halo_e2_high_agreement` | 0.4363 | 4.271 | 2.128 |
| `halo_loss_e2_mild_halo06` | 0.3695 | 4.927 | 0.814 |
| `halo_loss_e2_mild_halo10_faint04` | 0.3458 | 5.144 | 0.679 |
| `halo_loss_e2_high_halo10_faint04` | 0.4258 | 4.402 | 1.657 |

Interpretation:

- the current halo-band penalty reduces overall ink, but it also suppresses
  useful nearby line structure
- full-list halo-loss variants become too pale and lose recall
- high-agreement plus halo/faint loss is the best of the new variants, but it
  still does not explain or solve halo cleanly
- halo-specific metrics suggest some ink reduction is being traded for faint
  gray residue rather than true halo removal

Current priority:

- do not optimize directly for "remove halo" yet
- first identify what creates halo: atari generation, aux conditioning,
  generator loss, target mismatch, model architecture, binarization/deblur, or
  data alignment
- use controlled diagnostics before deciding the next corrective loss or model
  change

### Atari Halo Diagnosis

Added a no-training diagnostic to separate atari-origin halo from final-stage
halo amplification.

Implementation:

- `tools/evaluation/build_atari_halo_diagnostic_set.py`
  - builds a shuffled diagnostic list from `valid_train_milddup800_clean`
  - excludes the fixed lineart004 eval list
  - balances agreement score and rough edge density buckets
- `tools/evaluation/evaluate_halo_outputs.py`
  - now accepts `LABEL=DIR` model arguments while preserving old model-name
    behavior
- `tools/evaluation/compare_halo_amplification.py`
  - compares atari and final halo metrics sample-by-sample
  - writes amplification ratios and atari/final correlations
- `experiments/run_atari_halo_diagnosis.sh`
  - runs raw, dog, lucy_mild, and lucy_thin atari/final pairs on a shuffled
    diagnostic set

Completed:

- smoke: `atari_halo_diag_smoke6`
- main run: `atari_halo_diag_shuffle60`

Artifacts:

- `dataset/pairs_480/atari_halo_diag_shuffle60.txt`
- `results/atari_halo_diag_shuffle60_samples.csv`
- `results/halo_metrics_atari_halo_diag_shuffle60.csv`
- `results/halo_amplification_atari_halo_diag_shuffle60.csv`
- `results/halo_amplification_atari_halo_diag_shuffle60_summary.csv`
- `results/compare_atari_halo_diag_shuffle60.png`
- `logs/atari_halo_diag_shuffle60.done`

Summary:

| pair | samples | atari_halo_ink | final_halo_ink | amplification | corr |
|---|---:|---:|---:|---:|---:|
| `raw` | 60 | 0.2235 | 0.2235 | 0.948 | 0.928 |
| `dog` | 60 | 0.0930 | 0.2221 | 2.636 | 0.485 |
| `lucy_mild` | 60 | 0.0799 | 0.2147 | 2.906 | 0.492 |
| `lucy_thin` | 60 | 0.0632 | 0.1702 | 3.011 | 0.530 |

Interpretation:

- raw atari halo strongly predicts final halo, so raw atari defects are a real
  source
- dog/lucy postprocessing greatly reduces halo at the atari stage, but the
  final model regenerates halo-like ink around lines
- this points to aux-conditioned final-stage amplification, not only atari
  generation
- next diagnostic should isolate aux conditioning strength and reconstruction
  loss behavior, preferably with batched inference because one-image process
  startup is slow

### Aux Strength vs Loss Halo Diagnosis

Added and ran a split diagnostic for the two likely final-stage halo causes:

- aux conditioning strength: fixed `lucy_thin_aux_msgan` checkpoint, varied only
  the aux image passed at inference
- reconstruction/loss behavior: fixed `lucy_thin` aux, varied existing
  checkpoints trained with different loss/model settings

Implementation:

- `scripts/inference_i2i_batch.py`
  - batch inference wrapper that loads a checkpoint once per condition
- `tools/evaluation/transform_aux_strength.py`
  - deterministic aux variants:
    - `weak75`
    - `weak50`
    - `hard20`
    - `hard35`
    - `softcut20`
    - `blur`
    - `open`
- `experiments/run_aux_vs_loss_halo_diagnosis.sh`
  - runs aux-strength and loss/model comparisons on
    `dataset/pairs_480/atari_halo_diag_shuffle60.txt`

Artifacts:

- `results/halo_metrics_aux_loss_halo_diag_shuffle60_aux_strength.csv`
- `results/compare_aux_loss_halo_diag_shuffle60_aux_strength.png`
- `results/halo_metrics_aux_loss_halo_diag_shuffle60_loss_compare.csv`
- `results/compare_aux_loss_halo_diag_shuffle60_loss_compare.png`
- `logs/aux_loss_halo_diag_shuffle60.done`

Aux-strength summary:

| model | core | halo_ink | halo_faint | far_ink | far_faint |
|---|---:|---:|---:|---:|---:|
| `base_aux` | 0.0693 | 0.0632 | 0.2972 | 0.0206 | 0.1236 |
| `aux_identity` | 0.1963 | 0.1702 | 0.1815 | 0.0886 | 0.0917 |
| `aux_weak75` | 0.1667 | 0.1444 | 0.2450 | 0.0788 | 0.1085 |
| `aux_weak50` | 0.1323 | 0.1146 | 0.3264 | 0.0652 | 0.1391 |
| `aux_hard20` | 0.1302 | 0.1191 | 0.0000 | 0.0373 | 0.0000 |
| `aux_hard35` | 0.0466 | 0.0481 | 0.0000 | 0.0197 | 0.0000 |
| `aux_softcut20` | 0.0233 | 0.0212 | 0.0940 | 0.0115 | 0.0245 |
| `aux_blur` | 0.2675 | 0.2463 | 0.4126 | 0.1510 | 0.3416 |
| `aux_open` | 0.1042 | 0.0855 | 0.2685 | 0.0455 | 0.0978 |

Loss/model summary:

| model | core | halo_ink | halo_faint | far_ink | far_faint |
|---|---:|---:|---:|---:|---:|
| `base_aux` | 0.0693 | 0.0632 | 0.2972 | 0.0206 | 0.1236 |
| `loss_cleanup_msgan` | 0.0543 | 0.0465 | 0.3549 | 0.0190 | 0.1543 |
| `loss_bin12` | 0.2910 | 0.2779 | 0.5863 | 0.2238 | 0.8838 |
| `loss_bin20` | 0.2808 | 0.2691 | 0.6269 | 0.2098 | 0.9104 |
| `loss_struct08` | 0.0730 | 0.0637 | 0.3884 | 0.0260 | 0.1803 |
| `loss_struct08_msgan` | 0.0550 | 0.0469 | 0.3585 | 0.0191 | 0.1556 |
| `loss_width06` | 0.0724 | 0.0636 | 0.3884 | 0.0260 | 0.1803 |
| `loss_halo_high` | 0.1771 | 0.1479 | 0.2386 | 0.0654 | 0.1383 |

Interpretation:

- aux strength is a real control point; blurring the aux strongly worsens halo
  and far-background gray
- continuous or hard removal of faint aux ink reduces halo metrics, but
  `softcut20` and `hard35` also suppress core line strength, so they are
  diagnostics rather than immediate adoption choices
- bin12/bin20 style U-Net experts regenerate a large gray/halo field under the
  fixed `lucy_thin` aux condition
- cleanup+FM/structure cleanup variants keep halo ink lower, but still leave
  faint gray residue
- current conclusion: halo regeneration is a coupled effect of faint aux
  conditioning and model/loss behavior; U-Net/bin experts are especially risky
  for gray-field amplification

### Split Halo Into Background Haze And Line-Near Uncertainty

Updated halo evaluation so the previous single halo category is split into:

- `background_haze`
  - faint/gray ink in the far background away from GT lines
  - treated as unwanted haze/noise
- `line_near_uncertainty`
  - gray or excess ink in the band near GT lines
  - can be a line candidate, but becomes visible halo when amplified

Implementation:

- `tools/evaluation/evaluate_halo_outputs.py`
  - keeps old columns for compatibility
  - adds:
    - `line_near_uncertainty_ink`
    - `line_near_uncertainty_faint_ratio`
    - `line_near_strong_ratio`
    - `line_near_to_core`
    - `background_haze_ink_mean`
    - `background_haze_faint_ink_mean`
    - `background_haze_faint_ratio`
    - `background_haze_area_ratio`
    - `background_ink_area_ratio`
    - `background_haze_to_core`

New evaluation artifacts:

- `results/haze_uncertainty_metrics_atari_halo_diag_shuffle60.csv`
- `results/haze_uncertainty_metrics_aux_loss_halo_diag_shuffle60_aux_strength.csv`
- `results/haze_uncertainty_metrics_aux_loss_halo_diag_shuffle60_loss_compare.csv`

Atari/final split summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `raw_atari` | 0.2235 | 0.6769 | 0.1330 | 0.6478 |
| `lucy_thin_atari` | 0.0632 | 0.2972 | 0.0206 | 0.0837 |
| `cleanup_msgan` | 0.2235 | 0.7166 | 0.1605 | 0.7057 |
| `lucy_thin_final` | 0.1702 | 0.1815 | 0.0886 | 0.0722 |

Aux-strength split summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `base_aux` | 0.0632 | 0.2972 | 0.0206 | 0.0837 |
| `aux_identity` | 0.1702 | 0.1815 | 0.0886 | 0.0722 |
| `aux_softcut20` | 0.0212 | 0.0940 | 0.0115 | 0.0197 |
| `aux_blur` | 0.2463 | 0.4126 | 0.1510 | 0.2766 |

Loss/model split summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `loss_cleanup_msgan` | 0.0465 | 0.3549 | 0.0190 | 0.1017 |
| `loss_bin12` | 0.2779 | 0.5863 | 0.2238 | 0.6870 |
| `loss_bin20` | 0.2691 | 0.6269 | 0.2098 | 0.7069 |
| `loss_halo_high` | 0.1479 | 0.2386 | 0.0654 | 0.0900 |

Interpretation:

- `raw_atari` and `cleanup_msgan` contain both large background haze and
  line-near uncertainty
- `lucy_thin_atari` strongly reduces background haze, but `lucy_thin_final`
  regenerates line-near ink around GT lines
- `aux_blur` proves that smeared aux information creates both haze and
  line-near uncertainty
- `bin12/bin20` regenerate a large gray field under fixed `lucy_thin` aux,
  so they are unsuitable as-is for halo-sensitive routing

Next modeling rule:

- penalize or filter `background_haze` directly
- do not simply erase `line_near_uncertainty`; convert it toward a line core
  or route it to a model that can decide line ownership

### Rough Input Cleansing First Pass

Added a first-pass rough preprocessing diagnostic before rough-to-lineart/atari
generation.

Goal:

- test whether cleaning the rough input before atari generation reduces:
  - `background_haze`
  - `line_near_uncertainty`
- keep this as data cleansing / rough normalization, not yet a learned model

Implementation:

- `tools/preprocess/clean_rough_input.py`
  - `background`
    - estimates low-frequency background and soft-removes faint ink
  - `line`
    - uses ridge/DoG-like emphasis to suppress broad repeated sketch haze and
      keep line cores
  - `background_line`
    - background cleanup followed by line cleanup
  - `line_background`
    - line cleanup followed by background cleanup
- `experiments/run_rough_cleanup_diagnosis.sh`
  - cleans the shuffled 60-tile rough set
  - materializes atari using the existing ResNet-GAN atari checkpoint
  - evaluates split haze/uncertainty metrics
  - builds comparison montage

Artifacts:

- `results/haze_uncertainty_metrics_rough_cleanup_diag_shuffle60.csv`
- `results/compare_rough_cleanup_diag_shuffle60.png`
- `logs/rough_cleanup_diag_shuffle60.done`

Summary:

| model | core | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|---:|
| `raw_atari` | 0.2458 | 0.2235 | 0.6769 | 0.1330 | 0.6478 |
| `rough_background` | 0.0364 | 0.0261 | 0.1069 | 0.0112 | 0.0410 |
| `atari_background` | 0.2148 | 0.1973 | 0.7194 | 0.1242 | 0.6654 |
| `rough_line` | 0.0715 | 0.0540 | 0.2260 | 0.0254 | 0.1004 |
| `atari_line` | 0.2353 | 0.2153 | 0.6470 | 0.1230 | 0.6132 |
| `rough_background_line` | 0.0478 | 0.0342 | 0.1180 | 0.0140 | 0.0444 |
| `atari_background_line` | 0.2127 | 0.1956 | 0.7015 | 0.1201 | 0.6469 |
| `rough_line_background` | 0.0396 | 0.0288 | 0.0847 | 0.0123 | 0.0368 |
| `atari_line_background` | 0.2184 | 0.2009 | 0.6830 | 0.1204 | 0.6294 |

Interpretation:

- rough input cleansing itself works: both background haze and line-near faint
  are reduced heavily in the cleaned rough images
- the existing ResNet-GAN atari generator largely regenerates the gray/haze
  field after cleaned rough input
- this suggests the current atari generator learned a gray sketch-like output
  distribution, not only a pass-through of input dirt
- next useful step is not only stronger rough cleaning, but either:
  - retrain/fine-tune the atari generator on cleaned rough inputs and cleaner
    targets
  - or bypass the gray atari generator with a non-learned/softcut line-confidence
    aux for the next refiner

### Rough Cleanup Sweep Before Cleaned-Rough Atari Training

Expanded rough input cleansing before committing to cleaned-rough atari
fine-tuning.

Implementation:

- updated `tools/preprocess/clean_rough_input.py`
  - added `identity`
  - added background variants:
    - `background_mild`
    - `background`
    - `background_strong`
  - added line cleanup variants:
    - `line_mild`
    - `line`
    - `line_strong`
  - added chained variants:
    - `line_background_mild`
    - `line_background`
    - `line_background_strong`
  - added:
    - `softcut_only`
    - `edge_preserve`
- added `experiments/run_rough_cleanup_sweep.sh`
  - builds a balanced shuffled sample list
  - runs all rough cleanup variants
  - evaluates split haze/uncertainty metrics
  - builds montage

Completed:

- smoke: `rough_cleanup_sweep_smoke20b`
- main sweep: `rough_cleanup_sweep200`

Artifacts:

- `dataset/pairs_480/rough_cleanup_sweep200.txt`
- `results/rough_cleanup_sweep200_samples.csv`
- `results/haze_uncertainty_metrics_rough_cleanup_sweep200.csv`
- `results/compare_rough_cleanup_sweep200.png`
- `logs/rough_cleanup_sweep200.done`

Summary:

| model | core | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|---:|
| `identity` | 0.0606 | 0.0536 | 0.2554 | 0.0311 | 0.1267 |
| `background_mild` | 0.0300 | 0.0257 | 0.1227 | 0.0127 | 0.0507 |
| `background` | 0.0276 | 0.0235 | 0.1083 | 0.0116 | 0.0445 |
| `background_strong` | 0.0232 | 0.0197 | 0.0892 | 0.0096 | 0.0365 |
| `line_mild` | 0.0506 | 0.0444 | 0.1967 | 0.0228 | 0.0886 |
| `line` | 0.0571 | 0.0506 | 0.2268 | 0.0263 | 0.1052 |
| `line_strong` | 0.0647 | 0.0580 | 0.2575 | 0.0316 | 0.1254 |
| `line_background_mild` | 0.0286 | 0.0251 | 0.1007 | 0.0123 | 0.0447 |
| `line_background` | 0.0294 | 0.0259 | 0.0871 | 0.0128 | 0.0395 |
| `line_background_strong` | 0.0292 | 0.0258 | 0.0721 | 0.0126 | 0.0330 |
| `softcut_only` | 0.0481 | 0.0410 | 0.1456 | 0.0205 | 0.0593 |
| `edge_preserve` | 0.0316 | 0.0272 | 0.1104 | 0.0131 | 0.0457 |

Interpretation:

- all background/chained variants reduce `background_haze` strongly
- `line_background_strong` is best numerically for faint suppression, but likely
  risks deleting useful weak rough lines
- `line_background_mild` and `edge_preserve` are safer first candidates for
  cleaned-rough atari fine-tuning
- `line_background` is a reasonable more aggressive candidate if montage
  inspection accepts the loss of faint rough context
- avoid selecting solely by `halo_to_core`; it becomes unstable when core ink
  is heavily reduced

Next candidate set for atari fine-tune:

- safe: `line_background_mild`
- balanced: `edge_preserve`
- aggressive: `line_background`

### Cleaned-Rough Model Survey Launch

Decision:

- treat rough cleansing mode as a likely data-dependent choice
- preserve the current sweep finding rather than choosing a single universal
  cleanser too early
- next compare cleaned rough input distributions by training both:
  - direct `unet`
  - direct `resnet_gan`

Added:

- `experiments/run_cleaned_rough_model_survey.sh`
  - materializes train/eval cleaned rough for:
    - `line_background_mild`
    - `edge_preserve`
    - `line_background`
  - trains each mode with:
    - `unet`
    - `resnet_gan`
  - uses `--no-autocontrast` so cleanup is not undone by contrast stretching
  - evaluates fixed lineart004 metrics and split haze/uncertainty metrics
  - builds one montage
  - sends completion notification

Launched:

- unit: `lineart-cleaned-rough-model-survey-e2.service`
- tag: `cleaned_rough_model_survey_e2`
- epochs: 2

Expected artifacts:

- `logs/cleaned_rough_model_survey_e2.log`
- `logs/cleaned_rough_model_survey_e2.systemd.log`
- `logs/cleaned_rough_model_survey_e2.done`
- `results/fixed_output_metrics_cleaned_rough_model_survey_e2_compare.csv`
- `results/haze_uncertainty_metrics_cleaned_rough_model_survey_e2_compare.csv`
- `results/compare_cleaned_rough_model_survey_e2.png`

Result:

- completed successfully
- completion marker: `logs/cleaned_rough_model_survey_e2.done`
- montage: `results/compare_cleaned_rough_model_survey_e2.png`
- fixed metrics:
  `results/fixed_output_metrics_cleaned_rough_model_survey_e2_compare.csv`
- split haze/uncertainty metrics:
  `results/haze_uncertainty_metrics_cleaned_rough_model_survey_e2_compare.csv`

Fixed metrics summary:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `raw_resnet_gan` | 0.3062 | 7.343 | 0.842 |
| `line_background_mild_unet` | 0.3790 | 5.855 | 1.759 |
| `line_background_mild_resnet_gan` | 0.2370 | 8.463 | 0.394 |
| `edge_preserve_unet` | 0.3854 | 5.958 | 1.768 |
| `edge_preserve_resnet_gan` | 0.2568 | 7.857 | 0.536 |
| `line_background_unet` | 0.3385 | 6.931 | 1.561 |
| `line_background_resnet_gan` | 0.2258 | 8.469 | 0.439 |

Split haze/uncertainty summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `raw_resnet_gan` | 0.2956 | 0.7448 | 0.2037 | 0.6140 |
| `line_background_mild_unet` | 0.3567 | 0.4771 | 0.2141 | 0.5541 |
| `line_background_mild_resnet_gan` | 0.1646 | 0.8581 | 0.0741 | 0.5424 |
| `edge_preserve_unet` | 0.3783 | 0.3989 | 0.2158 | 0.5378 |
| `edge_preserve_resnet_gan` | 0.1743 | 0.8332 | 0.0717 | 0.4875 |
| `line_background_unet` | 0.3549 | 0.4800 | 0.2179 | 0.5449 |
| `line_background_resnet_gan` | 0.1664 | 0.8454 | 0.0724 | 0.4887 |

Interpretation:

- cleaned-rough `unet` is much better numerically than the old raw
  `resnet_gan` baseline
- `edge_preserve_unet` has the best F1 among this run
- `line_background_mild_unet` is close and has slightly better chamfer
- cleaned-rough `resnet_gan` under-produces ink and loses too much line recall
  after only 2 epochs
- ResNet GAN reduces background haze compared with raw resnet, but its line
  output is too sparse; this is not yet a good atari/refiner source
- U-Net benefits more immediately from cleaned rough input, but still carries
  gray-field/faint residue

Next direction:

- keep `edge_preserve_unet` and `line_background_mild_unet` as immediate
  cleaned-rough candidates
- if using ResNet GAN, try warm-start/fine-tune from the existing raw
  `resnet_gan` checkpoint instead of scratch 2-epoch training
- evaluate whether U-Net output should become a new structured aux source for a
  downstream cleanup/refiner

### Cleaned-Rough Warm-Start Survey Launch

Added:

- `experiments/run_cleaned_rough_warmstart_survey.sh`

Purpose:

- continue U-Net and ResNet GAN in parallel
- test whether longer U-Net training improves cleaned-rough results
- test whether ResNet GAN can recover recall when fine-tuned from an existing
  raw ResNet GAN checkpoint instead of scratch training

Conditions:

- cleanup modes:
  - `edge_preserve`
  - `line_background_mild`
- U-Net:
  - scratch
  - 5 epochs
- ResNet GAN:
  - warm-start from
    `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`
  - 5 epochs
  - low LR (`5e-5`, discriminator `1e-5`)
- all cleaned-rough runs use `--no-autocontrast`
- completion notification enabled

Launched:

- unit: `lineart-cleaned-rough-warmstart-e5.service`
- tag: `cleaned_rough_warmstart_e5`

Expected artifacts:

- `logs/cleaned_rough_warmstart_e5.log`
- `logs/cleaned_rough_warmstart_e5.systemd.log`
- `logs/cleaned_rough_warmstart_e5.done`
- `results/compare_cleaned_rough_warmstart_e5.png`
- `results/fixed_output_metrics_cleaned_rough_warmstart_e5_compare.csv`
- `results/haze_uncertainty_metrics_cleaned_rough_warmstart_e5_compare.csv`

Result:

- completed successfully
- montage: `results/compare_cleaned_rough_warmstart_e5.png`
- fixed metrics:
  `results/fixed_output_metrics_cleaned_rough_warmstart_e5_compare.csv`
- split haze/uncertainty metrics:
  `results/haze_uncertainty_metrics_cleaned_rough_warmstart_e5_compare.csv`

Fixed metrics summary:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `raw_resnet_gan` | 0.3998 | 5.476 | 1.732 |
| `cleaned_edge_unet_e2` | 0.3854 | 5.958 | 1.768 |
| `cleaned_mild_unet_e2` | 0.3790 | 5.855 | 1.759 |
| `edge_preserve_unet_e5` | 0.2500 | 8.375 | 0.551 |
| `edge_preserve_resnet_gan_warm_e5` | 0.2806 | 7.486 | 0.464 |
| `line_background_mild_unet_e5` | 0.2394 | 8.342 | 0.484 |
| `line_background_mild_resnet_gan_warm_e5` | 0.2811 | 7.387 | 0.509 |

Split haze/uncertainty summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `raw_resnet_gan` | 0.2894 | 0.5585 | 0.1053 | 0.3048 |
| `cleaned_edge_unet_e2` | 0.3783 | 0.3989 | 0.2158 | 0.5378 |
| `cleaned_mild_unet_e2` | 0.3567 | 0.4771 | 0.2141 | 0.5541 |
| `edge_preserve_unet_e5` | 0.3056 | 0.6504 | 0.1777 | 0.6025 |
| `edge_preserve_resnet_gan_warm_e5` | 0.2139 | 0.6998 | 0.0755 | 0.2982 |
| `line_background_mild_unet_e5` | 0.3090 | 0.5993 | 0.1832 | 0.5895 |
| `line_background_mild_resnet_gan_warm_e5` | 0.2204 | 0.7200 | 0.0828 | 0.3272 |

Interpretation:

- longer scratch U-Net training on cleaned rough degraded sharply compared with
  the 2-epoch cleaned-rough U-Net runs
- warm-start ResNet GAN recovered somewhat versus scratch cleaned-rough ResNet
  GAN, but still under-produced ink and did not reach raw ResNet GAN or
  cleaned-rough U-Net e2
- E5 models look cleaner/whiter in places but lose too much recall
- current best cleaned-rough signal remains the e2 U-Net family, especially
  `edge_preserve_unet` and `line_background_mild_unet`
- do not assume "more epochs" is better for the current cleaned-rough setup;
  early stopping or lower LR is likely needed

### Raw+Clean Mixed And 2ch U-Net Survey Launch

Decision:

- include raw/hazy rough data as a possible way to suppress white-collapse
- keep cleaned rough as haze-control signal
- test both mixed 1ch training and raw+cleaned 2ch training

Added:

- `tools/preprocess/build_mixed_rough_dataset.py`
  - duplicates each training sample as raw and cleaned rough with the same line
    target
- `experiments/run_raw_clean_mixed_2ch_survey.sh`

Conditions:

- cleanup modes:
  - `edge_preserve`
  - `line_background_mild`
- mixed 1ch:
  - train list duplicated to 1,600 rows
  - raw rough and cleaned rough both map to the same GT line
  - inference uses cleaned rough
- 2ch:
  - channel 1: raw rough
  - channel 2: cleaned rough
  - inference uses raw + cleaned
- U-Net only for this first pass
- epochs: 3
- LR: `7e-5`
- `pos_weight=6.0`
- `ink_weight=0.14`
- `binary_weight=0.06`
- no autocontrast

Launched:

- unit: `lineart-raw-clean-mixed-2ch-e3.service`
- tag: `raw_clean_mixed_2ch_e3`

Expected artifacts:

- `results/compare_raw_clean_mixed_2ch_e3.png`
- `results/fixed_output_metrics_raw_clean_mixed_2ch_e3_compare.csv`
- `results/haze_uncertainty_metrics_raw_clean_mixed_2ch_e3_compare.csv`
- `logs/raw_clean_mixed_2ch_e3.done`

Verification already run:

```bash
./venv/bin/python -m py_compile \
  lineart/model_zoo.py scripts/train_i2i_survey.py scripts/inference_i2i.py
bash -n experiments/run_model_survey.sh
```

One foreground survey attempt was stopped because sandboxed Python could not
initialize CUDA and DataLoader multiprocessing failed. The runner was changed
to `--workers 0`, then relaunched through user systemd with GPU-capable
execution.

Currently running:

- unit: `lineart-model-survey-e1.service`
- command: `EPOCHS=1 experiments/run_model_survey.sh model_survey_e1`
- systemd log: `logs/model_survey_e1.systemd.log`
- script log: `logs/model_survey_e1.log`

Expected outputs:

- `logs/model_survey_e1.done`
- `results/compare_model_survey_e1.png`
- `results/fixed_output_metrics_model_survey_e1_compare.csv`
- `checkpoints/model_survey_e1_unet_skip50/best.pth`
- `checkpoints/model_survey_e1_unet_gan/best.pth`
- `checkpoints/model_survey_e1_resnet/best.pth`
- `checkpoints/model_survey_e1_resnet_gan/best.pth`

Useful check commands:

```bash
systemctl --user status lineart-model-survey-e1.service --no-pager
tail -f logs/model_survey_e1.systemd.log
```

Next actions:

1. Wait for `logs/model_survey_e1.done`.
2. Inspect `results/compare_model_survey_e1.png`.
3. Use the survey only for output-tendency triage, not final quality ranking;
   `EPOCHS=1` is intentionally short.
4. Pick at most one or two promising families for longer controlled runs.

### Notification Operating Rule

For any training-plus-inference run of 2 epochs or more, wire completion
notification through `experiments/send_autoloop_notification.sh`.

Minimum rule:

- send a notification when the run writes its done marker
- send a failure/stopped notification if the service exits without the done
  marker
- use `experiments/watch_service_done_notify.sh` for already-running systemd
  jobs whose runner does not yet notify directly

Current resnet sharp follow-up notification watcher:

- training unit: `lineart-model-resnet-sharp-e5.service`
- watcher unit: `lineart-model-resnet-sharp-e5-notify.service`
- done marker: `logs/model_resnet_sharp_e5.done`
- montage: `results/compare_model_resnet_sharp_e5.png`
- metrics: `results/fixed_output_metrics_model_resnet_sharp_e5_compare.csv`

### Resnet GAN Refinement Direction

The current visual judgment is that `resnet_gan_advsharp` and its binary
fine-tune preserve more drawing-like structure than the U-Net group. U-Net
variants remain clearer black/white outputs, but many tiles look too similar
to the BCE baseline, and adding GAN to U-Net did not materially change the
failure mode.

Current direction:

- use ResNet+PatchGAN as an "atari" generator for drawing-like structure
- first map its limit with a short binary/ink sweep
- do not chase metrics alone; montage inspection decides whether improved F1 is
  just black growth or useful line selectivity
- after the limit is understood, build a line-refiner stage that can take
  rough plus ResNet-GAN output and target cleaner black/white line art

Completed useful checkpoint:

- `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`
- montage: `results/compare_model_resnet_binft_e3.png`
- metrics: `results/fixed_output_metrics_model_resnet_binft_e3_compare.csv`

`model_resnet_binft_e3_resnet_gan_advsharp_binft` metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `model_resnet_sharp_e5_resnet_gan_advsharp` | 0.3524 | 7.094 | 1.301 | 0.3511 | 0.3583 |
| `model_resnet_binft_e3_resnet_gan_advsharp_binft` | 0.3998 | 5.476 | 1.732 | 0.3337 | 0.5020 |

Next active sweep:

- tag: `model_resnet_binsweep_e2`
- unit: `lineart-model-resnet-binsweep-e2.service`
- epochs: 2
- base checkpoint: `checkpoints/model_resnet_sharp_e5_resnet_gan_advsharp/best.pth`
- candidates:
  - `resnet_gan_advsharp_bin04`
  - `resnet_gan_advsharp_bin08_ink10`
  - `resnet_gan_advsharp_bin12_ink12`
- expected montage: `results/compare_model_resnet_binsweep_e2.png`
- expected metrics: `results/fixed_output_metrics_model_resnet_binsweep_e2_compare.csv`
- expected done marker: `logs/model_resnet_binsweep_e2.done`
- notification: handled by `experiments/run_model_followup.sh`

### Resnet Binary/Ink Sweep Result

The 2-epoch binary/ink sweep completed.

Artifacts:

- done marker: `logs/model_resnet_binsweep_e2.done`
- montage: `results/compare_model_resnet_binsweep_e2.png`
- metrics: `results/fixed_output_metrics_model_resnet_binsweep_e2_compare.csv`
- checkpoints:
  - `checkpoints/model_resnet_binsweep_e2_resnet_gan_advsharp_bin04/best.pth`
  - `checkpoints/model_resnet_binsweep_e2_resnet_gan_advsharp_bin08_ink10/best.pth`
  - `checkpoints/model_resnet_binsweep_e2_resnet_gan_advsharp_bin12_ink12/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `model_resnet_binsweep_e2_resnet_gan_advsharp_bin04` | 0.3954 | 5.738 | 1.841 | 0.3346 | 0.4871 |
| `model_resnet_binsweep_e2_resnet_gan_advsharp_bin08_ink10` | 0.3925 | 5.819 | 1.752 | 0.3370 | 0.4734 |
| `model_resnet_binsweep_e2_resnet_gan_advsharp_bin12_ink12` | 0.3887 | 5.895 | 1.669 | 0.3389 | 0.4593 |

Interpretation:

- increasing ink/binary pressure reduces ink ratio and recall, with only small
  visual differences across the three sweep variants
- `bin04` is the most aggressive and highest F1 in this sweep, but it also has
  the highest ink ratio
- `bin12_ink12` is most conservative but gives up recall/F1
- none of the sweep variants is a clear improvement over the earlier
  `model_resnet_binft_e3_resnet_gan_advsharp_binft` checkpoint, which remains
  the current useful ResNet-GAN atari checkpoint

Current decision:

- stop broad ResNet-GAN parameter chasing for now
- treat ResNet-GAN as an atari/structure generator, not the final line model
- next build a line-refiner stage that consumes rough plus ResNet-GAN output
  and targets cleaner black/white line art

### 2ch Line Refiner Survey Started

Started the first 2-channel line-refiner survey.

Goal:

- freeze the current useful ResNet-GAN atari generator
- materialize atari outputs for the train/eval lists
- train line-focused refiners with input channels:
  - channel 1: autocontrasted rough
  - channel 2: ResNet-GAN atari output
- target clean black/white line art directly

Implementation changes:

- `lineart/model_zoo.py`
  - `build_generator(model_name, in_channels=...)`
  - checkpoint loading now honors saved `in_channels`
- `scripts/train_i2i_survey.py`
  - added `--aux-dir`
  - saves `in_channels`
  - supports binary confidence loss
  - discriminator uses the original rough channel when GAN is enabled with
    multi-channel generator input
- `scripts/inference_i2i.py`
  - added `--aux-input`
- `scripts/materialize_i2i_aux.py`
  - writes fixed atari outputs for a file list
- `experiments/run_line_refiner_survey.sh`
  - materializes train/eval atari outputs
  - trains and evaluates selected 2ch refiner candidates
  - sends notification on completion

Active run:

- tag: `line_refiner_e2`
- unit: `lineart-line-refiner-e2.service`
- epochs: 2
- atari checkpoint:
  `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`
- train aux: `results/line_refiner_e2_atari_train`
- eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet`
  - `refiner_skip50`
- expected montage: `results/compare_line_refiner_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_e2_compare.csv`
- expected done marker: `logs/line_refiner_e2.done`

### 2ch Line Refiner Survey Result

The first 2-channel refiner survey completed.

Artifacts:

- done marker: `logs/line_refiner_e2.done`
- montage: `results/compare_line_refiner_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_e2_compare.csv`
- atari train cache: `results/line_refiner_e2_atari_train`
- atari eval cache: `results/line_refiner_e2_atari_eval`
- checkpoints:
  - `checkpoints/line_refiner_e2_refiner_unet/best.pth`
  - `checkpoints/line_refiner_e2_refiner_skip50/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_e2_refiner_unet` | 0.3776 | 6.497 | 1.757 | 0.3563 | 0.4068 |
| `line_refiner_e2_refiner_skip50` | 0.3645 | 6.828 | 1.664 | 0.3549 | 0.3790 |

Interpretation:

- the 2ch refiner path is technically working: train/eval atari generation,
  2-channel checkpointing, inference, montage, and metrics all completed
- `refiner_unet` is better than `refiner_skip50` numerically and visually
  retains more candidate structure
- both refiners still mostly inherit the ResNet-GAN atari's soft/mid-gray
  character; this is not yet a clean black/white line-art refiner
- `refiner_unet` improves F1 over milddup but increases ink ratio, so montage
  inspection remains essential

Next likely direction:

- continue with plain `refiner_unet`, not `skip50`, as the first refiner branch
- make the refiner more explicitly line-targeted by reducing atari-copy
  behavior:
  - stronger binary/confidence pressure
  - stronger or better-balanced ink control
  - possibly staged training: warm with BCE/shape, then tighten with
    binary/ink
- compare against the raw ResNet-GAN atari in every montage to ensure the
  refiner is truly refining rather than only copying it

### 2ch Refiner Tightening Sweep Started

Started a short tightening sweep focused only on the plain `refiner_unet`
branch.

Rationale:

- `refiner_unet` showed better line pickup than `refiner_skip50`
- `refiner_skip50` was more conservative but did not look qualitatively better
- the next question is whether stronger binary/ink pressure can reduce
  atari-copy and mid-gray output without destroying useful line candidates

Active run:

- tag: `line_refiner_tight_e2`
- unit: `lineart-line-refiner-tight-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet_bin12_ink14`
  - `refiner_unet_bin20_ink16`
  - `refiner_unet_bin28_ink18`
- expected montage: `results/compare_line_refiner_tight_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_tight_e2_compare.csv`
- expected done marker: `logs/line_refiner_tight_e2.done`
- notification: handled by `experiments/run_line_refiner_survey.sh`

### 2ch Refiner Tightening Sweep Result

The focused `refiner_unet` tightening sweep completed.

Artifacts:

- done marker: `logs/line_refiner_tight_e2.done`
- montage: `results/compare_line_refiner_tight_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_tight_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_tight_e2_refiner_unet_bin12_ink14/best.pth`
  - `checkpoints/line_refiner_tight_e2_refiner_unet_bin20_ink16/best.pth`
  - `checkpoints/line_refiner_tight_e2_refiner_unet_bin28_ink18/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_e2_refiner_unet` | 0.3776 | 6.497 | 1.757 | 0.3563 | 0.4068 |
| `line_refiner_tight_e2_refiner_unet_bin12_ink14` | 0.3791 | 6.336 | 1.731 | 0.3565 | 0.4098 |
| `line_refiner_tight_e2_refiner_unet_bin20_ink16` | 0.3941 | 6.076 | 2.159 | 0.3439 | 0.4670 |
| `line_refiner_tight_e2_refiner_unet_bin28_ink18` | 0.3730 | 6.676 | 1.678 | 0.3599 | 0.3921 |

Interpretation:

- `bin12_ink14` is a small improvement over the first `refiner_unet` and keeps
  ink slightly lower
- `bin20_ink16` has the best F1/chamfer but reaches `ink_ratio=2.159`; montage
  suggests this is mostly extra dark structure rather than a clean jump toward
  finished line art
- `bin28_ink18` is too conservative and gives up recall/F1
- stronger binary/ink pressure does not fundamentally solve the soft atari-copy
  behavior

Current decision:

- keep `line_refiner_tight_e2_refiner_unet_bin12_ink14` as the conservative
  refiner checkpoint
- keep `line_refiner_tight_e2_refiner_unet_bin20_ink16` only as an aggressive
  recall/black-growth reference
- next step should change the training strategy, not just static weights:
  staged fine-tune or target/aux dropout to force actual line cleanup

### Staged 2ch Refiner With Aux Dropout Started

Started a staged 2-channel refiner run to test whether training procedure helps
more than static loss-weight sweeps.

Rationale:

- `bin12_ink14` is the current conservative refiner checkpoint
- `bin20_ink16` remains useful as a possible aggressive/MoE expert but is too
  black for global adoption
- static binary/ink weight changes did not fundamentally escape soft atari-copy
- staged training should first learn structure, then tighten line output
- aux dropout should make the model use the rough channel instead of treating
  the ResNet-GAN atari as the answer

Implementation:

- `scripts/train_i2i_survey.py`
  - added `--aux-dropout`
  - added `--aux-scale-min`
  - when triggered, the atari channel is weakened toward white
- `experiments/run_line_refiner_staged.sh`
  - stage1: structure warmup
  - stage2: resume from stage1 and tighten with binary/ink
  - sends completion notification

Active run:

- tag: `line_refiner_staged_e2e2`
- unit: `lineart-line-refiner-staged-e2e2.service`
- stage1 epochs: 2
- stage2 epochs: 2
- aux dropout: `0.25`
- aux scale min: `0.0`
- train aux: `results/line_refiner_e2_atari_train`
- eval aux: `results/line_refiner_e2_atari_eval`
- expected montage: `results/compare_line_refiner_staged_e2e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_staged_e2e2_compare.csv`
- expected done marker: `logs/line_refiner_staged_e2e2.done`

### Staged 2ch Refiner With Aux Dropout Result

The staged 2ch refiner run completed.

Artifacts:

- done marker: `logs/line_refiner_staged_e2e2.done`
- montage: `results/compare_line_refiner_staged_e2e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_staged_e2e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_staged_e2e2_stage1_structure/best.pth`
  - `checkpoints/line_refiner_staged_e2e2_stage2_tight/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_e2_refiner_unet` | 0.3776 | 6.497 | 1.757 | 0.3563 | 0.4068 |
| `line_refiner_tight_e2_refiner_unet_bin12_ink14` | 0.3791 | 6.336 | 1.731 | 0.3565 | 0.4098 |
| `line_refiner_staged_e2e2_stage1_structure` | 0.3899 | 6.193 | 1.962 | 0.3524 | 0.4413 |
| `line_refiner_staged_e2e2_stage2_tight` | 0.3743 | 6.596 | 1.775 | 0.3539 | 0.4015 |

Interpretation:

- stage1 improves F1/recall but mostly by adding more ink/mid-gray structure
- stage2 reduces ink versus stage1, but also gives up the stage1 recall gain
  and does not beat `tight_bin12`
- aux dropout plus staged training is technically working but did not create a
  clear quality jump toward black/white line cleanup
- current best conservative refiner remains
  `line_refiner_tight_e2_refiner_unet_bin12_ink14`
- `bin20_ink16` remains useful as an aggressive black-growth/recall expert
  candidate for possible MoE routing, not as global default

Current decision:

- do not keep increasing static or staged refiner weights blindly
- the bottleneck is likely supervision/target formulation: the refiner still
  learns soft atari cleanup rather than explicit line extraction
- next useful work should either:
  - design an MoE/router comparison using conservative vs aggressive experts
  - or introduce a more line-specific target/loss, such as skeleton/thinning or
    edge-map supervision, before longer refiner training

### Skeleton-Supervised 2ch Refiner Sweep Started

Started a line-specific refiner experiment before spending more time on router
work.

Rationale:

- router/MoE probing is useful, especially for using `bin12_ink14` as a
  conservative expert and `bin20_ink16` as an aggressive expert
- router oracle work may become long-running if expanded beyond the 8 clean eval
  tiles, so reserve that for an overnight batch after the oracle script and
  expert set are fixed
- current daytime focus is to see whether explicit line supervision can move
  the refiner away from soft atari cleanup

Implementation:

- `scripts/train_i2i_survey.py`
  - added skeleton target generation from GT line art
  - added `--skeleton-weight`
  - skeleton loss is an auxiliary BCE term; final output still targets normal
    line art
- `experiments/run_line_refiner_survey.sh`
  - added skeleton-supervised `refiner_unet` candidates

Active run:

- tag: `line_refiner_skeleton_e2`
- unit: `lineart-line-refiner-skeleton-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet_skel03`
  - `refiner_unet_skel06`
  - `refiner_unet_skel10`
- expected montage: `results/compare_line_refiner_skeleton_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_skeleton_e2_compare.csv`
- expected done marker: `logs/line_refiner_skeleton_e2.done`
- notification: handled by `experiments/run_line_refiner_survey.sh`

Router/MoE note for later:

- keep `line_refiner_tight_e2_refiner_unet_bin12_ink14` as conservative expert
- keep `line_refiner_tight_e2_refiner_unet_bin20_ink16` as aggressive
  recall/black-growth expert
- schedule oracle/router probing for a night run after deciding the expert set
  and whether the oracle should cover only clean eval or a broader train/test
  sample

### Direction Survey And Cleanup Refiner Prep

Added a model-direction overview:

- `doc/model_directions.md`

It records the current state, defers rough/line correspondence grouping to a
future MoE/data-split phase, and surveys these line-art directions:

- multi-scale PatchGAN
- feature matching loss
- line/structure perceptual losses
- diffusion/ControlNet-like refinement
- shallow residual cleanup refiner
- confidence/thickness multi-head
- soft morphology/line-width loss
- HED/DexiNed-style edge heads
- attention/Swin-like refiner

Short survey priority from the overview:

1. shallow residual cleanup refiner
2. multi-scale PatchGAN with feature matching
3. soft morphology / line-width loss

Prepared the first short survey direction, shallow residual cleanup:

- `lineart/model_zoo.py`
  - added `ResidualCleanupGenerator`
  - model name: `cleanup`
  - uses the atari input as a base ink probability and predicts a bounded
    correction around it
- `scripts/train_i2i_survey.py`
  - added `cleanup` to trainable model choices
- `experiments/run_line_refiner_survey.sh`
  - added cleanup candidates:
    - `cleanup_bin12`
    - `cleanup_skel06`
    - `cleanup_ink18`

Do not start the cleanup survey until the active skeleton sweep finishes, to
avoid overlapping GPU training services.

### Skeleton-Supervised 2ch Refiner Sweep Result

The skeleton-supervised refiner sweep completed training/inference, but the
runner initially failed during montage creation because the bash compare
argument `resnet_atari="$EVAL_AUX"` was parsed as an assignment. Fixed the
runner by quoting the full `LABEL=DIR` model argument:

- `--model "resnet_atari=$EVAL_AUX"`

Recovered the skeleton results from completed inference outputs.

Artifacts:

- done marker: `logs/line_refiner_skeleton_e2.done`
- montage: `results/compare_line_refiner_skeleton_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_skeleton_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_skeleton_e2_refiner_unet_skel03/best.pth`
  - `checkpoints/line_refiner_skeleton_e2_refiner_unet_skel06/best.pth`
  - `checkpoints/line_refiner_skeleton_e2_refiner_unet_skel10/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_tight_e2_refiner_unet_bin12_ink14` | 0.3791 | 6.336 | 1.731 | 0.3565 | 0.4098 |
| `line_refiner_skeleton_e2_refiner_unet_skel03` | 0.3675 | 6.805 | 1.620 | 0.3604 | 0.3792 |
| `line_refiner_skeleton_e2_refiner_unet_skel06` | 0.3784 | 6.467 | 1.873 | 0.3508 | 0.4166 |
| `line_refiner_skeleton_e2_refiner_unet_skel10` | 0.3767 | 6.509 | 1.800 | 0.3526 | 0.4098 |

Interpretation:

- skeleton auxiliary loss is technically working
- `skel06` and `skel10` are near the conservative refiner but do not clearly
  improve it
- `skel03` is too conservative
- skeleton target alone is not the missing ingredient for line cleanup

### Shallow Residual Cleanup Refiner Started

Started the first short survey from `doc/model_directions.md`: shallow residual
cleanup.

Rationale:

- U-Net refiner may be too free to softly repaint the atari image
- cleanup model is biased to predict bounded corrections around the atari ink
  probability
- this tests whether architecture bias helps more than more loss-weight tuning

Active run:

- tag: `line_refiner_cleanup_e2`
- unit: `lineart-line-refiner-cleanup-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `cleanup_bin12`
  - `cleanup_skel06`
  - `cleanup_ink18`
- expected montage: `results/compare_line_refiner_cleanup_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_cleanup_e2_compare.csv`
- expected done marker: `logs/line_refiner_cleanup_e2.done`

### Shallow Residual Cleanup Refiner Result

The shallow residual cleanup refiner survey completed.

Artifacts:

- done marker: `logs/line_refiner_cleanup_e2.done`
- montage: `results/compare_line_refiner_cleanup_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_cleanup_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_cleanup_e2_cleanup_bin12/best.pth`
  - `checkpoints/line_refiner_cleanup_e2_cleanup_skel06/best.pth`
  - `checkpoints/line_refiner_cleanup_e2_cleanup_ink18/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_cleanup_e2_cleanup_bin12` | 0.4155 | 5.080 | 1.821 | 0.3338 | 0.5590 |
| `line_refiner_cleanup_e2_cleanup_skel06` | 0.4033 | 5.345 | 1.385 | 0.3395 | 0.5032 |
| `line_refiner_cleanup_e2_cleanup_ink18` | 0.4139 | 5.138 | 1.657 | 0.3372 | 0.5435 |

Interpretation:

- cleanup architecture is numerically strong; all three variants beat prior
  conservative refiner metrics
- `cleanup_skel06` is notable because it keeps `ink_ratio=1.385`, close to the
  BCE baseline, while reaching `F1=0.4033`
- montage shows a new artifact: white/emboss-like correction halos around
  atari structures
- therefore the current cleanup formulation is not directly adoptable as final
  line art, but the architecture bias is promising

Current decision:

- keep `cleanup_skel06` as a promising cleanup reference, not as a final model
- next cleanup iteration should constrain the correction:
  - reduce max residual delta
  - regularize correction magnitude
  - optionally predict only darkening/ink-addition rather than both dark and
    light correction
- continue broader direction survey after this, especially multi-scale
  PatchGAN + feature matching and morphology/line-width losses

### Cross-Direction Survey Continuation

Decision after cleanup result:

- the cleanup direction is promising but not ready; preserve follow-up ideas in
  `doc/model_directions.md`
- do not deepen cleanup immediately
- continue the remaining cross-direction survey first, then decide which family
  deserves longer work

Next cross-direction order:

1. multi-scale PatchGAN + feature matching
2. soft morphology / line-width loss
3. line/structure perceptual loss

Evaluation rule:

- montage inspection remains the adoption gate
- distinguish true line cleanup from extra ink, texture, or halo artifacts
- keep useful-but-risky outputs as possible expert candidates rather than global
  defaults

### Multi-Scale GAN + Feature Matching Survey Started

Started the first remaining cross-direction survey: multi-scale PatchGAN with
feature matching.

Implementation:

- `lineart/model_zoo.py`
  - `PatchDiscriminator` can now return intermediate features
  - added `MultiScalePatchDiscriminator`
- `scripts/train_i2i_survey.py`
  - added `--multiscale-gan`
  - added `--feature-match-weight`
  - adversarial loss now handles single-scale and multi-scale discriminator
    outputs
- `experiments/run_line_refiner_survey.sh`
  - added candidates:
    - `refiner_unet_msgan_fm`
    - `cleanup_msgan_fm`

Active run:

- tag: `line_refiner_msgan_e2`
- unit: `lineart-line-refiner-msgan-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet_msgan_fm`
  - `cleanup_msgan_fm`
- expected montage: `results/compare_line_refiner_msgan_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_msgan_e2_compare.csv`
- expected done marker: `logs/line_refiner_msgan_e2.done`

### Multi-Scale GAN + Feature Matching Survey Result

The multi-scale PatchGAN + feature matching survey completed.

Artifacts:

- done marker: `logs/line_refiner_msgan_e2.done`
- montage: `results/compare_line_refiner_msgan_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_msgan_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_msgan_e2_refiner_unet_msgan_fm/best.pth`
  - `checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_msgan_e2_refiner_unet_msgan_fm` | 0.3611 | 6.794 | 1.730 | 0.3417 | 0.3878 |
| `line_refiner_msgan_e2_cleanup_msgan_fm` | 0.3977 | 5.629 | 1.333 | 0.3432 | 0.4776 |

Interpretation:

- multi-scale GAN + feature matching is not useful on the plain U-Net refiner
  in this short run
- the cleanup variant is much more interesting: it keeps `ink_ratio=1.333`,
  close to the milddup baseline, while reaching `F1=0.3977`
- montage shows `cleanup_msgan_fm` reduces the white/emboss halo seen in the
  first cleanup sweep, but still has pencil/texture-like mid-gray rather than
  final black/white line art
- this makes cleanup + GAN/FM a promising family for longer or constrained
  cleanup work, but not a final answer yet

Current decision:

- keep `line_refiner_msgan_e2_cleanup_msgan_fm` as the best balanced cleanup
  candidate so far
- do not pursue `refiner_unet_msgan_fm`
- continue remaining short surveys before deepening cleanup

### Soft Morphology / Width Loss Survey Started

Started the second remaining cross-direction survey: soft morphology / line
width regularization.

Implementation:

- `scripts/train_i2i_survey.py`
  - added `soft_width_loss`
  - added `--width-weight`
  - current width loss penalizes broad local ink spread using a local average
    pool term
- `experiments/run_line_refiner_survey.sh`
  - added candidates:
    - `refiner_unet_width06`
    - `cleanup_width06`
    - `cleanup_width12`

Active run:

- tag: `line_refiner_width_e2`
- unit: `lineart-line-refiner-width-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet_width06`
  - `cleanup_width06`
  - `cleanup_width12`
- expected montage: `results/compare_line_refiner_width_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_width_e2_compare.csv`
- expected done marker: `logs/line_refiner_width_e2.done`

### Soft Morphology / Width Loss Survey Result

The soft morphology / width loss survey completed.

Artifacts:

- done marker: `logs/line_refiner_width_e2.done`
- montage: `results/compare_line_refiner_width_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_width_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_width_e2_refiner_unet_width06/best.pth`
  - `checkpoints/line_refiner_width_e2_cleanup_width06/best.pth`
  - `checkpoints/line_refiner_width_e2_cleanup_width12/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_width_e2_refiner_unet_width06` | 0.3650 | 6.887 | 1.664 | 0.3556 | 0.3794 |
| `line_refiner_width_e2_cleanup_width06` | 0.4152 | 5.101 | 1.610 | 0.3400 | 0.5410 |
| `line_refiner_width_e2_cleanup_width12` | 0.4028 | 5.325 | 1.345 | 0.3405 | 0.4996 |

Interpretation:

- width loss does not help the plain U-Net refiner in this short run
- cleanup variants remain numerically strong
- `cleanup_width12` has a good ink ratio but montage still shows the cleanup
  halo/emboss artifact
- width loss alone does not solve cleanup artifacts; `cleanup_msgan_fm` remains
  the better balanced cleanup candidate so far

Current decision:

- do not deepen width loss by itself immediately
- keep width loss as a possible regularizer to combine with better cleanup
  constraints later
- proceed to line/structure perceptual loss survey

### Line / Structure Perceptual Loss Survey Started

Started the third remaining cross-direction survey: line/structure perceptual
loss using lightweight Sobel/DoG-style structure features.

Implementation:

- `scripts/train_i2i_survey.py`
  - added `structure_pyramid_loss`
  - added `--structure-weight`
  - compares DoG-like high-frequency residuals and Sobel gradients at two
    scales
- `experiments/run_line_refiner_survey.sh`
  - added candidates:
    - `refiner_unet_struct08`
    - `cleanup_struct08`
    - `cleanup_struct08_msgan`

Active run:

- tag: `line_refiner_structure_e2`
- unit: `lineart-line-refiner-structure-e2.service`
- epochs: 2
- reuses existing atari caches:
  - train aux: `results/line_refiner_e2_atari_train`
  - eval aux: `results/line_refiner_e2_atari_eval`
- candidates:
  - `refiner_unet_struct08`
  - `cleanup_struct08`
  - `cleanup_struct08_msgan`
- expected montage: `results/compare_line_refiner_structure_e2.png`
- expected metrics: `results/fixed_output_metrics_line_refiner_structure_e2_compare.csv`
- expected done marker: `logs/line_refiner_structure_e2.done`

### Line / Structure Perceptual Loss Survey Result

The line/structure perceptual loss survey completed.

Artifacts:

- done marker: `logs/line_refiner_structure_e2.done`
- montage: `results/compare_line_refiner_structure_e2.png`
- metrics: `results/fixed_output_metrics_line_refiner_structure_e2_compare.csv`
- checkpoints:
  - `checkpoints/line_refiner_structure_e2_refiner_unet_struct08/best.pth`
  - `checkpoints/line_refiner_structure_e2_cleanup_struct08/best.pth`
  - `checkpoints/line_refiner_structure_e2_cleanup_struct08_msgan/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `line_refiner_structure_e2_refiner_unet_struct08` | 0.3698 | 6.783 | 1.757 | - | - |
| `line_refiner_structure_e2_cleanup_struct08` | 0.4101 | 5.267 | 1.756 | - | - |
| `line_refiner_structure_e2_cleanup_struct08_msgan` | 0.3982 | 5.664 | 1.400 | - | - |

Interpretation:

- structure loss keeps the cleanup family numerically strong
- montage still indicates halo/emboss texture remains in cleanup-like outputs
- `cleanup_struct08_msgan` is better balanced than plain structure cleanup, but
  not a clear visual jump over `cleanup_msgan_fm`

Current decision:

- treat halo propagation from the atari source as the next bottleneck
- run a focused halo mitigation survey before deepening A/cleanup tuning

### Halo Mitigation Survey Started

Started a combined survey for the three halo mitigation ideas, using the
comparatively promising multi-scale-GAN + feature-matching setup as the base.

Goal:

- determine whether downstream halo is reduced by changing the atari source,
  converting the atari hint, or changing how the atari is applied

Implemented:

- `scripts/materialize_i2i_aux.py`
  - added `--aux-dir`, allowing 2ch checkpoints to materialize cleaned aux
    outputs
- `scripts/preprocess_atari_aux.py`
  - added atari preprocessing modes: `cutoff`, `dog`, `edgehint`
- `lineart/model_zoo.py`
  - added `maskcleanup`, which uses atari as context instead of residual base
- `scripts/train_i2i_survey.py`
  - added `maskcleanup` as a trainable model choice
- `experiments/run_halo_mitigation_survey.sh`
  - runs the three candidates and builds clean eval montage/metrics

Active run:

- tag: `halo_mitigation_e2`
- unit: `lineart-halo-mitigation-e2.service`
- epochs: 2
- completion notification: ntfy via `experiments/send_autoloop_notification.sh`
- base atari aux:
  - train: `results/line_refiner_e2_atari_train`
  - eval: `results/line_refiner_e2_atari_eval`
- candidates:
  - `source_msgan_aux`: generate aux with
    `checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth`, then train
    cleanup + msgan/FM on that cleaner aux
  - `dog_aux_msgan`: DoG/local-background-subtracted atari hint, then cleanup +
    msgan/FM
  - `mask_order_msgan`: `maskcleanup` with raw atari as context/mask rather
    than output base
- expected montage: `results/compare_halo_mitigation_e2.png`
- expected metrics: `results/fixed_output_metrics_halo_mitigation_e2_compare.csv`
- expected done marker: `logs/halo_mitigation_e2.done`

Verification before launch:

- `bash -n experiments/run_halo_mitigation_survey.sh`
- `py_compile` passed for updated model/training/inference/preprocess scripts
- confirmed `checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth`
  exists

Current status:

- service is running
- current phase: materializing `source_msgan_aux` train split

### Halo Mitigation Survey Result

The three-candidate halo mitigation survey completed.

Artifacts:

- done marker: `logs/halo_mitigation_e2.done`
- montage: `results/compare_halo_mitigation_e2.png`
- metrics: `results/fixed_output_metrics_halo_mitigation_e2_compare.csv`
- checkpoints:
  - `checkpoints/halo_mitigation_e2_source_msgan_aux/best.pth`
  - `checkpoints/halo_mitigation_e2_dog_aux_msgan/best.pth`
  - `checkpoints/halo_mitigation_e2_mask_order_msgan/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_msgan_e2_cleanup_msgan_fm` | 0.3977 | 5.629 | 1.333 | 0.3432 | 0.4776 |
| `halo_mitigation_e2_source_msgan_aux` | 0.4020 | 5.546 | 1.547 | 0.3430 | 0.4925 |
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| `halo_mitigation_e2_mask_order_msgan` | 0.3509 | 7.074 | 1.381 | 0.3519 | 0.3534 |

Initial interpretation:

- `source_msgan_aux` gives the best F1 among the three and slightly improves
  over the previous `cleanup_msgan_fm`, but increases ink ratio
- `dog_aux_msgan` has the best chamfer and lowest ink ratio, so it is the most
  interesting halo-suppression candidate for montage inspection
- `mask_order_msgan` underperforms in this implementation; using atari only as
  context/mask is not enough by itself in the short run

Next action:

- inspect `results/compare_halo_mitigation_e2.png` visually, especially
  whether `dog_aux_msgan` reduced halo/emboss texture without losing line
  continuity

### Halo Filter / Flow-Mask Deep Dive Started

Started the next combined deep dive after montage inspection suggested that
`dog_aux_msgan` produces desirable whiteness and useful chamfer, but still
shows visible halo.

Goal:

- deepen the atari postprocess path with deblur / edge-preserving filter
  variants
- rebuild the mask path so the auxiliary space controls line flow rather than
  acting as a plain region/context image

Implementation:

- `scripts/preprocess_atari_aux.py`
  - added `wiener`
  - added `lucy` using local Lucy-Richardson deconvolution
  - added `bilateral`
  - added `nlmeans`
  - added `flowdog`
  - added `flowmask`
- `lineart/model_zoo.py`
  - added `flowmaskcleanup`
  - model structure: rough-only base logits plus a correction term gated by
    soft aux ink/line-flow mask
- `scripts/train_i2i_survey.py`
  - added `flowmaskcleanup` to model choices
- `experiments/run_halo_filter_flowmask_survey.sh`
  - added a combined survey runner for filter variants and flow-mask variants

Notes:

- true external AI deblurring is not included in this run because no dedicated
  deblur model is currently available locally
- the existing `cleanup_msgan_fm`/`source_msgan_aux` path remains the learned
  dehalo/deblur reference for now

Planned candidates:

- filter/postprocess candidates using `cleanup + msgan/FM`
  - `wiener_aux_msgan`
  - `lucy_aux_msgan`
  - `bilateral_aux_msgan`
  - `flowdog_aux_msgan`
- mask-rebuild candidates
  - `flowmask_cleanup`
  - `flowmask_cleanup_dropout`

Verification before launch:

- `bash -n experiments/run_halo_filter_flowmask_survey.sh`
- `py_compile` passed for updated preprocess/model/train/inference scripts
- eval-size preprocess smoke tests passed for `wiener`, `lucy`, `bilateral`,
  and `flowmask`
- `flowmaskcleanup` forward pass returned finite `1x1x480x480` output

Expected run:

- tag: `halo_filter_flowmask_e2`
- epochs: 2
- expected montage: `results/compare_halo_filter_flowmask_e2.png`
- expected metrics:
  `results/fixed_output_metrics_halo_filter_flowmask_e2_compare.csv`
- expected done marker: `logs/halo_filter_flowmask_e2.done`
- completion notification: ntfy via `experiments/send_autoloop_notification.sh`

### Halo Filter / Flow-Mask Deep Dive Result

The combined filter / flow-mask deep dive completed.

Artifacts:

- done marker: `logs/halo_filter_flowmask_e2.done`
- montage: `results/compare_halo_filter_flowmask_e2.png`
- metrics:
  `results/fixed_output_metrics_halo_filter_flowmask_e2_compare.csv`
- checkpoints:
  - `checkpoints/halo_filter_flowmask_e2_wiener_aux_msgan/best.pth`
  - `checkpoints/halo_filter_flowmask_e2_lucy_aux_msgan/best.pth`
  - `checkpoints/halo_filter_flowmask_e2_bilateral_aux_msgan/best.pth`
  - `checkpoints/halo_filter_flowmask_e2_flowdog_aux_msgan/best.pth`
  - `checkpoints/halo_filter_flowmask_e2_flowmask_cleanup/best.pth`
  - `checkpoints/halo_filter_flowmask_e2_flowmask_cleanup_dropout/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_msgan_e2_cleanup_msgan_fm` | 0.3977 | 5.629 | 1.333 | 0.3432 | 0.4776 |
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| `halo_filter_flowmask_e2_wiener_aux_msgan` | 0.3782 | 5.211 | 0.968 | 0.3473 | 0.4217 |
| `halo_filter_flowmask_e2_lucy_aux_msgan` | 0.3932 | 4.901 | 1.110 | 0.3398 | 0.4772 |
| `halo_filter_flowmask_e2_bilateral_aux_msgan` | 0.3736 | 5.615 | 1.235 | 0.3558 | 0.4053 |
| `halo_filter_flowmask_e2_flowdog_aux_msgan` | 0.4046 | 4.697 | 1.550 | 0.3461 | 0.5238 |
| `halo_filter_flowmask_e2_flowmask_cleanup` | 0.2176 | 8.292 | 0.368 | 0.2998 | 0.1737 |
| `halo_filter_flowmask_e2_flowmask_cleanup_dropout` | 0.2255 | 8.454 | 0.492 | 0.2811 | 0.1920 |

Visual interpretation from montage:

- `flowdog_aux_msgan` is best numerically, but the improvement appears to come
  from heavier line/halo pickup rather than true halo suppression
- `lucy_aux_msgan` is closest to the previous `dog_aux_msgan` behavior while
  keeping ink ratio modest; it is the most reasonable deconvolution-style
  candidate
- `wiener_aux_msgan` becomes too sparse/thin overall
- `bilateral_aux_msgan` does not improve the visual tradeoff
- both `flowmaskcleanup` variants lose too much line information; the gating is
  too restrictive and the rough-only base is too weak

Current decision:

- keep `dog_aux_msgan` and `lucy_aux_msgan` as controlled white/hint candidates
- treat `flowdog_aux_msgan` as a high-recall / high-ink expert candidate, not
  as a halo-suppression solution
- redesign flow-mask with a stronger base model or soft line-direction target
  before running it again

### Lucy Parameter / Stronger Flow-Mask Survey Started

Started a follow-up survey based on the current reading:

- `lucy_aux_msgan` is the best balanced filter candidate so far, but not yet a
  decisive improvement
- need to check whether Lucy tuning improves without simply converting halo
  into stippled/halftone texture
- mask remains a likely main route, but the previous implementation used a weak
  shallow rough-only base and an overly restrictive gate

Implementation:

- `scripts/preprocess_atari_aux.py`
  - added `lucy_mild`
  - added `lucy_strong`
  - added `lucy_thin`
- `lineart/model_zoo.py`
  - added `flowmaskunet`
  - added `softflowmaskunet`
  - both use a U-Net rough base plus flowmask-gated U-Net correction
- `scripts/train_i2i_survey.py`
  - added the new flow-mask U-Net model choices
- `experiments/run_lucy_mask_deep_survey.sh`
  - compares Lucy parameter variants and stronger mask variants in one montage

Planned candidates:

- Lucy sweep with `cleanup + msgan/FM`
  - `lucy_mild_aux_msgan`
  - `lucy_strong_aux_msgan`
  - `lucy_thin_aux_msgan`
- mask rebuild
  - `flowmask_unet`
  - `softflowmask_unet`

Verification before launch:

- `bash -n experiments/run_lucy_mask_deep_survey.sh`
- `py_compile` passed for updated preprocess/model/train/inference scripts
- eval-size preprocess smoke test passed for `lucy_mild`
- forward pass passed for `flowmaskunet` and `softflowmaskunet`

Expected run:

- tag: `lucy_mask_deep_e2`
- epochs: 2
- expected montage: `results/compare_lucy_mask_deep_e2.png`
- expected metrics: `results/fixed_output_metrics_lucy_mask_deep_e2_compare.csv`
- expected done marker: `logs/lucy_mask_deep_e2.done`
- completion notification: ntfy via `experiments/send_autoloop_notification.sh`

### Lucy Parameter / Stronger Flow-Mask Survey Result

The Lucy parameter / stronger flow-mask survey completed.

Artifacts:

- done marker: `logs/lucy_mask_deep_e2.done`
- montage: `results/compare_lucy_mask_deep_e2.png`
- metrics: `results/fixed_output_metrics_lucy_mask_deep_e2_compare.csv`
- checkpoints:
  - `checkpoints/lucy_mask_deep_e2_lucy_mild_aux_msgan/best.pth`
  - `checkpoints/lucy_mask_deep_e2_lucy_strong_aux_msgan/best.pth`
  - `checkpoints/lucy_mask_deep_e2_lucy_thin_aux_msgan/best.pth`
  - `checkpoints/lucy_mask_deep_e2_flowmask_unet/best.pth`
  - `checkpoints/lucy_mask_deep_e2_softflowmask_unet/best.pth`

Metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| `halo_filter_flowmask_e2_lucy_aux_msgan` | 0.3932 | 4.901 | 1.110 | 0.3398 | 0.4772 |
| `lucy_mask_deep_e2_lucy_mild_aux_msgan` | 0.4021 | 4.645 | 1.158 | 0.3371 | 0.5245 |
| `lucy_mask_deep_e2_lucy_strong_aux_msgan` | 0.4039 | 4.983 | 1.186 | 0.3496 | 0.4836 |
| `lucy_mask_deep_e2_lucy_thin_aux_msgan` | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| `lucy_mask_deep_e2_flowmask_unet` | 0.3140 | 7.367 | 0.776 | 0.3660 | 0.2828 |
| `lucy_mask_deep_e2_softflowmask_unet` | 0.3071 | 7.527 | 0.635 | 0.3818 | 0.2621 |

Visual interpretation from montage:

- Lucy parameter tuning improves the numerical tradeoff over the previous
  `lucy_aux_msgan`
- `lucy_thin_aux_msgan` has the best F1/chamfer but the low precision/high
  recall pattern suggests it may be gaining from extra faint/halo-like texture
- `lucy_mild_aux_msgan` looks like the safer balanced Lucy variant: good
  chamfer, ink ratio close to `dog_aux_msgan`, and less aggressive than
  `lucy_thin`
- `lucy_strong_aux_msgan` is numerically close but does not improve chamfer as
  much
- stronger flow-mask U-Net variants still underperform; the output is smoother
  and more line-losing than desired despite using a stronger base

Current decision:

- keep `lucy_mild_aux_msgan` as the current best controlled Lucy candidate
- treat `lucy_thin_aux_msgan` as a high-recall candidate requiring artifact
  scrutiny, not as a clean win
- mask route likely needs a different formulation than output gating, such as
  a supervised line-flow/line-affinity target or router/expert use, rather than
  simply multiplying corrections by an aux-derived gate

### Overnight Router/MoE Then Line-Field Chain Started

User decision:

- stack the line-field correction plan as the next direction
- run router/MoE overnight first
- after router/MoE completes, run the initial line-field correction experiment
- no intermediate instruction wait or notification while user is sleeping
- notify only after the final line-field stage completes

Line-field correction plan:

- replace deterministic gate-style masking with supervised line correction
  fields
- initial practical formulation:
  - input: rough + `lucy_mild` hint
  - primary output: final ink probability
  - auxiliary output: GT skeleton/centerline probability
  - auxiliary output: local offset vector toward nearest GT centerline
- target generation is deterministic from GT line art, but the model learns
  probabilistic ink/centerline fields and a correction vector field
- first run is a 2epoch initial comparison, not the final architecture

Implemented:

- `scripts/evaluate_oracle_moe.py`
  - per-sample expert metric table
  - balanced oracle expert selection
  - copies selected expert outputs into an oracle result directory
- `scripts/train_line_field_refiner.py`
  - U-Net with 4 heads: ink logits, centerline logits, dx, dy
  - losses: BCE/L1/tolerant/ink plus centerline BCE and offset smooth-L1
- `scripts/inference_line_field.py`
  - writes normal line output
  - optionally writes center/dx/dy debug maps
- `experiments/run_router_then_linefield_overnight.sh`
  - chained overnight runner
  - router/MoE oracle first
  - line-field training/inference second
  - sends ntfy only at final completion

Verification before launch:

- `bash -n experiments/run_router_then_linefield_overnight.sh`
- `py_compile` passed for new oracle/train/inference scripts
- oracle smoke test passed on clean lineart004 eval
- `LineFieldUNet` forward pass returned finite outputs:
  - ink: `1x1x480x480`
  - center: `1x1x480x480`
  - offset: `1x2x480x480`

Planned active run:

- tag: `router_linefield_overnight_e2`
- linefield tag: `linefield_initial_e2`
- epochs: 2
- router oracle candidates:
  - `bin12`
  - `bin20`
  - `cleanup_msgan`
  - `dog`
  - `lucy_mild`
  - `lucy_thin`
  - `flowdog`
- line-field train aux:
  `results/lucy_mask_deep_e2_lucy_mild_train`
- line-field eval aux:
  `results/lucy_mask_deep_e2_lucy_mild_eval`
- expected router summary:
  `results/router_linefield_overnight_e2_oracle_summary.json`
- expected router montage:
  `results/compare_router_linefield_overnight_e2_oracle.png`
- expected line-field montage:
  `results/compare_linefield_initial_e2.png`
- expected line-field metrics:
  `results/fixed_output_metrics_linefield_initial_e2_compare.csv`
- expected done marker:
  `logs/router_linefield_overnight_e2.done`

Current progress:

- unit: `lineart-router-linefield-overnight-e2.service`
- status: running
- router/MoE oracle stage completed
- current phase: `linefield_initial_e2` training
- no final done marker yet

Router/MoE oracle interim result:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `line_refiner_tight_e2_refiner_unet_bin12_ink14` | 0.3791 | 6.336 | 1.731 | 0.3565 | 0.4098 |
| `line_refiner_tight_e2_refiner_unet_bin20_ink16` | 0.3941 | 6.076 | 2.159 | 0.3439 | 0.4670 |
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| `lucy_mask_deep_e2_lucy_mild_aux_msgan` | 0.4021 | 4.645 | 1.158 | 0.3371 | 0.5245 |
| `lucy_mask_deep_e2_lucy_thin_aux_msgan` | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| `router_linefield_overnight_e2_oracle` | 0.4317 | 4.598 | 1.582 | 0.3431 | 0.5971 |

Oracle selected experts on the 8 clean eval tiles:

- `lucy_mild`: 3
- `lucy_thin`: 2
- `flowdog`: 1
- `bin20`: 1
- `bin12`: 1

Generated router artifacts:

- summary: `results/router_linefield_overnight_e2_oracle_summary.json`
- montage: `results/compare_router_linefield_overnight_e2_oracle.png`
- metrics: `results/fixed_output_metrics_router_linefield_overnight_e2_oracle_compare.csv`

### Overnight Router/MoE Then Line-Field Chain Result

The chained overnight run completed and sent the final ntfy notification.

Artifacts:

- done marker: `logs/router_linefield_overnight_e2.done`
- router summary: `results/router_linefield_overnight_e2_oracle_summary.json`
- router montage: `results/compare_router_linefield_overnight_e2_oracle.png`
- router metrics:
  `results/fixed_output_metrics_router_linefield_overnight_e2_oracle_compare.csv`
- line-field checkpoint: `checkpoints/linefield_initial_e2/best.pth`
- line-field outputs: `results/linefield_initial_e2`
- line-field debug maps: `results/linefield_initial_e2_debug`
- line-field montage: `results/compare_linefield_initial_e2.png`
- line-field metrics:
  `results/fixed_output_metrics_linefield_initial_e2_compare.csv`

Final metrics:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 | 0.3028 | 0.3042 |
| `halo_mitigation_e2_dog_aux_msgan` | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| `lucy_mask_deep_e2_lucy_mild_aux_msgan` | 0.4021 | 4.645 | 1.158 | 0.3371 | 0.5245 |
| `lucy_mask_deep_e2_lucy_thin_aux_msgan` | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| `router_linefield_overnight_e2_oracle` | 0.4317 | 4.598 | 1.582 | 0.3431 | 0.5971 |
| `linefield_initial_e2` | 0.4061 | 5.481 | 2.843 | 0.3203 | 0.5622 |

Interpretation:

- router/MoE oracle is clearly useful as an upper-bound probe: expert choice
  improves F1 over any single current expert on the 8 clean eval tiles
- oracle expert choices:
  - `lucy_mild`: 3
  - `lucy_thin`: 2
  - `flowdog`: 1
  - `bin20`: 1
  - `bin12`: 1
- `lucy_mild` and `lucy_thin` are now both valid halo-mitigation candidates,
  with `mild` safer and `thin` more recall-oriented
- initial line-field refiner runs technically, but overproduces ink
  (`ink_ratio=2.843`) and looks too blurred/dark in montage
- line-field direction should not be abandoned, but the first formulation needs
  stronger ink/width constraints or a different output reconstruction before a
  longer run

### Halo / Agreement Hypothesis Next

Current hypothesis:

- halo may be amplified because many training pairs have weak rough-line
  correspondence
- if training/inference uses high rough-line agreement pairs, the atari/refiner
  should need less hallucinated soft structure, reducing halo

Dataset status:

- enough data exists for an initial test:
  - `valid_train_base_clean_unique.txt`: 469
  - `valid_train_milddup800_clean.txt`: 800
  - `valid_train_kurip_strict_f1_cham_ink.txt`: 540
  - `valid_train_kurip_strict_precision.txt`: 540
  - `valid_train_ako5_clean.txt`: 2498
- however, a strict proof needs source-controlled splits because source style
  and agreement can be confounded

Next planned experiment:

- compute or reuse rough-line agreement scores
- start with a practical high-agreement test using strict/matched lists
- compare against current mixed/milddup training under the same halo-sensitive
  setup
- add a halo-specific evaluation view, not only F1/chamfer/ink, because
  chamfer can improve when halo or stippled texture is counted as near-line ink

### Agreement / Halo Survey Started

Started the first direct halo/agreement hypothesis test.

Goal:

- test whether training on high rough-line agreement pairs reduces halo versus
  low agreement pairs under otherwise comparable conditions

Design:

- base train pool: `dataset/pairs_480/valid_train_milddup800_clean.txt`
- score all 800 tiles by rough/line edge agreement
- create equal-size splits:
  - high agreement: top 240
  - low agreement: bottom 240
- train both with the same setup:
  - model: `cleanup`
  - aux: `lucy_mild` train/eval hints
  - loss/gan setup: same cleanup + multi-scale GAN + feature matching family
    used in the Lucy halo experiments
- compare against:
  - `milddup800`
  - `dog_aux_msgan`
  - `lucy_mild`
  - `lucy_thin`

Implementation:

- `tools/evaluation/score_pair_agreement.py`
  - computes edge F1/chamfer/density agreement score
  - writes high/low split lists
- `tools/evaluation/evaluate_halo_outputs.py`
  - measures grayscale halo-band ink around GT line art
  - reports halo-band ink/faint ratio and far-background faint ink
- `experiments/run_agreement_halo_survey.sh`
  - scores splits, trains high/low candidates, builds montage and metrics

Verification before launch:

- `bash -n experiments/run_agreement_halo_survey.sh`
- `py_compile` passed for new evaluation scripts
- agreement scoring smoke test passed on clean lineart004 eval
- halo metric smoke test passed for `lucy_mild` and `lucy_thin`

Expected run:

- tag: `agreement_halo_e2`
- epochs: 2
- split size: 240 high / 240 low
- expected agreement scores: `results/agreement_halo_e2_agreement_scores.csv`
- expected high list: `dataset/pairs_480/agreement_halo_e2_high240.txt`
- expected low list: `dataset/pairs_480/agreement_halo_e2_low240.txt`
- expected montage: `results/compare_agreement_halo_e2.png`
- expected metrics: `results/fixed_output_metrics_agreement_halo_e2_compare.csv`
- expected halo metrics: `results/halo_metrics_agreement_halo_e2_compare.csv`
- expected done marker: `logs/agreement_halo_e2.done`

### 2ch Haze-Control Survey Launch

Purpose:

- follow up on `raw_clean_mixed_2ch_e3`
- simple raw+clean 1ch mixing became too thin, but raw rough + cleaned rough aux
  2ch U-Net kept F1/chamfer strong
- remaining issue: 2ch restored useful ink density but also restored some
  background haze / line-near gray

Plan:

- keep 2ch setup:
  - input ch1: raw rough
  - input ch2: cleaned rough aux
  - target: clean line
- test two cleaned rough modes:
  - `edge_preserve`
  - `line_background_mild`
- train three variants per mode:
  - `auxdrop50_scale35`: randomly weaken aux during training
  - `bghaze06`: add GT-far background haze penalty
  - `auxdrop_bghaze`: combine both controls
- after inference, test generation-output halo suppression on haze variants:
  - `threshold35`
  - `unsharp_curve`

Implementation:

- `scripts/train_i2i_survey.py`
  - added `--background-haze-weight`
  - added `--background-haze-radius`
  - loss penalizes predicted ink outside a max-pooled GT line support mask
- `experiments/run_2ch_haze_control_survey.sh`
  - trains/evaluates the 2ch variants
  - builds montage, fixed metrics, and split haze/uncertainty metrics
  - sends ntfy notification at completion

Verification before launch:

- `./venv/bin/python -m py_compile scripts/train_i2i_survey.py`
- `bash -n experiments/run_2ch_haze_control_survey.sh`

Expected run:

- unit: `lineart-2ch-haze-control-e3.service`
- tag: `2ch_haze_control_e3`
- epochs: 3
- expected montage: `results/compare_2ch_haze_control_e3.png`
- expected metrics: `results/fixed_output_metrics_2ch_haze_control_e3_compare.csv`
- expected haze metrics:
  `results/haze_uncertainty_metrics_2ch_haze_control_e3_compare.csv`
- expected done marker: `logs/2ch_haze_control_e3.done`

Result:

- completed successfully
- systemd unit disappeared after completion because it was launched with
  `--collect`
- done marker: `logs/2ch_haze_control_e3.done`
- original montage: `results/compare_2ch_haze_control_e3.png`
- readable montage with short labels:
  `results/compare_2ch_haze_control_e3_readable.png`
- fixed metrics:
  `results/fixed_output_metrics_2ch_haze_control_e3_compare.csv`
- split haze/uncertainty metrics:
  `results/haze_uncertainty_metrics_2ch_haze_control_e3_compare.csv`

Short display names:

- `clean-edge`: cleaned rough `edge_preserve` U-Net e2
- `clean-mild`: cleaned rough `line_background_mild` U-Net e2
- `prev-edge2`: previous raw rough + `edge_preserve` aux 2ch U-Net
- `prev-mild2`: previous raw rough + `line_background_mild` aux 2ch U-Net
- `edge-auxdrop`: `edge_preserve` 2ch + aux dropout/scale
- `edge-bghaze`: `edge_preserve` 2ch + background haze penalty
- `edge-both`: `edge_preserve` 2ch + aux dropout/scale + background haze penalty
- `mild-auxdrop`: `line_background_mild` 2ch + aux dropout/scale
- `mild-bghaze`: `line_background_mild` 2ch + background haze penalty
- `mild-both`: `line_background_mild` 2ch + aux dropout/scale + background haze penalty
- `*-th35`: `threshold35` generation-output postprocess
- `*-unsharp`: `unsharp_curve` generation-output postprocess

Fixed metrics summary:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `prev-edge2` | 0.3936 | 5.708 | 1.450 |
| `prev-mild2` | 0.3866 | 6.046 | 1.473 |
| `edge-auxdrop` | 0.3818 | 6.089 | 1.417 |
| `edge-bghaze` | 0.3855 | 5.871 | 1.370 |
| `edge-both` | 0.3734 | 6.239 | 1.338 |
| `mild-auxdrop` | 0.3832 | 6.084 | 1.447 |
| `mild-bghaze` | 0.3606 | 6.443 | 1.112 |
| `mild-both` | 0.3634 | 6.501 | 1.279 |

Split haze/uncertainty summary:

| model | line_near_ink | line_near_faint | bg_haze | bg_haze_area |
|---|---:|---:|---:|---:|
| `clean-edge` | 0.3783 | 0.3989 | 0.2158 | 0.5378 |
| `clean-mild` | 0.3567 | 0.4771 | 0.2141 | 0.5541 |
| `prev-edge2` | 0.3760 | 0.4499 | 0.2363 | 0.5539 |
| `prev-mild2` | 0.3726 | 0.4718 | 0.2339 | 0.5607 |
| `edge-auxdrop` | 0.3740 | 0.4652 | 0.2345 | 0.5600 |
| `edge-bghaze` | 0.3680 | 0.4819 | 0.2250 | 0.5641 |
| `edge-both` | 0.3647 | 0.4917 | 0.2231 | 0.5688 |
| `mild-auxdrop` | 0.3711 | 0.4735 | 0.2314 | 0.5616 |
| `mild-bghaze` | 0.3560 | 0.5016 | 0.2196 | 0.5687 |
| `mild-both` | 0.3601 | 0.4942 | 0.2202 | 0.5662 |

Postprocess finding:

- `threshold35` removes faint haze numerically and increases F1/chamfer, but
  pushes `ink_ratio` to about `4.5-4.7`
- montage shows black thickening / over-binarization, so this should not be
  treated as a production-quality fix
- `unsharp_curve` worsens background haze and should be rejected for this
  branch
- generation-output cleanup remains possible, but needs conditional/background
  targeting rather than global thresholding

Current decision:

- best current balance is `edge-bghaze`
  - full checkpoint/result prefix:
    `2ch_haze_control_e3_edge_preserve_2ch_bghaze06_e3`
  - training recipe:
    - raw rough as channel 1
    - `edge_preserve` cleaned rough as channel 2
    - U-Net, 3 epochs, no autocontrast
    - `--background-haze-weight 0.06`
    - `--background-haze-radius 9`
- `edge-bghaze` slightly reduces background haze relative to previous 2ch while
  keeping F1/chamfer degradation small
- aux dropout/scale did not materially improve haze behavior
- further local loss/model tweaks are likely diminishing returns

Next direction:

- shift focus to data and router/MoE
- grow and characterize datasets rather than continuing narrow parameter
  sweeps in this worktree
- route or group by properties that now look important:
  - rough/line agreement
  - solid black fill / large black regions
  - dirty or hazy rough input
  - line-near uncertainty versus far background haze
- use `edge-bghaze` as the current single-model baseline for comparisons

### Pair Feature Scan For Router/Data Splits

Added a reusable per-tile feature scanner:

- `tools/evaluation/score_pair_features.py`
  - scores rough/line agreement
  - rough dirty/haze statistics relative to GT line support
  - line-near uncertainty versus far-background haze
  - target ink and large dark connected-component / black-fill indicators
  - writes per-tile CSV, list/source summary CSV, and optional split lists
- `experiments/run_pair_feature_scan.sh`
  - scans current major train lists:
    - `valid_train_base_clean_unique.txt`
    - `valid_train_milddup800_clean.txt`
    - `valid_train_kurip_strict_f1_cham_ink.txt`
    - `valid_train_kurip_strict_precision.txt`
    - `valid_train_ako5_clean.txt`

Generated:

- `results/pair_feature_scan_current.csv`
- `results/pair_feature_scan_current_summary.csv`
- `dataset/pairs_480/pair_feature_scan_current_splits/`

Generated split counts:

| split | unique rows |
|---|---:|
| `agreement_high` | 895 |
| `agreement_mid` | 1676 |
| `agreement_low` | 1302 |
| `background_haze_high` | 839 |
| `line_near_uncertainty_high` | 706 |
| `black_fill_high` | 811 |
| `high_agreement_low_haze` | 384 |
| `clean_router_seed` | 734 |

Initial reading:

- `kurip` has the strongest rough/line agreement among the current source
  groups.
- `ako5` contributes many low-agreement and high-background-haze tiles.
- `housei` is the clearest source for large black-fill / solid dark-region
  behavior.
- `clean_router_seed` is the first practical seed list for router/data-split
  experiments because it keeps moderate/high agreement while avoiding the
  highest haze and black-fill extremes.

Next action:

- use `clean_router_seed`, `agreement_high`, `agreement_low`,
  `background_haze_high`, and `black_fill_high` as the first data axes for
  router/MoE feature analysis against the current expert set.

### Router Feature Probe

Added feature-stratified router/oracle probe tooling:

- `tools/evaluation/build_router_feature_probe_lists.py`
  - samples small source-balanced probe lists from feature CSV groups
  - writes per-group lists, a combined list, and group label CSV
- `tools/evaluation/analyze_router_oracle_features.py`
  - joins oracle choices back to feature labels
  - summarizes expert choices and metrics by probe group and source prefix
- `experiments/run_router_feature_probe.sh`
  - builds probe lists
  - materializes current expert outputs on train-split samples
  - runs oracle selection and feature-group analysis

Completed:

- tag: `router_feature_probe_current`
- count per group: 12
- unique combined samples: 63
- done marker: `logs/router_feature_probe_current.done`
- probe labels:
  `dataset/pairs_480/router_feature_probe_current_lists/router_feature_probe_current_labels.csv`
- oracle summary:
  `results/router_feature_probe_current_oracle_summary.json`
- feature summary:
  `results/router_feature_probe_current_oracle_feature_summary.csv`
- montage:
  `results/compare_router_feature_probe_current_oracle.png`

Probe-group oracle summary:

| group | oracle F1 | chamfer | ink ratio | expert counts |
|---|---:|---:|---:|---|
| `agreement_high` | 0.4358 | 5.429 | 0.930 | `lucy_thin` 5, `bin12` 2, `lucy_mild` 2, others 3 |
| `agreement_low` | 0.0630 | 13.698 | 2.034 | mixed, no reliable winner |
| `background_haze_high` | 0.2951 | 7.987 | 1.018 | `lucy_thin` 7, `lucy_mild` 3, `flowdog` 2 |
| `line_near_uncertainty_high` | 0.3464 | 6.520 | 0.942 | `lucy_thin` 9, `flowdog` 2, `edge_bghaze` 1 |
| `black_fill_high` | 0.5331 | 5.246 | 0.534 | `lucy_thin` 8, `lucy_mild` 4 |
| `clean_seed` | 0.3655 | 6.300 | 2.681 | `lucy_mild` 3, `lucy_thin` 3, `bin12` 2, `bin20` 2, others 2 |

Initial interpretation:

- `lucy_thin` dominates many feature groups, especially line-near uncertainty
  and black-fill-heavy probes, but it remains recall-oriented and needs visual
  artifact scrutiny.
- `agreement_high` is a good router-development regime because multiple experts
  are competitive and oracle F1 is meaningfully high.
- `agreement_low` is not solved by the current expert set. Its oracle F1 is
  very low, so this group should drive data repair/exclusion or a separate
  hallucination/data-alignment strategy rather than ordinary router tuning.
- `clean_seed` is mixed: bin experts still win some samples. This is useful for
  learning a first router because it contains nontrivial expert choice, but the
  high mean ink ratio suggests the balanced score still permits over-inked wins.

Next action:

- inspect `results/compare_router_feature_probe_current_oracle.png`
- adjust oracle score or add an ink/haze-aware penalty before training a router,
  because current balanced scoring can still choose over-inked bin outputs
- treat low-agreement samples as a separate data-quality problem instead of
  forcing them into the first router training set

### Agreement-Low QC And Repair Categories

Followed up on the question of whether `agreement_low` is effectively a
pair-rebuild group.

Added:

- `tools/evaluation/make_pair_feature_qc.py`
  - builds rough / line / edge-overlay QC montages from feature CSV rows
  - overlay convention:
    - rough edges: red
    - line edges: blue
    - overlap: green
- `tools/evaluation/classify_pair_feature_failures.py`
  - writes coarse repair/QC categories from feature metrics
  - emits one CSV and category-specific file lists

Generated QC:

- `results/agreement_low_qc_worst40.png`
- `results/agreement_low_qc_ako5_worst40.png`
- `results/agreement_low_qc_housei_worst40.png`

Generated classifications:

- `results/agreement_low_repair_categories.csv`
- `results/agreement_low_repair_categories_summary.csv`
- `dataset/pairs_480/agreement_low_repair_categories/`

Category counts over 1,302 unique `agreement_low` tiles:

| category | count |
|---|---:|
| `valid_low_correspondence_or_metric_failure` | 702 |
| `black_fill_or_solid_region` | 206 |
| `line_too_sparse_fragment` | 115 |
| `ako5_uninterpretable_rough_rebuild_or_exclude` | 105 |
| `dirty_or_hazy_rough` | 97 |
| `rough_too_sparse_vs_line` | 77 |

Source/category summary:

- `ako5`: 1,169 tiles
  - many are visually weak/noisy rough against sparse or semantically mismatched
    line fragments
  - worst cases look like pair repair / extraction-QC candidates, not useful
    router training data
- `housei`: 127 tiles
  - many are black-fill / solid-region or rough-too-sparse-vs-line cases
  - this is less clearly "wrong pair"; more likely a metric failure or a
    separate black-fill expert/router category
- `lineart`/`orig`: only 3 each in `agreement_low`; not a major source of the
  low-agreement problem

Current interpretation:

- `agreement_low` should not be treated as one homogeneous rebuild group.
- User visual inspection confirmed that, unlike `housei`, the `ako5` and
  overall worst montages include rough images that are not interpretable as
  usable underdrawings.
- Practical split:
  - `ako5_uninterpretable_rough_rebuild_or_exclude`, `dirty_or_hazy_rough`,
    `rough_too_sparse_vs_line`, and some `line_too_sparse_fragment` entries:
    repair/re-extract/review or exclude
  - `black_fill_or_solid_region`: keep separate as black-fill/solid-region
    routing or metric-special-case data
  - `valid_low_correspondence_or_metric_failure`: needs a second visual pass
    before deciding whether it is valid hallucination-style data or metric
    failure

Next action:

- treat `ako5_uninterpretable_rough_rebuild_or_exclude` as a hard repair /
  exclusion queue before any router training
- build a second-pass QC montage for `valid_low_correspondence_or_metric_failure`,
  sampled by source, because this is now the largest unresolved bucket

Additional generated QC:

- `results/agreement_low_qc_ako5_uninterpretable_worst40.png`

### Ako5 Uninterpretable Rough Origin Check

User visually inspected the `agreement_low` montage and pointed out that the
overall worst and `ako5` worst tiles contain rough images that are not
underdrawings at all.

Follow-up finding:

- the problematic names are `ako5_...`, not `ako5r_...`
- all 105 `ako5_uninterpretable_rough_rebuild_or_exclude` tiles are present in
  `valid_train_std15.txt`
- 0 of them are present in `valid_train_ako5_regions.txt`
- therefore the source is the old same-coordinate / std15 ako5 extraction path,
  not the newer region-matching ako5 extraction

Trace:

- old script: `tools/pair_extraction/prepare_ako5.py`
  - tiled full pages into fixed 480px same-coordinate grid
  - filtered only by rough autocontrast std >= 15
  - did not require line ink, rough/line edge agreement, semantic
    interpretability, or non-margin content
- later script: `tools/pair_extraction/refilter_ako5.py`
  - rebuilt `valid_train_ako5_clean.txt` from existing `ako5_*.jpg`
  - used rough autocontrast std and line ink only
  - still did not catch edge-less / margin / uninterpretable rough tiles

Generated provenance artifacts:

- `results/ako5_uninterpretable_provenance.csv`
- `results/agreement_low_ako5_uninterpretable_raw_ac_line.png`
- `results/ako5_001_saved_rough_grid_bad_marked.png`

Concrete evidence:

- 105 affected tiles span 21 pages, but 54 are from `ako5_001`
- `ako5_001` bad coordinates cluster heavily around page top, left/right
  margins, and near-empty grid cells
- rough stats for the 105 affected tiles:
  - mean gray: about 251 / 255
  - mean rough ink: about 0.015
  - median rough edge density: 0
  - edge F1: 0 for all affected tiles
- raw/ac/line montage shows many raw rough tiles are nearly blank, margin
  artifacts, text/noise, or scan/JPEG residue rather than usable underdrawings

Current decision:

- `ako5_uninterpretable_rough_rebuild_or_exclude` is confirmed data pollution
  from old full-page same-coordinate std15 extraction
- do not use these tiles for router training, base training, or expert
  comparison except as negative/audit examples
- region-matched `ako5r_*` data remains the safer ako5 path; future ako5
  rebuilds should start from region matching or add an explicit
  interpretable-rough/content gate before saving

### Ako5 Uninterpretable Hard Exclusion Lists

Hard-excluded the 105 confirmed uninterpretable ako5 rough tiles from current
affected training lists without overwriting the old lists.

Added:

- `tools/evaluation/filter_pair_list.py`
- `experiments/filter_ako5_uninterpretable_lists.sh`

Exclusion source:

- `dataset/pairs_480/agreement_low_repair_categories/agreement_low_ako5_uninterpretable_rough_rebuild_or_exclude.txt`

Generated replacement lists:

| old list | new list | old rows | new rows | removed |
|---|---|---:|---:|---:|
| `valid_train_std15.txt` | `valid_train_std15_no_ako5_uninterp.txt` | 6622 | 6517 | 105 |
| `valid_train_std15_clean_split_moredupes.txt` | `valid_train_std15_clean_split_moredupes_no_ako5_uninterp.txt` | 6598 | 6493 | 105 |
| `valid_train_milddup800_clean.txt` | `valid_train_milddup800_clean_no_ako5_uninterp.txt` | 800 | 733 | 67 |
| `valid_train_ako5_clean.txt` | `valid_train_ako5_clean_no_ako5_uninterp.txt` | 2498 | 2456 | 42 |
| `valid_train_warm_plan1.txt` | `valid_train_warm_plan1_no_ako5_uninterp.txt` | 3158 | 3116 | 42 |
| `valid_train_warm_plan2.txt` | `valid_train_warm_plan2_no_ako5_uninterp.txt` | 2824 | 2788 | 36 |

For each generated list, a matching `_removed.txt` audit list was also written.

Verification:

- all generated `*_no_ako5_uninterp.txt` lists have zero overlap with the 105
  excluded names

Current operating rule:

- use `*_no_ako5_uninterp.txt` variants for any future training or router/data
  analysis that would otherwise use the affected old lists
- do not launch new runs from `valid_train_std15.txt`,
  `valid_train_std15_clean_split_moredupes.txt`,
  `valid_train_milddup800_clean.txt`, or `valid_train_ako5_clean.txt` unless
  explicitly auditing the old polluted lists

### Broader Ako5 Bad-Rough Audit And Reformed Lists

Expanded the audit after the user noted the issue is not merely low agreement:
the bad `ako5` rough images are not underdrawings at all.

Ran full scans:

- `results/pair_feature_scan_std15_full.csv`
- `results/pair_feature_scan_std15_full_summary.csv`
- `results/pair_feature_scan_std15_no_ako5_uninterp.csv`
- `results/pair_feature_scan_std15_no_ako5_uninterp_summary.csv`

Finding:

- the initial 105 hard-excluded tiles were only the worst/visible subset from
  current feature scans
- full `valid_train_std15.txt` still contained many same-origin bad rough
  candidates from old `ako5_` same-coordinate page tiling
- broad old-ako5 bad-rough rule found 859 candidates
- union with the manually confirmed 105 produced a final exclude list of 884
  names

Generated:

- `dataset/pairs_480/ako5_uninterpretable_broad/ako5_uninterpretable_broad_candidates.txt`
- `dataset/pairs_480/ako5_uninterpretable_broad/ako5_uninterpretable_broad_new_candidates.txt`
- `dataset/pairs_480/ako5_uninterpretable_broad/ako5_uninterpretable_final_exclude.txt`
- `results/ako5_uninterpretable_broad_candidates.csv`
- `results/ako5_uninterpretable_broad_candidates_qc_worst80.png`
- `results/ako5_uninterpretable_broad_new_qc_worst80.png`

Reformed affected lists using the 884-name final exclude set:

| old list | new list | old rows | new rows | removed |
|---|---|---:|---:|---:|
| `valid_train_std15.txt` | `valid_train_std15_no_ako5_badrough.txt` | 6622 | 5738 | 884 |
| `valid_train_std15_clean_split_moredupes.txt` | `valid_train_std15_clean_split_moredupes_no_ako5_badrough.txt` | 6598 | 5714 | 884 |
| `valid_train_milddup800_clean.txt` | `valid_train_milddup800_clean_no_ako5_badrough.txt` | 800 | 728 | 72 |
| `valid_train_ako5_clean.txt` | `valid_train_ako5_clean_no_ako5_badrough.txt` | 2498 | 2428 | 70 |
| `valid_train_warm_plan1.txt` | `valid_train_warm_plan1_no_ako5_badrough.txt` | 3158 | 3088 | 70 |
| `valid_train_warm_plan2.txt` | `valid_train_warm_plan2_no_ako5_badrough.txt` | 2824 | 2762 | 62 |

Verification:

- every generated `*_no_ako5_badrough.txt` list has zero overlap with
  `ako5_uninterpretable_final_exclude.txt`
- `valid_train_base_clean_unique.txt`,
  `valid_train_warm_regions_clean_split.txt`, and
  `valid_train_warm_regions_clean_unique.txt` have zero overlap with the final
  exclude set

Regenerated current feature scan from cleaned lists:

- `results/pair_feature_scan_current_no_ako5_badrough.csv`
- `results/pair_feature_scan_current_no_ako5_badrough_summary.csv`
- `dataset/pairs_480/pair_feature_scan_current_no_ako5_badrough_splits/`

New scan summary:

- rows: 4,705, down from 4,847
- exact overlap with final exclude: 0
- broad bad-rough candidates remaining: 0
- overall agreement score improved from `-0.2347` to `-0.1987`
- `milddup800` agreement improved from `-0.2254` to `-0.0944`
- `milddup800` edge F1 improved from `0.2323` to `0.2553`

Updated defaults:

- `experiments/run_pair_feature_scan.sh` now defaults to
  `pair_feature_scan_current_no_ako5_badrough` and uses:
  - `valid_train_milddup800_clean_no_ako5_badrough.txt`
  - `valid_train_ako5_clean_no_ako5_badrough.txt`
- `experiments/run_router_feature_probe.sh` now defaults to
  `results/pair_feature_scan_current_no_ako5_badrough.csv`

Current operating rule supersedes the previous narrower exclusion rule:

- use `*_no_ako5_badrough.txt`, not `*_no_ako5_uninterp.txt`, for future
  training/router/data analysis
- old same-coordinate `ako5_` data remains suspect; prefer region-matched
  `ako5r_*` for future ako5 expansion

### Other Source Bad-Rough Check

Checked whether the same non-underdrawing / margin-noise bug appears outside
old `ako5_` data in the cleaned current scan:

- scan: `results/pair_feature_scan_current_no_ako5_badrough.csv`
- unique rows checked: 3,737
- final ako5 bad-rough overlap: 0
- broad bad-rough candidates remaining: 0 for `ako5`

Source check:

- `kurip`: no same-signature bad-rough candidates
- `lineart`: no same-signature bad-rough candidates
- `orig`: no same-signature bad-rough candidates
- `housei`: 10 low-information / edge-failure candidates

Generated:

- `dataset/pairs_480/other_source_badrough_review.txt`
- `results/other_source_badrough_review_qc.png`

Interpretation:

- the `housei` 10 are not the same bug as `ako5`; the roughs are interpretable
  but often very sparse, text/fragment-like, or paired with black-fill /
  solid-region targets that break the edge-agreement metric
- keep them as a review / black-fill / metric-special-case bucket, not as a
  broad hard-exclusion like old `ako5_` bad rough

### Bad-Rough Clean Retrain Survey

Ran a short e3 retrain survey on the cleaned mild list:

- train list:
  `dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt`
  (728 rows)
- eval list: `dataset/pairs_480/eval_clean_lineart004_8.txt`
- runner: `experiments/run_badrough_retrain_survey.sh`
- done marker: `logs/badrough_retrain_e3.done`
- montage: `results/compare_badrough_retrain_e3.png`
- fixed metrics:
  `results/fixed_output_metrics_badrough_retrain_e3_compare.csv`
- haze metrics:
  `results/haze_uncertainty_metrics_badrough_retrain_e3_compare.csv`

Results:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old edge_bghaze | 0.3855 | 5.871 | 1.370 | 0.3715 | 0.4041 |
| new edge_bghaze | 0.3665 | 6.486 | 1.505 | 0.3514 | 0.3882 |
| old dog | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| new dog | 0.4132 | 4.615 | 2.628 | 0.3060 | 0.6479 |
| old lucy_mild | 0.4021 | 4.645 | 1.158 | 0.3371 | 0.5245 |
| new lucy_mild | 0.4208 | 4.545 | 2.522 | 0.3067 | 0.6806 |
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| new lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |

Haze/uncertainty summary:

- new edge_bghaze slightly reduces faint haze but loses F1/chamfer; do not
  promote it over the existing edge_bghaze baseline from this e3 run
- cleanup variants improve F1/chamfer by increasing recall, but all new cleanup
  models substantially over-ink (`ink_ratio` 2.5-2.8)
- line-near faint ratio drops strongly for the cleanup variants, especially
  lucy_thin (`0.1906 -> 0.0955`), but the montage shows the gain comes with
  visibly thicker/darker strokes

Interpretation:

- badrough removal changes training behavior; it is worth retraining ako5-touched
  models rather than trusting the polluted checkpoints
- however, the e3 cleanup retrains are not direct replacements because they are
  too recall/ink heavy
- next cleanup retrain should keep the cleaned list but add a stronger ink or
  width constraint, or reduce recall pressure / aux darkness before considering
  longer training

### Bad-Rough Ink/Width Constrained Survey

Ran a follow-up e3 cleanup-only survey to counter the over-inking from
`badrough_retrain_e3`.

Runner:

- `experiments/run_badrough_inkwidth_survey.sh`
- tag: `badrough_inkwidth_e3`
- train list:
  `dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt`
- reused aux from `badrough_retrain_e3`
- cleanup settings:
  - `pos_weight=4.0` (down from 5.0)
  - `ink_weight=0.18` (up from 0.14)
  - `binary_weight=0.16` (up from 0.10)
  - `width_weight=0.08`
  - `structure_weight=0.06` (up from 0.04)
  - `adv_weight=0.025` (down from 0.03)
  - `aux_dropout=0.25`, `aux_scale_min=0.75`

Generated:

- `results/compare_badrough_inkwidth_e3.png`
- `results/fixed_output_metrics_badrough_inkwidth_e3_compare.csv`
- `results/haze_uncertainty_metrics_badrough_inkwidth_e3_compare.csv`
- `logs/badrough_inkwidth_e3.done`

Fixed metric summary:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old dog | 0.3957 | 4.785 | 1.160 | 0.3440 | 0.4826 |
| clean dog | 0.4132 | 4.615 | 2.628 | 0.3060 | 0.6479 |
| inkwidth dog | 0.3846 | 5.130 | 1.339 | 0.3236 | 0.4794 |
| old lucy_mild | 0.4021 | 4.645 | 1.158 | 0.3371 | 0.5245 |
| clean lucy_mild | 0.4208 | 4.545 | 2.522 | 0.3067 | 0.6806 |
| inkwidth lucy_mild | 0.3834 | 4.993 | 1.191 | 0.3153 | 0.4941 |
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| clean lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |
| inkwidth lucy_thin | 0.4013 | 4.747 | 1.659 | 0.2977 | 0.6264 |

Haze/uncertainty summary:

- `inkwidth_dog` returns haze and ink metrics almost to old dog, but gives up the
  clean retrain's F1/chamfer gains
- `inkwidth_lucy_mild` is also over-corrected; lower F1 than the old model
- `inkwidth_lucy_thin` is the only useful middle point:
  - ink ratio reduced from `2.789` to `1.659`
  - recall reduced from `0.7272` to `0.6264`
  - F1 falls from `0.4227` to `0.4013`, slightly below old lucy_thin `0.4095`
  - faint line-near uncertainty stays better than old lucy_thin
    (`0.1630` vs `0.1906`)

Interpretation:

- the first ink/width setting is too strong for dog and lucy_mild
- for lucy_thin, the direction is promising but should be relaxed:
  width around `0.04-0.06`, binary around `0.12-0.14`, and/or less aux dropout
  should target `ink_ratio` around `1.4-1.6` without giving up as much F1
- current best production candidate remains old `lucy_thin` or the clean
  `lucy_thin` only if over-ink can be tolerated; no automatic promotion from
  `badrough_inkwidth_e3`

### Bad-Rough Lucy-Thin Relaxed Survey

Ran a focused lucy_thin-only follow-up after the first ink/width survey
over-corrected into gray, weak strokes.

Runner:

- `experiments/run_badrough_lucy_thin_relaxed_survey.sh`
- tag: `badrough_lucy_thin_relaxed_e3`
- train list:
  `dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt`
- reused aux from `badrough_retrain_e3_lucy_thin_*`

Variants:

- `relaxed_a`: `pos_weight=4.3`, `ink_weight=0.16`, `binary_weight=0.12`,
  `width_weight=0.04`, `aux_dropout=0.10`, `aux_scale_min=0.85`
- `relaxed_b`: `pos_weight=4.5`, `ink_weight=0.16`, `binary_weight=0.14`,
  `width_weight=0.05`, `aux_dropout=0.15`, `aux_scale_min=0.80`

Generated:

- `results/compare_badrough_lucy_thin_relaxed_e3.png`
- `results/fixed_output_metrics_badrough_lucy_thin_relaxed_e3_compare.csv`
- `results/haze_uncertainty_metrics_badrough_lucy_thin_relaxed_e3_compare.csv`
- `logs/badrough_lucy_thin_relaxed_e3.done`

Fixed metric summary:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| clean lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |
| inkwidth lucy_thin | 0.4013 | 4.747 | 1.659 | 0.2977 | 0.6264 |
| relaxed_a | 0.4137 | 4.605 | 2.066 | 0.3022 | 0.6672 |
| relaxed_b | 0.4163 | 4.553 | 2.236 | 0.3021 | 0.6821 |

Haze/uncertainty summary:

- `relaxed_b` is the best numerical middle point from this survey:
  - F1 improves over old lucy_thin (`0.4095 -> 0.4163`)
  - chamfer is close to old (`4.536 -> 4.553`)
  - ink is still high (`1.252 -> 2.236`), but less extreme than clean
    lucy_thin (`2.789`)
  - line-near faint ratio remains improved over old
    (`0.1906 -> 0.1291`)
- `relaxed_a` is slightly less inky (`2.066`) but also slightly lower F1
  (`0.4137`)

Interpretation:

- relaxing the ink/width settings moves in the right direction compared with
  `badrough_inkwidth_e3`
- `relaxed_b` is the current best compromise if accepting some gray/thick
  output is okay
- none of the e3 relaxed variants fully solves the visual gray-stroke issue;
  the next step should test a post-binarization / contrast calibration pass, or
  train with an explicit output-threshold/blackness objective rather than only
  lowering ink width

### Bad-Rough Lucy-Thin Postprocess Calibration

Ran postprocess calibration on the best relaxed candidate
`badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b`.

Updated:

- `tools/compare/postprocess_line_outputs.py` now supports generic
  `thresholdNN` modes, not only the previously hard-coded thresholds.

Runner:

- `experiments/run_badrough_lucy_thin_postprocess_survey.sh`
- tag: `badrough_lucy_thin_postprocess`
- modes: `threshold48`, `threshold50`, `threshold52`, `threshold54`,
  `threshold55`, `unsharp_curve`

Generated:

- `results/compare_badrough_lucy_thin_postprocess.png`
- `results/fixed_output_metrics_badrough_lucy_thin_postprocess_compare.csv`
- `results/haze_uncertainty_metrics_badrough_lucy_thin_postprocess_compare.csv`
- `logs/badrough_lucy_thin_postprocess.done`

Fixed metric summary:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| clean lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |
| relaxed_b | 0.4163 | 4.553 | 2.236 | 0.3021 | 0.6821 |
| threshold48 | 0.4203 | 4.472 | 2.539 | 0.3009 | 0.7103 |
| threshold50 | 0.4163 | 4.553 | 2.236 | 0.3021 | 0.6821 |
| threshold52 | 0.4096 | 4.667 | 1.917 | 0.3024 | 0.6466 |
| threshold54 | 0.4015 | 4.822 | 1.612 | 0.3022 | 0.6096 |
| threshold55 | 0.3955 | 4.934 | 1.431 | 0.3021 | 0.5836 |
| unsharp_curve | 0.4172 | 4.269 | 4.442 | 0.2754 | 0.8699 |

Interpretation:

- hard thresholding removes the gray-stroke look, but exposes scratchy / rough
  binary marks rather than clean confident line art
- `threshold52` is the most practical compromise:
  - F1 roughly matches old lucy_thin (`0.4096` vs `0.4095`)
  - ink is reduced from relaxed_b `2.236` to `1.917`
  - output is black/white instead of gray
- `threshold54` and `threshold55` reduce ink further but give up too much F1 /
  recall
- `threshold48` keeps F1 high but remains over-inked
- `unsharp_curve` is not useful here: very high recall and chamfer, but ink
  explodes (`4.442`)

Next direction:

- postprocess can remove gray, but does not solve the underlying line-quality
  issue
- likely need training-time blackness / threshold-aware loss, e.g. match metrics
  around an output threshold while keeping width/ink regularization modest

### Bad-Rough Lucy-Thin Threshold-Loss Survey

Added threshold-aware differentiable losses to `scripts/train_i2i_survey.py`:

- `soft_threshold(pred, threshold, sharpness)`
- new args:
  - `--threshold-shape-weight`
  - `--threshold-ink-weight`
  - `--threshold-value`
  - `--threshold-sharpness`
- the added loss applies tolerant F1 and ink matching after a differentiable
  threshold approximation, so training sees the quality of the thresholded
  output rather than only the gray probability map

Runner:

- `experiments/run_badrough_lucy_thin_threshold_loss_survey.sh`
- tag: `badrough_lucy_thin_threshold_e3`
- train list:
  `dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt`
- reused `badrough_retrain_e3_lucy_thin_*` aux

Variants:

- `thresh_light`: `threshold_shape=0.04`, `threshold_ink=0.08`,
  `threshold=0.52@24`, `binary=0.12`, `width=0.04`
- `thresh_mid`: `threshold_shape=0.07`, `threshold_ink=0.12`,
  `threshold=0.52@28`, `binary=0.12`, `width=0.04`

Generated:

- `results/compare_badrough_lucy_thin_threshold_e3.png`
- `results/fixed_output_metrics_badrough_lucy_thin_threshold_e3_compare.csv`
- `results/haze_uncertainty_metrics_badrough_lucy_thin_threshold_e3_compare.csv`
- `logs/badrough_lucy_thin_threshold_e3.done`

Fixed metric summary:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| old lucy_thin | 0.4095 | 4.536 | 1.252 | 0.3198 | 0.5962 |
| clean lucy_thin | 0.4227 | 4.457 | 2.789 | 0.3005 | 0.7272 |
| relaxed_b | 0.4163 | 4.553 | 2.236 | 0.3021 | 0.6821 |
| relaxed_b threshold52 | 0.4096 | 4.667 | 1.917 | 0.3024 | 0.6466 |
| thresh_light | 0.4199 | 4.489 | 2.227 | 0.3033 | 0.6938 |
| thresh_light threshold52 | 0.4146 | 4.595 | 1.934 | 0.3044 | 0.6616 |
| thresh_mid | 0.4211 | 4.471 | 2.204 | 0.3041 | 0.6965 |
| thresh_mid threshold52 | 0.4172 | 4.551 | 1.922 | 0.3064 | 0.6652 |

Interpretation:

- threshold-aware loss improved the relaxed candidate:
  - raw `thresh_mid` is now close to clean lucy_thin F1 but with less ink
    (`0.4211`, ink `2.204` vs clean `0.4227`, ink `2.789`)
  - `thresh_mid threshold52` is a better black/white postprocess point than
    `relaxed_b threshold52` (`0.4172` vs `0.4096`)
- montage still shows thresholded output is scratchy / rough-binary-like, not
  truly clean final line art
- current best compromise is `thresh_mid` as a gray/soft output, or
  `thresh_mid threshold52` if black/white output is required
- next useful step is either:
  - longer training from `thresh_mid` settings, or
  - a more structural threshold loss that penalizes speckle / disconnected
    scratch texture after thresholding

### 2026-07-24: Lucy-Thin Cleanup Tuning Pause

現在の結論:

- `ako5` bad-rough 汚染を除外した再学習は有効で、傾向は見えた
- ただし `lucy_thin` 系の現行調整は、実用線画としてはまだ遠い
- soft 出力では灰色の確率場になり、threshold 後は小連結成分が大量に
  出てざらつく
- 単純な後処理や閾値 loss だけでは、線画らしい一本線・連続線には届かない

重要な判断:

- `lucy_thin` / cleanup 系の loss 微調整はいったんここで締める
- 次の主リソースは、モデル側の微調整ではなくデータ・preprocess 側へ移す

次に優先する方向:

- 下絵 preprocess の研究
  - ノイズ除去
  - グレー抑制
  - 揺らぎ・重ね描き線の抑制
  - ラフの候補線を「モデルが一本線として解釈しやすい入力」に寄せる
- データセット拡張
  - クリーンで一致度の高い rough/line ペアの絶対数を増やす
  - old `ako5_` のような同座標グリッド抽出ではなく、region-matched /
    alignment-aware な抽出を優先する
  - agreement / badrough / black-fill special-case のQCを継続し、汚染を
    増やさない

現時点の扱い:

- `badrough_lucy_thin_threshold_e3_lucy_thin_thresh_mid` は調査上の
  参考候補として保持
- 黒白が必要な場合の参考は
  `badrough_lucy_thin_threshold_e3_lucy_thin_thresh_mid_post_threshold52`
- ただしどちらも production 昇格はしない
- 今後の改善軸は「より良い入力前処理」と「より良い高一致データ」

### 2026-07-24: hamlabi Raw Dataset Import

Input:

- raw zip found in workspace: `dataset_hamlabi.zip`
- user requested `dataset_hamulabi.zip`; actual filename is `hamlabi`
- zip manifest contains 13 page-level line/sketch pairs
- all line/sketch pages are same-size `(4961, 7016)`

Extractor update:

- updated `tools/pair_extraction/prepare_kurip_tiles.py`
  - supports configurable `--zip-root`
  - supports Windows-style zip members such as `dataset_hamlabi\manifest.json`
  - supports configurable `--name-prefix`
  - supports `--exclude-page`

Extraction policy:

- treated hamlabi as page-aligned same-coordinate data, but with strict QC
- used 480px tile scan with:
  - `--name-prefix hamlabi`
  - `--min-rough-std 15`
  - `--ink-min 0.03`
  - `--min-edge-pixels 350`
  - `--min-f1 0.25`
  - `--max-chamfer 12`
  - `--exclude-page 0009`
- `page0009` was excluded because the accepted tile contained strong black page-edge fill

Generated:

- `dataset/pairs_480/valid_train_hamlabi.txt` = 12 pairs
- `dataset/pairs_480/train/rough/hamlabi_*.jpg`
- `dataset/pairs_480/train/line/hamlabi_*.jpg`
- `results/hamlabi_tiles_strict_noedge.csv`
- `results/hamlabi_tiles_strict_noedge_qc.png`
- `results/hamlabi_tiles_strict_noedge_qc_tail.png`
- `results/hamlabi_tiles_strict_noedge_qc_sample.png`

Interpretation:

- hamlabi currently adds only a small number of high-agreement pairs
- QC shows the rough/line relationship is interpretable and does not resemble the old
  `ako5_` same-coordinate bad rough contamination
- this is useful as clean high-agreement data, not as a large-volume training source

### 2026-07-24: results Cleanup After Bad-Rough Audit

Removed obsolete `results/` artifacts tied to old bad-rough / old-list work:

- `results/agreement_low_*`
- `results/ako5_uninterpretable_*`
- old same-coordinate ako5 inspection/QC artifacts:
  - `results/ako5_001_saved_rough_grid_bad_marked.png`
  - `results/ako5_align_*`
  - `results/ako5_aligned_qc.png`
  - `results/ako5_calib_qc.png`
  - `results/ako5_hightone.png`
  - `results/ako5_strict_survivors.png`
  - `results/inspect_ako5_pairs.png`
  - `results/refilter_ako5_pass.png`
- superseded pair feature scans:
  - `results/pair_feature_scan_current.csv`
  - `results/pair_feature_scan_current_summary.csv`
  - `results/pair_feature_scan_std15_full.csv`
  - `results/pair_feature_scan_std15_full_summary.csv`
  - `results/pair_feature_scan_std15_no_ako5_uninterp.csv`
  - `results/pair_feature_scan_std15_no_ako5_uninterp_summary.csv`
- old raw/clean mixed 2ch survey outputs based on pre-`no_ako5_badrough` lists:
  - `results/raw_clean_mixed_2ch_e3_*`
  - `results/compare_raw_clean_mixed_2ch_e3.png`
  - `results/fixed_output_metrics_raw_clean_mixed_2ch_e3_compare.csv`
  - `results/haze_uncertainty_metrics_raw_clean_mixed_2ch_e3_compare.csv`

Kept:

- `results/badrough_*` because those were post-exclusion retrain/tuning artifacts using
  `valid_train_milddup800_clean_no_ako5_badrough.txt`
- `results/ako5_region_*` because those refer to the separate region-matched `ako5r_*`
  path, not the old same-coordinate `ako5_` path
- cleaned feature scan artifacts such as
  `results/pair_feature_scan_current_no_ako5_badrough.csv`

### 2026-07-24: hamlabi Region-Matching Branch Start

Branch:

- `hamlabi-region-extraction`

Reason:

- user pointed out that the first hamlabi import repeated the old mistake:
  extracting by same XY coordinates alone
- revised assumption: for most raw manuscripts, same-XY crop equality is not a
  valid correspondence signal even if rough/line page sizes match
- the first `hamlabi_*.jpg` 12-pair strict extraction is therefore not a
  production dataset; keep only as a failed baseline / QC reference

Added policy doc:

- `doc/region_dataset_extraction_policy.md`

Policy:

- do not create new raw-manuscript training pairs by XY equality alone
- line image should provide semantic region proposals
- rough image should be searched with translation and scale candidates
- preserve aspect ratio and normalize by long side for region datasets
- produce candidate CSV/JSON/QC first; do not write a production train list
  until review decisions are recorded
- use VLM only as second-pass review / rerank / reject gate, not as the primary
  extractor

Added deterministic hamlabi matcher:

- `tools/pair_extraction/match_hamlabi_regions.py`

Current run:

- job: `hamlabi-region-match.service`
- log: `logs/hamlabi_region_match.log`
- input: `dataset_hamlabi.zip`
- pages: 13
- deterministic candidates generated: 72
- deterministic `candidate` rows: 34
- lower-score review rows: 38

Generated:

- `results/hamlabi_region_candidates.csv`
- `results/hamlabi_region_candidates.json`
- `results/hamlabi_region_candidates_qc.png`

Added VLM review helper:

- `tools/pair_extraction/vlm_review_hamlabi_regions.py`

Generated panel-only review artifacts:

- `results/hamlabi_region_vlm_review_panel_only.csv`
- `results/hamlabi_region_vlm_review_panels/` = 72 panels
- `results/hamlabi_region_vlm_review_panel_sheet_top24.png`

VLM status:

- local Ollama API at `127.0.0.1:11434` was not running
- no Ollama user service was found
- VLM review is prepared but not executed; run the helper without
  `--panel-only` after a local vision model is available

Initial visual interpretation:

- the region matcher is clearly better aligned with the desired goal than
  same-XY 480 tiles
- top candidates include same character / same scene rough-line pairs at
  panel or character scale
- remaining issues:
  - some regions are too large and close to full-panel crops
  - black-fill and margin fragments still pass deterministic scoring
  - VLM / human review is required before saving any training manifest

Follow-up user interpretation:

- quick visual review suggests the extracted region pairs are directionally good
- large regions should not be rejected immediately; they can be treated as
  parent regions and searched again internally for smaller character / body-part
  pairs
- black-fill-heavy regions are part of hamlabi's source style and should be kept
  with labels rather than treated as contamination
- for future MoE work, black-fill-heavy hamlabi samples may become a specialized
  expert/router branch

Updated:

- `tools/pair_extraction/match_hamlabi_regions.py`
  - adds `black_fill_ratio`
  - adds `feature_tags`, including `large_parent`, `black_fill`, `wide_panel`,
    and `tall_panel`
- `doc/region_dataset_extraction_policy.md`
  - black-fill regions are not automatic rejects
  - large/multi-character regions are parent candidates for recursive search
- `tools/pair_extraction/refine_hamlabi_large_regions.py`
  - reads large parent rows from deterministic candidate CSV
  - searches smaller child regions inside each parent
  - preserves parent bbox metadata for traceability

Child-region run:

- input parents: 21 large parent candidates
- output children: 110
- deterministic `candidate` rows: 82
- `review_low_score` rows: 28
- tags:
  - `black_fill`: 19
  - `tall_panel`: 13
  - `wide_panel`: 9
  - `large_parent`: 7

Generated:

- `results/hamlabi_region_child_candidates.csv`
- `results/hamlabi_region_child_candidates.json`
- `results/hamlabi_region_child_candidates_qc.png`
- `results/hamlabi_region_child_vlm_review_panel_only.csv`
- `results/hamlabi_region_child_vlm_review_panels/` = 110 panels
- `results/hamlabi_region_child_vlm_review_panel_sheet_top32.png`

Interpretation:

- parent candidates remain the better panel / character-scale pool
- child candidates add useful subregions but also many fragments; keep as a
  separate review lane rather than replacing parent candidates
- no hamlabi region candidate has been promoted to a training list yet

### 2026-07-24: hamlabi Parent Review Dataset Materialization

Materialized the preliminary Codex-reviewed parent candidates as a variable
aspect review dataset.

Input:

- review CSV: `results/hamlabi_region_codex_review_prelim.csv`
- selected decision: `accept_review`
- selected rows: 31

Generated:

- `dataset/regions_hamlabi_review/rough/` = 31 PNGs
- `dataset/regions_hamlabi_review/line/` = 31 PNGs
- `dataset/regions_hamlabi_review/manifest.json`
- `dataset/regions_hamlabi_review/manifest.csv`
- `dataset/regions_hamlabi_review/README.md`
- `results/hamlabi_review_accept31_materialized_qc.png`

Dataset properties:

- variable aspect, long side normalized to 768 px
- width range: 356-768 px
- height range: 474-768 px
- pages covered: 12 source pages
- not added to any `dataset/pairs_480/*.txt` train list

Review feature tags added to manifest:

- `dense_ink_or_black_fill`: 18
- `large_panel_or_character`: 11
- `narrow_region`: 4

Current interpretation:

- this is the first usable hamlabi review dataset artifact
- it remains review-stage, not production training data
- black/dense-ink cases are deliberately preserved as hamlabi style features
  for possible MoE specialization

### 2026-07-24: hamlabi Child Region Curated Review Dataset

Created a conservative child-region review lane after inspecting the child
candidate contact sheet.

Intermediate child review:

- `results/hamlabi_region_child_codex_review_prelim.csv`
  - heuristic `child_accept_review`: 33
  - `child_hold_review`: 39
  - `child_reject_prelim`: 38

After Codex visual pass, the heuristic accept set was too permissive and was
reduced:

- `results/hamlabi_region_child_codex_review_curated.csv`
  - `child_curated_review`: 17
  - `child_hold_review`: 56
  - `child_reject_prelim`: 37

Materialized:

- `dataset/regions_hamlabi_child_review/rough/` = 17 PNGs
- `dataset/regions_hamlabi_child_review/line/` = 17 PNGs
- `dataset/regions_hamlabi_child_review/manifest.json`
- `dataset/regions_hamlabi_child_review/manifest.csv`
- `dataset/regions_hamlabi_child_review/README.md`
- `results/hamlabi_child_review_curated17_materialized_qc.png`

Interpretation:

- child curated data is lower confidence than parent review data
- keep it separate as part / small-character / subregion candidates
- do not mix into parent review or `pairs_480` train lists yet

### 2026-07-24: hamlabi Codex VLM Final Review

Ollama review was not available, so Codex VLM review was used for the current
small candidate set.

Inputs:

- parent materialized review dataset: 31 rows
- child curated review dataset: 17 rows
- total reviewed rows: 48

Generated final review CSV:

- `results/hamlabi_region_codex_final_review.csv`

Final review decisions over 48 rows:

- `final_accept_black_fill`: 17
- `final_accept`: 6
- `final_accept_large_parent`: 5
- `final_accept_child`: 6
- `final_accept_child_black_fill`: 6
- `final_hold_child`: 5
- `final_hold_parent`: 3

Materialized accepted rows:

- `dataset/regions_hamlabi_final_review/rough/` = 40 PNGs
- `dataset/regions_hamlabi_final_review/line/` = 40 PNGs
- `dataset/regions_hamlabi_final_review/manifest.csv`
- `dataset/regions_hamlabi_final_review/manifest.json`
- `dataset/regions_hamlabi_final_review/README.md`
- `results/hamlabi_final_review40_materialized_qc.png`

Current status:

- final review dataset is ready as a variable-aspect region artifact
- it is still not a `pairs_480` train list
- next required work is training-loader/materialization design for variable
  aspect region datasets, or a deliberate conversion policy into model-size
  tensors

### 2026-07-24: Region Loader And Materialization Policy

Added variable-aspect region dataset support:

- `lineart/region_dataset.py`
  - reads CSV/JSON region manifests
  - resolves `final_rough_path` / `final_line_path`, or `rough_path` /
    `line_path`
  - supports `square_pad` fit mode
  - supports `resize_stretch` as explicit compatibility mode
  - returns model-ready square tensors while preserving source aspect ratio in
    `square_pad`

Updated:

- `scripts/train_i2i_survey.py`
  - `--region-manifest`
  - `--region-fit-mode {square_pad,resize_stretch}`
  - `--image-size`
  - rejects unsupported region+aux and region+skeleton combinations for now

Recommended region training invocation:

```bash
venv/bin/python scripts/train_i2i_survey.py \
  --model cleanup \
  --region-manifest dataset/regions_hamlabi_final_review/manifest.csv \
  --region-fit-mode square_pad \
  --image-size 480 \
  --checkpoint-dir checkpoints/hamlabi_region_smoke
```

Added fixed materializer:

- `tools/pair_extraction/materialize_region_manifest_square.py`

Added policy doc:

- `doc/region_materialization_policy.md`

Policy:

- canonical format remains variable-aspect manifest
- `square_pad` is preferred for fixed-size conversion
- `resize_stretch` is compatibility-only because it distorts raw manuscript
  geometry
- fixed materializations stay in separate directories and are not appended to
  existing `pairs_480` lists by default

Generated fixed materializations:

- `dataset/regions_hamlabi_final_review_480_squarepad/`
  - rough: 40
  - line: 40
  - size: 480x480
  - QC: `results/hamlabi_final_review40_480_squarepad_qc.png`
- `dataset/regions_hamlabi_final_review_768_squarepad/`
  - rough: 40
  - line: 40
  - size: 768x768
  - QC: `results/hamlabi_final_review40_768_squarepad_qc.png`

Verification:

- region loader returned valid 480 and 768 tensor batches
- fixed materialized sample images have expected square dimensions
- syntax check passed for loader, training script, and materializer

### 2026-07-24: hamlabi Final Review v2 After User Visual Correction

User visually reviewed `results/hamlabi_final_review40_materialized_qc.png`.

Corrections:

- remove mismatched final QC indices:
  - 26
  - 27
  - 28
  - 37
  - 38
  - 40
- manually subcrop these multi-panel / multi-region rows to keep only the
  matching panel/region:
  - 1
  - 2
  - 30
  - 34

Generated manual subcrop QC:

- `results/hamlabi_manual_subcrop_candidates_qc.png`

Generated corrected canonical dataset:

- `dataset/regions_hamlabi_final_review_v2/`
  - rough: 34
  - line: 34
  - `manifest.csv`
  - `manifest.json`
  - `README.md`
  - `removed_user_mismatch.txt`
  - `manual_subcrop_replacements.txt`
- QC:
  - `results/hamlabi_final_review_v2_34_qc.png`

Updated loader/materializer support:

- `lineart/region_dataset.py` now prefers `v2_rough_path` / `v2_line_path`
  before older final paths
- `tools/pair_extraction/materialize_region_manifest_square.py` now also
  prefers v2 paths and names

Generated corrected fixed materializations:

- `dataset/regions_hamlabi_final_review_v2_480_squarepad/`
  - rough: 34
  - line: 34
  - size: 480x480
  - QC: `results/hamlabi_final_review_v2_34_480_squarepad_qc.png`
- `dataset/regions_hamlabi_final_review_v2_768_squarepad/`
  - rough: 34
  - line: 34
  - size: 768x768
  - QC: `results/hamlabi_final_review_v2_34_768_squarepad_qc.png`

Current operating rule:

- use `regions_hamlabi_final_review_v2`, not the earlier 40-row
  `regions_hamlabi_final_review`, for future hamlabi region experiments
- the non-v2 datasets remain historical audit artifacts

### 2026-07-24: hamlabi Region v2 Loader Smoke Training

Purpose:

- verify the variable-aspect region manifest loader can drive the existing
  i2i survey training path before launching longer hamlabi expert runs
- avoid accidental CPU training when CUDA is hidden by the execution sandbox

Code updates:

- `scripts/train_i2i_survey.py`
  - added `--require-cuda`
  - training now fails fast if CUDA is required but unavailable
- `tools/compare/make_region_manifest_compare.py`
  - runs a checkpoint against a region manifest
  - applies the same `square_pad` / `resize_stretch` normalization as the
    region loader
  - writes rough / model / line-GT review montage

CUDA finding:

- inside the managed sandbox, `/dev/nvidia*` is not visible and
  `torch.cuda.is_available()` is false
- outside the sandbox, PyTorch sees `NVIDIA GeForce RTX 3060`
- long GPU jobs should therefore be launched with approved elevated
  `systemd-run --user` service execution

Smoke run:

- unit: `hamlabi-region-v2-smoke.service`
- manifest: `dataset/regions_hamlabi_final_review_v2/manifest.csv`
- rows: 34
- model: `cleanup`
- image size: 480
- fit mode: `square_pad`
- epochs: 5
- checkpoint dir: `checkpoints/hamlabi_region_v2_smoke/`
- log: `logs/hamlabi_region_v2_smoke.log`

Outputs:

- `checkpoints/hamlabi_region_v2_smoke/best.pth`
- `checkpoints/hamlabi_region_v2_smoke/epoch005.pth`
- `results/hamlabi_region_v2_smoke_outputs/`
- `results/hamlabi_region_v2_smoke_compare.png`

Result:

- training completed on CUDA
- loss decreased from `G=0.7049` to `G=0.5696`
- 5 epoch output remains very rough and gray, as expected for a smoke run on
  only 34 pairs
- the important result is that the corrected v2 region dataset can now pass
  through loader, training, checkpoint save, inference, and montage review

### 2026-07-24: hamlabi Region v2 480px Training Trials

Cleanup 80 epoch trial:

- unit: `hamlabi-region-v2-cleanup480.service`
- checkpoint dir: `checkpoints/hamlabi_region_v2_cleanup480/`
- log: `logs/hamlabi_region_v2_cleanup480.log`
- epochs: 80
- checkpoints:
  - `best.pth`
  - `epoch020.pth`
  - `epoch040.pth`
  - `epoch060.pth`
  - `epoch080.pth`
- montages:
  - `results/hamlabi_region_v2_cleanup480_epoch020_compare.png`
  - `results/hamlabi_region_v2_cleanup480_epoch080_compare.png`

Cleanup result:

- completed on CUDA
- loss decreased from `G=0.7049` to `G=0.4613`
- visual output remained mostly gray rough-copy enhancement
- measured epoch080 output over the 12-sample review:
  - mean intensity: `0.808`
  - std: `0.140`
  - black pixels `<0.2`: `0.000`
  - white pixels `>0.8`: `0.617`
- conclusion: the shallow cleanup model is not adequate for this
  rough-to-finished-line mapping with the current 34-pair hamlabi set

UNet 480 trial:

- unit: `hamlabi-region-v2-unet480.service`
- checkpoint dir: `checkpoints/hamlabi_region_v2_unet480/`
- log: `logs/hamlabi_region_v2_unet480.log`
- epochs: 80 target
- montage so far:
  - `results/hamlabi_region_v2_unet480_epoch020_compare.png`
  - `results/hamlabi_region_v2_unet480_epoch040_compare.png`

UNet interim result:

- epoch020 reached `G=0.4414`, already below cleanup epoch080
- epoch040 reached `G=0.3368`
- visually, UNet learns strong black area placement but collapses fine line
  structure into soft black blobs
- interim conclusion: higher capacity helps optimization but hamlabi black-fill
  regions dominate the signal; the next useful direction is likely stronger
  line/fill separation or region-type/expert splitting rather than simply
  extending the same loss longer

UNet final result:

- completed 80 epochs on CUDA
- final loss: `G=0.2593`
- checkpoints:
  - `best.pth`
  - `epoch020.pth`
  - `epoch040.pth`
  - `epoch060.pth`
  - `epoch080.pth`
- final montage:
  - `results/hamlabi_region_v2_unet480_epoch080_compare.png`
- output statistics over the 12-sample review:
  - epoch020: mean `0.642`, std `0.208`, black `<0.2` `0.081`, white `>0.8` `0.029`
  - epoch040: mean `0.723`, std `0.197`, black `<0.2` `0.059`, white `>0.8` `0.427`
  - epoch080: mean `0.778`, std `0.203`, black `<0.2` `0.061`, white `>0.8` `0.692`

Interpretation:

- 34 curated pairs are enough to validate the region extraction and training
  plumbing, but not enough to expect a clean hamlabi expert
- UNet optimizes much better than cleanup, but it mostly learns black-fill
  placement and loses fine line structure
- this supports treating hamlabi as a future MoE/expert source with explicit
  region-type splitting:
  - line-dominant character/panel crops
  - black-fill-heavy crops
  - multi-character/layout crops
- before larger training, prioritize increasing clean aligned pair count and
  separating fill masks from line targets

### 2026-07-24: Pair Expansion Policy After hamlabi Trials

Decision:

- apply the hamlabi variable-aspect region approach to future pair expansion
  for other datasets as well
- do not return to 480px fixed crops as the source pair creation rule
- do not rely on same XY coordinates or regular fixed-size rectangular tiling
  as the primary extraction method

Reason:

- the hamlabi trials showed that even visually plausible page associations do
  not make identical coordinate crops reliable
- successful same-XY matches are expected to be rare cases, not the default
  behavior of raw manuscript datasets
- with fewer than 50 pairs, training results are mainly useful for detecting
  failure modes and validating data plumbing, not for expecting clean expert
  quality

Updated extraction direction:

- anchor candidate generation on the finished line image
- cut semantically coherent units first:
  - panel / koma
  - character
  - face / body / hand
  - large parent region when the exact child is not separable yet
- search the rough image for matching content under translation, scale, and
  aspect-preserving normalization candidates
- keep oversized parent matches for recursive child search instead of forcing
  them into fixed 480px squares
- treat the rough-match search process itself as a major improvement target

Documentation:

- `doc/region_dataset_extraction_policy.md` now records this as the default
  policy for hamlabi and other future raw-dataset pair expansion work

### 2026-07-24: CPU-Only Region Search Loop

Goal:

- make variable-aspect pair exploration runnable while GPU training is active
  or while the operator is away
- keep the loop deterministic and review-gated; it should generate candidates,
  not silently promote a training dataset

Added:

- `tools/pair_extraction/run_region_search_loop.py`
  - restartable state file
  - parent/child profile rounds
  - CPU thread limits
  - `CUDA_VISIBLE_DEVICES=` to avoid occupying GPU
  - skip completed rounds unless `--force` is used
- `doc/region_search_loop.md`
  - operation notes
  - output paths
  - service command

Started service:

- unit: `region-search-loop-hamlabi.service`
- log: `logs/region_search_loop_hamlabi.log`
- state: `results/region_search_loop/state.json`
- output root: `results/region_search_loop/`
- CPU threads: 2

Current behavior:

- runs hamlabi parent candidate search profiles
- for each parent profile, runs child refinement profiles against oversized /
  large parent candidates
- writes CSV/JSON/QC montage artifacts per round
- does not materialize accepted training pairs automatically

### 2026-07-24: Valid Mask Trial for Partially Rewritten Regions

Motivation:

- some large region pairs are mostly matched but contain local redrawn /
  mismatched parts
- instead of dropping the whole pair, try excluding only mismatched target
  regions from the training loss

Added:

- `tools/pair_extraction/build_region_valid_masks.py`
  - reads a region manifest
  - normalizes rough/line with the same square-pad policy
  - writes `valid_mask_path` per pair
  - writes a QC montage with rough / line / valid mask / ignored overlay
- `lineart/region_dataset.py`
  - can load `valid_mask_path`
  - returns `(rough, target, valid_mask)` when a mask is present
- `scripts/train_i2i_survey.py`
  - added `--region-mask-key`
  - applies valid masks to BCE/L1 and masked variants of the region losses

Generated mask datasets:

- aggressive black-fill-aware trial:
  - `dataset/regions_hamlabi_final_review_v2_768_masked/`
  - result: too aggressive; black-fill regions caused large false ignores
- conservative line-target-only trial:
  - `dataset/regions_hamlabi_final_review_v2_768_masked_line_conservative/`
  - QC: `dataset/regions_hamlabi_final_review_v2_768_masked_line_conservative/valid_mask_qc.png`

Conservative mask statistics:

- rows: 34
- ignore ratio:
  - min: `0.0000`
  - median: `0.0204`
  - mean: `0.0246`
  - max: `0.1153`
- interpretation: conservative enough for a first masked-loss experiment

Smoke training:

- unit: `hamlabi-region-v2-mask-smoke.service`
- manifest:
  - `dataset/regions_hamlabi_final_review_v2_768_masked_line_conservative/manifest.csv`
- mask key: `valid_mask_path`
- model: `cleanup`
- image size: 480
- epochs: 5
- checkpoint dir:
  - `checkpoints/hamlabi_region_v2_mask_smoke/`
- log:
  - `logs/hamlabi_region_v2_mask_smoke.log`
- result:
  - completed on CUDA
  - loss decreased from `G=0.7031` to `G=0.5642`

Conclusion:

- the idea is implementable
- black-fill mismatch masking needs special handling and should not be enabled
  by default
- the safer first version is target-line unsupported-region masking:
  rough extra construction lines are left as input noise, while unsupported
  target line regions are excluded from loss

### 2026-07-24: Region Post-Alignment for Residual Slide

Reason:

- visually matching rough/line crops can still be shifted after normalization
- supervised rough-to-line training assumes pixel correspondence
- residual slide can encourage gray averaging, double lines, and misplaced
  target lines

Added:

- `tools/pair_extraction/post_align_region_manifest.py`
  - reads a region manifest
  - normalizes rough/line with square-pad or stretch policy
  - searches small residual rough translation against the line edge map
  - writes `aligned_rough_path`, `aligned_line_path`, `align_dx`,
    `align_dy`, and alignment score fields
  - writes `post_align_qc.png`
- `lineart/region_dataset.py`
  - now prefers `aligned_rough_path` / `aligned_line_path` before v2/final
    source paths
- `tools/pair_extraction/build_region_valid_masks.py`
  - now also prefers aligned paths when present

Generated:

- exploratory 24 px max-shift alignment:
  - `dataset/regions_hamlabi_final_review_v2_768_postalign/`
  - shifted: 32 / 34
  - mean score gain: `0.3418`
  - note: several rows hit the 24 px boundary; useful as a diagnostic but
    potentially too aggressive
- conservative 12 px max-shift alignment:
  - `dataset/regions_hamlabi_final_review_v2_768_postalign12/`
  - shifted: 32 / 34
  - mean score gain: `0.2386`
  - QC: `dataset/regions_hamlabi_final_review_v2_768_postalign12/post_align_qc.png`

Mask after conservative post-align:

- dataset:
  - `dataset/regions_hamlabi_final_review_v2_768_postalign12_masked_line_conservative/`
- QC:
  - `dataset/regions_hamlabi_final_review_v2_768_postalign12_masked_line_conservative/valid_mask_qc.png`
- ignore ratio comparison:
  - before post-align: mean `0.0246`, median `0.0204`, max `0.1153`
  - after 12 px post-align: mean `0.0200`, median `0.0138`, max `0.1106`
  - after 24 px post-align: mean `0.0191`, median `0.0136`, max `0.1120`

Smoke training:

- unit: `hamlabi-region-v2-postalign-mask-smoke.service`
- manifest:
  - `dataset/regions_hamlabi_final_review_v2_768_postalign12_masked_line_conservative/manifest.csv`
- model: `cleanup`
- mask key: `valid_mask_path`
- image size: 480
- epochs: 5
- result:
  - completed on CUDA
  - loss decreased from `G=0.7013` to `G=0.5629`

Policy:

- use conservative post-align before masked-loss training
- boundary-hitting shifts are review signals
- if a region needs different shifts in different subareas, keep it as a parent
  and search child regions rather than forcing one translation
