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
