# Halo Loss Work Log

## 2026-07-20

Goal:

- directly attack halo/faint gray output distribution after agreement split
  showed that high rough-line agreement improves structure but does not remove
  halo by itself

Current baseline context:

- `lucy_mild` and `lucy_thin` are both viable halo-mitigation candidates
- `agreement_halo_e2_high_agreement` improves F1/chamfer but overproduces ink
- low rough-line agreement clearly worsens faint/stipple failure, but high
  agreement is not a complete halo solution

Implementation:

- modified `scripts/train_i2i_survey.py`
  - added `halo_band_loss`
  - added `faint_gray_loss`
  - added CLI:
    - `--halo-weight`
    - `--halo-inner-kernel`
    - `--halo-outer-kernel`
    - `--faint-weight`
- added `experiments/run_halo_loss_survey.sh`
  - runs from shared main artifact root `/home/sh1/deepl/lineart`
  - uses halo-loss worktree code from `/home/sh1/deepl/lineart-halo-loss`
  - compares:
    - `mild_halo06`
    - `mild_halo10_faint04`
    - `high_halo10_faint04`
  - evaluates both normal fixed metrics and halo-specific metrics

Expected run:

- tag: `halo_loss_e2`
- epochs: 2
- expected montage: `results/compare_halo_loss_e2.png`
- expected metrics: `results/fixed_output_metrics_halo_loss_e2_compare.csv`
- expected halo metrics: `results/halo_metrics_halo_loss_e2_compare.csv`
- expected done marker: `logs/halo_loss_e2.done`

Result:

- run completed and wrote `logs/halo_loss_e2.done`
- montage: `/home/sh1/deepl/lineart/results/compare_halo_loss_e2.png`
- fixed metrics:
  `/home/sh1/deepl/lineart/results/fixed_output_metrics_halo_loss_e2_compare.csv`
- halo metrics:
  `/home/sh1/deepl/lineart/results/halo_metrics_halo_loss_e2_compare.csv`

Summary metrics:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `lucy_thin` | 0.4095 | 4.536 | 1.252 |
| `agreement_halo_e2_high_agreement` | 0.4363 | 4.271 | 2.128 |
| `halo_loss_e2_mild_halo06` | 0.3695 | 4.927 | 0.814 |
| `halo_loss_e2_mild_halo10_faint04` | 0.3458 | 5.144 | 0.679 |
| `halo_loss_e2_high_halo10_faint04` | 0.4258 | 4.402 | 1.657 |

Interpretation:

- `mild_halo06` and `mild_halo10_faint04` reduce ink too broadly and lose
  recall
- `high_halo10_faint04` is the best halo-loss variant, but still looks like an
  ink-budget tradeoff rather than a clean halo mechanism fix
- halo-band loss as currently defined penalizes legitimate nearby line width
  as well as halo
- faint-gray metrics increase in the full-list variants, so the loss may be
  converting halo into pale residue instead of removing its cause

Updated priority:

- pause direct halo-removal optimization as the primary goal
- focus next on identifying which factor creates halo:
  - atari generation blur/deconvolution
  - aux conditioning strength
  - GAN/reconstruction loss balance
  - target mismatch and alignment
  - architecture receptive field and skip behavior
  - thresholding or post-filter behavior
