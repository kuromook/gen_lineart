# Session Memory

Updated: 2026-06-11

## Current Objective

Pause model development and mine additional manuscripts for high-confidence
rough/line pairs before starting Mixture of Experts (MoE) experiments.

## Current State

- `shape1` is the adopted practical model.
  - checkpoint: `checkpoints/shape1/best.pth`
  - fixed-sample F1@2px: 0.9364
  - chamfer: 0.757
  - ink ratio to ground truth: 1.208
  - recall: 0.9809
- shape1 reduced thick/extra lines enough to enter a practical range.
- Adding clean-line details absent from the rough is explicitly out of scope.
- High-confidence ako5 region extraction is complete.
- `dataset_480/valid_train_ako5_regions.txt`: 418 saved pairs.
- `dataset_480/valid_train_warm_regions.txt`: 1,078 pairs.
  - 660 original non-ako5 std15 pairs
  - 418 new `ako5r_` region pairs
  - no old broad `ako5_` pairs
- Conservative extraction defaults:
  - region score >= 4.0
  - tile score >= 3.0
  - line ink 1.5% to 15%
  - edge F1 >= 0.35
  - chamfer <= 8.0

## Pair Mining Result

- ako5 yielded about 418 high-confidence pairs from about 50 pages.
- Practical yield estimate: about 8 pairs per manuscript page.
- The major result is that usable pairs can be extracted from uncertain,
  inconsistent manuscripts containing page mismatch, edits, deletions, and
  local deformation.
- The extraction pipeline is ready to be applied to other manuscripts.

## Next Theme: MoE

- Do not start full MoE training with the current 1,078 pairs.
- First mine more manuscripts and inspect whether natural expert clusters
  emerge from input characteristics.
- Rough planning targets:
  - 3 experts: minimum about 6,000 pairs / 750 pages; safer 900-1,200 pages
  - 4 experts: minimum about 8,000 pairs / 1,000 pages; safer 1,200-1,600 pages
  - practical general target: 10,000+ pairs / about 1,250 pages at ako5 yield
- Actual yield may differ by manuscript. Process an additional 100-200 pages
  first and recalculate yield and QC precision before committing to the full
  page target.

## Repository State

- Latest pushed commit: `b9e39f3 Add practical shape1 model and evaluation`
- `origin/main` includes shape1 checkpoints, comparison, outputs, metrics, and
  pair-extraction code.
- No training service is active.

## Resume Actions

1. Select the next 100-200 manuscript pages for mining.
2. Adapt/run the ako5 page and local-region matching pipeline.
3. Measure accepted pairs per page, false-pair rate, and input-feature
   distribution.
4. Continue mining toward several thousand pairs.
5. Only then define expert categories and begin a 2-expert pilot before
   considering 3-4 experts.

Detailed history is in `work_log.md`.
