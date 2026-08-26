# Results Layout

Updated: 2026-08-26 JST

## Policy (revised 2026-08-26)

**Old policy** (2026-07-25 through 2026-08-09): keep anything referenced
by exact filename from a current doc, on the theory that even a settled
experiment's image evidence might have residual value later.

**New policy, in effect now**: that theory was wrong in practice. By
2026-08-26 `results/` had grown to 166 top-level items / 1.6GB, browsing
it to find anything current cost real time, and the overwhelming majority
of the bulk was visual evidence for *already-settled, already-written-down*
findings from architecture-survey work (the GAN-era Direction 5/6/8/9
family, badrough/halo/router ablations, the old Direction 4 from-scratch
ControlNet attempts, superseded domain-LoRA isolation-chain intermediates,
etc.) that nobody was going back to re-examine visually. The written
conclusion in `doc/architecture_decisions.md` / `doc/model_directions.md`
/ `doc/model_results_summary.md` / `doc/work_log.md` is what actually
carries forward; the image/checkpoint evidence behind a *settled* finding
does not need to be kept "just in case."

**Current rule**: once a finding is settled and written down, delete its
supporting `results/` artifacts (montages, per-sample outputs, metric
CSVs) rather than preserving them by default. Do not wait for a
scheduled cleanup pass -- delete at the point the verdict is reached, in
the same session. Exceptions, kept deliberately:

- artifacts that are still a **functional dependency** of an active
  training/eval script (e.g. a caption CSV a runner script's
  `--caption-csv` argument points at) -- these aren't "evidence," they're
  inputs;
- the sample/reference output for the **currently-adopted** config of
  each active model family (so "what does the adopted model actually
  produce" stays checkable without a rerun);
- calibration data a still-used *methodology* depends on (e.g. the hand-
  built fidelity ranking behind the `bsds_f1` metric's validation);
- raw dataset provenance records (manifest/tile CSVs -- small, and the
  only record of exactly which files are in a training pool) for
  already-extracted, already-in-use raw sources.

A 2026-08-26 pass applying this rule took `results/` from 166 items/1.6GB
to 43 items/79MB. See `doc/work_log.md`'s 2026-08-26 entry for the exact
deletion list and rationale per category.

`config/results_manifest.json` (an unmaintained lightweight index that had
not tracked most of `results/`'s actual contents for some time) was
deleted rather than revived -- keeping a second, hand-maintained index of
`results/` in sync was itself part of the clutter problem, not a solution
to it. There is no results-manifest tooling to update going forward.

## What's Currently There

- **Active `diffusion`-branch work** (2026-08-04 onward): the domain-LoRA
  adopted-config reference samples
  (`domain_lora_{line,rough}_sd15base_sksv2_20260807_scale14/`), the
  `bsds_f1` calibration basis (`eval_metric_calibration_20260809/`), and
  everything from the current `clip_pairs` koma-pipeline run and the
  real-pairs ControlNet LoRA fine-tune (`clip_pairs_koma_*`,
  `controlnet_lora_realpairs_20260824*`).
- **Per-source raw-dataset provenance** (`results/<source>/` for
  `ako5ver2`/`fitness`/`hamlabi`/`gakuen`/`housei`/`fighting`): trimmed to
  just the manifest/tile CSVs (which tiles are in the training pool) --
  all QC/overlay images deleted 2026-08-26, the review they supported is
  long since concluded.
- `results/lessons/`: trimmed to its CSVs only (2026-08-26) -- the
  stroke-continuity direct-regression finding it supports is fully
  written up in `doc/architecture_decisions.md`'s "単段直接回帰" section;
  the montage images that made the case visually are no longer kept.
- `results/archive/`: old/historical, audit-only, untouched by this pass
  -- see `doc/README.md` for the standing rule not to read it without
  being asked.
- `lineart_profile_*.csv`: kept per the exception above -- these are the
  quantitative trail behind the domain-LoRA fidelity-budget decisions in
  `doc/diffusion_fidelity_budget_policy.md`.

## Going Forward

When a `results/` artifact's finding gets written down as settled (adopted,
rejected, shelved -- any final verdict), delete the artifact in the same
edit that records the verdict, unless it matches one of the four
exceptions above. Do not defer this to a later cleanup pass.
