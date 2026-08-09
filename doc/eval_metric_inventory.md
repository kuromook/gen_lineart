# Evaluation Metric Inventory (2026-08-09)

Stocktaking pass, triggered by the 2026-08-09 workstream C finding that
chamfer-to-GT does not measure conditioning fidelity (see `doc/work_log.md`,
"Workstream C Resolved"). Goal: classify every metric currently in use by
*what comparison it actually performs*, so future sessions don't reuse a
metric for a question it wasn't built to answer.

## The core distinction

Two structurally different questions get asked in this project, and they
need different metrics:

1. **"Does this generated output resemble finished line art / land close to
   GT?"** -- point-distance-to-reference metrics (chamfer, F1@tolerance) are
   fine for this, with the caveat below.
2. **"Did this output actually follow its input (rough / ControlNet
   conditioning), as opposed to hallucinating unrelated but plausible-looking
   content?"** -- point-distance-to-GT metrics answer this badly once output
   ink density is high, because dense-enough output makes nearest-GT-edge
   distance saturate low regardless of whether the *content* corresponds
   (validated 2026-08-09: chamfer-to-GT agreed with a hand-built per-tile
   fidelity ranking on only 1/6 decisive tiles; a conditioning-roundtrip
   metric agreed on 5/6 -- see `results/eval_metric_calibration_20260809/`).

A third family sidesteps both point-distance mechanisms entirely by
measuring an image's own structural statistics (line width, component
fragmentation, blank-cell distribution) with no reference-matching step at
all -- these are not exposed to the density-saturation failure mode because
there's no nearest-point search in the first place.

## Inventory

| script | comparison basis | question it answers | 2026-08-09 verdict |
|---|---|---|---|
| `tools/evaluation/evaluate_fixed_outputs.py` | point-distance (chamfer, `f1_2px`, edge extraction fixed 2026-08-09) **and** one-to-one bipartite-matched F1 (`bsds_f1`, added 2026-08-09 later) vs GT | "how close does output land to GT" | ink-extraction bug fixed 2026-08-09 (see below). **`bsds_f1` reaches 6/6 agreement with the hand ranking -- the best result of any metric tried, cheapest to run (no roundtrip/preprocessor needed).** Plain `chamfer_px`/`f1_2px` stayed at 1/6 even after the ink-extraction fix -- they answer "resemblance to GT," not "did it follow conditioning"; use `bsds_f1` for the latter now. |
| `tools/evaluation/condition_roundtrip_fidelity.py` | output's own re-extracted conditioning vs original conditioning (rough-derived), no GT involved. Three comparison functions computed side by side: `roundtrip_chamfer` (5/6), `roundtrip_ssim` (6/6, added 2026-08-09 later, matches ControlNet++'s established choice for edge/line-art conditions), `roundtrip_bsds_f1` (6/6, added 2026-08-09 later) | "did the output follow its conditioning" | **validated 2026-08-09**; `roundtrip_ssim`/`roundtrip_bsds_f1` are the current best (6/6). Right choice specifically when no GT exists yet. |
| `tools/pair_extraction/tile_region_manifest_480.orientation_similarity` | GF-HOG-inspired per-cell gradient-orientation-histogram intersection, output vs rough directly (no GT, no preprocessor) | "does local stroke direction correspond to input, independent of position" | **added + validated 2026-08-09 later**, 5/6 agreement. Not yet wired into a CLI script (function only) -- targets content-independent repeating-texture hallucination specifically (narrow orientation histogram), a different mechanism than point-matching or SSIM. |
| `tools/evaluation/measure_lineart_profile.py` | no reference matching -- single-image structural axes (line width, component density, blank-cell fraction, grid heterogeneity), compared as *distributions* against a real-line-art reference set | "does this look like real line art, structurally" | **not implicated** -- different mechanism, no point-distance step. Already the basis of the domain-LoRA fidelity-budget policy ([[project_diffusion_fidelity_budget]]). Correct tool for domain-LoRA style-fidelity questions. |
| `tools/evaluation/evaluate_stroke_stability.py` | no reference matching -- skeletonized component/fragmentation stats on the prediction alone (GT stats computed only for side-by-side reference) | "are strokes continuous or fragmented (wobble)" | **not implicated** -- same reason as above. Correct tool for "did training make strokes more confident vs more stable" questions. |
| `tools/evaluation/evaluate_halo_outputs.py` | distance-banded intensity averaging (ink density in fixed rings around GT ink: core/near-halo/far-background), not nearest-point chamfer | "is there soft gray smearing (halo) around real lines" | different mechanism, not re-examined this session (GAN/Direction-era artifact, not currently active in the diffusion branch's open questions). Likely fine but unverified against a hand ranking. |
| `tools/pair_extraction/*.py` (`match_koma_panels`, `match_hamlabi_regions`, `match_kurip_regions`, `tile_region_manifest_480`, etc.) | point-distance chamfer, but between two *real* scans (rough vs line of the same physical page) during dataset construction/alignment gating | "are these two crops actually the same region, well-aligned" | different question entirely -- no generative hallucination risk since both sides are real ink, so the density-mismatch confound that broke model-eval chamfer is much less likely here. Extensively empirically calibrated already (multiple `--max-chamfer` threshold tuning passes recorded across `doc/work_log.md`). Not re-examined this session; lower priority since it's a different comparison context, not because it's known-safe by the same argument as the structural-profile family. |
| `tools/evaluation/score_pair_agreement.py` | same as above (rough vs line chamfer, dataset curation) | "how well do this rough/line pair agree, for splitting high/low-agreement training data" | same as above -- different context, not re-examined. |
| `tools/evaluation/evaluate_gate_voting.py` | CNN classifier accuracy against labeled dataset-gate targets | "is the dataset-source-routing classifier accurate" | unrelated question (classification accuracy, not line-art fidelity) -- not implicated. |
| `tools/evaluation/compare_halo_amplification.py` | diffs two halo-metric CSVs (from `evaluate_halo_outputs.py`) sample-by-sample | downstream comparison tool, not a metric itself | n/a |
| `audit_pair_dataset_integrity.py`, `audit_split_leakage.py`, `build_atari_halo_diagnostic_set.py`, `build_clean_eval_set.py`, `filter_pair_list.py`, `transform_aux_strength.py`, `write_experiment_handoff.py`, `data_check.py` | dataset bookkeeping / list construction, no fidelity computation | n/a | not fidelity metrics, skip |

## Visual evidence

- `results/eval_metric_calibration_20260809/tile_fidelity_ranking.csv` --
  the hand-built per-tile fidelity verdicts (10 tiles) that the agreement
  numbers above are computed from, with a one-line reason per tile.
- `results/eval_metric_calibration_20260809/metric_agreement_summary.md` --
  the full agreement table across all 8 metric variants tried
  (chamfer-to-GT 1/6 up to bsds_f1/roundtrip_ssim/roundtrip_bsds_f1 6/6)
  plus the open implication about the 2026-08-08 LoRA-drop decision.
- `results/eval_metric_calibration_20260809/metric_fix_visual_check_montage.png`
  -- rough / bad_hallucination / good_structural / GT montage for all 10
  diagnostic tiles, with `evaluate_fixed_outputs.py` numbers (chamfer /
  f1@2px / bsds_f1 / ink×) printed under each model column so the metric
  can be checked against the image directly. Generated via:
  ```
  ./venv/bin/python tools/compare/make_multi_model_eval_compare.py \
    --sample-list dataset/pairs_480/diag_controlnet_same_coordinate_10.txt \
    --split train \
    --model "bad_hallucination=results/public_controlnet_lineart_anime_preprocessed_20260808" \
    --model "good_structural=results/public_controlnet_noLora_full_20260808" \
    --annotate-metrics \
    --output results/eval_metric_calibration_20260809/metric_fix_visual_check_montage.png
  ```
  (`--annotate-metrics` is a new flag on that script, added 2026-08-09.)

## Literature check (2026-08-09)

A background literature survey confirmed the density-saturation weakness
found today is a known, decades-old phenomenon in boundary detection
(Pratt's Figure of Merit -> BSDS precision-recall, early 2000s), still
active research territory in 2026 (MatchED, CVPR 2026). It also found that
our `condition_roundtrip_fidelity.py` framework independently converged on
the same design as ControlNet's own eval and ControlNet++'s formalized
"controllability" metric -- but ControlNet++ uses **SSIM** for edge/line-art
conditions specifically, not Chamfer. Full writeup, sources, and a
prioritized prototype list: `doc/eval_metric_literature_survey_20260809.md`.

Follow-ups from this survey, all **implemented and validated 2026-08-09
(later)** -- see the inventory table above and `results/
eval_metric_calibration_20260809/metric_agreement_summary.md` for the full
per-metric agreement numbers:
- ~~Add BSDS-style one-to-one bipartite-matched precision/recall/F~~ --
  done (`tile_region_manifest_480.bipartite_match_f1`, via
  `scipy.sparse.csgraph.maximum_bipartite_matching` on a KD-tree-restricted
  candidate graph -- a maximum-cardinality-matching simplification of
  BSDS's exact min-cost LP formulation, documented as such in the
  function's docstring). Reached 6/6 against GT directly in
  `evaluate_fixed_outputs.py` (`bsds_f1`) -- the best result of any metric
  tried, and it didn't even need the roundtrip framework.
- ~~Swap/add SSIM as `condition_roundtrip_fidelity.py`'s comparison
  function~~ -- done (`roundtrip_ssim`, `skimage.metrics.
  structural_similarity` on the raw grayscale conditioning maps before
  binarization). 6/6.
- Also added a GF-HOG-inspired orientation-histogram similarity function
  (`tile_region_manifest_480.orientation_similarity`) as a secondary,
  mechanistically-different check aimed at repeating-texture hallucination
  specifically. 5/6, function only (not yet wired into a CLI script).
- `evaluate_fixed_outputs.py`'s `f1_2px`/`precision_2px`/`recall_2px` are
  named like BSDS-family F-scores but use a materially weaker many-to-one
  match -- readers should not import BSDS-equivalent intuition from them;
  now documented in the script's own docstring and kept alongside (not
  replaced by) `bsds_f1`/`bsds_precision`/`bsds_recall` for continuity with
  historical CSVs.

## Open follow-ups

- ~~`evaluate_fixed_outputs.py`'s ink extraction~~ -- **done 2026-08-09**, see
  table above.
- ~~Literature-survey follow-ups (BSDS matching, SSIM roundtrip, GF-HOG
  descriptor)~~ -- **done 2026-08-09 (later)**, see "Literature check"
  above.
- `orientation_similarity` is a validated function but has no CLI wrapper
  yet -- wire it into a script if it needs to run routinely rather than
  ad hoc.
- `evaluate_halo_outputs.py` has not been checked against a hand ranking the
  way chamfer-to-GT and the roundtrip metric were; worth doing if halo-style
  GAN artifacts become relevant again (currently dormant, diffusion branch
  doesn't use the GAN-era pix2pix models this was built for).
- The `pair_extraction`/dataset-curation chamfer family (rough-vs-line
  alignment gating) was not re-examined against a hand ranking this session.
  It's a different comparison context (real-vs-real, not generated-vs-real)
  so the specific failure mode found today may not transfer, but that's an
  assumption, not a verified conclusion.
- See "Literature check" above for the newer, more specific follow-up list
  (bipartite matching, SSIM roundtrip, GF-HOG orientation descriptor).
