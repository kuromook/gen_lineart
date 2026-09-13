# Current Project State

Updated: 2026-09-06 JST (ControlNet track pointer, Active Goal, and Next
Actions rewritten. The descriptive sections in between -- Current Data
Direction, Current Model Interpretation, Current Extraction Rules, Current
Data Pipeline Stage, and the dated 2026-07-26/31 entries -- are still from
the raw-extraction era and have not been re-verified.)

This file is the first document to read. It should contain only active state,
current decisions, and next actions. Chronological details live in
`doc/work_log.md`; reusable extraction knowledge lives in
`doc/preprocess/raw_dataset_extraction_knowledge.md`.

Do not read files under `archive/` directories unless the user explicitly asks
for archived history or audit material.

## ControlNet Cross-Hatch Track: Closed 2026-09-06

The `lineart-controlnet-realpairs` track (ControlNet LoRA fine-tunes
hallucinating dense cross-hatch instead of clean line art) met its goal and is
closed. Three worktrees have descended from it, each with its own briefing in
`doc/initial_notice.md` (one of them already closed in turn):

- `../lineart-controlnet-sd15-refine` (branch `controlnet-sd15-refine`) --
  **CLOSED 2026-09-13**, same verdict as the SDXL track: the diffusion model
  does not beat the preprocessor. Re-measured on 192 tiles with Track B's
  protocol, the gap is **wider** than five tiles suggested -- `manga_line`
  alone scores 0.2847, against 0.2514 for round 1's best (`w=0.2`, -0.0333)
  and 0.2524 for round 2's (`w=0.4`, -0.0323), where the five-tile read had
  been -0.021. The reinterpretation is the part worth keeping: **the
  preprocessor already had near_white_frac 0.931** against GT's 0.924, so four
  sweeps spent lifting near_white from 0.400 into the 0.800s were climbing
  back toward what the conditioning map started with and never overtook it.
  The grey residual was not a gap the model was filling -- it was degradation
  the model introduced. It proposes no successor and folds into
  stroke-selection. Notices:
  `inbox/note_sd15_consistency_weight_result_20260910.md`,
  `inbox/note_track_a_closing_20260913.md`.
- `../lineart-controlnet-sdxl-fidelity` (branch `controlnet-sdxl-fidelity`) --
  **CLOSED 2026-09-11.** Its question is answered in the opposite direction
  from the premise: SDXL does not diverge from the rough, it copies its
  conditioning at f1 0.88. Fidelity was never the problem. What it established
  instead: **no model this project has trained beats a preprocessor run
  alone** (lineart_coarse 0.2639 > SDXL bare 0.2615 > manga_line 0.2566 >
  SD1.5 consistency 0.2354 > SDXL ft 0.1539, same five tiles), and decomposing
  that gap shows the residual is **deletion** on the lineart pool (the
  preprocessor lays 1.4x GT's ink) and **solid fills** on housei (its
  fill_ratio is 0.0% against GT's 24.5%). A delete-only oracle reaches
  **0.7425** against the preprocessor's 0.3231. Successor is the
  stroke-selection worktree below.
  Notices: `inbox/note_sdxl_correction_preprocessor_20260911.md`,
  `inbox/note_contamination_check_both_tracks_20260911.md`,
  `inbox/note_track_b_closing_and_selection_proposal_20260911.md`.
  Pool inventory (every source profiled, with the new fill_ratio):
  `../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md`.
- `../lineart-stroke-selection` (branch `stroke-selection`) -- **NEW
  2026-09-11, the current active direction.** Can the deletion be learned? Input is the preprocessor output
  rather than the raw rough, the label comes straight from the pair data (did
  this stroke match GT), and the ceiling is 0.74 against a current best near
  0.30. First move is to look at the oracle before trusting it. Proposal:
  `doc/track_proposal_stroke_selection_20260911.md`.
- `../lineart-pair-signal` (branch `pair-signal`) -- **NEW 2026-09-13,
  diagnostic.** Why did 8,467 pairs contribute nothing to any fine-tune? Five
  hypotheses ordered by cost; the first and cheapest has never been measured at
  all -- latent diffusion can only express its target through the VAE, and GT
  line art is white paper under 1-3px strokes, which is what a VAE handles
  worst. If the VAE cannot round-trip GT, no amount of training reaches it and
  that single fact explains every result above. Briefing:
  `../lineart-pair-signal/doc/initial_notice.md`.

Proposal with both directions: `doc/track_proposal_20260906.md`.

The ControlNet tracks stay on the **v2-based** pair snapshot copied into their
own `data/` (`train_list.txt`, 8,467 rows). User decision 2026-09-06: the
difference against the newer 8,798-tile v3 pool is not large enough to be
worth a re-baseline. Do not migrate them to v3 without a fresh decision.
The closed track's full work log is `doc/track_controlnet_realpairs_work_log.md`
on branch `controlnet-realpairs` (not present in this working tree).

**Cause, and six lessons that apply project-wide** (lessons 3-4 added
2026-09-06 from `../lineart-controlnet-sdxl-fidelity`, lesson 5 on 2026-09-10
from both tracks, lesson 6 on 2026-09-11; see the notices in `inbox/`). The cause was not on the
training side: six hypotheses (data pool, LoRA rank, epochs, an x0-vs-GT
consistency loss, caption vocabulary, a UNet-side LoRA) were each measured and
rejected. The base UNet is frozen in every ControlNet run, so its hatch prior
could never be trained away; raising `controlnet_conditioning_scale` to
overpower it moved gt_bsds_f1 0.1411 -> 0.2337 with no retraining.

1. **Do not compare ControlNet models at `controlnet_conditioning_scale=1.0`
   alone.** All eleven models bunch at f1 0.13-0.15 there because the
   hallucination dominates; re-measuring at each model's best scale reshuffled
   the ranking substantially. Sweep several scales.
2. **Do not judge on `orientation_entropy` alone.** It cannot separate a hatch
   mesh from the smooth boundary of a solid fill. Report `line_width_p50`
   (GT ~3.7) and `ink_ratio` (GT ~0.035) with it -- one model scored f1 0.2101
   while actually being a solid-fill blob at `line_width_p50` 40.92.
3. **Do not judge on `gt_bsds_f1` alone either -- it cannot see whether there
   is white paper under the ink.** It only asks whether strokes land near GT
   strokes. Two measured cases scored well while not being line art at all: a
   uniformly grey image took the best f1 of its sweep (0.2263) with 3.1% of
   pixels near white, and a nearly blank page took 0.2121 with 87.8% near
   white and no subject drawn. Report `bg_mode` (GT 255), `near_white_frac`
   (GT 94.8%) and `midtone_frac` (GT 1.8%) beside it. And `near_white_frac`
   is not self-sufficient either -- it cannot tell "white because it is clean"
   from "white because nothing was drawn", so read it with `line_width_p50`
   and the montage. **Landed here 2026-09-06** as `paper_profile()` in
   `tools/evaluation/measure_lineart_profile.py` (lifted from
   `paper_metrics()` in the SDXL track's
   `experiments/score_resolution_sweep_20260906.py`, thresholds kept
   identical so the numbers stay comparable with that sweep). Verified on 200
   GT tiles from the v3 pool: bg_mode 255, near_white_frac 0.951,
   midtone_frac 0.014 -- matching the GT anchors above. This is directly the
   axis `../lineart-controlnet-sd15-refine` needs -- its stated residual is
   "grey background, greyish lines".
4. **Resolution and conditioning scale interact; sweeping one alone can hide
   the effect entirely.** For the bare SDXL ControlNet, going 512 -> 1024 at
   cs1.0 is flat (0.2263 -> 0.2341), but at cs2.0 it improves monotonically
   (0.2416 -> 0.2568). Paper white behaves the same way: it appears only where
   1024 and cs2.0 meet (near_white 3.0% -> 81.5%). Keep isolating one variable
   at a time as the default, but when an interaction is plausible, run the
   grid -- "level the field at 1024" alone would have shown nothing here.
5. **"Does fine-tuning help?" is the wrong question; the training signal's
   design is the question.** Read together, the two tracks look contradictory
   and are not. On SDXL, three epsilon-MSE runs each made the bare ControlNet
   worse -- fine-tuning is not merely useless there, it is harmful. On SD1.5,
   fine-tuning with an auxiliary term added on top of epsilon-MSE
   (`scripts/train_controlnet_consistency.py`: decode the x0 estimate through
   the VAE, take an L1 on Sobel-edge agreement with the GT image) is what
   produced the new best config. The difference is not the architecture and
   not whether one trains, but **what is compared, against what, at which
   timesteps**. Note also the shape of the SDXL failure: the objective kept
   improving while the output got worse on every axis a human cares about --
   the same "optimize one indicator, lose line-art-ness" pattern this project
   keeps rediscovering, now with a loss curve that looked healthy throughout.

6. **Report `gt_bsds_f1` against the conditioning map's own score, or the
   number cannot be read.** On a ControlNet conditioned by a line
   preprocessor, f1-against-GT largely measures how faithfully the output
   copied that preprocessor -- so a model that does nothing scores best. The
   SDXL track's headline collapsed on exactly this: the conditioning map
   alone scores 0.3177 against GT, the bare ControlNet 0.3027, and
   distance-from-conditioning tracks f1 monotonically (0.88 -> 0.3027, 0.84
   -> 0.2961, 0.26 -> 0.1820). The baseline is cheap -- no inference, just
   score the conditioning images you already have. **This contamination has
   not been checked on the SD1.5 track**, whose 0.2354 is in the same
   position; it may well survive, since `consistency_weight` moved
   near_white 0.400 -> 0.779, which copying the conditioning cannot explain,
   and that would make it the project's first demonstrated case of the model
   contributing rather than the preprocessor.
   A companion metric was added for the same reason on 2026-09-11:
   `profile_metrics` now returns **`fill_ratio`** (share of ink in strokes
   thicker than 8px) beside `line_width_p50`, because width alone cannot
   separate a thick stroke from a solid fill -- the confusion that made this
   project read housei GT's line_width_p50 of 7.59 as "thick deliberate
   strokes" when it was measuring solid blacks. Calibrated against known
   cases: coarse_trained, the identified fill-escape model, 91.6%; a clean
   line-art model 7.1%; GT tiles 3.3%. A fill-excluded "corrected width" was
   tried and rejected -- it collapsed to ~1.9 everywhere, since what remains
   after removing fills is their own thin fringes.
   A second reading rule from the same review: **do not average across the
   `lineart` and `housei` pools.** They are different tasks, not different
   sources -- 5.9% vs 21.9% of GT ink is solid fill, 1% vs 38% of tiles are
   near-blank -- and the failure inverts between them (grey paper on one,
   an inability to lay solid fills on the other). Score them separately;
   `dataset/pairs_480/holdout_lineart_family.txt` and
   `holdout_housei_100.txt` are already split that way.

## Active Goal

**Close the gap between the preprocessor's output and GT.** Work happens in the
worktrees named above, not here; this tree is the common foundation (shared
scripts, dataset pipeline, project-level docs).

This replaces the goal that stood here until 2026-09-11, "make ControlNet-based
rough-to-line conversion actually follow the rough". That framing was wrong in
both halves. Fidelity was never the problem -- the SDXL stack copies its
conditioning at f1 0.88 -- and the baseline to beat was never another model:
**no model this project has trained beats running a preprocessor alone.** The
thing that already solves most of the task is `LineartDetector`, and the open
question is the residual it leaves.

That residual is measured, and it is two different problems by pool:

- **Deletion**, on the lineart-family pool: the preprocessor lays 1.4x GT's
  ink, so strokes must be removed. On the 192-tile `lineart_family` group, a
  delete-only oracle reaches f1 **0.7425** where `lineart_coarse` alone scores
  0.3231 and the best model this project ever trained scores 0.2514.
  This is `../lineart-stroke-selection`, and it is the active direction.
- **Solid fills**, on the housei/ako5 pools: the preprocessor's `fill_ratio` is
  0.0% against GT's 24.5%, because an edge detector structurally cannot fill.
  Over 12,000 tiles of this type have never been trained on or evaluated.
  **Deferred by user decision 2026-09-13** -- not dropped, just not now.

One finding from the now-closed `../lineart-controlnet-sd15-refine` survives
its closure and is still the only one of its kind here: its consistency loss
moved the output *away* from the conditioning map while moving it *toward* GT
(vs-conditioning 0.5009 -> 0.4618 as f1 rose 0.2175 -> 0.2354), the opposite of
the SDXL stack, which simply copied its conditioning. That mechanism did work.
It just never carried the output past the preprocessor -- on 192 tiles it
finished 0.032 short. **A mechanism can be real and still not be worth
keeping**, and that is the distinction to hold on to: the evidence is against
diffusion generation closing this gap, not against that particular loss doing
what it was designed to do.

Running alongside the deletion work is a **diagnosis**, not another attempt at
the gap: `../lineart-pair-signal` (branch `pair-signal`, opened 2026-09-13).
Every fine-tune this project has run on its 8,467 pairs either degraded the
result or improved mechanically without beating a preprocessor run. That the
pairs contributed nothing is settled; **why** has never been investigated, and
the answer decides what the pair data is still good for. It is not only a
post-mortem of the closed tracks: **Track C's labels come from the same pairs**
(did this preprocessor stroke match GT), so if the pairs are the problem, the
deletion work inherits it -- and if the problem is latent diffusion as a
vehicle, a pixel-space discriminative model never touches it. Getting at least
its first answer before Track C trains a baseline is the point.

**The raw-extraction goal that stood here through 2026-08 is done.** The
clip_pairs v3 re-extraction, the 8,798-tile combined pool
(`dataset/pairs_480/valid_train_combined_v3_20260830.txt`), its WD14 captions,
and its `lineart_anime` conditioning are all in place and verified 1:1 as of
2026-08-30. What remains on the data side is cleanup and a few open questions,
listed under Next Actions -- not an active build-out. The sections below this
one still describe that era and should be read as reference, not as current
direction.

Old leak-era `shape1` scores are not adoption targets. Use clean eval metrics
and montage review only as current references. Evaluate line art with the
BSDS-style one-to-one matching F1 (`gt_bsds_f1`), and always report
`line_width_p50`, `ink_ratio`, `fill_ratio` and the conditioning map's own
score alongside it -- see the six lessons at the top of this file.

## Current Data Direction

The immediate focus is raw-dataset expansion through region-matched,
variable-aspect extraction rather than same-coordinate 480px tiling.

Current ako5ver2 review target:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/`

Current filtered ako5ver2 manifest:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv`

Current ako5ver2 review status:

- source rows: 291
- excluded user-reviewed mismatches: 7
- held umbrella / mask-insufficient layer-difference rows: 3
- kept rows: 281

Important ako5ver2 review files:

- `removed_user_mismatch.csv`
- `held_user_mask_insufficient_umbrella.csv`
- `flagged_user_review_20260725.csv`
- `user_review_20260725_summary.txt`

Dataset-specific note:

- ako5ver2 has rough-only umbrella cases where rough contains an umbrella but
  the line art omits it, likely due to 3D, separate layer, or later compositing.
  These are documented in `doc/preprocess/raw_dataset_extraction_knowledge.md` and should
  not be trained as normal pairs under the current mask.

## Current Model Interpretation

The best recent single-model candidates remain halo-mitigation / Lucy-hint
variants, but none is a final production line-art model:

- `lucy_mild_aux_msgan`: safer balanced candidate
- `lucy_thin_aux_msgan`: higher-recall candidate requiring artifact scrutiny
- `dog_aux_msgan`: controlled white/hint candidate
- `flowdog_aux_msgan`: high-recall / high-ink expert candidate

Router/MoE oracle was useful as an upper-bound probe on clean eval, but it is
not a deployed router.

Initial line-field refiner ran successfully but overproduced ink and needs
stronger ink/width control or a revised formulation before deeper use.

## Current Extraction Rules

Use:

- `doc/preprocess/EXTRACTION_RULES.md` for procedure and gates
- `doc/preprocess/region_dataset_extraction_policy.md` for variable-region extraction
- `doc/preprocess/region_materialization_policy.md` for manifest/materialization policy
- `doc/preprocess/raw_dataset_extraction_knowledge.md` for dataset-specific exceptions

Current rules to preserve:

- do not promote same-XY crops as production data without region/content
  matching
- keep candidate generation separate from acceptance
- review QC before training
- use variable-aspect manifests first where possible
- materialize fixed-size square-padded copies only when a downstream tool needs
  them
- keep layer/prop differences as hold/tag cases unless masks explicitly make
  them safe

## Documentation State

`doc/work_log.md` is approaching the 5,000-line maintenance threshold.

Use:

- `doc/documentation_maintenance_policy.md`

Current maintenance plan:

- do not automatically delete or archive `work_log.md` sections without user
  review
- create reviewable compaction proposals first
- extract reusable knowledge into focused docs
- only then archive old chronological detail

## Current Data Pipeline Stage

The 768 px long-side normalization bottleneck identified during stroke-scale
filter design has been resolved by re-materializing keep281 near native source
resolution. See `doc/preprocess/raw_dataset_extraction_knowledge.md` for the scale
measurement and `doc/preprocess/region_dataset_extraction_policy.md` for the scale-band
and tile-score policy notes.

Current native pipeline artifacts:

- materialized regions: `dataset/regions_ako5ver2_native_20260725/` (165 of
  281 keep281 regions; 116 dropped as below 480 px native)
- masked regions: `dataset/regions_ako5ver2_native_20260725_masked_line_conservative/`
- strict tile candidates (post score cutoff):
  `results/ako5ver2_native_tiles_480_strict_cut25.csv`

A full-resolution review of the first strict pass (before the score cutoff)
found the tail contained at least one tile that passed every individual gate
but showed unrelated rough/line content. This is recorded as a standing policy
note: tile score is not a content-match guarantee, and thumbnail QC hides it.

Applying `--min-tile-score 2.5` removed that failure mode: 422 tiles from 112
regions, re-checked at full resolution with no remaining wild mismatches (only
sparse/faint tail tiles, correct semantic correspondence but loose alignment,
reviewed and accepted by the user). For scale, the earlier 768-normalized
strict subset was 90 tiles from 45 regions.

User approved proceeding to `--save` and a training pass. Saved, integrity
audit passed (0 findings), and trained
`ako5ver2_native_strict_cut25_480_warm_clean_bce_e10` (10-epoch BCE-heavy
warmstart, same recipe as prior keep281 control runs).

Result: loss decreased monotonically (0.3048 to 0.2617), but visual output
reproduces the same soft / density-map texture seen in every earlier
keep281-derived run under this recipe family. This isolates the remaining gap
to the model/recipe side, not the data pipeline: the native re-materialization
and stroke-scale tile filter are considered validated.

## 2026-07-26 Additional Native-Strict Sources: fitness, housei

Applied the same reviewed native-strict pipeline (region matching via
`match_kurip_regions.py` + strict filter via
`tools/pair_extraction/filter_matched_region_tiles.py`) to two more raw
sources, following the same "extraction methodology is validated, next work is
model-side" framing above. Goal: broaden the training pool for future
model-side experiments, not to reopen data-pipeline research.

- `fitness` (renamed from `kurip`, which was a person's username): 271 tiles
  from 37 regions, 0 integrity findings.
- `housei`: 25 tiles from 10 regions, 0 integrity findings. Small; the ceiling
  is the strict content-quality gates, not the tile-score cutoff.
- `fighting` (renamed from `lineart` earlier this session): 40 tiles, already
  recorded above.

Full details, routes, and rename rationale: `doc/preprocess/dataset_status.md` and
`doc/preprocess/raw_dataset_extraction_knowledge.md`.

Two overnight autonomous agents were assigned `fitness` and `housei`
originally; both were lost mid-task (their transcripts became unrecoverable,
likely from an environment restart) and their work was picked up and completed
directly. This surfaced an environment issue: long-running background
extraction jobs get silently killed around 10-13 minutes regardless of
execution method, with no traceback. Recorded as a standing operational note
in `doc/preprocess/raw_dataset_extraction_knowledge.md`; the practical workaround is
chunked `--offset`/`--limit`/`--append` runs, now supported directly in
`filter_matched_region_tiles.py`.

Per explicit user decision, all pre-session leak-era `kurip`-named data,
checkpoints, and one-off comparison scripts were deleted outright (not
renamed) during the `fitness` rename, since that material was not needed.
5 still-active infra scripts (`match_kurip_regions.py` and 4 others) still
carry the old name; renaming those is a separate, larger decision left open
(one of them, `prepare_kurip_tiles.py`, is shared with hamlabi).

None of `fitness`/`housei`/`fighting` have been trained on yet.

## 2026-07-26 Alignment Investigation: Root-Caused To Scale/Deformation

Combined-source training (ako5ver2 native + fitness + housei + fighting, 798
tiles) surfaced a clear per-source quality gradient in output crispness
(fighting best, ako5ver2 worst) that tracked chamfer distance almost exactly.
A post-hoc `chamfer<=12` re-filter (217 tiles) did not visibly improve output
in a direct same-sample comparison against the unfiltered pool, despite the
metric correlation holding — see `doc/preprocess/region_dataset_extraction_policy.md`
("Alignment Gate vs Style Gate") for the resulting architectural rule: keep
alignment gates (chamfer, strict-tolerance edge correspondence) as fixed
cross-dataset constants (`ALIGNMENT_*` in `tile_region_manifest_480.py`),
separate from style gates (ink/gray/width/black-fill), which stay per-source
tunable. `analyze_tile()` was refactored into `alignment_metrics()` +
`alignment_gate_pass()` / `style_metrics()` + `style_gate_pass()` to enforce
this structurally; regression-checked against prior saved results (identical
counts).

Root cause of why the chamfer-only refilter didn't visibly help: residual
misalignment is not just imprecise translation search. User's production-
process explanation, confirmed by test: line art is inked from a printed
rough with no production need to keep it pixel-aligned, and the finished line
art goes through a finishing pass that rescales/repositions content per panel,
per character, or occasionally a smaller partial region. A joint
translation+scale search on fitness's 5 worst-chamfer tiles cut chamfer by
25-38%, and 3 of 5 picked a non-1.0 scale — confirming real scale mismatch
that no current tool corrects for (`match_kurip_regions.py` is
translation-only; `match_hamlabi_regions.py` only tries a few discrete global
scales per parent region, not per-panel/per-character).

Decided out of scope for now: local mesh-level (non-uniform) deformation —
too open-ended to model generally, revisit if a good general method appears.

Planned fix (paused, blocked on external work): panel border lines are
composited from a separate layer in the original production file and are
absent from the finished line art layer itself, so panel boundaries cannot be
recovered from the flattened rough/line images alone (a first attempt using
long-line morphology detection on a real ako5ver2 page failed, flagging
character hair as false panel borders). User will extract the panel-border
layer as its own dataset on another machine. Once available, planned staged
approach: (1) segment pages into clean single-panel regions using that layer,
(2) verify alignment per panel (translation + one uniform scale expected, no
mesh deformation, so more tractable than whole-page matching), (3) split
further into per-character regions within a panel if multiple characters are
present, re-scoring alignment per character with only low-scoring cases
needing manual review, (4) defer finer sub-character regions, which may have
irregular/"special" deformation. Full detail:
`doc/preprocess/raw_dataset_extraction_knowledge.md` ("Residual Misalignment").

## 2026-07-26 (later) housei Koma Panel Segmentation And Tile Extraction

The panel-border-layer extraction unblocked for `housei` (delivered as
`dataset/raw_zips/dataset_housei_v2.zip`; ako5ver2/hamlabi still pending on
the other machine). Built `tools/pair_extraction/match_koma_panels.py`
(panel detection from the koma layer + per-panel translation/scale alignment
search) and ran it across all 18 housei pages: 81 panels, chamfer median
17.24 -> 14.55. Found and resolved two anomalies on review: `housei_004`
excluded (true page-level asset mismatch, rough is a ラフ layout sketch not
下絵, confirmed by user); `housei_010`/`011`/`012` flagged as a distinct
high-residual-misalignment cluster (real content correspondence, much higher
chamfer; ruled out contrast/faintness as the cause via a 4-method test).

Per user direction, filtered at panel granularity rather than page
granularity: a `chamfer <= 20.0` gate (a real distribution gap, not a fitted
elbow) kept 59/74 non-housei_004 panels, excluding housei_010/011/012's
panels specifically while keeping every other page's panels including the
otherwise-good pages that happen to contain those 3. Materialized
(`tools/pair_extraction/materialize_koma_panels.py`) and ran through the
existing native mask + strict-tile pipeline (`build_region_valid_masks.py`,
`tile_region_manifest_480.py`) unchanged: 58 tiles from 287 raw candidates,
0 integrity findings. Saved as
`dataset/pairs_480/valid_train_housei_koma_native_strict_20260726.txt`. Full
detail: `doc/preprocess/raw_dataset_extraction_knowledge.md` (`## housei`) and
`doc/preprocess/dataset_status.md` (`## housei` -> "Koma Panel Segmentation").

## 2026-07-26 (later still) housei_004 Fixed, Sub-Region Split, Yield Improved

User supplied `dataset_housei_v3.zip` (corrected `housei_004_sketch.jpg`,
confirmed the only changed file vs v2) plus, separately, `dataset_ako5_koma.zip`
and `dataset_hamlabi_koma.zip` (koma layers for the other two sources,
integrity-checked OK, not yet processed — user said proceed with housei first).

Re-ran housei_004 alone against v3: now passes cleanly (chamfer 12.3-18.6, was
30-40/excluded). Diagnosed the 58-tile yield as low via a per-gate funnel
measurement: `ink_range` rejected 80.9% of candidates (koma panels are
panel-border geometry, not content density, so much of a panel is blank),
alignment only 0.7%. Built `tools/pair_extraction/split_koma_panel_subregions.py`
(reuses hamlabi's page-level ink-connected-component region proposal, scoped
to one already-aligned panel) to crop dense content islands before tiling.
Result: 66 panels -> 179 sub-regions -> 75 tiles (up from 58), 0 integrity
findings. This supersedes the earlier 58-tile panel-level-only set. Full
detail: `doc/preprocess/raw_dataset_extraction_knowledge.md` (`## housei`).

## 2026-07-26 (even later) Per-Sub-Region Alignment Refinement

User's follow-up observation from looking at the sub-region tile QC
directly: content matches panel-to-panel, but zoomed in there's still
noticeable misalignment — asked how much character/region-level realignment
within one panel would help. Added `--refine-alignment` to
`split_koma_panel_subregions.py`: a small local translation+scale search per
sub-region, starting from the panel's own alignment (already roughly right)
rather than a wide from-scratch search. Ran across all 179 sub-regions in 3
chunks (chamfer improved ~8-11% per chunk; some individual sub-regions
needed a real correction, e.g. one 32px shift cut chamfer 19.1->16.3).
Re-tiled through the unchanged mask+tile pipeline: **85 tiles** (up from 75
unrefined, up from 58 at the original whole-panel level), 0 integrity
findings. This is now the current housei koma-panel training source,
superseding both earlier sets (left on disk, not deleted). Full progression:
58 -> 75 -> 85 tiles across whole-panel -> sub-region-split ->
+alignment-refinement. Full detail: `doc/preprocess/raw_dataset_extraction_knowledge.md`
(`## housei`).

## 2026-07-31 Koma Extraction Complete; Model Architecture Survey (6 -> 8 -> 9), Then Reconsider Direction 4

All 5 koma-pipeline sources (ako5ver2/fitness/gakuen/hamlabi/housei) are fully
extracted and combined: `dataset/pairs_480/valid_train_combined_koma_20260729.txt`
(1489 tiles: ako5ver2koma 536 / fitnesskoma 455 / gakuenkoma 202 /
hamlabikoma 164 / houseikoma 132), visually QC'd and already used for
training. There is no remaining raw-extraction backlog for these 5 sources —
the items below about "koma layers arrived but not processed yet" are
resolved and were left in this file well past their relevance; see
`doc/work_log.md` ("2026-07-29 (later still): Combined 5-Source Koma
Training Launch") for how this finished.

Current model-side status (see `doc/model_directions.md`): Direction 5
(shallow residual cleanup refiner) was tried in two forms on the combined
koma dataset — bidirectional (`cleanup`, `combined_koma_lucy_mild_msgan_20260729`)
and darkening-only (`cleanupdark`, `combined_koma_cleanupdark_20260730`) — both
converged to the same soft/marbled-gray F1@2px~0.40-0.43 / chamfer~4.5-5.6
ceiling. Directions 1/2/3/7 (multi-scale PatchGAN, feature matching,
structure/perceptual loss, soft width/skeleton loss) were also already tried
in some form pre-koma with no clear jump past that same ceiling.

**Decided plan:** try the remaining untested architecture directions in
order — Direction 6 (confidence/thickness dual-head, `dualhead` model,
in progress as of 2026-07-31 as `combined_koma_dualhead_20260731`), then
Direction 8 (HED/DexiNed-style multi-scale edge head), then Direction 9
(attention/Swin-like refiner block) — each a small delta on the existing
CNN+GAN pipeline, evaluated on the same 8-sample clean eval montage. Only
after those three, reconsider Direction 4 (diffusion/ControlNet-style
refinement), previously deferred for cost/data reasons.

In parallel with the 6/8/9 survey, the user is preparing new raw source
material (additional manuscript pages from the same artist, on a separate
machine) to grow the koma dataset beyond 1489 tiles. This is not a blocking
prerequisite for Direction 4 (the 5 existing sources are all the same
artist with some style variation, not different artists, so augmentation of
the current pool was judged a reasonably good fit for that narrower
generalization target) — it is opportunistic growth to do alongside the
architecture survey, revisited once Direction 4 is actually reached.

## 2026-07-31 (later) Direction 6/8/9 Survey Concluded; Unpaired-Rough Tested; Switching To A Direction 4 Branch

Direction 6 (confidence/thickness dual-head) and Direction 8 (HED-style
multi-scale side outputs) both initially failed (chronically under-inked)
because the new generators reconstructed ink from scratch with no anchor to
the aux/atari input, unlike the adopted `cleanup`/`cleanupdark` models
(`out = aux_logits + bounded_correction`). Fixed both to use the same
residual-anchor pattern and retrained; Direction 9 (bottleneck
self-attention) was implemented with the fix applied from the start.
**Result: Directions 5, 6, 8, and 9 all converge to the same soft/marbled
F1@2px ~0.40-0.42 ceiling (or below it, when undertrained) — no
architecture in this short survey produced a qualitative jump.**
`combined_koma_lucy_mild_msgan_20260729` (`cleanup` model) remains the
adopted best checkpoint. Full detail and numbers: `doc/model_directions.md`
(Directions 5/6/8/9 "Result" notes) and `doc/work_log.md` ("2026-07-31:
Direction 6/8/9 Survey Concluded").

Also prepared and tested the first unpaired-rough pool (`skima`, 626
pencil-only manuscript pages with no line-art counterpart, tiled to 4917
rough-only tiles at `dataset/unpaired_rough/skima/`) via a new
adversarial-only training branch (`--unpaired-weight` in
`scripts/train_i2i_survey.py`). Not adopted at either weight tried (0.03:
clear regression with a qualitatively different fragmented/binary failure
mode; 0.003: negligible effect, ~reproduces baseline) — a next step (not
yet attempted) would add a GT-free continuity regularizer to the unpaired
branch itself. Full detail: `doc/work_log.md`.

Explored Direction 4 (diffusion/ControlNet) feasibility: confirmed local
SD1.5-family checkpoints exist (`~/disk/checkpoint/Stable-diffusion/`,
anime-tuned merges preferred over plain SD1.5), installed
`diffusers`/`transformers`/`accelerate`/`peft` into the project venv, and
confirmed `ControlNetModel.from_unet()` builds correctly from a locally
loaded checkpoint. Full ControlNet training script not yet implemented.

**Decision: Direction 4 moves to its own branch**, since it is
architecturally unrelated to the CNN+GAN refiner family developed on
`cleanup-refiner`. This branch's architecture-survey work is considered
closed out as of this commit.

## Next Actions

Items 1-2 are the active work, each in its own worktree with its own briefing
and work log: item 1 tries to close the gap, item 2 diagnoses why every attempt
so far failed to. Item 3 is a deferred strategic question; items 4-7 are
common-foundation housekeeping, none of them blocking. Both ControlNet tracks
are closed as of 2026-09-13 and neither leaves work behind -- see the pointer
section above.

1. **Stroke selection** (`../lineart-stroke-selection`): learn to delete the
   preprocessor's spurious strokes. Input is the preprocessor output, not the
   raw rough; the label comes straight from the pair data (did this stroke
   match GT); on the 192-tile `lineart_family` group the ceiling is 0.7425
   against a best-ever trained score of 0.2514. First
   move, per its own briefing, is to **look at the oracle output before
   trusting the number** -- this project has been misled by a metric three
   times (orientation_entropy alone, gt_bsds_f1 alone, near_white_frac alone),
   and 0.74 is an oracle that consults GT, so it is an upper bound by
   construction, not an achievable score. Note also that the oracle's recall
   is capped at 0.604 because the preprocessor never finds the other 40% of
   GT's strokes, so the choice of preprocessor should be revisited on ceiling
   (recall) rather than on its own standalone f1 -- `lineart_coarse` scores
   best alone (0.2639) but that is a different criterion.
   Proposal: `doc/track_proposal_stroke_selection_20260911.md`.

   Both ControlNet tracks that preceded it are closed, and their results are in
   the pointer section above. One question they left open is now **answered:
   do not port the consistency loss to SDXL.** That call was explicitly gated
   on Track A's round-2 sweep, and the sweep came back worse than the figure it
   was waiting on -- -0.032 to -0.033 on 192 tiles against the -0.021 the five
   tiles had shown. Porting a loss that finishes below its own conditioning map
   on the cheaper architecture, onto the one that merely copies its
   conditioning, has nothing to recommend it.

   Track A also measured a ceiling this track needs: `manga_line`'s delete-only
   oracle reaches **0.5143** (recall capped at 0.3462) against
   `lineart_coarse`'s **0.7425** (recall 0.604), on the same `lineart_family`
   192-tile group, so the two are directly comparable. `manga_line` is the
   weaker basis for a selection approach.
2. **Pair-signal diagnosis** (`../lineart-pair-signal`): why did the pair data
   contribute nothing? Run in this order, and **stop and report after the
   first** -- it may settle everything. (a) **VAE round-trip ceiling**: encode
   and decode GT tiles and score the result against the GT they came from, on
   `gt_bsds_f1` and the paper axes, for both SD1.5 and SDXL. Inference only,
   minutes. `tools/evaluation/condition_roundtrip_fidelity.py` does not cover
   this -- it measures conditioning maps, not the VAE -- so it needs new code.
   If the round-trip scores near the current best of ~0.25, the ceiling was
   never reachable and nothing else needs explaining. (b) **Loss-versus-quality
   correlation**, using checkpoints that already exist, no training: Track B
   watched its loss fall the whole way down while every axis a human cares
   about got worse; whether that is systematic decorrelation or an accident is
   unknown. (c) Only then, pair correspondence quality -- which is the
   hypothesis that would also implicate Track C's labels.
   Briefing: `../lineart-pair-signal/doc/initial_notice.md`.
3. **Solid fills (the housei/ako5 pools): deferred, by user decision
   2026-09-13.** Not dropped -- the question was put and answered "not now".
   Recorded here so it stays visible rather than becoming a silent omission.
   Over 12,000 tiles across `ako5` and `housei` have never been trained on and
   appear in no evaluation set. They are not a harder version of the current
   task but a different one: GT there is 24.5% solid fill against the lineart
   pool's 4.0%, 27-38% of tiles are near-blank, and the preprocessor fills
   nothing at all (fill_ratio 0.0%), so an edge-detector-plus-selection
   pipeline cannot reach it by construction. The delete-only oracle tops out at
   0.3291 there against 0.7425 on the lineart pool. The natural moment to
   reopen it is when stroke selection has a real number on the lineart pool:
   that is what decides whether this is the other half of the plan or a
   separate project. Inventory:
   `../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md`.
4. The unpaired-rough adversarial-branch idea is **dormant, not to be picked
   up for now** (user decision 2026-09-06). It belongs to the shelved CNN+GAN
   line (`scripts/train_i2i_survey.py`, the `cleanup`/msgan family), so acting
   on it would mean returning to an architecture this project moved off. The
   remaining move, if it is ever resumed, is: add a GT-free
   continuity/self-consistency regularizer to the unpaired branch, then
   re-sweep `--unpaired-weight` between 0.003 (no effect) and 0.03
   (destructive: F1@2px 0.4175 -> 0.2381, as fragmented high-contrast
   stippling). Note the `skima` rough-only pool itself (626 pages -> 4,917
   tiles, `dataset/pairs_480/train/rough_unpaired_skima/`) still exists and
   may be worth using in the ControlNet context instead -- that would be a
   new idea, not this one.
5. Decide whether to rename the remaining `kurip`-named infra scripts, given
   `kurip` was a username (`match_kurip_regions.py` and others;
   `prepare_kurip_tiles.py` affects hamlabi too). Still open, unrelated to
   the work above.
6. Decide whether umbrella/layer-difference rows (ako5ver2) should be
   manually masked, tagged for future routing, or left held out. Still open.
7. Revisit whether `--max-soft-ink-ratio` needs a per-source
   `diagnose_gate_funnel.py` pass for ako5ver2/hamlabi/fitness/gakuen (only
   housei has an established relaxed value so far); yield may be
   conservative for the others under the shared default. Still open.
