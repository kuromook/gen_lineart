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
closed. Successors are two worktrees, each with its own briefing in
`inbox/initial_notice.md`:

- `../lineart-controlnet-sd15-refine` (branch `controlnet-sd15-refine`) --
  refine from the best config; the remaining gap is gray background/gray lines.
- `../lineart-controlnet-sdxl-fidelity` (branch `controlnet-sdxl-fidelity`) --
  SDXL does not cross-hatch but diverges from the rough. Note the SDXL runs so
  far were trained *and* inferred at 512 on a 1024-native base, so that
  divergence is not yet a fair reading.

Proposal with both directions: `doc/track_proposal_20260906.md`.

Both tracks stay on the **v2-based** pair snapshot copied into their own
`data/` (`train_list.txt`, 8,467 rows). User decision 2026-09-06: the
difference against the newer 8,798-tile v3 pool is not large enough to be
worth a re-baseline. Do not migrate them to v3 without a fresh decision.
The closed track's full work log is `doc/track_controlnet_realpairs_work_log.md`
on branch `controlnet-realpairs` (not present in this working tree).

**Cause, and two lessons that apply project-wide.** The cause was not on the
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

## Active Goal

Make ControlNet-based rough-to-line conversion actually follow the rough. Work
happens in the two successor worktrees named above, not here; this tree is the
common foundation (shared scripts, dataset pipeline, project-level docs).

- SD1.5: close the remaining gray-background/gray-line gap from the cs3.5 best
  config -- `../lineart-controlnet-sd15-refine`.
- SDXL: re-baseline at 1024, then re-measure condition fidelity --
  `../lineart-controlnet-sdxl-fidelity`.

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
`line_width_p50` and `ink_ratio` alongside it -- see the two lessons at the top
of this file.

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

Items 1-2 are the active direction and are carried out in the successor
worktrees, each of which has its own briefing and work log. Items 3-6 are
common-foundation housekeeping and open questions, none of them blocking.

1. **SD1.5 refinement** (`../lineart-controlnet-sd15-refine`): sweep the
   consistency-loss hyperparameters -- only a single point
   (`consistency_weight=0.1`, `consistency_max_timestep=200`) has been tried,
   and the effect is confirmed real. Then consider extending it to all
   timesteps, InnerControl-style (arxiv 2507.02321). Do **not** add a UNet-side
   LoRA, raise the LoRA rank, or forbid hatching by negative prompt -- all
   three were measured and are worse; the first also breaks the
   conditioning-scale lever itself.
2. **SDXL re-baseline** (`../lineart-controlnet-sdxl-fidelity`): retrain and
   re-infer at 1024. Every SDXL run so far used the SD1.5-era `--resolution`
   default of 512 on a 1024-native base, so the "SDXL diverges from the rough"
   reading is on hold until this is redone. Whether 1024 SDXL ControlNet LoRA
   fits in 12GB VRAM is itself unverified.
3. The unpaired-rough adversarial-branch idea is **dormant, not to be picked
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
4. Decide whether to rename the remaining `kurip`-named infra scripts, given
   `kurip` was a username (`match_kurip_regions.py` and others;
   `prepare_kurip_tiles.py` affects hamlabi too). Still open, unrelated to
   the work above.
5. Decide whether umbrella/layer-difference rows (ako5ver2) should be
   manually masked, tagged for future routing, or left held out. Still open.
6. Revisit whether `--max-soft-ink-ratio` needs a per-source
   `diagnose_gate_funnel.py` pass for ako5ver2/hamlabi/fitness/gakuen (only
   housei has an established relaxed value so far); yield may be
   conservative for the others under the shared default. Still open.
