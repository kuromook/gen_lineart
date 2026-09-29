# Current Project State

Updated: 2026-09-30 JST. The Track Ledger, Lessons, Known Tool Traps, Active
Goal and Next Actions are current as of this date; the ledger was rebuilt by
reading every track's own files after this file had fallen 13 days behind. The
descriptive sections further down -- Current Data Direction, Current Model
Interpretation, Current Extraction Rules, Current Data Pipeline Stage, and the
dated 2026-07-26/31 entries -- are still from the raw-extraction era and have
not been re-verified.

This file is the first document to read. It should contain only active state,
current decisions, and next actions. Chronological details live in
`doc/work_log.md`; reusable extraction knowledge lives in
`doc/preprocess/raw_dataset_extraction_knowledge.md`.

Do not read files under `archive/` directories unless the user explicitly asks
for archived history or audit material.

## Track Ledger

Reconciled 2026-09-30 by reading every track's own files, after this file had
fallen 13 days behind. Each track keeps its own briefing in
`doc/initial_notice.md` and its own `doc/work_log.md`; this ledger records only
status and where each got to.

**Where the tracks live -- this is why the ledger went stale.** Tracks A-F are
git worktrees of `/home/sh1/deepl/lineart`, so `git worktree list` finds them.
Tracks **G and H are not**: `/home/sh1/deepl/lineart-panel-generation` is a
**separate clone** of the same remote (`gen_lineart`), and
`/home/sh1/deepl/lineart-face-words` is a worktree of *that* clone. They push
branches to the same GitHub remote but are invisible to `git worktree list` run
here. Check `git branch -r` or the sibling directories, not just the worktree
list.

The lineage forked in two on 2026-09-17, after Track D's diagnosis (lesson 8)
closed the pixel-matching generative route:

- **the selection line** -- Track C, delete what the preprocessor over-draws;
- **the words line** -- Tracks F -> G -> H, treat strokes as a vocabulary and
  build up from representation. This is where all activity since 2026-09-21 is.

### The closed ControlNet lineage

The `lineart-controlnet-realpairs` track (ControlNet LoRA fine-tunes
hallucinating dense cross-hatch instead of clean line art) met its goal and is
closed, as are both of its successors.

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
- `../lineart-stroke-selection` (branch `stroke-selection`) -- opened
  2026-09-11. Can the deletion be learned? Input is the preprocessor output
  rather than the raw rough, the label comes straight from the pair data (did
  this stroke match GT), and the ceiling is 0.74 against a current best near
  0.30. Proposal: `doc/track_proposal_stroke_selection_20260911.md`.
  **Progress as of 2026-09-17**: the oracle survived its visual check (f1
  0.6765, precision 0.9948 -- dashed, but the surviving points trace GT's
  lines, so neither an over-aggressive oracle nor misaligned pairs), and a
  minimal pixel-level keep/drop baseline has been trained and evaluated.
  **A structural constraint has since been established, twice:** the
  preprocessor's output does not decompose into strokes. Its skeleton is short
  fragments joined by a dense junction mesh (262 tokens per tile against GT's
  27, 5.1% junction pixels against 1.5%), so **deciding keep/drop on cut
  skeleton segments cannot reach pixel-level selection** -- 0.301 against
  0.552 in one controlled comparison, and cutting finer (40/20/10/5px) only
  recovers to 0.162. This track had already found the same thing at
  connected-component granularity. The "just cut it finer" route is closed.
  Notices: `inbox/note_oracle_visual_check_pairs_look_sound_20260913.md`,
  `inbox/note_stroke_fit_is_measurable_but_weak_20260917.md`.
- `../lineart-pair-signal` (branch `pair-signal`) -- **CLOSED 2026-09-17,
  diagnosis complete.** Why did 8,467 pairs contribute nothing to any
  fine-tune? Five hypotheses, four of them eliminated by measurement:
  - **VAE ceiling: refuted.** GT round-trips at f1 **0.9605** (SD1.5) and
    **0.9806** (SDXL) on lineart_family, against a best trained model of 0.25
    and a delete-oracle ceiling of 0.74. The margin for error is wide too --
    adding isotropic noise at 40% of signal strength to the latent costs only
    0.048.
  - **Objective is nearly flat: confirmed, and sharpened.** Almost the whole
    loss drop happens in the first 1,000 steps; after that it moves 1.2% while
    f1 moves 18% and paper white goes 0.13 to 0.83. The loss is not unrelated
    to quality -- deltas correlate at Spearman -0.73 -- it simply has almost
    no gradient left along the axes that matter. See lesson 7.
  - **Loose pair correspondence: real but not the cause.** A third of GT's
    stroke length sits 3-8px off even in the raw rough, but training on the
    1,837 best-aligned pairs against a matched random control changed nothing
    that matters: strokes drawn where the conditioning map has none stayed at
    0.104 vs 0.113. What improved was only fidelity to the conditioning map.
  - **Scale: refuted.** 460 subset of 1,837 subset of 8,467, everything else
    identical: `gt_only` reads 0.046 / 0.045 / 0.045, flat across an 18x range,
    and the smallest arm scores highest at the final step.
  - **What remains is the objective's expressiveness** -- see lesson 8, which
    is the finding worth carrying forward from this whole track.
  Briefing: `../lineart-pair-signal/doc/initial_notice.md`. Notices:
  `inbox/note_vae_ceiling_refuted_20260914.md`,
  `inbox/note_loss_blind_to_quality_20260915.md`,
  `inbox/note_stroke_anchoring_and_pair_offset_20260915.md`,
  `inbox/note_h2_decided_flat_objective_20260915.md`,
  `inbox/note_training_pairs_alignment_manga_line_empty_20260915.md`,
  `inbox/note_h34_aligned_pairs_do_not_teach_placement_20260915.md`,
  `inbox/note_manga_line_emptiness_scale_and_contrast_20260916.md`,
  `inbox/note_hypothesis5_scale_refuted_20260917.md`.

### The selection line

- `../lineart-stroke-selection` (branch `stroke-selection`, **Track C**) --
  **active, idle since 2026-09-17.** Not superseded by the words line; it
  simply has not been picked up since its first real result. Its bullet above
  under the ControlNet lineage carries the oracle and decomposition findings.
  **Step 2 completed 2026-09-17**: a 16,201-parameter pixel-level keep/drop
  classifier, trained 3 epochs on all 8,467 pairs (~9 minutes), scored on the
  same group-A 192-tile batch as the oracle. f1 **0.3161** against the
  preprocessor's 0.3164 -- flat -- but the composition is not flat at all:
  **precision +0.068** (0.2373 -> 0.3057) and **recall -0.169** (0.5255 ->
  0.3566). It is deleting, and over-deleting; the montage confirms it has
  stopped simply echoing the conditioning map. Ceiling is 0.6765 on that batch.
  Named next variables, to be isolated one at a time: the conservative 0.5
  threshold, epoch count, `pos_weight`. Results:
  `results/stroke_selection_eval_20260917/` in that tree.

### The words line

Opened by an explicit user redefinition on 2026-09-18: **treat a stroke as a
word and a panel as a sentence** -- give strokes features, and take the picture
to be built from their combinations. A framework for this arrived from the user
on 2026-09-21, the **5P/5C**: ten elements said to govern composition
(composition, placement, plane, perspective, proportion, pose, counter=line
quality, character, coodinate=within-work consistency, concept). Three are
implemented as features (composition, plane, proportion); the rest are future
work, and the stated expectation is that the pieces make each other easier.

- `../lineart-stroke-grammar` (branch `stroke-grammar`, **Track F**) --
  **measurement phase closed 2026-09-21, kept readable.** Its result is a clean
  negative and it is the reason the rest of this line is shaped the way it is:
  **there is no inter-word grammar inside a panel.** Five measurements agree --
  autoregressive 6.94 bits/word against a 7.35 unigram (gate 6.85, narrowly
  failed); bigram 7.40, *worse* than unigram, so adjacency in canonical order
  carries nothing; set-cloze 7.398, below unigram; conditioning on 12 shot
  types 7.312 (0.036 bits); conditioning on 40 WD14 tags 7.370 (nothing).
  What it does have: **words are real** (500-word FSQ codebook, 92% used,
  23.2% change under detail removal), **their spatial placement has structure**
  (position prediction beats the 7.56 marginal at 6.31-6.39 under every
  condition), and panel **scene type** clusters into 12 classes a human can
  name. Also the finding already folded in as a Known Tool Trap: the
  preprocessor's output does not decompose into strokes. Notice:
  `inbox/note_stroke_fit_is_measurable_but_weak_20260917.md`. Its earlier
  discriminative framing is archived in that tree at
  `doc/archive/initial_notice_discriminative_20260918.md`.
- `../lineart-panel-generation` (branch `panel-generation`, **Track G**, and
  note the separate clone) -- **active, last commit 2026-09-25.** Given a
  concept, choose words and place them. Approval is recorded in its own
  briefing and work log ("起案・承認された", 2026-09-21); **the copy of its
  proposal still says draft/unapproved**, so that copy is stale, not the
  status. Where it got to, in order:
  - *2026-09-21, generation smoke*: set-at-once vs two-stage. Word bits
    **7.078 / 7.094** passed the pre-registered 7.312 gate; position bits
    6.680 / 6.658 beat the 7.563 marginal but **failed** the 6.388 cloze band,
    and the gate was not relaxed after the fact. **The visual gate failed**:
    the output is "prototypes scattered at statistically plausible positions",
    not a readable scene -- recorded in that tree as 蟻の群れ, an ant swarm.
    Leak checks and a "too good to be true" audit of the 0.662 scale bits were
    run and passed. Neither architecture was selected; the 0.022-bit gap is
    noise.
  - *2026-09-23, user redirection*: **do not start from placement.** While the
    words render as an ant swarm, a human cannot judge "does this read as a
    panel", so word rendering quality comes first. Recorded alongside it, and
    worth carrying: **bits do not guarantee a visual pass** -- position beats
    the marginal by 0.9 bits and the montage is still unreadable.
  - *2026-09-24, word reliability*: of 467 words, **49 reliable / 164
    borderline / 254 unreliable**, stable across seeds (0 flips). The dominant
    cause is that a single detail stroke changes the word. A word is a
    **shape class with scale discarded** -- cluster diameter varies about 6x
    within one word -- and Track F's names are correspondingly polysemous.
  - *2026-09-25*: an attempt to measure absolute vs relative position **failed
    its own instrument check and was halted**, handed to Track H to do first on
    face parts, where the relations are clear.
  - Also found here, and a trap for anyone reading that data: in Track F's
    `cluster_set` meta, **the `cy` column is x and `cx` is y**.
- `../lineart-face-words` (branch `face-words`, **Track H**, worktree of the
  Track G clone) -- **active, the current front of this line, opened
  2026-09-25.** Its proposal records user approval dated 2026-09-25. Pin down
  the face-part words (eye, nose, mouth, brow, ear) by attaching human
  *strategy labels* in a namespace separate from the machine word ids, then
  make a mechanical rule reproduce them. The user's framing is recorded as: the
  more cards you turn face-up, the easier the rest becomes; ears and mouths
  should be easiest because one artist does not vary them much. Methodology
  worth noting -- **held-out labels are assigned mechanically at labelling
  time**, before any result is seen, split two ways (by panel, and by series to
  test whether a rule survives a different artist). Progress: step 1's
  size-stratified montages of the "closed eye" candidates w342/w343/w344 are
  built and the tool checks passed, including an added negative control after
  a suspiciously perfect 1.00 overlay score; **awaiting the user's visual
  judgment.**
- `../lineart-aesthetic-judge` (branch `aesthetic-judge`, **Track E**) --
  **paused 2026-09-17, for a reason that matters more than the pause.** It set
  out to score "line-art-ness" against human judgement, so that places where GT
  and the conditioning map disagree could be settled by preference rather than
  by either one. The comparison UI was built and 300 pairs prepared; the user
  used it and reported that **you cannot call something beautiful when it is
  broken and unreadable** -- that is a broken/uninterpretable axis, not an
  aesthetic one. The material confirms it: of the three candidate sources, only
  the preprocessor output is interpretable line art (midtone fraction 0.072
  against 0.503 and 0.259 for the trained models), so most pairs would have
  been "clean line art vs broken something". **Re-open condition: when two
  interpretable outputs can be compared.** Its diagnosis is what points at the
  words line -- interpretability has to be produced before preference can be
  measured. UI and pairs are kept at `results/comparison_pairs_20260916/`.

Proposals, now held here rather than only in those trees' gitignored `outbox/`:
`doc/track_proposal_aesthetic_judge_20260916.md` (E),
`doc/track_g_generation_proposal_20260921.md` (G),
`doc/track_h_face_words_proposal_20260924.md` (H).

**External review.** Kimi (Kimi Code) read the project docs at the user's
request and sent Track F five recommendations on 2026-09-19; four were adopted
as-is and one was adopted with a correction. Copies:
`doc/external_review_kimi_20260919.md` and
`doc/external_review_kimi_reply_20260919.md`. The substantive points, because
they shaped Track F's design and generalise: negatives in a
selection-style test must be **stratified and hard** (matched on single-stroke
features, drawn from the same panel's other clusters, and stratified by *work*
rather than by corpus, or the model becomes a provenance detector); the
**endpoint-gap** feature must appear as a baseline; the **vocabulary-existence**
test is a precondition, not an optional extra; and **masked/orderless**
modelling suits pictures better than left-to-right autoregression, since
canonical order has to be invented. The correction is itself a lesson: the
0.651 endpoint-gap figure Kimi cited turned out to be **an artefact of the
tokeniser** -- segments cut at junctions end where they touch another stroke,
creating a characteristic ~4px gap -- and fell to 0.54-0.58 once strokes were
joined at panel scale. Baselines must be re-measured in the current unit.

Proposal with both directions: `doc/track_proposal_20260906.md`.

The ControlNet tracks stay on the **v2-based** pair snapshot copied into their
own `data/` (`train_list.txt`, 8,467 rows). User decision 2026-09-06: the
difference against the newer 8,798-tile v3 pool is not large enough to be
worth a re-baseline. Do not migrate them to v3 without a fresh decision.
The closed track's full work log is `doc/track_controlnet_realpairs_work_log.md`
on branch `controlnet-realpairs` (not present in this working tree).

## Lessons

**The cross-hatch cause, and eight lessons that apply project-wide** (lessons 3-4 added
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
7. **Never use training loss as a proxy for quality -- not the logged loss,
   and not a clean one either.** Added 2026-09-15 from
   `../lineart-pair-signal`, which measured both. The *logged* loss is one
   batch at a random `t`, and its between-checkpoint spread equals its own
   standard error, so Track B's "loss fell from 0.0341 to 0.0302" was a change
   indistinguishable from noise. Fixing the measurement does not rescue the
   idea: with `t`, noise and latents all held fixed, the true movement after
   step 1,000 is 0.00039 -- telling 0.0001 apart from single batches would
   need roughly 490,000 batches per checkpoint. Worse, the little that does
   move tracks **paper tone, not line structure** (r = -0.98 against paper
   white in one run; no significant relation to f1). So: **snapshot during
   training and score a holdout at each snapshot.** Enable
   `--eval-snapshot-steps` from the start -- no run before 2026-09-15 ever did,
   which is why diagnosing this needed a fresh training run rather than
   existing checkpoints. Run with `PYTHONUNBUFFERED=1`; logs were reaching disk
   only about every 1,000 steps.
8. **A conditioned generative objective converges on reproducing the
   conditioning map and adjusting tone. It cannot be made to add strokes the
   map lacks, or remove strokes the map has.** Added 2026-09-17; this is the
   central finding of `../lineart-pair-signal` and it explains every negative
   result above at once. Measured three ways: output skeletons sit near the
   conditioning map 36% of the time against 9% near GT alone, and move *toward*
   the map as training proceeds; where the map lacks a GT stroke, the model
   draws it 9.3% of the time; and neither better-aligned pairs (0.104 vs 0.113)
   nor 18x more pairs (0.045 / 0.045 / 0.046) shifts that. **Consequence for
   planning: do not invest in better pairs for generative training.** The pair
   data's value is as supervision for selection, and as the measuring
   instrument it has been all along. The one caveat the track states itself:
   the aligned-pairs arm ran 2,290 steps, so a far longer run cannot be
   strictly excluded, though its pre-registered criteria did not ask for one.

## Known Tool Traps

Collected 2026-09-13..17 from `inbox/`. Every one of these was hit by a track
that had no way to know, and several were hit twice. **Two are unfixed bugs**
-- see Next Actions.

- **`bipartite_match_f1` (the implementation behind `gt_bsds_f1`) can take
  tens of seconds to 12+ minutes on a single tile.** scipy's
  `maximum_bipartite_matching` approaches worst case on particular edge-point
  configurations, and it is not predictable from image density -- similar tiles
  hit it or do not. 98 of 584 pairs (16.8%) timed out in one run. Environment
  causes (cv2 thread contention, sharing a process with PyTorch/CUDA) were
  tested and ruled out. Work around it by dispatching per tile to a subprocess
  under a hard `timeout` and recording a miss:
  `tools/evaluation/vae_roundtrip_score.py`. **If a batch evaluation appears
  hung, suspect one slow tile before suspecting a hang.** The metric itself was
  left unchanged -- it is shared and validated.
- **`evaluate_fixed_outputs.py --split auto` resolves GT paths wrongly for
  168 of the 192 `holdout_lineart_family.txt` tiles.** It decides with
  `"train" if name.startswith("housei") else "test"`, and only the 24
  `lineart_004_*` tiles actually live in test. The default sample list was all
  `lineart_004_*`, which is why nobody noticed. **Unfixed.** Any past 192-tile
  number produced through `--split auto` should be re-checked; Track A's and
  Track B's 192-tile scoring used other scripts and was not audited.
- **The shared `evaluate_stroke_stability.skeletonize()` leaves a 2px-wide
  skeleton.** Cut it into segments at junctions and 64% of the skeleton
  classifies as "junction", shattering lines into dots (only 1.7% of segments
  reach 30px). Use skimage 1px thinning with crossing-number junctions
  instead (2.0% junctions, 70.4% of segments over 30px). Connected-component
  metrics such as `long_component_ratio` do not depend on width and are
  unaffected.
- **Binarising a conditioning map before skeletonising changes results by
  half.** A naive `>32` manufactures false junctions and 1-3px fragments from
  edge jaggies; filling holes first takes skeleton capture from 0.42 to 0.62
  and a measured ceiling from 0.193 to 0.301.
- **`manga_line` is nearly empty on the training pairs.** It holds 12% of GT's
  stroke length there (median 6%) against 30% on holdout, and is essentially
  blank on 19% of training tiles -- so Track A trained largely on "draw line
  art from an empty input", which fits its stroke-adding behaviour. Soft
  pencil, stroke width, blur and upscaling were each measured and eliminated;
  what the images show is extreme close-ups of thick soft graphite, where the
  structure the extractor looks for (thin dark lines on white) does not exist
  at that scale. **A fix is known but not applied**: downscale the rough to
  240px and auto-contrast before the extractor, taking GT agreement from 0.010
  to 0.185 on empty tiles (verified against a 180-degree-rotation chance
  baseline, so it is not amplified paper grain). It does not close the gap to
  holdout's 0.30, and **it must be applied to training and holdout together or
  it merely inverts the train/eval mismatch.** `lineart_coarse` does not have
  this problem (66% on training pairs, 60% on holdout).
- **Do not read "strokes drawn where the conditioning map has none"
  (`gt_only`) on its own.** A degenerate output that covers the whole tile in
  dither scores highest on it; across 15 snapshots it correlates with
  `gt_bsds_f1` at **-0.269**. It also tracks the "strokes the map does have"
  column at r 0.75-0.90 within an arm, so report the normalised form too, and
  always beside f1, `fill_ratio`, `near_white_frac` and a montage.
- **Per-pair alignment scores now exist for all 8,467 training pairs**:
  `../lineart-pair-signal/results/pair_alignment_strata_20260915/per_pair.csv`
  (rough / manga_line / lineart_coarse, each with `le3`, `3to8`, `gt8`,
  `chance_le3`, `aligned`, `density`). Usable for weighting or excluding
  training data. Chance is measured by rotating each conditioning map 180
  degrees; horizontal flip was rejected because panel borders survive it and
  inflate the baseline.

## Active Goal

**Build a representation of line art that a human can read, then generate from
it.** Work happens in the tracks listed above, not here; this tree is the common
foundation (shared scripts, dataset pipeline, project-level docs).

This supersedes the goal that stood here from 2026-09-11 to 2026-09-30, "close
the gap between the preprocessor's output and GT", which was framed entirely
around deleting the strokes the preprocessor over-draws. **That framing is not
refuted -- it is no longer the whole project.** Deletion is still live as Track
C and still has the widest measured headroom of anything here. But every attempt
to reach GT by matching pixels, whether by generating or by selecting, has
landed in the same place: at or just above the preprocessor's own score. On
2026-09-18 the user redefined the problem instead of pushing further on that
axis, and that redefinition is where the work now is.

**What survives unchanged, regardless of route.** These are measurements, not
strategy:

- **The baseline to beat is a preprocessor run, not another model.** No model
  this project has trained beats `LineartDetector` alone. Report the
  conditioning map's own score next to any figure (lesson 6).
- **The residual it leaves is two different problems by pool.** Deletion on the
  lineart family -- the preprocessor lays 1.4x GT's ink, and a delete-only
  oracle reaches f1 **0.7425** against `lineart_coarse`'s 0.3231 on the 192-tile
  group. Solid fills on housei/ako5 -- its `fill_ratio` is 0.0% against GT's
  24.5%, because an edge detector structurally cannot fill; over 12,000 tiles of
  that kind have never been trained on or evaluated, and targeting them is
  **deferred by user decision 2026-09-13**, not dropped.
- **Lessons 1-8 and the tool traps below hold for every route.** Lesson 8 in
  particular is why the generative-pixel-matching route is closed rather than
  merely unpromising.

**The current route, and why it is shaped this way.** Treat a stroke as a word
and a panel as a sentence. Track F then established a clean negative that
determines everything downstream: **there is no inter-word grammar inside a
panel** -- five independent measurements agree, and a bigram model is *worse*
than unigram. So generation cannot be "a grammar consumes a word sequence".
What does exist is a real vocabulary and real structure in **where** words sit.
Hence Track G's shape: the concept is supplied from outside, and the model's job
is only which words appear and where they go.

Track G has already run that and **failed its visual gate** while passing its
numeric ones -- prototypes land at statistically plausible positions and the
panel still does not read. The user's redirection from that result is the thing
to hold on to: **bits do not guarantee a visual pass**, and while the words
render as an ant swarm no human can judge whether a panel reads at all. So the
front of the work moved *upstream*, to making the words themselves legible and
nameable. That is Track H: pin down the face-part words first, because the more
cards are face-up, the easier the rest becomes.

Two things follow that are worth stating plainly, because they are easy to lose:

- **This route has not yet produced a readable panel.** It is currently
  building representation, not generating. Track G's one generation attempt is a
  recorded failure, and 254 of 467 words are unreliable.
- **Track E is the standing check on all of it.** It tried to score
  "line-art-ness" against human preference and stopped because nothing
  interpretable existed to compare -- you cannot ask which of two pictures is
  better when both are broken. Its re-open condition is the honest success
  criterion for the words line: **two interpretable outputs to put side by
  side.**

**The diagnosis that closed the previous route** was `../lineart-pair-signal`,
finished 2026-09-17. The pairs contribute nothing to generative fine-tuning not
because of the VAE, not because of loose correspondence, and not because of
scale, but because the objective converges on reproducing the conditioning map
(lesson 8). Two consequences, pointing opposite ways:

- **Do not invest in better pairs for generative training.** Better-aligned
  pairs and 18x more pairs both changed nothing.
- **The pair data is not devalued -- its role is confirmed.** It is supervision
  for selection, the source of Track H's strategy labels, and the measuring
  instrument every verdict in this file rests on.

One finding from the closed `../lineart-controlnet-sd15-refine` is still the
only one of its kind here and should not be lost: its consistency loss moved the
output *away* from the conditioning map while moving it *toward* GT
(vs-conditioning 0.5009 -> 0.4618 as f1 rose 0.2175 -> 0.2354), unlike the SDXL
stack, which simply copied. The mechanism worked; it just never carried the
output past the preprocessor, finishing 0.032 short on 192 tiles. **A mechanism
can be real and still not be worth keeping.**

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
score alongside it -- see the eight lessons at the top of this file, and
**never the training loss** (lesson 7).

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

Items 1-2 are the active fronts, both on the words line; item 3 is the
selection line, live but idle. Item 4 is two open bugs in shared tooling, item 5
a deferred strategic question, items 6-9 common-foundation housekeeping, none of
them blocking. Every ControlNet track is closed and none leaves work behind --
see the Track Ledger above.

1. **Face-part words** (`../lineart-face-words`, Track H -- the current front).
   Waiting on the user: step 1's size-stratified montages of the closed-eye
   candidates w342/w343/w344 are built and the tool checks passed, and the
   **visual judgement has not been given yet**. That judgement gates the rest:
   whether splitting a word by size separates closed eyes from panel borders and
   hair. After it, pre-register the labelling procedure and the rule's pass
   criteria, then run one full loop (label -> rule -> held-out check) on closed
   eyes before widening to open eyes, brows, nose, mouth, ears. Two
   methodological commitments already recorded there are worth keeping: labels
   live in a **namespace separate from the machine word ids**, and the held-out
   split is **assigned mechanically at labelling time**, before any result is
   seen, both by panel and by series.
2. **Panel generation** (`../lineart-panel-generation`, Track G). Blocked
   upstream by its own choice, and correctly so: it can generate, but the words
   render as an ant swarm, so no visual judgement is possible and placement
   measurement was halted after failing an instrument check. Its own next steps
   are recorded there -- prototype-plus-residual decoding to make words legible
   first, then re-judge, then placement. **Do not restart from placement**
   (user, 2026-09-23). When it resumes, the standing warning from its last run
   is that **bits do not guarantee a visual pass**: position beat the marginal
   by 0.9 bits while the montage stayed unreadable.
3. **Stroke selection** (`../lineart-stroke-selection`, Track C -- live but
   untouched since 2026-09-17). It is not superseded and it still has the widest
   measured headroom in the project: its minimal classifier sits at f1 0.3161
   against a 0.6765 ceiling on the same batch. The first result is genuinely
   informative -- flat f1 but **precision +0.068 and recall -0.169**, i.e. it
   learned to delete and it over-deletes -- so the named next variables are
   concrete: the conservative 0.5 threshold, more epochs, `pos_weight`. Isolate
   one at a time. Two constraints bound the route: per-segment keep/drop cannot
   reach pixel-level selection (measured twice, cutting finer does not rescue
   it), and selection can never add the ~16% of GT stroke length the
   preprocessor never finds, which is what caps oracle recall at 0.604. Lesson 7
   applies even though the objective is discriminative: score a holdout at each
   snapshot rather than watching the loss. Proposal:
   `doc/track_proposal_stroke_selection_20260911.md`.
4. **Fix the two tool bugs found in `inbox/` (both still open).**
   (a) `evaluate_fixed_outputs.py --split auto` mis-resolves GT for 168 of the
   192 `holdout_lineart_family.txt` tiles; it should resolve per tile by
   looking for the file rather than by a `housei` prefix test, the way the
   holdout runner already does. Then re-check any past 192-tile number that
   went through it. (b) `bipartite_match_f1`'s pathological slowness has a
   working mitigation in one track
   (`tools/evaluation/vae_roundtrip_score.py`, per-tile subprocess with a hard
   timeout) but nothing shared -- every track batch-scoring `gt_bsds_f1` needs
   it. Lifting that into the shared evaluation path is the cheap fix; changing
   the metric itself is not on the table, it is validated and comparisons
   depend on it. The unapplied `manga_line` emptiness fix (downscale to 240px
   plus auto-contrast) is a data-side decision, not a bug fix -- and it must be
   applied to training and holdout together or it inverts the mismatch.
   Details for all three: **Known Tool Traps** above.
5. **Solid fills (the housei/ako5 pools): deferred, by user decision
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
6. The unpaired-rough adversarial-branch idea is **dormant, not to be picked
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
7. Decide whether to rename the remaining `kurip`-named infra scripts, given
   `kurip` was a username (`match_kurip_regions.py` and others;
   `prepare_kurip_tiles.py` affects hamlabi too). Still open, unrelated to
   the work above.
8. Decide whether umbrella/layer-difference rows (ako5ver2) should be
   manually masked, tagged for future routing, or left held out. Still open.
9. **Keep this file from falling behind again.** The 2026-09-30 reconciliation
   is done -- the Track Ledger above was rebuilt from every track's own files,
   and the three proposals and the Kimi exchange are now held in `doc/`. What
   caused the drift is structural and still true: **Tracks G and H live in a
   separate clone** (`lineart-panel-generation`, with `lineart-face-words` as
   its worktree), so `git worktree list` run here cannot see them, and neither
   sent a notice to `inbox/`. Notices stopped on 2026-09-17 while those two did
   all the work of the following two weeks. When checking project state, list
   the sibling `lineart-*` directories and `git branch -r`, not just the
   worktrees. Whether to consolidate that clone back into this repo's worktree
   set, and whether the words-line tracks should send notices at all, are open
   questions for the user -- the current arrangement works, it is only invisible
   from here.
10. Revisit whether `--max-soft-ink-ratio` needs a per-source
   `diagnose_gate_funnel.py` pass for ako5ver2/hamlabi/fitness/gakuen (only
   housei has an established relaxed value so far); yield may be
   conservative for the others under the shared default. Still open.
