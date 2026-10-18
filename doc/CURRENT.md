# Current Project State

Updated: 2026-10-18 JST. **Track K opened** by user approval: the 1px
skeleton's connectivity, which is upstream of Tracks F, G and H. The error has
two directions -- over-merge, which the author can mark, and under-connection,
which has no mark and is the binding one (18.7% of arms joined). K1 registered
before Track H's 300-face analysis returned.

Previous: 2026-10-09 JST (Tracks G and H, two new lessons, a **withdrawn tool
trap** -- the `cy`/`cx` swap was a misdiagnosis -- and, later the same day,
Track G's resumed placement question settled: the reference frame carries the
win and lesson 13 now holds from both the classification and the prediction
side; see Known Tool Traps for two new entries on checking a box's
orientation and on writing a negative control. Later still: the distorted
position bins were measured and the verdict is **"big"** -- every position-bits
figure in Tracks F and G is replaced, no verdict flips, and 86% of the apparent
placement structure was the panel's aspect ratio, which is **lesson 14**. Track
J opened to take the idle GPU. **2026-10-10**: Track J's stage 1 is in -- the
preprocessor cannot fill, so for fills the baseline is *paint nothing* against
*paint everything* -- and two GT-only measurements moved its unit to the koma
panel, one of which says the line art on this corpus **encloses almost
nothing**, which the words line should know about. Two new tool traps came out
of it).
The Track Ledger, Lessons, Known Tool Traps, Active Goal and Next Actions are
current as of this date; the ledger was rebuilt on 2026-09-30 by
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
  same group-A 192-tile batch as the oracle. **Rescored under lesson 9 on
  2026-10-03, and the rescoring reverses the reading**: raw f1 0.3202 against
  the preprocessor's 0.3164 looked flat, but the classifier's output is sparser
  (ink_ratio 0.042 against 0.055) and therefore has a *lower* floor (0.1275
  against 0.1494), so in signal it is **+0.1928 against +0.1670 -- clear of its
  baseline by +0.0258.** The flatness was the hidden floor difference, not the
  result. `stroke_decomposition.py` agrees independently: `c_survival` falls
  1.0 -> **0.6283**, i.e. it deletes 37% of the strokes that should go.
  Composition was never flat either: **precision +0.068** (0.2373 -> 0.3057) and
  **recall -0.169** (0.5255 -> 0.3566) -- it is deleting, and over-deleting.
  Ceiling on that batch is the pixel-level oracle at signal **+0.4872** (raw
  0.5870). Rescore data: `results/signal_rescore_20261003/` in that tree.
  **Tuning closed 2026-10-03, and the way it closed is the finding.** Four
  variables were isolated one at a time -- decision threshold (0.3-0.7, no
  retraining), epochs (to 15), `pos_weight` (2.0 / 3.97 / 6.0), and capacity
  (16,201 -> 105,185 parameters, `channels=48,blocks=5`) -- and **every one of
  them was already at its optimum.** Threshold peaks at the default 0.50 and
  decays monotonically above it; 3 epochs converges and 15 moves signal by
  0.0014; the mechanically derived `pos_weight` 3.97 beats both neighbours; and
  **6.6x the parameters gives +0.1927, indistinguishable from the small model**,
  with montages that cannot be told apart. Final configuration: signal
  **+0.1942**. Four independent axes landing together on "the default is
  optimal" reads as the ceiling of this architecture rather than of the task:
  a flat conv stack with no pooling and no dilation has a receptive field of
  only 9-13px, and multiplying parameters without widening it changes nothing.
  **Untested and now the decision point**: (a) widening the actual receptive
  field -- dilated convolutions or more stages, a structural change rather than
  a capacity one; (b) whether the strict 2px-match label is itself hard to
  learn, which is delicate because redefining the label breaks comparability
  with the 0.6765 / 0.7425 ceilings. Commits `80d96c6`, `39c5387`, `ea450d6`,
  `0be448f`; montages under `results/epoch_sweep_20261003/` and
  `results/capacity_sweep_20261003/`.
  **2026-10-04: the direction has a zero ceiling and the track's framing is now
  a user decision.** The structural change was made (dilated RF 9->17px,
  parameter count held) and gives +0.0369 against `keep_all`, a +0.0049 nudge
  over +0.0320 -- on the noise boundary the four-knob sweep established, and
  visually indistinguishable. It is unaffected as a measurement and inherits the
  ceiling anyway: see lesson 10. The pixel oracle this track is measured against
  scores **0/15 usable**, as does the untouched rough and as does this
  classifier, against GT's 14/15. Selection inside "the rough's ink, minus some"
  cannot reach a usable drawing however good it gets. What stands: the selection
  is non-random (+0.0273 against ink-matched random deletion, 95% CI
  +0.0217..+0.0329, higher on 78.6% of 187 tiles) and 88% of the gain survives
  aggregation to whole stroke segments, so it is stroke selection rather than
  per-pixel nibbling. **Track C has raised three options and is not deciding
  between them**: continue scoped as preprocessing only, fold into a track that
  draws rather than selects, or close out.
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
  note the separate clone) -- **active, last commit 2026-10-09; the resumed
  placement question is settled and the track is now waiting on Track H's
  unit.** Given a concept, choose words and place them. Approval is recorded in its own
  briefing and work log ("起案・承認された", 2026-09-21); **the copy of its
  proposal still says draft/unapproved**, so that copy is stale, not the
  status. Where it got to, in order:
  - *2026-09-21, generation smoke*: set-at-once vs two-stage. Word bits
    **7.078 / 7.094** passed the pre-registered 7.312 gate; position bits
    6.680 / 6.658 beat the 7.563 marginal but **failed** the 6.388 cloze band,
    and the gate was not relaxed after the fact. **Every one of those position
    figures was replaced on 2026-10-09** -- on corrected bins they read 7.901 /
    7.880 against a 7.957 marginal and still fail the 7.789 band, so the
    verdicts stand and the margin over the marginal falls from 0.905 bits to
    0.077. See Known Tool Traps. **The visual gate failed**:
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
  - *2026-10-09, the halted question settled, with the face box as the
    reference frame.* Resumed on the foundation's instruction; the grid of
    probabilities was dropped for a point prediction scored by miss distance
    (panel short side), one knob only (word minimum n = 20), fitted on one
    series group and evaluated on the other, both directions reported
    separately, 40,228 in-box clusters after dropping duplicate panels. All
    four gates passed both ways, but the decomposition is the finding: median
    miss **0.347 absolute/per-word (A2) -> 0.157 box centre alone (F0) ->
    0.152 box+word (F3)**, so **97% of the "relative beats absolute" win is
    containment in the face box** and 3% is placing the word inside it
    (F3-F0 95% CI [-0.0078, -0.0043] / [-0.0074, -0.0046]; the negative
    control -- swap in a same-area box from another panel -- erases it). **The
    anchor is required**: predicting the box position too (C1, 0.350 / 0.340)
    is no better than absolute, in fact marginally worse (C1-A2 CI
    [+0.00004, +0.0040]), which measures the "if" in the user's claim that
    one fixed word determines the rest. In absolute coordinates the word adds
    nothing (A2-A1 CI spans 0 both ways). **In the eye/mouth size band
    (r = 0.1-0.25) the word's contribution is exactly zero** (F3-F0 -0.0001 /
    -0.0003), while clustering the in-box (u, v) of that band into three on
    the fit half alone reproduces, independently in both groups, **the same
    three peaks Track H found by a different implementation and a different
    figure** (0.256/0.742/0.496 u against H's 0.27/0.73/0.50). Knowing which
    of the three peaks would **halve the miss (0.074 vs F0's 0.149)**, and the
    word takes none of that: so **the relative structure is real and large,
    and the word does not point at it** -- lesson 13 confirmed from the
    prediction side. The author then judged the figures: orientation correct,
    the two eyes and the mouth below them visible, all 12 face boxes on faces,
    r >= 0.5 is "a circle" (not "contour/hair" as the track wrote, corrected),
    and **some of the 12 clusters are not one thing** -- the author seeing the
    unit trap directly. The author also noted the three peaks are coarse
    because profile and reclining faces are mixed in, so **0.074 is an upper
    bound on the ceiling, not the best attainable**, which strengthens rather
    than weakens the conclusion. Artifacts `results/framerel_20261009/`, tool
    `tools/gen/framerel.py` (written without reusing the coordinate parts of
    `placement_gt.py` / `relpos.py`, reading `meta.csv` by column name and
    taking panel size from the line PNG, so it does not sit on the distorted
    bins). Notices
    `inbox/note_track_g_frame_relative_placement_20261009.md` and
    `inbox/note_track_g_status_and_idle_capacity_20261009.md`. **Decided
    2026-10-09 (user): the next measurement waits for Track H's unit** --
    restricting to frontal faces was drafted but never registered or run,
    because with a mixed unit it would only measure a mixed unit's stability.
    `framerel.py` takes a new unit's labels in place of the word id, so the
    distance to 0.074 comes out in one run.
  - *2026-10-09 (third notice), the distorted position bins measured and the
    verdict is "big".* Given this by the foundation as work needing neither the
    GPU nor the author. Rebuilt the bins on matched denominators -- words,
    scales, order and the split unchanged, only the two position tokens -- and
    re-ran all five figures in matched pairs, same seed, same code, same
    session. **The figures all move (largest delta 1.449 bits against a
    pre-registered 0.270 "big" threshold) and no recorded verdict flips**; the
    numbers, the collapsed margins and the mechanism are in Known Tool Traps.
    The mechanism is the finding: **the position tokens were encoding the panel
    aspect ratio**, and a no-training frequency table on a 16-way aspect bin
    reproduces 86% of the cloze transformer's advantage -- lesson 14. **Track
    F's central negative is untouched**: word bits move by -0.056 to +0.004 and
    the cloze still falls far short of its 6.348 gate, so "no inter-word grammar
    inside a panel" never sat on the position bins, and the track's own
    pre-registration forbade using this run to re-read that negative. Its
    instrument checks failed four times in the day and **all four failures were
    in the checking side, never in the measured side** -- including a control
    that was vacuous by construction and a known-value cell that named a number
    without naming the quantity, both now in the traps. Track F's files were
    not touched: `train_cloze.py` was copied unchanged (sha256 verified against
    the original) and run with this track's own paths.
  - Also reported from here, and **withdrawn as to its cause on 2026-10-04**:
    the claim that Track F's `cluster_set` meta has `cy` and `cx` swapped. The
    columns are as named; what is transposed is `strokes`/`pts` and `dims`.
    The distorted position bins are real, and so is the consequence for every
    position-bits figure. Track G's own analyses are numerically unaffected but
    carry **swapped axis labels** -- see Known Tool Traps for the settled
    version, which it needs before it resumes.
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
  test whether a rule survives a different artist). **Pushed as of 2026-10-09**
  (`ca9d788` = `origin/face-words`), so it is no longer single-disk.
  Where it got to, in order:
  - *2026-10-04, the opening hypothesis is refuted.* The user judged the
    size-stratified montages of w342/w343/w344: across 150 instances there is
    **one closed eye as a drawing** (r2270 and r89070 are the same strokes in
    two overlapping panels of one page) and 34 pieces of hair. Splitting by
    size does not separate them because **these are not closed-eye words** --
    they are the class "arc, short bundle of strokes", holding hair, shading
    hatch, panel borders and an ear, while scoring among the best shape
    cohesion on record (0.53-0.56). The user's reading is the thing to carry:
    *an arch of hair is itself decided by its relation to what is around it,
    and cannot be judged from shape alone.*
  - *2026-10-04, every figure built so far was a mirror image*, caught against
    the line PNGs. The strokes are stored (y, x). This is lesson 12 and the
    settled axis statement in Known Tool Traps; the figures were rebuilt from
    the same seeds and row numbers.
  - *2026-10-04, shape-neighbour retrieval is weak and the detector works.*
    Collecting the nearest clusters to a verified seed by symmetric chamfer
    returns **5 ears in 40** for the ear seed and **2 closed eyes in 40**
    (seed excluded) for the eye seed -- about a tenth, consistent with the
    user's reading above, and the approach was dropped. What replaced it is an
    external anime face detector (`deepghs/anime_face_detection`
    `face_detect_v1.4_s`, distributed default threshold 0.307, knobs untouched)
    used **only as a labelling aid, never trained on**: on 42 panels the user
    counted **7 missed faces against 39 boxes with no false positive**, about
    15%, passing the pre-registered 20%/20% gate, and the misses are
    fragmentary cuts. Run over everything: **3,821 of 5,345 panels have a face,
    5,531 boxes, and 42,983 clusters (24% of 176,887) fall inside one.**
  - *2026-10-04, the face box turns out to carry the structure the words did
    not.* Density of cluster centres in box-relative coordinates shows **two
    peaks at the eyes** ((0.27, 0.40) and (0.73, 0.38)) and **one at the
    mouth** ((0.50, 0.73)) at relative size 0.1-0.25, a nose-shaped ridge below
    centre at r < 0.1, and a ring of contour and hair at r >= 0.5. Recorded
    sceptically in that track, correctly: a detector that places boxes by
    finding eyes and mouths will reproduce their positions, so the montages and
    the labels are what test it.
  - *2026-10-04..09, 1,003 instance labels and the first rule draft.* Verified
    here against `labels/strategic_labels.csv`. Cues compared leave-one-out
    within the training labels only: **position+size beats the word outright,
    and the word adds nothing on top of it** -- lesson 13. Orientation helps
    eyes and mouths and not brows, for a reason that redirected the track.
  - *2026-10-09, the unit is the problem, and the user's decision follows from
    it.* **41% of labelled part instances are mixed** and brows are 75% mixed,
    almost always with the hair beside them; 14.6% of all strokes are in no
    candidate cluster at all. So rather than tune a rule on the existing
    clusters, the track is **building line-level ground truth**: a tablet-capable
    UI (`tools/face/group_server.py`, port 8473) where the author selects
    strokes and groups them into parts, pre-registered on 2026-10-09 to measure
    mixing, fragmentation and omission, and to evaluate a regrouping mechanism
    against the same held-out split the instance labels use. It went live
    2026-10-09 20:06 with no production groups; **read from that track's files
    on 2026-10-10 07:24 there are 41 groups in `labels/face_groups.jsonl`**,
    the five trial groups being in a separate file and excluded by
    pre-registration. That is this file looking at the track's disk, not the
    track reporting -- it still owes its notice -- so treat the figure as a
    timestamp, not a status. The three pre-registered measurements need more
    than 41.
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
  been "clean line art vs broken something". Re-open condition was: when two
  interpretable outputs can be compared. **Met 2026-10-03, reopened, and
  stopped again 2026-10-04 -- having produced the most consequential result of
  the week.** The comparison itself was closed as unanswerable (lesson 11: it
  ranks the loss, not the method), but before stopping it ran an *absolute*
  judgement instead -- one image at a time, "is this usable as line art" -- and
  that is what established lesson 10's zero ceiling. It also caught the wrong
  baseline behind Track C's +0.0272. **Stopped, not concluded**: the untried
  route it records is re-rendering surviving strokes so the output reads as a
  sparse drawing rather than a holed rough. An unused build with disjoint tiles
  across arms sits at https://claude.ai/artifact/Mvvoqn3CAjtaNQ3ScqR9vG, worth
  running only if someone disputes the finding. Its diagnosis is what points at the
  words line -- interpretability has to be produced before preference can be
  measured. UI and pairs are kept at `results/comparison_pairs_20260916/`.

### Opened since the reconciliation

- `../lineart-image-prompt` (branch `image-prompt`, **Track I**) -- **opened
  2026-09-30**, read `## Approaches Never Taken Up` below for why. Does an
  IP-Adapter image-prompt channel change lesson 8's mechanism, or only add a
  second thing to copy? **Its first probe is void and the reason is worth
  carrying**: it ran 8 arms x 192 tiles and measured nothing, because cs was
  fixed at 2.5 -- a figure borrowed from Track B's *SDXL* bare ControlNet and
  from Track A's *LoRA-equipped* SD1.5, neither of which is this track's
  configuration (SD1.5, public ControlNet, no LoRA), for which no cs has ever
  been established. Every arm landed in the hatch-dominated regime (ink_ratio
  0.442 against the conditioning map's 0.054, near_white 0.162 against 0.870),
  where lesson 1 says differences collapse -- and they did: seven diffusion arms
  visually indistinguishable. **The root cause was that the probe was run from
  this foundation session**, where a long mixed context lost the binding between
  a number and the configuration that produced it. The track now leads with a cs
  sweep on the baseline alone, which also yields a figure this project does not
  have: the best cs for the bare public ControlNet on SD1.5. What survives from
  the void run is the plumbing anchor (conditioning map alone 0.3164, matching
  Track C's 0.3164) and the arm design.
  **2026-10-02, the sweep ran and the answer is that no cs is healthy in that
  configuration**, for a reason one layer below the borrowed number: the probe
  fed `LineartDetector(coarse=True)` maps to `control_v11p_sd15s2_lineart_anime`,
  a ControlNet trained on a *different* preprocessor, while the matching
  `control_v11p_sd15_lineart` sat unused in the same checkpoint directory. With
  the unpaired one, f1 is flat at 0.198-0.211 across cs 0.5-8.0 and
  `vs_condition_f1` **peaks at cs5.0 and reverses** -- the signature of the
  image collapsing rather than following; turning ControlNet fully off (cs 0.0)
  changes f1-against-GT not at all, and so does swapping in a deliberately wrong
  map. With the paired one, `vs_condition_f1` climbs monotonically to 0.872 and
  f1 to 0.3115, approaching the conditioning map's 0.3164 **from below without
  ever passing it** -- lesson 8, as a curve. Note also that cs is a plain
  multiplier on the ControlNet residuals with **1.0 the magnitude it was trained
  to emit**, so this project's habitual 2.0-3.5 is out of distribution by
  construction. The track's own question has an answer for content transfer:
  rescored with `stroke_decomposition.py` (lesson 9), the probe's arm leaking
  the tile's **own GT** as the image prompt is indistinguishable from the arm
  given an unrelated image -- every difference inside one standard error, win
  rates at chance. Those two arms hold ControlNet fixed and vary only the
  reference, so **that null belongs to IP-Adapter alone**, whatever state the
  spatial channel was in.
  **Closed 2026-10-04 (`7c2f470`) on a clean negative, and that closure
  includes the tone axis this file listed as untested.** About forty arms --
  pooled and patch checkpoints, uniform / style-only / style+layout injection,
  three reference types, `ip_scale` 0.4-1.0, at cs 1.0 and 2.5. Floor-corrected,
  the real use case (an honest, unrelated reference) scores **+0.1317 against a
  no-adapter baseline of +0.1317, equal to four decimals**: the image-prompt
  channel contributes exactly nothing. Leaking the tile's own GT buys +0.022 and
  still lands below the conditioning map alone (+0.1670) and below the bare
  matched ControlNet (+0.1636). The patch checkpoint does carry reference
  *content* -- leak and unrelated reference separate there, where the pooled one
  had them identical to four decimals -- but what it carries is the reference's
  content, not how to draw line art. Two results from it outlive the track:
  InstantStyle's layer-wise injection, rerun with SD1.5's own block indices
  behind a gate that fails loudly, is **the first arm on record to beat a
  baseline on `b_recall` and `neither_ink` at once** -- it needs the leak to do
  it, and `c_survival` never improves in any of its twenty-six arms; and the
  tone gain reported earlier was mostly content leaking in (`near_white` 0.464
  under uniform injection falls to 0.075 when only the style blocks are used).

- `../lineart-solid-fill` (branch `solid-fill`, **Track J**) -- **opened
  2026-10-09 by user decision**, reopening Next Actions item 5 after the GPU
  was found idle with nothing held anywhere that is GPU-bound. Predict the
  **solid fills** from the rough: a region problem, not a line problem. The
  reason this and not another GPU slot: the preprocessor's `fill_ratio` is
  **0.0%** against GT's **24.5%**, so this is the only residual where the
  standing result "no trained model beats the preprocessor" (lesson 6) cannot
  apply by construction -- and a loss settles what item 5 actually asked,
  whether fills are the other half of the plan or a separate project.
  **`fill_ratio` is normalised by ink, not by area**, and this file invited the
  area reading until Track J separated the two on 2026-10-09: 24.5% is the
  share of *ink pixels* belonging to strokes wider than 8px, and the share of
  the *tile* that is fill is **4-8%** (measured over all 13,046 tiles: `ako5`
  0.0757, `housei` train 0.0454, `housei` test 0.0460, `ako5r` 0.0404). Both
  belong next to each other wherever either is quoted -- the same failure as
  the "16-20%" above.
  **Counts, corrected 2026-10-09 by Track J and re-verified here**: the
  **11,868** first recorded in this entry was wrong twice over, counting
  `ako5r` inside `ako5` (a `^ako5` prefix match) and dropping `housei`'s test
  tiles. The pools hold **13,046 tiles that have never been trained on and
  appear in no evaluation set**: 11,450 in train (`ako5` 5,962, `housei`
  5,070, `ako5r` 418) plus `housei`'s **1,596 test** tiles, which are the
  holdout and are already separated as a different source; `ako5` has no test
  split of its own. Track J also has the first full-population profile of these
  pools (the 2026-09-11 inventory sampled 120 tiles each):
  `results/baseline_fill_20261009/gt_fill_profile.csv`, where `housei` test is
  **41.1% near-blank** against the sampled 38.3%, and roughly half its tiles
  contain any fill at all.
  **Reframed 2026-10-10 by the user, and this changes the unit before stage 2
  is registered**: the thing that is filled or not filled is a **region** closed
  off by lines, not a set of pixels, and a fill's outline is a line -- so if the
  line can be recovered from the region, fills stop being a separate kind of ink
  (the Active Goal has the author's wording and what follows from it). Three
  measurements on GT alone, no model and no GPU, come before any stage-2 design:
  **(1)** how much of a fill region's boundary is already traced by line ink in
  GT, which decides whether "recover the line from the region" is recovery or
  construction; **(2)** whether highlight fills in hair separate from solid
  fills by shape statistics alone, since the author calls them a texture;
  **(3)** whether the rough carries any indication of which regions are filled
  -- if it does not, stage 2 is not a prediction task at all. The region
  framing also runs into a measured limit: tiles that sit entirely inside a
  fill have no boundary in the window, so it may need panel-level inputs rather
  than 480px tiles. Stage 1 is unaffected and its pre-registration stands as
  written -- this is for the next registration, not a revision of the one being
  scored. Its
  briefing carries the four data traps (no `ako5` split, 26-38% near-blank
  tiles, `housei` `grid_cv` 3.18, `line_width_p50` cannot see fills), a
  two-stage plan -- **baseline inference first**, because neither pool has ever
  been scored at all, then the four-day slot for a region model against three
  mandatory baselines (paint nothing / paint every closed region / paint GT's
  mean share by area) -- and one prohibition worth repeating here: **the
  diffusion route is predicted not to produce fills at all**, by Track D's
  hypothesis 4 (content absent from the conditioning map is not drawn) plus
  lesson 8, so using it requires first building an instrument that could
  falsify that prediction.
  **Stage 1 came in on 2026-10-10, and every number below was re-derived here
  from the track's own `per_tile_*.csv` before being folded in** -- an
  independent pass over the pixel counts reproduces `report.txt` arm for arm:
  - **Nothing in the preprocessor family fills.** On the primary stratum
    (normal layer, GT has a fill, pools never averaged) the readable ceiling is
    micro IoU **0.0000** (`preproc@128`), against a GT fill area of 0.0939 on
    `housei` test (n=782) and 0.1201 on `ako5` held (n=474). **C1 holds**: the
    premise of the track is intact and stage 2 is cleared to register.
  - **The trivial arm is the comparison to beat, and its score is an identity,
    not a measurement.** `all_black` scores micro IoU 0.0939 / 0.1201 --
    exactly the GT fill-area share, because painting everything makes the
    intersection the GT and the union the tile (the track's IC2 confirms it on
    2,308/2,308 tiles; re-derived here as 0.093921 / 0.120147). "The trivial
    arm scores 0.094" and "GT fill area is 9.4%" are therefore **one number**,
    and any arm below it is worse than blacking in the whole tile. For fills
    the standing baseline is **not** the preprocessor -- lesson 6 cannot bite,
    the preprocessor scores zero -- but the pair *paint nothing* / *paint
    everything*.
  - **The adopted checkpoint does not fill either.**
    `combined_koma_lucy_mild_msgan_20260729`, put on these pools for the first
    time, reaches 0.0163 / 0.0152 and **fails its own off-by-one control**, so
    even that is not evidence of placement; it thickens strokes. Consistent
    with lesson 8 and Track D's hypothesis 4.
  - **Three corrections to the track's own notice, found here and sent back.**
    (i) The headline "the preprocessor's ceiling is micro IoU 0.0002" quotes
    `preproc@32`, an arm its own registration excludes from being read; the
    readable ceiling is 0.0000, and the largest number any variant reached is
    **0.1618** (`preproc@budget` on `ako5`, handed GT's ink count so that only
    placement was its job). (ii) `preproc@32` and `@otsu` were excluded for
    failing **NC2, the off-by-one control**, not the transpose control that the
    report's summary line names for all three; only `@budget` failed NC3.
    (iii) `preproc@128` is described in the C1 line as the best variant
    "passing the negative controls", but no control was run on it -- it predicts
    nothing, so there was nothing to control. The accurate statement is that
    **every variant predicting a nonzero amount failed a control.** None of
    this moves C1 or C2.
  - One verdict is **held**, correctly: on `ako5r` (n=23) `msgan` lands at
    0.0664, between the registered bars, to be decided on the montage before
    stage 2 registers. n=23 cannot carry more than that.

- `../lineart-stroke-topology` (branch `stroke-topology`, **Track K**) --
  **opened 2026-10-16** by user approval, from the user's own observation after
  finishing 300 faces in Track H: "the 1px thinning and the linking leave many
  parts with wrong connectivity". The defect is **upstream of Tracks F, G and
  H** -- the unit the words are made of is decided there -- and nobody could
  own it: Track F wrote the rule but is sealed and must not be modified, Track
  J measures it only as a side job to the fill, Track H is author-bound.
  Notice: `../lineart-stroke-topology/doc/initial_notice.md`. Proposal:
  `doc/track_k_stroke_topology_proposal_20261016.md`.
  - **The error has two directions and they are not the same error.**
    *Over-merge* (several things fused into one stroke) is what the author can
    mark in Track H's UI (`broken`, the "stroke to take apart" mark the user
    asked for on 2026-10-09). *Under-connection* (one drawn stroke left in
    pieces) has **no mark at all** -- and that is the binding one: Track J's
    calibrated chainer joins only **7,513 of 40,212 arms (18.7%)** in the line
    art, 12.5% in the rough. So the data in hand covers the minority error.
  - **What the author's mark already says**, from Track H's pre-registered
    105-face count: of the 74 faces that could carry the mark, **42 (57%) have
    at least one**; 90 marks in all, **0.9%** of the 10,204 strokes in those
    faces, 65 of them inside a part (~3% of grouped strokes), 1-3 per face
    typically and 7 at most. The 23 oldest production faces have no `broken`
    field, which records that the mark was **unavailable**, not absent. "Many"
    is 57% per face and 0.9% per stroke; both are the same fact.
  - **The current rule, read out** (Track F `tools/stroke/strokes.py`, read
    only): arms at a node are paired greedily by how close the outgoing
    directions are to 180 deg, `max_bend=35`, `reach=10px`, `spur_max=6`,
    `min_len=8`. It looks at the tangent angle alone -- no lateral offset, no
    width continuity, no reference back to the ink before skeletonization. The
    failures the author saw are predictable from the rule.
  - **Inherited from Track J, calibrated**: curvature at a junction is
    quantisation noise (`W_KAPPA=0` wins; only tangent agreement and lateral
    offset carry anything), and **thinning pulls the line toward the junction
    centre so the arm tips are deformed** -- skipping the last 2px leaves
    recall at 0.654 and drops wrong joins from 0.104 to **0.039**. Near a fill
    the skeleton is the fill's medial axis and not a drawn stroke at all.
  - **Registered before Track H's 300-face analysis came back**: K1, the
    over-merge marks are concentrated on strokes adjacent to a fill. Costs the
    author no time -- marks and fill masks both exist -- so it is step 1, on
    training-use faces only (H6's split), adjacency threshold printed before it
    is read, with a random-mark negative control. **If K1 is falsified the
    cause is in thinning generally** and the repair is not "take the contour
    near fills".
  - **K1 came in the same day and was re-derived here from the track's own
    `per_stroke.csv` before being folded in: the 2x2 (140/45 against
    12,671/12,234, RR 1.4876), the length-stratified MH-RR (1.5463) and the
    radius-12 share (64/185 adjacent) all reproduce exactly**, because the
    adjacency definition was registered and printed (`ADJ_PX=4.0`,
    `ADJ_FRAC=0.25`). **RR 1.487 against a registered bar of 1.5: 0.013 short,
    held, bar not moved.** Instrument checks 4/4, and the fake-mark control's
    95% band is [0.899, 1.155], far below the real 1.487.
  - **But K1b falsified the mechanism, which was this file's own prediction.**
    The decisive leg is the attributable fraction, not the ratio: AFe =
    P(exposed | marked) x (RR-1)/RR is **0.248** at fill radius 4, 0.136 at 8
    and 0.110 at 12 (re-derived here: 140/185 x 0.4876/1.4876 and so on, all
    three agree). **Read as fully causal, the loosest definition still leaves
    75.2% of the marks unexplained, and 89.0% once the mask is narrowed to
    black regions** -- at radius 12, 65.4% of the marked strokes touch no
    black region at all. The track's first version led with "RR stays flat
    across the radius", and **Track J's objection that a flat RR is not
    decisive on its own is correct**: a ratio can survive a narrowing of the
    exposure definition while the contribution collapses. What the
    measurement reads is adjacency to **thick ink**, not to a fill. Fixing the
    contour near fills does not reach most of the author's over-merge marks,
    and the fill-adjacency line is closed. The association itself survives
    (p ~ 7e-12, outside the control band); only its name changed.
  - Step 2 was already done, twice, by Track J while this track was being
    opened: **IC-j7** (2026-10-11) and **IC-j8** (2026-10-12, committed
    2026-10-16 as `0b5c9e9`). Of the chainer's rejoin failures, refusal by
    cost is 3.6% and the wrong arm 7.2% -- **11.8% together**, so the question
    the foundation routed was the minority. 81.1% are topology, and IC-j8
    splits those: `t_touch_split` 41% (repair flips 92%) and `t_relay_other`
    57%, whose **51/51 are the crossing's own fused middle ridge, 6.2-11.5px
    long against `MERGE_PX = 6.0`** -- fragments that missed absorption by
    0.2 to 5.5px, on a constant the 972-point calibration grid never varied.
    **The binding constraint on under-connection is fragment merging, not the
    angle.** The obvious repair is already refused by measurement:
    `merge_px_inf` flips 45% of `t_relay_other` and **breaks 111 of the 200
    trials that already rejoin (55.5%)**, absorbing real strokes. Only a
    bounded sweep is left, and it belongs to Track J.
  - **The artificial-cut instrument has a named blind spot, and it is a
    constructional one.** None of IC-j6's 311 trials contains a deliberate gap
    (`t_no_node` was 0 of 90), because an occluder is drawn by *adding* ink, so
    no white paper is left behind; and the current rule only collects arms
    whose node id is >= 0, so a free end is never even a candidate. A
    deliberate gap therefore joins at **0%, by architecture rather than by
    tuning**. The notice's "the weakness of artificial cuts is that their
    distribution may differ from real crossings" is upgraded for gaps to **will
    differ**, and the author's 300 real-ink faces are the only population that
    covers it. Track K registered amendment A1 before measuring: count
    free-end-to-free-end pairs at `GAP_MAX=24px` in a **separate box**, never
    added to the same-node count.
  - **K4, registered here before it is measured**: over-merge and
    under-connection are **two symptoms of one mechanism**, the fused ridge at
    a thick crossing. Below `MERGE_PX` the ridge is absorbed, the crossing
    becomes one node and any two arms may pair -- separate things fuse.
    Above it the two halves stay at distinct nodes and nothing can pair -- one
    stroke stays cut. The prediction is therefore **opposite in sign on the
    same variable**: the author's marks should sit at crossings whose ridge is
    at or below `MERGE_PX`, while `t_relay_other`'s ridges are already
    measured above it. Falsified if the ridge length at marked strokes'
    crossings is indistinguishable from unmarked ones. Costs the author
    nothing; must be stratified by stroke length (124 of 185 marks are in the
    longest quintile).
  - **K4's variable is Track F's threshold, not Track J's, and the foundation
    named the wrong constant.** The author's marks sit on strokes that **Track
    F's** tokenizer produced, so the operative rule is F's bridge collapse,
    `len(span) <= max(3.0, 1.5 x median(local ink width))` in
    `tools/stroke/strokes.py`, not the chainer's `MERGE_PX = 6.0`. Track K
    caught this and registered `ratio = bridge span length / that threshold`,
    with a support bar of **MH-RR >= 2.0** (chosen high because K1 came in at
    1.487 against 1.5), KS p >= 0.05 as the falsifier, and a second leg -- the
    sign must actually reverse between marked strokes and box-C candidates --
    before anything is called two symptoms of one mechanism.
  - **One correction back on the mechanism**: above F's threshold the crossing
    is *not* structurally unjoinable the way it is in the chainer. A
    non-bridge span keeps an arm at each node, so a relay through the ridge
    needs two successive pairings, each within `max_bend = 35`. In F's
    pipeline the over-split side is "two angle tests in series", not
    "impossible" -- J's architecture must not be read into F's rule.
  - **A third symptom, visible in F's own source comment**: "bridge spans are
    not linked: they stay their own (short, usually dropped) token, so a thick
    crossing leaves a gap of about one ink width". The collapsed ridge's ink
    is dropped from every stroke, which is the same defect J's
    `keep_absorbed` repair names. **Ink lost at thick crossings is measurable
    with no labels at all** (bridge-span pixels over skeleton pixels) and is
    the cheapest leg of K4.
  - **Lesson 12 refined, by Track K on Track J's objection**: a
    cross-implementation anchor only works on quantities that do **not** depend
    on the pipeline's own construction. K's skeletons and J's are not the same
    object -- F prunes spurs at 6px and reads a cached skeleton, J keeps every
    short fragment and skeletonizes on the fly -- so fragment, node and arm
    counts cannot be lined up between them, and K recorded that K2's box
    counts will not be printed beside J's buckets. IC-k1 agreed to 0.99
    because fill area does not depend on the skeleton at all. **Choose the
    anchor quantity for construction independence, not for availability.**
  - **K4a came in on 2026-10-16 and both registered primary values are
    refuted, which is the point of it.** Instrument checks 4/4, including
    IC-k5b matching the stroke pack 177/177 with no difference in a single
    stroke. Re-derived here from `per_panel.csv` and `per_bridge.csv`: strokes
    90,111 = pack = reproduction, bridges 35,799, two-node spans 91,567,
    skeleton 8,969,812px.
    - **A: refuted.** The median `len / ink_width` of a collapsed bridge is
      0.600 and the **empirical** null model's band is [0.596, 0.607] -- the
      observation sits inside it. F's source comment "a thick crossing leaves
      a gap of about one ink width" is **not supported**: the collapsed
      bridges are indistinguishable from the two-node span length distribution
      truncated at the threshold. "About one ink width" was the selection rule
      restated, not a property of the material.
    - **B: not supported.** Over all 91,567 two-node spans, rho(width, length)
      is **0.0205** against a bar of 0.30, and the shuffle control is 0.0443,
      i.e. *higher* than the observation.
    - **The side arm is the false positive this foundation's intervention
      prevented.** On the selected bridges alone rho is **0.6521**, and the
      track's first registration would have reported that as strong support.
      Selection on a function of both variables measures the cut, not the
      material.
  - **But K4a's headline needs the fill split before it is written, and this
    file supplied the number that shows why.** The loss-2 bridges -- the ones
    that survive as their own stroke -- have a median local ink width of
    **26.8px** and p90 **127.1px**; 92.7% exceed 8px. A drawn line in this
    material is about **2.7px** wide, and that figure is window-invariant:
    2.828 at koma scale against 2.828 on 480px tiles outside the fill mask,
    against 18.03 and 4.47 over all ink (this file, 2026-10-10). So the 7x
    width the track attributed to the native panel scale **is the fill, not
    the scale**, and most of loss 2 is a solid area's medial axis rather than
    anything inside a line crossing. Only ~7.3% of loss 2 sits at a width two
    crossing lines could produce. **The 21.2% is real and its name was wrong**:
    it is the share of the 90,111 stroke tokens in those panels that are
    collapsed-bridge fragments, dominated by fills.
  - **K5, registered before measuring, which reconciles K1 with K4a**: the
    loss-2 fragments are **depleted inside the author's part groups** and
    concentrated outside them, and within the groups the one- and two-stroke
    groups (175 of 489 in H6) are enriched in them relative to groups of three
    or more. The author marks what they tried to use for a part, and faces are
    not fills, which is why K1's marks are not explained by fill adjacency
    (AFe 0.248) while the token population is mostly fill-derived. Falsified
    if the loss-2 share is indistinguishable inside and outside the groups --
    in which case the 21.2% cannot be used to support the notice's premise
    that the word's unit is being eroded here. Needs IC-k7 first: IC-k5b
    checked counts, not stroke ids, and the track registered the id check as
    its own gate before any measurement that uses them.
  - **Box B's instrument has failed its stop condition twice and the track
    stopped both times, which is the registration working.** The box is the
    only window on a deliberate gap (a "skipped line"), since the artificial-cut
    instrument cannot construct one. Run 1 (B'): 9 of 30 constructed, 3
    misclassified, cause being the chord grazing the candidate's **own** ink on
    a curved line. Run 2 (B''), with B'-1/2/3 and the foundation's required
    over-correction target T4: all four classes reached 30 constructed,
    **T3 30/30 and T4 30/30**, T1/T2 **23/30**, cause being the attribution
    radius taken from the sampled pixel's ink width. B'-1b -- the radius taken
    at the nearest own skeleton pixel, `max(2.0, iw/2 + 1.0)` -- is approved
    2026-10-18 **on the condition that the 30/30 bar is decided on a disjoint
    band of sites**, because the fix was derived from the 7 failures; the bar is
    not relaxed, the exit is registered in advance (if it fails, the chord is
    the wrong geometry and the arc gets its own fresh band), and the radius
    budget is spent. Until the instrument passes, the material's counts (box B
    88: paper 65 / faint 23, `face.mouth` 12 of the faint) are description and
    not evidence, which is the track's own ruling.
    **Run 3 (B'-1b) failed on the test band**, which is the registered bar
    working: band (a) diagnosis T1/T2 24/30, band (b) test **27/30**, T3 and
    T4 30/30 on both with the attribution trap firing 5 times and staying ink.
    All three failures have the same shape -- the nearest own skeleton pixel is
    the candidate's own **tip**, with no skeleton inside the radius at all
    (3.000000000000021 against 3.0, 4.56 against 4.00, 3.91 against 3.00) --
    and the failing rows are the long ones (gap median 27.2 against 20.3px,
    `d_tip` 8.25 against 4.12). **The mechanism is the skeleton's retraction at
    a line's end** (median 10.8px measured here): a line's ink extends past its
    last skeleton pixel, so "within half a width of the skeleton" defines a
    body correctly mid-line and not at the end, and a chord that starts at a
    tip always starts in the candidate's own ink where no skeleton exists. A
    radius wide enough to cover it swallows the gap. **Approved 2026-10-18, as
    the third and last attempt: the collar is redefined, not widened** -- start
    measuring where the chord leaves the candidate's own connected ink
    component. The test that admits it is that **it adds no tunable number**:
    the arc adds a tangent length, a wider radius moves a constant, and the
    component exit is decided by the already-registered hysteresis ink rule.
    The binding condition is that **T4 must be rebuilt for the new definition**
    -- a third line that *touches* the candidate shares its connected
    component, so "exit your own component" would walk straight through the
    filling ink and report paper -- so T4 splits into T4-touch (8-connected to
    the candidate) and T4-sep, 30/30 each, plus a separately printed
    unmeasurable box for a chord that never exits. Band (c) is substitution
    900+, bands (a) and (b) are both diagnosis now, and if band (c) fails the
    gray axis is abandoned with no fourth attempt in any form.
    **Band (c) failed and the exit was taken (2026-10-18).** T1 30/30, T2
    30/30, T3 30/30, T4-sep 30/30 -- and **T4-touch 21/30**. The tip mechanism
    C-1 named was genuinely solved: the rows that failed band (b) at 27/30 came
    back 30/30. What broke is the single case the new definition cannot see:
    when the filling ink is 8-connected to the candidate, walking out of one's
    own component walks through the fill. The 10 failures have `t_a` median
    13.25px against 2.00 for the passes, 1.5 samples against 16.0, and
    `comp_same` 8/10 against 3/24; 6 were called paper with every sample at
    g_min 255, and 4 had no post-exit interval left at all. **The decisive
    sentence is the track's own**: without T4-touch, four rows of 30/30 would
    have been reported as a pass. The gray axis is abandoned, the author now
    owns the skipped-line-versus-faint distinction, and the request carries
    **three options** rather than two, since "the ink was never broken, only
    the skeleton" is identifiable in advance.
  - **Box B's one surviving measurement, and what it is not.** 59 of the 182
    candidates (32.4%) fall in the registered unmeasurable box: 56 where the
    interval after leaving one's own ink component is empty, 3 where the chord
    reaches the other tip without ever leaving it (`face.other` 25,
    `face.eye.open` 10, `face.mouth` 8). It survives the failed run because
    **its predicate never touches the gray classification** -- only the
    registered hysteresis ink rule and geometry -- which is also why the
    material's class counts were correctly left unread. But it is **not** "the
    gap is not a gap": 4 of the 30 constructed genuine fills landed in this
    same box, so it contains real filled gaps, and the predicate is close to
    monotone in gap length, so 32.4% has no scale on its own. Before any of it
    reaches the author it needs a name, a stratification by gap, the same
    predicate run on box A's 521 joined pairs as the trivial arm, and a split
    by `comp_same`.
    **All three conditions were met on 2026-10-18 and they retired the number,
    including the name this file gave it.** The trivial arm saturates (box A
    521/521, box C 75/79, box B 59/182) and the gap stratification shows why:
    at 2-4px box B is also 1.000 (n=46), at 12px and over it is 0.000 (n=53).
    The split by `comp_same` is decisive -- **55 of the 59 have
    `comp_same = 0`**, gap median 3.00px against a collar sum of 3.50px, so the
    two tips are in *different* components and no sample point fits between
    them. **The foundation's name, "the two tips' ink lies in one connected
    component along the chord", holds for 4 of 59.** It was coined in the same
    notice that told the track its own name overclaimed, and without splitting
    by the subcase that notice had just required. The 55 are not waste; their
    name is **"pairs closer than the thickness of their own ink bodies, with no
    sample point between them"** -- free ends 1-4px apart, which is a statement
    about the linking rule's reach and may be the purest instance of the
    architectural 0% (a rule that collects only arms with node id >= 0 cannot
    make a 3px free-end pair a candidate at all; unregistered reading, the
    foundation's, for the track to check).
  - **K7, the part-label axis, stopped on its own negative control
    (2026-10-18), and the registration was at fault, not the data.** The
    control permuted part labels **within `order`**, which preserves each
    stratum's composition, so its median came out at **1.3476** against a
    registered band of [0.95, 1.05]. The track stopped rather than widen the
    band or reread the failure as a property of the material. The diagnosis is
    right and the numbers check out here: mouth 50 of 146 = 0.3425, non-mouth
    132 of 636 = 0.2075, crude RR **1.6501** (CI 1.1137-2.7537), AFe **0.3940**
    (CI 0.1021-0.6369), Fisher p 7.4e-4, permutation p 0.044; the per-part
    table also reconciles to box B's 182. **The error was the pairing, not the
    band**: a within-stratum permutation has null value 1 for **MH-RR** and
    "whatever composition alone produces" for the **crude** RR, so [0.95, 1.05]
    was a correct band attached to the wrong statistic. Approved: MH-RR becomes
    the primary with the permutation distribution as its reference and the
    null-1 band restored as a real instrument check; the crude RR is demoted to
    secondary and printed beside its own permutation median, because the gap
    1.6501 against 1.3476 **is** the composition effect and is a result worth
    printing. **One instrument check comes first**: the stratum-size
    distribution. The permutation median 1.3476 and the observed MH-RR 1.3418
    agreeing to 0.006 has a boring explanation that must be ruled out --
    degenerate strata, where singletons make the permutation an identity and
    contribute nothing to MH, so both numbers are driven by the same few
    strata. No name goes on that coincidence until the counts are printed.
    **The counts came back and the strata are degenerate** (2026-10-18): 145
    strata over 782 pairs, median size 3 and max 70; only 39 are singletons
    (5.0%) but **only 37 strata carrying 323 pairs (41.3%) contribute to
    MH-RR**, and the MH numerator's weight is **19.3% in one stratum alone**
    (n=70) with 39.1% in the top five. The coincidence is left unnamed. The
    heavier number is not the 44.1% that cannot contribute but the 19.3%: the
    primary value is substantially one stratum's property, so a
    leave-one-stratum-out range is printed as description here, and a stability
    condition in that form is registered **before** the unread band is opened.
    That one stratum is itself a finding to chase -- 70 candidates against a
    median of 3 -- and plausibly the same face-level mechanism as `face.other`
    and the 56 continuous-ink cases.
  - **The stratification variable is the *face*, which settles a causal
    question the foundation's prescription had left open.** `order` is the face
    identifier (`k7_part_axis.py` keys labels by `(order, stroke_id)` and
    clusters the CI by it; `k4_main_leg.py` counts `n_faces` from it). The
    foundation made MH-RR the primary without first deciding whether the
    stratum is a confounder or a mediator -- had `order` been something a
    part's identity determines, stratifying on it would partly block the
    exposure, MH-RR would under-state the total effect and the crude RR would
    be the right estimate of it. A face is **upstream of both** the part label
    and the box-B rate, so it is a confounder, MH-RR is the correct adjustment,
    and the 1.6501-against-1.3418 gap is the between-face composition effect.
    Reports should call the stratum the face: "order" reopens the question on
    every reading, "face" closes it in one line.
  - **Strata where every unit is a case pull MH-RR toward 1, so the all-strata
    value is the conservative one.** The track printed MH-RR 1.3418 over all
    strata against 1.3502 over contributing strata and said the difference is
    that one-sided strata still contribute. The direction can be stated
    exactly: in MH-RR = sum(a·n0/n) / sum(b·n1/n), an all-case stratum has
    a = n1 and b = n0, so its numerator and denominator terms are both n1·n0/n
    -- equal mass on both sides, which shrinks the ratio toward 1. Keeping
    1.3418 as the primary is therefore right, 1.3502 is printed beside it with
    that reason, and the registration for the unread band states which of the
    two is the primary (the conservative all-strata form is recommended).
  - **The exposure stays "mouth versus the rest", and the reason is that the
    registered design was the conservative one.** The track offered to replace
    it with a per-part table after seeing that mouth is only 4th (face.other
    0.782 n55, glasses 0.545 n11, face.ear 0.388 n85, face.mouth 0.343 n146,
    face.nose 0.186, face.eye.closed 0.143, face.eye.open 0.085 n247,
    face.brow 0.083 n145). Computed here: face.other sits in the comparator, and
    removing it **raises** the mouth's RR from 1.6501 to **2.2356** -- so
    comparing against everything else diluted the result rather than
    manufacturing it, and moving the primary onto the table would mean picking
    the maximum of 8 post-hoc. The table is kept as post-hoc description with n
    on every row and **nothing below n = 30 used in a finding**. The one
    exception is **face.other, which gets its own registration**: it is the
    table's maximum *and* the only row with a mechanism written before the
    measurement -- the track had already identified cheek hatching as
    spur-pruning residue, which is the same mechanism as the 56 continuous-ink
    cases, so it belongs in that registration rather than in a post-hoc table.
  - **These 782 pairs are now read material, which changes what the rewrite can
    claim.** The crude RR, its CI, AFe, Fisher p and the permutation p were all
    printed while the primary was declared unread, so a re-registered MH-RR on
    the same pairs cannot be confirmatory. Split accordingly: on this material
    the question is **estimation, with AFe as the primary** (0.3940, CI
    0.1021-0.6369) under the frame adopted 2026-10-17, and the **confirmatory
    MH-RR moves to an unread panel band** -- the chainer has used only [0:171]
    of 4,049 panels -- with the registration written before the band is opened.
    **The author request does not wait for it**: what the request needs is the
    three options and a denominator split by the predicate, not the mouth's RR.
  - **A third mechanism of under-connection is now hypothesised, and it needs
    neither labels nor the author** (registered 2026-10-18, to be measured
    after the part-label axis and before the author request). The 56 cases are
    free-end pairs -- they are in box B precisely because they share no node --
    whose ink along the chord is nonetheless unbroken. If the ink is continuous
    while the skeleton is not, the defect sits in **skeletonisation and spur
    pruning**, not in the linking rule: `spur_max = 6` prunes the connecting
    spur and F's `build_links` only collects arms with node id >= 0, so nothing
    can rejoin it, by architecture. That is distinct from both the cost refusal
    (3.6%) and the ridge (`t_relay_other`). Primary value: the share of the 59
    with an 8-connected ink path between the tips and no skeleton path.
    Confound to split out with `comp_same` and the presence of another span's
    node in the connecting ink: a crossing line makes ink continuous without
    the pair being one stroke.
    **Measured 2026-10-18, and the mechanism shrank from 56 cases to 14.** The
    statistic is now box B's `comp_same = 1`, **14 of 182 (7.7%)**, which is
    non-monotone in gap (0.043 / 0.122 / 0.059 / 0.097 / 0.000 / 0.083) and so
    is not a gap-length proxy; parts are face.mouth 6, face.ear 2, face.nose 2,
    face.eye.open 2, face.brow 1, glasses 1. "A third of box B" is withdrawn,
    and what the author request can drop is 7.7%, not a third. At n=14 the
    frame changes: **existence here, prevalence on the unread band** -- the 14
    are enumerated and each checked by eye for continuous ink with no skeleton
    path, with no bar, because what a defect needs first is existence.
    **The confound that could remove the mechanism entirely**: the registered
    ink rule is `ink_mask(thr=128, grow=200)`, so a single gray-190 pixel
    between two strokes makes them one component, and the mechanism would then
    be the preprocessor's hysteresis rather than spur pruning. Required before
    any existence claim: re-decide the 14 at `thr=128` with no grow and print
    how many survive, plus each bridge's darkest gray and local ink width (a
    bridge thinner than this material's ~2.7px line is not a line).
    **Both checks ran the same day and the mechanism did not survive.** The
    predicate's name was wrong: box C is **79/79** on `comp_same` and the four
    cases where the predicate reads false are all `comp_same = 1` in the
    20-24px bin, so it is not component identity but **"no point exists along
    the chord where it leaves the candidate's own component"** -- a pair joined
    by a long detour has a straight chord that crosses paper. Renamed here too;
    this file had it as component identity. Then the hysteresis confound
    decided it. For each of the 14, the minimax path between the tips (the path
    minimising its darkest pixel) has a bottleneck gray of 0, 3, 62, 124, 130,
    131, 162, 183, 189, 190, 194, 195, 196, 198 -- **10 of 14 are at or above
    128, so they are not ink in the `thr` mask at all** -- and the bottleneck's
    local ink width is **2.0px in 13 of 14** against this material's ~2.7px
    line. At `thr = 128` with no grow, **4 survive**, and those four bridges
    are single-pixel threads. **The third mechanism does not exist in the form
    "the line is continuous"**: the enumeration was not run because existence
    itself fails, and what the author request can drop goes 56 -> 14 -> 4
    (2.2%). Arm 2 came back clean: the largest ink component's share is 0.4118
    across the 13 panels holding a `comp_same = 1` case against 0.4676 across
    the other 59, so K4a's fill trap did not fire this time -- a control worth
    recording precisely because it was a miss.
    **The trivial arms, written by the foundation under the division adopted
    today**: arm 1 is non-candidate free-end pairs from the same panel matched
    by gap bin, statistic = the difference in `comp_same` rate between
    candidates and matched non-candidates stratified by gap, **null value 0**,
    registered direction candidates higher, printed without a bar on this
    material; arm 2 is the largest ink component's share of each panel's ink,
    **no null value, descriptive**, to show the 14 are not concentrated in
    panels that are one blob -- the same trap as the fills that dominated
    K4a's 21.2%.
  - **The author request's denominator was split honestly, and its reading is
    registered before the answers arrive** (2026-10-18). Of box B's 182, the
    **55** whose two collars together exceed the gap (gap 2.00-9.22px, median
    3.00) are removed, because "did you skip the line here" is not a question a
    3px gap can pose to the author -- it is invisible at output scale. **127**
    are answerable (gap 2.24-23.09px, median 10.82), 40 are drawn at seed
    20261018, the 4 single-pixel-thread cases are listed separately, and the
    sheet never claims 182. The instrument measures the **presentation**, not
    the author: 8 repeats and 8 anchors among 56 tiles. Registered reading of
    the primary, with Clopper-Pearson at n=40 being +/-15 points: **28 of 40 or
    more reads as "mostly intentional"** ([0.535, 0.834]), **12 or fewer as
    "mostly dropped"**, and anything between reads as **"both exist in
    quantity"** -- which is not a failure but the finding that F's rule cannot
    be corrected toward either side. The sheet must also not leak: no part
    names on the tiles, nothing about the mouth hypothesis in the text, and no
    hint of the expected distribution.
  - **The debris points the other way: the hysteresis bridge is an untested
    explanation of the author's *over-merge* marks** (registration D, proposed
    by the foundation 2026-10-18, Track K to accept or decline). The heaviest
    fact left by the dead hypothesis is that **`grow = 200` counts a 2.0px
    thread of gray 130-198 as ink**. Turn that around: two separately drawn
    lines bridged by such a thread become one ink component, the skeleton runs
    through, and **two drawn strokes fuse into one token** -- which is exactly
    the direction the author *can* mark (`broken`). The earlier panel census
    is consistent: 258 components at 128 collapse to 64 at 200. K1 already
    refuted fill adjacency as the explanation of those marks (AFe 0.248), and
    this costs the author nothing because the 90 marks already exist. Shape:
    primary = the share of marked tokens whose skeleton passes through a pixel
    that is ink **only** under grow, against unmarked tokens, stratified by
    length and by face, MH-RR with AFe as the estimate; **null value = 1 for
    MH-RR** under permuting marks within a face. **The instrument check that
    comes first**, and without which the primary is not read: the share of
    *all* tokens whose skeleton touches at least one grow-only pixel -- if that
    is 90%, the predicate is saturated and separates nothing, which is this
    question's "paint everything" arm. Repair is **not** Track K's: the ink
    rule is shared by Track F's export and Track J's chainer, so if D stands
    the fix is a cross-track design change under its own registration and the
    user's call. Measure in K, repair elsewhere.
  - **The pruning-residue mechanism and the continuity mechanism are separate
    registrations, and bundling them was the foundation's error.** Face 267 --
    the stratum holding 19.3% of the MH weight, 70 pairs, box B rate 0.429 --
    settles it: its box B content is all 14 of its `face.other` strokes plus
    mouth 9, brow 4, eye.open 3, and **`comp_same = 1` is 0 of its 30 box-B
    cases**. The two mechanisms are cleanly separated inside one face, so
    `face.other` gets its own registration whose statistic is *not* `comp_same`
    but the ink left where `spur_max = 6` pruned the skeleton.
  - **The leave-one-stratum-out check came out against the foundation's own
    concern, which is worth recording as such.** MH-RR ranges over
    **[1.2647, 1.8729]**, never crossing 1, and the **maximum is reached by
    dropping face 267** -- the heavy stratum *dilutes* the estimate rather than
    producing it. The worry raised when the weight concentration was found is
    answered by measurement: 1.3418 stands as the primary, and the stratum is
    called the face from here on.
  - **A sealed track's docstring can be quantitatively wrong**: F's
    `strokes.py` says a bridge span stays "their own (short, usually dropped)
    token", but **53.4% of them survive** (19,101 of 35,799). Recorded here
    rather than fixed, because Track F's files are not modified.
  - **The touch merge is adopted (Track J, IC-j9, 2026-10-17), and this
    foundation's difference-form prescription was measured a second time.**
    Judged once on the reserved panels `[0:40]`, never read since IC-j6's
    registration: reference `stroke_chain` 0.619 rejoin / 0.072 wrong against
    the candidate's **0.711 / 0.072**, trial by trial 9 gained and **0 lost**,
    no trial newly wrong. Both refusal conditions checked before reading. The
    absolute bar that failed IC-j8, `wrong <= 0.059`, **is failed by the
    reference arm itself on this set too (0.072)**, so the trap holds on
    independent material. Recorded values moved with the adoption: the
    discovery set's rejoin **0.643 -> 0.752**, the reserved set's
    **0.619 -> 0.711**, `wrong` unchanged in both. `stroke_chain.py`'s sha is
    now `53b8406cfbb4f4d1`; the pre-adoption file is commit `1a82ea5`, and
    anything compared with a figure recorded before 2026-10-17 must pin it.
    **This does not reach Track K**, contrary to what this file routed on
    2026-10-17: see the file-level sha trap below.
  - **After the adoption the topology bucket is half closed**: the merge
    defect is fixed and the ridge is untouched, so `t_relay_other`'s 51 cases
    are now **66.2% of the 77 remaining failures**, up from 46.0% of all
    failures. The ridge repair needs a **two-node cost** -- the evidence is
    chain-local while the cost is node-local -- which is a design change, not
    a defect repair, and belongs to Track J under its own registration.
  - **A correction this foundation owes Track K**: the 2026-10-16 routing
    said loosening `MERGE_PX` is net-negative, from `merge_px_inf` breaking
    111 of 200. The bounded sweep separates the two. At `MERGE_PX = 10` the
    occluder's own ridge survives in **379 of 379** trials (reference also
    379/379, ratio 1.000), so the fall in `wrong` there is a real precision
    gain and the registered expectation that it would rise is refuted.
    Unbounded and bounded part company at this point.
  - **A caveat to carry out of stage 2**: pixel IoU and region-F1 **disagree
    in sign** at this signal level. At `r = 4` `rough_dark` beats both
    don't-paint and its mirror on pixel IoU (+0.0118, +0.0076) and beats
    neither on region-F1. Every stage-2 headline must name its metric. The
    verdict's robustness to `FILL_HALF_WIDTH_PX` (r in {2,3,4,6,8}, largest
    gap +0.0003 against a +0.05 bar) stands as the strong result.
  - **The GPU slot is taken: M1 trains 2026-10-19 to 10-22**, held on IC6's
    four criteria. Track J's bounded `MERGE_PX` sweep and the
    `FILL_HALF_WIDTH_PX` sensitivity run on CPU beside it, and Track K is CPU
    only, so nothing contends.
  - Step 2 is Track J's own registered next step run on IC-j6 (of the 34.6% of
    trials the chainer fails to rejoin, how many were refused by the cost
    break-even versus matched to the wrong arm -- different repairs), with the
    chainer staying in Track J and the result shared. The track is explicitly
    allowed to question 1px thinning itself and explicitly forbidden to shrink
    into a `max_bend` sweep. No GPU for now; the measurements are CPU.

### Not part of either line

- `../lineart-controlnet-realpairs` -- the **closed** cross-hatch track, kept as
  a plain directory (not a repo) because it holds physical copies of the pair
  data and checkpoints. Its history is on branch `controlnet-realpairs`. Listed
  here so an audit does not keep rediscovering it.
- `../lineart-cleanup-refiner` (branch `cleanup-refiner`) -- **dormant** since
  2026-08-04, from the CNN+GAN era (clDice topology loss, rough-fidelity
  metrics). Not part of the current route; see lesson 5 and Next Actions on why
  that architecture was left.

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

**The cross-hatch cause, and fourteen lessons that apply project-wide** (lessons 3-4 added
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
   score the conditioning images you already have. **Corrected 2026-10-10.** This paragraph stood for a
   month saying the contamination "has not been checked on the SD1.5 track",
   whose 0.2354 is in the same position, and that it might survive. The answer
   was already in this same file, in Current Data Direction: that track's
   consistency loss moved the output **away** from its conditioning map while
   moving it toward GT (vs-conditioning 0.5009 -> 0.4618 as f1 rose 0.2175 ->
   0.2354), so it is the one case on record that is *not* copying -- and it
   closed anyway on 2026-09-13, because the re-measurement on 192 tiles put
   `manga_line` alone at 0.2847 against its 0.2514 / 0.2524. The mechanism was
   real and still did not carry the output past the preprocessor. **Nothing is
   open here**; an open-sounding sentence was left standing after its own
   question had been answered elsewhere in the file.
   A companion metric was added for the same reason on 2026-09-11:
   `profile_metrics` now returns **`fill_ratio`** (share of ink in strokes
   thicker than 8px) beside `line_width_p50`, because width alone cannot
   separate a thick stroke from a solid fill -- the confusion that made this
   project read housei GT's line_width_p50 of 7.59 as "thick deliberate
   strokes" when it was measuring solid blacks. Calibrated against known
   cases: coarse_trained, the identified fill-escape model, 91.6%; a clean
   line-art model 7.1%; GT tiles 3.3%. A fill-excluded "corrected width" was
   tried and rejected -- it collapsed to ~1.9 everywhere, since what remains
   after removing fills is their own thin fringes. **Partly revised
   2026-10-10**: it does not collapse to a single value -- 1.91 on `housei`
   against 2.74 on `ako5` -- and because it is invariant to the window size
   while the all-ink width is not, it is the only usable scale check when a
   tile is compared against a panel. Rejected for style, kept for scale; see
   the Known Tool Traps entry on statistics taken over all ink.
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
9. **`gt_bsds_f1` has a floor, and the thing to measure is low `neither_ink`
   with high `b_recall` together -- f1 alone cannot be the yardstick.** Added
   2026-10-02 from `../lineart-image-prompt` (Track I). Measured on 192
   `holdout_lineart_family` tiles: an image with **no relation to the input at
   all** (ControlNet residuals multiplied by zero) scores **f1 0.1941** against
   GT and 0.3498 against the conditioning map, purely by being dense -- a 2px
   one-to-one match lands somewhere on a hatched field whatever it depicts.
   **Revised 2026-10-03, and the revision matters: that floor is not a
   constant.** It is a property of the individual output, so subtracting one
   number misreads in both directions. The same degenerate construction scores
   0.1941 on `holdout_lineart_family`, **0.1285** on `diag_valid5` and
   **0.1027** on housei; and density alone does not predict it either -- two
   arms at ink_ratio 0.425 and 0.443 floor at **0.148 and 0.204**, because ink
   concentrated into strokes hits a random GT less often than ink scattered as
   hatching. What to report instead is the **signal**, a per-output matched
   null:
   `signal = f1(output, its own GT) - mean_j f1(output_j of the same arm, that GT)`,
   j over tiles from other source images, GT held fixed and only the prediction
   swapped. Tool: `tools/evaluation/measure_f1_signal.py` (no GPU, runs on
   outputs already on disk). **Protocol: run the degenerate output through the
   same procedure first and read nothing else until its signal comes out near
   zero** -- that check caught two measurement errors in the run that produced
   these figures. Rescored this way the 2026-09-30 IP-Adapter probe's baseline
   is **-0.0012** on 192 tiles and **-0.0059** on five: not slightly above a
   floor, but exactly nothing.
   More importantly f1 cannot separate the two ways of failing, which look
   identical through it: **(a) the output is still the rough**, and **(b) the
   output added ink that suits the metric without being line art**. The split
   that does separate them, implemented as
   `tools/evaluation/stroke_decomposition.py` (no GPU; it runs on outputs
   already on disk), partitions strokes by where they are, at the same 2px
   tolerance:
   **A shared** (in the conditioning map and the GT), **B GT-only** (must be
   drawn), **C map-only** (must be deleted); then reports `b_recall`,
   `c_survival` (lower is better) and `neither_ink` (output ink far from both
   -- the "filling empty paper with invented imagery" axis). The conditioning
   map is the trivial baseline at exactly **b_recall 0 / c_survival 1.0 /
   neither_ink 0**, so anything that does not beat it on the first two is not
   moving the rough toward the line art, whatever its f1. Across 25 arms
   measured this way, **no arm achieved low `neither_ink` and high `b_recall`
   together** -- only a strict trade-off, which is the signature of buying the
   score with ink. Failure (b) looks like the unpaired-ControlNet config: two
   thirds of its ink outside both the map and the GT at every scale. Failure
   (a) looks like the paired one: `neither_ink` falls to 0.057 while
   `b_recall` falls with it to 0.102 and `c_survival` stays 0.887 -- the
   conditioning map, copied. Caveat to carry: `b_recall` is a proximity test,
   so a dense output earns it for free; never read it without `neither_ink`
   beside it. This is lesson 8 made measurable, and it is the first instrument
   here that agrees with what the eye reports ("the picture has not moved off
   the rough") rather than contradicting it.
10. **Line art is not a subset of the rough's ink. Any method whose output
    space is "the conditioning's ink, minus some" has a ceiling of zero.**
    Added 2026-10-04 from `../lineart-aesthetic-judge` (Track E) and
    `../lineart-stroke-selection` (Track C), who measured it from opposite
    sides on the same day. Track E put a blind, randomised, one-image-at-a-time
    question -- *is this usable as line art?* -- to 15 holdout tiles per arm:

    | arm | usable |
    |---|---:|
    | rough (conditioning, untouched) | **0/15** |
    | Track C's classifier | **0/15** |
    | **the pixel-level oracle** | **0/15** |
    | GT | 14/15 |

    Median 1.3s per item, so these were not marginal calls, and the anchor is
    really 14/14: the one GT rejection is `lineart_003_000`, a genuinely blank
    ground truth, so the judge rejected an empty sheet and the scale is sound.
    **The decisive row is the oracle.** It is this project's own definition of
    a correct deletion -- what `tools/selection/build_keep_labels.py` teaches,
    and the source of the 0.7425 ceiling everything in that line was measured
    against. Applied perfectly it is 0% usable, so a classifier converging on
    it converges on 0%. Deletion also moves nothing: rough = classifier =
    oracle, with no partial credit. Track C reached the neighbouring result
    from the measurement side the same day -- the label is not a stroke-shaped
    teaching signal, leaving 34.0% of even the oracle's own strokes partially
    erased. **What survives and what does not**: the classifier's +0.0320
    against `keep_all` and +0.0273 against an ink-matched random deletion are
    true statistics, and establish that its selection is non-random rather than
    a density effect. They are simultaneously **zero progress toward a
    drawing**, and that is the number that should govern investment. Deletion
    remains viable as a **preprocessing step**; it is not a route to the
    deliverable. Limits stated by the measurers: 15 per arm means "under about
    20%", not literally zero; and the same tiles appeared across arms, which
    can only depress the deletion arms -- but `rough` is itself at 0% and all
    three deletion arms are identical, so there is no gap for that to have
    manufactured. Data: `../lineart-aesthetic-judge/results/trackc_judge_20261004/`.
11. **A comparison between two lossy outputs of the same source returns a
    ranking of the loss, not of the method.** Added 2026-10-04 from Track E,
    which built the material and then closed the comparison rather than report
    it. Three granularities were tried -- per-pixel, stroke-aggregated,
    junction-chained -- each neutralising what decided the last, and each time
    the judge was deciding on breakage: of the first 77 judgements read, the
    side with fewer broken strokes won **23 of 24** decided pairs, the same
    count as the headline preference. The judge said so unprompted ("I am
    marking whichever destroys too much as NG"). **Corrected and sharpened the
    same day**: the comparison artifact's database was exported before deletion
    and held **203** judgements, not 77, and reading all of them withdraws the
    claim that the eye was not the limitation. The pre-registered instrument
    check -- does the judge prefer the *ideal* deletion over a random one of the
    same ink -- passed 17/17 **only at the dashed, pixel level**. At stroke
    level it ties **91.1%** of the time, and chained through junctions
    **100%**. Track C re-pulled all 203 records independently and confirmed
    92.5% ties pooled. So the sharper statement is not "the comparison is
    dominated by damage" but **"once the damage is removed there is nothing
    left to see"**: an ideal deletion and a random deletion of the same size
    are indistinguishable to a person. On the decision pairing -- classifier
    versus the undeleted rough, junction-chained -- 13 of 23 were ties and of
    the 10 decided the judge preferred **the undeleted rough on 7** (not
    separable at this n, but the point estimate leans against deleting at all).
    Intra-rater consistency on repeats was 9/12. This agrees with lesson 10
    from the other side: nothing in the deletion family is distinguishable from
    anything else in it, and none of it is line art. The f1-signal figures
    (+0.0320, +0.0273) were never claims about what a person sees and are
    unaffected. The artifact has been deleted; the only copy of the records is
    `../lineart-aesthetic-judge/results/trackc_judge_20261004/comparison_judgements_203.json`. Changing the unit of deletion moved the damage
    without removing it, because the damage *is* the output: the classifier
    removes a median 22.7% of the conditioning's ink and under 10% on none of
    the 192 tiles. **The check is cheap and belongs before the material is
    built**: take any axis that separates the arms -- tone, density, breakage,
    damage -- measure it, and see whether it predicts the preference. If it
    does, the human judgement adds nothing that was not already computable.
    This is adjacent to lesson 9 but distinct: lesson 9 is a metric scoring a
    relationship that is not there; this is a *human* judgement decided by a
    quantity already in hand.
12. **An instrument check that compares two things inside the same coordinate
    system cannot detect an orientation error. Anchor one leg of every check to
    something the pipeline did not produce.** Added 2026-10-04 from
    `../lineart-face-words` (Track H). Every montage that track and Track G had
    built was a **mirror image across the diagonal**, and the pre-registered
    check certified it perfectly: "share of an instance's points within 3px of
    its panel's own line art" came out at median **1.00, minimum 1.00**,
    because the red overlay and the grey background were both drawn from the
    same `strokes.npy` whichever way it was read. A negative control was added
    precisely because 1.00 looked too good, and it passed too -- swapping x/y
    drops the score to 0.00-0.02 and a 50px shift to 0.00-0.09 -- so the check
    had real sensitivity and was still blind to the single error present. What
    caught it was the **line PNG on disk**, an artefact nothing in the analysis
    chain produces, together with the user noticing that nothing in the montage
    stood upright. The check Track H runs now reads the same strokes against
    `dataset/<source>/line/*.png` and separates the two readings cleanly:
    0.999 against 0.11. Note what the error did and did not touch: chamfer
    distances, word ids, sizes and counts are invariant under a transpose, so
    the numbers survived and only the **pictures and every statement about
    direction** were wrong. That is the general shape -- a transpose is cheap to
    make, expensive to see, and leaves the metrics looking healthy. The count
    that track keeps of its own broken instruments stood at six before this one.
13. **The machine word does not carry what the thing is; position and size
    inside a detected frame do.** Added 2026-10-09 from Track H, measured on
    1,003 author-assigned instance labels inside anime-face-detector boxes.
    Leave-one-out within the training labels only -- optimistic by construction,
    since the same panel and the same face sit in the neighbourhood, so it is
    read **only as a comparison between cues**:

    | cue | open eye (prec/rec, base 0.27) | mouth (base 0.14) | brow (base 0.16) |
    |---|---|---|---|
    | position + size | 0.66 / 0.78 | 0.66 / 0.74 | 0.43 / 0.34 |
    | word alone | 0.60 / 0.50 | 0.33 / 0.10 | 0.00 / 0.00 |
    | position + size + word | 0.64 / 0.70 | 0.61 / 0.74 | 0.35 / 0.20 |
    | position + size + orientation | **0.72 / 0.85** | **0.73 / 0.87** | 0.43 / 0.34 |

    Nose and ear are below 20 positives and read as reference values only; both
    are at or near zero for every cue, which is a neighbour-vote problem at
    n=13-14 rather than evidence against position. **Adding the word to
    position+size makes every label worse**, and the word alone is at base rate
    for everything except open eyes. The vocabulary is spread far too thin to
    be an identity: 223 training instances occupy **79 of the 500 words**, the
    most frequent holding 11. Orientation (length-weighted mean stroke
    direction, doubled-angle) lifts eyes and mouths and leaves brows exactly
    where they were, because three quarters of labelled brows share a cluster
    with the hair above them and the hair's vertical strokes own the cluster's
    direction -- the trap below, not a failure of the cue. The author's claim
    that brows run roughly perpendicular to the hair is supported where the
    unit permits it: brows not mixed with hair sit at a median 16 degrees from
    horizontal with none past 60, bangs-only clusters at 71. **This is the
    quantitative form of "a word is a shape class with scale discarded": the
    shape class is real and the meaning is not in it.**

    **Confirmed from the prediction side the same day, by Track G, on a
    different question with a different tool and a different split** (series
    groups, not panels): asked to place a cluster inside its face box, box+word
    is indistinguishable from the box centre in the eye/mouth band (0.1492 vs
    0.1493, and 0.1451 vs 0.1454 the other direction), while knowing which of
    the band's three peaks a cluster belongs to **halves the miss (0.074)**.
    So the structure is there and large; the word takes none of it. Two tracks
    arriving at this from a classification task and a regression task is the
    reason to trust it.
14. **Put a no-training frequency table on one cheap descriptor next to every
    learned number.** Added 2026-10-09 from Track G's position-bin run. The
    cloze transformer looked at every other cluster in a panel and bought 1.228
    bits over the position marginal. A frequency table conditioned on **one
    16-way bin of the panel's aspect ratio**, with no model and no training at
    all, bought **1.054 of those 1.228 bits -- 86%** -- and once the bins were
    rebuilt on matched denominators the same table bought exactly nothing while
    the transformer kept 0.17 bits. The structure the learned model appeared to
    find was a leak from a broken normalisation, and **the thing that exposed it
    cost nothing to compute**. So the control is not optional: for any learned
    figure, name the cheapest descriptor of the input that could carry the same
    information -- the frame's shape, the ink total, the instance count -- and
    score a table on it. If the table gets most of the way there, the model is
    not the finding. This is the same move as the lesson under Track C
    (measure the computable axis before asking a person to judge), one level
    down: measure the computable axis before believing a model.
## Every f1 On Record, And What Can Still Be Read

**This section replaced an earlier one, and the earlier one was wrong.** On
2026-10-02 it subtracted a single floor of 0.1941 from every figure and reported
each as a share of the floor-to-oracle span. Track I measured the floor properly
the next day and it is not a constant: it belongs to the individual output. The
tables built on one floor are withdrawn. Kept here as a worked example of why a
retraction is a pass over every store that repeated the number, not a correction
in one place (Working Discipline, rule 3).

The replacement is **signal** -- see lesson 9 for the definition and
`tools/evaluation/measure_f1_signal.py` for the implementation. Only a number
with its signal measured can be ranked against another.

**Measured so far** (Track I, 2026-10-03; raw f1, that output's own floor, and
the signal between them):

| set | arm | raw | floor | **signal** |
|---|---|---:|---:|---:|
| 192 lineart_family | GT line art itself | 0.9948 | 0.1062 | **+0.8886** |
| 192 | conditioning map `lineart_coarse` | 0.3164 | 0.1494 | **+0.1670** |
| 192 | matched cs2.5 | 0.3115 | 0.1478 | +0.1636 |
| 192 | matched cs1.0 | 0.2745 | 0.1428 | +0.1317 |
| 192 | anime cs6.0 | 0.2114 | 0.1975 | +0.0139 |
| 192 | degenerate (residuals x0) | 0.1941 | 0.1942 | **-0.0001** |
| 192 | anime cs2.5 (the void probe's baseline) | 0.2023 | 0.2036 | **-0.0012** |
| diag_valid5 | conditioning map `lineart_coarse` | 0.2639 | 0.1078 | **+0.1561** |
| 5 | matched cs2.5 | 0.2637 | 0.1082 | +0.1556 |
| 5 | SDXL bare cs2.0 | 0.2612 | 0.1085 | **+0.1527** |
| 5 | SDXL bare cs2.5 | 0.2615 | 0.1096 | +0.1519 |
| 5 | SDXL bare cs3.0 | 0.2578 | 0.1113 | +0.1465 |
| 5 | **SDXL fine-tune cs2.0** | 0.1539 | **0.0825** | **+0.0714** |
| 5 | degenerate | 0.1332 | 0.1285 | +0.0047 |
| 5 | anime cs2.5 (void probe) | 0.1277 | 0.1336 | **-0.0059** |
| 192 | **Track C classifier vs `keep_all`** | -- | -- | **+0.0320** |
| 192 | Track C classifier vs ink-matched random deletion | -- | -- | +0.0273 |
| 192 | Track C classifier, dilated RF 9->17px, vs `keep_all` | -- | -- | +0.0369 |
| 192 | Track C classifier, as first measured against the wrong baseline | 0.3202 | 0.1275 | +0.1928 |
| 192 | Track C pixel-level oracle | 0.5870 | 0.0997 | **+0.4872** |
| housei 100 | **GT line art itself** | 0.9100 | 0.8443 | **+0.0657** |
| housei | conditioning map | 0.1573 | 0.1480 | +0.0093 |
| housei | degenerate | 0.0978 | 0.1027 | -0.0049 |

**Baseline correction, 2026-10-04.** The classifier's figures above were first
quoted against `condition`, which `measure_f1_signal.py` renders as the inverted
*raw* conditioning image. The deletion-only classifier does not operate on that;
it operates on `edge_map(conditioning)`, the contour map, and is a strict subset
of it on 192/192 tiles. Against the correct baseline (`keep_all` -- the contour
map with nothing deleted) the effect is **+0.0320**, not the +0.0272 reported on
2026-10-03. Direction unchanged, magnitude larger. Found by Track E,
re-derived and confirmed by Track C.

**Three things this changes.**

1. **The worry that drove the request was wrong, and the way it was wrong is the
   lesson.** SDXL fine-tune's 0.1539 looked like it sat below the floor. Its own
   floor is **0.0825** -- *lower* than the degenerate output's, because thick,
   fill-heavy ink hits a random GT less often -- so it carries +0.0714 of real
   signal. Judging it by one constant would have condemned a result that was
   merely weak. The original conclusion survives and sharpens: bare SDXL carries
   **+0.1527** against the fine-tune's **+0.0714**, less than half, so lesson 5
   holds after floor correction rather than in spite of it.
2. **The void probe was not marginal, it was null.** -0.0012 and -0.0059 against
   a construction that reads +0.0047 on noise. Yesterday's reading of it as
   "+0.008 above the floor, 1.5% of the span" was an artefact of the single
   constant.
3. **`gt_bsds_f1` has essentially no discriminative power on housei.** Feeding it
   **the correct answer** yields +0.0657. 38% of those tiles are near-blank and
   blank matches blank at high f1. This is stronger than lesson 6's "do not
   average across pools": on housei the metric does not work at all, so no f1
   comparison there means anything, cross-pool or not.

4. **The first rescoring of a real result reversed its reading, in the
   direction the raw numbers hid.** Track C's classifier looked flat against
   its baseline at raw f1 0.3202 vs 0.3164. Its output is sparser, so its floor
   is lower, so in signal it clears the baseline by +0.0258. **A sparse output
   is penalised by raw f1 and rewarded by signal** -- which is the direction
   that matters here, because deleting strokes is exactly what makes an output
   sparser. Any past comparison between outputs of differing density is
   suspect in the same way.

**Still unreadable.** Signal needs the outputs, so a figure whose outputs are
gone cannot be recovered, and **a raw value alone says nothing about where it
sits.** These remain unranked until rescored:

- Track A's round-1 **0.2514** and round-2 **0.2524**, and `manga_line`'s own
  0.2847 (192 tiles)
- Track D's instrumented snapshots, 0.2270 through 0.2715 (192 tiles)
- the old cs=1.0 default **0.1411** and the eleven-model band at cs1.0 around
  **0.13-0.15** (five tiles) -- the ones that prompted the request, still open,
  because realpairs' outputs are needed and their density is unknown
- the stroke-level delete-only oracles, **0.7425** and **0.5143** (Track C's
  *pixel*-level oracle is measured, above; these two are the stroke-level
  figures from the earlier decomposition)

Rescoring any of these is cheap where the outputs survive: no GPU, no training.
Whether they do is the first thing to check before quoting them again.

## Working Discipline

Added 2026-10-02 after a session in which the foundation itself produced a void
experiment. The failure was not forgetting a fact. Every fact needed was present;
what came loose was the **binding between a number and the configuration that
produced it** -- `cs=2.5` kept its "best" label and lost its "for SDXL bare, from
Track B" qualifier. That is what a long mixed-context session does, so the
mitigation is not to remember harder but to write bindings into artifacts, and to
let one check run without depending on anyone's attention.

These are deliberately few. This project has already shown that documentation
past a certain volume degrades rather than helps -- a briefing grew past 900
lines and stopped working -- so each rule below has to earn its place against an
incident that actually happened here.

1. **A number is not quotable without its configuration.** Write
   `cs2.5 (SDXL bare / 1024 / Track B)`, never `cs2.5 (best)`. Lesson 6 already
   enforces one instance of this (always report the conditioning map's own
   score); this generalises it. Four incidents trace to a bare number travelling:
   cs2.5 above; `0.2582` credited to a model when it was the preprocessor's;
   `line_width_p50 7.59` read as thick strokes when it was solid fill; the
   `0.651` endpoint-gap that was a tokeniser artefact.
2. **Every new measurement includes a cell whose value is already known, and if
   it does not reproduce, nothing else in the run is read.** Already the practice
   that caught the most: Track B verified its cs=1.0 column against historical
   values before trusting an eleven-model table; Track D reproduced 0.2514 as
   0.2513 before believing its snapshots; the void probe's conditioning-map cell
   matched Track C's 0.3164, which is how we know its plumbing was innocent and
   only its design was wrong.
3. **A retraction is a pass, not an entry.** When a number is withdrawn, correct
   every store that repeated it in the same sitting, and say what replaced it.
   `0.2582` stayed in memory as "best ever" for weeks after it was retracted.
   A reader who meets one retraction should be able to see which other numbers
   came from the same batch.
4. **Run `tools/audit_track_ledger.py` when picking this file up.** Read-only; it
   reconciles the ledger against the filesystem, finds tracks that exist but are
   unlisted and listed tracks that no longer exist, flags separate clones that
   `git worktree list` cannot see, and reports each track's commits-since-last-
   notice gap. It exists because the thirteen-day gap of 2026-09-17..30 was
   mechanically detectable the whole time and nothing was looking.
5. **The foundation session does not run experiments.** Its job is recording and
   routing. Theme work belongs in a track, started in a clean session from that
   track's briefing. This is a role boundary rather than a caution, because the
   caution failed.
6. **On what a picture depicts, the author decides.** Added 2026-10-04 from
   Track H. A session read that track's verified closed-eye seed as "an open
   mouth with teeth" off a 0.3x montage, wrote the retraction into the work log
   and propagated it into the findings; the user, who drew the page, restored
   it, and the full-resolution crop agrees with the user. Claude's reading of a
   drawing is a hypothesis for the author to confirm, never grounds to overturn
   their judgement -- and when a judgement is in doubt, pull the
   full-resolution crop instead of arguing from the montage. The cost here was
   not the misreading but the retraction built on top of it.
7. **Minimality is not the lever here, and two of its pieces are.**
   Considered and **declined** 2026-10-10: the `ponytail` skill
   (`github.com/DietrichGebert/ponytail`, MIT), a general "take the first rung
   that works" ruleset -- ask whether the code needs to exist, reuse what is
   there, prefer deletion, write the minimum, tag a knowing shortcut -- with
   `/ponytail-review`, `-audit` and `-debt` over the diff, the repo and those
   tags. It is **not** web- or game-specific and contains nothing about UI,
   which is worth recording because that is what it looks like from its README.
   Three reasons it is not adopted, in order of weight:
   **(a) Nothing in this project's recorded damage is of the kind it prevents.**
   Every incident in this file is a number travelling without its quantity or
   configuration (`0.2582`, `cs2.5`, `line_width_p50 7.59`, the "16-20%", the
   11,868 tiles), a void experiment, or an instrument measuring itself (Track G
   lost four instrument checks in one day, all on the checking side). None of
   them is over-building. The one case where excess did bite was a briefing that
   grew past 900 lines and stopped working, which is documentation volume and
   outside the skill's scope.
   **(b) What the ladder's first rung cuts is this project's main asset.** On the
   measurement side the rules deliberately ask for what nobody requested: an
   instrument check, four negative controls, a known-value cell, a no-training
   frequency table beside a learned number, pools kept apart instead of
   averaged. Without lesson 14's frequency table the aspect-ratio leak would
   still be read as placement structure; without an unrequested transpose and a
   50px shift, `preproc@budget` would have passed as placement on its off-by-one
   control alone. In a domain with no precedent, what makes a result refusable
   is the work itself, not overhead on it. The companion rule already in force:
   **a method is never rejected or scaled down because of its runtime.**
   **(c) The scoping that would have made it safe cannot be implemented.** The
   skill triggers on any coding task, so "tooling yes, instruments no" is not
   expressible; `full` is the default and `ultra` argues with the request, which
   runs against discipline 6 (on what a picture depicts, the author decides) and
   against the author's framings being this project's steering input. It would
   also sit as a fixed per-session cost in every track, and the cost that
   actually bites here is long mixed-context sessions, not code volume.
   **Two pieces are adopted on their own, without the skill**: an assert-based
   self-check in any new non-trivial logic (discipline 2 under another name),
   and a knowing shortcut tagged `shortcut: <limit>, <when to upgrade>` with
   those tags collected somewhere, which is the known-limit binding discipline 1
   asks for. **What would reopen this**: a UI that someone other than the author
   has to use, or a track's tooling becoming a real maintenance burden. Then it
   is installed project-scoped in that track's own tree, not by the foundation.

**When you suspect the knowledge has already muddled**, stop producing results
and run these in order: re-run the anchors (separates a broken tool from a broken
understanding), reconcile the index against the filesystem, then walk the numbers
this file currently quotes and check each still carries its configuration and a
traceable source -- marking the ones that do not. Resume after that, not before.

## Known Tool Traps

- **A control the reference arm itself fails cannot discriminate** (found
  2026-10-16 in Track J's NC-j8d, routed back as a reading correction, not an
  adoption request). The touch-merge adoption bar was `wrong <= 0.059`, taken
  from an earlier trial set where the calibrated chainer scored 0.039. On the
  311 trials actually used, the **reference arm scores 0.061** and so breaks
  the bar by itself, while the candidate scores an identical 0.061 (trial by
  trial: 34 gained, 0 lost). The leg was meant to ask "does this repair cost
  precision" and in fact asked "does it *improve* precision", which a
  zero-cost repair can never pass. **Write a cost leg as a difference from the
  reference arm** (`wrong_candidate - wrong_reference <= 0`), not as an
  absolute imported from another population. This is the fourth way a control
  is born meaningless, after the three already in the `moat` ledger.
- **The instrument graded its own pass condition, and leniently** (caught
  2026-10-18 by Track K itself, on the second run of K2-B'). The registration
  said each of three constructed target classes must land in the right box,
  30 of 30. The implementation set `stop_condition_passed` from
  `n_constructed > 0`, so the first run printed PASS on nine constructed
  targets of which three were misclassified. The track found it, tightened the
  code to the registration, and left the first PASS in the record. **A stop
  condition is code like any other: assert the registered threshold, by its
  number, in the same commit that writes the condition.**
- **A stop condition written about the material teaches nothing about the
  instrument** (routed 2026-10-18 to Track K). K2's A2 required the ink count
  along the chord to separate into a zero peak and a non-zero peak; it came
  out as a zero peak plus a thin tail, and the tail was a different mechanism
  (spur-pruning residue, 14 of 20 cases in two panels, 60% `face.other`). The
  collar was never the problem: **the population A2 expected does not exist in
  this material**. Whether a distribution is bimodal is not a property we
  control, so making it the instrument's pass condition means a differently
  shaped material leaves the instrument untested. **Register the decision
  rule and its thresholds, and make the stop condition constructed targets
  that must land in the right class.**
- **A one-directional fix needs a target that catches over-correction**
  (routed 2026-10-18). K's B'-1 excludes ink belonging to the candidate's own
  spans, which can only move cases toward "paper"; "ignore all ink near the
  tips" would pass the same stop condition and classify everything as paper.
  So a repair that moves cases one way is registered with a target on the
  other side -- here a gap a third line genuinely fills, which must still
  classify as ink.
  **T4 passed 30 of 30 on its first run (2026-10-18), and the attribution trap
  the registration named fired in 6 of those 30** (the darkest point was nearer
  the candidate's own span than the third line) and still classified as ink, so
  the exclusion is not over-reaching. The over-correction target earned its
  place on the first measurement -- and then earned it again, decisively: when
  the collar was redefined, the guard had to be rebuilt for the new definition
  (a third line *touching* the candidate shares its connected component), and
  the rebuilt **T4-touch came in at 21 of 30 while the other four rows were all
  30/30**. Without it the run would have been reported as a pass. **This rule
  is no longer a precaution; it has caught a false pass.**
- **A radius taken from the sample point collapses exactly where it is needed**
  (Track K, 2026-10-18, second failure of the same instrument). B'-1 attributed
  ink to the candidate's own span within a radius built from the ink width *at
  the sampled pixel*; the distance transform returns 0 to 2 at a stroke's edge,
  and grazing one's own ink always happens at the edge. T1/T2 failed 7 of 30
  with `d_own` 2.0-4.9px against radii of 2.0-2.4. The fix (B'-1b, approved) is
  the definition, not a constant: **whether a pixel belongs to a line's body is
  decided by that line's thickness, which is a property of the line, not of the
  pixel's position** -- so take the width at the nearest own skeleton pixel,
  `max(2.0, iw/2 + 1.0)`.
- **A fix derived from the failures passes on those failures by construction**
  (routed 2026-10-18 to Track K; it is Track J's IC-j12 lesson in another
  shape). B'-1b was built by inspecting the 7 cases that failed, so re-running
  on the same site band is diagnosis, not a check. The prescription: print the
  diagnosed band to show the 7 are fixed, labelled as diagnosis, and decide the
  30/30 bar on a **disjoint band of sites** drawn mechanically from the same
  18,194 candidates with a new seed. Never aggregate the two bands. Also
  registered before the run, as the foundation's own prediction: **T1/T2 will
  not reach 30/30 on the fresh band**, because one case of nine was already
  unreachable at any body radius -- and if that is how it fails, the conclusion
  is that the chord is the wrong instrument, not that the radius needs a third
  value. The number of fixes derived from the same material is budgeted, and
  two were spent.
- **The author of a repair is the wrong party to predict its yield** (observed
  across three Track K registrations, 2026-10-18). The track's mechanism
  readings were right all three times; its yield predictions were wrong all
  three times ("8 of 9 will be repaired" came in at 1 of 7; "all five rows
  pass" came in at 21 of 30), and a fourth mis-estimate the same day was of **what a control
  returns under the null** (1 against 1.3476), which is the same side of the
  line -- so the division widened: **the author registers the mechanism, the
  foundation registers the yield and the control's null value**, the latter
  written as a pair with the statistic's name.
  **The empirical half of this is withdrawn, the same day.** The claim that the
  track's mechanism readings were unbroken lasted hours: its registered
  "`comp_same = 1` is the majority of the 59" came in at **4 of 59**, which is
  a mechanism reading, making it 4 of 5. And in the same exchange the
  foundation named the 59 wrongly, in the very notice that told the track its
  own name overclaimed. **Neither party has a clean record on what the material
  contains**, so the procedural split stands on who is easier to fool rather
  than on a tally, and the rule that did the work is simpler: **split by
  subcase and print it before putting a name on the material.**
  The foundation's predictions of **where a guard
  would bite** were right all three times, which is the cheap direction: the
  mechanism was handed to it by the track's own notice. The 3-0 is not evidence
  that one party judges better -- the two prediction tasks differ and their
  biases point opposite ways. The division adopted from here: **the repair's
  author registers the mechanism, the foundation registers the yield.**
- **A null value belongs to a statistic, not to a study** (Track K's K7,
  2026-10-18, stopped by its own control). The registered negative control
  permuted part labels within `order` and required the result in [0.95, 1.05].
  It came out at 1.3476, because permuting within a stratum preserves that
  stratum's composition, so a **crude** RR pooled over strata does not return
  to 1 when the within-stratum association is destroyed. The same permutation
  **is** centred on 1 for **MH-RR**. So the band was correct and was attached
  to the wrong statistic: the failure was not the material, not the band's
  width, and not the control's idea. **Write the null value and the statistic
  as a pair in the registration**, and when a control fails, ask which of the
  two is wrong before touching either.
- **A coincidence between two statistics is a degeneracy check before it is a
  finding** (same stop). The permutation median of the crude RR (1.3476) and
  the observed MH-RR (1.3418) agreed to 0.006 although they estimate different
  things. The boring explanation to rule out first is degenerate
  stratification: singleton strata make a within-stratum permutation the
  identity and contribute nothing to MH, so both numbers can be driven by the
  same few strata. Required before interpretation: the number of strata, their
  size distribution, how many are singletons, and what share of the units sit
  in strata that carry both exposure levels and both outcomes.
- **"Compared against everything else" can be the conservative choice, and
  that is checkable** (ruled 2026-10-18). Track K suspected its pre-registered
  exposure (mouth versus all other parts) of being a design error once the
  per-part table showed mouth 4th of 8. Computed from the track's own table:
  the highest-rate part, `face.other` at 0.782, sits in the comparator, and
  removing it raises the exposed RR from 1.6501 to 2.2356. **A comparator that
  contains the strongest non-exposed group dilutes the estimate**, so the
  registered design did not manufacture the result and does not need replacing
  -- while moving the primary onto the table would mean choosing the maximum of
  eight cells post-hoc.
- **A count from a run whose stop condition failed is usable only if its
  predicate never touches the failed path -- and it still needs a scale**
  (ruled 2026-10-18 on Track K's 32.4%). The unmeasurable box survived because
  it is decided by the registered ink rule and geometry alone, with no gray
  classification in it. Two things it still lacked: a name that does not
  overclaim (it says the tips' ink lies in one connected component, not that
  the gap is not a gap -- 4 of 30 constructed real fills landed in the same
  box) and a trivial arm, because the predicate is nearly monotone in gap
  length, so the bare percentage has nothing to be compared against.
- **A definition can be right in mechanism and one case short in coverage**
  (Track K's C-1, 2026-10-18, and the right way to record a partial result).
  The collar redefinition solved exactly what it was derived for -- the rows
  that failed at 27/30 returned 30/30, and the retraction mechanism is closed
  -- and failed only where the filling ink is continuous with the candidate.
  Recorded as coverage missing at one place, not as a refutation of the
  reasoning, because the distinction changes what the next registration
  inherits.
- **An anchor whose correct answer is missing from the option set tests
  nothing** (ruled 2026-10-18 on Track K's author request). The sheet mixed 8
  box-A pairs -- continuous ink -- among the real cases, with a bar of "at most
  2 of 8 called a skipped line", to catch a crop that hides a connection. But
  the options were skipped-line / faint / can't-tell, so the truthful answer
  for an anchor, *it is not broken*, was absent: the author can only fall back
  on can't-tell, and "the presentation hid the connection" becomes
  indistinguishable from "the author was unsure". Required change: a fourth
  option, **"not broken (looks connected)"**, after which the anchor bar
  becomes option 4 at 6 of 8 or better with at most 1 of 8 called a skipped
  line. The same option then earns its keep on the real cases too -- an answer
  of "not broken" on a genuine box-B case says the 2x crop is hiding a real
  gap, which checks the decision to drop the 55 sub-collar cases.
- **An anchor set identifiable by a nuisance variable passes for the wrong
  reason** (same ruling). Box A's gaps are mostly 2-8px while the 127
  answerable cases have a gap median of 10.82px, so anchors drawn from box A at
  large are separable by gap length alone and the author could score them
  without ever looking at the ink. Box A holds about 48 pairs at 8px or more
  (27 + 10 + 7 + 4 across the bins), so the anchors are drawn **matched on gap
  length** to the 40 sampled cases.
- **A bar on raw agreement is confounded by the answer marginal** (same
  ruling). Reaching 6 of 8 on a repeat check is a 1.97% event when three
  options are answered at random, which makes it a real bar -- but if the
  author's answers run 80/10/10 the chance agreement is 0.66 and 6 of 8
  becomes a **45.2%** event, so the same bar means nothing. The marginal is
  printed beside the agreement, the bar is agreement at 6 of 8 *and* above the
  marginal-implied chance agreement, and when one option takes 70% or more the
  repeat check prints **uninformative**, not pass. Also: a repeated tile must
  sit at least two sheets from its first appearance, or what is measured is
  short-term memory of the tile.
- **A robustness arm must not contain a repair of its own** (Track K, found and
  corrected by the track, 2026-10-18, after the foundation required the arm).
  The harsher arm -- re-deciding continuity at `thr = 128` with no grow --
  printed 8 survivors because the implementation snapped a tip to the nearest
  ink within 2px when the tip did not land on the `thr` mask. The arm exists to
  be **harsher**; the snap made it gentler, and the correct count is **4**
  (provable without rerunning: `ink_plain` is a subset of `ink_hysteresis`, so
  a minimax bottleneck at or above 128 admits no path in the plain mask at
  all). An arm that carries a convenience of its own is not an arm, it is a
  second primary value. Had the arm not been required, "8 survived" would have
  been the reported number.
- **A diagnostic that predicts a fix's yield must run the instrument's own
  code path** (Track K, 2026-10-18, the most useful of the day). Before
  measuring B'-1b the track estimated it would repair 8 of 9 known failures; it
  repaired **1 of 7**, because the diagnostic script measured with a fixed
  3.0px collar while the instrument uses `max(2.0, 0.5 x tip ink width)`.
  **The prediction was made outside the instrument, so it was a prediction
  about a different instrument.** The fix's mechanism was still right -- the
  distance transform does collapse at an edge -- and only the yield estimate
  was wrong, which is the correct way to split it. A yield estimate quoted in a
  registration must come from the same call path it is predicting.
- **A threshold compared against a float distance can fail by 2e-14** (Track
  K, 2026-10-18). One of the three band-(b) failures was 3.000000000000021
  against a radius of 3.0. A case that misses by that much is not a
  measurement. The repair adopted here removes the comparison instead of
  tuning it, but any registration that compares a computed distance to a
  threshold **writes the tolerance as a number, or compares in integer pixel
  arithmetic**.
- **Skeletonization retracts at a line's end, so "within half a width of the
  skeleton" is not a definition of the line's body there** (measured by Track
  K, 2026-10-18: retraction median 10.8px in this material). The definition
  holds mid-line and fails at the ends, which means **any instrument that
  starts a measurement at a tip starts inside ink that has no skeleton**.
  Relevant to F, G and J as well as K, which is why it sits here and not only
  in the track entry.
- **A registered exit clause that names a successor *method* is a guess; name
  the *class* of change** (the foundation's error, 2026-10-18, caught by Track
  K with evidence). The exit written for B'-1b said the next attempt was
  limited to "the arc through the two tangents". The arc starts from the same
  two tips and traverses the same first few pixels of the candidate's own ink,
  so it could not have repaired any of the three failures: the clause named a
  method chosen by the foundation's guess rather than by the mechanism. The
  right form is **"the next change must be a definitional change derived from
  the mechanism; a constant may not be moved"**, which admits the collar
  redefinition and excludes the arc. Track K declining to follow the clause's
  letter, and stopping to ask instead, was correct.
- **Disjointness should be a property of the construction, not of a draw**
  (Track K's amendment to this file's prescription, 2026-10-18, accepted as the
  better form). The foundation prescribed a fresh band of sites under "a new
  seed 20261018b"; two independent permutations can and do collide, so the
  track used **the next interval of the same permutation** instead -- band (a)
  = 0..299, band (b) = 300..599, band (c) = 900+ -- where no overlap is
  possible by construction.
- **Replacing an approximation with the concept it approximated is a different
  kind of change from moving a number** (the principle that admitted Track K's
  third fix, 2026-10-18). Two fixes derived from inspecting failures had
  already been spent, and the budget was real. What let the third one through
  is that it introduces no quantity anyone can tune: box B asks whether the
  gap *between two strokes* is filled, the candidate's own ink is by
  definition not in that gap, and the collar -- a fixed 2-3px, then
  `max(2.0, 0.5 x iw)` -- was always an approximation of "skip your own ink".
  The connected-component exit states the same intention exactly, and takes
  its boundary from the hysteresis ink rule that was registered long before.
  **The test to apply to a late fix is not how many times the instrument has
  been touched, but whether the change hands anyone a new number to choose.**
- **A whole-file sha of a shared dependency is coarser than the dependency**
  (the foundation's error, 2026-10-18, corrected by Track K with evidence and
  then verified here). The foundation forecast that Track J's IC-j9 adoption
  would halt Track K's next K2 run on its sha check, and prescribed fixing all
  comparisons at the pre-adoption commit `1a82ea5`. Both were wrong: K's
  `J_SHA` was the post-adoption value from the start
  (`tools/stroke/k2_missed_candidates.py:47`, recorded in
  `results/k2_missed_candidates_20261017/check.json`), so every K2 number --
  box A 521, box C 79, box B 88 -- is already post-adoption and the tables need
  no split; and while `tools/evaluation/stroke_chain.py` changed by 31 lines
  added and 6 removed between `3d78816` and HEAD, the two functions Track K
  actually calls, `arm_geometry` and `perp_offset`, are **byte-identical**
  (checked here by extracting each body from both commits). The prescription is
  withdrawn. Keep the halt wiring -- halting is correct behaviour -- but note
  that a file-level sha halts on changes that cannot reach the caller, so the
  halt alone carries no information about whether a number moved; printing a
  sha over the called functions' bodies makes it informative.
- **A set-level zero on a regression count is not a cost bar, it is a refusal
  to trade** (Track J, 2026-10-18, found by its own measurement). `newly_wrong
  = 0 by set` made 0 of 30 sweep cells proposable, including two that broke
  nothing and cut `wrong` from 0.061 to 0.048 and 0.035; the next
  registration's surviving `broken = 0` refused a cell that gained 45 trials
  and cut `wrong` from 23 to 16 counts because it broke 2 trials of 379. Only
  an exactly additive repair clears such a clause, and IC-j9's adopted repair
  happened to be exactly additive, so the clause was carried for a week
  without ever being tested against a real trade. **Put the tolerance in the
  bar as a number, before the material is touched** -- and once a reserved set
  has been read, that number can no longer be written for that challenger:
  it needs a new set, which here exists (the chainer uses panels [0:171] of a
  4,049-panel corpus).
- **Selecting on a zero-valued count manufactures a zero** (Track J,
  2026-10-18). The challenger cell was picked by one mechanical rule, highest
  score among discovery cells with `broken = 0`, and broke 0 of 311 there and
  **2 of 379** on the never-read set. The recall and merit legs transferred
  with their direction predicted; only the cost half did not. This is the
  winner's curse in its count form: **if a grid cell is chosen partly because
  a count was zero, expect that zero not to replicate.**
- **A registered expected value must name the entry or the results file it
  came from; a date and a sample size are not a citation** (Track J,
  2026-10-18). IC-j10's second expected value cited "an r-census on 60 held
  panels recorded 2026-10-14". There is no entry on that date, no commit (the
  history goes 10-11 to 10-16) and no results file holding the number. The
  measurement came out at 0.8588 against a band of 0.63 +/- 0.10, and the band
  refuted nothing because **it was never a measurement**. The track found it
  by going back to reread the census as its own registration instructed. This
  one is not a flaw in a control's design: it is a registration resting on a
  measurement that does not exist.
- **The cached intermediate is not always the one that made the data, and the
  foundation approved the wrong path** (caught 2026-10-16 by Track K's IC-k5b,
  which failed 0 of 6 before anything was read). Track K's K2 registration --
  approved here -- pointed at the cached skeletons in
  `panel_skeleton_20260917/`, built with a plain `ink_mask` at threshold 128.
  The strokes the author actually labelled came from
  `export_panel_tokens_v2.py`, which uses **hysteresis ink**
  (`ink_mask(thr=128, grow=200)`: a pixel joins if it is below 200 and its
  8-connected component reaches below 128), skeletonizes that mask on the fly,
  and passes `dist * 2` as the width field. **Track F's own docstring says so
  in one line** -- "the skeleton has to be recomputed because the ink rule
  changed, so this script caches it alongside the tokens rather than reading
  the 2026-09-17 cache" -- and records why it matters: one panel goes from 258
  components at 128 to **64** at grow 200, because a plain cut breaks faint
  anti-aliased lines into dots. **How the ink breaks apart is exactly what a
  connectivity measurement is about**, so the cache would have measured
  different lines from the ones the author saw. After the path was matched,
  IC-k5b passed 6/6 with no difference in a single stroke. **Check which code
  path produced the labels, not which artefact is on disk**, and read the
  producing script's docstring before reusing a cache.
- **A control that removes the artifact along with the signal cannot separate
  them** (caught 2026-10-16 in Track K's K4a, before it was run). K4a proposed
  Spearman rho >= 0.30 between a collapsed bridge's length and the local ink
  width, with the control being a shuffle of the widths among bridges inside a
  panel. But a span *is* a bridge only if `len <= max(3.0, 1.5 x median
  width)`, so within the selected set the length is bounded above by a linear
  function of the width and a positive rho follows from the cut itself. The
  shuffle destroys that selection-induced pairing exactly as it destroys a
  real effect, so it falls to ~0 either way. **The separating control is a null
  model, not a shuffle**: redraw each bridge's length uniformly on
  `(0, max(3.0, 1.5w)]` -- the world where only the cut operates -- and require
  the real rho to clear that band. Two cheaper forms work as well: drop the
  selection and measure over every two-node span with the cut drawn as a line,
  or test the claim's own magnitude, since `len / width` piling at 1.5 is the
  cut while mass near 1.0 is the source comment's "about one ink width".
  **Also watch the kink**: below a median ink width of 2.0px that threshold is
  the constant 3.0 and carries no width dependence at all, and the project's
  measured fill-excluded widths (1.91 housei, 2.74 ako5) sit on top of the
  kink, so the two regimes must be reported apart or the kink reappears as a
  correlation.
- **An instrument built from artificial damage cannot contain damage the
  material never shows** (found 2026-10-16, Track K reading Track J's IC-j6).
  Cut-and-rejoin trials made by drawing an occluder over real ink contain **no
  deliberate gaps at all** -- 0 of 90 free-end cases -- because an occluder is
  drawn by adding ink and leaves no white paper; and the chainer collects only
  arms with a node id >= 0, so a free end is not even a candidate and a
  deliberate gap joins at **0% by architecture**. "The artificial distribution
  *may* differ from the real one" is the general caution; for a defect the
  construction cannot produce, it **will** differ, and only real material
  covers it. Name the defect classes the construction excludes, before
  reading the instrument's headline.

Collected 2026-09-13..17 from `inbox/`. Every one of these was hit by a track
that had no way to know, and several were hit twice. **Two are unfixed bugs**
-- see Next Actions.

- **`bipartite_match_f1` (the implementation behind `gt_bsds_f1`) can take
  tens of seconds to 12+ minutes on a single tile.** scipy's
  `maximum_bipartite_matching` approaches worst case on particular edge-point
  configurations, and it is not predictable from image density -- similar tiles
  hit it or do not. 98 of 584 pairs (16.8%) timed out in one run. Environment
  causes (cv2 thread contention, sharing a process with PyTorch/CUDA) were
  tested and ruled out. **If a batch evaluation appears hung, suspect one slow
  tile before suspecting a hang.** The metric itself is unchanged -- it is
  shared and validated, and every comparison on record depends on it.
  **The mitigation is now shared, as of 2026-10-09**: `bsds_guard.match_f1()`
  in `tools/evaluation/` dispatches one tile to
  `score_bsds_f1_one.py` under a hard cap and returns `None` instead of
  blocking, so a caller records a miss and the batch stays readable. Default
  cap 300s; `--bsds-timeout 0` restores the old uncapped in-process call.
  `evaluate_fixed_outputs.py` and `condition_roundtrip_fidelity.py` both go
  through it and both gained a `bsds_timed_out` column, with means over the
  survivors and the miss count printed. Verified on 10 tiles: the guarded and
  in-process CSVs are byte-identical, a 0.001s cap reports 10/10 missed rather
  than scoring anything, and `bsds_guard.py` run directly self-checks both.
  The per-tile pattern this generalises came from
  `../lineart-pair-signal/tools/evaluation/vae_roundtrip_score.py`, which is
  still the reference for a whole-batch runner with a thread pool.
- **The `"train" if name.startswith("housei") else "test"` split heuristic was
  wrong for 168 of the 192 `holdout_lineart_family.txt` tiles, and it was in
  nine places, not one. Fixed 2026-10-09.** Only the 24 `lineart_004_*` tiles
  live in test; the default sample lists were all `lineart_004_*`, which is why
  nobody noticed. Beyond `evaluate_fixed_outputs.py`, which is where it was
  reported, the same line sat in `evaluate_stroke_stability.py` and six
  `tools/compare/*.py` scripts. All nine now call
  `tools/evaluation/gt_paths.resolve()`, which decides by looking for the file
  and raises naming the tile when it is in neither split. Verified: 192/192
  tiles resolve, 168 to train and 24 to test, and a nonexistent tile raises.
  **And the blast radius recorded here was overstated, which is worth saying
  plainly rather than quietly dropping.** This file asked for every past
  192-tile number produced through `--split auto` to be re-checked. **No
  re-check is needed.** The two split directories share no basename at all --
  14,102 train tiles against 1,620 test tiles, zero overlap, measured
  2026-10-09 -- so the heuristic could never read a *different* tile's GT. It
  either resolved correctly or pointed at a path that does not exist, and both
  the evaluator and the compare scripts raise on a missing file. The bug blocked
  runs; it did not corrupt numbers. The only figures it could ever produce were
  from all-`lineart_004_*` or all-`housei` sample lists, both of which it
  resolved correctly.
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
- **The axis trap in Track F's `cluster_set`, settled 2026-10-04 -- and the
  cause this file carried until 2026-10-09 was wrong.** The withdrawn version,
  reported by Track G on 2026-09-23 and repeated here, was that
  `results/cluster_set_20260919/meta.csv` has `cy` and `cx` swapped. **It does
  not.** Track H settled it against an artefact outside the analysis chain --
  the line PNGs under `dataset/<source>/line/` -- on 42 panels: **`meta.cy` is
  y and `meta.cx` is x, exactly as the columns are named.** What is transposed
  sits one layer down: **`strokes.npy` and `cluster_set` `pts` are stored
  (y, x)** -- read as (x, y) only 0.11 of the points land on the PNG's ink,
  read as (y, x) 0.999 -- and **corpus `dims` is (W, H)**, not (H, W). Track
  G's companion claim that the line PNGs are stored transposed was the same
  misreading from the other side and is withdrawn too: the PNGs are upright.
  **What survives is the whole consequence.** The corpus position bins were
  built as `py = cy/dims[0]` = y/W and `px = cx/dims[1]` = x/H -- mismatched
  denominators under either reading -- so they are genuinely distorted.
  **The quantity that was 16-20% here until 2026-10-09 was mislabelled**, and
  the correction matters because it is what a track trying to reproduce the
  figure will aim at: the original 2026-09-23 record says *"py pins to top bin
  15 at 15.9%, px at 20.1%"*, which is the **share of clusters whose ratio
  reaches or exceeds 1 and is clipped**, not top-bin occupancy. Verified here
  on 2026-10-09 against the corpus: **top-bin occupancy is 17.96% (py) and
  22.39% (px)**, the top row plus top column is **40.3%**, and the clipped
  share is the recorded 15.9% / 20.1% -- the original note's companion figures
  ("cy <= H is 84%", "cx <= W is 80%") are that clipped share from the other
  side, 84.13% / 79.93%.
  **The blast radius is now measured, and the verdict is "big" (Track G,
  2026-10-09, `results/posbins_20261009/`).** Rebuilding the bins on matched
  denominators moves every position figure: marginal **7.563 -> 7.957**, cloze
  **6.315-6.388 -> 7.785-7.792**, two-stage **6.658 -> 7.880**, set-at-once
  **6.680 -> 7.901**, the largest delta 1.449 bits against a pre-registered
  "big" threshold of 0.270. **No recorded verdict flips** -- every sign holds --
  but the margins collapse: Track F's position gate from 1.175 bits to 0.165,
  Track G's two-stage from 0.905 to 0.077, the cloze-band failure from 0.270 to
  0.088. So the position-bits figures all need replacing, and no past
  conclusion needs withdrawing. The corrected corpus is kept at
  `../lineart-panel-generation/results/posbins_20261009/corpus_fixed.npz`, with
  words, scales, order and the train/test split identical to the original and
  only the two position tokens rebuilt.
  **What the distortion actually was, which is the part worth carrying**:
  `y/W` collapses toward zero on wide panels and exceeds 1 on tall ones, so the
  position tokens were **encoding the panel's aspect ratio**. With no learning
  at all, a frequency table conditioned only on a 16-way aspect-ratio bin
  reaches **6.510 bits** against the 7.563 marginal -- **86% of the advantage
  the cloze transformer obtained by looking at every other cluster in the
  panel** (1.054 of 1.228). On the corrected bins the same table gains exactly
  nothing (7.946 against a 7.946 marginal) and the transformer's remaining edge
  is **0.17 bits**. Corroborating, all without training: the correlation
  between a panel's mean `py` and `log2(H/W)` falls from **+0.909 to +0.012**
  (`px`, -0.913 to -0.120), the between-panel share of bin variance from
  0.509/0.494 to 0.122/0.099, and the clipped share from 15.9%/20.1% to
  **0.0%/0.0%**. Reproduced here independently from
  `../lineart-stroke-grammar/results/grammar_corpus_20260920/corpus.npz`:
  top-bin 0.1796/0.2239, top row+column 0.4034, correlations +0.909/-0.913,
  and the aspect-only table at 6.517-6.528 depending on the smoothing.
  **One convention to know before comparing any of these numbers**: the
  "marginal" is the **sum of the two token marginals**, H(py) + H(px), not the
  joint. On the original bins that overstates the baseline by 0.190 bits (joint
  7.373, measured here), because the broken normalisation correlated the two
  axes; on the corrected bins the two are nearly independent (joint 7.951
  against the sum 7.954), so the corrected margins are not artefacts of the
  convention -- an honest train-fitted 256-cell table scores 7.945 on the test
  panels and the cloze still beats it by 0.16 bits.
  Track G's own 2026-10-09 tool `tools/gen/framerel.py` deliberately does not
  use the bins at all, reading `meta.csv` by column name and taking panel size
  from the line PNG.
  **Track G's own GT analyses are numerically unaffected, with their axis
  labels swapped.** Verified here on 2026-10-09 by reading
  `tools/gen/placement_gt.py`: it takes `x = meta.cy` and divides by `dims[1]`,
  and `y = meta.cx` divided by `dims[0]`, so the two misreadings cancel and
  every normalised ratio it computes is the correct one wearing the other
  axis's name. Distances and symmetric statistics hold; **anything that reads a
  direction -- up/down, left/right, the sign of an angle, "the word above this
  one" -- is mirrored across the diagonal.** Track F's files were not modified
  and Track F is held, so this waits for whoever resumes it; its briefing
  carried the wrong cause from 2026-10-04 and was corrected on 2026-10-09.
- **The line-PNG check has no sensitivity to the orientation of a face box.**
  Reported by Track G on 2026-10-09 and verified here against
  `../lineart-face-words/results/h3_face_parts_20261004/faces.csv`. Lesson 12's
  remedy -- score points against the ink in `dataset/<source>/line/*.png` --
  only exercises `strokes` / `pts` / `meta`. A face box `(x0, y0, x1, y1)` can
  be read either way round and the PNG check will not move at all, so a track
  that uses the detector boxes needs its own cheap check: **does the box fit
  inside the panel.** Read by column name, 7.3% of the 5,531 boxes stick out
  and the 90th percentile of the overflow is **0.4 px** -- the detector
  clipping at the edge. Read with x and y exchanged, **40.5%** leave the panel
  and the 90th percentile is **1,018.6 px**. (Both fractions count overflow
  beyond a 2 px tolerance, which is what reproduces Track G's numbers; the
  overflow percentiles are over all boxes.) It discriminates because panels are
  not square: 94.6% of the 5,345 panels differ by more than 5% between the
  sides (94.3% of the 3,821 that hold a box), aspect ratio median 1.56. Any
  track handling the boxes -- H and G today -- should run it before reading a
  direction off a figure; it is shared here as
  `tools/evaluation/face_box_check.py`, which takes the csv as its one argument
  and reproduces the four figures above.
- **A negative control has to name one concrete error, not a direction.**
  Track G's first instrument check on 2026-10-09 failed its own negative
  control and the fault was in the control, not the data: the registration said
  "exchange x and y", and the implementation exchanged the axis order of `pts`
  while leaving the cluster centre `(cx, cy)` correct. That is a hybrid that
  cannot occur -- it transposes a prototype's local offsets about its own true
  centre -- and it scored **0.3854** against a <= 0.2 bar while the correct
  reading scored 1.0000. Re-run one error type at a time: exchanging x and y of
  the finished points scores **0.0095**, confusing `meta.cy` / `meta.cx` (the
  known defect in Track G's `placement_gt.py`) **0.0020**, a 50 px shift
  **0.0833**, and the impossible hybrid **0.3854** -- and that last one is high
  **only for small clusters** (0.94 at 63 px diameter, 0.71 at 69 px, 0.12 at
  303 px), because a local transpose about the right centre barely moves a
  small thing. The lesson is the phrasing: **"swap x and y" is a different
  mutation at every layer**, so a control is written as the specific confusion
  a real caller could make, one per row, each with its own expected score.
  **Two more ways the same check went wrong on 2026-10-09**, both from Track
  G's position-bin run, where the instrument checks failed four times and the
  measured side never did:
  (i) **A control can be vacuous by construction.** The mirror control for the
  rebuilt bins -- "read `(x/W, y/H)` instead of `(y/H, x/W)`" -- is the same
  corpus with its two position tokens exchanged, so every symmetric statistic
  (saturation, clipped share, marginal entropy) is **identical by definition**:
  0.0500 against 0.0500, 7.956639 against 7.956639. It passes whatever the code
  does. **An axis-name swap cannot be detected by a symmetric statistic at
  all**; only a check against something outside the coordinate system can see
  it -- here "do the dims fit the panel", 100% as (W, H) against 3.3% as
  (H, W). That is lesson 12 again, one layer up: the control inherits the
  blindness of the statistic it is built from.
  (ii) **A known-value cell has to record which quantity, not just the
  number.** The check aimed at this file's "16-20% saturating the top bin" and
  failed, because the figure in the original record is the **clipped share**
  (15.9% / 20.1%), while top-bin occupancy is 17.96% / 22.39%. Both are near
  20% and neither reproduces the other. The number alone is not a reproducible
  anchor; the quantity, the denominator and the population belong next to it,
  and that is now fixed in the axis trap above.
- **`cluster_set`'s unit is built from proximity alone, so it mixes parts and
  silently drops strokes.** Measured by Track H on 2026-10-09 against 1,003
  author-assigned instance labels inside detected face boxes. Clusters come
  from a Louvain cut (resolution 2.0) of a 4-nearest-neighbour stroke graph
  (`tools/stroke/label_clusters.py`) -- no orientation, no connectivity,
  nothing semantic -- so a brow and the bangs above it become one cluster while
  a part drawn in two separated pieces splits in two. **41% of the 575 labelled
  instances that contain a face part are mixed**, brows worst at 98 of 131
  (75%). And `build_cluster_set.py` admits only clusters of 3-32 strokes:
  **14.6% of all 2,168,753 strokes are in no candidate at all** (3.9% in
  clusters of 1-2, 10.7% in clusters of 33+), which structurally excludes parts
  drawn with one or two strokes -- nose, mouth, closed eye, exactly the ones
  hardest to label. Any claim of the form "word W means X" inherits all of
  this, and so does every per-word statistic in Tracks F, G and H.
- **`lineart_003_000` has a blank ground truth and is in
  `holdout_lineart_family`.** 480x480, every pixel 255 -- the only blank GT among
  the 192 (next lowest ink is 0.0141). Any f1 or signal computed against it is
  meaningless, so it silently contaminates every 192-tile average on record.
  Found 2026-10-04 when a human judge rejected it as "not usable line art",
  correctly, while rating GT. Exclude or flag it when reporting; past averages
  carry it. **How much it carries was then measured, by Track I before it
  closed: a uniform +0.001 shift that changes no comparison on record.** So
  this is a reporting-hygiene item and not a reason to re-run anything --
  worth stating plainly, because "every average on record is contaminated"
  invites a re-measurement that the size of the effect does not justify.
- **A summary statistic taken over *all* ink is a reading of the solid areas
  whenever fills are present.** `2 * median(distance-to-paper over ink)` reads
  **27.15px** on an `ako5` koma panel and **5.48px** on a 480 tile of the same
  material at the same drawing scale -- it looks like a 5x scale difference and
  is not one; the median is sitting inside the blacked-in areas, which are a
  larger share of the ink in the larger window. Measured over ink *outside* the
  fill mask the two agree at **2.74 and 2.74**. Found by Track J 2026-10-10,
  whose own first scale-free argument was built on the wrong leg; reproduced
  here over 24 panels and 197 tiles with an independent exact-EDT
  implementation (18.03 against 4.47 over all ink, **2.828 against 2.828**
  outside the fill). `line_width_p50` has exactly this exposure, and it is the
  same confusion that once made `housei` GT's 7.59 read as deliberate thick
  strokes (lesson 6). The rule: **before comparing any per-unit ink statistic
  across two window sizes, recompute it outside the fill mask.** Note also that
  the fill-excluded width, which lesson 6 records as having "collapsed to ~1.9
  everywhere", does separate the pools (1.91 `housei` against 2.74 `ako5`) and
  is invariant to the window -- which makes it the right instrument for a
  cross-unit scale check even though it is useless for the question lesson 6
  was asking it.
- **An arm can beat its off-by-one control and still carry no location
  information at all -- what it is scoring is the *amount* of ink.** Track J's
  `preproc@budget` (the preprocessor handed GT's own ink count, so that only
  placement was left to it) reaches micro IoU **0.1618** on `ako5` against
  0.0615 for the off-by-one control, which reads as a clear pass. But
  **transposing the prediction costs 0.0012, and shifting it 50px costs
  0.0003**: two transforms that destroy location change nothing, so the overlap
  is not placement. The only control that bites is re-pairing with another
  panel's GT, and that gap is just the between-panel difference in *how much*
  is blacked in. macro IoU at a seventh of micro (0.0230 against 0.1618) says
  the same from the other side. So: **a control that removes location must cost
  a localised arm something; if the score survives it, the arm is scoring
  quantity, and an off-by-one control alone will wave it through.** This is the
  sibling of the box-orientation trap above and of lesson 12 -- there the
  transpose was the error nothing could detect, here the transpose is what
  detected a different kind of false positive. One reporting note from the same
  report: the verdict word "localised" was printed **without the threshold it
  was judged against**, which is the records failure this file has now logged
  three times in two days; a judgement word travels with its bar or not at all.
- **Per-pair alignment scores now exist for all 8,467 training pairs**:
  `../lineart-pair-signal/results/pair_alignment_strata_20260915/per_pair.csv`
  (rough / manga_line / lineart_coarse, each with `le3`, `3to8`, `gt8`,
  `chance_le3`, `aligned`, `density`). Usable for weighting or excluding
  training data. Chance is measured by rotating each conditioning map 180
  degrees; horizontal flip was rejected because panel borders survive it and
  inflate the baseline.

## Approaches Never Taken Up

Recorded so they are not re-asked from scratch, and so that "we must have tried
that" is not assumed.

**ControlNet + IP-Adapter (image-prompt / image-embedding conditioning) has
never been tried, and was never even proposed.** Surveyed 2026-09-30 across
every worktree's `doc/`, `inbox/`, `outbox/`, `experiments/`, `scripts/`,
`tools/`, all branches' commit messages, and the Directions 1-9 survey in
`doc/model_directions.md`: zero occurrences, of the name or of the concept
(`image_encoder`, `CLIPVision`, T2I-Adapter, reference-only, unCLIP-style image
conditioning, exemplar conditioning). Direction 3's "learned line-art encoder if
available later" is a perceptual-loss feature extractor, not conditioning, and
Track E's "CLIP features + linear" feeds a judge, not a generator.

**It fell off by expiry, not by judgement.** The one sentence in the corpus that
would have covered it is `doc/work_log.md` 2026-08-08, Next Action 2: after
finding that stacking the unconditional line-domain LoRA on the ControlNet was
actively harmful, it says to *"revisit whether any form of style adapter is safe
to add only after the paired fine-tune itself is working."* The paired fine-tune
never started working -- Tracks A and B both closed below the preprocessor
baseline -- so the revisit never triggered.

**Tooling was never the constraint**: diffusers 0.39.0 with `IPAdapterMixin`,
and both `StableDiffusionControlNetPipeline.load_ip_adapter` and
`set_ip_adapter_scale` are available in the venv today, no new dependencies.

**Assessment (2026-09-30), against what is now measured.** It does not address
the failure in lesson 8, because that failure is *spatial*: where the
conditioning map lacks a GT stroke the model draws it 9.3% of the time, and
output skeletons sit near the map (36%) rather than near GT alone (9%). An
IP-Adapter embedding is pooled and spatially unlocalised -- it carries
appearance, not where a stroke goes -- so against this project's own
decomposition it addresses neither half: not which of the preprocessor's
1.4x-GT strokes to delete, and not the ~16% of GT stroke length the map never
finds that caps oracle recall at 0.604. The closest analogue this project has
actually measured points the wrong way: the 2026-08-08 result above, where a
strong global style prior **overrode** the ControlNet's conditioning.
Architecturally IP-Adapter sits nearer that failure than a fix -- it injects at
cross-attention (`attn2`), which is precisely the layer
`scripts/pnp_line_from_rough.py` deliberately left alone, its docstring naming
`attn2` as carrying the signal it wanted to keep steering with.

**Where it would fit is a different axis.** The one thing training demonstrably
moves is tone (near_white 0.13 -> 0.83; fixed-`t` loss correlates with paper
white at r -0.98 and not with f1), and a reference-image channel is the right
instrument for tone. That makes it relevant to Track E's question -- aim at the
base model's own look rather than at GT -- and most natural of all in a
line-art-to-line-art framing, where spatial content comes from the source and a
clean exemplar supplies style. **Note that no line-to-line theme exists in this
project**: `line2line` and equivalents return nothing across `doc/`, and
`lineart-cleanup-refiner` is the nearest thing and has no reference
conditioning either.

**If it is tested, the honest first move is inference-only**: public ControlNet
at its best config (`lineart_coarse`, cs2.5) plus IP-Adapter with a line-art
tile as image prompt, scored on the usual axes *plus* the conditioning map's own
baseline (lesson 6) and Track D's `gt_only` / `vs_condition` columns. Using GT
as the reference leaks, so like the delete oracle it bounds the mechanism rather
than demonstrating an achievable score -- which is this project's established
way of starting.

## Active Goal

**Build a representation of line art that a human can read, then generate from
it.** Work happens in the tracks listed above, not here; this tree is the common
foundation (shared scripts, dataset pipeline, project-level docs).

This supersedes the goal that stood here from 2026-09-11 to 2026-09-30, "close
the gap between the preprocessor's output and GT", which was framed entirely
around deleting the strokes the preprocessor over-draws. **As of 2026-10-04 that
framing is refuted, not merely narrowed** -- see lesson 10. A human judge found
the pixel oracle, which is this project's own definition of a perfect deletion,
**0/15 usable as line art**, the same as the untouched rough and the same as
Track C's classifier, against GT's 14/15. **Line art is not a subset of the
rough's ink**, so no amount of selection inside that output space reaches a
drawing. Deletion survives as a **preprocessing step** -- clearing clutter
before something else draws -- and its selection is measurably non-random, but
it is not a route to the deliverable.

The user's 2026-09-18 redefinition -- stroke as word, panel as sentence -- was
made before this was known, on the judgement that pixel-matching had stalled.
**It now has a second, harder reason behind it**: the thing that has to happen
is a mechanism that *draws* strokes at uniform weight, not one that chooses
among the rough's existing marks.

**Where the work is, as of 2026-10-04.** The words line (Tracks G and H) is the
active front, by user decision the same day. Tracks C, E and F are **held, not
closed** -- kept in place to be re-measured when a new yardstick exists, with
each one's trigger named under Next Actions.

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
- **Lessons 1-9 and the tool traps below hold for every route.** Lesson 8 in
  particular is why the generative-pixel-matching route is closed rather than
  merely unpromising.

**The current route, and why it is shaped this way.** Treat a stroke as a word
and a panel as a sentence. Track F then established a clean negative that
determines everything downstream: **there is no inter-word grammar inside a
panel** -- five independent measurements agree, and a bigram model is *worse*
than unigram. So generation cannot be "a grammar consumes a word sequence".
What does exist is a real vocabulary and -- as this was stated until 2026-10-09
-- real structure in **where** words sit. Hence Track G's shape: the concept is
supplied from outside, and the model's job is only which words appear and where
they go. **That second half has since been measured down to almost nothing**:
the position bins the figure rested on were distorted, and 86% of the apparent
placement structure was the panel's own aspect ratio leaking through a broken
normalisation (lesson 14). On corrected bins what a transformer gets from
seeing every other cluster in the panel is **0.17 bits**, and Track G's
frame-relative run put the same conclusion in panel widths: the reference frame
is worth 0.190 and the word inside it 0.006. So the vocabulary is real and
**the placement structure, as measured on these words, is not** -- which is the
same verdict from the two directions the project measured it, and it points at
the unit again rather than at a better placement model.

Track G has already run that and **failed its visual gate** while passing its
numeric ones -- prototypes land at statistically plausible positions and the
panel still does not read. The user's redirection from that result is the thing
to hold on to: **bits do not guarantee a visual pass**, and while the words
render as an ant swarm no human can judge whether a panel reads at all. So the
front of the work moved *upstream*, to making the words themselves legible and
nameable. That is Track H: pin down the face-part words first, because the more
cards are face-up, the easier the rest becomes.

**As of 2026-10-09 that front has moved upstream once more, and this is the
current shape of the route.** Track H put author labels on 1,003 instances and
found that the word is not what identifies a part -- position and size inside a
detected face box are, and adding the word makes the prediction *worse* (lesson
13). Then it found why the rest is not reachable by a better cue: the unit
itself is wrong. `cluster_set`'s clusters are cut from stroke proximity alone,
so 41% of labelled part instances mix two things -- brows with the bangs above
them, three times in four -- and 14.6% of all strokes never enter a cluster at
all. **So the words line is currently rebuilding its unit rather than its
model**: the author groups individual strokes into parts to make line-level
ground truth, against which mixing, fragmentation and omission can be measured
and a regrouping mechanism scored. Two assets came out of the detour and both
should be held on to -- an external face detector that works on these panels
(no false positives and 15% of faces missed on 42 audited panels, faces found
in 71% of 5,345 panels), and the finding that **frame-relative** coordinates
show the eye/mouth/nose structure that word-to-word placement never did. The
face box is the first reference frame this project has that is not itself
another mixed word.

**Later the same day Track G put sizes on that second asset, and the route's
bottleneck is now a number.** Asked to place a cluster, the reference frame is
worth a median **0.190** of the panel's short side and the word inside it
**0.006**; knowing which of the eye/mouth band's three peaks a cluster belongs
to would be worth another **0.075**, and the author's reading that profile and
reclining faces are mixed into those peaks means even that is an underestimate.
So the missing piece is not a better placement model and not more words: it is
**a label that says which peak** -- that is, the unit. Two tracks reached this
from opposite directions on the same day, one classifying parts and one
predicting positions, which is the reason the route is being rebuilt at the unit
and not anywhere else.

**The user raised the importance of fills on 2026-10-10, with a reason that
reaches back into the words line.** Recorded as the author's framing (Working
Discipline 6), in the author's own terms: in hair, a **region** -- an area
closed off by lines -- is sometimes filled and sometimes not, and **if the line
can be recovered from the region, fills become much easier to handle**, because
a filled area is not a separate kind of ink but a region that has an outline.
Separately, the **highlight fills in hair are a kind of texture**, and being
able to split that from the line art is what keeps a pipeline from growing
branches it cannot otherwise avoid. Two consequences the foundation draws from
it, which need keeping apart because they need different things:
- **The ground-truth side needs no rough at all.** Whether a fill's boundary
  coincides with line ink, and whether highlight fills separate from solid
  fills by shape alone, are measurable on GT, with no model and no GPU. And it
  lands on the unit problem: the reason `cluster_set` mixes parts is
  overwhelmingly hair and shading hatch -- w342/w343/w344 turned out to be
  "arc, short stroke bundle" holding hair, hatch and panel borders; three
  quarters of labelled brows share a cluster with the bangs above them; 14.6%
  of all strokes enter no cluster. **So a line/fill/texture decomposition is
  not only Track J's business: part of the words line's unit problem is a fill
  problem.**
- **The rough side has an unmeasured premise.** Predicting which regions are
  filled requires that the information be present in the rough. If it is not,
  the task is not prediction but proposal, and the goal changes. That is the
  fork Track J's stage 2 has to settle before it registers anything.
One constraint is already measured and bears on both: some `ako5` tiles sit
entirely inside a fill with no boundary inside the 480px window (0.5% entirely
filled, 31.3% near-blank), so **a region-level formulation may not fit the tile
size this project evaluates on** and may need panel-level inputs.
**Both of those were then measured on GT alone, 2026-10-10, and both came back
against the easy reading** (Track J, `results/region_closure_20261010/`; the
direction and the size of each were reproduced here from the images with an
independent implementation):
- **The 480px window is the wrong unit, and by a wide margin.** On a 480 tile
  **87.6%** (`housei`) and **92.9%** (`ako5`) of the solid-fill *area* sits in
  a connected component whose own boundary is outside the window; on a koma
  panel of the same material at the same drawing scale it is 26.4% / 48.5%.
  Stage 2 therefore takes the **koma panel** as its input unit. A model shown a
  480 tile is being asked about an object it is never shown whole.
- **The line art does not enclose anything, and this one is not only Track J's
  problem.** Taking strokes alone (`ink & ~fill`) and flood-filling what they
  close gives **1.2 regions per panel** on both `ako5ver2` and `housei` koma
  (5.7 on `clip_pairs_v3`), the largest background component is **100.0% of the
  background in every panel inspected**, and **95-100% of the fill area lies
  outside any enclosed region**. So "propose closed regions, then classify
  each" has no ground truth to be scored against on this corpus. Track J's
  answer is to take the 面 to be the **fill connected component**, keeping the
  model pixel-wise while the metric becomes region-wise. The consequence worth
  carrying to the words line: **any proposal that assumes a part or a region
  can be recovered as an area enclosed by the line art cannot be built on this
  corpus by that construction** -- offered as a fact about the corpus, not a
  finding about Track H's or Track G's own units.

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
score alongside it -- see the nine lessons at the top of this file, and
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

Items 1-2 are the fronts on the words line -- Track H on the author's time and
Track G on the machine's -- and **item 5 is the third front as of 2026-10-09**,
Track J on the GPU, which until that day had nothing to do. Item 3 is the three held tracks and what would justify resuming
each. Item 4 is two open bugs in shared tooling, **item 5 is now an active
third front -- Track J, the solid fills, opened 2026-10-09 to take the idle
GPU** -- items 6-9 common-foundation housekeeping, none of them blocking. Every ControlNet track is closed and none leaves work behind, and
**Track I closed itself on 2026-10-04** -- see the Track Ledger above.

1. **Face-part words** (`../lineart-face-words`, Track H -- the current front,
   and a live session is on it as of 2026-10-09). **The push is done**:
   `ca9d788` equals `origin/face-words`, verified here, so the
   "single-disk" flag raised on 2026-10-04 is cleared. It remains a worktree of
   Track G's separate clone, so neither `git worktree list` here nor
   `tools/audit_track_ledger.py` can attribute it -- the audit prints it as a
   separate clone, which is the best it can do.
   Why it exists, from Track G's first notice: a word is a shape class with
   size removed, so words mix meanings and dilute any placement measured on
   them; fixing a small, trustworthy set of words is the prerequisite Track G
   stopped for. It assigns author strategy labels (eye, nose, mouth, brow, ear)
   in a namespace separate from `wNNN`, at instance level, with a held-out part
   fixed at labelling time -- and **the user allowed those labels as ground
   truth for evaluation, an explicit exception to "names are interpretive
   only"**. It owes its own notice; the ledger entry above is this file reading
   its work log, not the track speaking.
   **Where it stands and what the foundation is waiting for.** The opening
   hypothesis is refuted, the words turned out not to carry part identity
   (lesson 13), and the track has moved up a level: the unit itself is wrong, so
   it is collecting line-level ground truth through the grouping UI, zero
   production groups so far. Three things to watch from here rather than act on:
   - **The three measurements that ground truth is for are pre-registered and
     unanswered**: how many true groups one current cluster spans (mixing), how
     many clusters one true group splits into (fragmentation), and how much of a
     part's ink never enters `cluster_set` at all. Those numbers decide whether
     the regrouping mechanism is a repair or a replacement, and they are the
     first thing to fold in here when they land.
   - **The held-out discipline is now carried across two label sets.** The face
     queue inherits the instance queue's series groups and the same
     page-sha1 rule, and the track verified zero panels disagree between them,
     so a mechanism can be scored without a leak across the two label types.
     This is worth protecting if anything proposes re-splitting.
   - **A rule's pass criteria are still unset by design** (precision 0.80 /
     recall 0.60 per label on the panel-held-out side is the recorded
     candidate, with fewer than 20 positives counting as undecidable). They are
     to be fixed before the held-out side is read. Note the counting problem
     already visible: at the random order's rate, reaching 20 held-out
     positives needs roughly 1,000 labels for mouth and nose, 2,400 for ear and
     7,000 for closed eye -- so either the order stops being uniform (which
     would bias what "position predicts" means and was deliberately avoided) or
     the rare parts get a different instrument. Unresolved, and the user's call.
2. **Panel generation** (`../lineart-panel-generation`, Track G) -- **resumed
   and settled the same day, 2026-10-09; the slot is open again and has been
   re-routed.** It had been idle since 2026-09-25, when the work moved to
   Track H and its own blocker sent it there. The reasoning behind the decision is a resource one and worth keeping,
   because it will recur: Track H's line-level ground truth occupies **the
   author**, which is this project's scarce resource, while the machine sits
   idle -- and **no held experiment anywhere in the project is GPU-bound**, the
   heavy work having all been closed by lesson 8. So the work to pick up is work
   that needs neither the GPU nor the author's eyes, which is what this one is.
   The question it halted on is now answered -- **placement is relative only
   once a reference frame is given, and the word contributes almost nothing
   inside it**; the numbers, the author's visual verdict and the three peaks
   are in the Track Ledger entry for 2026-10-09, and the one-line consequence
   is in lesson 13. Both notices are in `inbox/`
   (`note_track_g_frame_relative_placement_20261009.md`,
   `note_track_g_status_and_idle_capacity_20261009.md`); every figure in them
   was re-derived here from `results/framerel_20261009/` before being folded
   in, and the box-orientation numbers were recomputed from `faces.csv`
   independently. **Not resumed, and still on hold**: the generative line
   (PosAR-W, set-AR vs two-stage, scale-up), untouched.
   **What the track now waits for** is Track H's unit: `tools/gen/framerel.py`
   takes a new unit's labels in place of the word id, so the distance to the
   0.074 ceiling comes out in one run, right after item 1's three measurements
   land. Restricting to frontal faces was drafted and deliberately **not**
   registered -- with a mixed unit it would measure a mixed unit.
   **The re-routed slot, decided 2026-10-09 by the foundation under the user's
   standing permission to give Track G other tracks' work**: measure the real
   blast radius of the distorted corpus position bins (the trap above) and
   re-report the figures that sit on them -- rebuild the bins with matched
   denominators and recompute the marginal 7.563, the cloze 6.315-6.388 and
   Track G's own 6.658. The track proposed it itself, it reported the
   distortion in the first place, and it needs neither the GPU nor the author's
   eyes. The precedent for the shape of the answer was Track I on
   `lineart_003_000`: if the shift is small, "no re-measurement needed" can be
   closed; if it is large, the premise Track F resumes on has changed.
   **Done the same day, and the answer is "large" (third notice, folded into
   Known Tool Traps).** Every position figure is replaced, no verdict flips, the
   margins collapse, and the mechanism is that the tokens were encoding the
   panel's aspect ratio -- 86% of the learned advantage reproducible by a
   frequency table with no training, which is now lesson 14. Track F's premise
   has changed in exactly one direction: its word negative is untouched and its
   **placement** claim is much weaker than recorded.
   **The slot is open again and the next assignment is named**: finish the
   sweep. Which of the *remaining* figures in Tracks F and G sit on the
   distorted bins, and what do they read when rebuilt -- specifically Track F's
   "238 words have a placement preference that replicates" and its weak
   relative placement (residual cosine 0.15 against a null near zero), plus any
   figure in either track's logs derived from `bins_of`. Some of it was computed
   on real coordinates rather than bins and is therefore unaffected; the
   deliverable is the list, separated into "unaffected", "replaced, verdict
   holds" and "replaced, verdict changes", so whoever resumes Track F reads one
   table instead of re-deriving this. Same conditions: no GPU, no author's eyes,
   pre-registration first, Track F's files untouched.
   Deliberately *not* chosen for the same slot, and why, so the decision is not
   re-litigated: Track C/E's re-render route is the one named untried thing in
   the deletion family, but its verdict needs the author's eyes and would
   compete with the labelling; Track G's generation scale-up is cheap but needs
   a visual gate and leaves word purity unsolved; the solid-fill pool would use
   the GPU but has no route to it after lesson 8, and reopening it is a separate
   user decision (item 5).
   First notice received 2026-10-04 and it closes the
   13-day gap; three things in it change the picture here:
   - **The blocker is word purity, not tooling.** A word is a shape class with
     size removed, so one word spans roughly 6x in size and mixes meanings --
     the user reads w342/w343/w344 as closed eyes where Track F named them
     panel frame, bangs and hair tips. **Relative placement measured on mixed
     words is diluted**, which is why fixing a small set of words came first.
     That is Track H.
   - **"Do not restart from placement" (2026-09-23) was about ordering, not
     about dropping placement.** Rendering had to become readable first,
     because the ant swarm could not be judged at all. It since was: real
     cluster instances replaced decoded prototypes (token match 5,345/5,345,
     A-column F1 0.78 at precision 1.00; the black blobs are real solid fills,
     not a width bug) and the user judged the words recognisable. The goal was
     then restated as **analysing placement on the ground-truth word
     segmentation, with no generative model, measuring word trustworthiness
     while doing it.**
   - **The instrument check failed three times and was stopped before any main
     run, with no verdict on the claim it was testing** -- the user's domain
     claim that placement is decided by relative position rather than absolute
     coordinates. v1 (32x32, offsets in reference-word sizes), v2 (16x16,
     direction x log-distance) and v3 (relation types fixed on the training
     half) all failed their relocation or range checks. The diagnosis is that
     relations do transfer (type-1 median distance 4.1 reference sizes) but a
     reference large relative to the panel makes "within 4 sizes" cover
     everything, and shrinkage tuned on the whole set erased the few tight
     pairs. **Grid probabilities have too many knobs for this question**; the
     first candidate on return is a point prediction scored by miss distance,
     not yet registered.

   Measured on GT and replicating across halves (62 works -> 36 groups, 380
   near-duplicate panels dropped): per-word median detail-flip **0.275** (the
   global 23.2% was a frequency-weighted average), **238 words** have a
   placement preference that replicates, and relative placement between
   different words is consistent across halves but **weak** -- residual cosine
   0.15 against a null near zero. The word-reliability split **49 / 164 / 254**
   is the second version, three-valued by bootstrap with zero pass/fail flips
   across three seeds; the first version flipped 8 of the top 20 words on the
   random draw alone and is superseded. Reliability there means **boundary
   stability, not freedom from mixed meaning** -- which is exactly the blocker
   above. The generative line (PosAR-W, set-AR vs two-stage) is **on hold, not
   abandoned**; a train/sample mismatch in the placement sampler was found and
   fixed, worth 0.011 pos bits, and did not explain the collapsed compositions.
   The standing warning from its last run still holds: **bits do not guarantee a
   visual pass** -- position beat the marginal by 0.9 bits while the montage
   stayed unreadable.

   **Two things were routed to its briefing on 2026-10-09 and should be the
   first things read when it resumes.** (a) The axis correction: the swap it
   reported was a misdiagnosis, its own analyses are numerically fine but every
   *directional* reading in them is mirrored, and the distorted corpus bins are
   still distorted -- Known Tool Traps has the settled version. (b) Track H's
   results, which bear directly on the question G halted: the words do not
   carry part identity (lesson 13), the unit mixes parts (the trap above), and
   the structure G was looking for in word-to-word placement does appear
   cleanly in **frame-relative** coordinates once a face box supplies the
   frame. If relative position is to be measured again, the face box is the
   first reference frame this project has that is not another mixed word.
3. **Tracks C, E and F are held, not closed** (user decision 2026-10-04). The
   words line resumes as the active work; these three stay in place so they can
   be re-measured when a new yardstick exists. **Held means the state must be
   durable and the trigger must be named**, so each briefing records what would
   justify picking it up again rather than leaving a future session to re-derive
   it:
   - **C** (`../lineart-stroke-selection`) -- what would change the verdict is
     not a new metric. Lesson 10 came from a human absolute judgement, and it
     found the *perfect* deletion unusable. The one thing that could move it is
     a different **output format**: Track E's untried route of re-rendering the
     surviving strokes so the result reads as a sparse drawing rather than a
     holed rough. What was measured was holed roughs.
   - **E** (`../lineart-aesthetic-judge`) -- stopped because a comparison
     between two lossy outputs ranks the loss (lesson 11). It reopens when there
     are two outputs that differ by something *other* than damage. Its unused
     build, with tiles disjoint across arms, sits at
     https://claude.ai/artifact/Mvvoqn3CAjtaNQ3ScqR9vG.
   - **F** (`../lineart-stroke-grammar`) -- closed its measurement phase on a
     clean negative: no inter-word grammar within a panel, five measurements
     agreeing. That negative was measured on *the vocabulary it had*. If the
     words line produces a better vocabulary -- which is exactly what Track H is
     working on -- the question is worth re-asking on it, and that is the
     trigger. **Two corrections were written into its briefing on 2026-10-09**,
     both aimed at whoever resumes it: the axis note it received on 2026-10-04
     named the wrong cause (the columns are as named; the transpose is in
     `strokes`/`pts` and `dims`, and the distorted bins remain distorted), and
     Track H has since measured that the unit `cluster_set` is built on mixes
     parts in 41% of labelled cases and excludes 14.6% of all strokes. The
     second one sharpens the trigger rather than the negative: a better
     vocabulary now plainly means a better **unit**, not just a better codebook.
     **Measured 2026-10-09, and the answer splits F's record in two**: its word
     negative is untouched (word bits move by -0.056 to +0.004 and the cloze
     still falls far short of its 6.348 gate), while **every position figure is
     replaced and the placement claim is much weaker** -- the marginal goes
     7.563 -> 7.957, the cloze band 6.315-6.388 -> 7.785-7.792, and 86% of what
     looked like placement structure was the panel's aspect ratio (lesson 14).
     A corrected corpus identical in every other token is kept at
     `../lineart-panel-generation/results/posbins_20261009/corpus_fixed.npz`, so
     a resume starts from that rather than rebuilding it.
4. **Both tool bugs are fixed, 2026-10-09 -- what remains under this item is
   the data-side decision.** (a) The `housei`-prefix split heuristic now
   resolves by looking for the file, through
   `tools/evaluation/gt_paths.resolve()`, in all **nine** places it appeared
   rather than only the one that was reported. The re-check of past 192-tile
   numbers this item asked for **is not needed, and the reason is recorded
   under Known Tool Traps**: the two splits share no basename, so the heuristic
   could only resolve correctly or point at nothing, and every caller raises on
   a missing file. (b) `bipartite_match_f1`'s unbounded worst case now has a
   shared guard, `tools/evaluation/bsds_guard.match_f1()`, used by the two
   shared batch scorers; the metric is untouched. Guarded and unguarded output
   is byte-identical on the verification batch.
   **Still open and still a user decision**: the unapplied `manga_line`
   emptiness fix (downscale to 240px plus auto-contrast). It is a data-side
   change, not a bug fix, and it must be applied to training and holdout
   together or it merely inverts the train/eval mismatch. Details: **Known
   Tool Traps** above.
5. **Solid fills: reopened 2026-10-09 by user decision, as Track J
   (`../lineart-solid-fill`, branch `solid-fill`).** Deferred on 2026-09-13,
   reopened when the GPU was idle and the survey found **nothing held anywhere
   in the project that is GPU-bound** -- which is structural, not an oversight:
   the heavy generative line closed with lesson 8, Track I closed on a
   four-decimal null, the deletion family closed with lesson 10, and the CNN+GAN
   family was left at lesson 5. What remains on the words line is bounded by the
   author and by labels, neither of which a GPU shortens. Fills are the one
   exception, and the ledger entry above has the reasoning and the counts.
   Over 12,000 tiles across `ako5` and `housei` have never been trained on and
   appear in no evaluation set. They are not a harder version of the current
   task but a different one: GT there is 24.5% solid fill against the lineart
   pool's 4.0% -- both **ink-normalised**; by tile area the fill in these pools
   is 4-8%, and the two quantities are not interconvertible because the mask
   that measures area is a dilated opening, see the Track J entry -- 27-38% of tiles are near-blank, and the preprocessor fills
   nothing at all (fill_ratio 0.0%), so an edge-detector-plus-selection
   pipeline cannot reach it by construction. The delete-only oracle tops out at
   0.3291 there against 0.7425 on the lineart pool. The natural moment to
   reopen it is when stroke selection has a real number on the lineart pool:
   that is what decides whether this is the other half of the plan or a
   separate project. Inventory:
   `../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md`.
   **Stage 1 is done (2026-10-10) and its verdict is that the premise holds**:
   the preprocessor's readable ceiling is micro IoU 0.0000 and the adopted
   checkpoint fails its own off-by-one control, so the comparison for fills is
   *paint nothing* against *paint everything* (whose score is the GT area share
   by identity: 0.0939 / 0.1201). **Two GT-only measurements then moved the
   unit before stage 2 was registered**: a fill region does not close inside a
   480px window (87.6% / 92.9% of fill area is in a border-cut component), so
   the input is now the **koma panel**; and the line art encloses almost
   nothing (1.2 regions per panel, 95-100% of fill area outside any of them),
   so the 面 is taken to be the fill connected component rather than a proposed
   closed region. Stage 2 is registered in the track's log on `clip_pairs_v3`
   koma panels, split on **work family** with the `ako` series collapsed (9
   families, 969 panels, 19.3% held out, fixed before any score) because
   whether a character's hair is blacked in is a property of that character.
   **The one item that was waiting on the user is decided: yes, 2026-10-10.**
   A **non-adversarial segmentation model is outside** the CNN+GAN prohibition,
   stage 2 may proceed and may spend the four-day slot. What is still closed is
   the adversarial objective itself -- the `msgan`/`cleanup` i2i family, a GAN
   term, another `--unpaired-weight` sweep -- so the prohibition narrowed by
   exactly one reading and not by a family. Track J's falsifier for that
   reading stands as registered; if it fires, the reading is withdrawn and that
   goes in the headline.
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
9. **Keep this file from falling behind again.** Status 2026-10-09: the
   mechanism now half works. Track G sent its first notice on 2026-10-04 and it
   is folded in; Track H has pushed its branch but **still owes a notice**, so
   everything above about it is this file reading that track's work log, which
   is second-hand by construction. The audit ran clean today apart from
   uncommitted paths in two finished trees (`cleanup-refiner`,
   `controlnet-sd15-refine`) -- worth a look only to decide whether they are
   worth keeping, since both tracks are closed.
   The 2026-09-30 reconciliation
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
