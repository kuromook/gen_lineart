# Track K's record -- stroke connectivity, 2026-10-16 to 10-19

Moved out of `doc/CURRENT.md` on 2026-10-24, verbatim: the two Track K ledger
bullets, 1,078 lines, for a track that closed its tree on 2026-10-19 and is
dormant. The ledger keeps the entry Track K itself wrote to be read instead of
these bullets -- *"Where Track K stands, 2026-10-19"* -- plus its restart
condition; everything below is the working record behind it.

Both axes are here: the connectivity axis (K1-K7, the author's 40 and 56
judgments, box B's two stopped instruments, the collapsed-bridge mechanism and
the cost arithmetic) and the second axis that opened and closed on 2026-10-19
(K19-K22, the free-end gap rule). **Across six registrations Track K never moved
a bar, and four of the six would have passed on a bar moved by less than a
hundredth** -- which is why the near-misses are kept at full length here rather
than summarised into "did not pass".

The track's own files are the other store: `../lineart-stroke-topology` at
`25e4dec`, pushed to `origin/stroke-topology`, with
`doc/track_k_stroke_topology_proposal_20261016.md` here in `doc/`.

- **Track K's second axis closed itself the same day it opened, 2026-10-19:
  K19 landed on the registered "the axis dies" branch, and the letter L
  reservation is released** (no `stroke-bridging` branch and no
  `../lineart-stroke-bridging` worktree was ever created). Commit `78856b2` in
  K's tree.
  - **Instrument checks first, all three passed.** Repeats 7/8 against a
    chance agreement of 0.3657 (expected 2.93, mode 0.50); anchors 7
    "not broken" / 0 "intentional skip" / 1 undecidable, and **4/4 on the four
    whose widths could be matched**, i.e. the subset the width defect does not
    touch was answered perfectly; the cross-check scored **6/6 on the six
    committed items** against the band of 5/6 (exact upper tail 0.0261)
    declared before unblinding.
  - **The two undecidable items both moved to a committed answer**, which is
    the foundation's registered reading of them: not drift, but the boundary
    getting sharper. The decision to keep them out of the bar (trap 21) is what
    made that legible instead of a failure.
  - **Primary, on the registered denominator and bar**: 42 cases at <=8px,
    14 kasure / 19 intentional skip / 8 undecidable, **precision 0.3333, CP95
    [0.1957, 0.4955]**. The upper limit misses the 0.50 bar by **0.0045** and
    **K did not move it**. The <=6px reference arm (n=23) reads 0.4783
    [0.2682, 0.6941], which is the held branch, and the primary governs.
  - **The answer-check that matters most.** Before unblinding, Track K added
    the column the foundation's table lacked: at a true precision of 0.625 the
    probability of reaching "the axis dies" is **0.000** -- *this paper cannot
    kill the axis*. **It killed it.** So the truth was not 0.625: **the n=8
    point estimate was noise**, and the proposal's headline had been written on
    it.
  - **Cost instrument rebuilt (K21) and all three arms came up empty**: part
    groups saturated (zero cross-group on every arm), ambiguity 2.1%, width
    mismatch degenerate on box B (median 0.000). The proposal's own stated
    limit -- that cost can only be seen through the author's answers -- is
    confirmed rather than worked around.
  - **The answers took 152 seconds for 52 items (median 2.4s each), and Track K
    refused to use that for pass or fail because no speed band was
    registered.** The evidence for judgement quality is the 6/6 cross-check
    against items answered slowly last time. This is the discipline working in
    the direction that costs something: an unregistered statistic that looked
    damning was left out of the verdict.
  - **A third "must not do" statement now stands**, next to unconditional
    loosening and repair keyed on the collapsed bridge: **a rule that closes
    free ends by gap length.**
  - **Track K's tree closed 2026-10-19** (`25e4dec`, pushed to
    `origin/stroke-topology`, working tree clean). Its own closing record is
    worth keeping: **across six registrations the bar was never moved**, and
    every one of them landed within a hair of it -- K1 **1.487** against 1.5,
    K4 **1.8435** against 2.0, K19's upper limit **0.4955** against 0.50 (three
    just outside), K5 **0.5154** against 0.50 (just inside), K7 stopped because
    the band did not hold and its primary was never read, and K2-C abandoned
    its axis at the registered exit. Four of the six would have passed on a
    bar moved by less than a hundredth.
  - **Track K leaves one instrument for the other tracks**: the author-judgment
    web UI, `tools/web/judge.html` plus `tools/stroke/k22_tiles.py` in that
    tree. **Swap the manifest and the tiles and it works for any
    multiple-choice question put to the author.** Answers land in the
    artifact's shared store and are read directly with `ArtifactData`, which
    removes the hand-off that caused K10's `uid` mix-up -- distributing PNG
    sheets and taking answers back by number. 52 items took 2.5 minutes.
    **Routed to Tracks H and J on 2026-10-19**, since H spends the author's
    time continuously and J's fill/line boundary may need the same kind of
    answer.
  - **What Track K carries forward**, as it put it: three "must not do"
    statements, one confirmed mechanism (the collapsed bridge, PAF 0.5759,
    supported on the held-out series), **one direction the numbers have not
    killed** (carry both tokenisations downstream rather than deciding -- which
    changes Track F's output format *and* the model's input, so it was never in
    K's hands), and one correction returned to this ledger (the 0.5px
    retraction). **`face_groups.jsonl` has since grown** -- 518 rows and 402
    orders at 2026-10-23 against the 417 this was written on -- so the input K
    was waiting on exists; K is quiet by the user's 2026-10-19 decision, not
    for want of marks. The
  axis is new -- its own registration, its own entry -- but the five
  instruments, the ink mask, the box structure and the 52-tile paper all live
  in K's tree, and Track I's first probe is the recorded case of what a
  re-derivation costs. Proposal:
  `../lineart-stroke-topology/doc/track_proposal_stroke_bridging_20261019.md`
  (`98dce58`), bands in `b1eab40`.
  **Approved to proceed without waiting for the author's 28 judgments**, which
  are the *outcome* of a registration already closed (bar 0.50, three-way
  reading, abstention condition, repeat band 5/6 declared before unblinding).
  Waiting would stake the axis on a coin flip: at the point estimate the
  judgments **confirm with probability 0.350, leave it held with 0.650, and
  cannot refute at all** -- Track K's own column, and the number that matters
  most for spending the author's time. Three pieces of work do not depend on
  them: the 54 cases that cannot be asked (the gap is not visible in the
  output, which is why even a perfect result covers 44% of the footprint), the
  free-end joining rule itself (**absent from Track F**, which is sealed), and
  the part-group cost instrument, which is **saturated** -- zero cross-group
  cases on every arm, because box B's pairs are in nearly the same part by
  construction, so the real error of fusing two lines inside one part cannot be
  seen.
- `../lineart-stroke-topology` (branch `stroke-topology`, **Track K**) --
  **opened 2026-10-16, dormant since 2026-10-19**: the tree closed at `25e4dec`
  (pushed, working tree clean). **Its restart condition is met and it stays
  dormant anyway**: `face_groups.jsonl` has grown past the 2026-10-16 state it
  was waiting on -- 518 rows / 402 orders at 2026-10-23, and Track H reports 97
  complete faces at `order` >= 301 all carrying a `broken` field, 62 of them
  with at least one mark and 116 marks in all (train and holdout together) --
  which covers K3's frame of 40 new faces and 40 new marks. K is quiet by the
  user's 2026-10-19 decision, that it does not have to happen now, and Track H's
  notice explicitly does not ask for it back. Its closing record and both axes
  are in the Track K entries above. Opened by user
  approval, from the user's own observation after
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
    a line's end** (**median 0.5px**, corrected 2026-10-19; this entry first
    said 10.8px, see the correction note under Known Tool Traps): a line's ink
    extends past its
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
    (CI 0.1021-0.6369) -- which is `(RR-1)/RR`, the fraction among the exposed;
    the share of all box-B cases a perfect repair would remove is
    **PAF 0.1082**, Levin and Miettinen agreeing, so K7 is framed in the asked
    direction and passes the two-formula check, Fisher p 7.4e-4, permutation p 0.044; the per-part
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
    **The author answered all 56 the same day; both instrument checks passed
    and the primary landed between the registered bars.** Repeats **8 of 8**
    against a chance agreement of 0.4123 (expected 3.30) with the modal answer
    at 0.562, under the 0.70 that would have printed uninformative. Anchors
    **8 of 8 answered "not broken" with 0 called a skipped line** -- the
    fourth option turned a bar that the old form would merely have passed into
    an active confirmation that the crop is not hiding connections. And **0 of
    40 real cases were called "not broken"**, which is the check on dropping
    the 55 sub-collar cases. **Primary: skipped line 22 of 40 = 0.550, CP95
    [0.385, 0.707]**, faint 9, can't-tell 9 -- the registered middle reading,
    **"both kinds exist in quantity"**. Its consequence is the day's real
    result: **F's rule cannot be repaired by loosening alone.** About half the
    answerable gaps were drawn that way on purpose, so extending the linker's
    reach destroys intended gaps at about the rate it recovers dropouts, and
    **an unconditional loosening is refuted by labels** for the first time in
    this project. **Sensitivity that must be printed beside it**: excluding the
    9 can't-tells gives 22 of 31 = 0.710, CP95 [0.520, 0.858], which sits in
    the "mostly intentional" band -- so the conclusion depends on how
    missingness is handled, the registered denominator was all 40, and the
    0.710 reading is not available as a post-hoc headline. The registration for
    the unread band states in advance that can't-tell stays in the denominator.
  - **The two post-hoc findings from the author's answers are one finding.**
    The track offered (1) gap length separates the answers (skipped median
    11.96px against faint 5.66px, Mann-Whitney p=0.0057) and (2) `face.other`
    is 7 of 7 faint against 2 of 33 elsewhere (Fisher p=1.9e-6). But faint is 9
    cases of which **7 are `face.other`**, so only **2 faint cases exist
    outside it**: the gap-length result's faint arm is 78% cheek hatching, and
    the two statements are the same 7 cases seen two ways. Confirming both on
    the unread band would confirm both and learn nothing. The registration must
    therefore **make them compete**: does gap length still separate *within*
    `face.other`, and does faint exist at all outside it? If the unread band's
    faint cases are again mostly `face.other`, the operative variable is the
    part -- the cheek hatching -- and gap length is its proxy, which moves the
    repair out of the linking rule and into how hatching is drawn and inked.
    Worth noting what does line up: the author's first visual reading ("the
    cheek hatching is faint output, a different thing from a skipped line"),
    K7's `face.other` box-B rate of 0.782, and today's 7 of 7 all point at the
    same row, which is what registration A stands on.
    One more limitation to carry: the 40 were drawn from the 127, which is the
    gap distribution truncated from below, so faint cases are under-represented
    and the gap-length separation is **attenuated** -- the truncation cannot
    have manufactured it.
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
    **The saturation check passed (2026-10-18): 0.4700 over the 25,035 tokens
    of K4's main leg**, far under the 0.90 that would have made the predicate
    useless, so D can discriminate. The track first measured it over all 90,111
    tokens, noticed that is not the set D's primary rides on, and re-measured --
    this file's "a diagnostic must run the instrument's own code path", applied
    by the track to itself; the answer did not move (0.4619 -> 0.4700) but the
    re-measurement is recorded. Numbering was checked too: `sid_not_found = 0`,
    and 25,090 rows against 25,035 unique (panel, sid) because one panel can
    carry several faces.
    **The bars, written here under today's division**: the primary is the
    **population attributable fraction, PAF, bar 0.30** (written as "AFe" when
    it was first sent, which the vocabulary entry below corrects), chosen from
    what a repair must achieve -- the repair is a cross-track change to an ink
    rule whose removal fragmented a panel from 64 components back toward 258,
    so a mechanism worth that discussion must be one whose perfect repair
    removes about a third of the marks. (The second justification offered at
    the time, that K1 was refuted at 0.248, is **suspended**: that number is
    consistent only if fill adjacency at radius 4 has a population prevalence
    near 0.6766, which has not been printed.) PAF and MH-RR are
    one bar written twice, through **PAF = p(RR-1)/(p·RR+1-p)**, so
    **RR\* = (0.3+0.7p)/(0.7p)**: at p = 0.4700 the companion bar is
    **MH-RR >= 1.912** (p 0.30 -> 2.429, 0.20 -> 3.143, 0.15 -> 3.857, 0.10 ->
    5.286), and a changed predicate means re-reading RR\* off that formula
    rather than keeping the number. **Falsifier**: the 95% CI upper bound of
    MH-RR below RR\* is a refutation, not a held verdict -- checked for reach
    with 185 marks at p=0.47, where a true null gives [0.750, 1.333] (upper
    bound under 1.912, so refutation lands) and a true RR of 2 gives [1.483,
    2.697] (lower bound over 1, so support lands). **Abstention before
    reading**: if either the expected exposed or unexposed mark count under the
    null falls below 20, do not read the primary -- print it and come back,
    because adding marks from the unread band or Track H's new faces comes
    first.
  - **The registered predicate was too weak for its own mechanism, so D gains
    D-prime.** "The skeleton touches at least one grow-only pixel" is 47%
    prevalent and is carried by tokens whose grow share is tiny (per-token
    median 0.0, q75 0.0741, q95 0.4516), while the mechanism says the grow
    pixels sit at a **bottleneck**: two drawn lines bridged by a gray-190
    thread. It is also confounded by length in the hypothesis's own direction
    -- a longer token touches more pixels, and **124 of the 185 marks are in
    the longest quintile**. So the primary predicate becomes **D-prime:
    removing the grow-only ink pixels disconnects the token's skeleton into two
    or more parts** -- the token owes its own connectivity to grow -- with D
    kept as a secondary, printed separately and never pooled. D-prime's
    prevalence is printed first, because the abstention condition depends on
    it.
  - **The permutation has to run inside the whole stratum, not inside one of
    its factors** (the foundation's correction to D's registration, 2026-10-18,
    one level deeper than K7). The track wrote "MH-RR stratified by length and
    face, null value 1 under permuting marks within face": permuting within one
    factor while stratifying on two does not return the statistic to 1, which
    is exactly K7's failure again. Permuting within (face x length) cells
    instead would be degenerate at 185 marks over ~145 faces. So the design is
    settled as **stratify by length decile only, treat the face as the CI's
    cluster** (the cluster bootstrap the track already runs), and **permute
    marks within length decile**, which pairs the null value 1 with the
    statistic actually computed. The K7 stratum-contribution print is required
    before the primary is read: how many of the ten deciles carry both exposure
    levels and both outcomes, and what share of the marks sit in them.
    **D was refuted the same day, and it was the foundation's hypothesis.**
    With the abstention print passing (184 marks, D-prime prevalence 0.3033,
    expected exposed marks 55.8, RR\* re-read from the formula as **2.4129**),
    8 of 10 deciles contributing and all 184 marks inside them, and the
    negative control recovering the null (**permutation median 1.0017**, 95%
    interval [0.8542, 1.1504]), the primary came in at **MH-RR 1.1388, CI
    [0.9706, 1.3105], AFe 0.0404** -- the CI's upper bound under RR\*, so
    **refuted, not held**. The crude association was 1.83 (marked tokens are
    0.554 D-prime against 0.3033 overall) and length-stratification took it to
    1.14: **the foundation's warning was only half right**, since D-prime does
    *not* escape the length confound either -- longer tokens owe their
    connectivity to grow more often, predicate or no predicate. What D-prime
    did buy is the clarity of the negative: **30.33% of tokens really do owe
    their connectivity to grow and it still does not predict the marks**, so
    the mechanism exists and is unrelated, rather than merely being diluted.
    The grow bridge stays on the books as a **preprocessing** observation (cut
    gray median 160, 28.5% at or above 190; width median 2.00px, 52.2% under a
    2.7px line) for whoever touches the ink rule, and is closed as a topology
    question. (Mark counts: 184 tokens here against K4's 185 rows, the
    difference being one panel carrying several faces.)
  - **Two bars of different shapes are being applied to the same marks, and
    the surviving mechanism is K4.** K4's main leg was **held** at MH-RR
    1.8435 against its registered bar of 2.0 -- **that verdict stands and is
    not reopened**. **The 0.331 this file quoted beside it is invalid**, and
    Track K caught it within hours: see the vocabulary entry below. K4's RR has
    **exposure and outcome swapped** relative to the question, so no
    attributable fraction computed from it means anything, and the comparison
    "K4 0.331 against D 0.0404" put a mis-framed number beside a correctly
    framed one. What stands is that K4 must be **recomputed in the asked
    direction** -- exposure = passes a collapsed bridge, outcome = marked,
    MH over length deciles, the same design as D -- before any bar is
    registered for it. Crude, in that direction, the RR is **13.136** and PAF
    is **0.6692** by both formulas, but direction 2 fell from 4.456 to 1.8435
    under stratification, so direction 1 will fall materially too and is not
    to be guessed. **This also answers whether the band has a
    companion -- it does.** One opening of `[171:]` carries three things: K4's
    AFe confirmation, the part-versus-gap-length discrimination designed so it
    cannot confirm both, and registration A's confirmation if A produces a
    statistic first. Order: **registration A on the read material first**,
    since the part-versus-gap question is about the same `face.other` cluster
    and designing it without knowing what that cluster is would mean measuring
    one candidate blind.
  - **The two leading mechanisms were both dismissed on a swapped-direction
    RR, so "no single mechanism explains the marks" is withdrawn.** Read in the
    asked direction, **K1 (fill adjacency) has PAF 0.5030** crude with both
    formulas agreeing, and **K4 (collapsed bridge) has PAF 0.5759**,
    Miettinen, CI [0.4590, 0.6861], adjusted over length deciles (Levin
    0.3926). **Both clear the 0.30 bar**, and both were failed by bars applied
    to an RR with the mark as the exposure. Their registered verdicts stand --
    K1 unsupported at RR 1.487 against 1.5, K4 held at MH-RR 1.8435 against 2.0
    -- because a verdict is not re-read after the fact; what changes is what
    gets measured next. The trajectory this file proposed, that the track would
    close on "the defect is real but no single mechanism explains it", rested
    on the mis-framed 0.248 and 0.331 and is withdrawn. The three deaths of the
    day are unaffected: the gray axis, the continuity mechanism and the
    hysteresis bridge stay dead.
  - **K1 and K4 are probably one mechanism, and arithmetic says so before any
    new measurement.** Attributable fractions of **disjoint** exposure groups
    cannot sum past 1, since the attributable cases are disjoint sets --
    and **0.5030 + 0.5759 = 1.0789**. (Even the conservative pairing, with
    Levin's 0.3926, sums to 0.8956 and leaves almost nothing unexplained.) So
    the two exposures must overlap. The mechanism side agrees: **K4a
    established that the loss-2 collapsed bridges are mostly a fill's medial
    axis** (local ink width median 26.8px, 92.7% over 8px), which makes "passes
    a collapsed bridge" largely a restatement of "is adjacent to a fill". So
    the next measurement is **the overlap, on read material, before
    registration A**: the 2x2 of fill adjacency against collapsed-bridge
    passage over the 25,035 tokens, each exposure's PAF conditioned on the
    other, and the over-1 sum printed as a clue to which estimate is inflated.
    Direction is fixed as exposure = predicate, outcome = mark; Miettinen is
    the primary with Levin as a sensitivity; the crude two-formula agreement is
    checked before anything is adjusted.
    **Measured, and "one mechanism restated" is refuted -- the arithmetic was
    right and the inference past it was not.** Not disjoint does not mean the
    same: Jaccard **0.2026**, P(fill-adjacent | bridge) 0.6846,
    P(bridge | fill-adjacent) 0.2235, and **bridge passage keeps PAF 0.4533
    among tokens that are not fill-adjacent at all**. Track K's distinction is
    the one that matters: K4a's finding is about **what kind of bridge**
    collapses, not about which tokens the two exposures cover. The four-cell
    table is the clearest form (mark rate, ratio to neither): both 0.03807
    (20.86x), bridge only 0.01895 (10.39x), fill only 0.00312 (1.71x), neither
    0.00182. Re-read here, the two behave as **roughly independent
    multiplicative factors** rather than a hierarchy -- fill adjacency's RR is
    2.009 within bridge-passing tokens and 1.708 within the rest, so there is
    no strong interaction, and the joint cell's 20.86x sits only a little above
    the 17.8x multiplicativity predicts -- with bridge passage much the
    stronger factor.
  - **The registered exposure is bridge passage, not the union** (decided
    2026-10-18). The union clears the bar more comfortably (prevalence 0.5632,
    89.19% of the marks, PAF 0.7344 [0.6267, 0.8364]) and is still the wrong
    choice, for two reasons. **PAF rises with the width of the exposure**, so
    folding in 9,948 fill-only tokens at a crude 1.71x inflates it by
    prevalence rather than by mechanism -- widening an exposure to clear a PAF
    bar is the same move as relaxing the bar. And decisively: **an exposure has
    to be something a repair can change.** Being next to a fill is a property
    of the drawing and cannot be repaired; passing a collapsed bridge is a
    property of one line of Track F's rule,
    `len(span) <= max(3.0, 1.5 x median(local ink width))`. So the exposure is
    **bridge passage: prevalence 0.1667, 72.43% of the marks, PAF 0.5759
    [0.4617, 0.6854]**, with fill adjacency demoted to a stratifier and a
    secondary (its independent contribution, MH-RR 2.77, and the four-cell
    table printed again on fresh material). The bar is **the PAF CI's lower
    bound at or above 0.30**, which the read estimate predicts, with a CI upper
    bound below 0.30 as a refutation; the null value is MH-RR 1 under permuting
    marks within length decile, and the pre-read prints are the crude
    two-formula agreement, the stratum contributions, and the abstention counts.
  - **The reserve exists and was designed for this on 2026-10-04 -- the
    foundation got this wrong twice, both times by reasoning about the material
    instead of reading it.** First it prescribed confirming on Track J's
    chainer band `[171:]` without checking that the outcome, the author's
    `broken` mark, comes from labels and not from panels. Then it concluded no
    reserve existed at all and that only the author's future marks could serve
    -- without opening `labels/holdout_plan.json`, which has held one since
    2026-10-04: seed 20261004, **test_panel 35 faces / 63 marks / 29 panels**
    and **test_series 35 faces / 72 marks / 24 panels**, 10 whole works set
    aside for the second, **70 marked faces and 135 marks entirely outside the
    177 panels K4 read**. **The foundation cites the file when it states a fact
    about a track's material**, now in the traps. (Also recorded: the `broken`
    arrays contain duplicates, 246 entries collapsing to 185 unique
    (order, sid), so K4's 185 is the right count and no labelled mark is
    missing from `per_stroke.csv`.) **No new demand on the author is needed**,
    and K3 stays stopped -- `face_groups.jsonl` is identical to K1's snapshot,
    so no labels have been added.
  - **The abstention condition was written on the wrong basis, and the fix is
    not a relaxation of the bar.** The condition said "at least 20 expected
    exposed marks under the null", which at prevalence 0.1667 gives 10.5 and
    12.0 in the two bands and would abstain. But **requiring precision under
    the null asks for it in the scenario that needs it least**: if the null
    holds, refuting only needs the PAF CI's upper bound under 0.30, which a
    small exposed cell delivers, while precision matters on the supportive
    side, where the exposed cell is three times the null expectation. Computed
    here at ~269 tokens per panel and the discovery effect of MH-RR 4.8785:
    test_panel reaches a **PAF lower bound of 0.329** if the effect replicates
    and a **PAF upper bound of 0.135** under the null; test_series, **0.334**
    and **0.124**. Both directions land in both bands, so the condition is
    replaced by the thing the 20 stood for -- **before the material is read,
    compute and print that the registered bar is reachable both ways at this
    band's n and prevalence; a band where it is not is not read** -- and **the
    bar itself does not move**. Margin is thin: 0.329 and 0.334 sit just over
    0.30, so **a held verdict is a live outcome and is to be reported as held**,
    not as nearly cleared. The two bands are **not pooled** -- test_panel asks
    about reproduction within the same art style, test_series about
    generalisation across works -- are **read once, simultaneously**, since
    holding two bands must not become two attempts, and **disagreement between
    them is itself a finding**: panel confirming while series fails would make
    the mechanism style-specific. K7 and registration A, both label-derived,
    are registered **at the same time**, because coming back to the same band
    with a later idea is the dullest way to spend it.
    **Registered 2026-10-18, and only (0) opens the band.** Two things settled
    it. First, **(i) cannot be answered by the holdout at all**: its outcome is
    the author's skipped-versus-faint judgment, which exists only for K10's 40
    cases, all of them in train -- the foundation had asserted it could ride the
    holdout, its third material claim of the day made without reading, and
    again stopped by the track quoting the file. Second, **the proportionate
    abstention condition needs only n and the exposure prevalence, so it can be
    computed without reading the outcome** -- which means a registration can be
    abstained from *without opening the band*, leaving the record that removes
    any later reason to reopen it and spending nothing.
    So: **registration A abstains on structure** (the holdout holds 8 + 12 = 20
    `face.other` groups against train's 55; a median difference with a
    face-clustered CI cannot reach the supportive side there), and stays a
    description on read material with face 267 named, printed at `thr` only and
    with grow per the standing constraint. Its `residue_frac` predicate is
    approved as written.
    **K7's question abstains and also leaves the confirmatory queue.** It does
    not reach: computed here from 4.42 pairs per panel, a box-B share of 0.233,
    mouth prevalence 0.1867 and the discovery MH-RR of 1.3418, test_panel has
    ~51 box-B cases and a supportive CI of **[0.78, 2.32]**, test_series ~31
    and **[0.66, 2.72]** -- **the CI includes 1 even if the effect replicates
    exactly**, and lowering the bar to fit the power is choosing the bar from
    the material. The deeper reason is that **no decision depends on the
    answer**: the mouth is already known to be 4th of 8 on read material
    (0.343, behind `face.other` 0.782, glasses 0.545 and `face.ear` 0.388),
    nothing is repaired per part, and the author's actual claim -- that skipped
    lines are concentrated in mouths -- was answered by K10's 40
    (`face.mouth` 8 skipped / 1 faint / 5 can't-tell of 14). So K7 closes as
    description. **Its registration is kept, not discarded**, because it was
    written before any read, and the material it would need is now quantified:
    **about 247 box-B cases** to put the CI above 1 at MH-RR 1.3418, against
    the discovery's 182 -- the number to hand the user if they ever ask whether
    more labelling pays.
  - **The holdout was opened once and the mechanism survived on the harder
    band (2026-10-18).** `test_series` -- ten whole works held out, the
    cross-work generalisation test -- gives **MH-RR 3.9018, PAF 0.4855, CI
    [0.3245, 0.6400]: support**. `test_panel` gives **MH-RR 4.3629, PAF 0.4894,
    CI [0.2578, 0.7581]: held**. **No overall verdict is registered**, because
    this file wrote "report the bands separately, do not pool" and never wrote
    a combination rule, so none is invented after the fact. What can be said:
    the point estimates are nearly identical (0.4894 against 0.4855), the
    verdicts differ on **precision** (8 of 10 contributing deciles against 6 of
    10) and not on effect, the held band's interval is not in conflict with
    0.30, and both sit a little under the discovery's 0.5759 [0.4617, 0.6854].
    **This file's registered expectation was wrong in a useful direction**: it
    said that if the panel band confirmed while the series band failed the
    mechanism would be style-specific, and the result is the reverse, so **the
    mechanism is not style-specific**. The instruments held throughout -- crude
    two-formula agreement exact in both bands, the **negative control
    recovering the null in both** (0.9577 and 0.9880), and the proportionate
    abstention printed before the outcome was read. **The condition for
    discussing a repair, PAF >= 0.30 on held-out material, is met.**
  - **The benefit is measured and the cost is not, so no decision goes to the
    user yet.** Two warnings already point the same way: K4a found the loss-2
    collapsed bridges are mostly a fill's medial axis (local ink width median
    26.8px), so raising the threshold eats fill axes, and K10's author answers
    already refuted unconditional loosening from the other direction. The next
    registration therefore puts **the primary value on the cost side**, on
    train material and declared as design exploration rather than a test: sweep
    the collapse rule's coefficient with the trivial arms at both ends (current
    value, collapse everything, collapse nothing), and score against **the
    author's part groups**, which cost nothing and measure **both** directions
    -- *fixed* = a marked token splits with its pieces landing in different
    part groups, *broken* = an unmarked token that sat inside a single part
    group splits. That last count is **the first number anyone has put on
    over-splitting**, the direction the author cannot mark, which is the
    constraint Track K was opened over. Primary: **broken / fixed, bar < 1.0**
    -- marks are 0.74% of tokens, so mis-splitting 1% of unmarked tokens breaks
    as many as the repair fixes. Secondary: the share of newly split or
    absorbed structure with local ink width over 8px, against K4a's 92.7%, as
    the fill-axis check. The bar is written before the sweep, and **a
    coefficient chosen from the sweep is a design choice, not a confirmed
    value** -- confirming it needs new marks, the K3 material.
    **The fragment-to-part correspondence rule is removed rather than
    written** (2026-10-18; the track asked for it and was right that such a
    rule manufactures its own result). The foundation's first phrasing --
    *fixed* = a marked token's pieces land in different part groups --
    **demanded information the labels do not contain**: `broken` says the token
    should be taken apart and never says *where*, and the author assigned the
    fused token a single part. So the definitions become **split or not**:
    *fixed* = a token carrying `broken` splits at this coefficient, *broken* =
    a token without the mark splits, since the author had the tool and did not
    use it. The part groups drop out of the primary and become the cost's
    breakdown, where an unmarked token that sat wholly inside one group is the
    strongest cost. No pixel correspondence and no voting are needed, because
    **the skeleton is unchanged across the sweep** -- skeletonisation is
    upstream and only the span-to-token assembly moves -- so "did this token
    split" is decided on one fixed pixel universe. One degree of freedom fewer.
    **Required exclusion**: only faces where the `broken` field exists may
    supply controls; the 23 oldest faces record the mark as **unavailable, not
    absent**, and leaving them in the denominator dilutes the cost with
    material that could not carry a mark. Secondary: the fragment-count
    distribution in both arms, since a marked token splitting into 2 and into 7
    are the same "fixed" and different phenomena.
  - **The cost bar is demanding, and the foundation registers a prediction that
    the unconditional sweep fails.** With about 24,850 unmarked tokens, an
    unmarked split rate of 0.5% already gives 124 broken against 148 fixed at
    80% of marks (ratio 0.84), and 1.0% gives 249 against 148 (1.68). So
    **bar < 1.0 requires holding the unmarked split rate under roughly 0.5%
    while splitting most marked tokens**, which no global lowering of the
    coefficient can do, because a global threshold splits unmarked tokens at
    the same rate. **Registered before the sweep: no uniform coefficient clears
    the bar.**
  - **So the arm the mechanism actually implies rides the same sweep: a width
    gate.** K4a established that the collapsed bridges are mostly a fill's
    medial axis -- local ink width median 26.8px, 92.7% over 8px -- against a
    drawn line's ~2.7px, window-invariant. The repair the mechanism points to
    is therefore not a lower threshold but **collapse only bridges whose local
    ink width is at line scale, and never at fill scale**, which is a
    **narrowing** rather than a loosening and adds one degree of freedom whose
    value comes from an existing measurement. Arms: coefficient
    {0, 0.75, 1.0, 1.25, **1.5 current**, 2.0, 3.0, inf} x {no gate, width gate
    W} -- same bar, same reference, same prints, **one sweep**, so that nothing
    comes back for a second pass. The over-8px secondary doubles as the check
    that the gate is working.
    **Every arm failed the bar (2026-10-18/19), the registered prediction held,
    and the width gate the mechanism implied failed too.** Best arm, coefficient
    1.5 with a 4px gate: **113 fixed, 2,895 broken, ratio 25.6**; the strong
    cost alone (unmarked tokens that sat inside one part group) is 1.87 to 6.12
    across arms, never under 1. Both trivial arms are the worst (26.8 and 40.1),
    which is how we know there is no good point in between, and the ratio barely
    moves with the knob (26-28 across coefficients 0 to 1.25) -- a global
    threshold **trades fixes for breaks at a fixed ~26:1**, so this was never a
    question of choosing the setting. The gate did what it was built for (the
    over-8px share rises to 0.75-0.79, fill-scale bridges held back) and the
    ratio still did not move, which the track flagged as the heavy fact.
    **It is not a mystery, and the number was available before the sweep**: the
    exposure's precision is P(mark | passes a collapsed bridge) =
    134/4,182 = **3.20%**, so acting on every bridge-passing token has a floor
    ratio of (4182-134)/134 = **30.2**. The gate raised precision to **3.76%**
    and bought 25.6. **A ratio under 1.0 needs precision over 50% -- 13.3x the
    best arm.** So the conclusion is stronger than "not worth it now":
    **no rule keyed on collapsed-bridge passage can reach the cost bar**,
    whatever the threshold or the gate.
  - **PAF 0.49 and a 26:1 cost ratio are not in conflict, and this is the
    distinction the whole sweep was for.** They are two readings of one table:
    the attributable fraction answers *how much of the problem is here* (half
    of it), precision answers *does acting here hit anything* (3.2%), and the
    two diverge exactly when the exposure is common (16.7%) and the outcome
    rare (0.74%). **A PAF bar licenses discussing a repair; whether one can be
    built is decided by precision.** The bar was the right gate for the
    discussion and never implied the repair existed. The requirement for any
    future attempt is now a specification rather than a hunt: **a feature that
    distinguishes where to split must raise precision inside the
    bridge-passing set from 3.8% to over 50%**; anything less does not bring
    the ratio under 1. The one direction the numbers do not kill, unmeasured
    and unregistered: a repair that **does not decide** -- carrying both
    tokenisations downstream instead of committing to one -- which is an
    architecture change, not a threshold change.
  - **The cost ratio is algebraically precision, and the identity held across
    all 21 arms** (verified by the track, 2026-10-19: broken/fixed equals
    (1-precision)/precision to rounding in every arm). So the sweep carried one
    number per arm, and its **floor was computable from the four-cell table
    before it ran** -- precision 3.20% implies a ratio of 30.2 for any rule
    acting on the whole exposure. What it did add, and what was not knowable in
    advance, is **how much a width gate could raise precision: 0.56 points,
    3.20% to 3.76%**. Worth carrying as a design habit: **before sweeping,
    check whether the target metric is an algebraic function of something
    already measured** -- that part is already answered, and only the rest
    needs the sweep.
  - **Track K's own summary of 2026-10-18/19**:
    `inbox/note_track_k_day_summary_20261019.md`. It records, besides the
    results, that the author spent **one session of 56 judgments** in total,
    that instruments stopped the work **5 times** (twice preventing a reported
    pass), that this foundation's readings were stopped **twice** and its
    circular check once, and that material was mis-identified 3 times by the
    foundation and twice by the track -- **every one of them before a result
    was published**. Track K is quiet until new marks exist.
  - **Where Track K stands, 2026-10-19 (read this instead of the bullets
    above).** *Established*: the connectivity error has two directions and only
    over-merge is markable; the author's marks concentrate on tokens passing a
    collapsed bridge (PAF 0.4855, CI [0.3245, 0.6400], confirmed on ten
    held-out works) and that mechanism is **not style-specific**; fill
    adjacency is a second, weaker and roughly independent factor; F's docstring
    is quantitatively wrong (53.4% of bridges survive); K4a's 21.2% is fills.
    *Dead*: box B's gray axis, the continuity mechanism, the hysteresis bridge
    as an explanation of the marks, and any repair keyed on collapsed-bridge
    passage (precision ceiling ~4% against the >50% a cost bar needs).
    *Two statements of what must not be done*, both from labels: unconditional
    loosening is refuted by the author's 40 judgments, and the collapse rule
    cannot be retuned to exploit the confirmed mechanism. *Left*: a repair that
    does not decide -- carrying both tokenisations downstream -- unmeasured and
    unregistered; and a precision-raising feature, which needs new marks (K3)
    and must clear a 13x specification written before any search begins.
  - **Two corrections the foundation owes on this leg.** The "23 oldest faces
    without a `broken` field" was carried over from a 2026-10-16 note and never
    checked; **measured, it is 14 faces and 7,173 tokens**. And the track's own
    first pass counted the cost over all 90,111 tokens before catching that the
    author only ever saw the 25,035 in face regions -- **the same trap one level
    below the distinction this file had just drawn**, with the first ratios
    (85-139) kept in the record beside the corrected 25.6-40.1.
  - **The day's one positive result has a cheap successor: a conditional rule.**
    The author's answers refuted loosening *without a condition*; gap length
    separated the labels (11.96px against 5.66px, p=0.0057), so "join short
    gaps, leave long ones" is possible in principle. Estimate it -- AUC and a
    threshold with its error rate -- on the **31 decided answers**, as an
    estimate and not a test, **with and without `face.other`**, because 7 of
    the 9 faint cases are cheek hatching and the threshold may be nothing more
    than "exclude the hatching", which would move the repair back into
    registration A. The threshold is registered as a number before the band is
    opened. This is the "measure a computable axis before asking the judge"
    rule run backwards: **the judge's answers calibrating a computable axis.**
    **Measured, and the suspicion was right.** On the 31 decided answers the
    gap-length AUC is 0.8232 with a best threshold of 5.39px and 5 errors, but
    the rule is **one-sided** -- it catches 22 of 22 skipped lines and 4 of 9
    faint -- and dropping `face.other` leaves **2** faint cases, so the 0.8864
    AUC on 24 is not usable, while `face.other` alone is 0 of 7 skipped. What
    can be written: the gap-length threshold **cannot be separated from the
    fact that 7 of the 9 faint answers are `face.other`**. What cannot: that
    gap length repairs anything, or that an AUC of 0.886 predicts anything.
    **The repair's location moves to registration A.**
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
