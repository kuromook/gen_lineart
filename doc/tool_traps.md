# Known Tool Traps -- the evidence

Moved out of `doc/CURRENT.md` on 2026-10-24, verbatim, when that file had grown
to 6,820 lines. **The rules live in the index** in `doc/CURRENT.md § Known Tool
Traps`; this file holds each one's incident, mechanism and numbers. Nothing was
deleted and no measurement was dropped in the move: the 83 entries below are the
section's own text, in its own order, with a `T<n>` number added so the index can
point at one.

Two things to know before reading:

- **Numbers here are not a current state.** Several entries are the record of a
  retraction, and the retracted figure is quoted next to its withdrawal on
  purpose. `doc/CURRENT.md`'s opening paragraphs are the only place that says
  what currently holds.
- **T63-T75 and T80-T81 are not traps.** They are fill-line findings and status
  that accreted into this section during 2026-10-23/24. They are kept here
  because they are the record; the fill line's live state is in
  `doc/CURRENT.md`.

Add a new trap by appending it here with the next number, and adding its one-line
general form to the index in `doc/CURRENT.md`.

- **T1. A long run's own log can be the thing a crash destroys** (Track J,
  2026-10-19, caught in advance while M1 was in flight). The trainer writes
  `train_<arm>.jsonl` with mode `"w"`, and there is no `--resume`, so a crash
  plus a relaunch **truncates the holdout curve that the 4,000-step
  checkpoints were supposed to make recoverable**: the weights survive and the
  record of how they got there does not. The procedure is to copy the jsonl
  aside before relaunching, and the general form is that **a multi-day run's
  recoverability has to cover its log, not only its checkpoints** -- an
  append-mode slip is invisible until the first crash, which is the worst
  moment to discover it.
- **T2. A rotated chance arm's height is a property of the structure you rotate,
  not of the structure you measure** (Track D, 2026-10-19, refuting a
  prediction the foundation had registered). Both directions of the
  rough-to-line correspondence use the same null -- rotate one skeleton 180
  degrees -- and the foundation predicted the precision side's chance would sit
  *above* the recall side's 0.000-0.012, because the rough's segments are
  shorter and more numerous and a short segment satisfies a median-distance
  test more easily by luck. **The mechanism was confirmed and the prediction
  still failed**: the rough does have 2.2-3.1x the segments at 19.6-41.8px mean
  length against the GT's 49.0-70.1px, but the chance arm came out *lower* on
  average (0.0334 against 0.0523), with `prec_chance > chance_rec` in only
  20-39% of pairs. The reason is that the recall side rotates **the rough**,
  which carries 1.4-2.5x the GT's skeleton length, and a denser rotated
  structure lands near things by accident more often. **Rotating the denser of
  the two raises the floor**, so a chance arm is not comparable across two
  directions unless the same structure is rotated in both -- and when it is
  not, the difference in the floors is the asymmetry, not the signal. The
  named suspect for an inflated null (centrally symmetric panel borders, which
  survive a 180-degree rotation exactly as they survive the horizontal flip
  that was rejected for that reason) **did not need to be called**: the nulls
  came out low.
- **T3. A threshold on grey reads pencil grain as strokes, so an unmatched-length
  figure is not an unused-input figure** (Track D's own caveat on its result,
  adopted 2026-10-19). `gray < 200` skeletonises grain and tone into short
  fragments -- clippairskoma's roughs are largely texture -- so of the ~half of
  rough skeleton length with no GT line within 8px, some is grain rather than
  misplaced line. The measurement stands as registered; **the reading is
  bounded by what the threshold admits.** Sibling of the trap above about a
  summary statistic over all ink being a reading of the solid areas: there a
  threshold-free statistic read the fills, here a threshold reads the paper.
- **T4. "It cannot be checked from here" needs the right command before it is
  written down** (foundation's own, 2026-10-19). This file and
  `doc/worktree_policy.md` recorded GitHub's default branch as unverifiable
  because `gh` is not authenticated. It takes no authentication at all:
  `git ls-remote --symref origin HEAD` prints `ref: refs/heads/main`. The same
  shape as the other two misses of the week -- a fact about someone else's
  material asserted from the first tool that failed, rather than from the tool
  that answers the question.
- **T5. A bar placed on a ratio can be refuted while the mechanism it was placed
  to test is actually happening** (Track J, IC7, reported 2026-10-19; the
  sibling of Track K's "the ratio was precision all along"). The leg asked
  whether fusion could be scored, with the bar `npred/ngt < 0.90`. The measured
  ratio was **1.777**, so the bar was broken -- and **fusion was nevertheless
  occurring**: 14.2% of the GT area sat inside a fused part. A **second**
  mechanism was moving the same ratio the other way: already-dilated small
  blobs that appear across the 500px filter (`d=8.485`, 57.5% of the predicted
  area). Two mechanisms, opposite signs, one ratio, and the ratio reported
  neither. **A criterion that tests a mechanism names the counting
  decomposition, not a ratio** -- how many of what land where, each side
  printed separately. J's next registration drops the ratio and puts a bar on
  each of the two shares; they already carry numbers, at three dilations:
  fused-GT share and crossing-prediction share read **0.142 / 0.575** at
  `d=8.485`, **0.105 / 0.374** at `d=4`, **0.048 / 0.113** at `d=1`
  (`../lineart-solid-fill/results/fill_koma_20261010/ic7_npred_decomposition.txt`,
  48 panels).
  **Fixing the grid point is not the same as fixing the bar** (Track J,
  `221b26f`, 2026-10-19, and the distinction is worth keeping). J pinned the
  two dilations at which the shares will be judged -- `d=8.485` (written into
  IC7's registration `324b2a4`, from `0.41421 * sqrt(1500/pi)`) and `d=4.000`
  (IC6's shared integer radius, fixed 2026-10-16) -- both of which existed
  **before** the decomposition table did, so neither was chosen by looking at
  it. It then **refused to let today's 0.142 / 0.575 become the bars**: a
  tolerance read off the table is a tolerance known in advance to admit the one
  number it was taken from, which is the same objection the foundation raised
  against an imported chainer tolerance on 2026-10-18, now landing on the
  foundation's own suggestion. The bars will instead be **derived from the
  mechanism** (e.g. IC6's census that 2.0% of GT fill area lies in components
  under 500px, carried through the geometry of the dilation) and computed
  before the leg runs.
  **The discriminator is the second mechanism's direction against the
  conclusion drawn, not whether the bar passed or broke.** The foundation first
  wrote the latter ("a passed bar is safe in a way a refuted one is not") and
  **Track J refuted it the same day with IC7 itself**: had the second mechanism
  pushed that ratio *down* instead, the bar would have been **passed** and a
  fusion that is not happening declared to be happening -- same statistic, same
  trap, only the verdict reversed. So a bar is robust **only when every second
  mechanism that can be named for its statistic moves it against the conclusion
  being drawn.** J's own one passed ratio bar, IC-j12's rejection clause
  (379/379 unabsorbed, ratio 1.000, bar `< 0.80`), is robust by that test and
  not by the false one: its named second mechanism (loosening `MERGE_PX`
  lengthens fragments past `parts_of`'s `PART_SHARE`, so they stop counting as
  absorbed) *lowers* the ratio, which is the direction that **fires** the
  rejection -- opposite the conclusion.
  The foundation audited Track K's own two ratio bars against this shape on
  2026-10-19, since K had bars on an attributable fraction and on a cost ratio.
  Both survive, by direction and by algebra, which is what the false
  generalisation was built on top of:
  - The **PAF >= 0.30 bar** (0.4855): every dilution known in that table --
    all-case strata, marks arising in the unexposed arm -- pulls MH-RR toward
    1, i.e. **against** the conclusion that the fraction is large, so a hidden
    second mechanism can only understate it.
  - The **cost bar `broken/fixed < 1`**, which was refuted. It survives for an
    algebraic reason: the ratio is
    `(1 - precision) / precision` identically, a strictly monotone function of
    **one** counted quantity, so there is no numerator and denominator for two
    mechanisms to move apart; and the decomposition was printed per arm (four
    arms, precision 0.0243 to 0.0376).
  - Where the trap **does** bite: that precision is pooled over the whole
    exposed set, so a sub-population inside it with high precision would be
    invisible in the 3.8%. That is exactly what K's registered requirement
    specification already demands of any new search (a feature that lifts
    precision from 3.8% past 50%), so **J's lesson and K's spec are one
    instruction in two vocabularies**: name the split of the exposed set, not
    the ratio over it.
- **T6. Do not repair a renamed reference with a repository-wide `sed`** (Track J,
  2026-10-18). `sed s/diffusion/foundation/` across this repo would rewrite the
  dozen-plus `BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/...` lines in
  `experiments/*.sh` and `scripts/*.py` into paths that do not exist, and
  nothing reports it until a job is launched. The `diffusion` -> `foundation`
  rename needed six operative lines, all in `doc/worktree_policy.md`; the other
  hits are citations, a *technique* name inside a prohibition clause, the
  separate branch `diffusion-controlnet`, and that checkpoint path. See
  `doc/worktree_policy.md`'s tail section for the four kinds.
- **T7. A control the reference arm itself fails cannot discriminate** (found
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
- **T8. The instrument graded its own pass condition, and leniently** (caught
  2026-10-18 by Track K itself, on the second run of K2-B'). The registration
  said each of three constructed target classes must land in the right box,
  30 of 30. The implementation set `stop_condition_passed` from
  `n_constructed > 0`, so the first run printed PASS on nine constructed
  targets of which three were misclassified. The track found it, tightened the
  code to the registration, and left the first PASS in the record. **A stop
  condition is code like any other: assert the registered threshold, by its
  number, in the same commit that writes the condition.**
- **T9. A stop condition written about the material teaches nothing about the
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
- **T10. A one-directional fix needs a target that catches over-correction**
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
- **T11. A radius taken from the sample point collapses exactly where it is needed**
  (Track K, 2026-10-18, second failure of the same instrument). B'-1 attributed
  ink to the candidate's own span within a radius built from the ink width *at
  the sampled pixel*; the distance transform returns 0 to 2 at a stroke's edge,
  and grazing one's own ink always happens at the edge. T1/T2 failed 7 of 30
  with `d_own` 2.0-4.9px against radii of 2.0-2.4. The fix (B'-1b, approved) is
  the definition, not a constant: **whether a pixel belongs to a line's body is
  decided by that line's thickness, which is a property of the line, not of the
  pixel's position** -- so take the width at the nearest own skeleton pixel,
  `max(2.0, iw/2 + 1.0)`.
- **T12. A fix derived from the failures passes on those failures by construction**
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
- **T13. The author of a repair is the wrong party to predict its yield** (observed
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
- **T14. A null value belongs to a statistic, not to a study** (Track K's K7,
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
- **T15. A coincidence between two statistics is a degeneracy check before it is a
  finding** (same stop). The permutation median of the crude RR (1.3476) and
  the observed MH-RR (1.3418) agreed to 0.006 although they estimate different
  things. The boring explanation to rule out first is degenerate
  stratification: singleton strata make a within-stratum permutation the
  identity and contribute nothing to MH, so both numbers can be driven by the
  same few strata. Required before interpretation: the number of strata, their
  size distribution, how many are singletons, and what share of the units sit
  in strata that carry both exposure levels and both outcomes.
- **T16. "Compared against everything else" can be the conservative choice, and
  that is checkable** (ruled 2026-10-18). Track K suspected its pre-registered
  exposure (mouth versus all other parts) of being a design error once the
  per-part table showed mouth 4th of 8. Computed from the track's own table:
  the highest-rate part, `face.other` at 0.782, sits in the comparator, and
  removing it raises the exposed RR from 1.6501 to 2.2356. **A comparator that
  contains the strongest non-exposed group dilutes the estimate**, so the
  registered design did not manufacture the result and does not need replacing
  -- while moving the primary onto the table would mean choosing the maximum of
  eight cells post-hoc.
- **T17. A count from a run whose stop condition failed is usable only if its
  predicate never touches the failed path -- and it still needs a scale**
  (ruled 2026-10-18 on Track K's 32.4%). The unmeasurable box survived because
  it is decided by the registered ink rule and geometry alone, with no gray
  classification in it. Two things it still lacked: a name that does not
  overclaim (it says the tips' ink lies in one connected component, not that
  the gap is not a gap -- 4 of 30 constructed real fills landed in the same
  box) and a trivial arm, because the predicate is nearly monotone in gap
  length, so the bare percentage has nothing to be compared against.
- **T18. A definition can be right in mechanism and one case short in coverage**
  (Track K's C-1, 2026-10-18, and the right way to record a partial result).
  The collar redefinition solved exactly what it was derived for -- the rows
  that failed at 27/30 returned 30/30, and the retraction mechanism is closed
  -- and failed only where the filling ink is continuous with the candidate.
  Recorded as coverage missing at one place, not as a refutation of the
  reasoning, because the distinction changes what the next registration
  inherits.
- **T19. An attributable fraction and a precision answer different questions, and
  diverge when the exposure is common and the outcome rare** (established with
  numbers 2026-10-19, Track K's cost sweep). PAF 0.49 said half the author's
  marks sit on tokens passing a collapsed bridge; precision said 3.2% of such
  tokens carry a mark. Both are true, and the second is what a repair lives on:
  a rule that acts on the exposure fixes one token for every 26 it breaks, and
  **a cost ratio under 1 needs precision over 50%**. The floor was computable
  from the four-cell table before any sweep ran. **A PAF bar is a licence to
  discuss a repair, not evidence that one can be built** -- when the bar is
  cleared, the next number to put on the table is precision.
- **T20. A ground-truth label answers only the question it was collected for**
  (ruled 2026-10-18, the foundation correcting its own cost design). The
  author's `broken` mark means "this token should be taken apart" and carries
  no information about **where** the split belongs, and the author assigned the
  fused token one part group. A metric phrased as "the fragments land in
  different part groups" therefore demands information the labels do not hold,
  and the fragment-to-part correspondence it needs would have been a new
  degree of freedom invented to supply it. Phrased as **split or not**, the
  metric needs no correspondence at all. Related exclusion from the same
  ruling: a face whose `broken` field does not exist records the mark as
  **unavailable, not absent**, so it cannot serve as a control.
- **T21. An append-only log read as a set of rows double-counts its keys** (Track
  K, 2026-10-18, found and fixed by the track before it read anything).
  `labels/face_groups.jsonl` carries up to three rows per `order` -- 417 rows
  against 301 unique keys, 113 duplicated -- so the first holdout implementation
  inflated the marks to 75 and 106. Read such a file as **latest-row-per-key**,
  and **print the row count beside the unique-key count** so the gap is
  visible. Track F's loader was safe because `G.S["faces"]` is a dict keyed by
  `order`, which is also why K4's 185 was right. **A bug found late is swept
  backwards**: every earlier result that read these labels prints whether it
  went through the dict path.
- **T22. A registration earns reserved material only by naming the decision its
  answer changes** (ruled 2026-10-18, retiring K7's confirmatory leg). K7 asks
  whether the mouth predicts box-B membership. On read material the mouth is
  **4th of 8** parts, nothing is repaired per part, and the author's own claim
  -- skipped lines concentrated in mouths -- was already answered by the 40
  judgments. Confirming a weak fourth-place enrichment would change nothing,
  and the holdout cannot reach it anyway (the supportive CI includes 1 even at
  the discovery effect size). **A question whose answer changes no decision
  closes as description and does not spend reserved material.** The companion
  rule, from the same ruling: **the proportionate abstention condition is
  computable from n and the exposure prevalence alone, so a registration can be
  abstained from without opening the band** -- the printed computation leaves
  the same durable record at no cost to the reserve.
- **T23. The foundation cites the file when it states a fact about a track's
  material** (its own failure, twice on 2026-10-18). It prescribed a
  confirmation band borrowed from another track without checking where the
  outcome variable came from, then declared that no holdout existed without
  opening the plan file that had described one since 2026-10-04. Both were
  reasoning about material instead of reading it, and the second would have
  sent a needless request to the author. A session whose job is recording and
  routing has no standing to assert a fact about material it has not read.
- **T24. A PAF bar is meaningless unless the exposure is the narrowest set the
  repair would touch** (ruled 2026-10-18 on Track K's exposure choice). PAF
  grows with exposure prevalence even at a weak risk ratio, so a wide exposure
  and a strong mechanism produce the same number: the union of fill adjacency
  and bridge passage reaches PAF 0.7344 at prevalence 0.5632 by folding in
  9,948 tokens elevated only 1.71x. **Widening an exposure to clear a PAF bar
  is the same move as relaxing the bar**, and the defence is to pair the bar
  with the requirement that the exposure name something a repair can change --
  here one line of Track F's collapse rule, rather than the artwork's own
  property of lying next to a fill.
- **T25. "AFe" was naming three different quantities, and two of them were one
  quantity fed an RR from the wrong direction** (found by Track K, 2026-10-18,
  which stopped before registering a bar because the definition flipped the
  verdict). The vocabulary, fixed here: **AFe = (RR-1)/RR**, the fraction among
  the exposed; **PAF = p(RR-1)/(p(RR-1)+1) = p_case(RR-1)/RR**, the share of
  *all* cases a perfect repair would remove. The two PAF formulas are
  algebraically the same **on the crude table**, so computing both and checking
  they agree to the printed digits is a check on the **framing** -- and it is
  the check that fires here. Corrected by Track K the same day: the agreement
  holds only crude. Levin's form assumes a crude RR and a homogeneous
  population and is biased when fed an adjusted one, so **under stratification
  disagreement is normal** (K4 adjusted: Levin 0.3926 against Miettinen 0.5759)
  and the estimator to report is **Miettinen's**, with Levin's printed as a
  sensitivity and not as a check. K4's pool table compares P(bridge | marked) 0.7243 with
  P(bridge | unmarked) 0.1625 and calls the ratio 4.456 an RR: that is an RR
  with **marked as the exposure and bridge as the outcome**, a fine descriptive
  statement that cannot carry an attributable fraction. In the asked direction
  the RR is (134/4182)/(51/20908) = **13.136** and both PAF formulas return
  **0.6692** exactly. Fed direction 2's RR, they return **0.1233 and 0.3314**,
  disagreeing by 0.2082 -- which is how the error announces itself. D, by
  contrast, looked correctly framed -- and that reading was **a circular
  check of the foundation's own making**: the "Miettinen" value it agreed with
  was computed from a `p_case` the foundation had **back-solved out of the
  consistency relation**, so agreement was guaranteed. With the observed
  `p_case` the two are 0.0404 against 0.0676, and D was in direction 2 as well.
  **A check that derives one of its own inputs from the relation it is
  checking always passes; every input must come from the data.** The refutation
  of D survives anyway, which is the one comfortable part: re-measured in the
  asked direction, MH-RR 1.3121 with CI [0.9413, 1.8574] under RR\* 2.4129, and
  PAF 0.1319 (Miettinen) / 0.0865 (Levin) with both CI upper bounds under 0.30.
  **D is refuted under three definitions and both directions.** Every attributable fraction reported
  from here prints both formulas. K7 passes. K1's 0.248 does **not**
  revive: the foundation's algebra implied a prevalence near 0.6766 and the
  measured P(fill-adjacent) is **0.5106**, so 0.248 was a mis-framed quantity
  and not a valid one awaiting a number. In the asked direction K1 is crude RR
  2.9823 with **PAF 0.5030**, both formulas agreeing.
- **T26. An identity key must be derived from the thing identified, not from its
  position in the output** (Track K, found while scoring, 2026-10-18). The
  sheet's `uid` was assigned after the tiles were shuffled, so a repeated tile
  and its first appearance carried different uids and repeats could not be
  identified by key at all. Scoring fell back on (panel, s0, s1) and printed
  **9 pairs, 9 agreeing**; one of those "pairs" was two *anchors* drawn from
  the same span pair. With (panel, s0, s1, gap) it is **8 pairs, 8 agreeing**,
  and the conclusion is unchanged -- but a second defect was hiding inside the
  first: **the anchor set contained a duplicated span pair**, so anchors are
  drawn from mutually distinct span pairs from here on.
- **T27. An anchor whose correct answer is missing from the option set tests
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
- **T28. An anchor set identifiable by a nuisance variable passes for the wrong
  reason** (same ruling). Box A's gaps are mostly 2-8px while the 127
  answerable cases have a gap median of 10.82px, so anchors drawn from box A at
  large are separable by gap length alone and the author could score them
  without ever looking at the ink. Box A holds about 48 pairs at 8px or more
  (27 + 10 + 7 + 4 across the bins), so the anchors are drawn **matched on gap
  length** to the 40 sampled cases.
- **T29. A bar on raw agreement is confounded by the answer marginal** (same
  ruling). Reaching 6 of 8 on a repeat check is a 1.97% event when three
  options are answered at random, which makes it a real bar -- but if the
  author's answers run 80/10/10 the chance agreement is 0.66 and 6 of 8
  becomes a **45.2%** event, so the same bar means nothing. The marginal is
  printed beside the agreement, the bar is agreement at 6 of 8 *and* above the
  marginal-implied chance agreement, and when one option takes 70% or more the
  repeat check prints **uninformative**, not pass. Also: a repeated tile must
  sit at least two sheets from its first appearance, or what is measured is
  short-term memory of the tile.
- **T30. A robustness arm must not contain a repair of its own** (Track K, found and
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
- **T31. A diagnostic that predicts a fix's yield must run the instrument's own
  code path** (Track K, 2026-10-18, the most useful of the day). Before
  measuring B'-1b the track estimated it would repair 8 of 9 known failures; it
  repaired **1 of 7**, because the diagnostic script measured with a fixed
  3.0px collar while the instrument uses `max(2.0, 0.5 x tip ink width)`.
  **The prediction was made outside the instrument, so it was a prediction
  about a different instrument.** The fix's mechanism was still right -- the
  distance transform does collapse at an edge -- and only the yield estimate
  was wrong, which is the correct way to split it. A yield estimate quoted in a
  registration must come from the same call path it is predicting.
- **T32. A threshold compared against a float distance can fail by 2e-14** (Track
  K, 2026-10-18). One of the three band-(b) failures was 3.000000000000021
  against a radius of 3.0. A case that misses by that much is not a
  measurement. The repair adopted here removes the comparison instead of
  tuning it, but any registration that compares a computed distance to a
  threshold **writes the tolerance as a number, or compares in integer pixel
  arithmetic**.
- **T33. Skeletonization retracts at a line's end, so "within half a width of the
  skeleton" is not a definition of the line's body there.** The definition
  holds mid-line and fails at the ends, which means **any instrument that
  starts a measurement at a tip starts inside ink that has no skeleton**.
  Relevant to F, G and J as well as K, which is why it sits here and not only
  in the track entry.
  **Corrected 2026-10-19: the size is 0.5px, not the 10.8px this entry carried
  for a day, and the 10x error was the foundation's** (Track K measured it and
  sent the correction). On box B's **364 tips**, extending outward along the
  tangent in 0.5px steps with the instrument's own
  `ink_mask(thr=128, grow=200)` and stopping on leaving the tip's own connected
  component: **median 0.5px** (quartiles 0.5-0.5, max 5.5), against `iw/2` of
  1.0px at the same tips, correlation 0.737, and the stop reason was *outside
  ink* in **364/364**. It is the geometry of a round cap: at a 2px stroke width
  the ink ends about 1px past the last skeleton pixel. **No measurement of
  10.8px exists anywhere in `results/` or the work logs**; what does exist is
  **10.82px as the median gap length** of box B's 127 answerable cases
  (`work_log.md` line 2312). The direction of the entry was right, the size was
  a borrowed number wearing the wrong name.
  **What depended on it, re-derived**: the entry was used to argue that a
  radius wide enough to cover the retraction *swallows the gap itself*. At
  0.5px against a 10.82px median gap, **it does not**, so that argument is
  withdrawn -- a 1px wider radius would have covered the three failing rows
  (`d_own` 3.00, 4.56, 3.91 against radii 3.00, 4.00, 3.00) without reaching
  the gap. The decision it supported still stands, but **on the
  pre-registration ground only**: the radius had already moved twice and the
  registration said it does not move a third time. A decision keeps its force
  from the rule it was registered under, not from the number quoted beside it.
- **T34. When one quantity's median equals another quantity's median, suspect a
  swapped name before a real coincidence** (Track K's general form of the
  correction above, 2026-10-19). 10.8 arrived as a retraction and was in fact
  the gap median sitting two sections away in the same log.
- **T35. A registered exit clause that names a successor *method* is a guess; name
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
- **T36. Disjointness should be a property of the construction, not of a draw**
  (Track K's amendment to this file's prescription, 2026-10-18, accepted as the
  better form). The foundation prescribed a fresh band of sites under "a new
  seed 20261018b"; two independent permutations can and do collide, so the
  track used **the next interval of the same permutation** instead -- band (a)
  = 0..299, band (b) = 300..599, band (c) = 900+ -- where no overlap is
  possible by construction.
- **T37. Replacing an approximation with the concept it approximated is a different
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
- **T38. A whole-file sha of a shared dependency is coarser than the dependency**
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
- **T39. A set-level zero on a regression count is not a cost bar, it is a refusal
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
- **T40. Selecting on a zero-valued count manufactures a zero** (Track J,
  2026-10-18). The challenger cell was picked by one mechanical rule, highest
  score among discovery cells with `broken = 0`, and broke 0 of 311 there and
  **2 of 379** on the never-read set. The recall and merit legs transferred
  with their direction predicted; only the cost half did not. This is the
  winner's curse in its count form: **if a grid cell is chosen partly because
  a count was zero, expect that zero not to replicate.**
- **T41. A registered expected value must name the entry or the results file it
  came from; a date and a sample size are not a citation** (Track J,
  2026-10-18). IC-j10's second expected value cited "an r-census on 60 held
  panels recorded 2026-10-14". There is no entry on that date, no commit (the
  history goes 10-11 to 10-16) and no results file holding the number. The
  measurement came out at 0.8588 against a band of 0.63 +/- 0.10, and the band
  refuted nothing because **it was never a measurement**. The track found it
  by going back to reread the census as its own registration instructed. This
  one is not a flaw in a control's design: it is a registration resting on a
  measurement that does not exist.
- **T42. The cached intermediate is not always the one that made the data, and the
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
- **T43. A control that removes the artifact along with the signal cannot separate
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
- **T44. An instrument built from artificial damage cannot contain damage the
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

- **T45. `bipartite_match_f1` (the implementation behind `gt_bsds_f1`) can take
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
- **T46. The `"train" if name.startswith("housei") else "test"` split heuristic was
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
- **T47. The shared `evaluate_stroke_stability.skeletonize()` leaves a 2px-wide
  skeleton.** Cut it into segments at junctions and 64% of the skeleton
  classifies as "junction", shattering lines into dots (only 1.7% of segments
  reach 30px). Use skimage 1px thinning with crossing-number junctions
  instead (2.0% junctions, 70.4% of segments over 30px). Connected-component
  metrics such as `long_component_ratio` do not depend on width and are
  unaffected.
- **T48. Binarising a conditioning map before skeletonising changes results by
  half.** A naive `>32` manufactures false junctions and 1-3px fragments from
  edge jaggies; filling holes first takes skeleton capture from 0.42 to 0.62
  and a measured ceiling from 0.193 to 0.301.
- **T49. `manga_line` is nearly empty on the training pairs.** It holds 12% of GT's
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
- **T50. Do not read "strokes drawn where the conditioning map has none"
  (`gt_only`) on its own.** A degenerate output that covers the whole tile in
  dither scores highest on it; across 15 snapshots it correlates with
  `gt_bsds_f1` at **-0.269**. It also tracks the "strokes the map does have"
  column at r 0.75-0.90 within an arm, so report the normalised form too, and
  always beside f1, `fill_ratio`, `near_white_frac` and a montage.
- **T51. The axis trap in Track F's `cluster_set`, settled 2026-10-04 -- and the
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
- **T52. The line-PNG check has no sensitivity to the orientation of a face box.**
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
- **T53. A negative control has to name one concrete error, not a direction.**
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
- **T54. `cluster_set`'s unit is built from proximity alone, so it mixes parts and
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
- **T55. `lineart_003_000` has a blank ground truth and is in
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
- **T56. A summary statistic taken over *all* ink is a reading of the solid areas
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
- **T57. An arm can beat its off-by-one control and still carry no location
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
- **T58. Per-pair alignment scores now exist for all 8,467 training pairs**:
  `../lineart-pair-signal/results/pair_alignment_strata_20260915/per_pair.csv`
  (rough / manga_line / lineart_coarse, each with `le3`, `3to8`, `gt8`,
  `chance_le3`, `aligned`, `density`). Usable for weighting or excluding
  training data. Chance is measured by rotating each conditioning map 180
  degrees; horizontal flip was rejected because panel borders survive it and
  inflate the baseline.
- **T59. An audit read a path the ledger says does not exist as a claim that it
  does.** `tools/audit_track_ledger.py` matched `` `../lineart-...` `` with a
  bare regex, so Track K's released letter-L reservation -- recorded correctly
  in the ledger as "no `../lineart-stroke-bridging` worktree was ever created"
  -- came back as a finding: "ledger names it but no such directory exists".
  Found 2026-10-19 while checking that the start-up command in the new
  `CLAUDE.md` runs. **The record in this file was already right; the tool was
  the broken side**, and its finding pointed a reader at a correct entry to
  "fix" it. Fixed by reading the negation out of the sentence the name sits in
  (`claimed_dirs` / `NON_CLAIM`, with `--self-test`). The general form:
  **a check that scans prose for identifiers will count a name written in order
  to deny it.** **A second instance arrived 2026-10-23 from an unrelated
  direction**: the sweep written to find unimplemented controls counted `IC-j13`,
  whose only occurrence in Track J's log is the sentence *"what I am **not** doing
  is registering IC-j13"*. Two instances in four days in two unrelated checks, so
  this is a property of **any** prose-scanning check and not of the audit tool.
  It is the expensive direction of a false positive, because such a check has no
  author standing by to say "that one is fine" and its output is read as
  authoritative -- discipline 4's most of all, being the one check that depends on
  nobody's attention.
  **A third false positive from the same tool the same day, on a different
  reader.** The audit matched a notice to its sender by looking for a
  `` `lineart-<topic>` `` directory name in its opening, and Track J's slot
  verdict identifies itself as *"branch `solid-fill`"* -- the branch, not the
  directory. So the notice counted for nobody, and the audit reported
  *"`lineart-solid-fill` has committed 5d past its last notice -- unreported
  results are likely"* **against the one track that had just reported a four-day
  slot in full**. Fixed by reading the branch as a last resort
  (`notice_senders()`, branch `<topic>` -> `lineart-<topic>` per CLAUDE.md §5),
  with the two spellings added to `--self-test` (6 cases). The shape to keep:
  **a check that matches records by one spelling of a name will report the
  absence of everything spelled the other way**, and the report reads as a
  finding about the track rather than about the reader.
- **T60. A layer can be present, intended and fully specified, and still measure as
  empty, because its content is a render-time effect rather than pixels.**
  Found 2026-10-23 by the extraction session. A CLIP STUDIO tone layer keeps its
  shape in `LayerLayerMaskMipmap` and its density, LPI and angle in
  `LayerEffectInfo`, while `LayerRenderMipmap` -- the only field the extractor
  and the surveyor read -- is **empty**, since the dots are composited when the
  page is drawn. So every tone layer in this corpus logged `drawn_ratio = 0`,
  and a project whose every instrument counts ink had **no way to notice that an
  entire category of area was there**. The foundation compounded it by reasoning
  from the discard side: "you already identify these layers, so export them
  instead of discarding them" was right for ベタ and wrong for トーン, where
  nothing was being discarded because nothing had been seen. **The check: when a
  class of content is expected and the measurement says zero, ask whether the
  measurement can see that class at all before concluding it is absent.**
  **Widened 2026-10-26 by Track M, from the extraction side's third reply, and
  the widening changes what kind of trap this is.** The reader **fills unsaved
  blocks with 0**. So for a layer whose `body_init == 255` -- tone bodies, fill
  bodies, paper -- that is **partially** saved (`0 < body_stored < body_blocks`),
  the measured value is a **lower bound**, not a value. `drawn_ratio` goes
  through the same reader, so every `drawn_ratio` on such a layer is a lower
  bound too. **What looked like a question about the meaning of zero is a
  question about the meaning of every value: zero is not special, it is the case
  where the lower bound reached 0.** The partition is now five, not two:
  `255` + nothing saved (**possibly the full canvas**) / `0` + nothing saved
  (**empty**) / `0` + something saved (**the contents really are all 0**) /
  `255` + **fully** saved (**the value means what it says**) / `255` +
  **partially** saved (**a lower bound**). Where `body_init == 0` there is no
  bias at all, the fill value being the same as `InitColor`. **M got the same
  one-line caveat wrong twice in two days** -- first the direction of the
  implication (`255` implies 0-fill, not the reverse), then the coarseness of the
  condition (`body_stored > 0` lumping *fully* and *partially* saved together,
  whose values mean different things) -- **both times from writing a rule about
  how to read an instrument without knowing the instrument's innards**, hence the
  practice it registered: **when you register a rule for reading someone else's
  column, send that notice to whoever built the column.** **The same property
  should be expected of every coverage column computed by this reader, including
  `raster_cover` and `mask_cover`, whose non-zero values on partially-saved
  `255` layers would also be lower bounds** -- asked of the extraction side
  2026-10-26, not assumed, **and answered the same night, narrowing it twice.**
  **`raster_cover` can be a lower bound** (same reader) **but is not one in the
  deliveries that exist**: across `grad_v2`'s 258 layers and `tone_v2`'s 619,
  **no layer is a partially-saved `body_init 255`** -- the 96 at `255` are all
  **saved nothing**, which is the "reads 0, is the full canvas" form, and the
  other 162 are `init 0` and mean what they say. **The mechanism is real and the
  instance count here is zero**, which is the outcome Track M had pre-committed
  to writing as *理屈だけで実体が無い*. **`mask_cover` is not a lower bound at
  all**: a different reader (`mask_coverage`) fills both unsaved blocks and
  canvas overflow with **`InitColor`** rather than 0, and its CSV already carries
  `init`, `blocks`, `stored` and the dimensions. **The columns that can be lower
  bounds are exactly those the render reader computes** -- `drawn_ratio`,
  `black_among_drawn`, `drawn_gray_mean`, `ink_px`, `preview_dark`,
  `raster_cover`, `alpha >= 1` -- **and only on `body_init 255` layers that are
  partially saved**, whose full count will come from the `alpha >= 1` table the
  author ordered first. **`preview_dark` and `black_among_drawn` being on that
  list reaches Track L's selections**, which used both as gates on 2026-10-26.
  **And the extraction side volunteered an unmeasured one**: the mask reader
  **does not add `LayerMaskOffset`**, so coverage may move on offset layers, and
  **neither the magnitude nor the number of affected layers is measured** --
  carried as open, not as small. **A column of lower bounds supports "at least this
  much" and nothing else: no mean, no ranking, no threshold in the direction of
  the bound.** The
  sibling failure in the same reply is the mirror image -- **a class defined by a
  pixel statistic collects everything that shares the statistic**: page
  furniture (paper, MASK, トンボ, bleed and panel frames, 集中線) is 100% pure
  black among drawn pixels exactly as a fill is, 5,480 layers of it, so it has
  to be dropped by name before any fill count means anything.
  **The subtler sibling, 2026-10-23 evening: a metric can be valid on one
  construction and silently invalid on another, and still return a plausible
  number.** The foundation asked whether `肌グラデ` rides the same hill as the
  blurred tone, on `mask_mid_per_boundary`. The extraction side answered that
  **the question is malformed**: a gradient-fill layer cuts its area with a
  **hard** mask and softens the paint *inside* it, while a blurred tone layer has
  a **soft mask**. `mask_mid_per_boundary` measures the mask, so **the softness of
  a gradient fill is invisible to it in principle**. The dangerous part is what
  it returns instead -- on ako5's 17 gradient layers, **p10 0.000 / p50 0.78 /
  p90 1.07 / max 1.67, two of eleven above 1.0**, which lands squarely on the
  hard hill (tone's hard hill is p90 0.93). **A number in a believable range is
  not evidence that the instrument measured the thing.** The earlier version of
  this trap was an instrument that returned *zero* for a class it could not see;
  this one returns a number that reads as an answer. The check that catches it is
  not a range test: it is **asking what the instrument is physically attached
  to** -- here, the mask -- and whether the property lives there.
  **Third instance, 2026-10-23, from the extraction side and pointing the other
  way.** Its first inventory reported **84 works with ベタ and 1,246 ベタ layers**;
  the corrected figures are **55 works and 299 layers**. The cause was the layer
  name `black`, which in this corpus is a **line** layer -- the ink pass, not a
  fill -- and only **70 of 1,012** `black` layers met the fill signature at all.
  Its own words: the same structure of error as the foundation's 519-layer
  reading, *"画素統計ではなく名前の側で"* -- on the name side instead of the pixel
  statistic. The pair is now symmetric and both halves are on record: **a class
  defined by a pixel statistic collects everything that shares the statistic,
  and a class defined by a name collects everything that shares the name.**
  Neither side is the safe one, and the only thing that held in either case was
  requiring the name **and** the signature to agree before a layer counts.
  **Fourth answer, 2026-10-27, and the exposure moved class rather than
  vanishing.** The 905-page `alpha >= 1` table arrived (62,105 layers,
  `(page_path, layer_id)` duplicates **0**, missing pages **0**, errors **0**)
  and carries the whole five-way partition. Partially-saved `255` -- the only
  class that is a bound -- is **540 layers on 238 pages**, and **every one is
  a correction layer (`layer_type` 4098/4096, 403) or paper (1584, 137):
  `mask_fill`, tone, gradient and raster have 0.** Reproduced here from the
  CSV without using the supplier's table: the five classes come out **13,265 /
  3,934 / 43,625 / 741 / 540** exactly, and of the 540 only **246 rendered at
  all** -- the other 294 carry no value to be a bound of. Both of M's known
  cells PASS: class 1's 13,061 drawn layers are all `alpha_ge1_ratio` 0, class
  4's 721 all above 0. **But the column is still unusable for the layers it
  was ordered for, through a different class.** Of the **14,663 `mask_fill`
  bodies** on these pages, **12,937 -- 88.2% -- sit in class 1 (`255`, nothing
  saved), which reads 0 and may be the entire canvas.** A fill body being a
  uniform colour is correct rather than defective, since its shape lives in
  the mask and its coverage in `mask_cover`. **So T60's hazard for fill layers
  was never the lower bound: it is that an honest zero and a full canvas print
  the same character.** The fear was registered in class 5, the class-5 count
  for fill is 0, and the reading of the column it was meant to protect is no
  better than before -- *下限は消えたが、0 の読みは変わらない。* **And one column name was found
  measuring two quantities across two tables.** `drawn_ratio` agrees with the
  existing `layer_colour` on **55,671 of the 55,810** layers where both have
  values; the **139 that disagree are all paper**, and the cause is a
  definition difference rather than a reader difference -- `layer_colour`
  repaints paper with the drawing colour over the whole canvas, this table
  leaves the pixel data as stored. **For paper it is `layer_colour`'s 1.0 that
  reaches the page.** Two tables, one column name, two quantities, and the
  only reason it is visible is that the supplier joined them and printed the
  mismatch count. It is the across-instrument form of T134 (g), where one
  column name was being given two kinds inside one instrument and a startup
  check could refuse it; **between two instruments there is no startup to
  refuse at, so the join is the check.** **Scope, stated because it is not the
  corpus.** The 905 pages are **872 (request 3) plus the 33 the inventory
  never had** -- verified here against `pages_missing33_20261026.txt`, **all
  33 present** -- so this is the first table in the project that **covers**
  the inventory hole (T115) instead of carrying it. It is **905 of the 7,667
  `.clip` on disk**, so *tone and fill have no lower bounds* is a statement
  about these pages. **Still unmeasured**, and the supplier said so without
  being asked: how much page area the 403 partially-saved correction layers
  actually affect, since they were not re-counted with `InitColor` filled in.
  **And the foundation got the denominator of that very sentence wrong, by the
  mistake this entry had just registered.** It published **14,759 and 87.7%**,
  which is `layer_type == 2` counted off the CSV -- **not `v_layer_kind ==
  mask_fill`**, the axis it named. The supplier's kind-side count is
  **14,663** (88.2%), the difference is **96 layers**, and the foundation
  **had seen the discrepancy while computing** (classes 2 and 3 came out 314
  against 306 and 1,436 against 1,348), noted that the two axes were not
  identical, and **then printed the type number under the kind's name
  anyway**. The retracted figures are *~~14,759~~* and *~~87.7%~~*; the
  conclusion is unchanged, because what it rests on is that the great majority
  of fill bodies sit in class 1, which both axes say. **Where the two axes
  diverge is itself structure**: classes 1 and 4 agree **exactly** (12,937 and
  72), and all 96 of the disagreements sit in classes 2 and 3 -- the
  `body_init == 0` classes -- so `v_layer_kind` is using something beyond
  `layer_type` that moves type-2 layers out of `mask_fill` precisely where the
  body has no `255` fill behind it. **Which 96 is the supplier's to say and
  not the foundation's**, the CSV carrying no kind column at all. The lesson
  is the narrow one and it is this trap from the inside: **a CSV that lacks
  the column you want does not license the nearest column that correlates with
  it**, and the tell was present and read and discarded --
  *二つの軸が一致しないと気付いた上で、片方の数をもう片方の名前で印字した*. T127 is the count-agrees-set-differs
  form, T138 the name-agrees-quantity-differs form, and this is the third:
  **the axis was substituted knowingly and the substitution was not carried
  into the label.**
  **Fourth instance of this exact shape, 2026-10-27, and Track M found it in a
  rule it had written hours earlier.** Its addendum 88 registered an exclusion
  predicate -- `alpha_ge1_ratio == 0` and `body_init == 0` and `body_stored ==
  0` means *shown to be empty* -- and **applied to addendum 65's layer
  (`slimeH1`) it is false**: the measurements really are 0/0/0 with `rendered`
  true, and **the original-size mipmap holds a finished 7016x6543 picture at
  alpha 255 across the whole canvas**. The column reads the render mipmap
  only. **M had already registered, in addenda 80 and 82, that this layer's
  judgement must not use this column** -- and did not apply its own
  registration when writing the rule. **39 of the 229 candidate layers have
  the same shape, every one `has_image_material`.** So the series runs T60 ->
  addendum 84 (every value on a partially-saved `255` is a bound) -> addendum
  88 (all three ordering conditions already retracted) -> **here: a column's
  rule applied to layers it had itself placed outside that column's reach**,
  which is the foundation's `layer_type`-for-`v_layer_kind` substitution in
  the same week and the same family -- *射程の外だと自分で登録した層に、その列の規則を当てた*. **Two
  method points travelled with it and both are better than the finding.**
  First, **M turned the disagreement into a check rather than loosening the
  tolerance.** Its import check fired T138 independently on the 315 delivered
  layers: **13 disagreed, all paper, all class 5**, new table 0.51-0.99
  against the delivery's 1.0, and **zero in the opposite direction** -- so
  instead of widening the tolerance it added two assertions, *if it is a lower
  bound then new <= delivered* and *the row must be class 5*, and paper aside
  the match is **275 of 275 exact**. A disagreement whose direction is
  predicted by the mechanism is evidence, and the right response to it is a
  narrower check, not a wider one. Second, **a known cell can be a literal
  whiff, and M wrote it as one.** Addendum 88's cell was *addendum 65's layer
  appears in the new matrix*, and that page is already settled by the author's
  own instruction, so it never enters the matrix: **the cell as written cannot
  run.** M did not record it as PASS. It tested the diagnosis by **applying
  the rule to that layer** and printing **old rule FAIL / new rule PASS** from
  the instrument. *空振りを PASS と書かない* -- a cell that cannot fire is neither a
  pass nor a fail, and calling it a pass is reading a result from a run that
  did not happen (T132's family from the other side, and 判定保留 as a positive
  act). The repair narrows only: `rendered` must be true, because **a 0 from a
  layer that could not be drawn is not a measurement**, and the pixels must
  have nowhere else to live. **The order went from 160 to 199 layers and not
  one change loosened anything**, with coverage of the 488 non-folder layers
  on 38 pages at **100%** (226 delivered / 199 ordered / 33 paper / 30 shown
  empty / **0 uncovered**).
  **And the foundation called a different population a wider one,
  2026-10-27.** Relaying the 905-page table to Track L it wrote *the 0 on the
  65 priority pages held in a wider population too* -- and **the two tables
  share 8 pages**: 12% of L's 65, **0.9% of the 905**, verified here by
  intersecting the two page lists. They are very nearly disjoint, so **the
  905-page raster result does not re-confirm the 65-page one**; what exists is
  **a second independent piece of evidence, which is the stronger thing to
  have** and must not be counted as one result confirmed twice, since two
  independent legs survive one of them collapsing. L's own framing:
  *1本を2本と数えないほうが強い*. On the 8 shared pages the row counts match exactly once
  folder rows are accounted for (109/80 against 80, 8/8, and so on), with **0
  disagreements** -- so the join is clean and the coverage is 8.2% of L's
  table. **The rule is the cheap one**: before writing that a result holds in
  a wider population, **intersect the page lists and print the overlap**. This
  is 曝露や母集団を広げるのも棒を動かすこと read backwards -- not widening a population to pass,
  but **describing a sideways move as a widening**, which claims a
  confirmation that was never computed.
- **T61. A fill can be white, and an instrument that counts ink cannot see it.**
  Found 2026-10-23 by the extraction session while fixing the above. Its fill
  signature read `plane0`, which is **255 for a white fill exactly as it is for
  blank paper**, so coverage cannot separate *this area is painted white* from
  *nothing is here*. The independent check adopted is **`preview_dark` >= 0.5
  from `CanvasPreview`**, the composited page, which carries colour rather than
  coverage. Every instrument in this project counts ink, so the general form is
  worth keeping beside the tone trap: **coverage is not colour, and what colour
  an area is cannot be read from a field that only says whether the area was
  touched.**
- **T62. Every track tree carries a frozen copy of this file, and the copy says it is
  the first document to read.** Found 2026-10-23 by Track J during its
  close-out: its own `doc/CURRENT.md` was **1,957 lines, frozen 2026-10-09**,
  against the live file's 5,639 -- **13 days and 3,682 lines behind** -- and a
  fresh session obeying its opening paragraph would take *a Track J that has not
  run its slot*, *a 519 that has not been retracted* and *a verdict word that
  does not know its two forms* as the present. **Swept across all trees the same
  day: ten carry one**, from `solid-fill`'s 13 days behind to
  `cleanup-refiner`, `controlnet-sd15-refine`, `controlnet-sdxl-fidelity` and
  `stroke-selection` at **82 days and 5,267 lines**. **None of them has
  `CLAUDE.md`**, which was written today on `foundation`, so its branch-detection
  section reaches no existing track.
  **Track J's reading of it, which is the day's rule a fourth time**: the name
  promises currency, and a promise is an implication, not a check. **It is the
  worse form of the stale-briefing accident** reported that morning, because a
  briefing does not claim to be current and this file does.
  **Fixed where it propagates**: the claim in `foundation`'s copy is now
  conditional -- it opens by telling the reader to check whether they are inside
  a worktree -- and `CLAUDE.md`'s reading table says the same, so **every track
  cut from here on inherits both**. **And the audit now tests the property**
  rather than the implication (check 7, `frozen_ledger()`): it reports a tree
  whose local copy **still makes the claim**, which is what a fix has to remove,
  so a tree that replaces its copy with a pointer goes quiet and a tree that
  merely warns about it in the briefing does not. **A briefing warning is the
  weaker fix, because it has to be read first.** J fixed its own tree that way
  (`70bb589`) and the audit still reports it, correctly.
  **Track J then removed its copy rather than warning about it** (`168b886`,
  verified against `frozen_ledger()`, which now returns `None` for that tree --
  the audit went from eleven findings to ten). Its replacement is a **19-line
  pointer** offered as a template for the other eight, and the fourth of its
  four points is the one worth copying: **`git merge foundation` conflicts here,
  and the resolution is to take the foundation side** -- written where whoever
  hits the conflict will read it, so that resolving it the other way cannot
  restore the frozen copy. J's own judgement on deletion: all 1,957 lines are
  superseded in the live file, so no fragment was worth keeping.
  **Swept 2026-10-23 by user approval: all ten trees, in one pass.** Each tree's
  `doc/CURRENT.md` is now a **19-line pointer** on J's template, and
  **`CLAUDE.md` is added in the same commit** -- which was the heavier half, by
  Track J's reading: a session that reads a frozen ledger **also has no
  branch-detection rule**, so the two compound, and it takes an old ledger as
  the present *and* fails to establish which tree it is in. **It was ten and
  not eight** (this file's own count was wrong): `solid-fill` was already fixed,
  but `stroke-selection` and the newly cut `fill-region` both carried one, the
  latter current on the day and stale the next. Freeze dates ran **2026-08-02 to
  2026-10-23**, lags **67 to 5,454 lines**. **Eight finished trees were pushed**
  (all had upstreams, so no new remote branch was created); `stroke-selection`
  was pushed after the finding below; `fill-region` is commit-only, because its
  own session pushes its branch (CLAUDE.md §5). Every tree was clean and nobody
  was sitting on any of them. **Not swept**: `lineart-face-words` and
  `lineart-panel-generation`, separate clones sharing one git directory, with a
  live session on the first -- **the branch belongs to whoever is sitting on
  it**, so Track H is told instead.
  **And the sweep immediately made the audit report its own commits as a
  track's unreported work.** `stroke-selection` jumped to *"committed 19d past
  its last notice -- unreported results are likely"*, on the strength of the
  housekeeping commit the foundation had just made in it. **The reporting gap
  asks whether a *track* has done work it has not reported**, so the fix is a
  property test rather than a marker: the gap is now computed from the last
  commit that touches **something other than `doc/CURRENT.md` and `CLAUDE.md`**,
  the two files the foundation maintains in every tree
  (`last_commit(own_work_only=True)`). After it, the gap column reads exactly as
  it did before the sweep. **A check that cannot tell a fix from a symptom will
  report the fix**, and this one was contradicted by the housekeeping done to
  satisfy another of its own findings.
- **T63. Three requests are in flight to the extraction side that the foundation did
  not carry, and that is the part worth fixing (2026-10-24).** Track L wrote
  them into this tree's `outbox/` and delivered them itself, and the user pushed
  the priority one directly:
  **`request_composited_page_20261024.md`** (composited finished pages at native
  resolution -- the standard that has not been through a filter, with four
  questions to answer first, the load-bearing one being whether the preview is
  native), **`request_full_layers_addendum_20261024.md`**, and
  **`request_priority_65pages_20261024.md`** with `priority_pages_20261024.txt`
  -- **65 pages, ako6 31 / ako7 34, 129 koma, about a third of the 178**, chosen
  because *the stalled verification runs on exactly that range*. Its order of
  priority: **the full layer inventory for those 65 pages first, statistics only,
  no rendering** -- one item that moves the work and costs almost nothing. Two
  standing conditions in it are worth keeping: **native canvas, same `pid`
  naming**, and **do not select by page-level cleanliness** -- clean and mixed
  koma sit side by side on one page, so **cutting by page throws away 94% of the
  usable data** and the koma-level selection stays on this side.
  **Going direct is not the problem; the foundation not knowing is.** Fewer hops
  means less distortion, and the track that holds the measurement writes the
  better request. **But the foundation's picture of what is in flight was wrong
  until the user said so.** A mechanical test for "is this recorded" was looked
  for and does not exist: **89 of 112 correspondence files are not named in this
  ledger**, because correspondence is normally summarised rather than cited, so
  "unnamed" cannot mean "unrecorded". **What does work is cheaper**: the
  start-up procedure listed `inbox` only, and a request the foundation did not
  carry lands in **`outbox`**. `CLAUDE.md` now lists both.
- **T64. The composited-page route is dead for area boundaries, answered 2026-10-24:
  `CanvasPreview` is a quarter of native and the native composite is not in the
  file.** 7016x6543 -> **1754x1635**, 4961x7016 -> 1240x1754, **exactly 4.00x**
  (checked here). CSP stores only the reduced version. **This was the
  contingency Track L had named in advance** -- *"if the resolution turns out to
  be reduced, the request changes shape"* -- which is why asking the four
  questions before the work was the right order. Two side answers: **the halftone
  is composited but crushed** (60L at 600dpi is a ~10px period at native, so
  **2.5px at a quarter** and it averages to grey -- a page at 48% tone coverage
  shows 21.6% midtone in the preview); and **the preview looks like visible
  layers only, on one layer's evidence** (`visibility=0` at paper darkness 0.064
  against a median 0.720 over 50 visible layers) -- **direction, n=1, weak.**
  **And building a true native composite would be heavier than the extraction
  work itself**, needing opacity, blend modes, clipping, halftone generation,
  gradient rendering whose direction and shape are undecoded, and folder
  compositing -- **which has already gone wrong once on panel frames**. The
  extraction side's warning is the part to keep: **a mistake there produces a
  plausible-looking image, so visual inspection cannot catch it.**
  **The counter-proposal it made instead is better than what was asked for, and
  it is adopted.** What this side actually wants is *the fill painted inside line
  layers*, so **hand those layers over one by one at native resolution** rather
  than a composited page. The full inventory carries `drawn_ratio`,
  `black_among_drawn` and `preview_dark`, so **this side specifies which layers
  to export**. Its sentence: *for the "counted invisible as a negative" problem,
  **the layers that were dropped are themselves the answer, and the composited
  page is only a proxy for them.*** **And it moves the selection to this side
  where it can be disclosed**, which was the only real gain the composited page
  had offered.
- **T65. "Statistics only, so it is light" was wrong, and the correction is worth
  keeping.** `drawn_ratio` and its siblings **cannot be had without drawing each
  layer**: **about 85 s/page** measured (an ako6 page carries 178 layers, 55 with
  content), so **the 65-page inventory is about 90 minutes**, not minutes. The
  fill extraction had been fast because it drew **only the candidates**.
  Running; output `page_layers_priority65.csv`, **pre-selection, all layers**,
  with `visibility` / `opacity` / `has_mask` / `has_gradation_fill` /
  `tone_enabled` / `tone_density_pct` / `tone_lpi` / `rendered` / `drawn_ratio` /
  `black_among_drawn` / `drawn_gray_mean` / `preview_dark` per layer. **The
  request had called item 1 "statistics only, no rendering" and this file passed
  that on** -- a cost estimate inherited from the wrong operation.
- **T66. "The channel drops messages in both directions" is RETRACTED within the hour:
  nothing was ever lost (2026-10-24).** Verified over SSH from here -- all four
  files sit in `C:/Users/sh1/code/extract_line_and_sketch/inbox/`
  (`request_composited_page` 01:02, `..._addendum` 01:17,
  `request_priority_65pages` + `priority_pages` 01:23). **The delivery route was
  working the whole time.**
  **Two separate things had looked like one.** (1) **The extraction side had no
  procedure for checking its own inbox** -- the day's traffic ran over messages,
  so the habit of looking for files never formed; it noticed only when the author
  asked whether the priority request had arrived, and then **read this tree's
  `outbox/` over SSH before looking in its own inbox**. (2) **The foundation's
  reply went to a second, older destination**, because `CLAUDE.md` recorded the
  procedure and not the path -- that half stands. **The symptoms resembled each
  other and were reported as one cause, unverified.**
  **Its general form is sharper than this file's**: it had **no procedure**, not
  a missing parameter -- and **it used "no message arrived" as evidence that "no
  file arrived", which are different routes, so it is not evidence.** The day's
  five units were numbers attached to the wrong thing; this one is **the silence
  of one channel read as the state of another.**
  **And the foundation acted on that diagnosis without checking the symptom.**
  It opened an SSH session to find the destination path and **did not look for
  the three files while it was there**, then rewrote `CLAUDE.md` on a peer's
  report. The check was one `dir` away in a shell already open. **A peer's
  diagnosis of its own failure is a report, not a measurement** -- the rule this
  file applies to peers' findings, not applied to a peer's self-diagnosis.
  **The file-per-exchange rule stands on its own footing**: the author
  instructed it (*direct exchange is fine, but nothing may be left without a
  record*), and its reason is recording, not delivery.
  **The author's instruction: direct exchange is fine, but nothing may be left
  without a record.** Adopted on both sides: **every exchange leaves a file --
  `inbox/` for what arrives, `outbox/` for what goes out -- and a message is
  only a pointer to the file.** The extraction side's replies now land as files
  (`inbox/reply_composited_page_20261024.md` is the first), and the foundation
  does the same. **This also repairs the gap found an hour earlier from the
  other end**: the start-up procedure now lists `outbox` as well as `inbox`, so a
  file left by anyone is seen even when the message carrying it was lost.
  **And the first file the foundation sent under the new rule could not be
  delivered, for the same class of reason.** `CLAUDE.md` said *"push notices to
  the extraction tool by SCP"* and **recorded no destination path**; the guess
  `/Users/sh1/inbox/` failed on a Windows host. The real one, found by listing
  the remote home, is **`sh1@192.168.1.5:C:/Users/sh1/inbox_from_lineart/`**, now
  written down, and delivery was verified by listing it afterwards. **The
  procedure was recorded and its parameter was not** -- the same shape as the
  address that had moved from `.9` to `.5`. **A procedure without its parameters
  is an instruction to guess.** **And the path found that way was the wrong one
  of two**: the live destination is
  **`C:/Users/sh1/code/extract_line_and_sketch/inbox/`**, while
  `inbox_from_lineart/` had been used once in August -- so **"found it on the
  remote disk" is not "found the one in use"**, and the extraction side has asked
  that everything standardise on the first. It copied the misdirected reply
  across itself.
- **T67. The whole-corpus tone scan finished 2026-10-23: 7,667 pages over four
  groups, three failures** (`CHNKSQLi not found`, corrupt files). The aggregate
  is in Next Actions item 11.
- **T68. The "against the radius-4 estimated mask" caveat has a number now, and the
  number is worse than the caveat implied.** Track L, ako6, this configuration:
  at the region unit the mask recovers **17.8%** of the production fill. Amendment
  J3 required every fill headline to name that mask; **the qualifier was always
  about an unknown, and the unknown has been measured once**. It does not move
  Track J's verdict -- every arm there sat on the same mask and the comparison
  was internal -- but any **absolute** claim about how much fill is recoverable
  now has a figure to answer to.
- **T69. Say this as three files, not as a loss. Track L spent three exchanges failing
  to get it across and traced the failure to its own words "deleted" and
  "discarded".** The wording that lands first time:
  > **There are three different files. (1) the author's manuscript (`.clip`),
  > which has every fill layer in it. (2) the delivered fill layers, which also
  > have them. (3) this project's dataset image, which was **composited from
  > (1)'s line layers only** and is therefore a *different image*, not a copy.
  > The fill is in (1), is in (2), and was never in (3) -- and (3) is the only
  > one this project has ever used.**
  **Nothing was destroyed and nothing is missing from the author's files.** On
  `page0118`, **61.0% of the fill layers' 2,293,008 black pixels are white in the
  line image**. A side-by-side at native resolution is at
  `../lineart-fill-region/results/explain/raster_vs_production_fill.png`.
  **This file's earlier wording would have reproduced the same
  misunderstanding** in the next reader, which is why it is corrected rather
  than supplemented: *"discarded"*, *"drops out"* and *"missing"* all imply an
  act on one object, and the truth is two objects with different contents.
- **T70. This project's line rasters DO NOT CONTAIN a large part of the fill the
  author separated -- two works now, and it reaches further than the mask.** The
  raster lacks **46.4% of the production fill's pixels in ako6 and 40.4% in
  ako7** --
  the same level in both, measured by Track L in one run with ako6's values
  carried as the known cell and **reproducing to four decimals** (`black` keep4
  0.994, `beta` ink-presence 0.1421). **So "one work, not yet a claim" is
  upgraded**: the loss is not an artefact of one manuscript. **Which means Track
  J's "the rough does not mark the fill, 96%" was measured against a raster that
  is itself missing fill**, as was every fill figure built on `fill_mask.py`.
  **This is the single most load-bearing finding in the fill line**, because it
  moves the target rather than the instrument -- and it was reachable only once a
  production standard existed, which is what the inventory bought.
  **What is still not claimed**: the *per-layer* rule. The share of separated
  fill that survives differs 2.7x between the two works (14.2% / 38.1%) and
  **why is unmeasured**. And **ako8 onward is unmeasured** -- the inventory's
  eight-page sample shows `beta`-named mass at 0, but that is a sample with no
  shape term, so **the end of the generation is not known** and saying it needs
  pixels.
- **T71. The separation is progressing, not static, and one step of the author's
  "back and forth" now has a direction.** From ako6 to ako7 the area share
  inverts: `beta`-named **30.8% -> 53.1%**, `line`-named **54.3% -> 37.5%**, and
  the largest single layer name goes from the unnamed `線画` (41.4%) to **`black`
  (53.1%)**. `black` is fill in both (keep4 0.994 / 0.993, median inscribed
  radius 135.2 / 174.4px). **So 2018-2019 is a generation mid-move**, which is
  a claim about **how the author separated the layers** and **not** about how
  much fill the raster keeps -- that second thing turned out to be a width
  threshold in this project's own extractor (Track L's entry).
  why a date cannot stand in for it.
- **T72. An external standard for areas is now within reach, on a distinction Track L
  made after finding a hole in this file's own map (2026-10-23 night).** The map
  said three routes fail for three different reasons -- whole layers mix line and
  area, cutting with radius 4 is circular, and name-selected `beta` layers are
  largely absent from the raster. **Routes 1 and 2 were not independent.** The
  foundation's proposed remedy for route 1 was *"take only the pixels in a layer
  that pass the shape test"*, and **that is precisely route 2's circularity**:
  cut the pixels with `keep4` and the standard's **boundary** is set by the
  radius-4 opening, which is then what evaluates it. **A map of independent
  failures hid a dependency between two of its entries.**
  **The distinction that opens the road: selecting is not cutting.**
  - **Cut pixels** -> the standard's **boundary** comes from the instrument ->
    **circular**.
  - **Select whole layers** -> the boundary stays the author's, and radius-4
    decides only **which layers are admitted**, never their shape -> **not
    circular**. Every pixel of the standard is then one the author drew.
  **Measured, `keep4 >= 0.95` with no pixel cutting:**

  | | layers | line contamination | fill-area coverage |
  |---|---:|---:|---:|
  | ako6 | 316 | **0.6%** | **86.0%** |
  | ako7 | 140 | **1.2%** | **76.7%** |
  | ako6 `beta`-name, non-tone | 59 | 1.4% | 10.3% |
  | ako7 `beta`-name, non-tone | **2** | 0.2% | **1.1%** |

  So route 1's defect -- 13.7% line contamination -- **falls to 0.6-1.2% while
  covering most of the fill**.
  **And Track L withdrew its own recommendation in the same message.** The
  name-selected standard (`beta`-named, non-tone) that its first work had pointed
  to **collapses on ako7: 2 layers, 1.1% coverage**, because as `black` layers
  increase they move **under tone folders**, so the non-tone `beta`-named set
  shrinks. **The second work killed what the first suggested** -- the second time
  that day a structure seen in ako6 had been written as a rule.
  **The limit is stated with the result, not after it.** The layers `keep4`
  excludes are the ones **fused with lines**, which is **exactly where the mask
  is weakest**, so **agreement against this standard is biased in the mask's
  favour**. The bias is bounded by what is left out: **14.0% of the fill area in
  ako6, 23.3% in ako7**, and Track L has put those two numbers on the same line
  as any agreement figure it reports.
  **Measured 2026-10-24 as L7, and then RETRACTED the same day on a hole the
  user found.** The numbers are kept here because the retraction is the content.
  Truth was taken to be the author's fill; the main value was `precision`, so
  `1 - precision` was read as the false-positive rate:

  | stratum | koma | precision | **FP rate** | recall | mask area | true fill area |
  |---|---:|---:|---:|---:|---:|---:|
  | **CLEAN** | 35 | 0.2868 | **71.3%** | 0.0226 | 1.99% | 25.18% |
  | LEAKY | 94 | 0.7039 | 29.6% | 0.9235 | 13.91% | 10.60% |

  **The hole, in the user's words**: *"if you run the radius-4 method on clean
  line-only information, the image has no fill in it to find -- you haven't made
  that blunder, have you?"* **Half of it was that blunder and the other half was
  worse.**
  **L6's "clean koma" means the *delivered* fill layers are in dedicated layers
  and do not leak into the raster. It does not mean the raster has no fill.**
  The delivery contains only layers that passed a **colour** test
  (`black_among_drawn >= 0.95` and the rest). **If the author painted fill inside
  an anti-aliased line layer, that layer fails the test and never enters the
  delivery at all** -- not even as a "mixed" layer, so `share_mixed` could not
  see it in principle. **So the raster holds real fill that the ground truth
  knows nothing about: the mask found it correctly and it was counted as an
  error.**
  **The evidence is in the blob sizes.** Of the 2,194,338 px called false
  positive, **54.2% sits in components of 5,000px or more** (27.7% at 5k-20k,
  26.5% above 20k), and **a radius-4 opening cannot leave a 5,000px blob behind
  from thick strokes** -- so at least that share is the standard's absence, not
  the mask's error. A native 1:1 crop shows the mask outputting a hair mass and
  a window frame with **11 pixels (0.0%)** corresponding in the delivery
  (`results/explain/fp_koma_01704_...png`).
  **Withdrawn: the 71.3%, the CLEAN/LEAKY precision contrast, and the sentence
  this file had called a deeper question than L1's** -- *"the mask's accuracy is
  a property of its input, not of the mask"*. The contrast cannot separate *fill
  leaked into the raster* from *fill the delivery never captured*, and the
  sentence rested on the contrast. Identity still holds (`truth_px` agrees with
  L6's `clean_px` on all 129 koma); it was never the arithmetic.
  **The recall column is NOT a finding, and Track L found that itself before
  reporting it.** `mask` is a subset of `ink`, so `mask ∩ truth` is a subset of
  `ink ∩ truth` and **`recall <= leak` by construction**; CLEAN's pixel-weighted
  leak is **0.0554**, so a recall of 0.0226 being small is **half-automatic --
  the stratum's own definition caps it.** What is informative is only what sits
  inside that cap: **of the 2,161,246 reachable pixels the mask took 882,485 =
  40.8%**. **`precision` carries no such bound**, which is why **71.3% is the
  result and the small recall is not.** The day's rule applied to one's own
  output, caught by walking the construction after seeing the number.
  **What survives is what rests on the delivered layers themselves, and all of
  it is a LOWER BOUND** -- fill outside the delivery can only add. **Two of this
  file's inherited null values move, in that direction.** On CLEAN:
  **`all_black`'s IoU 0.0939 / 0.1201 -> at least 0.2518**, and **`blank`'s
  region-F1 0.0084 -> 0.0000**, because every koma in that stratum has area in it
  and "paint nothing" cannot score. **The trivial-arm bar more
  than doubles**, so every fill verdict has to be re-read on this stratum before
  it can be quoted against clean data.
  **And "1.2 regions per koma" is gone.** The author's fill has **at least median
  3.0 connected components per koma** (mean 3.9, max 16, >=500px, stable at 3.0
  across both strata), against the ledger's *"this corpus's line art encloses
  almost nothing -- about 1.2 regions per panel"*, which came from the inferred
  mask. **So "there are hardly any areas, so a region proposer is a poor bet"
  does not hold on the real thing** -- and that reading was the basis of Track
  L's own unit constraint, so **the candidate ordering can move.**
  **What this does not do, stated by Track L rather than asked of it**: Track J's
  **arm-to-arm comparisons remain valid**, every arm having sat on the same
  ground truth, so the ordering is intact. What broke is the **absolute claims,
  the trivial arms' null values, and the region unit** -- three things, and **not
  J's verdict.**
- **T73. The user redirected the fill line on 2026-10-24, and the reason is not
  derivable from anything here.** Their words: **research should proceed first
  on data where line and fill are cleanly separated; measuring mixed data as
  mixed is low priority** -- *because what was measured cannot be stated*. And:
  **if Track J's assets are to be validated, validate them on the cleanly
  separated data.** So Track L's own recommendation -- re-measure door 2 (the
  rough does not mark the fill) against real ground truth -- **was de-prioritised
  by the user**, and the agreement run above was overtaken rather than refused.
- **T74. How much cleanly separated material exists: 35 koma, and the distribution is
  what matters (Track L's L6, pre-registered at mixed-layer contribution < 0.10
  and leak < 0.10).** ako6: 201 koma, 56 separated, 99 leak-free, **12 both**;
  ako7: 136, 52, 67, **23 both**. **Total 35 koma over 28 pages, 38,982,636 true
  fill pixels, median fill area per koma 19.0%.**
  **The decile histograms are two spikes with a hollow middle** -- mixed-layer
  share `[108, 32, 13, 19, 21, 14, 9, 5, 6, 110]`, dedicated-layer leak
  `[69, 56, 7, 7, 7, 2, 5, 2, 3, 82]`. **A koma is either cleanly separated or
  completely mixed, so the drawing method switches per koma**, and **the count
  therefore does not depend on where the threshold sits** (78 at 0.20/0.20, 88 at
  0.30/0.30). **So the generation wobbles at koma granularity, not only per
  work** -- which is a finer grain than the author's "it goes back and forth" had
  been written at here.
  **Scale, stated as the limit it is**: 35 koma is enough to measure and **not
  enough to train**; 39M pixels make pixel-level quantities precise; and door 2's
  real re-measurement **cannot run on 28 pages** when the original was 167+73
  koma on a different population. **The growth path is plausible and
  unmeasured**: ako6/ako7 give 35 of 337 koma, about 10%, and the bimodality
  suggests other works hold clean clusters too -- but **how many works keep fill
  in separate layers at all has not been measured**, so that 10% extrapolates to
  nothing. A request to the extraction side is the user's call and **has not been
  made**.
- **T75. Track L's L7, running 2026-10-24: validate Track J's assets on the clean
  data, with the false-positive rate, and with a stratum that can move the blame
  off the mask.** Truth is **the author's fill, not the inferred mask**. Main
  value **`precision = |mask ∩ truth| / |mask|`, so 1 - precision is the
  false-positive rate**. **The design's sharp part**: `CLEAN` (leak < 0.10) and
  `LEAKY` (leak >= 0.50) go in the **same table**, because **if precision is
  determined by the stratum, then the mask's accuracy is a property of whether
  fill leaked into its input and not a property of the mask**. Also on real
  truth: the trivial arms (`all_black`'s IoU is the true fill area share,
  `blank`'s region-F1) and **the connected-component count of the author's
  fill** -- **answered 2026-10-24: median 3.0 components per koma on the author's
  fill, against** Track J's *"this corpus's line art encloses almost nothing, about 1.2
  regions per koma"* came from the inferred mask and is being retaken against the
  real thing.
- **T76. `git add -A` put the author's artwork into a public repository, and a
  force-push does not take it back out.** 2026-10-23, the foundation's own
  error, start to finish because the whole shape is reusable.
  **What happened**: the extraction deliveries arrive by SCP into `data/`, a
  directory created that day and **not added to `.gitignore`** -- while this
  project's convention keeps data out of the repository (`dataset/*` and
  `results/**` are both ignored). Two `git add -A` commits carried **1,018 files,
  about 99MB**, and both were pushed to `origin/foundation` at
  `github.com/kuromook/gen_lineart`, **which was public**. Contents: **884
  binary fill masks at 7016x6543 covering 107 pages** (black = fill, no
  gradations, so the composition is legible and the drawing is not), and **134
  part-layer files including five `_line.jpg` page composites at full resolution
  with 256 grey levels** -- those five are the artwork itself.
  **Why removal is not what a force-push does.** A commit is a content-addressed
  object; a branch is a pointer. The force-push moved pointers, and **the
  protocol has no "delete object" operation** -- a client can only say which refs
  exist. Pruning unreferenced objects is GC, **GitHub does not expose GC to
  users, and it serves `git fetch <sha>` for unreachable objects**. Measured: a
  fresh bare repo fetched `--depth=1` by SHA pulled **285MB** and read a named
  PNG at its exact byte size. **So after the force-push the data was still
  retrievable by anyone.**
  **And the SHAs were not a secret: this file published them.** The commit
  message recording the mistake named both offending commits, and so did the
  trap entry below. **Documenting the error created the path to the data.** A
  retraction has to ask whether its own text is an index to the thing being
  retracted.
  **The verification trap, hit twice before being noticed.** `git fetch origin
  <sha>` **succeeds as a no-op when the object is already in the local
  repository** -- and ours was, through the backup refs. Both times the first
  reading was wrong in the safe-looking direction once and the alarming
  direction once. **The only valid test is a fresh bare repo holding no objects**,
  and the final one must be anonymous, since a probe over SSH shows only what a
  key-holder sees.
  **The fix that worked, and why each step was that step.** (1) `data/` into
  `.gitignore` and `git rm -r --cached`, so it stops growing -- the tip is clean
  but history is not. (2) **Private first**, which is immediate and certain while
  the rest is arranged. (3) `git filter-branch --index-filter` over
  `7d7087d^..` for **both affected branches in one invocation**, because
  rewriting them separately would give the shared commits two different new SHAs
  and leave a duplicated history. (4) **Delete the repository and recreate it**
  -- the only self-service way to destroy the objects, **viable only because
  forks and network count were 0**; with a fork, the objects move into the fork
  network and survive. Stars, watchers and issues were also 0, so nothing was
  lost but the URL's continuity. (5) **An explicit refspec, never
  `push --mirror`**: mirror pushes every ref, and the rollback tags were the one
  remaining path to the data, so a mirror would have re-uploaded it. `foundation`
  went first, which is what sets the default branch. (6) The rollback refs were
  kept until the **anonymous** probe passed, then deleted and `git gc
  --prune=now` run. **Verified after**: `fsck` clean, all 21 branches matching
  origin, both objects pruned locally, and `.git` fell from **6.8G to 207M**,
  most of it unrelated unreachable history that had been accumulating.
  **The general lesson is not "be careful with `git add -A`".** It is that
  **this repository had a convention against committing data and the convention
  lived only in `.gitignore` entries for the directories that existed when it
  was written.** A new directory inherited nothing. **A rule expressed as a list
  of instances does not cover the next instance** -- which is why `data/` is now
  ignored as a directory rather than by what sits under it.
- **T77. A quoted commit hash can stop resolving, and then it is a number that looks
  like provenance and leads nowhere.** Found 2026-10-24 by running this file's own
  muddle procedure (discipline 3's "walk each quoted number to its source") over
  the ledger's hashes: **34 quoted, 4 dead** -- three of them the very commits the
  history rewrite destroyed, named by the entry *about* that rewrite, and one a
  Track L briefing hash that the same rewrite renumbered (`30c5104` ->
  `7d9f080`). **The entries were right and their citations had rotted under
  them.**
  **Now checked mechanically** (`audit_track_ledger.py` check 7,
  `dead_hashes()`): every `` `hash` `` in the ledger must resolve, and the four
  known-dead ones are declared in `DEAD_HASHES_OK` **with a reason** -- so a
  deliberate dead reference is a one-line declaration and an accidental one is a
  finding. **Near-zero false positives, and it caught something the day it was
  written.**
  **The general form**: a rewrite, a dropped branch or a typo breaks a reference
  **without touching the sentence around it**, so the prose keeps asserting
  traceability that no longer exists. **This is the fifth member of the
  implication family** -- after a name, a key, a tag and a recorded address --
  and the same rule closes it: **test the property, do not read the record.**
- **T78. If an external standard arrives through a filter, what the filter dropped is
  not absent -- it is invisible. Counting invisible as negative turns an
  instrument's correct firing into an error.** Track L's wording, 2026-10-24,
  after the user asked whether its clean stratum simply had no fill in it to
  find. **It did have fill**: the ground truth was the *delivered* layers, the
  delivery admitted only layers passing a colour test, and fill painted inside
  an anti-aliased line layer fails that test and **never appears in the delivery
  at all** -- so the instrument that found it was marked wrong. The measured
  consequence: 54.2% of the "false positives" sat in components of 5,000px or
  more, which a radius-4 opening cannot produce from strokes.
  **This is the sharpest form of the week's recurring failure.** The others were
  a number attached to the wrong unit; this one is **a negative class built out
  of a gap in the standard's own intake**. The check it implies: **before
  counting anything as a negative, ask what the standard's acquisition path
  cannot admit**, and treat every figure resting on it as a bound in the
  direction of that gap.
  **It also refutes a framing this file had written and praised twice** -- *"they
  test colour, we test shape, neither was built from the other, so agreement is
  informative"*. **The axes are not independent: the colour test is the entrance
  to the shape test.** Two instruments are only two witnesses when neither sits
  upstream of the other's intake.
  **And the same shape appeared on the network the same day.** The extraction
  host's recorded address, `.9`, returned `No route to host` with an `INCOMPLETE`
  ARP entry, and Track L was about to conclude the machine was down; it had moved
  to `.5`. **A recorded address is an implication too** -- the same class as a
  name, a key or a tag, and the same rule applies: test the property, do not read
  the record.
- **T79. A combined `git log -- pathA pathB` cannot tell you which commit added which
  path, and its ordering invites you to think it can.** 2026-10-23: having
  pushed two manuscript deliveries into the repository by `git add -A`, the
  foundation ran `git log --diff-filter=A -- <ako6 manifest> <part_layers>` to
  find out which commit was which, got two commits back, and **read the output
  order as the mapping**. The output is ordered by date. Checked per path by
  Track L and then here: **`7d7087d` added the ako6 delivery (884 files, ~75MB)
  and `51623d3` added the part-layer export (134 files, ~23MB)** -- the exact
  opposite of what `f38a601`'s commit message states, and of what was reported
  to the user and to Track L. **The count was right (two commits, ~99MB) and the
  correspondence was wrong**, which is the day's recurring failure on a new
  unit: commit against path. **The consequence was material**: `fill-region`
  descends from `7d7087d` and not `51623d3`, so what Track L's branch carries is
  the **75MB ako6 delivery, 3.2x the figure the foundation gave it**.
  `f38a601`'s message is left standing with this entry as its correction, since a
  commit message is a store. **(Those three SHAs -- `f38a601`, `51623d3`,
  `7d7087d` -- no longer resolve: the history rewrite that removed the data
  rewrote them, so the messages survive under new hashes and these numbers are
  dead references kept because the entries name them. Verified 2026-10-24.)** **The check: one path per query, or read the
  `--name-only` output rather than the commit order.**
  **Second instance within the hour, same shape, found by Track L.** The
  foundation made backup tags before the rewrite and then **kept working**, so
  both tags went stale -- `backup/fill-region-pre-data-rewrite` was one commit
  behind, and **the missing commit was L's hold record itself**, which replaces
  the briefing's *"this tree pushes its own branch"* with a hold. **Losing that
  one commit would have put a fresh session back on the instruction to push**,
  restoring the exposure the rewrite exists to remove. The foundation's own tag
  was also a commit behind. Both re-pointed; nothing was lost, since the old
  targets are ancestors. **L's framing, which names what the two instances
  share: a snapshot read in place of a moving ref.** Reading output order as a
  mapping and reading a tag as a branch are the same act -- **inferring the
  correspondence of two things from their ordering or their name instead of
  asking each one.** **Two provisions follow**: a backup ref taken before a
  rewrite is only valid if nothing commits after it, so **it is taken at the
  moment of the rewrite and not in advance**; and **verification compares
  against the live ref, never the tag**, because a change made after the tag
  would otherwise pass as something that never existed rather than as a
  difference.
- **T80. The koma corpus holds the same page more than once, at different drawing
  stages, and a naive check reads that as duplication.** Found 2026-10-23 while
  answering the extraction side's question about crop geometry.
  `dataset/regions_clip_pairs_v3_koma_panels_20260826` holds **5,018 panels over
  1,241 page records but only 1,132 distinct page geometries**: **95 geometries
  exist in more than one version, covering 667 panels (13.3%)**, paired across
  source folders (`ako6/page0051` <-> `done/ako6-3`, `pink/page0010` <->
  `pink/page0028` <-> `done1/pink10`). **No group straddles train and held, so
  there is no split leakage**; inside Track J's 969-panel held set, **9 groups /
  69 panels (7.1%) are alternative versions, all in `2017_pink`**.
  **The methodological half matters more than the counts.** The first pass
  matched pages by panel-geometry signature and reported *339 duplicate panels,
  46 of them in the held set*. A pixel check refuted it: **only 1 of 31 sampled
  groups was pixel-identical.** Same crop dimensions on every panel, different
  content -- the same drawing at different stages. The pink pairs differ in a
  median of **0.39% of pixels** (ink IoU median **0.886**), which is what proves
  they are one drawing rather than two pages, but the tail is wide: ink IoU p10
  **0.387**, min **0.031**, and `ako6/page0051_p1` carries ink **0.1907** where
  `done/ako6-3_p1` carries **0.0067** -- one version has the fill, the other is
  nearly blank. **A geometric signature identifies a page; it does not identify
  a state.** Reporting the first result would have been this file's own
  failure mode twice over -- a count published from a proxy that was never
  checked against the content. **The rule is not "do not use a proxy", which is
  what the first wording here invited.** Narrowing with a cheap signature and
  then sampling is the right default; what went wrong is that the sample's
  disagreement was treated as noise. And Track J, which did hash all 5,018
  panels, states plainly that it did not get there by judgement -- **it went
  looking for an explanation of why its grouping found 129 groups where this
  file found 95, picked the first pair it expected to be a false positive, and
  the false positive was real.** So the count mismatch is what summoned the
  direct test. J's wording, adopted in place of the foundation's:
  **when a proxy has been used to narrow, do not close until the side the proxy
  excluded is explained.**
  **Why it is load-bearing for Next Actions item 11**: comparing the radius-4
  mask against a production ベタ layer on the *unfinished* version of a page
  reads as *"the mask misses every area"* when the truth is *"that version is
  not drawn yet"*, and **`ako6`, the chosen trial work, has 150 of its 227
  panels (66%) inside a multi-version group**. **It would have been the week's
  fourth false positive, and reported on the very inventory the retracted 519
  came from** (Track J's framing). So **page identity is fixed by coordinates,
  not inferred from shape**.
  **The identifier was there; the relation was not.** The manifest's
  `content_fingerprint` differs between versions (`page0051` and `done/ako6-3`
  carry different fingerprints; 63 pids, one fingerprint each, **none shared
  across pids**), so this corpus never confused two states -- what was missing
  was any record that two pids are **one page at two stages**. And the crop is
  recoverable without asking anyone: `manifest.csv` carries `work_id`, `slug`,
  `page`, `pid` and **`x0,y0,x1,y1` in page coordinates** (`ako6` tops out at
  x1=6491 / y1=6541 inside the stated 7016x6543 canvas, and bbox size equalled
  panel size on 40 sampled rows with no mismatch), so **the page frames asked of
  the extraction side were not needed and the request was withdrawn**. One
  open step is closed: Track J traced `scale`/`dx`/`dy` to
  `tools/pair_extraction/match_koma_panels.py:search_panel_alignment()`, a
  per-panel joint translation-and-scale search added with the koma pipeline in
  `acc4138` on **2026-07-26** -- so the guess that they came out of that month's
  alignment investigation was right. **In `materialize_koma_panels.py` the line
  panel is a plain `crop_page(line_gray, x0, y0, x1, y1)` that never touches the
  three columns**, while the rough panel is scaled about the panel centre and
  then offset. So `scale != 1.0` means **the rough page was drawn at a different
  scale than the line page**, and the correction lands only on the rough -- which
  is why 153 of 227 rows are 1.0. **Consequence for item 11: matching a
  production ベタ layer against the line page is unaffected by those columns and
  needs no overlay check first.** The caveat applies only if the rough is ever
  the comparison target, and there it compounds: the alignment only maximises
  per-panel edge agreement, and Track D measured on 2026-10-19 that **the pair is
  loose in both directions -- about half of the rough's ink has no GT line within
  8px** -- so "the mask missed the area" read against a rough inherits that
  looseness on top of the version problem.
  **Track J measured it from the content side the same evening and the reading
  above is both corrected and strengthened** (`98686ae`, its
  `tools/evaluation/audit_koma_duplicates.py` with the numbers in `--self-test`).
  Reproducing the geometry grouping gave J *more* groups than this file found
  (129 / 989), because a shared canvas with a single full-page panel collects
  unrelated works -- and the first such pair J checked expecting coincidence,
  `044_2021_nanami` against `045_2021_nurse`, came back **pixel difference
  0.0000, ink IoU 1.000**: a **byte-identical page across two different
  `work_id`s and two different families.** So J defined duplication on content
  instead -- **line and rough both byte-identical** -- and hashed all 5,018
  panels: **201 duplicate groups, 412 panels (8.2%), 3 of them crossing
  families, 8 groups / 19 panels closed inside the held set (2.0%, not 7.1%)**,
  and **0 groups straddling train and held**, confirming the leakage conclusion
  on a test of content rather than shape. **Both numbers are real and they count
  different things**: 69 is how many held panels are *another version of a page
  already in the set*, 19 is how many are *the same image twice*, and only the
  second bears on leakage.
  **The verdict was recomputed rather than argued about.** Keeping one panel per
  duplicate group takes the normal layer from 956 to 946: `blank` 0.0084 ->
  0.0085, best non-learned 0.0110 -> 0.0111, M0/M1/NC4 unchanged, C1's margin
  over `blank` +0.0026 either way, C2 legs -0.0012 / -0.0017 -> -0.0012 /
  -0.0018, C3's gap -0.0093 -> -0.0094. **Nothing moves before the fourth
  decimal**, and C4 is structurally unaffected since each control sits on the
  same panels as the arm it excludes. **So it does not join the headline**, and
  J drew the distinction this file should keep: the radius-4 mask is **an
  unchecked estimate of unknown direction**, while this is **an effect measured
  and bounded at the fourth decimal** -- J3's condition cuts the other way here,
  and a clean negative must not be softened by a caveat that has been measured
  not to touch it. The 10.40% figure travels with "2.0% of this population is an
  exact duplicate".
  **And the two audits whose names cover this did not cover it.**
  `audit_split_leakage.py` and `audit_pair_dataset_integrity.py` both operate on
  `dataset/pairs_480` and **have never looked at the koma dataset at all**; J
  states it had not known about the duplication and that the ledger should say
  **unchecked**, not *already reported*. **Third instance that day of a name read
  as a measurement** -- after `NC4` and the extraction side's `black` -- and this
  **Fourth instance an hour later, and the new entry caused it.** Track L's
  ledger entry opens *"active, opened 2026-10-23"* and goes on to describe the
  proposal's **"eight closed doors"** -- and the status reader, scanning the
  first 900 characters for `closed|dormant|paused`, printed Track L as
  **`(finished)`** on the day it was created. Fixed by narrowing **where the
  check is allowed to look**: `entry_is_finished()` reads only the entry's
  opening (260 characters) and lets an explicit `active` win over a stopped word
  in the same opening, so *"active, idle"* stays active; three cases added to
  `--self-test`, now eleven. **Three of the four instances are this one script**,
  which is the price of the one check that depends on nobody's attention: it
  reads prose, and prose uses status words for other things.
  one is the dangerous sub-case, because the name belonged to a *check*, so its
  existence was read as coverage. **An inventory named wrongly gets caught when
  its counts stop adding up; a check named wrongly never does, because a check
  that reports nothing is a check that passed.** The general rule, from J:
  **a key, a name or a class carries an implication, and an implication is not a
  check -- test the property directly.** It has lesson 14's reach, and Track J
  registered both this and the check-naming sentence in its own tree the same
  evening, which is the store list working as intended for once.
  **Why the split held, stated as a mechanism rather than as reassurance.** Of
  the 35 duplicate groups spanning different `work_id`s, **11 would have landed
  on both sides had the split been keyed on `work_id` instead of `family`**
  (`051_2022b_4th` vs `059_2023b_4th`, `065_2024_ako3B` vs `067_2024b_ako3B`);
  stripping trailing digits and merging `ako` swallows them. **So it is a
  consequence of the design and not an accident** -- but the family key was
  registered for *character identity*, not duplicate pages, and the two hazards
  only coincide because both take the form "the same work appears under another
  id". **It is also incomplete**: `nanami` against `nurse` is byte-identical
  *across* families, invisible to the key, and it landed train/train. **That part
  was luck.** Hence the direct check: **do not rely on what a key implies, test
  the property.**
  **For item 11, J's addition**: the comparison target must be fixed at the start
  as *"that version of that page"* and never *"that page"*.
- **T81. Every solid-fill number on record inherits a mask that has never been
  checked against anything.** Stated by Track J 2026-10-23, of its own corpus:
  `fill_mask.py` infers the fill from a *flattened* line-art raster by distance
  transform at radius 4, and **`IC1` through `IC7` all take that mask as given**
  and check the metrics built on top of it. So the one core the whole track rests
  on is the one instrument no instrument check has been aimed at -- not through
  negligence but because **there was nothing to check it against**. The
  manuscripts' production ベタ layer would be the first external standard it has
  ever had (Next Actions item 11), which is why J asks not only for agreement but
  for **the direction of the error**: a mask that is dilated and a mask that is
  deficient send every existing figure the opposite way when re-read. Until then,
  a fill figure is quotable only as *"against the radius-4 inferred mask"*, the
  same way a diffusion f1 is quotable only beside its conditioning map.
  **The evening's correction removed the date this caveat was going to end on.**
  The standard was expected to be the extraction side's ベタ layers; after the
  `black` correction the supply is **299 layers over 55 works** rather than
  1,246 over 84, and the filtering that produced it is **name-based and has
  already been wrong once**. Track J's reading, adopted: the qualifier is now a
  caveat **with no end date** -- not a larger one, since no number moved and the
  comparison was always on one shared mask, but one that can no longer be
  written as *"until item 11 brings the production layers in"*. Getting an
  external standard is a **smaller and different errand** than it looked this
  morning, and item 11 should be registered as if it has none.
- **T82. A multi-day run's completion procedure must not be addressed to the session
  that launched it.** Track J's handoff note of 2026-10-19 opens its checklist
  with "read `slot_20261019.log` and both jsonls **in the session it finishes
  in**". The run was deliberately detached at PPID 1 so that nothing depended on
  a session staying open; it finished 2026-10-19 23:43:35, and the session that
  wrote the instruction had exited around 14:00 the same day. **The two
  requirements are incompatible by construction**: detaching is precisely what
  makes the launching session's lifetime irrelevant, so an instruction addressed
  to that session cannot be carried out. This file had already recorded that day
  that "the slot will finish with nobody watching", so the gap was foreseen and
  only the wording was never fixed. The cost: a four-day GPU window spent and
  its verdict still unread four days later. **The rule: the completion procedure
  belongs to the tree** -- the briefing, or a file the briefing names --
  addressed to whoever arrives next, never to "this session". Sibling of the
  `"w"`-mode jsonl trap from the same note: a multi-day run's recoverability has
  to survive the session that started it, and that covers its instructions as
  well as its weights and its log.
  **Second instance 2026-10-23, which settles the form.** **Detaching achieves
  "independent of the session" and "unable to tell the session" at the same
  time.** `PPID=1` is the right choice for a multi-day slot, so this is not an
  objection to it but a demand for the matching provision, in three stages --
  Track J's wording, and J had the first two in place at 10:46, before the
  foundation raised the question:
  **Third instance, 2026-10-23 evening, and it lands on stage (2).** On the
  extraction machine, **memory pressure killed the completion watcher twice
  while all three real jobs survived** (`ako6`'s fill extraction, the saturation
  pass, the final tone group) -- three concurrent runs on one box. **So the
  harness-tracked watcher is itself a single point of failure, and under memory
  pressure it is the *first* thing to go** -- the opposite of the assumption
  that the watcher outlives the run. **Why it went first is not measured**: the
  extraction side corrected this file's first wording, which said the watcher
  was the cheapest thing to reap. The harness notice says only that the session
  was **idle** when memory got tight and says nothing about cost, so "cheap
  therefore reaped" is an inference and "idle therefore reaped" is the other
  candidate. The provision is the same under both; the mechanism is open.
  **Stage (3), the independent second route, is what covered it, and it was
  exercised rather than theorised**: the foundation asked for status over the
  session channel and got the page counts back. That is the first time this
  file's three-stage provision has been tested end to end, and the stage that
  earned its place is the one added last.
  **(1)** detach the run. **(2)** In the same sitting, stand up a *separate*
  harness-tracked watcher -- not the run itself, which would throw away the
  point of detaching. J's is a `run_in_background` until-loop over the log
  (pid 842759 under its session, verified here), and it greps **the failure
  signatures as well as the success one**, because a watcher that waits only for
  success cannot tell a crash from work in progress. **(3)** The watcher is
  itself a single point of failure and has no watcher of its own, so **a verdict
  from a multi-day slot gets a second, independent path** -- another session.
  The costs are asymmetric: one message against a verdict left for another two
  days. **Stage 3 was the foundation's addition and J considered two stages
  sufficient**, which is recorded because the provenance is the evidence that
  neither side had the whole form.
  **What stage 3 needs in order to work**: this project's start-up procedure
  (`ls -t inbox | head`) catches a notice that **arrived**, not one that failed
  to, and the audit's notice-gap cannot fire for a track that committed and
  reported the same day. So the second path needs the **expected completion time
  written down** where a later session reads it -- which is why the header above
  carries one.
  **The foundation's own error here belongs in the entry.** It asserted that
  nothing would wake Track J, from the launch form of the scoring process alone,
  **without checking whether a second process existed** -- the same shape as the
  trap two entries up: reading one of two adjacent records.
- **T83. Two files can each delegate an instrument to the other, and nothing notices
  until a verdict needs it.** Found by the foundation 2026-10-23 while reading
  Track J's tree to see what reading the slot would take. `NC4` (the context
  shuffle, M1's control) is required by J's pass criterion C2 -- *M1 - M0 >= 0.03
  **and** M1's NC4 dropping >= 0.03, both or the context stream is not what
  produced the difference*. `tools/evaluation/score_fill_koma.py:29` says NC4
  "belongs to M1; **it lives in the trainer**"; `scripts/train_fill_seg.py`'s
  docstring says it "**lives in the scorer, not here**". **It exists in neither**,
  so the registered verdict of a four-day run was unreadable from the moment it
  was registered. Each file is individually coherent -- the comment names the
  right owner, and both authors believed the instrument was somebody's -- which
  is why no review of either file finds it. **And why grepping did not catch it:
  the name was already taken.** In stage 1 `NC4` *was* the 50px shift and *was*
  implemented (`nc4_shift50`, `score_fill_baselines.py`); stage 2 renumbered that
  control to `NC5` and reassigned `NC4` to the context shuffle. Asking "is NC4
  covered" returned working code -- for the old meaning. **The renumbering is the
  first layer and the mutual delegation only the second**, so the mechanical check
  is **a control name whose meaning changed between stages**, not a name missing
  from code. **Swept 2026-10-23 over Tracks J, K and G, 33 registered names: this
  is the only instance** (of the other six flags, three were names labelling the
  block beneath them, one a registration explicitly declined, one a control Track
  K ran and read out whose label lives in its log rather than its code, and one a
  delegation whose target does implement it). So the convention is sound and this
  is one incident with a mechanism. Why this one: the thirty `IC-*` checks are run
  immediately, so a missing one surfaces in hours, while **a pass criterion is read
  once, at the end** -- so look there. Track H is safe for the matching reason, its
  C1/C2 being anchored to figures it has already produced and re-reads every cycle.
  The registration-time rule is now discipline 2's second paragraph.
  **The form both of 2026-10-23's failures share, Track J's reading:** each was
  *reading one of two adjacent records*. `NC4` needed two files crossed, and
  either alone is coherent; the foundation's retracted enclosure claim sat in
  section 2 of a work-log entry whose section 3, two screens down, measures it.
  Stated so as to be actionable rather than as "read more carefully":
  **before quoting a passage that supports a claim, search the same document for
  a passage that measured it.** J's own countermeasure, for the other direction:
  a refuted framing belongs in the briefing's current-state section *with its
  measurements*, not only deep in the work log, so a fresh session cannot pick
  it up at the entrance.
  **A sibling, added by Track J the same evening, after the extraction side's
  `black` correction.** Both failures are the same act -- **a name read as a
  measurement**. J assumed `black_among_drawn == 1.00` meant *a fill layer*; the
  extraction side assumed `black` meant *a black area*. Neither was checked
  against the content until something forced it, and in J's case the thing that
  forced it was being asked to supply the number as an external standard. The
  rule, J's wording, adopted project-wide:

  > **An inventory counted by name is not an inventory of what the name refers
  > to. It may not be quoted as a standard until the signature has been checked
  > on the content side.**

  **The exposure is project-wide**, the same way lesson 14's is: anywhere a
  figure here rests on names. Swept 2026-10-23 -- the live instances are the
  manuscript inventory's layer-count line (marked at the point of citation in
  Next Actions item 11) and the retracted 519, which was this failure's first
  instance and is already withdrawn in four stores. **The counts that survive
  the sweep are the ones where a signature was checked**: the 444 tone layers
  actually in use, the 70 of 1,012 `black`, the 5,480 furniture layers dropped
  by name *after* the pixel statistic had failed to separate them.
