# shortcuts

Known limits, in the form the foundation's fourth notice asked for:
`shortcut: <limit>, <when it is raised>`. One file so a limit cannot hide in
a docstring. Added to as tools are written; nothing is removed, only marked
raised with the date.

## stage 2 (fill segmentation on koma panels)

- `koma_windows.py` — shortcut: a panel smaller than 480px in a dimension is
  padded with background white rather than being scored at its own scale,
  raised when a panel set with many sub-480 panels is used (clip_pairs_v3's
  median min side is 1235px, so this affects few panels; the count is printed
  by every run).
- `koma_windows.py` — shortcut: overlapping predictions are averaged and then
  thresholded at 0.5, raised if a run shows the seam lines in the stitched
  output, which the IC4 sheet would make visible.
- `region_metrics.py` — shortcut: the matcher is greedy in descending IoU, not
  optimal (Hungarian), raised if any arm's verdict sits within 0.01 region-F1
  of a threshold, since greedy and optimal can differ only when two GT regions
  compete for one prediction.

## stage 1 / chainer (carried over)

- `stroke_chain.py` — shortcut: `MERGE_PX = 6.0` absorbs junction stubs up to
  about twice the stroke width, raised if a material with a different stroke
  width is used, since the stub's length scales with the width.
- `stroke_curves.py` — shortcut: `np.polyfit` emits RankWarning on
  near-collinear short segments; the class is decided by `line_rms` there, so
  it is noted rather than silenced. Raised if the piecewise fit is ever
  promoted out of "exploratory".

- **Two junction clusters with a 1px bridge are never merged.** `fragments()`
  records one node per fragment end (`max(sorted(hit), key=hit.get)`), so a
  1px fragment lying between two junction clusters has *both* ends assigned to
  the dominant cluster. `merge_nodes` then sees `n0 == n1`, cannot union, and
  the arms on either side of the crossing sit at nodes no matching can reach
  across. Measured on IC-j7 case 1b: a straight crossing at -55 deg splits
  into node 1 and node 2 with a zero-length fragment between them, and the two
  halves of one straight line cannot be rejoined. The same crossing at -70 deg
  gives one node and rejoins.
  *shortcut: a crossing whose skeleton yields two clusters bridged only by a
  fragment that touches both is unjoinable; raised when `merge_nodes` is
  rewritten to union every cluster a short fragment touches, rather than the
  two its ends were assigned to.*

- **IC-j6 measures crossings only. Deliberate gaps are not in it at all.**
  `make_trial` draws the occluder as *added* ink (`ink | occ_mask`), so the
  paper is never blank between the two halves and a junction always forms.
  Measured consequence: `t_no_node` (neither half touches a junction) is
  **0 of 90** topology failures. The chainer has no mechanism for a free tip
  either -- `build_links` only ever collects arms whose node id is >= 0, so a
  tip with no junction is not a matching candidate at all, and a drawn-on-
  purpose gap is unjoinable at **0%**, not "sometimes missed". `free_end`
  exists only as an IC-j8 counterfactual.
  *shortcut: the deliberately broken stroke is unmeasured and unjoinable;
  raised when free tips are censused on real ink and a gap instrument is
  registered (this is item 2's thread).*

- **`wrong` only counts contamination by the artificial occluder.** A real
  drawing joins separate strokes on purpose, and there is no ground truth for
  that in this trial set, so "joined things that should stay apart" is
  measured only through a proxy.
  *shortcut: precision is a proxy; raised when a stroke-level ground truth for
  intentional joins exists.*

- **`MERGE_PX` and `C_UNMATCHED` were never swept.** The 972-combination grid
  covers `TAN_SKIP`, `TAN_WIN`, `KAPPA_REF`, `THETA_MAX`, `D_MAX`, `W_KAPPA`
  only. IC-j8 then measured that 46% of the rejoin failures are ridges 6.2 to
  11.5px long against `MERGE_PX = 6.0`, and IC-j7 that `THETA_MAX / THETA0`
  equals the `C_UNMATCHED` break-even exactly.
  *shortcut: the two constants that look load-bearing carry their as-written
  values; raised by the bounded sweep registered 2026-10-12.*

- shortcut: IC-j6's `rejoin` figure of record changed on 2026-10-17 when
  IC-j9 adopted the touch merge into `stroke_chain.py`
  (sha `e9e49267c339f885` -> `53b8406cfbb4f4d1`). Pre-adoption 0.643
  (`[40:171]`) / 0.619 (`[0:40]`); post-adoption 0.752 / 0.711, `wrong`
  unchanged. Raised when: any entry quoting 0.643 is re-read -- it is a
  pre-adoption reading and stays as written. `score_touch_merge.py` now
  refuses to print a verdict unless its reference arm is the pre-adoption
  sha, because with the repair installed both its arms are one code path.
- shortcut: the post-adoption failure buckets have NOT been re-measured.
  `t_relay_other` at 66.2% of remaining failures is arithmetic on IC-j8's
  measured parts (111 failures, 34 gains all inside `t_touch_split`), not a
  measurement. Raised when: a registration needs the post-adoption profile --
  it must re-run the bucket classifier, not cite that number.

- shortcut: a cost leg on a regression count uses zero tolerance
  (`broken = 0 by set`, and IC-j11's `newly_wrong = 0 by set`), which refuses
  any trade and has now twice rejected a candidate better on every aggregate
  -- IC-j12's cell gained 45 trials and lost 2 of 379. **When it is raised:**
  in the registration that precedes a **third, never-read** chainer band,
  never by re-reading `[0:40]`, `[40:171]` or `[171:271]`. Three conditions
  on that registration, the first two added by the foundation on 2026-10-18:
  (a) the observed **0.53% (2/379) is now known**, so a tolerance may not be
  set to a round number that happens to clear it -- it must be derived from
  something that is not that observation (e.g. the band-to-band variability
  of `broken` measured on a band that is not the one being read), and the
  conflict is stated in the registration; (b) the **required gain** is
  written down too -- whether 45 rejoins and `wrong` 23 -> 16 are worth
  0.53% breakage can only be decided before the numbers are seen; (c) the
  IC-j12 verdict of **NOT ADOPTED stands** and is not revisited by any of
  this.
- shortcut: the chainer's "reserved" index ranges (`[0:40]`, `[40:171]`,
  `[171:271]`) are over the raw 5018-file `line/` directory, so they are
  reserved within Track J's chainer work only and are not disjoint from the
  koma `train` split the fill tracker fits `freq_table` on. **When it is
  raised:** if a *learned* arm is ever evaluated through
  `calibrate_stroke_chain`, at which point the ranges must be re-cut through
  `koma_split`.
- ~~shortcut: IC-j10's cv2 nesting violations are counted but their rim
  distance is not asserted; rim confinement (<= 1.0px) is established for the
  exact-EDT masks only.~~ **RAISED 2026-10-18** by the paired C1
  re-measurement (`tools/evaluation/c1_paired_nesting.py`,
  `results/fill_koma_20261010/icj10_c1_paired.txt`): cv2's worst rim distance
  is **1.00px** on the same 49 panels, and the lattice accounts for 98.9% of
  cv2's violation count.
- shortcut: the IC-j10 registration's second expected value (core area
  `r=8 / r=4` = 0.63) cites "an r-census on 60 held panels, recorded
  2026-10-14" that is **not in this repository** -- no entry, no commit, no
  results file on that date. **When it is raised:** it is not; the number is
  withdrawn as a registered expectation in the IC-j10 result entry and the
  quantity is re-registered from the 956-panel measurement if it is ever
  wanted again.
- shortcut: a registered expected value may cite only a date and a sample
  size, with no entry or results file behind it -- IC-j10's D2 did, and the
  measurement it named could not be found afterwards. **When it is raised:**
  now, as a rule -- every registered expected value names the entry or the
  file it comes from, and one that cannot is not registered.
- shortcut: a full-held region-F1 pass costs **hours**, and the cost is set by
  a handful of multi-million-pixel panels rather than by the panel count. Three
  measurements: the 2026-10-18 arm scoring ran **7h26m at 27.6 s/panel** over
  969 panels (`results/fill_koma_20261010/per_panel_held_nc2repaired.csv`,
  21,302 rows); IC6 ran in **under 49 min** (registration committed 22:05,
  artefacts written 22:54 on 2026-10-16); IC7 is measuring **12.1 s/panel**
  (200/969 in 40m15s, pid 709204). **No size guard is installed, and that is a
  decision, not an omission** -- dropping the large panels would bias the
  sample toward low-fill material and break comparability with the 969-panel
  reference, so the cost is paid instead. **When it is raised:** when a
  *learned* arm has to be scored more than once inside a GPU slot. The cut is
  then **caching, not subsampling**: `region_metrics.match` calls
  `regions(gt)` on every invocation, so a 22-row panel rebuilds the GT's
  connected components 22 times. To be measured before it is believed.
- shortcut: a wall-time estimate taken with `--limit N` **is not a sample**.
  IC7's registered "~90 min" came from 12 panels at 3.2 s/panel, but those are
  the first 12 by sorted name and the real rate is **12.1 s/panel (~3h15m)** --
  the registration's own fallback applies (the run is still read and the
  overrun becomes this entry, never a reason to cut panels). Same family as
  the F3 defect, where 49 panels were compared against 969. **When it is
  raised:** now, as a rule -- a cost estimate either uses a mechanical spread
  (every k-th panel, as IC-j10's exact-EDT leg did) or is written down as an
  unverified guess.
