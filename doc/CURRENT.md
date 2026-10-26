# Current Project State

Updated: 2026-10-24 JST (early). Items 1-5 are 2026-10-23: **the day Track J's
slot closed, Track L opened and produced a result, and the foundation put the
author's artwork on a public repository.** Item 6 is this file's own compression,
done 2026-10-24. In order of what a reset session needs:

**1. The fill line has a new target, and it is not the instrument.** Track L's
first job, run the day its tree opened, compared the radius-4 mask to `ako6`'s
**production** fill layers -- the first external standard this project has ever
had. The error direction is **deficit** (area ratio 0.482, 64.0% of the
production fill missed, region-F1 **0.1355** with 1.48x the region count, so it
**shatters** rather than shrinks). **But the mask is not the main cause**: the
line raster -- **a separate image composited from the line layers, not a copy of
the manuscript** -- does not contain **46.4% of the production fill's pixels in
`ako6`, 40.4% in
`ako7`**, two works at the same level with `ako6`'s values reproduced as the
known cell. **So Track J's "the rough does not mark the fill, 96%" and every
figure built on `fill_mask.py` were measured against a raster that is itself
missing fill.** This moves the target rather than the instrument and is the
heaviest open item in the fill line.

**2. Track J's four-day slot closed cleanly and left five rules.** C3 fired,
C4 removed both models (M1 falls to the off-by-one control), C1 and C2 failed
on every leg, the sign was identical in all nine threshold cells, and
`all_black` recovers 5-8x more fill pixels than either trained model. **The
headline is "no arm predicts fill in this formulation", not "no learned model
is needed"** -- a distinction Track J proposed against its own verdict. No
number from the slot survives; the rules do (Working Discipline 1's
verdict-word sibling, the name-counted inventory, the check that must name what
it reads, "an implication is not a check", and "do not close until the side the
proxy excluded is explained").

**3. The production standard exists now, and `black` is a generation.** Two
works delivered: `ako6` **781** fill layers over 107 pages and `ako7` **399**
over 71, in `data/incoming_manuscript/`, with the selection rule recorded in
each manifest. **70-75% of the production fill lives in layers named
`線画 ...`**, so a name-based design would have dropped it. The author's process
knowledge: `black` was *meant* as the fill, the digital line pass tends to fuse
it, fill-only `black` layers also appear, and going back to analogue separates
them again -- **so whether `black` is fill is a property of the generation, and
2018-2019 is a generation mid-move** (`beta`-named area share 30.8% -> 53.1%
from `ako6` to `ako7`). **The raster's loss is a separate thing and is ours**: a
16.0 width cap in `merge_line_candidates` drops the widest layers, so one giant
fill layer per page accounts for it -- a knob, not a fact about the corpus
from `ako6` to `ako7`). The whole-corpus tone scan also finished: **17,695 used
tone layers of 86,308 configured, five densities and three rulings covering
99.8%**.

**4. The foundation published the author's artwork and then removed it.**
`git add -A` carried **1,018 files, ~99MB** into a public repository -- 884
binary fill masks and **five full-resolution line-art pages**. **A force-push
does not remove data**: GitHub serves unreachable objects to `fetch <sha>`, and
this file's own commit message had published the SHAs. Fixed by making the
repository private, rewriting both affected branches in one pass, **deleting and
recreating the repository** (possible only because forks were 0), and pushing an
explicit refspec -- a mirror push would have re-uploaded it through the rollback
tags. Verified gone from a fresh anonymous clone. **`data/` is now ignored as a
directory**, and the lesson is in the traps: **a rule written as a list of
instances does not cover the next instance.**

**5. Five units of one failure, in one day.** Name against signature, layer
against content, screening against ranking, population against measurement,
work against page -- and later commit against path, snapshot against moving ref,
and one work's mechanism against a rule. **Each time the number was right and
the thing it was a number *of* was wrong.** Three of the day's false positives
came from `tools/audit_track_ledger.py` itself, which now carries checks 7 and
the narrowed status reader (`--self-test`, 11 cases).

**6. This file was compressed on 2026-10-24, and the shape is the point.** It
had grown **4,361 -> 6,820 lines on 2026-10-23/24**, against a project whose one
handover document stopped working at 900. **It is now 2,436**, with nothing
deleted and no measurement dropped -- every numeric token removed from a section
was looked up in the file that received it before the change was committed. What
moved, and the rule each move followed:

- **The rules stay in the reading path; the evidence moves to a focused doc.**
  `Known Tool Traps` was 1,768 lines of 83 incidents and is now a grouped index
  of one line per rule, citing `doc/tool_traps.md` as `T<n>`. Track K's 1,078
  lines went to `doc/track_k_record.md`, keeping in the ledger the summary **Track
  K itself wrote to be read instead of its bullets**. Next Actions item 11's
  1,045-line record of one day's correspondence went to
  `doc/fill_line_record_20261023.md`.
- **A section called "Current" is not a check either.** Five sections named
  `Current <something>` had been frozen since 2026-07; they are in
  `doc/archive/current_20260726_raw_extraction_era.md` with the seven dated
  entries of that era.
- **A check has to follow the prose it was written for.** Moving sections took 19
  quoted commit hashes out of `tools/audit_track_ledger.py`'s check 7 without
  raising anything -- a check going quiet is this project's most expensive
  failure shape -- so the check now reads `HASH_SOURCES`, and its comment says to
  add to that list the next time prose with provenance leaves this file.
- **What was deliberately dropped**: three numbers, all of them a finished run's
  pace (M0 at step 29,050 of 40,000, 1.47 it/s, 50,950 steps remaining). A
  progress reading is not a measurement of the material.
- **Retractions were kept and one was repaired.** The index line for T75 said L7
  was "running"; L7 was measured and retracted the same day, and the line now
  says so and points at where. If a figure here has a retraction, they travel
  together -- that is the half of the bloat that was correct as a record and
  wrong as something to read.

**Open, as of 2026-10-24 morning.** **The file everything was waiting on has
arrived, and reading it is Track L's work, not this session's.**
`inbox/page_layers_priority65.csv` landed **02:25**: every layer of the 65
priority pages, pre-selection -- **7,362 rows over 65 pages** (`ako6` 31 /
`ako7` 34, none missing), 1,818 folders, 5,544 raster layers, **2,963 with
content**, 20 columns including `visibility` / `opacity` / `has_mask` /
`rendered` / `drawn_ratio` / `black_among_drawn` / `preview_dark`. It cost
**~25 s/page**, ~25 minutes in all, against the forecast 85 s.
**The quantity the fill line was after, as the extraction side framed it**:
**759 layers pass its ベタ colour signature and 128 fail it while being dark on
the page** (`preview_dark >= 0.5`), top names `線画` 24 / `effect` 17 / `black`
11, `black_among_drawn` median 0.823 -- **and the reason offered for that 0.823,
"just under the 0.95 cut, where an anti-aliased line layer carrying ベタ would
fall", is RETRACTED by Track L's measurement the same morning** (below).
**The 128 is the extraction side's own cut, and it asks us to re-cut with ours**
(every column is in the file). `preview_dark` is measured on the 1/4 preview, so
it has **no boundary precision**.
**Routed to Track L at 07:39** --
`outbox/reply_l_page_layers_priority65_20261024.md`, copied with the CSV, the
JSON and the notice into `../lineart-fill-region/inbox/`, which had to be
created. **`inbox/` is gitignored, so a file that lands here is invisible to a
track until it is copied**; the notice was addressed to Track L from the start.
Reading the distributions sets where to cut, which is pre-registration, and this
session holds the band rather than setting it (規律5). **L was already on it**:
`tools/evaluation/l8_inventory_reconcile.py` was in its tree at 07:38,
uncommitted, with P1/P2/P3 written before looking.
**The foundation's receipt check** -- that what arrived arrived, not what it
says: 7,362 rows x 20 fields with nothing ragged, no duplicate
`(page_id, layer_id)`, and `rendered=True` 5,544 / `drawn_ratio > 0` 2,963 both
matching the notice. Four things a re-cut has to know, all in the reply:
- the CSV carries a **UTF-8 BOM**, so the first column name is `\ufeffpage_id`
  unless it is opened `utf-8-sig`
- **`layer_type` holds raw CLIP values beyond the documented 1/2/3** (`0` 1,796,
  `256` / `1584` / `800` 65 each, `4098` 1) and **disagrees with the `folder`
  column in **206 rows** (this file first said 141 and the count is corrected
  below), so filtering by `layer_type==1` is not filtering by
  `folder==False`, and which one was used belongs in the registration
- **`用紙` comes back `drawn_ratio 1.0` / `black_among_drawn 1.0` on all 65
  pages** -- white on the page, **fully opaque as measured** (not black: the
  quantity is alpha, corrected 2026-10-25, and the extraction side's own
  `draw_color_main` is `[255,255,255]` on the paper rows it could not render)
  -- so **any cut on this quantity passes the paper layer**
- **31 layers have content but no `preview_dark`** (`マンガ用グラデーション` 15,
  `rendering_shadow_*` 9, `背景グラデ` 3), which a `preview_dark >= 0.5`
  condition **drops silently**
**A fact about the 65-page pool itself, found by the foundation on 2026-10-24
while checking a different claim, and not previously recorded anywhere**:
**the pool holds 16 pairs of pages that are two revisions of the same page.**
Measured as Jaccard overlap of each page's multiset of
`(name, ancestors, drawn_ratio)` triples: 16 pairs at **0.598 to 0.900**, each
pairing a `<work>/pageNNNN` against a `<work>/doneN/<work>-NN` -- the background
and effect layers byte-identical by `drawn_ratio`, the character line and fill
layers different. `ako6/page0117` vs `ako6/done6/ako6-57` is one: 110 of 127
layers identical, the `線画 *` and `black` layers all different.
**The pair list is not quotable without the rule that made it, which Track L
established by re-deriving it and getting a third value.** The rule used here:
**all 7,362 rows**, triples of `(name, ancestors, drawn_ratio)`, Jaccard
**>= 0.5** -- which gives **17 pairs over 34 pages**, 16 of them within one work
(0.598 to 0.900) and the 17th across works at 0.693. Excluding the layers with
nothing drawn gives **16 pairs over 32 pages** instead, because empty layers
match each other and lift the overlap. **What is stable and what is not**:
**15 same-work pairs survive every rule tried**, and under this rule there is a
**gap** -- no same-work pair falls between 0.4 and 0.598 -- so the threshold is
not the fragile part, the **population** is. The one rule-dependent same-work
pair is `ako7/page0175` against `ako7/done2/ako7-40` (0.598 with empty layers,
**0.370** without); the cross-work pair sits in every version (0.693 / 0.664 /
0.537) and the shift is not even monotone, since
`ako6/page0118`-`ako6/done6/ako6-56` *rises* from 0.691 to 0.710. **So the pool
holds 49 or 50 distinct pages**, and that reading is what does not move. This
file said "about 32 pages" first and the count under its own rule is 34.
which every per-page figure on this page is computed over -- it is not wrong,
it is a pool with a known multiplicity, and any **per-page** rate reads about a
quarter heavier than a distinct-page rate would. **Which member is the finished
one is not decidable here**: the folder names suggest it and 規律6 says the
author decides. `inbox/work_duplicates.json` does **not** cover this -- it
matches whole works across year folders and names as its own limit that it does
not separate 同じ作品の別版. **One pair does not fit the pattern**:
`ako6/done4/ako6-18` against `ako7/page0073` at 0.693, across works, 97 of 140
layers identical. Unexplained, deliberately not theorised, and a question for
the extraction side.
**Track L read the distributions the same morning (L8, `b9a2831`, pushed) and
the colour signature is not what removes areas from the standard.** Against the
author's own ベタ as delivered (496 layers, `ako6`+`ako7` priority pages,
`beta_manifest.json`'s 判定 mapped onto the inventory): bam median **0.9986**,
**100% at or above 0.95**. Against the dark layers that the **area floor**
(`dr < 0.002`) cut: 1,414 layers, bam median **1.0000**, 87% at or above 0.95.
**The cut side is the blacker one** -- this author's line layers carry **no
anti-aliasing at all** (`線画 skin` and `線画 hair` both bam 1.0), so
`black_among_drawn` does not separate line from area in this corpus and the
0.823 story does not hold here.
**What the standard has never seen**: 2,177 layers with content that are neither
delivered nor in `excluded`, **1,551 of them dark on the page**. Split by which
axis cut them, **the area floor alone cut 1,207** (median **20,657 px**, 257
layers above 46,000 px, spread over **61 pages**) and **colour alone cut 63**.
**The 128 exists and is not the bulk.**
**Track L's own registration missed and is kept**: P3 predicted bam would
concentrate below 0.95 and 1,239 of 1,551 came back at or above it. Its
**規律2 anchor passed on the second run and FAILED on the first** -- 496/496 =
**1.0000** recovered both ways after the fix (reverse 0.9940, 3 layers over),
against a pre-registered **0.98 both ways**, with the first run at **0.9745**
and **the bar not moved**; the cause was its own surrogate for the name family
reading the layer name while the extraction side reads the ancestor folders, an
axis the registration had declared as surrogate. **Three delivered layers pass
all four written conditions of 判定 and are not in `excluded`** (one at
opacity 79, two at 256), so **the written rule does not determine the delivery**
and a query is out to the extraction side.
**The band check this session owes is not payable yet, and Track L says why**:
whether those 1,207 layers are line or area **is not decidable from the
statistics** -- a 2.7 px line running 7,600 px is 20,000 px, the same as a hair
mass, an iris or a shadow. So at **07:42** it asked the extraction side not for
a condition but for **all 2,963 content-bearing raster layers of the 65 pages,
unselected, at native resolution** (~0.25 GB, `layer_id` in each filename), and
**named no layers on purpose** -- naming them would put its own selection inside
the standard, which is the shape of the retraction it had made the day before.
Requests #2 (the composited page) and #5 (mixed layers) are **withdrawn**.
The condition, and the band check with it, comes after `keep4` runs per layer on
those pixels; **until then Track L publishes no number about mask precision**,
and when the pixels land they go under `data/` and are never committed.
**The foundation's four structural notes all landed**, and one was a live
defect: the **31 layers with content and no `preview_dark`** were being dropped
silently by L's own `preview_dark >= 0.5` (all undelivered, median
`drawn_ratio` **0.00696**, above the area floor, 20 of them
`has_gradation_fill=True`) and are now counted separately as "layers whose
darkness is unknown". `用紙` was **harmless** -- all 65 peak at `preview_dark`
0.4153 and never entered a dark set -- and is excluded explicitly anyway. On
`folder` versus `layer_type`, L uses **`folder`**, because **`folder` is the
exact complement of `rendered` on all 7,362 rows** (verified here), and
**the disagreement with `layer_type` is 206 rows, not the 141 this file first
said** -- 141 counted `folder=True & layer_type=2` (49) and
`folder=False & layer_type=0` (92) and left out the 65 roots at
`folder=True & layer_type=256`. 79 of those rows have content and sit in L's P2
population.
**The order the author set, unchanged**: look at the distributions, **then**
write a file naming the `page_id`/`layer_id` to export at **native resolution
with grey levels** -- the line layers that contain fill, which the colour test
had been dropping. **Do not fix the selection condition before seeing the
distributions**; that is how the retracted L7 happened. Deliver by SCP to
`C:/Users/sh1/code/extract_line_and_sketch/inbox/` and **verify with `dir`**.

**Three notices arrived from the extraction side between 00:47 and 01:51 on
2026-10-24, and two of them change what is written above.**
- **The composited-page route is dead for area boundaries.** `CanvasPreview` is
  **exactly a quarter** of native (7016x6543 -> 1754x1635, and 4.00x on `gakuen`
  too) and **no native composite exists in the file**; halftone is burnt in but
  at 1/4 a 60L screen's ~10px period averages to grey. It is enough for the
  `preview_dark` statistic and not for a boundary. Visibility is probably
  respected (one `visibility=0` layer at 0.064 against 0.720/0.798) on a sample
  of **one layer**.
- **The gradient layers' masks were measured**, 1,754 layers over 85 works, and
  **the 2026-10-23 first run of that survey is invalid** (it stopped at the body
  raster and never measured the mask in 68% of layers). 99.4% carry a mask with
  content; **the mask is hard and the paint is soft** (edge p50 **0.744** against
  **131.1**), and mixing the two produces a meaningless 77.3. Coverage separates
  by use, 1 : 7 : 21 for `乳首グラデ` / `肌グラデ` / `背景グラデ`, and `背景グラデ`
  is the most area-like family (**67 of 67 hard edges**) and the smallest (69
  layers). 59 layers (3.4%) cover a tenth of a page or more.
- **"The channel drops messages in both directions" is retracted by the
  extraction side itself**: all three requests had arrived, and what was missing
  was its own habit of looking in its inbox. One real fault remains -- a reply
  with no destination recorded went to `C:/Users/sh1/inbox_from_lineart/`.
  **The one destination from now on is
  `C:/Users/sh1/code/extract_line_and_sketch/inbox/`.**

**A hole the audit now closes, found 2026-10-24 from one line of Track L's**:
**10 of the 14 trees do not ignore `data/`.** The rule went into the
foundation's `.gitignore` with `f7372e6` -- the 2026-10-23 incident's own fix,
after 99MB of the author's artwork reached a public repository -- so every
branch cut before it, and every branch that has not merged since, is one
`git add -A` from the same incident. `fill-region` is one of them and is about
to receive **0.25GB of native-resolution layers**; it is working around the hole
by keeping the pixels in the foundation's tree and reading them by absolute
path, which is sound and is not the fix. **The fix is one line on each branch**
(`git checkout foundation -- .gitignore`), and it belongs to whoever sits there.
Nothing is tracked under `data/` in any tree today -- checked -- so the exposure
is prospective, **but it was closer than that sounds**: Track L ran `git add -A`
in its tree this morning to commit L8, and nothing was swept in only because
`results/**` and `inbox/` were already ignored. **`fill-region` is fixed**
(`1d77d86`, pushed, three lines taken verbatim from the foundation), and its own
reading of it is the one to keep -- it had *stated* the hole in a notice and not
closed it, so knowing and checking were on different sides. **Seven of the ten were fixed by this session the same day, on the user's
instruction to take the dormant trees only**: `aesthetic-judge` `46f34c0`,
`cleanup-refiner` `3428d0f`, `image-prompt` `c22bfc7`, `pair-signal` `4235dc3`,
`stroke-grammar` `de593f6`, `stroke-topology` `0e7eb02`, `solid-fill` `4af934c`
-- each tree clean before and after, each `.gitignore` taken verbatim from
`foundation`, **and all seven pushed on the user's instruction the same day**
(the foundation may act on a dormant branch with nobody sitting in it, and the
outward step was the user's to approve, which is why it was asked for
separately). Every commit touches `.gitignore` and nothing else, checked before
the push; all seven were fast-forwards.
**Two are left and both belong to someone else**: `face-words` has a live
session in it, and `panel-generation` is Track G, active and waiting on H.
**The remedy was quietly breaking something else, which is why each tree was
checked after**: the foundation's rule read `venv/`, several trees keep `venv`
as a **symlink** to the foundation's, and `venv/` does not match a symlink -- so
the first import left the symlink untracked in three trees. Fixed at the source
first (`f6faae0`, the rule is now `venv`), then re-imported: **T85**.
**`tools/audit_track_ledger.py` now checks the `data/` rule as check 9**
(asked of git, so a rule in any form counts; a plain directory that is not a
repo is not flagged), self-test 13 cases. The rule was already in CLAUDE.md and
in every briefing, which is exactly what made it worth checking mechanically:
**T84**.
**The audit carries 12 checks from 2026-10-25, self-test 28 cases**, and the
two new ones read each live track's own `doc/work_log.md`, dated level-2
sections from the last two days only: **11** a result whose own measurement is
registered by no heading in that log (registration cannot be checked by reading
a result; the one mechanical trace it leaves here is a heading), and **12** a
PASS or FAIL with no bar *and its number* in the section -- "FAIL (below the
bar)" names a threshold without stating it. On the first run 11 found nothing
and 12 found four, all of them verdicts whose threshold is genuinely absent
(two known-cell PASSes among them). **A third check was written for discipline
9-1 and removed after it ran**: it reported eight findings on two tracks and
every one was a false positive, because to a scanner the sentence that performs
the practice ("落とした数を別に印字する"), the sentence that reports a zero
("落としたページ 0") and the sentence that omits the reasons are one text. That
is **T93** again, so the removal is recorded rather than the check: **9-1 is not
checkable on prose**, its mechanical form is the per-reason table the instrument
prints, and before the run it is `moat`'s provisional rule 11. The audit also
flagged its own ledger on the first run: `0d975ac` and `4941c60` are commits in
the **extraction tool's** repository, now declared in `DEAD_HASHES_OK` with that
reason (**T77**'s family -- a hash that resolves somewhere else is provenance
only for a reader standing in the other repository).

**Two more deliveries were verified complete by Track L the same morning**, and
these are counts, not readings: tone **1,302 / 1,302** PNGs over 65 pages
(max 31 layers on a page, one page with none) and gradient **155 / 155** over
the same pages (max 14, 19 pages with none), **nothing missing, nothing extra,
no errors** against what each manifest points at. No covering notice has arrived
for either, so whether the extraction side considers them finished is a query
out. Their pixels are unread on purpose: the 2,963 unselected layers come first.

**The 2,963 native layers arrived the same day, and the extraction side's list
of what it owes Track L is now empty.** `note_all_layers_priority65_20261024.md`
with 1,591 + 1,372 PNGs under `data/incoming_manuscript/ako{6,7}/all_layers/`,
**733MB**, native resolution, 8-bit grey, not binarised, named `{pid}_L{layer_id}.png`.
**The value is coverage and not colour** -- `255 - 被覆`, so an area painted
white comes out black, the same reason `用紙` does; page darkness still has to
come from `preview_dark`. **The foundation's receipt check passes exactly**:
2,963 files on disk, and the census's own
`folder=False & rendered=True & drawn_ratio>0` selection is **2,963 with 0 on
either side** -- every delivered file maps to a census row and every selected
row has a file. 2,581 layers were not exported and the manifest's `skipped`
gives the reason for all of them as nothing drawn; 0 failed to render.
`tone_state` is `none` 2,758 / **`disabled` 197** / `enabled` 8, and the 197 are
the layers that appeared in neither earlier delivery. **The extraction side
retracted one of its own sentences in the covering note**: "an enabled tone layer
cannot appear here because `LayerRenderMipmap` is empty" is wrong for 8 layers on
`ako7_page0068`, where the raster covers the whole page and **the shape is in the
mask, not in the PNG** -- 3 of those 8 are in the tone delivery and 5 are not,
which it says it has not confirmed. The three layers the written 判定 did not
explain were answered earlier (an **invalid** tone effect, a condition that was
not in the rule), the size question was superseded by the extraction side's own
revised 0.52-1.07GB estimate before delivery and landed inside it at 733MB, and
the tone and gradient covering notices arrived.

**A structure DB over the whole corpus exists as of 2026-10-25, and no notice
announced it.** `db/manuscripts.duckdb`, **99,364,864 bytes**, built 10:16 from
`db/clip_structure.jsonl` (353MB, a scan of 7,634 pages -- **not all of them:
the disk held 7,667 `.clip` files and 33 were absent from the inventory, one of
them a whole work; **closed 2026-10-26, `pages` is now 7,667 / `layers` 312,898**;
T115** -- and **306,009 layers**, ~2h idle), documented in the extraction side's `doc/structure_db.md`
(its commit `4941c60`): one row per page, per
layer (folders included), per page x SQLite table, plus the colour and koma-frame
CSVs already delivered. Joins are **by `page_path` and `layer_id`, never by work
or page name** (same-named alternate versions exist). It reads the SQLite tables
only -- **no layer is rendered** -- so any question about pixels still needs a
render. **Track L found it by reading the extraction tree**; the foundation
verified the files by `dir` and fetched only the doc
(`inbox/note_extract_structure_db_20261025.md`, copied to L and M). **The 99MB
body was deliberately not fetched**: that is the size of the 2026-10-23 incident,
and it goes under `data/` only when a question needs it -- SQL questions are
cheaper to send than the DB is to hold. Three caveats are the extraction side's
own and travel with every column quoted from it: the `name_*` marks
(`name_balloon` / `name_lineart` / `name_frame` / `name_3d` / `name_sfx`) rely on
the author's naming and **miss layers**; `blend_mode` / `clip_raw` /
`colour_type_index` are **raw values whose meaning is unverified** (as are
`v_work_colour`'s 0 / 0.02 / 0.5 colour cuts); and **a field being present is not
a pixel being present -- an empty layer still has a mipmap**, which is T-M3
written by the other side first. Two columns are already checked against things
this project holds: **`has_frame_line` matches the author's frames on the 65
pages**, and **`z_global` is 0-at-root**, the direction Track L used for L15's
occlusion. In the same hour the **reason-by-reason line-selection ledger** is
being generated -- `config/line_selection_ledger_priority65.csv` with **`reason`
and `failed`** columns alongside the statistics, **[41/65] at 11:31**, each page
cross-checking `ledger_line_id == delivered_line_id` with JPEG diff 0 -- which is
request (3) of **T86** and the thing that turns A/B/C from an estimate into a
count (**[47/65] at 11:42**, all 47 matching the delivery). **The notice exists
from 11:42**, after the foundation asked for one: `inbox/note_structure_db_20261025.md`,
386 bytes -- the DB, its size, its layer count, where the documentation is, and
that the body was not sent. That is the whole of what a notice has to be, and it
is what the record was missing for an hour and a half.
**Owed**: nothing, in the sense that both sides have paid. **L registered L10
before running it (`1004fba`, 10:40) and the foundation's band check is
delivered** (`outbox/band_check_l10_20261024.md`): three of four predictions
pass the band, the registration is genuinely prior, and the pooling clause is
right. **What does not pass is P7** -- "keep4 is bimodal and the middle is thin"
carries no number, so no value can falsify it. **The foundation offered to let it
be quantified before the body was read, and Track L declined on the grounds that
it had already read the body** (`60849ac`): a number chosen now would be moving
the bar, not repairing the registration, and the `<15%` bar L first reported came
from the analyser written *after* the run. **So P7 is recorded as neither PASS
nor FAIL but "no test", with its 43.87% kept as a measurement and reserved as
material for the next registration's bar**. L withdrew its own earlier
"P7 FAIL" wording in the same pass.
**The foundation proposed L6's measured middle band at the panel level as that
bar, twice, and withdraws it the same day**: Track L's cleansing inventory (C2
below) finds that L6's panel set is a product of the dataset raster, so its
middle band is raster-contaminated and cannot serve as a known value. **What
replaces it is not another number but a condition**: a bar for a layer-level
quantity has to come from a layer-level population, and L6's does not. The
retraction is recorded here rather than silently dropped, because the
recommendation went out in two band checks. **The offer was correct as a band
rule and wrong about L's state, and L's reading is the one that stands**: the
repair is available only to a party that has not looked.
**The foundation's reading of L's KC2 failure, which L called its own
registration mistake**: two thresholds were made into one. KC2 is an identity
against a **published column** and must therefore use `gray<230`, because that
is what `drawn_ratio` is; **keep4's own mask threshold is a separate free
parameter** and does not have to equal it. **Which threshold is "the area the
author painted" is settled by the author's process, not by the number that makes
KC2 pass** (規律6) -- choosing the passing number is the instrument fitting
itself. Three further remarks, each a consequence of a rule adopted the same
day, and **L measured all three**: the 746 mismatching layers were a
**cross-generation total**, and split, they are **not** skewed by generation
(`ako6` 406/1,591 = 0.2552, `ako7` 340/1,372 = 0.2478) -- **what they are skewed
by is the name**, with `線画` 117, `black` 98, `レイヤー 1` 59 at the top of the
mismatches and `線画` 431, `線画 skin` 308, `線画 hair` 204 at the top of the
matches. **`線画` heads both lists, so some `線画` layers carry grey content and
others do not, and the name does not determine the family *within a single
page*** -- which belongs beside "a black layer is fill is a function of the
generation" in lesson 6, and widens it: the name fails to determine the family
not only across generations but inside one page. Second, **"version duplication
does not affect layer counts" is true in general and false for this family**,
since the background base layers are identical across versions, so the 125 rows
are **96 distinct compositions** (`base` 62 -> 48, `lineart`/`線画` 20 -> 14) and
13 of the 28 family pages are one of a pair -- **and P10's separation survives
the de-duplication**: `base` at keep4 >= 0.8 is 62/62 and **48/48**, the
`lineart`/`線画` members are 0/20 and **0/14**, so the split is not an artefact
of the multiplicity. Third, **P8 and P10 share 7 layers**, which L's report now
states, so passing both is not two independent confirmations. **L10's own framing of P10 is the strongest thing in the
registration**: the 125 were chosen by the author's words, so keep4 agreeing
with them is keep4 agreeing with the author rather than with its own shape
criterion. Track H holds H19 at **+0.044 against a
0.05 bar, bar not moved**, owes the foundation a notice (Next Actions item 9),
and has the five-page part-layer export to review with the author; one question
is out to the extraction side (why `line`-name ink-presence is 1.0000 in `ako7`
and 0.8371 in `ako6`).
**Awaiting a user decision**: whether to ask the extraction side to look for
cleanly-separated material in other works -- it was to be **judged after the
distributions above, and the distributions have now answered the conditional
part**: the criterion **has** stopped being the colour test, since the area floor
is what removes areas from the standard, so the question to put to other works
is about area and not colour. And **two candidate tracks queued behind L**:
**colour manuscripts were approved the same day and are now Track M in the
ledger above** (`../lineart-colour-manuscripts`, briefing `94c19cb`), with the
author's three answers -- the base layer carries the area division and 濃淡 sits
on top of it, single illustrations go in, open the tree. **One candidate is
still queued and undecided: the gradient masks**, which the survey above has
just made measurable (the mask is an area, and inverting one that marks
everything-but-the-character yields the character).
**The question Track L raised and did not ask has been answered, and it was not
M-only** (規律6; L declined to extend M's answer into its own context, and
the foundation declined to route the extension, so it went to the author as a
question). Author's words, kept whole because the asymmetry in them is the
result:

> キャラクターの仕上げに対してこのように作業したモノクロ原稿は存在はする ただほんの僅か
> また、3Dレンダリングから作成した背景についてはベース領域の上に線画クリッピングという構成になってるものが多くある(キャラクター線画ではないので基本的に対象ではないが、**コマの排他的な部分を表すことが多いのでこのベース面は使いようはあるかもしれない**)
> 同様に背景(背面処理トーン)の配置にもこの手法がつかわれている

**So "base layer plus 濃淡" is the main line in colour and background-only in
monochrome.** For Track L's open item the answer is **no change**: character
finishing done that way is ほんの僅か, so the 1,207 layers stay a question about
pixel morphology and **keep4 remains the authoritative line**; the 2,963-layer
request is unchanged. **What the answer opened is a different door** -- the
background base areas, measured as L9 and entered in Track L's standing homework
above. For Track M it fixes an asymmetry that its candidate 2 has to carry:
the same structure holds in monochrome **for backgrounds only**, never for
characters.

**The lineage, compressed 2026-10-24.** Each of these is now a ledger entry or a
record file rather than a paragraph here.
- **2026-10-19**: Track K closed its tree (`25e4dec`, pushed) and its second axis
  closed itself the day it opened -- K19 at precision 0.3333, CP95 upper 0.4955
  against a 0.50 bar, **bar not moved** -- so the letter-L reservation was
  released. Across six registrations K never moved a bar and four of the six
  would have passed on one moved by less than a hundredth. Track D returned the
  precision side of the pair measurement: **the pair is loose in both
  directions**. Track L was proposed on a lead the author supplied and this file
  could not derive -- the manuscripts carry ベタ and トーン as separate production
  layers -- which **corrected the premise**: ベタ was being discarded as expected,
  but **トーン was never seen at all** (in CSP its dots are not pixels), so **none
  of the 519 layers was tone** and 519 itself mixed fills with balloons and page
  furniture. 45,533 layer rows stand in its place.
- **2026-10-18**: Track K opened by user approval on the 1px skeleton's
  connectivity, upstream of Tracks F, G and H. The foundation's `diffusion`
  branch was renamed `foundation`.
- **2026-10-09/10**: Tracks G and H, two lessons, Track G's placement question
  settled on the reference frame, **lesson 14** (86% of the apparent placement
  structure was the panel's aspect ratio, and every position-bits figure in F and
  G is replaced), Track J opening to take the idle GPU, and the finding that this
  corpus's line art **encloses almost nothing**.
- **2026-09-30**: the ledger was rebuilt by reading every track's own files after
  this file had fallen 13 days behind. The structural cause -- Tracks G and H in a
  separate clone, invisible to `git worktree list` -- is still true and is in
  Next Actions item 9.

The Track Ledger, Lessons, Known Tool Traps and Next Actions are current as of
the Updated date above, **and so is Active Goal**: its "where the work is"
paragraph was rewritten on 2026-10-24, having stood at 2026-10-04 and named
neither Track J nor K nor L. Status still belongs to the Track Ledger; Active
Goal is the route. The raw-extraction-era sections that used to sit below it are
archived (item 6).

**Check where you are reading this from first.** If this copy is inside a track
worktree (`../lineart-<topic>/doc/CURRENT.md`), it is **frozen at that branch's
start point** and is not current -- the live file is `../lineart/doc/CURRENT.md`,
and on 2026-10-23 the copies on disk ran 13 to 82 days and 3,300 to 5,300 lines
behind it. A track session reads `doc/initial_notice.md` first, not this file.

Read from `foundation`, this file is the first document to read. It should contain only active state,
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
  diagnosis complete. Reopened 2026-10-19 for one measurement by user
  approval, which it delivered the same day, and closed again.**
  **The pair is loose in both directions, and now by measurement.** The recall
  side had been on record since 2026-09-15; the precision side -- for each
  rough skeleton segment, the median distance to the GT skeleton, the same
  construction mirrored -- was measured on all 8,467 pairs against a rotated
  null. Notice `inbox/note_track_d_precision_20261019.md`, results in
  `results/pair_alignment_strata_20261019/`, commits `791356f` / `08220e0`.
  - **The reconciliation was exact first**, as registered: 8,467/8,467 pairs
    matched by name and the maximum absolute difference on every recall column
    against 2026-09-15 was **0.00e+00**.
  - **Placement information exists on both sides, and the precision side holds
    about half of it.** `prec_aligned` (measured minus rotated) runs
    **0.0920-0.1687** by pool against the recall side's 0.1742-0.3014, a ratio
    of **0.505-0.775**, and it is not an artifact of the median: `>0` holds in
    **91-97% of pairs individually**.
  - **The new number is `prec_gt8` = 0.464-0.660**: about half the rough's
    skeleton length has **no GT skeleton within 8px**, against 0.198-0.315 on
    the recall side. Read together: **two to three tenths of the final line has
    nothing under it, and about half of what the author drew has nothing over
    it.**
  - **Track D's own caveat, which the foundation adopts**: reading `prec_gt8`
    as "half the author's input is unused" **overstates it**. `gray < 200`
    skeletonises pencil grain and tone into short fragments -- clippairskoma's
    roughs are largely texture -- so part of the unmatched half is grain, not
    misplaced line. The measurement was left exactly as registered; only the
    reading is bounded. Montage:
    `results/pair_alignment_strata_20261019/montage_precision_vs_recall.png`.
  - The secondary readings came out **mixed, per pool** on the registered
    window, which is the honest outcome for a reference whose flaw was already
    recorded (it measures the many-to-one crowding, not placement).
  - This is consistent with hypothesis 4 and lesson 8 from this track's own
    closure: a conditional objective learns to copy what it is given, and what
    it is given disagrees with the target **this much in both directions**. Why did 8,467 pairs contribute nothing to any
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
  test whether a rule survives a different artist). **Pushed; `origin/face-words` is
  `43a6e88` (2026-10-18)**, so it is no longer single-disk. The track commits
  only on the user's word, so its work log runs ahead of its branch by design.
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
  - *2026-10-23, **the track's first notice**, which replaces the readings this
    file took off its disk.* Verified here before folding in:
    `labels/face_groups.jsonl` is **518 rows, 402 distinct `order` (0-401),
    written 2026-10-23 00:45**, and `origin/face-words` is `43a6e88`. Read
    latest-row-per-key (the trap above): **373 complete -- 261 train, 68 panel
    holdout, 44 series holdout -- and 29 skipped.** The holdouts have been
    counted and never opened.
    - **The three pre-registered measurements were answered on 2026-10-12
      (H6, 68 faces) and the answer is replacement, not repair.** A Track F
      `cluster_set` captures a true line-level group **19%** of the time;
      **68% mixed, 22% fragmented, 7% never enters a cluster at all.**
      Quoting this beside the 2026-10-09 figures requires the denominator:
      **41% mixed and 14.6% uncovered are per cluster-mark (1,003 marks)**,
      68% is **per true group**. Both stand; they are different measurements,
      and Track H asked that they not be printed without that distinction.
    - **The mechanism is per-stroke part prediction plus a proximity merge**
      (tau = 0.04 x face size), scored as group recall (true and predicted
      group overlapping >= 80% each way), 5-fold by series group, mean of 3
      seeds, **train split only**. Per-stroke (H7-B) **0.278**, flat in data
      (H10: +0.007 going from 50% to 100%); a transformer encoder over all
      strokes of a face (H11-T2) **0.321** and **not** flat (H16: +0.067).
      **H17, 2026-10-23: 193 -> 260 training faces lifts the original 193 from
      0.319 to 0.351**, up on all three seeds -- the current best. Per-stroke
      precision/recall: eye 0.71/0.79, mouth 0.67/0.66, nose 0.58/0.55, brow
      0.54/0.47, ear 0.52/0.38, all short of the proposed rule bar 0.80/0.60.
    - **What did not work, every one pre-registered and no bar moved**:
      learning the merge (H12, **-0.077**, which the track records as its own
      design error), a learned cut on the proximity merge (H13, -0.012 to
      -0.024), supplying true face orientation (H10c +0.017, H15 -0.009),
      eye-relative position (H7-C), orientation from the head box (H9b, failed
      all three). **The ceiling is set by per-stroke part identity, not by the
      merge**: true parts through the same merge reach 0.83-0.94.
    - **Next, registered**: H18 retrains when orders 401-500 are marked, and
      **+0.025 over H17 on the original 193 buys another 100 faces; short of
      that, marking stops.**
    - **H19 read out 2026-10-23 and FAILED by six thousandths, bar not moved.**
      Adding the pixels around each stroke -- aimed at brow/hair and ear/hair --
      gives **Q1 - Q0 = +0.044 on the original 193 faces, up on all three
      seeds, against a registered 0.05.** All four instrument checks passed
      first. So the effect is real, consistent and **not worth what was
      registered for it**, and Track H printed it that way rather than rounding
      toward the bar.
      **Where this leaves the part-identity ceiling**: H19 was the attempt to
      get the hair distinction out of *pixels*, and it falls short, while the
      2016-2018 manuscripts carry `線画 HEAD hair` as **a production label** --
      the distinction H19 was trying to infer, already decided. That is an
      argument for the manuscript route and **not a recommendation**: H decides
      with the author, and the layer names have not been checked against their
      own pixels yet.
      **And H19 joins a pattern this file should name rather than enjoy.**
      Track K closed six registrations without moving a bar, and **four of the
      six would have passed on a bar moved by less than a hundredth**; H19 is
      now +0.044 against 0.05. **Two readings fit and this file cannot choose
      between them from the inside**: either the bars are set at about the right
      place and these effects are genuinely too small to buy, or the bars are
      being set just above what this problem affords, in which case a string of
      near-misses is what a well-run project looks like on its way to concluding
      nothing. **The discipline is identical under both** -- the bar does not
      move after the fact -- but the second reading is a reason to ask, when
      registering, *what would make this worth 0.05* rather than only *is 0.05
      the right number*.
    - **Track G's dependency is not payable yet**: at group recall 0.351 the
      unit is not ready to hand over, and Track H says so itself.
  - *2026-10-25: **H18 fails its bar, and the user paused the track on the
    material rather than on the method.*** 260 -> 317 learning faces lifts the
    original 193 from 0.351 to **0.369, against a registered +0.025**, so the
    author's marking stops; H19 (pixels around the stroke) is **+0.044 against
    0.05**, negative control -0.018. **No bar moved, and the track proposed its
    own stop.** The user's reading of the ceiling is **the material**: the
    marking set shows **unnaturally cut panels** (the author saw them; the
    effect on H's numbers was small because the marking was of faces), and H's
    strokes come from Track F's `panel_pack_20260919` thinning, whose source is
    the pair-dataset extraction program. **So H is paused, not closed**, on two
    conditions: the stroke-connection repair coming out of the region work, and
    clean line art -- then the 619 marks (order 0-502) are re-run with the line
    art swapped, which is H's own design and needs an old-to-new stroke
    correspondence H will build **after** the material is in sight.
    **Two consequences.** The 5-page part-layer export request is withdrawn and
    **was never relayed**: no `request_*`/`push_*` in this tree's `outbox` and
    the extraction side's `inbox` holds no such file (checked by `dir`, 23
    entries), so **a request that never left cannot be retracted** -- T91 read
    from the other end, and the withdrawal closes inside this project. And the
    **body-and-hands track the author proposed is deferred** until the extraction
    dirtiness is addressed, because two new yardsticks on dirty material cannot
    separate a ceiling of the material from a ceiling of the formulation; what
    H would hand it (the marking UI, the T2 net, group recall, the registration
    pattern) and the author's warning that **clothing complicates the body** are
    held in Next Actions rather than dropped. **The dirtiness case is drafted by
    L and M, not by H** (the user: it is already visible there); H contributes
    its panel-crop observation as an *observation with provenance*, owed in the
    form of native 1:1 crops and a count with a denominator.
- `../lineart-fill-region` (branch `fill-region`, **Track L**) -- **active,
  opened 2026-10-23 by user approval**, worktree of this repo, briefing
  committed at `7d9f080`. Catching the fill as an **area**, in a clean tree, on
  the user's judgement that Track J went toward muddle and stopped, and that
  only some ideas have been refuted rather than the question. Proposal:
  `doc/track_l_fill_region_proposal_20261023.md` (161 lines, eight closed doors
  with their measurements, five candidates, the trivial arms, the unit
  constraints). **It borrows Track J's instruments read-only and does not borrow
  the framing** -- J's tree houses prohibitions and near-closed registrations
  alongside the tools, correct for J's registrations and exactly what L must be
  free to reconsider. **Its first job is the instrument check, and the check is
  bidirectional**: `fill_mask.py`'s radius-4 mask has never been compared to
  anything external, and the extraction side's layer selection has no shape
  term, so each side holds the instrument the other lacks. **The motive, in
  Track J's wording**: a person can look at the outline and tell which area is
  filled; cheap closing cannot extract it; what can is unmeasured.
  **First result, 2026-10-23, the same day the tree opened: the radius-4 mask has
  met an external standard, and the error direction is DEFICIT -- but the mask is
  not the main cause.** 225 koma over 62 pids, all train split, version-exact by
  `pid` (notice `inbox/note_fill_region_20261023_instrument_check.md`, 90 lines).
  **Anchors first**: `blank` region-F1 reproduces at **0.0084 with the strata
  counts 956/13 intact**, `koma_split` self-test passes, and a new **IC-L1** puts
  the crop box on the page at ink **IoU 1.000** while printing the 2px-offset
  rows at 0.37-0.72 -- **so the test is shown to have resolution**, which is the
  form an instrument check should take. **The scale correction item 11 handed to
  L turned out to be unnecessary**: all 5,018 panels are native crops and the
  production masks are in the same page coordinates, so radius 4 is the same four
  pixels on both sides.
  **The comparison**: production fill 116,782,490 px against the mask's
  56,341,841 = **area ratio 0.482**; the mask **misses 64.0% of the production
  fill** and **25.4% of what it does paint is not production fill**. Both
  negative controls fall (off-by-one **0.0513** against 0.2788, mirror **0.1068**
  against 0.3206), so it is measuring position and not quantity. Radius
  sensitivity 0.4027 / 0.3599 / 0.3073 at r=3/4/6. **At the region unit it is
  region-F1 0.1355** (recall 0.1779, precision 0.1199) -- and the region count is
  **1.48x** the truth while the area is half, so the mask does not shrink
  uniformly, **it shatters**.
  **The headline, and the reason this is not simply a verdict on the mask**:
  production pixels present as ink in the line raster run **14.21% for
  `beta`-named layers against 83.71% for `line`-named ones**, so the mask cannot
  be charged for the difference. **The mechanism written here first -- "the
  extraction side's fill-classified layers are discarded when the raster is
  composited" -- is CORRECTED twice, in wording and in substance. In wording,
  because nothing is discarded: the raster is a separate image built from the
  line layers, so a fill layer is simply **not among its inputs** (the
  three-files statement above). In substance, by Track L's second work**: in `ako7` **38.05% of
  them are present**, against ako6's 14.21%. The surviving form: **how much
  separated fill stays in the raster varies by work (14.2% / 38.1%); "separated,
  therefore dropped" is a matter of degree and not a deterministic rule, and
  which layers drop cannot be predicted from names alone.** L's own diagnosis is
  that a mechanism seen in one work had been written as a rule -- **the day's
  pattern a fourth time, a name's implication taken for the mechanism** -- and
  that the second work is what caught it. **Restricted to area that
  exists as ink at all, the radius-4 opening keeps 72.8%** -- so path loss
  (primary) and instrument loss (secondary, about 27%) had been folded into one
  number. **L read this first as "the tone folder is the cause" and corrected it
  the same day**: it was correlation, name family and folder being confounded
  (`line`-named layers have 0 of 549 under a tone folder, so the 2x2 degenerates).
  **Pushed 2026-10-23 (`7a4df98`)**, after the repository was recreated, and
  verified `data/`-free from an **anonymous** clone of that branch -- a second
  path beside the pre-push check. **Track L is at a stopping point.** Its own
  next moves are not things it will ask for: the extraction side's answer on why
  compositing differs by work, or `ako8`/`ako9` pixels, which are what the end of
  the generation needs. **Its standing homework is the one this file cannot
  answer either**: how to build an external standard for areas at all, given
  that whole layers mix line and area, radius-4 cutting is circular, and the
  `beta`-named layers are the ones largely absent from the raster.
  **L9 (2026-10-24, `61f761b`, pushed) put a candidate into that homework, and
  it came from the author and not from the statistics.** Pulling the census by
  the author's own words (`rendering` / `base` / `背景` / `背面` / `bg`) gives
  **125 content-bearing layers over 28 pages, both works** (ako6 104, ako7 21),
  and **every one of them has コマ in `ancestors`, without exception**. In 41
  folders over 22 pages a `base` and a line layer sit in the same folder in a
  `back`/`middle`/`front` depth stack, each step a pair of **base (large, nearly
  pure black) and line art (small, grey)** -- e.g. page0056 `/p/コマ 2/"N"/middle`
  holds `rendering_base_middle0001` at gray 0.6, dr 0.0339 beside
  `rendering_lineart_middle0001` at gray 39.9, dr 0.0030. **The foundation
  reproduced the selection from the delivered census**: the same 125 / 28 /
  104 / 21 under "rendered and `drawn_ratio` > 0" (166 rows before the content
  filter, コマ in `ancestors` on all 166), and the known cell page0028 `bg base`
  at **dr 0.129759**, matching the manifest's `excluded` row exactly. **The 76-vs-62
  disagreement was reconciled the same day, and the body number is 62.** The two
  rules differ and the finding does not: L's 76 is *family AND undelivered AND
  excluded by `pd<0.5` alone*, the foundation's 62 is *family AND `base` in the
  name*. They share 55; the 21 only in 76 are `rendering_render_*` layers that
  join the family through `rendering` and carry no `base` in the name; the 7 only
  in 62 are layers that pass every rule and **are actually delivered**. Both
  carry the finding: gray median 0.55 either way, bam >= 0.95 on 76/76 and 62/62,
  `pd<0.5` on 76/76 and on 55/62. **62 is kept in the body because it is
  definable in one word and does not depend on L's `why()` implementation**, and
  L agreed; 76 is the inner reading *taken by the excluding axis alone*
  (55 of them plus the 21 `rendering_render_*`). The same shape as 141-vs-206,
  and L's own reading is that its definition was the more complicated one.
  **The 166-of-166 statement is the one to use**: L confirmed it, and コマ in
  `ancestors` on all 166 rows before the content filter is stronger than 125 of
  125.
  **What the reconciliation turned up is a fault in the standard itself, and it
  runs the other way from yesterday's retraction.** Those 7 layers -- `bg base`
  x6 and `base` x1, `category=beta` in the delivery manifest -- mean **7 of the
  author's 496 delivered ベタ layers on these pages are the background base
  areas the author said are 基本的に対象ではない**. The colour test cannot
  separate ink from a base area because both are pure black, so a base area
  whose page is dark above it (`preview_dark` 0.5072 to 0.9249 on these 7) walks
  straight through. **Verified here against the delivery manifest independently
  of L**: 496 beta rows on the 65 pages, 7 in the family, all 7 with
  `category=beta`, and their `drawn_ratio` median **0.02873 against 0.00433 for
  all 496 -- 6.63x**. So the standard is dirty **in both directions**:
  yesterday's retraction was *areas the standard never saw counted as absent*,
  and this is *the standard holding as an area something the author does not call
  one*. **The count is all that goes here.** What those 7 do to a コマ-level
  score needs the overlaps, so it waits for the pixels rather than being turned
  into a share now -- L's call, and the right one, since a share would walk off
  with its mechanism attached.
  **One of the 7 is a second revision of another**: `ako6/page0117` and
  `ako6/done6/ako6-57` carry the same `bg base` at `drawn_ratio` 0.003839, so
  the 7 rows are **6 distinct background compositions** (see the pool note
  above).
  **L's P5 half-missed and the bar was not moved**: it predicted the base areas
  would fail the colour test for not being black; they pass it, and what excludes
  them is `preview_dark`, because line art and tone sit on top of them so the
  finished page is not dark there. **This is why L8's area floor must not be
  generalised into "the reason areas fall out of the standard"** -- see lesson 15.
  **These 125 are not ベタ** and are not counted as ベタ, because the author says
  キャラクター線画ではないので基本的に対象ではない; they are a candidate
  **external standard for areas**, on the author's own hint that they often mark
  コマの排他的な部分. **What cannot be said yet: exclusivity.** Whether the base
  areas tile a コマ without overlapping needs the pixels, and **all 125 are
  inside the 2,963 already requested**, so nothing new is owed by the extraction
  side. The sample is **28 of 65 pages**, so how many exist in the whole corpus
  is a separate count -- a name scan on the extraction side, **not yet asked
  for**.
  **Track L's cleansing inventory, 2026-10-24, six items, and the reason
  discipline 9 exists.** L's own framing: *cleansing is not hygiene around the
  research, it is the research*. The numbers are in that tree's `doc/work_log.md`;
  the index is here.
  **C1 -- the dataset's panel line-art raster is a different page from the
  author's** (T86): 63.0% of the layers' ink arrives, **54.0% of the area layers
  are missing outright** against 5.1% of line layers, per-panel median 0.795,
  **140 of 248 panels below 0.90**. It is the *prediction-side input* of L7, L12
  and L13, so those arms were scored against something other than the ground
  truth. **The measurement stands and its meaning changed on 2026-10-25** (T86):
  the raster is a line-only instrument, so **4.7% of the ink is line material it
  lost** and the rest is its selection rule plus the `black` ambiguity. **The
  arms' scoring problem is unaffected** -- an input that omits 37% of the page's
  ink is the wrong page whether the omission was intended or not.
  **C2 -- so L6/L7's "35 clean panels" are a product of the raster** (T74
  corrected): all 35 have area layers the raster dropped, `in_raster` median
  **0.1395** against **1.0000** for the LEAKY set. Not "panels where the fill is
  absent" but "panels where the raster lost the fill layers".
  **C3 -- `preview_dark >= 0.5` was borrowed into the barrier**: 789 white or
  pale layers (balloons, white-painted foreground objects) plus 68 `/text` layers
  fell out of it, and the barrier is what stops a flood. **The constant was
  right for its own role and wrong for this one** -- the case discipline 9-2 is
  written from. Its effect on L14/L15's leak and the `O` rate (0.18/0.24) was
  still being measured when L stopped.
  **C4 -- `fill_mask`'s opening answers "area or stroke", not "ベタ or effect"**:
  **354 of 1,325 regions (26.7%)** come from `effect` and `描き文字`, and they
  stay in L14's primary figures because lesson 8 forbids cutting by name.
  **C5 -- a fattened line becomes an area**: `rendering_lineart_front0001`, a 3D
  renderer's output, makes **32 "regions" in one panel** out of jagged thick
  strokes. This is what made L13 FAIL.
  **C6 -- `has_mask` is True on all 2,963 layers** (T87), so it cannot split
  anything, and L14's attempt to split on it came back empty.
  **What this costs and who pays.** The cheap half L can do itself: an inventory
  of every selection condition in use, as *condition / role / how many it dropped
  / what calibrated it*, and a `raster_frac` check over the existing panel sets
  using C1's own instrument. **The expensive half is a question for the
  extraction side and not a request from L** -- Track L measures from layers and
  does not need the raster at all -- namely **what rule the current panel raster
  drops layers by, since it cannot be derived from the delivered census**: by
  name and by parent folder both the kept and the dropped appear, the commonest
  immediate parent among the dropped being `トーン` at 108. Rebuilding the panel
  raster is the costly option behind that answer.
  **Second result, same night: whether `black` is a fill layer is a property of
  the GENERATION, and the author answered by telling L to measure.** The author's
  process knowledge, which nothing here could derive: when the tone-layer method
  began, **`black` was intended as the fill**; but in a digital line-art workflow
  **fill tends to get put in during the line pass**, so `black` came to be
  **fused with the line while `black` layers that are fill only also began to be
  made**; and **after line art went back to analogue, the fill separates from
  the line again**. **So there are generations in which `black` should be called
  fill**, and ako6/ako7 need **not just layer names but the statistics**.
  **ako6's verdict, on all 107 pages and 781 layers rather than a sample:
  `black` IS fill.** Median `keep4` (the share of a layer's own ink surviving
  the radius-4 opening) **0.994**, p10 0.955, median inscribed radius
  **135.2px** -- *a mass, not a line* -- over 177 layers holding 28.2% of the
  fill area; `hair black` 16 layers at 0.979. Against that, the **unnamed `線画`
  layers (144, holding 41.4% of the fill area) are bimodal**: `keep4` >= 0.9 for
  **59** and <= 0.2 for **26**. **One name denotes both pure fill layers and
  pure line layers.** Per page: **both kinds 77, fused only 25, separated only
  0** of 102, separated share median 0.496 -- so the author's *"fused, yet
  fill-only ones also made"* holds **simultaneously inside one work**, and **the
  name fails in both directions**.
  **This gives the raster question its mechanism, which is the part that
  matters.** The line raster is composited from the layers classified as *line*,
  so **a separated fill layer is not an input while fused fill rides along inside
  a line layer that is**: ink-presence **14.2% for
  separated `black` against 83.7% for fused fill inside `線画 ...` layers**, and
  **ako6's raster is missing 46.4% of the production fill's pixels**
  (66,226,148 of 123,596,784). **The "function of the work's generation" reading
  written here is RETRACTED: the extraction side found the actual mechanism in
  this project's own code, and it is a width threshold** (the question sent 2026-10-24,
  answered the same night).
  `extract_clip_dataset.merge_line_candidates` drops layers when the composited
  line width exceeds **`LINE_MERGED_MAX_WIDTH = 16.0`**, **removing them in order
  of their own single-layer width**, and rejects the page if more than half must
  go. **Fill is by definition wide, so it is removed first.** The guard was added
  in August for the opposite reason -- to stop one fill layer inside a
  line-named layer from costing a whole page (`aljanne2 page0017` had been lost
  entirely over one such layer out of 42).
  **Measured per layer, almost nothing is lost.** On `ako6 page0028`, whose page
  retention reads 0.2034: **8 of its 9 line-named fill layers survive at
  exactly 1.0000**, and the only casualty is the single layer at `drawn_ratio`
  **0.348**, retained 0.0099 -- **and that one layer holds about 80% of the
  page's fill area**, which is the whole of the page-level figure. **A page
  aggregate here is one layer's fate wearing a page's name.**
  **So the ako6/ako7 difference is not about generations.** Layers at
  `drawn_ratio >= 0.2`: **5 in ako6 against 2 in ako7**, and **`線画`-named among
  them: 3 against 0** -- ako7's two large ones are called `color` and
  `レイヤー 23`, which never enter the composite at all. That is the entire
  source of `line`-family ink-presence reading **1.0000 in ako7 and 0.8371 in
  ako6**. Page retention inside ako6 ranges **0.17 to 1.00** and **does not track
  line width** (0.44 / 0.38 / 0.36 all occur at width 1.91). **The function is
  "does this page carry a giant fill layer", not "which generation is this".**
  **The general form, rewritten and now predictive**: *a raster loses fill when a
  page carries a fill layer large enough to push the composite past the 16.0
  width cap, and what it loses is that layer -- widest single-layer width first.*
  **Which layer drops is predictable**, where before this file said it could not
  be predicted from names.
  **Narrowed 2026-10-25 by the extraction side's own code** (`reply_raster_layer_
  drop_rule_20261025.md`): **that rule is real but it is a post-merge rule, and it
  only reaches layers that entered the merge pool.** A fill named `black` never
  gets there -- it is rejected earlier by the statistics (`black_among_drawn`
  capped at **0.98**, so a layer is excluded for being *too* black) and by the
  name family, and `black` is not in the line-name list because **this corpus uses
  `black` for line layers and for fills alike**, which the generator's own comment
  says. **So the width cap explains the fills that are named `線画`, not fills in
  general**; the general form above is correct for that family and wrong as the
  mechanism of the missing fill. **And the 16.0 is measured on the page reduced to
  a 1,024px long edge**, not in native pixels -- on a 4,000px page it is about 60
  native pixels, and the cap is calibrated as the last net against taking a fill
  for a line (healthy 11.21 against bad 43.93).
  **And the consequence is better than the earlier reading allowed.** The missing
  fill is **not absent from the manuscripts and not lost in production**: it is
  **discarded by a threshold in this project's own extraction step**, which means
  **there is a knob**. The loss stands as measured (46.4% / 40.4% of production
  fill absent from the raster) and **every figure built on `fill_mask.py` still
  inherits it**; what changes is that recovering it is a pipeline question rather
  than a fact about the corpus. **This is the second time today that naming the
  mechanism moved a finding from "a property of the material" to "a parameter of
  ours"** -- the first being tone's invisibility in `LayerRenderMipmap`.
  **And ako7 cannot be judged from the inventory**, which holds colour
  statistics and no shape. The colour-test mass by name runs ako6 0.182 /
  2018_ako7 0.174 / 2019_ako7 0.322 / ako8 0.000 / ako9 0.044 / ako10 0.000 --
  **and that table ranks nothing**, being an eight-page sample. **The
  "statistics" the author means are the `keep4` side**, which is sayable only
  because ako6 carries both.
  **And the briefing's worry about the delivery is refuted in pixel terms**:
  **91.5% of the delivery's 450,998,301 paint pixels pass the shape test**, with
  8 layers (0.24%) failing -- the "75.7% below 0.01" was a **count of layers**,
  which is discipline 7-7 landing on the foundation's own sentence. One of the 8
  is **`パース定規 1`**: the shape test caught a name the exclusion families miss,
  and it goes to the extraction side.
  **2026-10-24 evening, the author's three remarks are settled and the occlusion
  line is closed.** The panel frames arrived from the extraction side, the layer
  z-order arrived after them (`layer_z_order_priority65.csv`, **7,362 rows x 13
  cols**, parent empty on 65 rows = one root per page, parents missing from the
  inventory **0**, frame folders **245/245** agreeing with `koma_manifest.json`
  -- which the extraction side itself annotated as **not a strong known cell**,
  the same function having produced both orders; the direction "0 is the bottom"
  rests instead on a pilot page with no blend modes or clips compositing to
  **1.58/255** against the CSP preview bottom-up and **131.55** reversed, and
  **that check was not repeated on the priority 65**). `z_global` is a bottom-up
  walk of the tree, **not the resolved composite order**, so it is T-M3's
  neighbourhood and is to be used only with `parent_id` in hand.
  **(a) The panel correction, which is Track L's own retraction.** A `コマ N`
  folder is not one panel: the delivered frame masks give **186 folders -> 322
  8-connected components**, and measuring per component kills the "fragmentation"
  reading of the earlier bulletin (**top_share median 0.9794 / 0.9860**, merge
  1.0000 both pools). **The 0.5736 for ako7 is withdrawn.** What survives is the
  **omission**: of 307 components (whole-page `モザイク` 8 removed), **26.7% /
  19.9% of the area falls in no delivered rectangle, 58 components (18.9%) lie
  almost wholly outside, 36 of 65 pages affected** -- the rectangles are accurate
  where they exist, and the author's panels are about one fifth not covered, so
  the 248 panels were **a biased subset, not a distorted cut**.
  **(b) The author's occlusion claim is correct and does not help.** With real
  z-order and an acyclic selection (occluders **612 passed**; dropped
  `visibility=3` 128, `/sketch set` 96, `visibility=0` 56, other 4; `OCC_front`
  never referencing F), the overlap `|R and OCC_front|/|R|` has **median 0.0000
  in both pools** (regions 625 / 938; mean/p90 0.0263/0.0164 and 0.0423/0.0295;
  above 0.5: 10 = 1.6% and 30 = 3.2%). **"Nothing is drawn under what sits in
  front" is true as measured.** But the contour those occluders back is **about
  2% of the author's fill contour** (support +0.018 / +0.026), and the fill
  procedure does not move: W2 -> W6 **+0.000 / +0.002**, W3 -> W7 **+0.008 /
  -0.005** on `IoU>=0.5`; W7 against its control **2.53x / 1.28x**.
  **(c) And this is the C3 case settled with numbers.** The earlier substitute
  barrier (`preview_dark>=0.5`, the borrowed constant that dropped 789 pale
  layers) produced gains of **+0.468 / +0.551**; with the real z-order **they do
  not reproduce at all** (+0.000 to +0.008), and on the 87% of regions whose
  overlap is below 0.01 (541/625, 814/938) the difference is +0.002 / +0.001.
  **The gain was the circularity, not the occluders** -- discipline 9-2's L-side
  example is confirmed rather than weakened, which is what the user held the
  distribution of discipline 9 for.
  **(d) The author's own panels and frame lines do not rescue the procedure
  either.** On the author's panels, with the frame lines **and the panel edge
  itself** as barriers (free space = panel mask and not barrier, which is what
  CSP does), regions 1,325 -> **1,563**: ako6 W0 0.0944 / W2 0.1152 / W3 0.1296
  against control **0.0757**; ako7 W0 0.0416 / W2 0.0586 / W3 0.0778 against
  control **0.0707**. **The null rose when the instrument was corrected and in
  ako7 it passes the non-circular arms**; the 2.1-2.2x of the rectangle era
  becomes **1.1-1.7x** (ako6 1.25-1.71 / ako7 0.59-1.10). Frame lines contribute
  **+0.0128 / +0.0032**, so the "all-or-nothing, therefore it could matter"
  expectation was wrong.
  **Caveat the foundation found on 2026-10-25 while paying L17's band check**:
  **that control is not on the arm's population** -- `per_region.csv` holds
  **real 625/938 against shuffled 608/919**, and what the control loses is the
  large regions (every region above **0.5** of its panel area is absent from the
  control in both pools; the `>0.2` class goes 26 -> 19 and **20 -> 10**),
  because a 90%-inside placement cannot be found for them. **For the 1.1-1.7x
  statement the effect is at most 0.002** (if all 17/19 missing regions had
  failed: 0.0757 -> 0.0736, 0.0707 -> 0.0693) and the direction understates the
  gap, so the statement stands -- but **it is fatal for a purity-type quantity**,
  whose denominator is the region's perimeter. Registered as **T90**.
  **L17 is registered, instrumented and stopped before its own arm (2026-10-25),
  and the band check is paid.** The question is the user's: **take the words from
  the lines adjacent to a fill area** -- the first route in this line that runs
  **area -> line grouping** and therefore does not need a closed contour. Ground
  truth is the author's own layer names (`線画 skin` 172, `線画 hair` 130, ...),
  used as labels only and never for selection (L8's lesson). `purity(R) = max_L
  |adj(R,4) and L| / |adj(R,4)|`, `n_lay90`, ink assigned to the frontmost layer
  by `z_global`. Known cells **KC16-KC19 all PASS**, and the foundation
  reproduced KC19 independently (**ako6 625 / ako7 938 / 1,563**, matching
  `arm=real` in `l15_zorder_occ/per_region.csv`) and verified that **`purity`
  appears nowhere in the known-cell log**, so L has not looked and still holds
  the repair. **The foundation's ruling** (`outbox/band_check_l17_20261025.md`):
  the procedure L asked for is granted -- L runs the **control arm only**, over
  **5 seeds**, and reports the null -- **but the band is written in that reply as
  a function of the null, before the null is seen**, because "report the null,
  then the foundation picks a number" only moves the bar-fitting from L to the
  foundation. **P34 is refused**, and the first ground is arithmetic: `purity <=
  1.0`, so if the null exceeds 0.80 the companion clause `arm - control >= 0.20`
  **cannot be satisfied at all** -- L16's unreachable 0.30 in a new dress, which
  is what an additive bar does to a quantity with a ceiling. **P35 is refused as
  written** because `n_lay90 >= 1` always and a ring touched by one layer scores
  1 for free: **L's degeneracy check was one unit too high** (line layers per
  *panel*, median 6, one-or-fewer 6/246 = 2.4%), while purity degenerates per
  *region*. The band therefore reads only on **regions whose ring is touched by
  two or more layers**, takes as primary the **share of regions above a purity
  threshold** rather than the median (the deliverable is one word per region, and
  a median hides the tail), requires the separation to exceed **3x the null's
  seed-to-seed spread** before the run is readable at all, normalises against the
  ceiling (**half the remaining headroom**) instead of an additive 0.20, and
  fixes now where the reading goes **if the null itself saturates** (threshold to
  0.98, then "undecidable"). **And a wording correction before the run**: purity
  is a measure of **unambiguity, not of correctness** -- the label *is* the
  argmax, so no region can have a wrong word, and "the fill area works as the
  wand" is a claim discipline 6 reserves for the author (P37's disclosure). A
  PASS may say only that **the ink adjacent to a fill area concentrates in one
  layer per region**.
  **2026-10-25, the control arm ran and the band is fixed at a value, not a
  formula.** L placed the regions with a per-region seed so both arms see the
  same placement, dropped the regions no seed can place **from both arms** (the
  dropped set is the large end: **17 / 19 regions, 2.7% / 2.0%, median area
  748,881 / 783,136 px, up to a whole panel**), and reported, over 5 seeds at
  d=4px on regions whose ring is touched by two or more layers:
  **q0 = 0.1344 / 0.1558, r0 = 0.0172 / 0.0543, m0 = 0.6498 / 0.6672, k0 = 2.0 /
  2.0** (ako6 / ako7). KC20 PASS on synthetic data only -- a zero-displacement
  control reproducing the arm **must not be run on the real pages**, where it
  *is* the arm. The remaining 608 / 919 **agree to the digit with the count the
  foundation took from the old control**, so the same 17 / 19 were what the
  earlier T90 gap consisted of. **[Marked 2026-10-26, then resolved the same hour: the 17
  stands and the foundation's mark was wrong.** L's run prints **15** because
  `report_unplaced` filtered on `pur != ''`, which silently dropped the two
  regions whose purity is undefined -- and they are **the pool's two largest**,
  4,265,910 px each, twice the 2,015,180 px it printed as the maximum, with
  `adj` = 0 at every distance, so no layer ink touches them at all. 625 - 608 =
  17 and 938 - 919 = 19 both reconcile, ako7's 19 was never short, and the
  disclosure built to stop regions being dropped silently was dropping them
  silently (T104's shape: a filter removes the extremes). L prints the
  undefined case as undefined now. **The two regions have identical area because
  they are two versions of one page** (`ako6_page0030` and
  `ako6_done8_ako6-39`), which puts a duplicate-version pair inside Track L's
  own population and gives author question C a second consumer.] **The band, with the pre-registered formula's
  arguments filled in: `q1 >= 0.5672` (ako6) / `q1 >= 0.5779` (ako7), both pools,
  and `k1 <= 2`**; the readability floor `q0 + 3 r0` (0.1860 / 0.3187) is the
  weaker clause. The saturation branch does not fire (`q0 < 0.50`;
  `share >= 0.98` nulls are 0.0269 / 0.0346).
  **And the degeneracy filter is what makes this a test rather than a
  tautology.** In the control, **38%** of rows touch no ink at all (1,185/3,040
  and 1,732/4,595) and **about half of those that do touch exactly one layer**
  (922/1,855 = 0.4970 and 1,600/2,863 = 0.5589), where `purity = 1.0` and
  `n_lay90 = 1` hold by definition. Without the two-layer filter the null would
  be **0.5646 / 0.6276** -- level with the 0.5672 bar in ako6 and **above the
  0.5779 bar in ako7**, so the control alone would pass. **The foundation's
  one-unit-too-high finding about the degeneracy check is confirmed in numbers by
  the control itself.** The `|adj|` strata run in the predicted direction (ako6
  `<200` share 0.4118 against `>=1000` 0.1269), and because both arms use the
  same region shapes and differ only in placement, the real arm's denominators
  will be **larger** -- the denominator bias now runs **against** the arm, so a
  PASS cannot be its artefact and only a FAIL could be partly explained by it.
  Four bar-less disclosures were added before the run (single-layer regions in
  the real arm, where they are a **success** mode rather than a degeneracy but
  still cannot carry the bar; the unfiltered figure; the 17 / 19 excluded large
  regions measured in the real arm only; the strata and the d grid), and L
  weakened the registration's PASS wording before running, as asked.
  **2026-10-26: L17 was re-measured on the fixed instrument (`31c4ef1`) and the
  route stays closed, but one branch changed.** The bars rebuilt from this run's
  own control are **0.5552 / 0.5750** (the withdrawn 0.5672 / 0.5779 came from
  `q0` 0.1344 / 0.1558; close is not the same, and both moved run to run), the
  arm is unchanged at **0.2565 / 0.2933**, and the foundation reproduced every
  printed figure. **Read with the floor as a condition on superiority and not as
  a verdict gate -- the reading this ledger fixed before the run -- the two pools
  are one shape**: the arm misses the bar by **0.2988 = 5.28 `r0`** (ako6) and
  **0.2817 = 4.99 `r0`** (ako7), far outside the luck of the draw, so **P34'
  FAILs readably in both**; and separately **"the arm is above the control" is
  not established**, at **+2.58 / +2.54 `r0`** against a 3 `r0` floor.
  **What changed is ako6's branch, and `r0` is why**: 0.0172 -> **0.0565**, 3.3x,
  the T97 signature (ako7 0.0543 -> 0.0564, flat), so ako6's earlier *readable*
  FAIL (gap 0.1221 against 3 `r0` = 0.0516) is now in the pre-registered
  "neither PASS nor FAIL" branch. **So the honest headline is not "the
  conclusion did not change" but "the route is closed by the distance to the
  bar, not by the floor"** -- and the floor's own clause is held open in both
  pools. **One claim of L's is refused**: *more draws would settle it* does not
  hold, because `r0` is max-min and therefore non-decreasing in the seed count,
  so adding seeds raises the floor and (q1-q0)/`r0` does not converge upward.
  Refusing to add seeds after seeing the result is right; the stronger reason is
  that it could not answer. **P36's wall survives the fix** (d=4: 0.0919 /
  0.0932 -- two pools, and the single "9.2%" is ako6's value standing for both).
  **P35' PASSes and still cannot be quoted** (`k1` 2.0 against `k1 <= 2`, with
  `k0` also 2.0).
  **The superseded 2026-10-25 run, kept with its mark: the route is closed,
  `q1` = 0.2565 / 0.2933 against a bar of 0.5672 / 0.5779.** Regions 269 / 358 at d=4px on the two-layer
  stratum, **1.91x / 1.88x the null**, median purity 0.7951 / 0.7730 against
  0.6498 / 0.6672, `k1` 2.0 / 2.0. **ako6 FAILs readably and ako7 is unreadable**
  (`q1 - q0` = 0.1375 against `3 r0` = 0.1629, the pre-registered "neither PASS
  nor FAIL" branch), **so P34' does not hold.** Two notes on the foundation's own
  rules, held open rather than used to re-read the cells: the floor was written
  as a verdict gate when its content is a condition on the *superiority* claim --
  the arm is deterministic, so only the bar jitters (0.5 r0 = 0.009 / 0.027) and
  ako7 misses the bar by 0.2846, about ten times that -- **and the next band
  writes the bar and the superiority floor separately**; and **P35' PASSes as
  registered but cannot be quoted**, because `k0` is also 2.0 / 2.0, a bar the
  control satisfies (T7's family). **The mechanism of the FAIL is that the fills
  sit on the part boundaries**: the single-layer share is **lower in the arm**
  (0.3858 / 0.4986) than in the control (0.4970 / 0.5589), so the author's fills
  touch *more* layers than chance placement -- hair against face, clothes against
  skin -- and the ink along their rings mixes exactly the layers one wanted to
  separate. **The reason is not "too few lines".** The `|adj|` strata run against
  the arm (`>=1000` 0.2952 against `<200` 0.0000) so the FAIL is not a denominator
  artefact, and the unfiltered figures are the predicted tautology (arm 0.5434 /
  0.6457 against control 0.5646 / 0.6276, **the arm below the control in ako6**).
  **And the ceiling is smaller than the bar, independently**: only **9.19% /
  9.32%** of the line ink lies within 4px of a fill ring (6.1% at d=2, 15.2% at
  d=16), and the word it gives is **hair** -- in the band's population of 627
  regions, hair 46.1% against face 4.8%, over **21 distinct names**. **Even a
  PASS would have delivered a tenth of the vocabulary, and that tenth is hair.**
  The heavy end of the author's fills is **not in the band's population** (the
  15 / 19 regions no seed can place, median area 748,881 / 783,136 px): median
  purity 0.8579 / 0.8407, `share >= 0.90` **0.0000 / 0.3158**, `|adj|` median
  60,276 / 41,173 px, and no control, so no bar. P37's granularity (1.50 / 2.17
  regions per layer) is the author's to judge (discipline 6). **The honest
  sentence is not "fill areas give no words" but "the words they give are hair,
  and they are a tenth of the vocabulary".** Track L also caught and fixed a bug
  in the *name* column mid-report (label index read as a layer id), re-ran, and
  got the same `q1 / k1` because the counted quantities never used the names --
  and records the instance that matters: **the wrong name table made it suspect
  the population, which was clean** (`線画*` is 1,585 of 1,681 LINE layers, 94%).
  **L18 is registered (`1fb1128`) and the foundation wrote its band form the same
  day, before any of L18's numbers existed.** The question comes from the user's
  route out of L17: if the maths cannot close it, a UI can -- but **the UI's
  ceiling is P36's 9.2%**, so L18 asks whether a word attached to the ink
  touching a fill **propagates along the 8-connected component** and how far the
  coverage then reaches. The foundation's ruling, issued against the registration
  and not against a result: **the two registered bars (coverage, honesty) trade
  off against each other -- a huge component raises one and sinks the other -- so
  the primary is a single quantity**, `cov_honest` = propagated ink lying in
  components with `comp_pur >= 0.90`, over P36's own denominator (`ako6
  43,279,719` / `ako7 45,598,813` px, which KC24 matched exactly, and that match
  is what lets "9.2% grew to X" be stated as a ratio at all). A component that
  receives two different words enters `cov1` and **never** `cov_honest`, decided
  now, because a row whose outcome a convention settles is the thing the
  foundation ruled against in Track M's `0/0` case the same morning. **The bar
  and the floor are written separately this time**, which is the L17 lesson
  applied: the bar is **`cov_honest >= 0.30`**, taken from the registration's own
  section 6 ("9.2% -> three or four tenths and the UI's single click moves a
  bundle") because **a bar belongs where a decision changes and the foundation
  invented no number**; the superiority of the arm over chance placement is a
  *floor* (`>= 3 r0`), and the reading for "A holds, B does not" is fixed in
  advance as **"propagation carries enough ink for the UI, but what carries it is
  the ink's connectivity, not where the fills are"** -- which is not a loss.
  A third bar, `honest_rate = cov_honest / cov1 >= 0.50`, keeps a wide reach from
  being a pile of undo work. **The primary weighting is pixels, the opposite of
  L17's per-region median, for the same reason**: the unit is the unit of what is
  delivered, and here the delivery is moving ink. **And the degeneracy gate is
  ruled into pixels**: L proposed closing if the median component spans three or
  more layers, but counted over components **one giant component cannot appear in
  the median**, so the gate is the pixel-weighted median, with the per-panel
  largest-component ink share printed and the panels split at 0.5 as a reading
  stratum. The geodesic grid may not supply a passing cell; a finite `g` that
  satisfies honesty where `g=infinity` does not is a **disclosure**, and the
  cutoff is a design value belonging to the author (discipline 6).
- `../lineart-colour-manuscripts` (branch `colour-manuscripts`, **Track M**) --
  **active, opened 2026-10-24 by user approval**, worktree of this repo cut at
  `e9d680b`, briefing committed at `94c19cb`. **Nobody has worked in it yet.**
  Taking the **area division** from colour manuscripts, on the author's premise
  that **colour puts a colour in every area, so the division is complete rather
  than sparse** -- against ベタ, which covers hair, tops and tights and nothing
  else. Proposal: `doc/track_m_colour_manuscripts_proposal_20261024.md`
  (123 lines, four closed doors with their measurements, the stock, three
  candidates, the trivial arm). **It does not take over Track L's homework**:
  whether a colour area can serve as the standard for a filled area is a
  separate registered question, because a coloured area is not a filled area and
  the decision being learned is not the same decision.
  **The author answered three things on approval, and the first one reordered
  the candidates before any measurement was taken.** (1) The manuscripts are
  built as **a base layer that indicates the area, with 濃淡 layers on top of
  it**, some exceptions, and **the base is what to go after** since finding areas
  is the current target. (2) **Single illustrations go in** -- a colour-painted
  manga page is likely less tractable than a single illustration, which is made
  more carefully. (3) Open the tree. **The consequence of (1) is that the area
  division is already a layer and not something to recover from composited
  pixels**: the proposal's candidate 1 (quantise the composite, take connected
  components) is handicapped by exactly the 濃淡 sitting above it, and candidate
  2 (read the layer structure) becomes the main line. **The reordering comes from
  the author's process knowledge and not from a measurement, and the
  registration has to say so** -- the same category as the ベタ/トーン lead of
  2026-10-23, which corrected this file's premise and which nothing here could
  have derived. **What must not follow from it is naming the base layer**: the
  condition has to be measured and registered, and the exceptions the author
  mentions are part of what gets measured.
  **Stock, counted from the delivered census** (`inbox/work_color_mode.csv`,
  289 works at 5 pages each): **28 all-colour works over 233 pages**, 47 mixed
  over 1,566, 214 all-monochrome; the koma corpus already holds **427 panels
  over 6 candidate works**, `2022b_hyoui_color` the largest at 190. The largest
  all-colour items are `2017 illustration` 79p, `2022b hyoui_color` 45p,
  `2019 fringe3` 24p, `2018 skima10` 17p. **A rough 468 colour pages inside the
  47 mixed works is an estimate and not a measurement** (each work's sampled
  ratio times its page count).
  **Verified at birth rather than assumed**: `data/` is ignored in this tree
  (`check-ignore` passes; the branch was cut after `f7372e6`), and the inherited
  `doc/CURRENT.md` was replaced with a pointer **before any session read it** --
  the first tree opened since check 8 existed.
  **First job is the instrument check, not a result**: page-level selection
  (work-level classification is a closed door -- 47 mixed works, `fringe2` falls
  to the monochrome side 2/5), **calibrating the saturation threshold** (the
  "saturation 24 over 2% of pixels" rule is uncalibrated, its only ground being
  that JPEG bleed reaches about saturation 10), and identifying the base layer.
  Its null is **taken fresh in this pool** -- Track J/L's region-F1 floor
  (`blank` 0.0084) is not carried in. The layer inventory it needs from the
  extraction side **queues behind Track L's 2,963 layers**. No GPU at this stage.
  **The monochrome side of the author's answer is asymmetric, and candidate 2
  has to carry that** (confirmed 2026-10-24 by L9, after L put the question to
  the author): the base-plus-濃淡 construction is the main line in colour, but in
  monochrome it holds **for backgrounds only** -- 3D-rendered backgrounds with
  line art clipped over a base area, and 背面処理トーン placement -- and
  character finishing done that way is ほんの僅か. So "colour and monochrome pose
  the same structural question" is true of backgrounds and false of characters,
  and M's registration says which it is testing.
  **Track L has offered the shape of its reconcile for the third of those three
  checks** (identifying the base layer): `tools/evaluation/l8_inventory_reconcile.py`
  takes the delivery side's selection rule, lays it back over the full census and
  turns the result into a known cell, which is the same move M needs once its
  own full layer inventory arrives. **It is a shape, not a number** -- nothing of
  L's pool transfers. The foundation calls L when M reaches that stage.
  **M is working and has already run its own correspondence** (notified to the
  foundation rather than left hidden, `note_m_request_sent_20261024.md`): four
  requests to the extraction side at 08:17, pushed by SCP, **landing confirmed by
  matching SHA256 on both sides**, explicitly queued behind Track L's export, and
  its registration is `doc/track_m_registration_20261024.md` in its own tree.
  The extraction side has accepted them and returned **two things that change M's
  instrument check before any measurement**: (1) **layer colour is readable** --
  a colour layer's block is 5 bytes per pixel, the leading 256x256 alpha and then
  interleaved B,G,R,x, and alpha x colour rebuilds CSP's own preview; confirmed on
  **one layer of one page** (`2018 skima10 page0073`) and to be tested across the
  8-page pilot. **Monochrome layers share the 5-byte form and only the alpha plane
  was ever used, so no existing delivery's numbers change.** (2) **there is no
  native composited colour page in the file** -- `.clip` carries only
  `CanvasPreview`, and **its reduction differs by work** (1/4 for ako6/ako7/gakuen,
  **1/2** for `skima10 page0073`). Building a native composite means compositing
  every layer with opacity, blend mode, clipping and folders -- the work Track L
  withdrew as #2 for being heavy and producing plausible wrong pictures. The pilot
  will carry the preview at its own size with the reduction in the manifest and
  **not offered as a substitute for native**.
  **2026-10-25, the foundation paid the band check Track M had owed since the
  previous day's eleven bulletins, and the first thing it changed is what may be
  quoted.** The three p-values reproduce exactly from M's own win/loss rows
  (0.1250 / 0.6875 / 0.2188) **but the design is weaker than "n=8" suggests**:
  ties leave the sign test, so the effective n is **7 / 6 / 6**, and at n=6
  nothing but **6-0** passes two-sided alpha 0.05. **"The bar is not met" on the
  registered set is therefore the design's near-certain output, not evidence that
  A0 is at chance** -- M's own reading, confirmed. What carries the conclusion is
  the effect size, which has no power problem: **A0 over-segments by one to three
  orders of magnitude** (precision 0.002-0.068, region-count ratio 4.9-1,156x,
  recall 0.192 / 0.239 against the corrected standard of 183 bases -> 823
  components -> 801 regions). **The p-values are not the claim.** Rulings on the
  three questions: **(1) keeping the five of eight rows whose coverage ratio falls
  outside [0.9, 1.1] is correct** -- dropping them leaves n=3, which cannot pass
  any sign test, so "exclude" and "delete the test" are the same option, and the
  ratio also correlates with the standard's completeness -- **but a window five of
  eight rows fail is not a sanity window**; it is printed as a column, and the
  denominator it now uses must be stated, T-M3 having made that quantity 1.0000 on
  seven of eight pages. **(2) A page whose standard is empty (`page0073`) is kept
  in the registered set as registered, and excluded from the test set by a rule
  declared in advance**, because the sign of such a row is fixed by the `0/0`
  convention -- `blank` is 1.0 by definition and any arm that proposes a region is
  0 -- which is T-M4's shape again, a null winning by convention rather than by
  material. **The correction is safe to make because it changes nothing**: if that
  page is N1's single loss, N1 becomes 6-0 at p=0.03125 and passes, **while N2
  stays at 0.6875 or 0.375 and the bar, which demands all three, fails either
  way** (M is asked to print which row it was rather than let the foundation
  assume). **(3) The test set's size is now a design input**: tolerating one loss
  needs an effective **n >= 10**, tolerating a quarter needs **n >= 20**, ties
  excluded -- and the registered set produced 2 ties in 8, so the page count must
  be budgeted above the effective n. **Written before the request-4 rasters are
  ordered**, since adding pages after reading the result is widening the exposure.

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
  2026-09-30, closed 2026-10-04** (`7c2f470`, on a clean negative; the closure
  is read out further down this entry). Read `doc/approaches_not_taken.md`
  for why it opened. Does an
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

- `../lineart-solid-fill` (branch `solid-fill`, **Track J**) -- **CLOSED
  2026-10-23 on a clean negative; nobody is sitting in the tree.** The verdict
  and its anchors are in Next Actions item 5, and the registered headline is
  **"no arm predicts fill in this formulation"** -- not "the non-learned route
  suffices". **Found 2026-10-24: this entry had carried no status word at all
  while item 5 and the Active Goal both said closed**, so the ledger disagreed
  with itself and the audit read the track as active. Opening history, kept:
  **opened 2026-10-09 by user decision**, reopening Next Actions item 5 after the GPU
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
    not a measurement.** (**0.2518 on the cleanly-separated stratum** -- Track L's
    L7, 2026-10-24, so this null value is population-bound and more than doubles
    there.) `all_black` scores micro IoU 0.0939 / 0.1201 --
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
  **opened 2026-10-16, dormant since 2026-10-19**: the tree closed at `25e4dec`
  (pushed, working tree clean). **Its restart condition is met and it stays
  dormant anyway**: `face_groups.jsonl` has grown past the state K was waiting on
  -- 518 rows / 402 orders at 2026-10-23, covering K3's frame of 40 new faces and
  40 new marks -- and K is quiet by the user's 2026-10-19 decision, that it does
  not have to happen now. Track H's notice explicitly does not ask for it back,
  so restarting K is a question of whether it is worth a session, not of whether
  the input exists. **Its second axis opened and closed the same day** (K19 landed
  on the registered "the axis dies" branch) and the letter L reservation is
  released: no `stroke-bridging` branch and no `../lineart-stroke-bridging`
  worktree was ever created.
  **The 1,078-line working record is `doc/track_k_record.md`** -- both axes, the
  stopped instruments, the author's judgments, the cost arithmetic. Why it
  exists: across six registrations K **never moved a bar**, and four of the six
  would have passed on a bar moved by less than a hundredth (K1 1.487 against
  1.5, K4 1.8435 against 2.0, K19's CP95 upper **0.4955** against 0.50, K5
  0.5154 against 0.50 just inside, K7 stopped on its band, K2-C exited), so the
  near-misses are the content.
  - **Where Track K stands, 2026-10-19 -- the track's own summary, written to be
    read instead of its bullets.** *Established*: the connectivity error has two
    directions and **only over-merge is markable** (under-connection has no mark
    and is the binding one -- the calibrated chainer joins **7,513 of 40,212 arms
    (18.7%)** in the line art, 12.5% in the rough); the author's marks
    concentrate on tokens passing a collapsed bridge (**PAF 0.4855, CI [0.3245,
    0.6400]**, confirmed on ten held-out works) and that mechanism is **not
    style-specific**; fill adjacency is a second, weaker, roughly independent
    factor; **Track F's docstring is quantitatively wrong** (53.4% of bridges
    survive); K4a's 21.2% is fills.
    *Dead*: box B's gray axis, the continuity mechanism, the hysteresis bridge as
    an explanation of the marks, and **any repair keyed on collapsed-bridge
    passage** (precision ceiling ~4% against the >50% a cost bar needs).
    *Three statements of what must not be done*, all from labels: unconditional
    loosening (refuted by the author's 40 judgments), retuning the collapse rule
    to exploit the confirmed mechanism, and **a rule that closes free ends by gap
    length**.
    *Left*: a repair that does not decide -- **carrying both tokenisations
    downstream** -- unmeasured and unregistered, and it changes Track F's output
    format *and* the model's input, so it was never in K's hands (see Next
    Actions item 9); and a precision-raising feature, which needs new marks (K3)
    and must clear a 13x specification written before any search begins.
  - **The instrument it leaves for the other tracks**: the author-judgment web
    UI, `tools/web/judge.html` plus `tools/stroke/k22_tiles.py` in K's tree.
    **Swap the manifest and the tiles and it works for any multiple-choice
    question put to the author**; answers land in the artifact's shared store and
    are read with `ArtifactData`, which removes the hand-off that caused K10's
    `uid` mix-up. 52 items took 2.5 minutes. **Routed to Tracks H and J on
    2026-10-19.**
  - **The discipline cost something here and it is worth remembering how.** The
    answers took 152 seconds for 52 items and **Track K refused to use that for
    pass or fail because no speed band was registered**; the evidence for
    judgement quality is a 6/6 cross-check against items answered slowly the
    previous session. Before unblinding K19, the track added the column this
    file's table lacked: at a true precision of 0.625 the probability of reaching
    "the axis dies" is **0.000** -- *this paper cannot kill the axis*. **It killed
    it**, so the n=8 point estimate the proposal's headline had been written on
    was noise. And the one correction K returned to this ledger is the **0.5px
    endpoint retraction** (the 10.8px here was wrong).
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

**The cross-hatch cause, and fifteen lessons that apply project-wide** (lessons 3-4 added
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
   **A third axis, added 2026-10-24 from the user, and it is the one this
   project kept crossing without noticing: the generation of the workflow is a
   pool.** The author's production process shifted over the years, so the layer
   structure shifts with it, and **every assumption that broke on 2026-10-23/24
   was an assumption about how the author worked** -- not about what a drawing
   depicts. The list, as Track L assembled it: *a black layer is fill* (it is a
   function of the generation, the user's own correction); *a `beta` name gives
   a clean standard* (it collapses in `ako7`, 2 layers over 1.1%); *line layers
   are anti-aliased* (this author's are pure black, bam 1.0); *a name identifies
   a family* (`線画` heads both the matching and the mismatching list of L10's
   KC2, so the name does not determine the family **inside one page**, let alone
   across generations); *a base area fails
   the colour test for not being black* (it passes; `preview_dark` is what
   excludes it); *the area floor is why areas fall out of the standard*
   (family-dependent, lesson 15). **So `ako6` and `ako7` do not go on one row**,
   for the same reason `lineart` and `housei` do not, and **no new rule is needed
   -- what was missing was recognising the generation as a pool.** The **16
   revision pairs** found in the 65-page pool the same day are the same thing one
   level down: two states of one page are two pools on one row, and the per-page
   rate reads about a quarter heavy because of it.
   **And the goal is not fewer surprises, it is cheaper ones** -- Track L's
   formulation, adopted. The workflow really does shift and this project only
   sees it from outside, so aiming at an instrument that cannot be surprised
   bankrupts; a universal instrument is not the objective and would not pay for
   itself. The price was paid twice on record and the two prices differ by one
   thing: **L7's 71.3% false-positive rate** landed in the ledger first and had
   comparisons built on it, so the retraction became a pass over every store;
   **L8's P3** was pre-registered, so the mechanism died, the count lived, and
   the correction was one line. **The difference is only whether the mechanism
   was registered as a prediction.**
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
15. **A mechanism offered with a number is not part of the measurement, and a
    plausible mechanism is the easiest thing in the world to agree with.** Added
    2026-10-24, from the same mistake made twice in one morning by two sessions
    that did not know of each other's version. The extraction side reported 128
    layers that fail its ベタ colour signature while being dark on the page,
    with `black_among_drawn` median **0.823**, and offered the reason: just
    under the 0.95 cut, where **an anti-aliased line layer carrying ベタ** would
    fall. This file copied the count and the reason together at 07:40. Track L,
    independently, had **pre-registered the same mechanism as P3** -- that bam
    would concentrate in 0.5-0.95 for exactly that population. Both were wrong
    the same way: measured, the author's delivered ベタ sits at bam median
    **0.9986** with 100% at or above 0.95, the dark layers the **area floor**
    cut are blacker still (1,414 layers, median **1.0000**, 87% at or above),
    and **this author's line layers carry no anti-aliasing at all** (`線画 skin`
    and `線画 hair` both 1.0). P3 came back 1,239 of 1,551 above the cut.
    **The count survived and the explanation did not**: the 128 are real, and
    what removes areas from the standard **in this family** is the area floor
    (1,207 of 1,551 unseen dark layers) and not the colour test (63).
    **That last clause was generalised too far for one day and L9 corrected it:
    there is no single reason an area falls out of the standard.** The background
    base layers are nearly pure black, pass the colour test and are large, and
    what removes them is **`preview_dark`** -- a property of the finished page
    and not of the layer. Different family, different axis. The two stand together
    here at Track L's own suggestion, because one of them looks like a slip and
    the pair shows what it is: a mechanism arrives already feeling measured,
    and the only defence is to register it as a prediction -- which is what made
    L's version visible as a miss instead of invisible as an assumption.

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
   **The reverse side, 2026-10-24: a number can carry its construction and still
   fail to communicate.** Track L needed three exchanges to convey that 40-60%
   of the production fill is absent from this project's rasters. The construction
   *was* attached -- *"the raster lacks 46.4% of the fill"* -- and **that sentence
   was itself the problem**, because "lacks" and "discarded" describe an act on
   one object where the truth is **two objects with different contents** (the
   three-files statement in the traps). **So discipline 1 is necessary and not
   sufficient: the construction has to be stated in a form that cannot be read
   as a different claim.** The repair was to replace the wording everywhere
   rather than annotate it, because an annotated phrase keeps being quoted.
   **A fifth, 2026-10-23, where what travelled was a *corpus* rather than a
   number.** The foundation told the extraction side to trial `ako5` and
   `housei` because *"every fill figure on record sits on those pools"* -- true
   of the 480px tile figures, and not of the koma panel corpus Track J actually
   used (5,018 panels, 57 works, 2015-2024). **It inferred the instrument's
   corpus from the corpus of the figures it was quoting**, and the pick was the
   worst available: `ako5`'s fill is not a separate layer at all. So the rule
   covers more than numbers -- **a population is quotable only with the
   measurement it was the population of**, and "the pools our figures sit on" is
   two different sets depending on which figures.
   **The verdict-word sibling, added 2026-10-23 from Track J's C3 reading.**
   A conclusion of the form *"the simple thing suffices"* -- no learned model is
   needed, the frequency table is enough, the preprocessor is the baseline --
   **describes two materially different situations and the registered wording
   cannot tell them apart**. Form A: the arms work and the simple one is as
   good, which is actionable. Form B: **every arm including the trivial ones
   sits in one band, so the simple one ties by default**, which says nothing
   about models and everything about the formulation. 2026-10-23 was form B.
   **So any such verdict carries the trivial arm's score in the same sentence,
   and when the best arm does not clear the trivial arm by the registered
   margin, the verdict is "nothing in this formulation does the task" and not
   "no model is needed".** Track J proposed this against its own verdict and the
   foundation approved it **for every track**, because lesson 14 made a
   no-training control mandatory project-wide and so gave every track the same
   exposure. **Why it is legitimate after the fact**: it is a *tightening*.
   Moving a bar after seeing results is the sin because it lets something pass
   that would not have; this restricts what may be claimed and lets nothing
   through. **The asymmetry is the test, not the timing.**
2. **Every new measurement includes a cell whose value is already known, and if
   it does not reproduce, nothing else in the run is read.** Already the practice
   that caught the most: Track B verified its cs=1.0 column against historical
   values before trusting an eleven-model table; Track D reproduced 0.2514 as
   0.2513 before believing its snapshots; the void probe's conditioning-map cell
   matched Track C's 0.3164, which is how we know its plumbing was innocent and
   only its design was wrong.
   **The sibling at registration time, added 2026-10-23**: every quantity in a
   pass criterion names either a value already on record or the file that will
   produce it. **A name that names another name is where this fails** -- see the
   `NC4` trap below. Discipline 2 binds the instrument before a result is read;
   this binds the quantity before the work starts, and it is checkable the
   moment a criterion is written down.
   **The path sibling, added 2026-10-24 from Track M: a known cell belongs to a
   *path*, not to a measurement.** M's cell that feeds the standard back in as a
   prediction **only ever runs at native scale (s=1)**, so the enlargement path
   carried no cell at all, and without one **"the arm scores low" cannot be told
   apart from "the arm is misregistered"**. It added IC-M7 (take its own enlarged
   output as the standard, expect 1.0). One cell per measurement satisfies the
   rule as written and still leaves a limb unchecked.
   **The generation sibling, added 2026-10-24 on Track L's proposal: a run that
   crosses generations carries a known cell from each generation it crosses.**
   One known cell satisfies the rule as written and **cannot detect a generation
   shift if it comes from the same generation as the rest of the run** -- which
   is precisely the shift that broke five assumptions in two days (lesson 6).
   **It has already worked once and was never written down**: L5 kept an `ako6`
   cell in the run while turning to `ako7`, and that is how `ako7`'s difference
   became visible instead of becoming a result. The rule is the practice, not a
   new demand.
   **The naming sibling, added 2026-10-25 night after three known cells failed
   in one evening, each at a different level: a known cell's name states what is
   compared, in which direction, and at what granularity.** T97 was the run
   level (a cell computed inside one run cannot see per-run drift), T100 the rule
   level (a cluster decided by the name cannot test a rule about the name), T101
   the direction and granularity level -- the foundation's "two instruments agree
   layer for layer" was a containment, and Track L's own `KC11`, named *our
   matcher agrees with `region_metrics.match`*, compared **only the three counts**
   `matched`/`n_pred`/`n_gt` and never which prediction paired with which truth.
   A matcher that gets the counts right and the pairs wrong passes it. **L's own
   `KC9` was honest because its name carries the `⊆`.** So the direction and
   the granularity belong in the name, where they are checkable before the cell
   runs. **By the end of that evening the list of levels was five, and it is
   the most reusable thing the day produced**: the **run** (T97 -- a bar
   computed inside one run cannot see per-run drift), the **rule** (T100 -- a
   cluster decided by the name cannot test a rule about the name), the
   **direction** and the **granularity** (T101 -- a containment read as an
   agreement, three counts read as a pairing, a sum read as a per-panel
   agreement), **whether the cell runs at all** (T102 -- `KC7`/`KC9` read a
   `truth4` image that is nowhere on either disk, and `l13_known_cells.py`
   raised `KeyError` from `os.environ[...]` at import, so the caller could not
   tell *no check* from *no environment*; it now prints the reason and FAILs),
   and **whether the cell's own answer is right** (T103 -- the `ProjectName`
   anchor's known cell was the `superset` decisions, which the author then
   refuted 2/2). **L's constants survived the missing image at a finer
   granularity than the cells had used** -- `page_truth.csv` 65 rows per page,
   `per_panel.csv` 248 rows per panel -- so L12's re-run is anchored page by
   page and panel by panel rather than by a total, and the method was never
   lost (stage 1 of `l12_panel_regions.py` rebuilds the image).
   **And L's repair carries a second rule: when the instrument does not
   expose the intermediate** -- `region_metrics.match` returns counts, not pairs
   -- **re-implementing it in order to compare is comparing with your own copy.**
   L anchored on four pairings derived by hand from the circles' radii
   (`gt1↔pred1` IoU 0.8863, `gt3↔pred4` 0.9506, `gt2` unreachable at 0.23 and
   below `min_px`, `gt4` overlapping nothing), which is an anchor because it
   exists without the instrument. Both legs now pass (`1116143`).
   **The second sibling, from the 2026-10-23/24 exchange: publish in a form that
   can be measured back.** Eight corrections passed between the foundation, the
   extraction session and Track L in one day, and **almost none was caught before
   publishing** -- the extraction side counts three of its own deliverables
   corrected *after or just before* release and none before; the foundation
   stopped exactly one table before it went out, and only because the other side
   had written its instrument's design out in prose. **So "check before you
   publish" is not what worked.** What worked was that every party shipped
   **statistics and not verdicts, disclosed how it counted, and put the limits in
   the same message** -- which let the other side recompute and disagree. The
   asymmetry that made it work: **one side held the whole population and the
   other held the construction, so each could see the other's error and neither
   could see its own.** Prefer the form that invites the recomputation over the
   hope of catching it alone.
   **And the case where no known value exists, Track J's wording, 2026-10-23**:
   **a newly built control, lacking a known value, must carry an identity
   inside itself.** It produced two while building `NC4`: `ctx_of` asserted
   byte-equal to the trainer's own context channel, and the derangement
   asserted to be a permutation with zero fixed points. The foundation's notice
   asked for the two anchors and had no rule for this; J supplied it.
   **The instrument/interpretation split, added 2026-10-24 on Track L's
   proposal, and it is where the damage actually was.** Self-checks exist for
   instruments and **nothing checks the interpretation**, yet every collapse of
   2026-10-23/24 was on the interpretation side: `ic_form` (a disc at r=20 scores
   1.0000, a 3px line 0.0000), `ic_geom` (IoU 1.000 at zero offset, 0.37-0.72 at
   +2px) and `anchor_blank` (0.0084 reproduced) have **never** failed, and the
   radius-4 opening got nothing wrong -- what was wrong was the sentence *"what
   the opening leaves is ベタ"*. **So a tool keeps measurement and verdict in
   separate functions, and the verdict is registered as a prediction** (lesson 15
   says register the mechanism; this says where it lives in the code, which is
   the half that was never registered).
   **What this does not reach, stated plainly because pre-registration cannot
   cover it**: *what the standard never saw.* No prediction can be written about
   something invisible, and L7 was exactly that case. **The only defence there is
   not to take someone else's selection as the standard**, and on 2026-10-24 that
   defence cost 733MB of native layers. Cheap this time by luck -- but a problem
   whose price is a known quantity of disk is a tractable one.
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
   **And it cannot stand in for the user on an outward-facing action, which
   Track L had to point out on 2026-10-23.** The foundation had told a track not
   to push, for a reason of its own, and later wrote *"the hold is lifted, you
   may push"*. **Track L did not push on that**: it waited for the user, on the
   grounds that **approval for an action that leaves this machine is the user's
   and the foundation cannot proxy it.** It was right, and the wording here was
   wrong. **The hold had two components and they were conflated** -- the
   foundation may lift the condition *it* imposed and say so, but **lifting a
   condition is not granting an approval.** The same asymmetry applies to any
   peer session: a release from the party that imposed a hold removes that hold
   and nothing else.
6. **On what a picture depicts, the author decides.** Added 2026-10-04 from
   Track H. A session read that track's verified closed-eye seed as "an open
   mouth with teeth" off a 0.3x montage, wrote the retraction into the work log
   and propagated it into the findings; the user, who drew the page, restored
   it, and the full-resolution crop agrees with the user. Claude's reading of a
   drawing is a hypothesis for the author to confirm, never grounds to overturn
   their judgement -- and when a judgement is in doubt, pull the
   full-resolution crop instead of arguing from the montage. The cost here was
   not the misreading but the retraction built on top of it.
   **A branch of this rule that nothing here anticipated, 2026-10-23.** Asked
   whether the black mass in a tone folder's `black` layer counts as fill, the
   author answered **not with a verdict but with an instruction to measure**: the
   name does not settle it, look at the statistics. **So the author's authority
   can be exercised over the *method of deciding* rather than over the answer**
   -- and this rule must not be read as "wait for the author to say which", since
   sometimes the author's answer is "that is a measurement". Track L's reading,
   adopted: **what measurement could reach was only "which one ako6 is"**, while
   the part only the author holds -- that separation and fusion **go back and
   forth with changes in the workflow** -- did come from the author and could not
   have come from anywhere else. **Measurement answered the instance; the author
   answered the structure.**
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
   **A record hole, filled 2026-10-25: this project's own measurement
   discipline was exported as a public plugin on 2026-10-10/11 and this file
   never said so.** `kuromook/moat` (`/home/sh1/deepl/moat`) holds the general
   form of disciplines 1-3 and their relatives as five skills -- `moat`,
   `moat-register`, `moat-controls`, `moat-audit`, `moat-retract` -- named after
   the user's own reading on the day the minimality skill was declined (*the
   detours have been paying for the outer moat*). The user raised it when the
   foundation proposed writing a **new** cleansing skill: the proposal was the
   wrong shape, because a second home for one discipline is the duplication this
   file's own opening warns about. **Installed user-scoped the same day**
   (`~/.claude`, so it reaches every tree regardless of when the tree was cut --
   the `.claude/skills/` route would have reproduced T84/T92 exactly), measured
   cost **~425 tok always-on** with bodies loaded on invocation, which retires
   the third objection above. **It does not retire the foundation's band
   checks**: a skill can approve its own design, and L17's two design errors (a
   floor written as a verdict gate, a bar the control also satisfied) were found
   by the outside eye. **Discipline 9 was deliberately NOT merged into it**, on
   the user's objection that selection rules look like instrument design and may
   instead be a property of this material -- the three rules, a fourth way a
   control is born useless, and four abstracted incidents live on the branch
   `provisional-selection`, with the condition that would kill them written into
   the section: *if the whole cost was the upstream rule being undocumented, the
   rule is "get the upstream rule in writing" and it is one line, not three*.
   **And the installed plugin is the working tree itself** (`installLocation` is
   that path), so a checkout in `/home/sh1/deepl/moat` decides which rules are
   in force -- the foundation left it on `main`. T92's family, one level up.
   **§3 of `CLAUDE.md` was compressed the same day, after the install** -- that
   order, because the reverse leaves a pointer with no target in any tree
   without the plugin. The general rules are now one pointer to the installed
   skills, and what stays is only what is this material's: the preprocessor
   baseline, a generation being a pool, the author deciding what a picture
   depicts, discipline 9, and the two riders `moat` does not carry (the
   "configuration" includes a property of **the output itself**, and the known
   cell is **one per generation**). **The items are now numbered as this ledger
   numbers them**: the old §3 listed selection as its item 7 while every
   citation in the project calls it 規律9, so the file a new session reads
   first disagreed with the file it then cites. 151 -> 150 lines.
   **Discipline 9 states in the file that it is not in the installed `main`**,
   so no reader can take it on the skill's authority. **The renumbering had a
   cost outside the file**: the cleansing notice already distributed to five
   trees named its own 正本 as "§3 の規律7", so aligning the numbers aged
   every letter that cited one. Repaired the same sitting (discipline 3) by
   **appending a dated correction and redistributing**, not by rewriting the
   body -- a sent letter is a record (T92).
8. **Conformal prediction is adopted as a component in one named place, not as
   a programme.** Raised by the user 2026-10-23 as a general method; run
   through the same test as item 7 -- a general method is adopted only where it
   answers damage this project has actually taken. It passes in exactly one
   place: **deciding which strokes a pre-annotation pass may fill in for the
   author** (Next Actions item 12). It turns "group recall 0.351" into a
   statement about cost under an error budget the author sets, and it comes
   with a **trivial arm built in** -- emitting all five parts always covers and
   saves no work -- so the quantity to register is the share of singleton sets
   at a fixed alpha, against that arm. Three pitfalls, each of a kind this file
   already records: **(a)** per-part difficulty is very uneven (eye 0.71/0.79
   against ear 0.52/0.38), so it must be **class-conditional (Mondrian)** or
   the hard parts are absorbed by the easy ones; **(b)** exchangeability is at
   the **face**, not the stroke -- a face's strokes arrive as a dependent
   bundle, and calibrating per stroke **overstates coverage**, which is this
   file's recurring instrument-measures-itself error; **(c)** the series
   holdout **deliberately** breaks exchangeability, so the guarantee does not
   transfer -- and the size of the coverage shortfall there is a free
   measurement of how artist-specific the rule is. **Premature for the fills**:
   per-pixel prediction sets are the wrong unit, and the trap above about
   `preproc@budget` is what a per-pixel score does when the unit is a region;
   after the region unit exists, risk control on a region-level loss is the
   right form. **The condition on adopting it**: a coverage figure is quoted
   with its calibration population and its conditioning, exactly like any other
   number here -- "90% coverage" is marginal, is not 90% for ears, and is not
   90% for a new artist. Discipline 1 was written about numbers of this shape.

9. **A selection is a measurement, and what it silently drops is part of the
   result.** **Recorded here as adopted; how it is distributed to the tracks is
   the user's open decision as of 2026-10-24 night, pending Track L's measurement
   of the author's occlusion hypothesis** -- that measurement *is* a step of the
   cleansing workstream, and its outcome could reshape 9-2's L-side example (the
   789 pale layers). The rule itself rests on two independent witnesses, so the
   pending result bears on the example and not on the rule. The notice sits in
   the four trees' inboxes marked provisional.
   **The result came back the same night and it confirms the example rather than
   reshaping it**: with the real z-order, the gains the substitute barrier had
   produced (**+0.468 / +0.551**) do not reproduce **at all** (+0.000 to +0.008),
   so the gain *was* the borrowed constant's circularity. **The trigger the user
   set is met, and on 2026-10-25 the user ordered the distribution**, so the
   notice went out as final to the five live trees (`colour-manuscripts`,
   `face-words`, `fill-region`, `panel-generation`, `stroke-selection`) with its
   provisional header replaced by the confirming numbers; **CLAUDE.md section 3's
   three-line discipline 7 is the authority and the notice is its provenance and
   case list.** The same day the discipline reached four more places, all of them
   the same shape: **T90**, a control arm measured on a smaller population than
   its arm (9-1 applied to the null); the two-layer filter in L17, where dropping
   the filter raises the null to 0.5646 / 0.6276 and ako7's control alone clears
   the bar, so **a selection left uncounted is how a test becomes a tautology**;
   **T91**, a channel the rules had no slot for, leaving a delivery whose request
   exists on neither side; **T92**, a replicated `CLAUDE.md` still naming a
   corrected address in 10 of 11 trees; and **T93**, a check whose predicate is a
   quoted string flagging its own remedy.
   Added 2026-10-24 from the user's observation that *the instrument
   work goes fine and the data keeps betraying it*, and from **Track L and Track
   M arriving at the same failure independently, neither knowing the other's
   version**: a borrowed brightness or saturation constant is not merely
   uncalibrated, it **deletes a whole kind of material in silence**, and what is
   deleted appears in no output, so no result can reveal it. **M**: the `24` in
   "saturation 24 over 2% of pixels" was hiding **82 pale pages** whose
   `color_ratio` is exactly 0 while they carry pixels at saturation 8-23 (one
   page is 0.999091 above saturation 8 at `color_ratio` 0.0), **across 32 works,
   13 of which the old bar called all-monochrome**; the population went 790 ->
   **872**. **L**: `preview_dark >= 0.5` dropped **789 white or pale layers**
   (speech balloons, white-painted foreground objects) plus 68 `/text` layers
   **out of the barrier** -- and the constant itself was never wrong, it is
   **correct for choosing what enters the author's ベタ** and was **borrowed for
   building the barrier**, where *what stops a flood is not necessarily dark*.
   **Track L's reading, adopted: cleansing is not hygiene around the research, it
   is the research** -- the whole of its C1-C6 inventory is in the ledger above.
   Three rules, each mechanical rather than a caution, in the shape of discipline
   4:
   **9-1. A selection prints what it dropped, counted by reason -- including the
   `continue` inside the instrument.** L12 already did this at the outer level
   ("1,981 layers passed, the rejects printed by reason") and **that print is the
   only reason C3 was found**. M's `clip_bases` **skipped 24 chains whose base
   was `None` without printing anything**, and that silence became the ground for
   an instrument rule. **The paired form, M's wording: when two inventories of
   the same thing are held, itemise the row-count difference before using
   either** -- M had computed 1,005 - 706 = 299 = 173 folders + 126 with nothing
   drawn **and had not read what the 173 were**, which cost it 167 -> **183**
   bases and 504 -> **823** components. Counting a difference is not reading it.
   **9-2. A constant carries the role it was calibrated for, and is never
   borrowed across roles.** The same `preview_dark >= 0.5` was right for "enters
   the ground truth" and wrong for "enters the barrier". **When the role changes
   the constant is re-derived, and the role travels with the number** exactly as
   discipline 1 makes a configuration travel with a figure -- a constant inside
   an instrument is a quoted number too. **And when a constant chosen for the
   instrument's stability conflicts with the standard's completeness, the
   standard wins**: M's area floor at 1e-4 had the smallest swing in component
   count (1.49x) and **silently removed 25% of the silhouette layers the author
   drew**, so it was re-taken at 1e-5 with the cost (2.00x / 4.37x) written down.
   **9-3. The null comes before the bar, and a high null indicts the standard
   first.** L15 set F's bar at 0.50 **without asking for the null, which was
   0.72-0.74** -- a bar below chance, met by default; CLAUDE.md already assigns
   the null and the band to the foundation and L drew its own. **M's converse,
   adopted**: when a null scores *high*, suspect the standard before the arm --
   its area floor left label 0 in, the margin entered the standard as one huge
   region, and **N2 (the whole page) matched it at 0.6667/1.0000**. **A null is a
   check on the standard, not only a floor for the arm.**
   **What none of the three reaches**, and it is the same limit discipline 2's
   instrument/interpretation sibling names: a rule can be correct and still be
   **correct only while its material is complete**. M's z-order rule (the
   component on top wins) is right in CSP and broken under T-M3, because
   *what the composite shows does not exist in the export* -- 823 components
   became 355 regions and `skima12H` went from 148 to **1**. One trap surfaced in
   two places in the same instrument, which is why 9-1 counts rather than warns.

**When you suspect the knowledge has already muddled**, stop producing results
and run these in order: re-run the anchors (separates a broken tool from a broken
understanding), reconcile the index against the filesystem, then walk the numbers
this file currently quotes and check each still carries its configuration and a
traceable source -- marking the ones that do not. Resume after that, not before.

## Known Tool Traps

**The rules are here; the evidence is in `doc/tool_traps.md`**, which holds each
entry's incident, mechanism and numbers verbatim (moved 2026-10-24, when this
section was 1,768 of this file's 6,820 lines). An entry is cited as `T<n>` there.
Nothing was deleted. Read a trap's full entry before citing its numbers -- several
of them exist to record a retraction, and the retracted figure sits next to its
withdrawal.

### 対照・帰無・事前登録の設計

- **T2** A rotated chance arm's height is a property of the structure you rotate, not of the structure you measure
- **T5** A bar placed on a ratio can be refuted while the mechanism it was placed to test is actually happening
- **T7** A control the reference arm itself fails cannot discriminate
- **T8** The instrument graded its own pass condition, and leniently (caught 2026-10-18 by Track K itself, on the second run of K2-B')
- **T9** A stop condition written about the material teaches nothing about the instrument (routed 2026-10-18 to Track K)
- **T10** A one-directional fix needs a target that catches over-correction (routed 2026-10-18)
- **T12** A fix derived from the failures passes on those failures by construction
- **T13** The author of a repair is the wrong party to predict its yield (observed across three Track K registrations, 2026-10-18)
- **T14** A null value belongs to a statistic, not to a study (Track K's K7, 2026-10-18, stopped by its own control)
- **T15** A coincidence between two statistics is a degeneracy check before it is a finding (same stop)
- **T16** "Compared against everything else" can be the conservative choice, and that is checkable (ruled 2026-10-18)
- **T17** A count from a run whose stop condition failed is usable only if its predicate never touches the failed path -- and it still needs a scale (ruled 2026-10-18 on Track K's 32.4%)
- **T19** An attributable fraction and a precision answer different questions, and diverge when the exposure is common and the outcome rare (established with numbers 2026-10-19, Track K's cost sweep)
- **T22** A registration earns reserved material only by naming the decision its answer changes (ruled 2026-10-18, retiring K7's confirmatory leg)
- **T24** A PAF bar is meaningless unless the exposure is the narrowest set the repair would touch (ruled 2026-10-18 on Track K's exposure choice)
- **T25** "AFe" was naming three different quantities, and two of them were one quantity fed an RR from the wrong direction
- **T27** An anchor whose correct answer is missing from the option set tests nothing (ruled 2026-10-18 on Track K's author request)
- **T28** An anchor set identifiable by a nuisance variable passes for the wrong reason (same ruling)
- **T29** A bar on raw agreement is confounded by the answer marginal (same ruling)
- **T30** A robustness arm must not contain a repair of its own
- **T31** A diagnostic that predicts a fix's yield must run the instrument's own code path (Track K, 2026-10-18, the most useful of the day)
- **T35** A registered exit clause that names a successor *method* is a guess; name the *class* of change (the foundation's error, 2026-10-18, caught by Track K with evidence)
- **T36** Disjointness should be a property of the construction, not of a draw
- **T37** Replacing an approximation with the concept it approximated is a different kind of change from moving a number (the principle that admitted Track K's third fix, 2026-10-18)
- **T39** A set-level zero on a regression count is not a cost bar, it is a refusal to trade (Track J, 2026-10-18, found by its own measurement)
- **T40** Selecting on a zero-valued count manufactures a zero (Track J, 2026-10-18)
- **T41** A registered expected value must name the entry or the results file it came from; a date and a sample size are not a citation (Track J, 2026-10-18)
- **T43** A control that removes the artifact along with the signal cannot separate them (caught 2026-10-16 in Track K's K4a, before it was run)
- **T44** An instrument built from artificial damage cannot contain damage the material never shows (found 2026-10-16, Track K reading Track J's IC-j6)
- **T53** A negative control has to name one concrete error, not a direction.
- **T57** An arm can beat its off-by-one control and still carry no location information at all -- what it is scoring is the *amount* of ink.
- **T78** If an external standard arrives through a filter, what the filter dropped is not absent -- it is invisible.
- **T90** A control arm that silently drops the cases it cannot place is measured on a different population from the arm -- and the drop is biased along size (found 2026-10-25 by the foundation in Track L's `l15_zorder_occ`: real 625/938 regions against shuffled 608/919, every region above 0.5 of its panel absent from the control, `>0.2` 20 -> 10 in ako7). Discipline 9-1 applies to the control, not only to the arm.
- **T101** A containment is not an agreement -- measure both directions before calling two instruments consistent (2026-10-25: the foundation read L's 2,581 ink-zero layers and the ledger's 2,598 `empty` as agreeing layer for layer; L made it a known cell and found `drawn_ratio<=0 -> ink_px==0` exact at 0/4,399 exceptions but the reverse carrying 31, up to 0.194 of a page drawn with no ink). The one-sided check passes on exactly the half that holds. **The same shape at the summary level, found by L the same night in its own `KC11`**: a cell named *our matcher agrees with `region_metrics.match`* compared only the counts `matched`/`n_pred`/`n_gt`, which a matcher with the right counts and the wrong pairs passes; and since that function returns no pairs, re-implementing it to compare would compare with one's own copy, so the repair anchors on four pairings computed by hand from the circles' radii (`1116143`).
- **T104** A default supplied with `or` erases the condition's own extreme value -- `(f(x) or -1) >= 0` turned Track L's 78 layers at `preview_dark == 0.0` into -1 and dropped them, and the 15 layers that carry the finding were among those 78 (2026-10-25, found by cross-checking against the gate's own printout of 875; without that cross-check the number published would have been 797). On a threshold, 0 is the most informative value.
- **T132** A file checked against its own manifest is a transcription check, not a content check, and it passes over any error that happened before both were written. Track L's **KC55** compared each delivered mask PNG's `< 128` fraction with the manifest's `mask_ratio_ge128` and found **0 disagreements of 14** -- **on the broken first version as well as on the corrected one** (2026-10-26). It could not have caught the defect: the reader filled unsaved blocks with 0 instead of `InitColor`, and **both the PNG and the manifest row were produced from that same wrongly-filled array**, so they agreed exactly while both were wrong. What actually caught it was the extraction side opening one layer at native resolution beside the page preview. L's own statement of the repair is the rule: **write a check's reach narrower than its name** -- KC55 verifies the *polarity convention* and the transcription, and verifies nothing about whether the material is right. L also named the family correctly: this is **`has_mask`'s shape occurring inside a known cell** (T87, a column named for a check that performs none), and T129's independence rule one level in -- there two instruments shared a reader, here a check and its subject share a source array. The practical form: **a check whose two sides come from one computation measures that computation's self-consistency and nothing else**, so a known cell needs at least one input the instrument under test did not produce. **L took inventory against this the same day**, and three specific findings stand: **KC55 fails** (delivered PNG against a manifest from the same computation, which is how it passed on the broken version), **KC53 fails** (same instrument, same delivery), and KC54 is partial (manifest against `z_global`, both from the extraction side). **What does not stand is the generalization.** L's inventory covered only the cells used by the pending unrun registration, and the foundation turned it into *nearly every known cell this project has held was built from the delivery* -- **withdrawn as unmeasured**, in the same message that praised the inventory. Synthetic cells do exist elsewhere in the track (`l14:91` placing a mark inside the fill, `l14:102` building both ground truth and prediction synthetically, `l15:92` putting a barrier far away, `l15:100`, `l35:54`'s KC48). L also withdrew the scope of its own count: the `grep` that produced it counted `np.zeros`/`np.ones` per `*_known_cells.py`, and **most of those are accumulators**, not synthetic cells -- *a number that comes out in countable form is not necessarily measuring the quantity its name claims*. **The full classification is deferred rather than estimated**: L will classify every cell while implementing the synthetic-or-delivered print, and will not fill it in with a guess now. Two scope errors stacked here, L's and the foundation's, in the same shape as the morning's three-deep tone chain -- **a measurement whose scope is unstated gets generalized by its next reader**, and the repair both sessions keep arriving at is to write the population beside the number. Two consequences L drew, both adopted. The first is a distinction worth as much as the rule: **KC53 asks whether this is the same instrument as before; KC56 asks whether the instrument is right -- and until today only the former existed.** So the new KC56 uses **no delivered data at all**: a small synthetic canvas with an occluder over the region and another beside it at explicit z, requiring the ratio to be a known value on the overlap and **0.0000** on the neighbour, plus a hand-made mask pushed through the O3 path with the post-mask shape matching the hand-decided shape pixel for pixel; if it fails, none of O1/O2/O2'/O3 is read. The second is an output change: **every run prints, beside its list of known cells, whether each is synthetic or delivered** -- because *n known cells passed* says nothing without n's composition, and **n cells all built from the delivery catch no error in the material**. L applied the same shape to the 128 threshold, which is the reader's choice and not the delivery's: when a mask is applied, print how much the shape moves as the threshold varies -- **unchanged means the threshold is inert, changed means the threshold is ours**. The foundation's own half of the repair, since it entered a supplier's figure unverified four times today and L measured instead only because the foundation had written *do not adopt it, measure it yourself*: **a number taken from a supplier carries its measurer in the ledger until someone here has measured it**, which is a check rather than a resolution, and T129's lesson is that only a check stops the person who wrote the rule. On the numbers this closes: L measured v2 with its own instrument rather than adopting the supplier's figure, getting masked retention of **median 0.9880, min 0.9756, max 0.9990, 21 of 21 above 0.95** over the 21 layers that have a PNG -- so masks drop about **1.2%** of the ink, the 46-layer figure of 98.3% differs only because it is a different population (46 layers against the beta raster, not 21 against the layer PNG) and is **not treated as a disagreement**, and L34's reported figure returns to **beta loses about 11% of its ink at r=4** with the 21 layers' uncertainty too small to matter at that magnitude. The earlier interval [0.8186, 0.9332] was the worst case over *unknown* shapes; the shapes are now known to move by 1.2%, so it collapses.
- **T131** When a population is defined partly by a function and partly by its callers, any count taken outside a caller is stale **by construction**, and the fix that made the instrument right makes the bare count wrong. Track L's `gated_layers()` returns the **raw** LINE arm of **1,681** layers, while the removal of the **289** layers that also sit in the delivered beta -- the repair L itself made that morning after KC44 -- lives in the *calling* code, so the arm L18 and L36 actually score is **1,392**. Asked how many mask-bearing layers with ink fall in the arm, L queried the set directly and answered **21 of 1,681 = 1.2%**; through the caller it is **12 of 1,392 = 0.86%** (2026-10-26). In L's own words: **the instrument was fixed and the counting used the old definition** -- *今朝自分で直した漏れを、直した側の腕で数えませんでした*. The same exchange produced **five different quantities for one question, no two equal and none of them the foundation's 25**: name contains `線画` **38**, in the delivered beta **21**, raw gated arm **21** (a *different* 21), **the scored arm 12**, and the foundation's own predicate **17** -- with two unrelated 21s inside a single message, which is T127 demonstrated in one page. The rule is architectural, not clerical: **a selection must not be expressible in two places.** Either the function applies every condition and callers cannot widen it, or it returns the alternatives under explicit names (`raw` and `scored`) so that no bare count is possible -- T110's recommendation, that a rule belongs in the instrument rather than in whoever remembers it, applied to a population rather than to a gate. What the correction does not touch: L36's reading, since 1.2% and 0.86% are both under one percent and both arms face the same standard; what it changes is **which arm a sentence is about**, which is the thing a reader cannot recover from the number alone.
- **T130** A verdict's robustness is a claim about **which side of the bar**, not about the size of the perturbation -- and the breakdown point is not a sensitivity. The foundation told Track L that L34's 保留 was safe because *a median does not move under 4.23% contamination*, 21 layers of 496 whose delivered shape is the pre-mask one. **That is the breakdown point: the fraction needed to move a median arbitrarily far.** Replacing 21 values moves the median by up to twenty-one order statistics, and L measured the actual interval on `form_keep.csv`: `keep2` median **0.9641** can sit anywhere in **[0.9560, 0.9787]**, `keep4` median **0.8892** anywhere in **[0.8186, 0.9332]** -- a width of **0.1146** (2026-10-26). *The median does not move* was simply false. The verdict survives for the reason L supplied instead: **r=2's worst case 0.9560 is still above 0.95 and r=4's worst case 0.9332 is still below it, so no interval crosses the bar**, and the line arm's 1,509 layers are untouched, so *the two conditions never overlap* holds. The reported figure changes even though the verdict does not: ~~beta loses **11%** of its ink at r=4~~ becomes **beta loses 7-18%**, median 0.8892 with the stated interval, and L declined to narrow it by arguing that a mask can only shrink a shape -- *likely* is not a number, and the ordered masks will give the real one. **Third instance in one day of the foundation reaching a right conclusion by an argument that does not hold** -- the others being *ink at alpha 25 over white renders no darker than 230* for T119, where the premise was false because the PNG is alpha-encoded, and *no verdict depends on it* for T112, which was the stated reason for declining the run that now decides L18. So the rule has a procedural half: **state the worst case over the undetermined quantity and check whether it crosses the bar**, which is checkable, rather than asserting that a perturbation is too small to matter, which sounds like a measurement and is an intuition. Near T124 in family: there a condition was stated and never discharged, here a justification was asserted in place of a condition that could have been computed in one pass.
- **T129** A derived column's zero reports what the reader looked at, not what the file holds -- and two instruments agreeing is not independent confirmation when they share a reader. Image-material layers keep their picture in **`ResizableOriginalMipmap`**, and the extraction tool read only `LayerRenderMipmap`, so `drawn_ratio = 0` meant *this reader found nothing where it looked* (2026-10-26). On `gran page0016`, layer L228 `slimeH1`: the render mipmap's **728 blocks are all empty** while the original mipmap's **728 blocks all hold content -- colour, bpp 5, the full 7016x6543 canvas at alpha 255**. Across request 3's **1,819** image-material layers, **every one of the 555 reported empty or unrenderable carries content in the original mipmap** (1,264 + 555 = 1,819), so **the whole class holds pictures** and the earlier *external block id absent* account of 49 of them was true only of the render mipmap. **The foundation's claim is withdrawn, not weakened**: *資料画像は既に消されていて、ファイルの構造だけが残っている* is wrong -- the pictures are there. Note the shape of the retraction, because it is the registrable part: the claim was corrected **three times in one day, each time more weakly** -- first *no pixels*, then *no pixels at alpha 26 or above* when `drawn_ratio`'s threshold was disclosed, and only now the cause, which was never about a threshold at all but about **which table the reader opened**. A weakening that keeps a claim alive is how a wrong claim survives disclosure. And the corroboration the foundation leaned on -- *M's column agrees completely, the absence of multi-colour layers is because the picture is gone* -- was **two paths into the same reader's single blind spot**, which is why it agreed: **independence is a property of the instruments' inputs, not of their authors**. **The foundation then broke this rule within hours of writing it.** Asked to settle whether a layer mask's high values show or hide, it proposed counting a tone mask's black fraction and comparing it with `mask_cover` as an external standard -- and `mask_cover` is produced by the same reader, so the comparison was guaranteed to agree and could not discriminate (the extraction side said so plainly). What settled it was the foundation's **earlier** proposal, which it had replaced with the cleverer one: crop to the layer's bounding box at native resolution and compare against the page preview. On `ako6 page0125 beta11` the layer is a vertical black band whose mask has saved blocks over only part of it, with the arm shape white, and the preview shows the band with the arm cut out. **The cause was the reader filling unsaved blocks with 0**, while `Offscreen.Attribute`'s `InitColor` is 0 for render mipmaps and **255 for layer masks**, so a mask that hides only where it was drawn read as a mask that hides almost everywhere. **The polarity convention was right all along** -- high means visible -- and refilled correctly the beta retained inside its mask is a median **98.3%** (min 76.8%, max 100%, all 46 at InitColor 255), so the overstatement in those 21 standard layers is about **1.7%, not 86%**, and L34's `keep4` interval of [0.8186, 0.9332] collapses back toward its point estimate. Two lessons beyond the independence rule: **a proposed ground truth must be shown not to share the instrument under test**, and **replacing a working method with a more elegant one requires re-checking the new one's premises** -- the first proposal was sound and the second was not. What stands: `L28 black` on the same page is genuinely empty, all blocks empty with no original-mipmap field at all, so the 77-layer claim divides layer by layer rather than holding or falling whole. Two live consequences: **`drawn_ratio` may not be read for `v_layer_kind = 'image_material'`**, and the delivery rule *every non-folder layer with `drawn_ratio > 0`* (T119) therefore **never delivered the 555** -- a real gap, produced by a sound rule resting on a broken measurement. **Second class, found hours later and by a different route** (T128's correction): a layer painted through a mask keeps its form **in the mask**, which `all_layers/` does not contain, so **2,072 of the 2,252 mask-bearing layers read `drawn_ratio = 0`** while their shape sits in a raster nobody delivered -- tone, gradient and fill among them. **So the mechanism has two confirmed instances in different tables and must stop being discovered one class at a time**: `drawn_ratio = 0` is not evidence of emptiness for any kind whose content lives outside `LayerRenderMipmap`, and the standing question to the extraction side is **which of the eleven `v_layer_kind` values keep content elsewhere, enumerated from the schema rather than met one by one**. **Answered the same evening, from the fields alone**: the reader misses **`mask_fill`** (form in the mask), **`image_material`** (original mipmap), **`gradation`** (only `GradationFillInfo`, and how the form is decided is **unknown**) and **`3d`** (**unknown**) -- with *unknown* written as unknown, which is the useful answer. A third-class candidate is flagged rather than buried: **`DrawUpperMipmapFor*` fields, meaning unknown, on about 2% of `raster` layers**. And one line reaches beyond the four kinds: **no mask has ever been applied, for any kind**, so a delivered PNG is always the **pre-mask** shape. The foundation counted what that costs: of the **2,963** delivered layer PNGs, **180 (6.07%)** belong to mask-bearing layers and **172** of those are not caught by the full-coverage gate, with a `drawn_ratio` median of 0.0060 -- thin, but **25 of them are line-named** (`線画 hair` 11, `線画` 10, `線画 cloth` 4), so they sit inside Track L's word-emitting LINE arm while **their delivered raster overstates what the page actually shows**. Those 180 are mostly gradients (`マンガ用グラデーション` 46, `肌グラデ` 37, `乳首グラデ` 31, `背景グラデ` 12), and **155 gradient masks exist on disk**, so for them the masked shape is recoverable rather than lost.
- **T128** A bitfield compared with `=` drops every row that has any other bit set, and when the drop is counted under a borrowed reason the count reads as diligence. `visibility` is a bitfield: the **1s bit is visibility and the 2s bit is the mask**, which the extraction side stated on 2026-10-26 while reporting that **`visibility = 1` was the main cause of Track M's selection defect -- every layer carrying a mask was being dropped**. The foundation then grepped for the same shape and found it live in Track L: `l15_zorder_occlusion.py` holds `if vi != '1': cut[f'visibility={vi} (可視でない)'] += 1; continue`. Over the 7,362-layer inventory the values distribute `{0: 140, 1: 4672, 2: 83, 3: 2459, 5: 4, 7: 4}`, so that line discards **2,690 layers of which 2,467 are visible** -- **2,459 of them visible-and-masked** -- from the occluder candidates of an *occlusion* analysis. **The foundation then named the class wrongly, and Track L measured it before using it** (2026-10-26): over the 5,544 non-folder layers, **all 1,936 whose names look like screen tone (`tone`, `トーン`, `NNL NN%`) carry the mask bit, with zero exceptions -- and 316 mask-bearing layers are not tone-named at all**: `調整` 65, `肌グラデ` 54, `マンガ用グラデーション` 52, `乳首グラデ` 39, **`べた塗り` 38**, `背景グラデ` 18, `線画 *` 29. **So the mask bit marks *painted through a mask rather than drawn as strokes* -- tone, gradient, adjustment and fill -- and the foundation's repeated "which is every tone layer" read a one-way implication backwards**: tone implies mask, mask does not imply tone. The rule that follows is T128's own applied one level up -- **name an arm after the condition that fired, not after what you believe the condition selects** -- so L's arms are *mask-bearing included* rather than *tone included*. A near-miss in the same measurement shows why the measurement was needed: L nearly reported *468 tone-named layers without a mask*, and **all 468 were folder rows** (T88's family); among layers the implication has no exceptions. Worth its own line: **40 of the dropped carry the name `べた塗り`** (38 was a miscount, `べた塗り 6` and `7` were missed), which is the material this track exists to study -- **and all 40 have `drawn_ratio = 0.0` and no layer PNG, so they could never have entered the standard; the standard is untouched.** **The consequence this entry originally drew is withdrawn.** It said the occlusion analysis had lost *tone, precisely the material that occludes*. Measured: of the **1,936** tone-named layers only **8 carry ink**, and those eight are full-canvas tone swatches at `drawn_ratio = 1.0` already caught by the full-coverage gate; of **2,252** mask-bearing layers only **180** carry ink at all. What arm O2 actually adds is **129 layers, every one a gradient** (`マンガ用グラデーション` 46, `肌グラデ` 36, `乳首グラデ` 24, `背景グラデ` 12, `フキダシ` 4, `線画` 3, three others), **no tone-named layer among them**, and thin -- `drawn_ratio` median **0.005710**. **Tone was not dropped by the equality test**; but the sentence that followed -- *it was never delivered with a shape at all* -- **is itself withdrawn an hour later**, by L, and the foundation's original conclusion comes back with a different mechanism. **The masks are delivered**: `ako6/tone/` holds **393** `_mask.png` and `ako7/tone/` **226**, `grad/` holds **85** and **70**, `fill/` holds **0** (a solid fill's form is the delivery itself), with `tone_manifest.json` and `gradient_manifest.json` carrying **`layer_id`**, so they join the inventory and `z_global` directly. Of the manifest's **1,238** items -- all inside the 65 priority pages -- **1,238 join `z_global` with no exceptions**, **619 have a mask file on disk** and the rest are unverified, `mask_cover` runs median **0.0082** / p90 0.0537 / max **0.2900**, and **none reaches coverage 0.95**, so they are shapes that could occlude. L's error was counting inside `all_layers/` and writing a claim about the whole delivery -- **the very mechanism T129 had just named, committed thirty minutes after reading the sentence that nobody knows whether there is a third instance**. Where this leaves the chain, since it is three corrections deep: **the foundation's conclusion (the occlusion analysis is missing tone) holds; the foundation's mechanism (the `visibility` equality test dropped it) was wrong, and L was right to correct it; L's replacement (no shape was delivered) was wrong too.** The live mechanism is that **the instrument reads only `all_layers/`**, so the repair is not a bit fix but reading `tone_manifest`. So the count *18% of occluder candidates were lost* is arithmetically right and substantively wrong, and the expectation that restoring them would let occlusion explain more does not hold. The bit fix stands regardless -- the printed reason was false either way. **The exact-set fact from the same sweep**: over 5,544 non-folder layers, `drawn_ratio > 0` and *PNG present* coincide on **2,963**, `drawn_ratio = 0` and *PNG absent* on **2,581**, and **no other combination occurs**. **The reason printed is false**: it says *not visible* about layers that are visible, so 規律9's drop-by-reason was honoured in form while misnaming the cause, and **a drop counted under a wrong reason is worse than an uncounted drop, because the count looks like the check working**. Two rules. **Test a bit, never a value** -- `int(v) & 1` where the question is visibility -- and a field whose distinct values exceed its apparent semantics (six values for a yes/no) **is a bitfield until proven otherwise**. And **a drop reason must be derived from the condition that fired, not written beside it**, or it will keep asserting the original guess after the guess is refuted. Note where the information actually lived: `has_mask` is True on all 2,963 layers and carries nothing (T87), while the mask bit sat in `visibility` under a different name -- **the column named for the check was empty and the check's answer was inside a column named for something else**.
- **T127** The count agreeing is not the set agreeing, and a cardinality check is exactly the check that passes. The author said of the recovered work `2017\ol`: *29ページのうち21ページ以降はなぜか同じページが入ってるので21ページ以降落としてもいい*, and the filenames run `page0002, 0005, 0010, 0031...0057` with `0036` missing, so the foundation enumerated three readings of *21st* and reported their sizes -- **9 by position in the filename ordering**, 26 by page number, 7 by position within the contiguous run. The answer came from the author's own ordering in `ol.cmc`, where page order and printed nombre agree **20/20** for the first twenty and the last nine carry nombres 5, 2, 8-14 that repeat earlier pages, giving `page0005, 0002, 0047, 0048, 0010, 0038, 0039, 0040, 0041`, which the author then confirmed on sight (2026-10-26). **That set has nine members and shares not one page with the nine the filename ordering gives.** Both the foundation and Track M had computed and reported that cardinality; had the nine been dropped on it, nine pages would have been deleted, nine kept that should have gone, and **every count in the ledger would still have reconciled** -- 29 minus 9 is 20 either way. The only instrument that could detect it was showing the author the actual pages, which is the condition the foundation had added to M's plan (*identify by signature, then confirm with the author, ten seconds*) for a different reason. Two rules. **When a set is identified, verify membership, never size** -- a size check on a set-valued claim is a null check that reports success. And **it is the near-miss that makes this registrable**: three readings were enumerated honestly and the sizes printed, and that honesty produced a number which looked like corroboration of the wrong set. Companion to T111 (a statistic that names no number until its convention does) at the level of sets rather than estimators.
- **T126** The population's boundary was inherited from the reader's format support, and nobody ever registered it. Every census in this ledger -- 7,634 pages, then 7,667, the 872 rendered, the 183 bases, the 65 priority pages -- counts `.clip` files, because `.clip` is what the extraction tool can open. **`.clip` was never a declared scope; it was a capability.** On 2026-10-26, tracing `.cmc` link targets, the extraction side found **1,986 `.lip` files** under `F:\manuscripts` (2015: 665, 2014: 613, 2016: 327, 2020: 257, 2019b: 112, 2012: 12; `2015\tomoe_dl` and `2016\tomoe_dl` hold 315 each), absent from the DB and from every request's population. **A `.lip` is a bare SQLite file** -- it begins `SQLite format 3`, where a `.clip` is a `CSFCHUNK` container holding SQLite plus pixels -- so the current reader cannot open one, **and whether it can be read at all is untested**. The immediate casualty is this entry's own neighbour: T115 recorded *the range is closed -- of 316 surveyed works, 27 are absent from the works list and 26 hold no `.clip` at all, so `2017\ol` is the only work lost this way*, and **those 26 are largely the `.lip` works**, so the closure holds only inside `.clip`. The extraction side withdrew its own phrasing (*原稿が無いわけではなかったので、あの書き方は言い過ぎでした*) and the foundation had ratified it into the ledger as closed, which is the part that belongs here. Scale: 1,986 against 7,667 means **every number in this register covers about 79% of the pages known to exist**. T125's shape one level out -- there the unit of analysis came with the instrument, here the population's outer boundary did -- and the same rule applies: **a registration must state the population's extent in terms of the material, not in terms of what the tool happens to open**, because a capability limit and a scope decision are indistinguishable once the count is printed. The first measurement is minutes, not hours: open **one** `.lip` with `sqlite3` and list its tables, which separates *same schema* from *different schema* before anyone plans a scan. **Both halves resolved the same day**: the core tables match (`Offscreen`, `Mipmap`, `MipmapInfo` identical in columns; pixels inline in `BlockData` in the same `BlockDataBeginChunk`+zlib format rather than referenced into an external chunk, so request 3's *external block id absent* failure cannot occur here), and **the author ruled 入る** on the facts Track M assembled -- 31 works, 27 distinct names, only 3 sharing a path with a `.clip` work, and **three same-page-count groups spanning years** (`tomoe_dl` 315/315, `ako4-6DL` 145/145, and a triple `akogare1-3DL`/`akogareFANBOX` at 112/112/112). So the version-cluster question now extends outside `.clip` as well. The foundation had sequenced the readability trial before the author's ruling and M asked first; M's own assessment is the right one -- *the author was made to rule on material nobody yet knew could be read* -- though with the ruling in hand readability becomes a question of procedure rather than of population. The denominator is now written as **7,667 `.clip` + 1,986 `.lip` = 9,653 files**, against which the 872-page rendered delivery is **9.0%**. **Measured the same day, and the duplicate hypothesis held**: ten random `.lip` pages all read at **0.8s each**, so a full structure scan is minutes; and of the same-name cross-year groups the foundation had flagged from their page counts, **all four -- `tomoe_dl`, `ako4-6DL`, `akogare1-3DL`, `dlgran` -- are byte-identical duplicates**, with `FANBOX` matching on **102 of 112**. **`.lip` holds 1,224 distinct sha256 among 1,986 files**, so 762 are copies and the new material is **1,224 pages, not 1,986**. The compilation reading is **undecided**: `ako4-6DL`'s page count does not match `ako4`+`ako5`+`ako6`, and counts cannot settle it. **A caution followed, and it closed the same evening**: `1,224` counts *distinct* pages while `7,667` counts *files*, so for some hours the only sayable total was **9,653 files**. **The sweep finished**: the `.clip` side holds **6,194 distinct contents among 7,667 files**, and the corpus is **7,418 distinct among 9,653** -- 6,194 + 1,224 = 7,418 exactly, so the halves now add. Against distinct pages the 872-page rendered delivery is **11.8%** rather than 9.0% of files. **And 1,473 `.clip` files are byte-identical copies of another**, which is an order of magnitude more duplication than Track M's cluster census had found (116 clusters over 248 pages, built from statistical signatures); the two are not directly comparable, but **sha256 now exists for all 7,667, so clusters can be rebuilt from exact identity instead of inferred from signatures**. The user has since directed that `.lip` be included, so the scope is settled by the user rather than inferred.
- **T125** The level of analysis was never a registered choice; it came with the instrument, and every registration since has specified arms, bars, strata and seeds **on top of a unit nobody chose**. The author settled it on 2026-10-26: *割れているなら別の面 基本単位として / これは線の単位を1本と数えるのと同じ意味での単位 / **複数の面のまとまりとなってはじめて意味を持たせられるものも多い***. So a 面 is **a unit for counting, not a unit of meaning** -- the author draws the parallel to counting one stroke as one line, which is Track K's result in the author's own words -- and meaning lives in a **まとまり**, a grouping of several 面. Track L's whole fill line, and `l18_propagate` in particular, is built at the 面 level: one word per region, `seed_of` returning a single layer label per region, and `cf` discarding any ink component two labels claim. **If the word belongs to a grouping, most of those conflicts are not errors but artifacts of charging a grouping to one region**, which gives T123's non-monotonicity its mechanism from the author's side: the author's base unit is the finer one, and the finer the partition the more conflicts arise. Track L named the thing that makes this registrable -- **the primary is fixed at the 面 level, and that was not a registered decision, the instrument simply was that way from the start**. Two rules. **A registration must state its unit of analysis explicitly, beside the arms and the bars**, because an unstated unit is inherited rather than chosen and no later check examines it. And **the unit of analysis is the author's to set** (規律6's deepest form): it is not a modelling convenience but a fact about the work, and asking for it is not the same as asking what a picture shows. What follows is a question, not an answer: whether a まとまり has a name or a boundary in the file decides at which level L17/L18/L35/L36 and the colour track's standard should be defined -- and T113 applies to asking it, so it needs native crops on which the author marks which 面 belong together, not a sentence.
- **T124** A condition can be written down and still do no work, because what travels forward is the conclusion. **Three instances in one day, from two sessions** (2026-10-26). Track L wrote *the floor is relative so a ceiling cannot cut it -- **if** the arm and the control fall by the same amount, the comparison with the floor survives*: the antecedent is in the same sentence, yet the operative claim, *B is safe*, was asserted unconditionally, and nothing in the instrument or the registration was obliged to discharge it -- conflicts arise from how a region's boundary samples line ink, the control displaces its regions, and no one held a reason to think the two rates equal. The foundation did the same thing twice: *no verdict depends on it, the arm sits below half the floor in both versions* was a magnitude claim never checked (ako6 is 0.0953 against 0.1641, which is 58% of the floor, not below half), and it was the stated reason for declining the run that now decides whether L18 closes; and *the missing layers cannot be told from the kept ones on any inventory column* was a universal negative drawn from five of twenty columns. The common shape is not carelessness about the condition -- in L's case the condition was printed -- it is that **a stated antecedent with no assigned discharger is indistinguishable in use from an assertion**. So the rule is procedural, not rhetorical: **a conditional claim must name who discharges the antecedent and when, or it must be written as the weaker claim that needs no condition.** L's repair is the model: print `conf_px/ink` per arm per standard so the antecedent becomes a measured column, keep the bar unchanged, and withdraw only the sentence that asserted safety. Near T111 and T96 in family (a quantity travelling without the thing that fixes its meaning), but the missing piece here is **an obligation rather than a definition** -- which is why re-reading the sentence never catches it and only asking *what would have to be true, and who is measuring it* does.
- **T123** A penalty rule inside a summary statistic can make it non-monotone in the very thing it is supposed to measure, so a *more* correct standard scores *lower*. `cov_honest` excludes any ink component claimed by more than one layer label: `seed_of` returns a single layer label per region -- the argmax over line ink within 4px of the region's boundary -- and `cf = int(len({t[1] for t in lst}) > 1)` sends the whole component's area to `conf_px` instead of the numerator. **Splitting a region creates new boundaries, each fragment claims its own label, and one component claimed twice is dropped.** Track L demonstrated it synthetically with KC48, changing the segmentation and nothing else -- **not one pixel of the ink union moved, and the component count stayed at one** (2026-10-26): one line component with 4,800px under layer 1 and a 240px tip under layer 2 gives **honest 5,040, conflict 0, labels [1]** undivided, and **honest 0, conflict 5,040, labels [1,2]** divided in two, with **purity identical at 0.9524 in both**. So the quantity answers *does the propagation cover the author's fill* **and** *does the standard's segmentation agree with the line layers' structure*, inseparably -- and since what the right segmentation is **is the project's open question**, the statistic entangles its answer with the unknown. Two consequences. L18's comparison survives, both arms facing the same segmentation, but its absolute numbers **0.0953 / 0.0775** are coverage of *the 300 layers selected by `keep4>=0.95`, segmented by a radius-4 opening* -- **the segmentation is part of the constitution, not only the layer set**, and another segmentation's `cov_honest` is another quantity. And L35, whose standard is an order of magnitude finer, is **predicted to drive `cov_honest` toward zero through the conflict rule rather than through propagation failing**, which is why the foundation now requires the conflict fraction under both standards on a single page **before** the 29-hour run, not after. The general rule: **when a statistic carries an exclusion rule, test it by changing only the partition and holding the measured material fixed** -- KC48's form, a known cell in which the thing being measured is provably identical.
- **T122** A pre-registration can carry a check that cannot pass and lack the one check that matters, and neither shows up until the instrument runs. Track L registered **KC41** for L35 as *every one of L19's 551 region names must appear among the new standard's names, zero missing* -- and it has **no passing path at all**, because a region's name is `{work}_{pid}_c{ci}_r{gi}_...` where `gi` is the component's index within its panel, so **changing the standard necessarily renumbers every region** (2026-10-26). Replaced by containment, the check then failed at **0.8465**, and the cause was not wiring: **two things had changed at once** -- the layer set (L's own 300 classified layers to the author's 496 delivered beta) and the ink (radius-4 opening to raw ink) -- and *raw ink is a superset of opened ink* holds only over the same layer set. **T110 inside a known cell.** Pinned to the delivered beta and varying only the ink, coverage is **1.0000, no exceptions**. The omission is the other half: with the standard switched to the author's delivered beta, **289 layers sit on both sides at once** -- in the standard that defines a region and in the arm's input that emits the words -- so the arm would be predicting fill from the very layers defining it, and **the registration contained no check for this**, meaning a contaminated `cov_honest` would have been read as a result. L added **KC44**, detected the 289, and the instrument now removes them from the arm, taking the word-emitting population from **1,681 to 1,392** -- which also means **L35's arm is not L18's arm and the two may not be subtracted** (T112's rule, applied before the fact rather than after). Two rules for every registration from here: **a check must have a state of the world in which it passes, and that state must be compatible with the change being measured**; and **when a standard is replaced, the first check is whether any member of the new standard also feeds the arm.** Note what worked: all four of the surviving cells pass, KC43 prints *this check is empty* on its 0 findings (T102's requirement), and KC42 shows the standard genuinely moved -- a 6px bridge gives 2 regions under the old definition and 1 under the new.
- **T121** A length constant carried into a new resolution is not merely uncalibrated -- it is wrong by the ratio of the scales -- and when its known cell travels with it, the gate certifies the constant at its original scale and can never detect the move. `FILL_HALF_WIDTH_PX = 4.0` is defined in `lineart-solid-fill/tools/evaluation/fill_mask.py` **for Track J's 480x480 tiles**, and its self-check is stated at that scale: *a disk of radius 20 survives almost entirely, a 3px line disappears completely*. Track L has been applying it to **native layer PNGs on a 7016x6543 canvas** -- about a **14x scale difference, never redrawn** (found by L while answering the foundation's order to redraw it for the unit question, 2026-10-26). The part L did not state, and the sharper half: **`l10_keep4_all_layers.py` imports the known cell along with the constant** -- its KC1 prints *disk r=20 keep4 / 3px line keep4* on synthetic 400x400 arrays -- so the cell that guards `keep4` across all 2,963 native-scale layers **passes trivially and says nothing about the material it guards**. **And the role itself turned out to be scale-dependent, not only the constant** (L34's full run, 2026-10-26): redrawn over 496 delivered beta layers against 1,509 `線画`-named layers, the two registered conditions **never overlap at any radius** -- beta's median clears 0.95 only at `r=2`, where line stands at 0.7963, and line falls under 0.05 only from `r=8`, where beta has dropped to 0.7678 -- so the verdict is **保留**, and `(C)` gets no number. The reason is visible in one figure: **the author's own beta layers lose 11% of their ink to a radius-4 opening** (median 0.8892 pooled; ako6 0.9297 against ako7 0.8291, generations again behaving as pools), because solid fill at native resolution **has thin, pointed parts**. At 480x480 the downscale removed exactly those parts, which is why *tell fill from line by form_keep* worked there. **A borrowed constant can be redrawn; a borrowed role may not survive the new scale at all**, and the one-page smoke test that showed beta 0.9524 at `r=6` was not representative of the 0.7969 the full population gives. A borrowed constant hides behind a borrowed gate: 規律9's *draw the constant again for each role* needs its companion, **draw the known cell again for each scale**, and L's repair is exactly that (KC38 disk r=100 survives at r=32, KC39 a 20px line vanishes at r=12, KC40 the same line survives at r=8 -- the third added so the pair is not one-sided, T107). What does **not** follow: `keep4`'s agreement with the author in L27 (`keep4 >= 0.95` wrong on **0/17** called fill and **1/57** called line) was obtained against the author's own judgement at native scale, independently of the constant's justification, so **the justification is void and that evidence stands**. The propagation to watch is the population: `l7_mask_on_clean.py` defines its ground truth as *production layers with `keep4>=0.95`*, so if the recalibration picks any r other than 4, every population defined by that expression is a different population and must be listed, not silently re-run. Sibling of T110 (a gate lost on reuse); here the gate was carried and that is what made it useless.
- **T120** A bar's margin can be thinner than the instability of the variable the bar is drawn on, and then PASS is not a verdict with room to spare -- it is an unresolved comparison. After T116 fixed `preview_dark`'s version-to-version spread at up to **0.1092**, the foundation asked Track L the one bounded question that decides whether the `>= 0.5` solid-fill gates are affected: how many layers sit within 0.11 of the threshold. The answer is **not zero** -- **402 layer-instances over 58 pages**, counted on both sides because a one-sided count makes a gate look safe (upper side could fall out, lower side could fall in): `l12_panel_regions` 105+68 of 2620 = 0.0660, `l8_inventory_reconcile` 14+12 of 614 = 0.0423, `l15_occlusion` 108+95 of 2877 = 0.0706 (2026-10-26, `l33_preview_dark_margin.py`, which reads `l8`'s `FAMILY_CUT` **by AST rather than copying it**, so the constant cannot drift). The bite is in the known cell: `l8` recovers **exactly all 496 delivered beta layers, recall 1.0000 / precision 0.9940** (3 extra), and the room to the 0.98 bar is **nine layers on the recall side** -- while **all fourteen upper-side window layers are themselves delivered beta**. Worst one-way reading: recall 482/496 = **0.9718 FAIL**, or precision 496/511 = **0.9706 FAIL**. Foundation's ruling: **the PASS is not withdrawn** -- a one-way drift of 0.11 has never been observed, and L32's six pairs run five one way and one the other, which is what differing save moments look like -- but **the PASS is not guaranteed against the instability either**, and no number may be quoted from these gates as though its margin were established. The general form, and new to this ledger: **a margin must be compared with the noise of the gate's own variable, not only with the bar.** T97 fixes the floor as a function of the run's own control; this fixes the margin as a function of the gate variable's own spread, and nothing in the ledger's gates measured that until the question was asked.
- **T119** Before calling an absence a gap, apply the receipt check already in the ledger -- and an arm named after a name must carry the condition that gives the name content. Track L's running `l34_radius_calibrate` printed OpenCV `imread_ ... can't open/read file` warnings; the foundation cross-checked `inbox/page_layers_priority65.csv` against the delivered `all_layers/` directories, found the **line arm missing 173 of 1,682 layer PNGs (0.1029) while the beta arm had all 496**, claimed it **could not distinguish the missing from the kept on any inventory column** -- having examined **five of the twenty** (`layer_type`, `folder`, `visibility`, `has_mask`, `tone_enabled`), while the column that separates them perfectly was the twentieth -- called the direction of the bias unidentifiable, and asked the extraction side whether this was a gap or a scope (2026-10-26). **It was neither a gap nor unknowable.** The delivery's rule is *every non-folder layer whose `drawn_ratio` is above 0*, and the correspondence is exact with no exceptions: **`drawn_ratio > 0` is 2,963 rows and 0 of them lack a PNG; `drawn_ratio = 0` is 2,581 rows and 0 of them have one** (the only unmatched file in the directory is `all_layers_manifest.json`). **`drawn_ratio` is a column in the very CSV `l34` reads**, and the foundation's own receipt check for that delivery had already recorded *`rendered=True` 5,544 / `drawn_ratio > 0` 2,963, both matching the notice* -- so the rule, the column and the count were all in hand, and a notice went out to ask a question this ledger had answered. Two rules. **A registered arm selected by a layer's name is a population of names until the condition that gives the name content is added to it** -- `線画`-named without `drawn_ratio > 0` admits 173 rows that the inventory itself says hold no ink, and T117 is the same sentence for material: a name determines neither what a layer holds nor that it holds anything. **Third instance the same day, and the cheapest to see**: the 33-row structure CSV arrived without a unit column, the foundation inferred millimetres from preview ratios, got every row right, and asked the extraction side to add a unit column -- **`pages.canvas_unit` had been there all along** (2 = mm, 1 = inch, NULL = 0 = px, 34 pages on the NULL side), and the inference was luck, since `x.clip` at 3000mm would have looked the same. Two general shapes fall out of it. **A derived extract silently drops the field that fixes its values' meaning** -- so a request for a CSV must enumerate those fields, here `canvas_unit` and `dpi`, not only the quantities -- and **the convention's default pointed the wrong way**: with absent meaning pixels, anyone reading the extract reads pixels, and only the millimetre rows are silently off by `(dpi/25.4)^2`. **Fourth instance, hours later, with a different repair**: the foundation asked the extraction side to *carry `f_sha256` in the `.lip` structure scan so the side is not walked twice* -- and all 1,986 hashes, with size and mtime, had already been computed during the duplicate measurement and were already in the hash table. Looking in this ledger would not have helped, because the work was done elsewhere and minutes earlier; **the repair is to ask what exists before specifying what to add**, which costs one extra sentence in the same message. The three earlier instances are answered by consulting what is in hand, this one by asking, and together they say the standing move before any request is *what do you already have*. And **an absence is interpreted against the delivery's selection rule, which must be recorded on receipt and then actually consulted**; the first move on an unexplained absence is to look up that rule, not to write to the supplier. What the 173 cost the measurement is **nothing**, and that is measurable rather than argued: the layer PNGs are **alpha-encoded, `g = 255 - alpha`** -- verified against the ledger, where the fraction of pixels at `g <= 229` reproduces `drawn_ratio` to all six printed digits on every layer sampled -- so `ink = im < 128` is exactly **alpha >= 128**, the two thresholds nest (128 above 26), and `drawn_ratio = 0` entails no ink by inclusion. Track L confirmed it directly in the extraction side's ledger, which carries `ink_px` even for rows with no PNG: of **2,581** rows at `drawn_ratio = 0`, those with `ink_px > 0` number **zero**, and the check was shown non-empty first (T102). The foundation had instead argued it through compositing -- *ink at alpha 25 over white renders no darker than 230* -- which reaches the same conclusion from a false premise, since the PNG is not a composite over white. The adverse reading the foundation ordered (score the 173 as 1.0) would have been the single most wrong repair available, and the event it guarded against cannot occur. A separate consequence of the same encoding: `drawn_ratio > 0` is **much weaker** than *has ink* -- `page0028_L17` is `drawn_ratio` 0.0726 against an ink fraction of 0.0083, nearly nine tenths of its pixels faint -- so the 2,963-layer population that all of Track L's measurements inherit (through `gated_layers()`, imported by L17/L18/L22/L31) is *layers with a pixel at alpha 26 or above*, which must never be restated as *layers with ink*. One real correction does survive from the exchange: the delivery note's phrase *not a single pixel drawn* is inaccurate, because `drawn_ratio` counts only pixels at **alpha 26 or above**, so a layer among the 2,581 may still carry faint or few pixels -- every claim that an image *is gone* must be stated at that threshold.
- **T118** A wait loop whose fallback value is a legal input to its own probe never ends, and the cost is time rather than a wrong number. Track L watched a run with `pgrep -f ... || echo 1` feeding `ps -p <pid>`: when `pgrep` finds nothing -- the very event the loop waits for -- the fallback supplies `1`, and `ps -p 1` succeeds because init always exists, so the `until` loop read "still running" forever. **L sat on a finished run for about two hours**: the measurement started 12:17 and ended 12:21:52, and the report reached the foundation at 14:32 (2026-10-26). Nothing was wrong with the measurement or its output; the instrument that was supposed to say *it is done* could not express that state. The rule: **a liveness loop must test for the positive event** (the output file exists, the log's last line is the terminator) **and no fallback may be a value the probe accepts** -- `|| echo 1` is the worst possible default here precisely because PID 1 is the one PID guaranteed to exist. T104's family (a default erases a true absence), but in a different consequence class: T104 and T110 produce a number that is wrong, this produces no number at all and spends wall-clock instead, so none of the ledger's numeric gates can catch it. Sibling in shape to T108 -- a check whose silence is indistinguishable from success. **Second instance the same day, same family, different layer**: L launched L35's smoke test as `... 2>&1 | grep -v WARN | head -30` in the background, and **`grep`'s block buffering held every `flush=True` progress line**, so sixteen minutes produced no output at all and the run was killed as uninformative when it was in fact working. `flush=True` in the writer is defeated by buffering in any reader downstream of it. The rule: **a background run writes to a file directly, never through a pipe, and Python runs `-u`** -- and in both instances the instrument that was supposed to report state could not express it, so the operator watched attentively and saw nothing.
- **T117** A rate near zero, measured only where the label could be determined, is not a rate near zero when the undetermined bucket outweighs the positives. Track L's L31 row (A) -- *do two different materials land in one component* -- came back **0/625 components in ako6 and 2/938 = 0.0021 in ako7** (pixel weight 0.0000 and 0.0134). In the same run, components holding two or more layers **of which at least one has no determinable material** are 24/625 and 23/938, at pixel weights **0.1605 and 0.2420** (2026-10-26). The undetermined bucket is therefore **twelve to eighteen times the measured positive by weight in ako7, and infinitely so in ako6**, and L states plainly that whether those belong to row (A) cannot be told from this measurement. The cause is that material is read from the layer's name, and `black` (97 layers), `線画` (40), `MASK` and `white` do not name a material -- L correctly refused to fold them into the known set. **So (A) may not be reported as "mixing is essentially absent"**; what was measured is *mixing among components whose layers could all be named*, a different quantity. 規律9 has so far been applied to the denominator -- count what the instrument dropped, by reason -- and this is its form for the **numerator's eligibility**: a case excluded for want of a label is not a negative, and printing it beside the rate (which L did) is what makes the rate readable. The repair is upstream and cheap: **ask the author what material a layer named `線画` or `black` holds**, which converts the largest term from unmeasured to known. Near T111 in consequence (the number is not yet a number) but the gap is in the **population of the numerator**, not in the estimator.
- **T116** A quantity read off a cached derivative measures the moment the cache was written, not the object the ledger is reasoning about. `preview_dark` is computed on the **`CanvasPreview` image CSP saved inside the `.clip`** -- the composited page as it stood when the author last saved -- and not on a redraw of the layer set that was delivered (the extraction side's own definition, 2026-10-26). Track L found the demonstration without looking for it: among the 45 layers left in L28's material, **six are the same layer present in two versions** with identical `layer_id`, `ink_px`, `canvas` and `drawn_ratio` to six figures, and their `preview_dark` differs anyway -- 0.0186, 0.0203, 0.0251, 0.0390, 0.0653 and **0.1092** (L70, 線画 hair). **Inside such a pair the layer's own pixels and the downscale factor are identical, so hypothesis 1 -- thin strokes greying out under the resize -- cannot produce the difference**; it has to come from the compositing side. Two readings remain, occlusion by other layers and two different save moments, and **the direction does not settle it either: five pairs are darker in the `page00NN` version and one in the `doneN` version**, which is what differing save times would look like and not what consistent occlusion would. Either way the consequence is the same and is stronger than the limit the foundation had drawn (*do not use it as a line-art gate*): **`preview_dark` is not a coverage measure for the delivered artifact at all.** What it does *not* yet show: the observed version-to-version spread is at most **0.11**, while the `>= 0.5` threshold still in use in `l12_panel_regions` / `l8` / `l15` was calibrated on separations of **0.25 and more** (speech balloon 0.25 vs line art 0.90 vs bleed 1.00), so those solid-fill gates are **unverified at the margin rather than shown to be wrong** -- the bounded question that decides it is how many layers in their populations sit within 0.11 of 0.5. T99's sibling: there a file's mtime dated the copy and not the drawing, here a preview's darkness dates the save and not the delivery. The general rule: **before a derived image is used as evidence about an object, establish that it was derived from the version of the object in hand.**
- **T115** The census that every page-count in this ledger rests on was never compared with the disk, and when something finally compared them it found a whole work missing. `pages` in the structure DB holds **7,634 pages** -- the number this ledger cites as 全数 (CURRENT.md:358, and the population named in every question routed to the extraction side) -- while the disk holds **7,667 `.clip` files**. The 33 absent from the inventory are **29 pages of `2017\ol\ol\page*.clip`, an entire work** (with its `ol.cmc` beside it), plus four loose files at year roots: `2021\tiacut.clip`, `2021\x.clip`, `2023b\add.clip`, `2023b\udonge.clip`. **`2017/ol` appears nowhere in `doc/CURRENT.md`, `doc/glossary.md` or `results/`** -- it was never in any population, any stratum or any count, so no instrument could have reported it as dropped. **Why the inventory omitted them is now known, and it is worse than a naming rule**: the page list was never built by walking the disk at all, but by `rglob`-ing under each row of a *works* list, and that works list was derived from a sample survey of 2026-08-12 by the line `pages = [p for p in work["ページ"] if p.get("status") == "ok"]; if not pages: continue` -- so **a work whose sampled pages all failed to read is dropped silently and entirely**. `2017\ol`'s eight sampled pages all returned `TimeoutError: page exceeded 180s` (its pages run 100-210MB), and the whole work left the population. The second rule is flat: a work is *a folder under a year folder*, so four files sitting directly in a year folder belong to no work and no `rglob`. Of the survey's 316 works, 27 are absent from the works list and 26 of those hold no `.clip` at all, so **`2017\ol` is the only work lost this way** -- **[CORRECTED 2026-10-26: only within `.clip`. Those 26 are largely `.lip` works, 1,986 pages of manuscript that no census has ever counted; see T126. The closure claimed here was ratified by the foundation and is wrong as stated.]** and the 33 are the complete difference (the extraction side walked the whole of `F:\manuscripts` to confirm: list-not-on-disk **0**, disk-not-in-list **33**, and 7,634 + 33 = 7,667). **And the recovered work is the heaviest in the corpus**, which is the part that generalises: `2017\ol` runs a median of **225 layers** and **150MB** per page (max 327 and 213MB) on a **497x277mm at 600dpi canvas = 11,740 x 6,543 px, 1.67 times ako6's area**, and carries **1,003 `image_material` layers across 29 pages** -- against 1,819 in the whole 872-page rendered delivery, a different population but a comparable count, and `image_material` is exactly the axis Track M needs for the reference-material question. So **a timeout does not thin a population at random: it removes members from the top of the cost distribution**, and the exclusion correlates with every property that makes a work interesting. The structure of all 33 read without error on 2026-10-26 and `pages` is now **7,667** / `layers` **312,898** (the foundation's receipt check: the delivered CSV's `n_layers` sums to **6,889**, exactly 312,898 - 306,009) -- but **that does not retire the timeout**, because the August survey was *rendering layers* while this scan only read SQLite tables, so whether `ol`'s layers render inside a limit is still unmeasured. The rule this yields is the general one: **an exclusion caused by a resource limit must be recorded as pending, never as absence** -- a timeout and an empty folder reach `if not pages` looking identical, and the material here is not unreadable, it merely did not finish in 180 seconds on one August afternoon. T104's family, at the scale of a whole work and with three months of latency. **Measured on 2026-10-26, and it corrects this entry's own earlier phrasing**: the timeout was not one afternoon's bad luck. All 29 pages render, in **2h04m on one idle process -- mean 256s per page, median 248, min 112, max 355 -- and 28 of the 29 exceed 180s**, the single exception being the 93-layer first page at 112s. So the August limit was reporting the truth about cost, deterministically, and **the defect was never the limit but that a cost signal was written down as absence**: had the survey emitted *ol: 8/8 sampled pages exceed 180s* as a row, this ledger would have known three months ago that `ol` is its most expensive work. The structural repair belongs in that survey script -- **a run under a limit must emit the timeout as data, not drop the subject**. A cost model falls out as a by-product, and the foundation fitted it from the delivered timings: elapsed seconds correlate with **total** layer count at **r = +0.901** (better than with layers holding content, +0.829), and a through-origin fit gives **1.086 s per layer** with residuals from -84 to +38s. It is a **within-work** rate -- one canvas at 11,740 x 6,543 -- so it may not be carried across canvas sizes without an area term; what is new is that **the DB already holds a layer count for every page, so the cost of any render order is computable before the order is placed**, which is what nobody could do in August. The repair is now structural rather than a one-off re-count: `file_hashes` will be built by walking the disk, keeping `in_page_list` per file, and printing both the unreadable count and the not-in-`pages` count on every rebuild -- **the comparison becomes a scheduled measurement instead of an accident**, which is exactly what this entry says was missing. What makes this registrable is not the hole but **how it surfaced**: it was a side effect of proposal 1, the all-files sha256 sweep, running at lowest priority -- a request the foundation made for a different reason, and only because T114's wrong claim forced the question of whether rows and files correspond. **Both of the only two occasions on which this ledger was compared with the disk were accidents of some other errand, and both found something** (T114: 872 rows, 0 missing but never once checked; here: 33 files, one whole work). A census is an instrument and has had no control. The consequence that bites now: the **116 duplicate clusters / 248 pages** were built from `pages`, so **a file absent from the inventory cannot be counted as a duplicate even if it is one** -- `2021\x.clip` and `2023b\udonge.clip` sit at year roots, exactly the shape of the stray working copies that generate those clusters -- and the duplicate census's zero for them is therefore **unmeasured, not measured**, with the 8 version questions (A--H) possibly missing a member. The rule: **a count called 全数 must name the thing it enumerated, and an inventory is not the disk**; comparing the two is itself a measurement and must be scheduled rather than inherited. T88's shape one level up -- there the manifest dropped folder rows (167 -> **183** bases) and reading only the manifest turned an omission into an instrument rule; here the page table dropped files and reading only the page table turned an omission into a denominator. Family of T102/T114: the instrument reports absence the same way it reports not having looked.
- **T114** Output from `&`-chained commands on a remote shell does not stay in step with the `echo` markers meant to separate it, so a failure can surface under the next command's heading. The foundation ran `dir /B "<missing path>" 2>nul & echo ---sub--- & dir /B /S "...2019b...page0186*" 2>nul` over ssh, read `The system cannot find the path specified.` as belonging to the second `dir`, and told both the author and the extraction side that **a ledger row named a file that does not exist**. The file exists (4,605,379 bytes, 2022-05-06 07:19 PM), and Track M's full sweep of all 872 rows against `dir /B /S "F:\manuscripts\*.clip"` found **0 missing** (2026-10-26). Reproduced deliberately afterwards: with a missing path first and a real file second, the error prints *after* the marker and the real file prints after that, and `2>nul` does not suppress it. Two rules: **a chained remote command may not be read positionally** -- send one command per call when the answer matters -- and **a negative existence claim may not rest on a single invocation**, because the tool reports absence the same way it reports its own failure. The strongest form of T102's family: there the check could not run, here the check ran and its silence was read as an answer about the wrong object. The schema proposals the claim was used to motivate all survive; only the example was wrong, and the real argument is stronger -- those 872 rows had **never once been checked against the disk** until the claim forced it.
- **T113** Every gate watches the instrument; none of them watches whether the question has an answer. Track M's page-kind UI passed, in one morning, a JS syntax check, an HTTP 200, a key diff of all 22 fields the JS reads, a live `go()` run driving the served page, three image gates, and a full sweep of the served HTML for stale wording -- the foundation ran the last three itself before handing over the URL -- and the question it asked had no answer, because **the picture was a composite of layers and the screen never said which layer the question was about** (2026-10-26). The author diagnosed it in one line: *"複数のレイヤが合成されたものが表示されているようだが…左端に合成前の各レイヤを並べて、どれについて答えてほしいかを問うてほしい"*, and a page turns out to hold two or three kinds at once, so no single answer exists. **The only instrument that detected it was the escape hatch**: all three answers came back as free text (100%, against 25% on the area-count UI), each naming several kinds -- *下絵、最下段だけ線画が入ってる* / *資料、ラフ、セリフ*. A gate that runs the page cannot tell a well-posed question from an ill-posed one, so **the escape hatch is not a courtesy to the respondent, it is the detector for this failure**, and its rate is the signal. Cost: 2.0 minutes of the author's time, spent after the foundation had verified and handed over. The repair is upstream of every other rule -- **say on the screen what the answer is about, and when the image is a composite, show the parts** -- and it is the first rule in `doc/ponytail_skeleton.md`, which until now began with *run it yourself end to end*: the foundation did run it end to end, and end-to-end was not enough.
- **T112** A filter switched on by a command-line flag changes the *control's* population as well as the arm's, and in the opposite direction to the one intended. Track L's `--exclude-gapfill` run reads `if EXCLUDE is not None and gi not in cname: continue`, and `cname` holds only regions that carry a name, which is awarded only where `adj != 0` at the original position. The true arm never had the nameless regions -- it `continue`s on them -- but **the shuffled control was placing and scoring them**, so turning the flag on removed 45 pre-registered regions from the arm and **~658 nameless rows from the control** (shuffled-only keys 229 -> 0) at the same time (2026-10-26). The cross-version arm comparison stays readable, since both versions see named regions only and differ by exactly the 45 (keys 1007 -> 968); the **cross-version control and floor comparison does not**, because the 45 and the nameless rows are not separated in the control's -0.0029 / -0.0108 movement. **[WITHDRAWN 2026-10-26 15時台: each version's own verdict is NOT unaffected.]** Track L traced the gate while implementing L36 and found it is `if EXCLUDE is not None and gi not in cname: continue` -- **it only fires under `--exclude-gapfill`**, so the **default version, which is the registered primary, is the broken one and the flag version is the symmetric one**. The flag did not contaminate the control; it incidentally repaired it. Therefore the default run's control scores regions the arm never had, its floor is not a floor for the arm's population, and **the primary verdict does depend on this**. The morning's *229 keys control-only, 0 under the flag* and the afternoon's *127 control-only, 309 arm-only* are **the same defect seen from both ends**: `cname` is only assigned where `_adj != 0` at the region's true position, and with the gate dead the shuffled arm places and scores the nameless ones anyway. The foundation also has to withdraw its reason for declining the repair run that morning -- *no verdict depends on it* -- which is exactly the run (L36) that now decides whether L18 closes. The other reason given stands: widening a registered population after seeing the result is what the registration prevents. **The lesson is that these are two different requirements and refusing the first does not justify refusing the second**: *do not widen the registration after the fact* and *the control must be symmetric with the arm* are independent, and a symmetric re-reading, pre-registered and printed beside the unchanged primary, is not a widening. L found it **by counting rows before comparing them** -- reading only the four-column table would have given "the control fell" as a finding. The foundation declined the repair run: no verdict depends on it (the arm sits below half the floor in both versions), and widening the registered population after seeing the result is the thing the registration exists to prevent. The ledger instead carries the rule: **a control column means something only within its version; never subtract one across versions.**
- **T111** A registered summary statistic does not name a number until its convention is named too, and the convention cannot be chosen after the data are in. The author's sixteen area counts gave twelve numbers, `1,1,1,1,1,1,2,2,2,4,4,4`, read against a pre-registered bar of **IQR >= 2**; the foundation reported **2.50 (bar crossed)** from `statistics.quantiles` (Python's default, exclusive, R type 6) and Track M reported **1.50 (bar not crossed)** from `np.percentile` (linear, R type 7), with Tukey's hinges giving exactly **2.00** in between (2026-10-26). **Two sessions read opposite verdicts off the same twelve integers**, and nothing either of them did was wrong -- the registration named `IQR` and stopped. The repair is to write the method into the registration, not to pick the one that passes. Note what does *not* rescue it: the top bin is `4以上`, so the sample is right-censored and every convention's IQR can only grow -- but *direction is safe* is not *the bar was crossed*, and under linear interpolation the censored values must actually reach **6** before it is. The verdict therefore stands at **undecided**, which is a different thing from either reading. Near T96 in form (a number that travels without the thing that fixes its meaning), and the first instance in this ledger where the ambiguity was in the **estimator** rather than in the quantity or the population.
- **T110** Rebuilding an instrument silently drops the gates the previous one passed. Track M's `m27_author_crops.py` has carried a `flat()` gate since it was written -- *discard the window if fewer than two colours hold 5% of it* (lines 98-102, called at line 220) -- and the reused `m47` queue builder did not call it, so **two of eighteen crops handed to the author were two-colour images**: the window fell inside a component that covers the whole canvas, and 99.14% of it was flat cyan (2026-10-26). The gate was not removed by anyone's decision; it simply was not carried across, and nothing in the new instrument's output announced its absence. The repair M adopted is to **call `m27.flat` itself** rather than restate the rule, and to **print the colour count as an independent column as well**, since one gate will miss a future shape the other catches. The sharpest demonstration is on the identical object: all three of m27's run logs carry `捨てる(窓が単色) 000_2017_illustration_sunshine2_page0002 L61`, and **m47's question 0 is `work=000_2017_illustration_sunshine2_page0002, lid=61`** -- the same layer m27 discarded three times was handed to the author. On the evidence itself: the first justification offered was that discard line, while the logs sat in the session's scratchpad and `results/` held none, so the foundation rejected it and M moved the three logs into `results/m27_author_crops/`. **A citation that points at a place which disappears is not a reason, even while it still resolves** -- the instrument's comment outlives the session that wrote it, so a comment may only point at what also outlives it. Walkability has three tiers, and Track L sharpened this one the same day: a session scratchpad **disappears**; a file under `results/` survives but is gitignored, so it **never reaches a clone**; only a **committed instrument that can rebuild the file** travels everywhere. The strongest form of "you can walk it" is therefore not the file but the tool that regenerates it. The form that survives a lost log is the opposite direction: **m47's own output proves the gate did not run**, because `flat()` would necessarily have dropped a two-colour window. A source line is not an event; T106's family. Sibling of T108 -- there a check was never written, here a check was written, passed once, and was lost on reuse.
- **T109** A composite key is only unique if every one of its parts is global, and a part that counts within a container repeats once the page holds two containers. The region key `(pid, comp, region)` has been cited in this ledger three times -- *keys are not unique*, the arm's 1,007 keys against 1,153 rows, and the caveat on L36's 127/309 -- **without ever being registered, so the number travelled while the reason did not**. Track L found the mechanism on 2026-10-26 by running KC46 before the body: the same key `('...page0028', 2, 1)` carried **two different names, `_002` and `_003`**, and the cause is that **`ci` is the component index within a *frame* from `koma_manifest`, while a page holds several frames**, so the same `ci` recurs on the same page in a different frame. The repair is a page-wide panel serial `pan` added to the row, which is an identification column and moves no arm, bar, standard or seed. What this cost: nothing, **because the known cell ran first** -- had the body run, the two arms' region sets would have been compared on colliding keys and **a collision would have been read as a difference between the arms**, which is the exact quantity L36 exists to measure. The general rule: **before two sets are compared by key, show that the key is unique inside each of them**, and a key assembled from a container-local counter is unique only once the container is named in it. Sibling to T94 (a fingerprint identifies content, not the page it came from): both are keys that look total and are not. **And the ledger's own failure mode, named by L: assigning the number became the reason not to diagnose it.** L found the non-uniqueness in the morning, gave it this number, wrote *indications, not numbers* beside it, and went on using the key for hours -- *a trap number is a label for a mechanism, and when the entry holds no mechanism the number licenses continued use of the defective thing as a known caveat*. What produced the diagnosis was a known cell failing and creating a reason to look. **So an entry with no mechanism is a debt, not a record**, and this register should be read for numbers that have never been written down.
- **T108** A startup gate that checks *syntax* and a server that answers *200* say nothing about whether the page renders, and a reused instrument's only joint is **the names of the keys in its queue**. Track M's area-count queue wrote the field as `area_px`; the reused `l19_fill_label_server.py` reads `it.area.toLocaleString()` inside `info()` <- `showImg()`, which `go()` calls **before** it builds the answer buttons, so one missing display-only key threw a TypeError that took the whole render with it: **no answer buttons appeared and `keymap`/`okeys` stayed empty, so none of 1/2/3/4, T, M, C, B, N responded** -- two symptoms, one cause (2026-10-26, found by the author, after the foundation had handed over the URL on the strength of "the startup gate passed, the JS passed a syntax check, HTTP 200"). None of those three executes `go()`. The mechanical check that would have caught it is a set difference: every `it.<key>` in the JS against the keys `payload()` adds plus the keys in a queue item -- 22 keys read, **3 absent** (`area` throws; `scale`/`ctx_scale` do not throw and instead print **"(原寸)"** over a reduced image, which lies to the author). The name of the check is again the danger (see memory *names-are-not-checks*): "gate passed" named a gate that did not look here. Sibling of T105 -- there *launched* was mistaken for *running*, here *starts* was mistaken for *works*.
- **T107** A two-way `else` never checks the side it falls through to, **and a degeneracy check written for one end of the range certifies nothing about the other**. Second form, 2026-10-26: Track L registered **KC51** for L37 as *the quantity is not empty*, and the actual failure was **saturation** -- `用紙` carries full-canvas alpha 255, so `below` was always the whole canvas and **all 80 layers came back `ov = 1.0000`**, a trivially-1 quantity that the known cell waved through because it was watching the opposite end. The trap was already in this ledger as **L23** and was not applied. L's repair is the shape to copy: add **KC52** for saturation, keep KC51, exclude layers with coverage >= 0.95 from `below` (86 of them), print `below_frac` on every row, **and draw no new constant** -- the gate is `l12`/`l15`'s existing one, so the bar did not move. The rule: **a check against degeneracy must cover both ends, because a quantity that is always 1 and a quantity that is always 0 are the same defect and only one of them is usually imagined.** Track L wrote `w = 'ako6' if '_ako6_' in pid else 'ako7'` in **15 places**; against the authoritative columns it disagreed **0 times on 65 pages and 0 times on 7,362 layers**, so no number it produced was ever wrong -- and **one page of a third work would have entered the `ako7` pool in silence**, because `ako7` is asserted and never tested (2026-10-26, found by L auditing its own filename dependencies after `pageNNNN` fell). The repair is a function that *fails* when the work is undetermined, and **the reason to make it is not that it was wrong but that it would be quiet when it became wrong**. T102's family from the data side: there a check could not run, here a check was never written. The same audit found that `pageNNNN` splits L's 65 pages **perfectly** -- 40 current, 25 `done*` -- which is a coincidence L deliberately did **not** adopt, since Track M's corpus breaks that convention in both directions; **a rule that agrees perfectly today and silently stops agreeing later is the worst kind of anchor**.
- **T106** A cluster built by thresholding a similarity over *summary statistics* presents itself as a claim of identity, and the strongest-looking column inside it is the criterion that formed it. Track M's clusters are the Jaccard >= 0.5 of per-layer signatures `md5(name | layer_type | drawn_ratio | sat_mean | top_colour_share | colours_q16)` -- **no pixel is read** -- so "cluster 18's columns all agree" is the clustering rule restated, not independent evidence, and every "the contents match" written from it must read **the statistical signatures match**; pixel identity stays unverified until the sha256 asked of the extraction side arrives (2026-10-26, found by M checking the foundation's own framing: the foundation had said "if the cluster was built on a fingerprint then content identity is already established", which was wrong about this instrument). The label `decided_by='identical_metadata'` was accurate from the first day and the prose around it was not. **The question put to the author inherits the defect**: asking *which of these two is current* presents the pair as one page in two copies, which the threshold never established -- and the author's own answers refuted it twice in one night, A being a real duplicate and B being two partly overlapping branches with no order. Near T94 (a fingerprint identifies content, not a page) and one level worse: this fingerprint does not read the content either.
- **T105** *Launched* and *running* are two states, and a launcher shell reports only the first: it exits the moment the body is detached, so the body's death produces no notification and the session believes a registered run is in flight. Track L's `--exclude-gapfill` version, authorised and registered the same night, **never ran a single page** -- `pans[-1]['drop_gapfill'] = drop_gf` was written before the matching `pans.append(...)`, so it raised `IndexError` on the first panel, and had it not raised it would have tagged *the previous panel* (2026-10-26). The repair is to make the launcher confirm the body is alive after a delay, not to trust the launch. T102's sibling: there the check could not run, here the run could not run, and both are silent by construction.
- **T102** A check that cannot run fails as silently as a check that does not exist (2026-10-25, Track L: `KC7` and `KC9` read a `truth4` image absent from both disks, and `os.environ['L13_TRUTH']` raised `KeyError` at import, so the caller saw neither a pass nor a reason). Print why it cannot run, and FAIL.
- **T103** A known cell is worth no more than its own answer -- the `ProjectName` anchor's cell was the `superset` decisions, and the author refuted that rule 2/2 the same night, voiding the cell rather than the arm (2026-10-25).
- **T100** A known cell has to be independent of the rule it tests -- a cluster decided by the name cannot test a rule about the name, and the agreement it prints measures identity, not correctness (the foundation's own specification, 2026-10-25: 0/42 at p=2.3e-13 under a coin, which reads as a refutation and is a tautology). The run-level form of this is T97; this is the rule-level form.
- **T98** A bar written as "higher than the null" with no resolution is not a bar -- its ordering flips with the choice of statistic (the foundation's own bar on the succor sharing query, 2026-10-25: arm 0.30 against null 0.15 on the pair mean, but p=0.335 by permutation, and the reverse ordering on the per-page maximum 0.6 against 0.63). A bar on a comparison needs the null's distribution, not its point value.

### 閾値・定義・単位

- **T99** A file's modification time is the time it was copied, not the time it was drawn -- and when that order matches the folder-name order, deciding by time is deciding by name in disguise (2026-10-25, the extraction side's own distribution: 4,312 of 7,634 pages within a few hours of 2022-05-06, in folder-name order, seconds apart; 10 of the 11 undecided version clusters lie entirely inside it). `decided_by` must name the rule that actually decided.
- **T3** A threshold on grey reads pencil grain as strokes, so an unmatched-length figure is not an unused-input figure (Track D's own caveat on its result, adopted 2026-10-19)
- **T11** A radius taken from the sample point collapses exactly where it is needed (Track K, 2026-10-18, second failure of the same instrument)
- **T18** A definition can be right in mechanism and one case short in coverage
- **T26** An identity key must be derived from the thing identified, not from its position in the output (Track K, found while scoring, 2026-10-18)
- **T32** A threshold compared against a float distance can fail by 2e-14 (Track K, 2026-10-18)
- **T33** Skeletonization retracts at a line's end, so "within half a width of the skeleton" is not a definition of the line's body there.
- **T34** When one quantity's median equals another quantity's median, suspect a swapped name before a real coincidence (Track K's general form of the correction above, 2026-10-19)
- **T48** Binarising a conditioning map before skeletonising changes results by half.
- **T50** Do not read "strokes drawn where the conditioning map has none" (`gt_only`) on its own.
- **T54** `cluster_set`'s unit is built from proximity alone, so it mixes parts and silently drops strokes.
- **T56** A summary statistic taken over *all* ink is a reading of the solid areas whenever fills are present.
- **T61** A fill can be white, and an instrument that counts ink cannot see it.

### 道具の実装とバグ

- **T6** Do not repair a renamed reference with a repository-wide `sed` (Track J, 2026-10-18)
- **T21** An append-only log read as a set of rows double-counts its keys
- **T38** A whole-file sha of a shared dependency is coarser than the dependency
- **T45** `bipartite_match_f1` (the implementation behind `gt_bsds_f1`) can take tens of seconds to 12+ minutes on a single tile.
- **T46** The `"train" if name.startswith("housei") else "test"` split heuristic was wrong for 168 of the 192 `holdout_lineart_family.txt` tiles, and it was in nine places, not one.
- **T47** The shared `evaluate_stroke_stability.skeletonize()` leaves a 2px-wide skeleton.
- **T49** `manga_line` is nearly empty on the training pairs.
- **T51** The axis trap in Track F's `cluster_set`, settled 2026-10-04 -- and the cause this file carried until 2026-10-09 was wrong.
- **T52** The line-PNG check has no sensitivity to the orientation of a face box.
- **T55** `lineart_003_000` has a blank ground truth and is in `holdout_lineart_family`.
- **T58** Per-pair alignment scores now exist for all 8,467 training pairs
- **T79** A combined `git log -- pathA pathB` cannot tell you which commit added which path, and its ordering invites you to think it can.
- **T83** Two files can each delegate an instrument to the other, and nothing notices until a verdict needs it.

### 運用・文書・長時間ジョブ・配送

- **T1** A long run's own log can be the thing a crash destroys (Track J, 2026-10-19, caught in advance while M1 was in flight)
- **T4** "It cannot be checked from here" needs the right command before it is written down (foundation's own, 2026-10-19)
- **T23** The foundation cites the file when it states a fact about a track's material (its own failure, twice on 2026-10-18)
- **T59** An audit read a path the ledger says does not exist as a claim that it does.
- **T62** Every track tree carries a frozen copy of this file, and the copy says it is the first document to read.
- **T63** Three requests are in flight to the extraction side that the foundation did not carry, and that is the part worth fixing (2026-10-24).
- **T65** "Statistics only, so it is light" was wrong, and the correction is worth keeping.
- **T66** "The channel drops messages in both directions" is RETRACTED within the hour: nothing was ever lost (2026-10-24).
- **T69** Say this as three files, not as a loss.
- **T76** `git add -A` put the author's artwork into a public repository, and a force-push does not take it back out.
- **T77** A quoted commit hash can stop resolving, and then it is a number that looks like provenance and leads nowhere.
- **T82** A multi-day run's completion procedure must not be addressed to the session that launched it.
- **T84** A `.gitignore` rule written after a branch was cut does not exist on that branch. The `data/` rule came in with the 2026-10-23 incident's own fix and 10 of 14 trees never got it; the audit checks it from 2026-10-24 (check 9). **2026-10-25, the half of it that kept the finding alive: the remedy named a ref that does not resolve where it was sent.** The audit said `git checkout foundation -- .gitignore`, but the two trees in the other clone (`face-words`, `panel-generation`) have no local `foundation` ref, and their `origin/foundation` is a fetch from whenever -- `0ddb43e`, **108 commits behind**, with no `data/` rule in it. Track H followed the advice as well as it could, aligned to that stale ref, and reported the fix in good faith while the audit kept flagging the tree. The check was right and its *instruction* was the defect (T93's family: a remedy is a string, not a check). The audit now resolves the ref per tree and prints `git fetch origin foundation && git checkout origin/foundation -- <file>` with how far behind that ref is, or says to copy from the foundation tree when no ref resolves (self-test 33 cases).
- **T85** A `.gitignore` rule with a trailing slash does not match a symlink, so importing the integration tree's ignore file into a tree that symlinks `venv` silently un-ignores it. The fix for one hole opened another, in the same command.
- **T86** The dataset's panel line-art raster is **a different page from the author's manuscript**: only **63.0%** of the layers' ink reaches it and **54.0% of the area layers are missing outright** (line layers 5.1%), per-panel median 0.795 with **140 of 248 panels below 0.90**. Any arm whose input is that raster is predicting against something other than the ground truth. The rule the raster drops by **cannot be derived from `page_layers_priority65.csv`** -- by name and by parent folder, both the kept and the dropped appear (the commonest immediate parent among the dropped is `トーン`, 108) -- so it is a question for the extraction side, not a filter anyone here can reconstruct. **Asked 2026-10-25 with the user's approval and answered the same hour, and the answer reclassifies the number**: the raster is a **line-only instrument** (`extract_clip_dataset.py`'s `process_clip`, commit `0d975ac`), so most of what Track L's population carried was never meant to arrive. Of the 83,023,430 px that the named families account for (35.4% of the measured ink, 83.1% of the `<0.5` class), **A: 12,633,293 px = 5.4% of the ink is not line material at all** (`effect`, `描き文字　白フチ`, `MASK`); **B: 11,038,367 px = 4.7% is line material genuinely lost** (`線画` 10.56M via the width rule, `rendering_lineart_middle0001` 0.48M via the 3D rule); **C: 59,351,770 px = 25.3% is `black`, which the author uses for both roles**. **So the raster's defect, for this project's purposes, is 4.7% and not 37%** -- the rest is a selection rule doing its job and one ambiguous name. **The reason-by-reason ledger arrived 2026-10-25 18:04 and it does NOT turn A/B/C into a count** -- the foundation's own request was at fault. 7,362 layer rows over the 65 priority pages, eight `reason` values, `unexplained` zero, and three independently built numbers landing on one: `adopted` flag 1,874 = `reason==adopted` 1,874 = the check file's `delivered_n` summed 1,874, with all 65 pages matching `delivered_line_id == ledger_line_id` and **zero pixels differing by more than 64** (mean |diff| at most 0.2225/255). `empty` is 2,598 layers (35%) at ink 0, which confirms the structure DB's caveat from a different instrument. **But A/B/C cannot be restated from it: the unit and the population both changed** -- px became the sum of per-layer `ink_ratio` (page fractions, double-counted where layers overlap), and the 248-panel corpus became the 65 priority pages. On these pages the same name families are A 0.60% / B 1.45% / C 1.41% of a summed ink of 109.68 (adopted 3.80%), and **reading 4.7% as "now 1.45%" would be a configuration error twice over**. Asked for the two things that would make them comparable: a page-pixel column, and whether the ledger can be run over the 248-panel corpus's pages. **Both answered by 18:17, and one of them corrects this entry: the 248 panels' source pages ARE the 65 priority pages.** Track L's `l_248koma_page_paths.csv` (65 rows, `n_koma` summing to 248) and the ledger's page set are equal in both directions, so no 248-page run was needed and **the population difference stated above was wrong at the page level**. What remains is the unit (being fixed: `page_w`/`page_h`/`page_px` arrived, `page_w*page_h == page_px` on all 65 rows, 2,983,869,720 px in total, and a re-run started 18:17 that prints exact `ink_px` because `ink_ratio` was rounded to six places) and **the cut -- 248 panel rectangles against whole pages, which is the real difference** (ink on a page and ink inside a panel are different quantities). The foundation sent the panel definition: 248 rows pulled from the koma corpus manifest by L's own panel names, **every rectangle's area matching L's `area` at a ratio of 1.0000**, so these rectangles are L's definition. **What was not verified and must be before counting**: which coordinate frame those rectangles live in -- `scale` takes five values from 0.85 to 1.05 and `dx`/`dy` are non-zero on 129 of the 248 rows -- so one page (ako6 page0028, whose 7016x6543 both sides have confirmed) is to be aligned first, because **a frame off by one scale still produces plausible numbers**. Their 245 author frame folders and L's 248 panels are different definitions and both are informative; only the 248 can restate A/B/C. A side lesson in the same file: the bare `page` string has 64 distinct values for 65 pages, so joins go on `page_path` or `content_fingerprint`, never on a page name. **The frame was settled within the hour: the rectangles are pixel coordinates of the native line image.** Cropping the delivered 7016x6543 `line.jpg` of ako6 page0028 by the rectangle reproduces all four corpus panels, **including p1 (scale 1.05, dx/dy -208/-160) and p4 (0.95, -512/-528)**, so `scale`/`dx`/`dy` do not touch the line side -- they are presumably the rough's alignment, which they say they did not verify. **They printed a control with it**: shifted +/-3px, no shift is best, and the `author`-frame rows came out identical to the pre-rect run, which is the re-run's known cell. The foundation asked for two things before this becomes a project fact: **the max |diff| and the count of non-matching pixels instead of a mean printed to one decimal** (100 flipped pixels in a million average to 0.0255, which prints as 0.0), and **the per-panel agreement as a column over all 248** rather than a sentence about the four, since the run already reads both sides -- that turns a frame verified on one page into a count. Also flagged: the join was made by rewriting the rect `pid`'s `__` to `_` to reach L's `page_id` (248 rows, none unresolved), which works but is a join on a name in a project that has been bitten by that; the rect file carries `content_fingerprint`, so one cross-check on the fingerprint validates the string transform. **Both asks came back within ten minutes, measured on all 248 and with a separate instrument from the run**: `max_abs_diff` 0 and `n_px_diff_nonzero` 0 on **248/248**, `inside_canvas` and `shape_equal` 248/248, and the shift control giving a minimum of **7,190** differing pixels (median 287,322) over the 48 offsets within +/-3px -- so the frame is a counted check, not a one-page assumption, and "mean 0.0" is replaced by `max_abs_diff = 0`. **The fingerprint advice was the foundation's error and is now T94**: it identifies output content, not a page, and 132 of the 248 panels (35 of 65 pages) resolve to more than one `.clip`, so the key stays `page_path`. **The column was split on the next pass** -- `fp_unique_same_page` 116/248 beside `fp_page_in_candidates` 248/248, the old boolean matching the first on all 248 rows and the eight shared columns identical row for row. **That comparison was only possible because the copies relayed to L and M had not been overwritten**: the CSV was replaced in place under its own name while the covering note was versioned, so the predecessor existed only in the two trees it had been distributed to. T92 from the other side -- asked that data files carry the version in the filename too. **What the ledger did establish is sharper than the recount**: splitting the 846 `not_line_stat` layers by how many LINE conditions they missed gives 132 layers that missed exactly one, **and their ink 5.68 exceeds the 4.16 of everything adopted**; of those, **95 layers (ink 5.18) were decided by `black_among_drawn <= bad_max` alone** and 37 by `drawn_ratio >= drawn_min`. **So one constant draws the boundary**, and the question put to the extraction side is `bad_max`'s value, the role it was calibrated for, and where it was set -- the shape 規律9-2 is about, testable as soon as the provenance is known. Also: the nine `線画` layers dropped by the width rule all fail `bad_max` as well, the name rule dropped only 2 layers in 65 pages (`MASK` was already out on statistics), and `render_3d` concentrates 86 layers and ink 7.15 in 15 pages. **And the dropped layers' `visibility=1` / `opacity=256` says nothing**: the selector never reads either column. **The provenance of the constants came back at 18:09 and the ruling is that discipline 9-2 applies, in four shapes at once.** The extraction side answered from git and the code's comments only, writing "記録なし" where there was no record. **(a) Role**: `bad_max = 0.98` is, in their words, for *choosing the line-art layer when building rough/line pairs* and **was never calibrated to separate ベタ from line** -- and it is the constant that alone decides the 95 marginal layers carrying ink 5.18. **(b) Unit**: the other four (`bad_min` 0.30, `drawn_gray_max` 110, `drawn_min` 0.025, `ink_max` 0.55) were derived from the p5/p50/p95 of **composited page images**, and they say so: *the unit is not a layer, it is the composited page image, and it was carried straight into selecting layers*. A share of a page and a share of one layer are two quantities with one vocabulary. **(c) Pool**: that derivation is one work (`housei`, verified by eye on 53 pairs of `ako5`), now applied to 59. **(d) Circularity, and their own commit message says it**: 0.98 was picked by comparing 0.88/0.98/1.00 on 338 QC'd pages, QC reads the page's line image, so fill ink entering as line is invisible to it -- `de18f10` reads *coverage rewards added ink, so contamination looked like improvement*. **This is L's `preview_dark>=0.5` again.** So: **a number taken off the raster cannot be read as a count of line against fill**, only as a count of what this selection rule passed. **What the ruling does not say**: 0.98 is not wrong -- it recovered the real line art on 21 of 51 bad pages against 28 of 287 good pages gaining a layer -- but **nothing between 0.98 and 1.00 was ever measured, and no per-layer ground truth (fill or line) has ever been held against it**; the comment that non-pure-black pixels mean an antialiased line is an explanation, not a calibration, and they say so. **And the retracted 519 came back as the inventory of the far side of the boundary**: raising `bad_max` to 1.00 would newly admit **519 layers on 74% of the good pages -- balloons, paper furniture and fills**, the same number and the same composition as the withdrawn "519 production fill layers" of 2026-10-19. The retraction was right (it was not a count of fill layers) and **this is what it was**: the set this constant excludes. So "what would moving the bar to 1.00 buy" is half answered already, and the other half cannot be answered without per-layer ground truth -- the candidate being M's known-cell 8 pages, the only set carrying the author's own standard. **The request to the extraction side therefore changes shape**: not *move 0.98* but *let us build a small per-layer ground truth so the boundary can be calibrated at all*, which is L, M and the author's work, not theirs.
- **T87** `has_mask` is **True on all 2,963 layers**: a column with a check's name that performs no check, and an attempt to split on it comes back empty. The same family as "names are not checks" -- silence reads as normal operation.
- **T88** "Not in the inventory" is not "does not exist". The delivered CSV carries folder rows with `parent_id` / `order_in_parent`; the pilot manifest **drops folder rows**, and reading only the manifest turned an omission into the ground for an instrument rule. Rebuilt from the tree: 167 -> **183** bases, **16 of which are folders** named for body parts, and components 504 -> **823**.
- **T89** `drawn_ratio` is **not a clipping layer's coverage**, because the export does not apply clips: effective against declared has median **0.042** over 303 of 698 layers and **135 layers declare the whole canvas**. The trap reaches the denominator as well as the numerator -- taking an occlusion denominator as "the sum of all layer alpha" returned **1.0000 on 7 of 8 pages**.
- **T91** An order handed over outside the written channel leaves no record on **either** side. The 2026-10-24 z-order request produced a 7,362-row delivery in ten minutes, and on 2026-10-25 the foundation found that **neither that request nor the panel-geometry one exists anywhere in the extraction side's tree** (its `inbox/` ends at 10-24 17:26; a whole-tree search finds only the delivered CSV, the delivery note and the generating script) while the foundation's own `outbox/` held no copy. The delivery cites a request its recipient does not hold. **The rules had no slot for it** -- `reply_*` was defined as track-bound -- so the gap was in CLAUDE.md, not in anyone's attention; `outbox/push_<topic>_<date>.md` with a one-line delivery record is now the slot (Track L's diagnosis and proposal, adopted). Retroactive copies exist for both orders, marked **route unknown**. **It recurred on 2026-10-25 in the other direction**: a 99MB structure DB and its documentation existed for over an hour with no notice in the extraction side's `outbox/` and nothing in the foundation's `inbox/`, and were found only because Track L read the other tree. The foundation's receipt asks for **a one-line notice at creation** rather than a written-up one -- the placement is the record, and the body can be read later. **And the receiver's own name is not stable: 2026-10-25 19:27 a network drop restarted this session, `lineart-ec` -> `lineart-b9`**, so every letter already sent names an address that no longer exists -- the same shape as the 386-byte notice that went nowhere that morning. The letters are records and were not rewritten; a one-page notice of the new name went out on the written channel instead. **The file in `inbox/` is the thing that survives a session; a session name is good for as long as the session is.** Nothing was lost in the outage: the tree was clean and pushed to `4f01d6e`, and the extraction side's `outbox` held nothing newer than 18:38.
- **T92** A replicated file is a copy, not a record. `CLAUDE.md` is copied into every tree at branch time, so a constant corrected on the foundation keeps answering with the old value: on 2026-10-25 **10 of 11 track copies still named `192.168.1.9`** after the extraction host had been corrected to `.5`, and Track L, following its own copy, concluded the machine was unreachable. Now audit check 10, scoped to live trees. **And the family reaches outside git-tracked files**: an installed plugin whose `installLocation` is a working tree serves whatever that tree has checked out, so a branch left switched would change the disciplines in force for every session on the machine. (And `ping` is not a liveness test there -- Windows drops ICMP, so `.5` is silent while `ssh` works.)
- **T97** A seed built from `hash(str)` makes a run unreproducible, and a check that lives inside one run cannot see it. Track L's `place()` seeded on `abs(hash(pid))`; **Python randomises string hashing per process** (the same string gave 87097 / 76801 / 59348), so **the control landed its regions somewhere else on every run**: the control's `cov_honest` median moved **0.0945 -> 0.1178** (ako6) and **`r0` tripled, 0.0213 -> 0.0655**, which moves the floor `control + 3r0` from **0.1584 to 0.3143**, and the placeable set itself changed, so the real arm's population differed run to run. **The earlier repair is why it stayed hidden**: L17 §5 made both arms share one seed *within a run*, which guarantees the within-run comparison and says nothing about reproducibility across runs. **So a known cell has to be a value recorded in an earlier run, not a value computed twice in this one** -- the run-level form of `moat` rule 2 (*an agreement between two parts of one pipeline is not an anchor*). Fixing `PYTHONHASHSEED` is not the repair, it moves the dependency into the environment; `zlib.crc32` is. **And a bar that is a function of the null must be computed in the same run as the arm**: only absolute constants (A 0.30, C 0.50) may be frozen across runs, or the luck of one draw is frozen into the bar. The foundation's published floor numbers (0.1584 / 0.2348) are **withdrawn as numbers while the formula stands**. `r0` as max-min also grows with the seed count, so 5-seed and 10-seed `r0` are different quantities; the registered max-min stays primary because it is the stricter floor, with `3xSD` printed beside it for arithmetic reasons that have nothing to do with the arm's value. **The disclosure that L saw the real arm before the fix is harmless and the reason is numerical**: the arm (0.0953 / 0.0773) sits **below its own run's control** (0.1178 / 0.1150), and every admissible floor sits at or above the control, so no choice of floor could pass it. **L17 is being re-measured** -- not because the direction is in doubt (the gap is 0.31 against a drift of 0.02-0.05) but because all four of its numbers are outputs of an instrument that does not reproduce; the report is to be headed *the defect was fixed and the conclusion did not change*, and the old numbers stay with that mark on them. The degeneracy gate needs no re-run: it reads the real data and never places anything.
- **T95** The pool is the duplicate cluster, not the work and not the year. Three tracks produced the same shape on 2026-10-25: Track M finds **63 cross-work clusters and 248 of 872 pages (28.4%) involved in duplication**; Track L finds **24 of its 551 regions coming from one cross-work pair**, so `ako6` and `ako7` are **not disjoint pools**; Track H's year table silently pooled `2019` with `2019b`. And the key collides across year folders: `ako7\ako7\page0073.clip` **exists under 2018, 2019 and 2019b as different files** -- the layer-selection ledger and the koma manifest hold the 2019 one (34 ako7 pages, all 2019), M's 872-page delivery holds the 2018 one, and both records were right. So **only the full `page_path` identifies a file** (T94's other half: not the fingerprint, not work+page), and **the author's confirmation attaches to the file they were shown** -- here the 2019 path. Operationally: the numbers do not change, the word "pool" does. Whenever two or more pools are named, the crossing clusters and their page count go in the same paragraph. CLAUDE.md §3's *a generation is a pool* is right and insufficient: **a year folder is not a pool boundary either.**
- **T96** Measure the resolution of the numbers you were handed before counting dirt in them. Track M nearly reported **167 layers with over 5% relative error** when `drawn_ratio` was merely **rounded to six decimals**; the same evening the extraction side's `ink_ratio` turned out to be rounded the same way, which puts up to `page_px x 5e-7` (5-25 px) per layer into any count converted back to pixels, and they re-ran to print exact `ink_px`. **Two independent instances in one evening**: resolution is a property of the instrument and it lands in the same digits as the defect.
- **T94** A content fingerprint identifies content, not the page it came from, so it cannot be the join key. The foundation advised joining the panel corpus to the manuscripts on `content_fingerprint` instead of a name transform; the extraction side checked and the advice was wrong. The fingerprint is computed from the **output** line/rough JPEGs (not the `.clip`), and **132 of 248 panels -- 35 of the 65 pages -- resolve to more than one `.clip`** (`ako6\done4\ako6-14.clip` beside `done5`, `2019\ako7\...\page0068.clip` beside `2019b`), so the fingerprint is one-to-many there: 116 panels hit one source, 128 hit two, 4 hit four. The name-based join was *consistent* with it (the named page was among the candidates for 248/248) but the fingerprint cannot decide between duplicates. **The key stays `page_path`.** Two further readings: **duplicate-content sources are the majority case in this corpus, not an exception** (over half these pages), which is a fact for any work that has to say which version was used; and the check file's own `fp_page_equal` column is True on only 116 of 248 because it means *uniquely resolves*, while the claim it supports is *is among the candidates* -- a boolean travelling without its definition, the shape of discipline 1.
- **T93** The remedy quoted the sentence it removed, so the check flagged its own fix. The pointer files that replaced the frozen ledgers explain that the old copy said *"this file is the first document to read"* -- which is exactly what check 8 searched for; a date clause hid it until a ledger commit on 2026-10-25 lit up ten repaired trees at once. A check whose predicate is a quoted string matches the text about the problem as readily as the problem.

### 素材の性質と外部標準(塗り)

- **T20** A ground-truth label answers only the question it was collected for (ruled 2026-10-18, the foundation correcting its own cost design)
- **T42** The cached intermediate is not always the one that made the data, and the foundation approved the wrong path
- **T60** A layer can be present, intended and fully specified, and still measure as empty, because its content is a render-time effect rather than pixels.
- **T64** The composited-page route is dead for area boundaries, answered 2026-10-24: `CanvasPreview` is a quarter of native and the native composite is not in the file.
- **T67** The whole-corpus tone scan finished 2026-10-23: 7,667 pages over four groups, three failures (`CHNKSQLi not found`, corrupt files)
- **T68** The "against the radius-4 estimated mask" caveat has a number now, and the number is worse than the caveat implied.
- **T70** This project's line rasters DO NOT CONTAIN a large part of the fill the author separated -- two works now, and it reaches further than the mask.
- **T71** The separation is progressing, not static, and one step of the author's "back and forth" now has a direction.
- **T72** An external standard for areas is now within reach, on a distinction Track L made after finding a hole in this file's own map (2026-10-23 night).
- **T73** The user redirected the fill line on 2026-10-24, and the reason is not derivable from anything here.
- **T74** How much cleanly separated material exists: 35 koma, and the distribution is what matters (Track L's L6, pre-registered at mixed-layer contribution < 0.10 and leak < 0.10). **CORRECTED 2026-10-24: the 35 are a product of the dataset raster, not of the manuscripts.** All 35 are panels whose area layers the raster dropped -- their area layers' `in_raster` median is **0.1395** against 1.0000 for the LEAKY set -- so "panels where the fill is not present" was "panels where the raster lost the fill layers". The count stands as a count of raster behaviour and not of clean material.
- **T75** Track L's L7, registered 2026-10-24: validate Track J's assets on the clean data, with the false-positive rate and a stratum that can move the blame off the mask. **The entry still says "running"; L7 was measured and then RETRACTED the same day** on a hole the user found -- the retraction is inside T72, and its numbers are kept there because the retraction is the content.
- **T80** The koma corpus holds the same page more than once, at different drawing stages, and a naive check reads that as duplication.
- **T81** Every solid-fill number on record inherits a mask that has never been checked against anything.

## Approaches Never Taken Up

**The reasoning is in `doc/approaches_not_taken.md`** (moved 2026-10-24,
verbatim, 231 lines). This section keeps the claims, so that nothing below is
re-asked from scratch and "we must have tried that" is not assumed.

**The paper scout's nine reports, triaged against Track K's spec (2026-10-19).**
- **One entry is of the right type** -- individual fibre separation in 3D fibrous
  materials -- and it is the one the scout rated **lowest**, while the five
  highest-rated entries are losses Track K cannot use. **The mis-rating is
  systematic, not random**, so the fix is a focus-area line, not a re-read.
- **The reusable reading is about the instrument**: a 13.3x precision lift is
  **information**, and a loss function is not information. K's question is *what
  distinguishes a junction from a continuation*; the scout's focus areas ask
  *how to train for connectivity*.
- **CtrLoRA (ICLR 2025, arXiv:2410.09400) would reopen a closed line.** It is the
  only paper in the log that treats the **condition encoder** as the suspect, and
  its budget fits -- but the ControlNet lineage is closed by Tracks A and B, and
  **the bar for reopening is a new yardstick, not a new paper.** Logged, not
  promoted.
- **One of its premises was wrong and this file repeated it**: the frozen VAE's
  round-trip is not an unmeasured ceiling. Track D measured GT round-tripping at
  f1 **0.9605** (SD1.5) and **0.9806** (SDXL) on lineart_family against a best
  trained model of 0.25, so the "run the round-trip test first" caveat is void.
  T23's trap -- cite the file when stating a fact about a track's material --
  walked into a second time, and the entry to cite was four hundred lines up in
  this document.

**ControlNet + IP-Adapter has never been tried, and was never proposed**
(surveyed 2026-09-30 across every worktree's docs, scripts, branches and the
Directions 1-9 survey: zero occurrences of the name or of the concept). **It fell
off by expiry, not by judgement** -- the 2026-08-08 note said to revisit style
adapters *after* the paired fine-tune worked, and Tracks A and B both closed
below the preprocessor baseline, so the revisit never triggered. Tooling was
never the constraint.
- **It does not address lesson 8, because that failure is spatial.** Where the
  conditioning map lacks a GT stroke the model draws it 9.3% of the time, and
  output skeletons sit near the map (36%) rather than near GT alone (9%). A
  pooled, unlocalised embedding touches neither which of the preprocessor's
  1.4x-GT strokes to delete nor the ~16% of GT stroke length the map never finds
  (oracle recall cap **0.604**). Architecturally it injects at `attn2`, the layer
  `scripts/pnp_line_from_rough.py` deliberately left alone.
- **Where it would fit is a different axis**: the one thing training demonstrably
  moves is **tone** (near_white 0.13 -> 0.83; fixed-`t` loss correlates with
  paper white at r -0.98 and not with f1), which makes it Track E's question and
  most natural in a line-to-line framing -- **and no line-to-line theme exists in
  this project.**
- **If it is tested, the honest first move is inference-only**: public ControlNet
  at `lineart_coarse` cs2.5 plus a line-art tile as image prompt, scored with the
  conditioning map's own baseline beside it (lesson 6) and Track D's `gt_only` /
  `vs_condition` columns. Using GT as the reference leaks, so like the delete
  oracle it bounds the mechanism rather than demonstrating a score.

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

**Where the work is, as of 2026-10-24.** Two fronts: the words line's unit
(Track H, on the author's time, with Track G waiting on its unit and holding an
open slot) and **the fill line (Track L)**, which is the only one with work in
flight. Tracks C, E and F are **held, not closed** -- kept in place to be
re-measured when a new yardstick exists, each trigger named under Next Actions
item 3. Track J closed on a clean negative 2026-10-23 and Track K is dormant.
The Track Ledger is where status lives; this section is the route.

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
reaches back into the words line.** The author's framing (Working Discipline 6),
in the author's own terms: in hair, a **region** -- an area closed off by lines --
is sometimes filled and sometimes not, and **if the line can be recovered from
the region, fills become much easier to handle**, because a filled area is not a
separate kind of ink but a region that has an outline. Separately, the
**highlight fills in hair are a kind of texture**, and splitting that from the
line art is what keeps a pipeline from growing branches it cannot otherwise
avoid.

The consequence for this route, and it is the reason the fill line is not only
Track J's business: **the reason `cluster_set` mixes parts is overwhelmingly hair
and shading hatch** -- w342/w343/w344 turned out to be "arc, short stroke bundle"
holding hair, hatch and panel borders, three quarters of labelled brows share a
cluster with the bangs above them, and 14.6% of all strokes enter no cluster.
**So part of the words line's unit problem is a fill problem.**

**Three measurements on GT alone decided the shape of it** (Track J, 2026-10-10,
`results/region_closure_20261010/`, direction and size reproduced here from the
images with an independent implementation; the full figures are in Next Actions
item 5):

- **The 480px window is the wrong unit, by a wide margin.** On a 480 tile 87.6%
  (`housei`) / 92.9% (`ako5`) of solid-fill *area* sits in a component whose own
  boundary is outside the window; on a koma panel of the same material at the
  same drawing scale it is **26.4% / 48.5%**. A model shown a 480 tile is being
  asked about an object it is never shown whole. Some `ako5` tiles sit entirely
  inside a fill with no boundary in the window at all (**0.5% entirely filled,
  31.3% near-blank**).
- **The line art does not enclose anything.** Strokes alone (`ink & ~fill`)
  flood-filled give **1.2 regions per panel** on `ako5ver2` and `housei` koma
  (**5.7** on `clip_pairs_v3`), the largest background component is **100.0% of
  the background in every panel inspected**, and 95-100% of the fill area lies
  outside any enclosed region. **The author's own fill gives median 3.0 regions**
  (Track L, 2026-10-24), which is the number to compare against rather than 1.2.
  So "propose closed regions, then classify each" has no ground truth to be
  scored against on this corpus, and **any proposal that assumes a part or a
  region can be recovered as an area enclosed by the line art cannot be built on
  this corpus by that construction** -- a fact about the corpus, not a finding
  about Track H's or Track G's units.
- **And the rough does not carry the answer either**, which was the other fork:
  predicting which regions are filled requires the information to be in the
  rough. It is not, so the task is proposal rather than prediction -- Track J's
  verdict (item 5) closed the prediction formulation, and what replaces it is an
  external standard, which is Track L.

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
listed under Next Actions -- not an active build-out. The sections that used to
follow this one and still described that era were archived on 2026-10-24; see
"The 2026-07 raw-extraction era" below for where they went.

Old leak-era `shape1` scores are not adoption targets. Use clean eval metrics
and montage review only as current references. Evaluate line art with the
BSDS-style one-to-one matching F1 (`gt_bsds_f1`), and always report
`line_width_p50`, `ink_ratio`, `fill_ratio` and the conditioning map's own
score alongside it -- see the nine lessons at the top of this file, and
**never the training loss** (lesson 7).
## The 2026-07 raw-extraction era (archived)

Twelve sections moved to `doc/archive/current_20260726_raw_extraction_era.md` on
2026-10-24, verbatim: the ako5ver2 region review and keep281 manifests, the
native re-materialization and the `--min-tile-score 2.5` tile filter, the
fitness/housei native-strict sources, the alignment root-cause, the housei koma
segmentation, and the Direction 6/8/9 survey. **Five of them were named "Current
<something>" and had been frozen since 2026-07** -- the name was the only thing
keeping them in the reading path.

The live route to that material is the focused docs, not the archive:

- procedure and gates: `doc/preprocess/EXTRACTION_RULES.md`
- variable-region extraction: `doc/preprocess/region_dataset_extraction_policy.md`
- manifests and materialization: `doc/preprocess/region_materialization_policy.md`
- dataset-specific exceptions (the rough-only umbrella cases): `doc/preprocess/raw_dataset_extraction_knowledge.md`
- which architecture families were tried: `doc/model_directions.md`, `doc/model_results_summary.md`

One standing policy note from that period is still worth reading where it sits:
**a tile score is not a content-match guarantee, and thumbnail QC hides it** --
found at full resolution in the strict pass's tail.

## Next Actions

### いま宙にあるもの (2026-10-26 未明、仕切り直し用。ここだけ読めば続けられる)

**ユーザーの方針 (2026-10-25/26)**: *この時期の原稿制作は運用が安定しておらず、作業途中の
コピーが残ったまま、命名規則も安定していない。これらは名前や統計量だけで判断しづらく、
こちらに問い合わせてもらうのがベターだ。* -> **版・重複の同定は作者に問い合わせる。名前や
統計量の錨はこれ以上探さない** (今夜落ちた4つ: 名前順 0/42 はタウトロジー T100 /
`ProjectName` 61/65 同値 / `superset` は作者に 1/3 / 更新時刻は一括複写 T99)。
通達 `outbox/notice_version_identity_ask_the_author_20261026.md` を M・L・抽出側へ配送済み。
**作業途中のコピーは本番より層が多いことがある**ので「層が多い方が後」は成り立たない。

| 宙にあるもの | 状態 | 次の一手 |
|---|---|---|
| **依頼4 本走** | 抽出側で 23:30 開始、779頁・7,773層 | 基盤は何もしない。終われば M が照合 |
| **版の反転 27塊** | 作者の回答で札は決定 (23 の `b` + 83・85 + gal 31・32) | M が `version_roles` を更新。**バイト照合は描き直し(最大25頁)の判定だけ** |
| **原稿の国勢調査 + webUI** | **新 track `census` を 2026-10-26 に新設**(`../lineart-census`、branch `census`)。ユーザーの指示: *都度チェックするには重すぎる。集計して持つ。集計データなので github にあげも OK。確認する webUI があるとありがたい* | **決定済み**: **公開は段 A のみ**(年・形式・種別ごとの件数と分布。**作品名・層名・頁パスを出さない** —— `origin` は匿名で読める)/ 全量は `data/census_full/`(追跡外)/ **順序は 器具 → UI**。仕様 `doc/census_spec.md`、引き継ぎは track の `doc/initial_notice.md`。**基盤は実装しない**(規律5) |
| **`2017\ol` は原稿集に入るか** | **作者の裁定 2026-10-26: 「はいる」。スコープの保留は解けた。** 但し書き **「29ページのうち21ページ以降はなぜか同じページが入ってるので21ページ以降落としてもいい」** —— **その「21」がどれを指すか決まっていない**(基盤の検算: 並びの21番目 = `page0049` 以降 **9件** / 頁番号>=21 = `page0031` 以降 **26件** / 連続する並び `0031`-`0057` の21番目 = `page0051` 以降 **7件**。**29件すべて (バイト数, 層数) が異なるので大きさには現れない**) | **1頁も落とさない。** 作者の言葉は**許可であって要求ではない**(M の判断、基盤が是認)。繰り返しは層の表の署名で同定し、**同定した頁を作者に見せて確認を取る** —— 測定で作者の読みを置き換えない(規律6)。**母集団に入れるのは同定の後**、差分は理由別に印字 |
| **~~`2017\ol` は原稿集に入るか~~(旧)** | **新規(2026-10-26)。** 29頁・層中央225・150MB/頁・画布 ako6 の1.67倍・`image_material` **1,003層**(872頁全部で1,819)。**3か月どの数にも入っていなかった**(T115) | **1行の問い。基盤から出す、順番はユーザー。** ラスタの発注(約2,600枚 / 1.1GB / 抽出側の機械時間 数時間)は**この答の後**。時間制限は無い(180秒は8月の標本調査だけ) |
| **まとまりに名前や境界は在るか** | **新規(2026-10-26)。作者が自分から「複数の面のまとまりとなってはじめて意味を持たせられるものも多い」と述べた。T125** | **列の中で最も効く問い** —— L17/L18/L35/L36 と色トラックの標準を**どの水準で定義するか**が決まる。**ただし文では聞けない**(T113)。**原寸の切り出しに作者がまとまりを印す形**が要る。**器具を作るのは track**(M が直近に 275問・出口1.1% の UI を作った)。**出す順はユーザー** |
| **`線画`/`black`/`MASK`/`white` の材質** | **新規(2026-10-26)。T117。** `線画` 40層・`black` 97層 | **場合2(接する別材質が1成分)の測定の前提。** 1行で聞ける |
| **作者への問い 8件 (A-H)** | 通達 §4 に完全パスで一覧 | **出すのは基盤から、順番はユーザー**。A が 9塊で最も効く |
| **`preview_dark` の出所照会** | **回答到着 (2026-10-26)。定義は CSP が保存した合成済みプレビューの上の暗さで、`>= 0.5` は**ベタ抽出**のために引かれ、線画の較正はされていない** | **T116 で確定: 納品物に対する被覆の指標として使えない** (L の自然な対照対6つ。同じ層が版で最大 0.1092 違う)。門として残るのは `l12_panel_regions`/`l8`/`l15` のベタ選別だけで、**margin 未検証** |
| **L34 半径の引き直し** | **完了・判定保留。2つの条件は一度も重ならない** —— beta の中央が 0.95 を満たすのは `r=2` のみ(そこで line 0.7963)、line が 0.05 を切るのは `r>=8`(そこで beta 0.7678)。KC38/39/40 PASS、腕 beta 496 / line 1,509、落とし0 | **(C) の数は出さない。** 機序は T121 —— **借りた役割が尺度を越えられなかった**(作者のベタ層自身が r=4 で墨の11%を失う)。**道1(納品 beta 層そのものを標準に)を基盤が承認**、条件は事前登録・T120 の継承(14/496)を登録に明記・道2 の代わりにしない |
| **L31 単位のずれの頻度** | **完了・棒を越えた = 「多い」** (12:21 終了、報告は 14:32。T118 の待ち受けバグで2時間)。**主値は半径に依存しない読みだけ**: ako6 **0.1168 / 0.1341**・ako7 **0.3150 / 0.2176**(L が自分で訂正。0.3411/0.5600 は opening 後の量で `radius=4` に依存し、L34 が保留に落ちたので引用できない)。**【作者の答で読みが半分逆になった 2026-10-26】場合3(同部位が白抜きで割れる)は作者も2と数えるので一致** —— **ずれているのは反対側、opening が作った隙間**(ako6 48/73・ako7 49/112)。**数は撤回しない。訂正は「内訳のどちら側がずれか」。** そして**「1,186 は下限」とは書けない**(場合2 は足りない側、場合4 は多すぎる側で、両方向に境界の付かない誤差)。**場合4 は作者の問いではなく `radius` の較正の問いになった** —— 作者の答が、聞けない問いを測れる問いに変えた | **(B) の判定は確定。** ただし**厳しい読み (生の墨だけ = 作者の隙間) では ako6 0.1168/0.1341 が棒を下回り、「多い」は ako7 が単独で支えている** -> **プールを跨いで平均しない**。(C) は `radius=4` の引き直し待ち、(A) は T117 |
| **L の45層の原寸切り出し** | `results/l28_preview_dark_crops/` に 45枚、質問文は L の `outbox/`。**送っていない** | **ユーザーの合図待ち**。作者の時間の列は4つ |
| **L18 の標準の構成** | **L35 が発見(2026-10-26)。L18 の標準は作者の納品ベタの 41.4%(205/496)しか覆っていなかった**。作者だけ291のうち **289 は L18 の腕の入力側に居た** | **撤回ではない。** 比較の言明は立つ(腕も対照も同じ標準、床は走行内の対照=T97)。**絶対値 `cov_honest` 0.0953/0.0775 は「300層の proxy 標準」に対する被覆で、作者のベタに対する被覆ではない。** 基盤が L に出した1問: **289層は腕と対照に同じ形で入るか**(入らなければ T112 の領域) |
| **L18 / L17** | **両方完了。L18 (`6e09517`) は4セルすべて A×B×C で FAIL (KC30 PASS)、L17 (`31c4ef1`) は棒から約5 `r0` で FAIL、優越は未確立。`--exclude-gapfill` 版が走行中 (2時間)** | 基盤が両方の算術を検算、印字された数は全数一致。**L へ返した指摘5件** (ako6 の枝が変わった / 保留の主語 / 引きを増やしても決まらない / P35' の `k0` / 除外大面 17→15) |
| **A/B/C の数え直し** | **止めてある** (`ink_px` はアルファ >= 128 で灰色の材料が重み0) | `ink_px` の `kind=raster` 行の定義回答待ち |
| **作者の時間の順 (決着)** | **短い4件を全部出す、L19 は未回答の問いの後。発注は「問題ないと考えているが先に質問」= 見込みは前向き、承認は未発行** | 3件は発送済み。**L19 は出せる状態で待機**。見積りは現行40頁で **約3.3時間 (暫定・按分)** に下がった |
| **作者の手元にある問い** | **1-B の A〜H (7件) / M の2問。1-A は回答済み** | 基盤が受けて逐語で台帳に入れ、L・M・抽出側へ配る |
| **34層の3択** | **紙ではなく L の UI で** (作者の指示)。45層 -> 現行40頁で **34層** | L が UI に載せる。名前・`preview_dark`・`keep4`・`black_among_drawn` は画面に出さない。門を混ぜる |
| **M の原寸切り出し** | **現物が無い** (「必要の言明」であって束ねた切り出しではなかった) | M が作成。列の位置が上がった |

**E60 は宙に無く、決着して実行も済んでいました (2026-10-26 にユーザーの指摘で発覚)。**
この表は E60 / E240 / やらない の三択を*ユーザーの判断待ち*として載せていたが、**ユーザーは
2026-10-25 に E60 を選び、L が L27 として事前登録し、作者が答え、結果も出ている** ——
作者 **64問 / 3.8分 / 中央2.0秒**、枠は掃き出された描画層 2,963 から L25 の 90層を除いた
無作為60層 (seed 20261026)、L25 の `E_random` 15層と同じ枠なので**合わせて74層**、
門 (既知セル5枚) **5/5 同じ答**。器具 `tools/evaluation/l27_bounds.py`。
**この台帳は 43/54 を「動かないもの」の一覧の中でだけ引用しており、L27 自身の記録を持って
いなかった** ので、ここに入れる:

| 定数 | 事象 | 件数 | 割合 | 両側95% |
|---|---|---|---|---|
| `black_among_drawn <= 0.98` | **この条件が拒んだ**層のうち「線」 | **43 / 54** | **0.7963** | [0.6647, 0.8937] |
| 同 | **この条件が受けた**層のうち「絵ではない/下絵」 | 8 / 20 | 0.4000 | [0.1912, 0.6395] |
| `preview_dark >= 0.5` | **L の母集団から外した**層のうち「完成の絵」 | 3 / 17 | 0.1765 | [0.0380, 0.4343] |
| `keep4 >= 0.95` | 「ベタの形」に振ったのに作者は「線」 | 0 / 17 | — | [0.0000, 0.1951] |
| 同 | 「線の形」に振ったのに作者は「ベタ」 | 1 / 57 | 0.0175 | [0.0004, 0.0939] |

**濃縮の無い枠で、この条件は落とした層の8割が完成の線画**。L25 の 19/19 (境界ぎわ) と
L26 の AUC 0.7750 (符号が逆) に続く**3つめで、母集団に最も近い読み**。区間の下側でも 0.66。
**通した20層のうち8層が「絵ではない/下絵」**なので、**条件はほぼ裏返っている**。
`keep4` は無作為の枠でも形の試験として保つ (0/17 と 1/57)、`preview_dark` は 3/17 で
`bad_max` ほどではないが清くもない。**言葉の意味も作者の指摘で直っている**: 「43/54 が線画」は
「線画が43層失われた」ではなく「**この1条件が線画を43回拒んだ**」で、実際の損失は門の全条件を
通して数え直さないと言えない。**そして基盤が売り込んだ「上界4.9%」は出ていない** ——
それは0件のときの形で、件数が多い側に出た行では役に立たないので、**L が両側95%区間を
結果を見た後に足し、足したことを明記した** (棒を動かしたのではなく、同じ量の正直な要約)。
自由入力7件 (`描き文字` 4 / `エフェクト` 1 / `フキダシ` 1 / `3D背景線画` 1) は登録の語に
入らないので数に入れず出した —— **役割の軸に名前が足りない、の3度目**。
**この行が宙に残っていた理由を記録に残す**: 判断・登録・実行・回答が同じ晩に全部済んだのに、
台帳の表だけが選択肢のまま凍っていた。**決まった選択は、決まった時に表から消す。**

**45層について、L は切り出しを見た時点で自分の仮説を落としました (`567fbed`)**: *`preview_dark`
は細い線を罰している* は**成り立たない** -- 1番 (`線画 hair`、墨 682,614) も7番 (墨 301,909) も
**大きな黒い面がそのまま入っている**。代わりに名前の形から **クリッピング** (下の層に合わせて
切り抜く設定) の仮説が立ち、45層のうち10層が `線画 hair`/`cloth`/`skin`/`face` 系で、
**`page0073` では5つが揃って落ちている**。**これが当たりなら `preview_dark` の判定は正しく、
L の条件を直す必要はありません。** そして **Track M の T-M3 (クリップ層のアルファは被覆でない)
と同じもので、独立に2回目** -- 二つの track が別の入口から同じ性質に当たったので、
**M の台帳から L に当てる価値があります**。筆幅の事前登録は**出さない** (仮説が変わったので
棒も腕も変わる。測るべきは z 順と下の層の被覆)。**走る前に聞く方が早い、の3回目**で、
今回は**切り出しを見ただけで仮説が1つ落ちた**。選別の内訳も印字済み:
875 -> 名前に線画/line なし 674 -> 1万画素未満 129 -> `bad<0.95` 27 -> **残った45**。
質問は **A 完成の線画 / B クリッピング / C その他**の3択で、**上から数枚で十分**と書いてある
(作者の負担を45枚にしないため)。

**L18 の確定 (2026-10-26 未明)**: 器具を直して測り直し、**結論は動かなかった** -- 4セルすべてで
A (`cov_honest>=0.30`)、B (床 = 対照+3`r0`)、C (`honest_rate>=0.50`) の**12判定が全部 FAIL**。
ako6 は 対照 0.0876 / `r0` 0.0255 / 床 **0.1641** に対し腕 0.0953、ako7 は 0.1001 / 0.0414 /
床 **0.2243** に対し腕 0.0775。**基盤が全部再計算して一致**、棒はこの走行の中の対照から計算
(T97 の直し)、KC30 PASS (L19 の551面が全部在る、欠け0 -- T102 を満たす)。
**基盤が固定した読み方が一つ**: ako7 の腕は対照の中央より低いので *反転* と読めるが、
`r0` の単位では **ako6 +0.30、ako7 -0.55** で**どちらも ±1 r0 の中**。`r0` は seed 間の
最大-最小、すなわち引きの運の幅なので、**正しい言明は「腕は対照である」**であって「低い」では
ない -- そう書くと次に誰かが逆向きの信号を読む。**FAIL の理由は「対照と区別できず、床にも棒にも
届かない」。** 撤回済みの数には印が付いた (`r0` 0.0213 -> 0.0655 は走行ごとに動いていた分、
床 0.1584/0.3143/0.2348)。
> **【2026-10-26 15時台 構成の追記。撤回ではない。】L18 の標準は、作者の納品ベタの 41.4% しか
> 覆っていなかった。** L35 が標準を納品 beta(496層)に替えて測ったところ ——
> **両方に在る 205 / こちらだけ 95 / 作者だけ 291**。そして**作者だけの291のうち 289 は、
> L18 が標準ではなく腕の入力側(語を出す側)に置いていた層**で、`keep4` は
> **中央 0.5960 / 最大 0.9491**(全部 0.95 の下)。
>
> **比較の言明は立つ**: 腕と対照は**同じ標準・同じ入力**に向き合っており、床はこの走行の
> 中の対照から計算されている(T97)。**標準が一様に欠けていれば腕と対照が同じだけ動くので、
> 「腕は対照である」は標準の構成に依存しない。**
> **ただしこの対称性は、この欠陥について確かめる必要がある** —— T112 で、
> 対照が腕の `continue` した行を採点していた実例が出ているため。
> **基盤が L に出した1問の答(2026-10-26、構成による)**: **入る。** `sub`・`lab`・`cl`・`P` は
> **腕のループの外で1回だけ作られ**、両腕が同じ `lab`・`cl` を読み、**違いは `R` だけ**。
> **289層が片腕にだけ効くことは構成上ありえない。この脅威は消えた。**
>
> **だが L18 はまだ閉じない。** L が隣で見つけた**別軸**が同じ比較に効く ——
> **両腕の採点した面集合が一致していない**(`SEEDS[0]` で 共通698 / 対照だけ**127** /
> **本来だけ 309**、seed を変えても同形)。機序: **等化されているのは「置けるか」だけで、
> 「置いた先に線のインクが在るか」(`adj == 0` の落とし)は等化されていない。**
> **規模は腕の3割**で、腕 0.0953 対 床 0.1641 が動きうる。**T112 そのもの**、かつ
> **登録は「置けるか」を明示的に等化しているので、登録の意図(対応のある比較)に対する不備。**
> **基盤が承認した走行1つ**: `regions_*.csv` に面の**名前**の列を足す(腕・棒・標準・seed は不動)。
> **事前登録を先に書かせる** —— 主値は登録のまま、**副値として共通集合に限った `cov_honest` を併記**
> (棒ではなく頑健性の読み)、帰結は先に明記(**共通集合でも FAIL なら L18 は FAIL で閉じる /
> 判定が変わるなら FAIL は撤回し問いは開く**)。**T109 により 127/309 は名前で取り直すまで目安。**
>
> **絶対値の言明は狭める**: `cov_honest` **0.0953 / 0.0775** は
> **「`keep4>=0.95` で選んだ300層を*半径4で開いた分割*」に対する量**であって、
> **作者のベタに対する被覆ではない。** **層集合だけでなく分割も構成の一部**であり、
> **別の分割の `cov_honest` は別の量**(L の提案を採用、T123)。
> **そして T123 により、この量は測るべきものについて単調ではない** ——
> 競合の規則があるので、**より正しい分割が、より低い値を出しうる。**
> **この2つの数を「作者のベタをどれだけ覆ったか」として引用してはいけない。**
> **設計の答え(下記)は比較に乗っているので残る**が、**「伝播は作者のベタに届かない」と
> 読み替えることはできない。**
>
> **L35 は L18 の検算にもならない** —— KC44 で289層を腕から外すので語を出す層が
> **1,681 → 1,392**。**腕の母集団が違うので引き算しない。**

**設計の答えとしては: 成分を伝う伝播で1クリックは伸びない。**
P36 = 9.2% の壁は残り、**L19 実測の 1,186面 ≈ 274分 (原稿65頁で4.6時間) がそのまま作業量**で、
**伝播では下がらない**。純度が成分の大きさで落ちる併記は**対照でも同じ向き** (0.1940) なので
判定にも読みにも使わない。
**The gap-fill exclusion turned out not to be a filter (L, `e3b4ff5`)**: removing a region
removes a word, which changes which component receives a word and therefore the purity
upstream, so **the excluded version cannot be made by filtering the full run's CSV** -- it is
a second measurement on a smaller population, not a subset analysis of the first. L put the
condition **into the instrument's startup gate**, so a run that cannot read the full version's
record stops at launch instead of after two hours, and prints the full version's A/B/C beside
its own, labelled *a record of the previous run* (same instrument, same seeds, KC30 passing).
**The foundation's one addition**: each version must print **its own control and floor**, since
the floor is a function of the control inside that run (T97) and the exclusion moves the
control too -- two arms against one floor is the next misreading waiting to happen. **A
reporting condition belongs in the startup gate**, which is T102 from the useful side: the
check that cannot run is the one that fails silently, so make it fail loudly and early.
**And one line from L worth keeping as a rule (`f4d31ac`)**: it had verified that the excluded
version cannot be made by filtering, but had not put that in the headline -- *what you verified
and what the reader receives are two different things*. A verified fact outside the headline
travels as the claim the headline makes. The table is now four columns per version
(arm, control, `r0`, floor) so two arms cannot be read against one floor.

**Three fronts and a queue.** Item 1 is Track H on the **author's time**, which
is this project's scarce resource; item 2 is Track G on the machine's, slot open
with its next assignment named; item 11 is the fill line, now Track L, and it is
the only one with something in flight. Item 3 is the three held tracks and the
trigger that would resume each, item 4 one data-side decision left from two
fixed bugs, **item 5 is Track J, closed 2026-10-23 on a clean negative**, items
6-10 housekeeping, none of them blocking, and item 12 a registration waiting on
H18. Every ControlNet track is closed and leaves nothing behind.

**The order the user set on 2026-10-23 still stands**: (1) item 11's GT-only
measurements, which need neither the GPU nor the author; (2) item 12's cost
baseline, which needs no instrument at all; (3) H18's read-out, before which the
marking stream is not to be touched; (4) then item 12's registration. The third
proposal of that day, conformal prediction, is ruled on in Working Discipline
item 8 -- adopted as item 12's mechanism, premature for the fills, not a track.

1. **Face-part words** (`../lineart-face-words`, Track H -- the front on the
   author's time, live session as of 2026-10-23, pushed `43a6e88`). It is a
   worktree of Track G's separate clone, so neither `git worktree list` here nor
   the audit can attribute it; the audit prints it as a separate clone, which is
   the best it can do.
   The opening hypothesis is refuted -- the words do not carry part identity
   (lesson 13) -- so the track moved up a level: the **unit** is wrong, and it is
   collecting line-level ground truth through the grouping UI. **373 faces
   complete as of 2026-10-23**, author still marking. Three things to watch
   rather than act on:
   - **The three measurements the ground truth was for are answered** (H6,
     2026-10-12, 68 faces): mixing **68%**, fragmentation **22%**, ink in no
     cluster **7%**, a cluster capturing a true group **19%** of the time. They
     settle it -- the regrouping mechanism is a **replacement, not a repair** --
     and the track has built one. Denominators differ from the 2026-10-09
     cluster-mark figures; the ledger entry has both.
   - **The held-out discipline is carried across two label sets**: the face
     queue inherits the instance queue's series groups and the same page-sha1
     rule, zero panels disagreeing, so a mechanism can be scored without a leak
     across label types. Protect this if anything proposes re-splitting.
   - **A rule's pass criteria are still unset by design** (the recorded
     candidate is precision 0.80 / recall 0.60 per label on the panel-held-out
     side, fewer than 20 positives counting as undecidable), to be fixed before
     the held-out side is read. The counting problem is already visible: at the
     random order's rate, 20 held-out positives needs roughly **1,000 labels for
     mouth and nose, 2,400 for ear, 7,000 for closed eye** -- so either the order
     stops being uniform, which would bias what "position predicts" means and
     was deliberately avoided, or the rare parts get a different instrument.
     The user's call.
2. **Panel generation** (`../lineart-panel-generation`, Track G) -- **the slot is
   open and its next assignment is named.** The question it halted on is
   answered: **placement is relative only once a reference frame is given, and
   the word contributes almost nothing inside it** (ledger entry 2026-10-09,
   lesson 13). The generative line (PosAR-W, set-AR vs two-stage, scale-up) is
   **on hold, not abandoned**, with one standing warning from its last run:
   **bits do not guarantee a visual pass** -- position beat the marginal by 0.9
   bits while the montage stayed unreadable.
   - **The named assignment: finish the sweep.** Which of the *remaining* Track
     F and G figures sit on the distorted position bins, and what they read when
     rebuilt -- Track F's "238 words have a placement preference that
     replicates", its weak relative placement (residual cosine 0.15 against a
     null near zero), and anything in either track's logs derived from
     `bins_of`. Some was computed on real coordinates and is unaffected. **The
     deliverable is one table in three buckets** -- unaffected / replaced,
     verdict holds / replaced, verdict changes -- so whoever resumes Track F
     reads it instead of re-deriving it. Same conditions as before: no GPU, no
     author's eyes, pre-registration first, Track F's files untouched.
   - **What it waits for is Track H's unit.** `tools/gen/framerel.py` takes a new
     unit's labels in place of the word id, so the distance to the **0.074**
     ceiling comes out in one run. Restricting to frontal faces was drafted and
     deliberately **not** registered: with a mixed unit it would measure a mixed
     unit.
   - **The instrument check failed three times and was stopped with no verdict**
     on the user's domain claim that placement is decided by relative position.
     v1 (32x32, offsets in reference-word sizes), v2 (16x16, direction x
     log-distance) and v3 (relation types fixed on the training half) all failed
     their relocation or range checks: relations do transfer (type-1 median
     distance 4.1 reference sizes) but a reference large relative to the panel
     makes "within 4 sizes" cover everything, and shrinkage tuned on the whole
     set erased the few tight pairs. **Grid probabilities have too many knobs for
     this question.** First candidate on return, not yet registered: a point
     prediction scored by miss distance.
   - Its own GT measurements that nothing else carries: per-word median
     detail-flip **0.275** (the global 23.2% was a frequency-weighted average),
     and the three-valued word-reliability split **49 / 164 / 254**, where
     reliability means **boundary stability, not freedom from mixed meaning** --
     which is the blocker itself.
3. **Tracks C, E and F are held, not closed** (user decision 2026-10-04). Held
   means the state is durable and **the trigger is named**, each in its own
   briefing:
   - **C** (`../lineart-stroke-selection`) -- not a new metric. Lesson 10 came
     from a human absolute judgement and found the *perfect* deletion unusable;
     what could move it is a different **output format**, Track E's untried
     re-render of the surviving strokes as a sparse drawing rather than a holed
     rough. Holed roughs are what was measured.
   - **E** (`../lineart-aesthetic-judge`) -- stopped because a comparison between
     two lossy outputs ranks the loss (lesson 11). It reopens when two outputs
     differ by something *other* than damage. Its unused build, tiles disjoint
     across arms, is at https://claude.ai/artifact/Mvvoqn3CAjtaNQ3ScqR9vG.
   - **F** (`../lineart-stroke-grammar`) -- closed its measurement phase on a
     clean negative (no inter-word grammar within a panel, five measurements
     agreeing) **measured on the vocabulary it had**. The trigger is a better
     vocabulary, which now plainly means a better **unit**, not a better
     codebook -- Track H's work. **Its record is split in two**: the word
     negative is untouched (word bits move -0.056 to +0.004, the cloze still far
     short of its 6.348 gate), while **every position figure is replaced and the
     placement claim is much weaker** -- marginal 7.563 -> 7.957, cloze band
     6.315-6.388 -> 7.785-7.792, and 86% of the apparent placement structure was
     the panel's aspect ratio (lesson 14). A corrected corpus identical in every
     other token is kept at
     `../lineart-panel-generation/results/posbins_20261009/corpus_fixed.npz`, so
     a resume starts from it.
4. **Both tool bugs are fixed (2026-10-09); what remains is the data-side
   decision.** The `housei`-prefix split heuristic resolves by looking for the
   file (`tools/evaluation/gt_paths.resolve()`) in all **nine** places it
   appeared, and `bipartite_match_f1`'s unbounded worst case has a shared guard
   (`tools/evaluation/bsds_guard.match_f1()`), byte-identical output on the
   verification batch. Why the past 192-tile numbers need no re-check is in
   T46.
   **Still open and still the user's**: the unapplied `manga_line` emptiness fix
   (downscale to 240px plus auto-contrast). It is a data-side change, not a bug
   fix, and **it must be applied to training and holdout together** or it merely
   inverts the mismatch (T49).
5. **Solid fills -- Track J (`../lineart-solid-fill`), closed 2026-10-23 on a
   clean negative.** The registered headline is **"no learned model is needed for
   this"**, and the distinction Track J insisted on against its own verdict:
   **"no arm predicts fill in this formulation"**, not "the non-learned route
   suffices". The 969-panel table (11,620 rows) passed both anchors before any
   verdict was read -- A1 agreed with the tree's own table **969 of 969** panels,
   A2 reproduced `score_holdout` to 1e-9 -- and every figure is **against the
   radius-4 estimated mask** (amendment J3), normal stratum, 956 panels.
   - **C4 removes both models first.** M0 falls to NC3 (mirror); **M1 falls to
     NC2** -- its score against its own panel's ground truth does not beat its
     score against the neighbour's. A model that cannot tell its own panel from
     the next one has not learned the panel.
   - **C1 FAIL, not narrowly**: best arm `rough_dark@budget` **0.0110** --
     +0.0026 over `blank`, +0.0072 over `all_black`, +0.0102 over its own NC2 --
     against a bar of **0.05**.
   - **C2 FAIL on both legs**: M1 - M0 **-0.0012**, M1 - NC4 **-0.0017**, against
     +0.03. **The leg NC4 made readable says the sharpest thing in the report:
     giving M1 another panel's context scores slightly higher than giving it its
     own.** The 4-channel whole-koma context is not used as context -- from an
     instrument that did not exist that morning.
   - **Not an artefact of threshold placement**: region-F1 over nine
     (MIN_REGION_PX, MATCH_IOU) cells, **sign identical in all nine**. Halving
     the match bar lifts M0 over `blank` by +0.0114, a fifth of C1's bar, and
     both C2 legs stay negative.
   - **What the models do**: GT fill is **10.40%** of normal-stratum panel area;
     M0 paints 0.71%, M1 0.34%. `npred/ngt` **0.531 / 0.328** -- about half the
     GT count -- while the **area** ratio is 0.068 / 0.033, so each predicted
     region is **14-30x too small** (IC7's speckle defect; the area ratio is the
     sharper index). M0 exceeds zero on **17 of 956** panels, M1 on 11, `blank`
     on 8. And **`all_black`'s pixel IoU is 0.1040 against M0's 0.0204** --
     blacking in the whole panel recovers **5-8x more fill pixels than a trained
     model**, while winning nothing at region level (0.0037). Any claim that a
     model learned where fill goes clears that row first.
   - **The floor and the ceiling travel in the same sentence as the verdict**:
     `blank` **0.0084**, the preprocessor's readable ceiling **pixel IoU
     0.0000**, and **every arm, learned and not, sits inside that band. Nobody
     won.** The 13 near-blank panels are **NO VERDICT** by C5 and IC6, printed
     only.
   - **Why the pool was worth the slot, and still is for someone else.** Over
     **12,000 tiles** across `ako5` and `housei` have never been trained on and
     appear in no evaluation set: GT there is **24.5% solid fill** against the
     lineart pool's **4.0%** (both ink-normalised; by tile area 4-8%, and the two
     are not interconvertible because the mask that measures area is a dilated
     opening), 27-38% of tiles are near-blank, the preprocessor fills **nothing**
     (fill_ratio 0.0%), and the delete-only oracle tops out at **0.3291** there
     against **0.7425** on the lineart pool. Inventory:
     `../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md`.
   - **Two unit findings that outlived the verdict**: a fill region does not
     close inside a 480px window (87.6% / 92.9% of fill area is in a border-cut
     component), so the unit is the **koma panel**; and the line art encloses
     almost nothing -- 1.2 regions per panel from the inferred mask, **median
     3.0 from the author's own fill** (Track L, 2026-10-24).
     The prohibition that still holds: a **non-adversarial** segmentation model
     was ruled outside the CNN+GAN prohibition (user, 2026-10-10), and the
     adversarial objective itself stays closed -- the `msgan`/`cleanup` i2i
     family, a GAN term, another `--unpaired-weight` sweep.
6. The unpaired-rough adversarial-branch idea is **dormant, not to be picked up
   for now** (user decision 2026-09-06): it belongs to the shelved CNN+GAN line.
   The remaining move, if ever resumed, is a GT-free continuity /
   self-consistency regulariser on the unpaired branch, then a re-sweep of
   `--unpaired-weight` between **0.003 (no effect)** and **0.03 (destructive:
   F1@2px 0.4175 -> 0.2381, fragmented high-contrast stippling)**. The `skima`
   rough-only pool itself (**626 pages -> 4,917 tiles**,
   `dataset/pairs_480/train/rough_unpaired_skima/`) still exists and may be
   worth using in the ControlNet context instead -- that would be a new idea,
   not this one.
7. Decide whether to rename the remaining `kurip`-named infra scripts, given
   `kurip` was a username (`match_kurip_regions.py` and others;
   `prepare_kurip_tiles.py` affects hamlabi too). Still open, unrelated to the
   work above.
8. Decide whether umbrella/layer-difference rows (ako5ver2) should be manually
   masked, tagged for future routing, or left held out. Still open.
9. **Keep this file from falling behind again.** The mechanism half works, and
   the half that does not is named: **Track H still owes a notice**, asked twice
   (2026-10-19, `../lineart-face-words/inbox/note_foundation_h_owes_a_notice_20261019.md`),
   with the reason stated plainly -- on one day Tracks K, D and J each sent a
   notice and **each corrected a number in this file**, so the H entry is likely
   to carry the same kind of error. Four lines are wanted, the fourth being what
   this ledger has wrong.
   **The structural cause of the 2026-09-30 drift is still true**: Tracks G and H
   live in a **separate clone**, so `git worktree list` run here cannot see them.
   When checking project state, list the sibling `lineart-*` directories and
   `git branch -r`, not just the worktrees. Whether to consolidate that clone
   back, and whether the words-line tracks should send notices at all, are open
   questions for the user -- the arrangement works, it is only invisible from
   here.
   **Track K is quiet indefinitely** (user, 2026-10-19), not for lack of time but
   because it does not have to happen now. What changed since moves the cost and
   not the decision: `labels/face_groups.jsonl` is at **518 lines / 402 orders**
   (417 when the above was written), so K3's frame of 40 new faces is **already
   covered** and the only missing input to restart is **40 marks**; its
   instruments and bands survive in `origin/stroke-topology` at `25e4dec`.
   Track H is explicit that its notice is not a request to restart K. **The
   connectivity line's next possible move is not K's**: it is carrying both
   tokenisations downstream, which changes Track F's output format and the
   model's input, so it belongs to F and G.
10. Revisit whether `--max-soft-ink-ratio` needs a per-source
   `diagnose_gate_funnel.py` pass for ako5ver2/hamlabi/fitness/gakuen (only
   housei has an established relaxed value so far); yield may be conservative
   for the others under the shared default. Still open.
11. **The fill line: line / ベタ / tone / gradient on ground truth, at region
    unit.** Proposed by the user 2026-10-23, **now Track L**
    (`../lineart-fill-region`, open, pushed `7a4df98`) -- so the live state is
    that track's ledger entry and its own `doc/initial_notice.md`, and what is
    left under this item is what the foundation owes.
    **The day's 1,045-line record is `doc/fill_line_record_20261023.md`**,
    verbatim: the layer inventory, the trial-work pick, the class scheme, the
    deliveries, the colour and tone surveys, Track J's door, and the retractions
    each produced. Read it before quoting any figure from that day.
    - **In flight, and nothing else moves until it lands**:
      `page_layers_priority65.csv` from the extraction side -- every layer of the
      65 priority pages (`ako6` 31 / `ako7` 34), pre-selection, with
      `visibility` / `opacity` / `has_mask` / `rendered` / `drawn_ratio` /
      `black_among_drawn` / `preview_dark`. ~85 s/page, started about 02:00 on
      2026-10-24. **Look at the distributions first**, then name the
      `page_id`/`layer_id` to export at native resolution with grey levels.
      **Do not fix the selection condition before seeing them** -- that is how
      L7 was retracted the same day it was measured.
    - **Five production classes, and they need different instruments**: line /
      ベタ / hard tone / blurred tone / gradient fill. The last two are both "an
      area represented only in part" in the author's words but are built
      differently -- a blurred tone has a **soft mask**, a gradient fill has a
      **hard mask with graded paint inside it**. Measured 2026-10-24 on 1,754
      gradient layers: 99.4% carry a mask with content, the mask edge is hard
      (p50 **0.744**) while the paint is soft (p50 **131.1**), and **mixing the
      two produces a meaningless number**. So **the gradient is the better area
      source, not the weaker one** -- the opposite of how this file first filed
      it. `背景グラデ` is the most area-like family (67 of 67 hard edges) and the
      smallest (69 layers).
    - **The population is measured, not assumed.** Tone: **17,695 used layers
      over 2,047 pages (26.7%) and 79 of 243 works**, five densities
      (11/20/10/30/22 = 96%) x three rulings (60L alone 91.5%, 99.8% with 55L
      and 30L) -- a small discrete set, which bounds what "hard tone" has to be.
      Production fill: `ako6` **781** layers over 107 pages, `ako7` **399** over
      71. Both works are also top for tone, so taking tone up later needs no new
      work selection.
      **Join on `(year, work)`, never on the work name**: the name pools
      `2018_ako7` + `2019_ako7` + `2019b_ako7` and inflated this file's own
      recomputation to 1,140 against their 484.
    - **The motive, in the form that survives** (Track J's wording, replacing the
      retracted one): **a person can look at the outline and tell which area is
      filled; cheap closing cannot extract it; what can is unmeasured.** J
      refuted *a per-pixel U-Net against the radius-4 mask at this scale*, and
      asked that this be an independent track so a clean negative closes as a
      negative -- **a region proposer is the seat J keeps open**, and is neither
      the morphology nor the U-Net already refuted.
    - **Two retraction mechanisms from that day, both adopted.** When a
      retraction pass and a briefing rewrite run on the same day, **the store
      list is structurally one short**, because a new store is being created
      while the pass walks it -- Track J found the fourth copy of the enclosure
      framing itself and now posts one line to `inbox/` on any day it rewrites
      its current-state section. And a caveat attached for disclosure
      (amendment J3, "against the radius-4 estimated mask") **must not become a
      reservation**: the mask is shared by every arm including `blank`, so it
      bounds absolute claims about recoverable fill and does not weaken C1-C3.
12. **Pre-annotated marking: let a model propose the groups and have the author
    correct them -- under Track H, registered separately, and not before H18
    reads out.** Proposed by the user 2026-10-23, on the reasoning that
    correcting a proposal may cost less than marking from blank, and that it
    may become usable as the label count grows even if it is not usable now.
    - **Not a new track.** It needs Track H's UI, its labels, its model and its
      registration practice -- all four. A separate track re-derives them, and
      Track I's first probe is this project's recorded case of what
      re-derivation costs. But its success criterion is **author time**, not
      group recall, so it needs its own registration and its own bar inside H.
    - **It must not touch the current marking stream.** H18 is live and reads
      out on orders 401-500 marked from blank, with a +0.025 bar; pre-annotating
      them mid-stream puts a second variable into the data-scaling slope H16
      and H17 measured. **Start at order 501.**
    - **The load-bearing hazard is anchoring, not accuracy.** If the author
      corrects machine proposals, the labels become partly a function of the
      model being evaluated -- and these labels are ground truth by the user's
      explicit exception to "names are interpretive only". Two cheap controls:
      **flag assisted faces** in `face_groups.jsonl` (it already carries
      `annotator`, `status`, `usage`) and keep them out of any population used
      to score the model, or score with and without; and **measure the
      anchoring directly** by putting a small set of faces to the author twice,
      blank and pre-filled, interleaved, comparing the groups that come back.
      Track K's author-judgment UI is the instrument for the second.
    - **The cost baseline exists already and needs no instrument.**
      `face_groups.jsonl` carries a `datetime` per row, so **time per face from
      blank is recoverable from the rows already on disk** (successive
      differences, breaks capped). Measure it before building anything, or
      whether it got cheaper can never be said. This is discipline 2's known
      cell for the cost question.
    - **Both branches of H18 lead somewhere this matters.** If the slope holds,
      the next hundred faces get cheaper. If it does not and marking stops,
      pre-annotation is **the only thing that reopens the data route at all**.
    - Mechanism: Working Discipline item 8 (conformal prediction, class-
      conditional, calibrated at face level).
13. **The dirtiness of the line art -- drafted by L and M, and two things wait
    on it.** Opened 2026-10-25 on the user's decision. The author reads part of
    Track H's ceiling as the material: the extraction program's line art is
    dirty and under-thinned, and H's marking set shows unnaturally cut panels.
    **H is paused and the body-and-hands track is deferred until this moves**
    (item in Track H's ledger entry). **The case is not H's to make**: the user
    says it is already visible in L and M, so each of those tracks sends the
    foundation `note_<track>_dirtiness_*.md`, the foundation merges the two into
    one request and pushes it to the extraction side.
    - **The form is a measurement, not an impression**: one named defect per
      item, with the upstream step that produced it (or "unknown", which is the
      most important field), a count with its denominator and the population's
      name, native 1:1 crops under `results/` (規律6), and **what would change
      downstream if it were fixed** -- the field that makes the request
      refusable.
    - **Selection and defect must be separated** (T86, 規律9). The raster's
      63.0% was a selection rule, not a defect rate, and the same confusion is
      available here: unnaturally cut panels may be the koma cutting rule
      rather than breakage. H's observation travels with its provenance (Track
      F `panel_pack_20260919` thinning, from the pair-dataset extraction
      program) and with H's own words that the effect on its numbers was small.
    - **The conclusion may legitimately be "put the selection rule in
      writing".** That is provisional rules 11-13 naming their own killer, and
      the precedent is on record: getting the raster layer-drop rule in writing
      changed what 63.0% meant. A one-line outcome is a sufficient outcome.
    - **Register before the clean line art arrives**: the known cell it must
      reproduce (one per generation), and the null value for "cleaner" -- after
      the swap, instrument difference and material difference are no longer
      separable. H's old-to-new stroke correspondence is H's instrument, built
      once the material is in sight.
    - **First evidence in, 2026-10-25 the same day, and the foundation's band
      check on it.** Track H put the author back through all **482 panels** of
      order 0-502 in a report-only UI: **12 panels reported (2.49%)**, native
      1:1 copies of all twelve, listed by `slug` + `content_fingerprint`
      (`/home/sh1/deepl/lineart-face-words/results/panel_anomaly_20261025/`).
      The author's reading: **layers from another page look composited in --
      a different kind from a layer mask failing to remove what it should.**
      Band check: the arithmetic adds up both ways; six slugs and ten
      fingerprints, with succor's seven spread over **five distinct pages**, so
      **not one corrupt export**. **The concentration is real**: 7/26 against
      5/456 elsewhere is **24.6x**, one-sided Fisher **p = 3.8e-7** (expected
      0.65 panels in succor at the overall rate). **The order trend is not**:
      4,3,3,2,0 looks like later marking being cleaner, but twelve panels
      scattered uniformly put seven or more in the first 197 with probability
      **0.17**, so it is indistinguishable from flat and no one may read
      improvement off it. **What the 2.49% cannot carry**: the 482 are panels
      that a face detector found a face in *and* the author reached within
      order 0-502 -- two selections (規律9) -- and **the selection is biased in
      opposite directions by anomaly type**: a panel with another page's layers
      composited in gains figures and so gains detected faces, while a panel
      emptied by the same class loses them (`copy` p29, a nearly empty panel,
      got in anyway, so the under-represented side is not zero). It is a rate
      per face-bearing panel and must not be multiplied over the corpus.
      **This one is on the defect side, not the selection side**: a cutting
      rule does not composite the wrong page, so it is a different kind from
      the raster's 63.0%. H also warns the twelve may not be one phenomenon
      (overlapping line art, a black square mass, blurred tone surfaces, a
      nearly empty panel) and declines to judge which -- the author's and
      L/M's call (規律6). **The foundation asked H for one more number: the
      panel counts of the 53 works with zero reports**, because the present
      table holds only the denominators of the six works that reported, so
      *work* and *production year* cannot be separated -- 2019's `fringe` is
      1/5 as well, and a generation is a pool. **And the five marked faces
      among the twelve stay in the training population for now**: removing them
      would move the population with no registered question attached, so they
      are recorded by `order` as known dirt and the swap is where that choice
      gets registered.
    - **The zero rows came back the same evening and they settle it: the
      concentration is the WORK, not the generation.** With all 59
      denominators (`per_work_all.csv`), **2019 without succor is 1/48 against
      a background of 5/456, p = 0.43** -- the year table's 8/74 was succor
      carrying it. **succor survives the multiple comparison**: Fisher 3.8e-7,
      **Bonferroni over 59 works 2.2e-5**, while the next-smallest raw p is
      housei's 0.025 on a denominator of one panel and corrects to 1. The
      95% intervals do not overlap (succor 11.6-47.8%, the rest 0.36-2.54%).
      **So the first question to the extraction side narrows from "the line art
      is dirty" to "what is different about 030_2019_succor, or about its pass
      through the program"** -- and because the author's reading is about
      compositing, the structure DB can be asked before anything is
      rasterised: the five pages against reported-free pages of the same work,
      joined on `page_path` and `content_fingerprint`, never on title.
      **One relation the foundation found and is NOT reporting as a result**:
      the five non-succor reports all sit in works with few face-bearing
      panels (1, 4, 5, 5, 9); threshold-free, the mean work size of a reported
      panel is **4.80 against 13.71 expected, permutation p = 0.0018**. It is
      held as a hypothesis for two reasons -- **the statistic was built after
      looking**, and **the selection bias already on record predicts it** (a
      broken panel yields no detected face, so a badly handled work has both
      few panels in this population and a higher reported rate, one cause
      producing both). succor is not explained by it (26 panels, on the large
      side). **Two riders on the year column**: it is the second field of
      `work_id`, not a verified production year (five works have a non-year
      there -- 42 panels, zero reports, so adding them anywhere only lowers
      that year's rate), and H's year table silently pools the `b`/`c`
      suffixes, which may be separate batches of one year (a generation is a
      pool).
      **The copying hypothesis is closed on a bound, not on a comparison
      (2026-10-25 night).** The extraction side compared layers from the .clip
      files directly -- the sha1 of each layer's compressed pixel block, nothing
      expanded or rendered -- over all 69 pages of succor, keyed on `page_path`,
      with template blocks removed two ways (repeated inside one page, or
      present on more than half the folder's pages). **No picture layer is
      byte-identical across pages anywhere in the work.** The foundation's own
      sweep of `succor_sharing_hashes.csv` found exactly **three** non-template
      blocks shared across pages, all three the empty ones they had already
      identified (81,616 bytes on 61 pages, the same size on 24, 160,240 on
      three). **The five reported pages carry 407 picture layers, so 0/407 is a
      one-sided 95% upper bound of 0.73%** (0.078% over all 3,848 in the work),
      and the reported rate is **27% in succor, 2.49% overall** -- the mechanism
      is three orders short, which is a stronger statement than any comparison
      of counts. **And the comparison the foundation had registered does not
      work**: the bar said "if the five reported pages are not higher than
      pages with no report, the hypothesis falls", and on the pair mean the arm
      **is** higher (0.30 against 0.15) -- but a permutation over all 5-page
      subsets of the 69 (200,000 draws, seed printed, computed in the same run)
      puts it at **p = 0.335** (raw counts p = 0.622), and the per-page maximum
      the extraction side reported reverses the ordering (0.6 against 0.63).
      **A bar on a point comparison with ten pairs has no resolution** (T98,
      the foundation's error). The scope stands as registered: copying followed
      by any redrawing, any shift, or mixing at the composite stage is invisible
      to this instrument, so the finding is **"as-is copying cannot explain
      it"**, and the composite stage remains the live candidate. **The deeper
      pixel comparison is declined for now** -- H is paused, the author reports
      little downstream harm, and the dirtiness of extraction is the front.
14. **Two authorisations given 2026-10-25 night, and the records that back them.**
    - **Request 4 (the silhouette alphas) is ordered.** Three approval records
      exist and they are **not the same kind**: `5d31507` (10-24, work_log line
      551) is **Track M's own two three-way choices and the user picking the
      recommended option -- not the author's words**; `46fd70f` (10-25) is the
      author's **verbatim "ok"** to the question *may request 4 be placed*; and
      the user told the foundation the same night **"出してよい (inside the
      10-24 approval)"**. **The foundation declining to act on the first was
      right**: a choice made from options someone else wrote is that person's
      sentence, not the author's. M placed the order at 20:15 (letter plus two
      lists, SHA256 matched both sides). **The numbers moved and both
      directions are on one page**: layers **10,622 -> 8,076** (1,222 duplicate
      folder-descendant rows removed, population 872 -> 786), pages **872 ->
      786** (`superseded` 74 and `duplicate_content` 12 dropped), size **~1.7GB
      -> ~2.0GB** because the previews were counted separately. The known cell
      is the **first 8 pages / 303 layers byte-matched against the pilot, and
      the rest is not read if they differ** -- the only place to stop before
      reading 2GB. **The known cell passed the same night: 303/303 alphas
      and 8/8 previews byte-identical, nothing named-but-missing**, so the main
      run (779 pages, 7,773 layers, ~1.2GB + ~0.7GB) is inside the 10-24
      authorisation and waits only on M's own reconciliation. **A met condition
      is not a new approval**, and the foundation said so in the same letter.
      One identity conflict came with it and is ruled the same way as T94:
      `2018\skima10\skima10\page0064.clip` has 39 layers in the layer list and
      no row in the page list because `version_roles` marks it `superseded` --
      **the page list is the population, the superseded file's 39 layers stay as
      known-cell material only**, and both records are correct. **[Withdrawn
      2026-10-26: the author answered that `2018\skima10` and
      `2018\skima10\skima10` are separate branches with no newer one, so
      `page0064.clip` is a current page of its own and those 39 layers are
      population material. No new order was needed -- they were already in
      hand.]** **The anchor the foundation asked for came back within the hour and it
      failed, and the failure found four things worth more (2026-10-25 night).**
      The extraction side could not tell which 8 clusters were meant, so it
      counted **all 65 clusters that hold both a `final` and a `superseded`**,
      split by `decided_by`. **[Marked 2026-10-26: 65 counts clusters, not
      duplicates.** The clusters are the Jaccard >= 0.5 of per-layer *statistical*
      signatures (T106), and the author has now called one of them -- G,
      `2019b\hitoduma\CD.clip` against `CDrev2.clip` -- **"completely different
      things"**, the first measured false positive of the clustering. The anchor
      test's verdict does not change, since it failed anyway, but every figure
      that reads 65 as a number of duplicates is reading an upper bound.] Name order puts `final` later in
      **0 of 42** `identical_metadata` clusters, 11/16 `+year`, 3/6 `superset`,
      0/1 `mtime_needed`; the foundation reproduced every count from the 65-row
      file. **So no anchor**: the only rows independent of the name are the 6
      decided by content (`superset`, 3/6 -- a coin) and the 11 lying wholly
      outside the bulk copy (name 3/11, mtime 0/11), and **the 10 undecided
      clusters stay `undecided`**, which the foundation had committed to in
      advance. **But 0/42 is not an absence of signal -- a coin gives 2.3e-13.**
      Those 42 were decided *by the name*, so the known cell the foundation
      specified was **not independent of the rule under test** (T100), and what
      the agreement measured was identity. The shape of the deciding rule is
      readable from the 42: the `superseded` side carries a **`b` suffix on the
      year folder** in 23 (`2019`->`2019b` 18, `2022`->`2022b` 5), **`_bk`** in
      4, is **nested one level deeper** in 2 -- and in **11 the `final` is the
      deeper side** (`comics4\underworld\underworld` 9, `gal\gal` 2), so the
      nesting rule runs both ways and no single rule can be written. **`decided_by`
      must name the tie-break actually used, not the evidence of equality.**
      **Four riders, none of them a number the foundation will move itself.**
      (i) **Three resolutions run against what the names mean**: `skima12chara`
      is `final` over `skima12chara`**`Done`**, `done4\ako6-14` over
      `done5\ako6-14` (and the extraction side's own in-file counter is *higher*
      in `done5`), and `superset` made `ako4_`**`backup`**`.clip` the `final`
      over `ako4.clip`. A naming habit is the author's to state (規律6), so
      these are held with both sides printed and the one-line author question
      is queued -- **no flip on the foundation's guess**. (ii) **Five clusters
      pair pages whose file names differ, three of them across different
      works**: `2019\road\page0076` with `2018\skima11\skima12`,
      `aljanne2\page0029` with `phantom\PhantomThief\page0026`, and
      `ako7\page0073` with `ako6\done4\ako6-18`. **A duplicate cluster whose
      members sit in different works is not a version chain** -- identical
      content, two real pages, which is T94 and T95 arriving together -- so
      **both stay in the population and the duplication goes in its own
      column**. **[Withdrawn for two of the three, 2026-10-26.** The reason
      above -- *two real pages, each of its own work* -- is an inference from the
      folder's parent, and the author refuted it twice: `ako6\done4\ako6-18` is
      **a pre-merge scattered page of `ako7`**, and `aljanne2\page0029` is **a
      reference copy of a `PhantomThief` page** (`aljanne2` is its sequel), so
      each keeps **one representative**, not both, and for the second the
      representative is the original rather than the folder's side -- otherwise a
      PhantomThief page is counted as an aljanne2 page. **`road\page0076` with
      `skima11\skima12` the author does not know**, so it stays `undecided`.
      **Two separate mechanisms make a folder not a work** -- the pre-merge stash
      and the reference copy into a sequel -- so a per-work page count taken from
      folders is wrong in two directions.] The population moves up to 3 pages (786 -> 789) and
      `skima11`, `phantom` and `ako6` were each silently missing one of their
      own pages: **not a whole work, so it would have stayed missing**.
      (iii) The folder pair `2016\gal` is resolved **both ways** across three
      clusters (31, 32 make `gal\gal` the `final`, 33 makes `gal` the `final`),
      so "which folder holds the newer version" is not a property of the
      folder; the content-based one is the one to trust and the other two are
      to be re-examined. (iv) **A candidate anchor that uses neither the name
      nor the filesystem**: the `Project` table's counter inside the .clip
      (`done4\ako6-19` = 150, `done5\ako6-19` = 154) survives copying. The
      request went out **with its bar written first**: adopted only on **6/6**
      of the content-decided clusters (a coin gives 1/64), not adopted at 5/6
      (7/64), the null being name order's own 3/6 on the same six, with the
      count of missing and tied values printed before the result, and the
      statement *there is no save-order field inside the file* recorded as the
      outcome if it fails. **Both came back inside the hour, and the author
      answered too (2026-10-25 night).** The author, asked which of
      `2019`/`2019b` and `2022`/`2022b` is newer: *"b付きが新しい　どちらも"*;
      and on the two `superset` clusters: *"2022b\sketch のほうが新しい
      作業もすすン出る頁がおおくある"*. **One rule covers both answers -- the
      `b` suffix is newer -- and it reverses 25 of M's clusters**, the 18
      `2019`/`2019b`, the 5 `2022`/`2022b` and clusters 83/85. **Three
      consequences, in rising order of weight.** (1) The 23
      `identical_metadata` flips wait on a **byte comparison**, because the
      author's rule is *bytes identical -> take the newer year folder* and
      `identical_metadata` is metadata, not bytes: byte-identical means
      relabelling with a mapping file (the delivery's `source` would otherwise
      name the non-current path), byte-different means re-rendering at most 23
      pages. Clusters 83/85 need no byte check -- the author spoke about
      content -- and flip now. **The main run was not stopped**: 779 of the 786
      pages are untouched by this. (2) **The `ProjectName` anchor is FAIL as
      registered**, 4/6 with the other two tied, and the degeneracy print asked
      for in advance says *why* in a stronger form than the bar does: ties in
      **61 of 65** clusters, all 58 metadata-identical clusters tied, while
      **10 of the 11 undecided clusters have distinct values** -- a quantity
      that degenerates exactly where an independent decision exists and
      separates only where none does **cannot be validated where it would be
      used**. `final` never carries the smaller number in any of the 65, so the
      direction is kept as a registered hypothesis and no author time is spent
      on it; the 11 stay `undecided`, which blocks nothing. (3) **The `superset`
      rule is refuted in direction by the author, 2 for 2** -- the only two of
      its six clusters the author could speak to. *More layers* is not *later*
      in this manuscript, and the author's own words point at a different
      mechanism (*more pages are further along* in the `b` folder, a statement
      about the set of pages, not about one page's layer count). **Two of the
      foundation's rulings go with it.** Its advice that `2016\gal`'s
      content-based resolution was the one to trust is **withdrawn** -- all
      three gal clusters are flagged and held, and `gal` carries no `b` to
      decide them -- and **the `ProjectName` known cell is void**, because it
      was built from the `superset` decisions. **So four of the foundation's
      known cells failed in one evening, each at a different level** (see
      discipline 2's naming sibling). **What did gain a number**: the two
      mtimes lying outside the bulk-copy window agree with the author 2/2
      (2022-10-19 over 2022-03-07, 2022-07-16 over 2022-05-01), the first
      evidence that a timestamp outside that window may be a drawing time --
      T99 is about timestamps inside it. **And the next author questions are written and
      queued, not asked. The foundation's first count of them was wrong and is
      corrected here**: the one-line question it proposed covered **two
      different folder shapes**, not one, and 14 clusters rather than 15.
      **(A) a grouping folder**: `2018\comics4\underworld\underworld` against
      `2018\underworld\underworld`, **9 clusters** (pages 0001-0007, 0009,
      0010), both sides inside 2018 so the year decides nothing -- and
      `comics4` holds 5 further clusters (`china_v2` 3, `mars_v2` 2) which are
      cross-year and already decided, so the answer would also tell us what
      that folder is. **(B) the same name repeated one level down**:
      `2018\skima10` against `2018\skima10\skima10` (2 clusters, `final` the
      shallow side) and `2016\gal` against `2016\gal\gal` (**3 clusters
      resolved both ways in our own ledger** -- 31 and 32 make the deep side
      `final`, 33 the shallow one), **5 clusters**, and this is the pair where
      the ledger contradicts itself. Three singles sit outside both:
      `skima11\skima12chara` against `skima12chara`**`Done`**,
      `ako6\done4\ako6-14` against `done5\ako6-14`, and
      `ako4\color\ako4_`**`backup`** against `ako4.clip`. **The order of the
      author's time is the user's to set.**
    - **The per-layer ground truth is approved** (the user: *already gave the
      OK*), repurposing L19's UI so one truth calibrates `preview_dark`,
      `keep4` and `bad_max`. **The foundation's condition is registration
      before the author is asked**: strata on each of the three constants'
      margins (kept side and dropped side), **plus one random stratum**, with
      the counts fixed in advance, and a few of L19's unambiguous regions mixed
      in to measure the human answer's own reproducibility (6 of 37 answers
      changed on re-ask). **Asking only near one constant's boundary would let
      that constant select the sample that corrects it** -- a warning about
      *calibration* only, and the foundation narrowed it the same night:
      **a boundary-drawn sample is the right sample for refuting a boundary**,
      because the claim is about where the boundary sits.
      **The truth came back in five minutes of the author's time (92 layers,
      median 2.1s) and it convicts `bad_max`**: in the 0.95-1.00 band, 2
      layers pass and **19 are dropped, and all 19 are finished line art**,
      with the mechanism falling out of the definition -- **and the
      definition turned out not to be the one in the name (2026-10-25 night,
      from the extraction side's own code)**. `cov` is the layer's **alpha
      plane only** (`clip_render.py` line 214), never the grey plane and never
      the colour, and the statistics run on `g = 255 - alpha`, so
      `black_among_drawn` is **the share of drawn pixels that are at least 224
      alpha -- near-opacity, not blackness** (and `ink_px` is alpha >= 128,
      `drawn_ratio` alpha >= 26). **The numbers do not move; the mechanism
      sentence does**: `<= 0.98` refuses **layers whose edges are not soft**,
      so line art drawn without antialiasing sits at 1.0 and is refused while
      meaning to refuse fills, which is what the author's 19/19 says one level
      down. **A white opaque balloon fill counts as "ink" on this definition**
      -- the same property Track L had recorded from the other side (*coverage
      is not colour*), now arrived twice independently. **And discipline 9-2 is
      at the base of the whole stack**: `g<128`, `g<230` and `g<32` are
      unchanged since the first commit (`96db0bc`, 2026-06-06) of
      `features.py`, where they were written to tell line art from rough on
      **composited page JPEGs, in which `g` really is brightness**, and they
      are now applied to **layer alpha** -- role and unit both changed, with
      **no calibration record** ("記録なし", in their words). Every LINE
      constant, `bad_max` included, sits on top of that reinterpretation.
      **The foundation's own documents stop calling `ink_px` ink** (it is
      `cov128_px`, a half-opacity coverage); the extraction side is explicitly
      *not* asked to rename its columns, because renaming would make every
      existing ledger unreadable. **Track L then did the thing that keeps an
      explanation from becoming a T103: it turned the received explanation into
      a check instead of a fact (`KC34`, `bd3bb1b`).** L's own layer PNGs are
      the extraction side's `g` itself, so the same thresholds were applied to
      them: **30 layers drawn at random, 16 with a PNG on disk, all four
      quantities agreeing within the rounding width, zero discrepancies, both
      directions, layer granularity** (the 14 without a PNG counted and
      printed). **Two layers show the quantity's identity outright**: a 3D layer
      (`ako6_page0117 L110`) spans `g` 144-255 over 112 distinct values and
      scores `bad = 0.0` -- **no pixel of it is more than 44% opaque** -- while
      `ako7_page0069 L23` holds **two** distinct values, 0 and 255, and scores
      **1.0**. So `black_among_drawn` is **the amount of edge softness**: a
      layer with no antialiased pixel at all sits at 1.0. That also settles
      KC33's "grey" layers -- they are **semi-transparent**, grey 145 being
      alpha ~110. **Nothing measured moves** (the author's 19/19, AUC 0.7750
      and its sign, 43/54 in the unselected frame, the constant's lack of
      calibration), and **the author's 64 answers are uncontaminated by the
      name**, because L's UI never displayed the layer name, `preview_dark`,
      `keep4` or `black_among_drawn` -- registered in advance, not argued
      afterwards. **The author-facing vocabulary changes with it**: not *black
      or not black* but **soft-edged or hard-edged**, which maps onto the
      author's own practice of drawing with antialiasing off and is therefore
      checkable. L also corrected the dirtiness case's L23 item the same night:
      the paper layer's PNG is **fully opaque, not black**, so *z-ordering every
      layer does not reproduce the manuscript* stands, but its reason moves from
      a claim about the data to a claim about our own compositing, which was
      reading alpha as darkness. So the question to the extraction side rises from *what value*
      to **can this quantity separate line from fill at all** -- 9-2's deeper
      form, a borrowed *quantity* rather than a borrowed constant, and the
      registered next number is the overlap (or AUC, null 0.5) of
      `black_among_drawn` between layers answered "line" and "fill", which
      scores the quantity without choosing a threshold. **It was measured
      within the hour and it refuted the foundation's expected answer: the
      quantity is not beaten, the sign is.** AUC(line > fill) is **0.7750
      [0.6379, 0.8962]**, so line layers carry the higher pure-black share --
      and the rule keeps the LOW side (`<= 0.98`), which keeps the fill-ish
      and drops the line-ish, exactly what L25's 19/19 said one level down.
      **One ceiling goes with it**: 0.775 is not 0.97, so even oriented
      correctly a single threshold on this quantity misclassifies a lot, and
      the extraction side's own reason for the condition was a **tail cut**
      (*a layer at exactly 1.00 is a binary fill or balloon*) **on a tail that
      is mixed**: 0.88 lost line art, which they found and fixed, and 1.00
      admits the 519 balloon/paper/fill layers. **No point on this one
      quantity satisfies both**, so the final form of tonight's ruling is
      neither the value nor the sign but **one quantity carrying two jobs**.
      `preview_dark` fails on its own terms (**0.4417 [0.2288, 0.6591]**,
      containing the null), and `keep4`'s 0.9970 is **a tautology to be
      labelled as one in the table** -- it passes fill shapes by construction,
      so its definition is making the label. The concentration check holds
      (0.7750 -> 0.7500 -> 0.8435 as the defining strata come out). **And the
      random stratum cannot deliver the unconcentrated AUC at the size being
      discussed**: fills are about 1 in 16, so 60 random layers hold ~3.8 of
      them and give a +/-0.19 interval, wider than the +/-0.13 already in hand
      -- 60 buys the error upper bound (~4.9% at zero errors) and nothing
      else, and an unconcentrated AUC needs ~240 random layers (~15 fills,
      10-15 minutes of the author). The foundation recommends the 60 and says
      why: **the sign claim is already carried** by 19/19 plus the interval
      plus the concentration check, so 240 answers a different question, and
      that question only matters after the orientation is fixed.
      **The heavier finding is not `bad_max`**: 7 of the 18 layers `keep4`
      passes at its boundary are **not pictures at all**, and `keep4` claims
      nothing about role, so **the instrument has a test of shape and no test
      of role** -- the hole that 5 rough-sketch, 4 sound-effect and 1 effect
      layer fall into. The next design writes a role quantity rather than
      fixing a threshold. **`preview_dark` is neither cleared nor retracted**:
      875 layers dropped stands as a count, what they were is unmeasured, and
      0 of the 4 dropped in the random stratum were finished line art.
      **A 60-layer random stratum yields an upper bound, not a rate**: zero
      errors in 60 bounds it at 4.9% one-sided, while a true 2% rate shows
      zero 30% of the time, so 1% and 4% are not separable and the frame the
      sample is drawn from has to be registered before drawing it.
      **And the author repaired the instrument's axes, not its values**: the
      five choices mixed shape (`keep4`'s axis) with role (`bad_max`'s), which
      the author caught before the run -- the same separation this whole day
      turned on, and it saved all 92 answers. L asks what a layer
      is, M's known-cell 8 pages ask where a region is, so they do not overlap;
      the order between them is the user's.
      **Track L fixed the wording of all of this the same night (`e8c3887`) and
      the fix matters outside L**: "dropped" and "kept" were doing four jobs, so
      they are now **refused / accepted / excluded / routed**. `keep4` is a
      **routing and discards nothing**, and the sayable form is *an unselected
      frame in which `black_among_drawn <= 0.98` refused 54 layers, 43 of them
      finished line art* -- **not** *40% of the line art was lost*. The layers
      in question were never in the delivery, and **mixing the two turns the
      request to the extraction side from "pick these up" into "stop what you
      are doing"**. Three of the foundation's four cautions on the koma ledger
      also came back corrected: the `reason=empty` one **does not reach L**
      (L's gate never reads `reason`; its first condition is `drawn_ratio <= 0`,
      and all 4,399 such layers have `ink_px == 0` -- the gap between L's
      ink-zero 2,581 and the ledger's `empty` 2,598 **is exactly those 17
      layers**, two of which are the author-confirmed cross-page duplicate pair.
      **The foundation called that "two instruments agreeing layer for layer"
      and L turned the phrase into a running known cell (`kc33`, `8dd0bac`),
      which refuted it: the agreement holds in one direction only.**
      `drawn_ratio <= 0` implies `ink_px == 0` with **0 exceptions in 4,399**,
      but the reverse has **31** -- layers with `ink_px == 0` and `drawn_ratio`
      up to **0.194**, every one of them `black_among_drawn = 0.0` with
      `drawn_gray_mean` 144-145, by `reason` `render_3d` 11 and
      `not_line_stat` 20. **A containment is not an agreement** (T101), and the
      asymmetry is the two quantities' definitions: `ink_px` counts ink and
      `drawn_ratio` counts opaque pixels, so **a layer can cover a fifth of the
      page and carry no ink**. None of the 31 pass L's gate, so no number of
      L's moves -- but **it lands on the foundation's own planned recount**:
      weighting A/B/C by `ink_px` silently gives **3D and tone layers weight
      zero**, which is T-the-same-vocabulary again with the sign reversed (the
      `empty` caution was *no drawn pixels yet some ink*; this is *drawn pixels
      yet no ink*). **So `ink_px`'s own definition is now asked for** -- which
      pixels it calls ink, at what threshold, and for which role that threshold
      was drawn -- before any A/B/C is restated in it); the JPEG one is moot
      because **L holds no number counted from the ledger raster** (A6 is PNG on
      both sides); and **the foundation's own statement of L's gate was short by
      two conditions** -- it is `folder=False & rendered=True`, `drawn_ratio > 0`,
      `preview_dark >= 0.5`, coverage < 95%, `/text` and `/sketch set` excluded,
      and only all five give 1,981. The double-count caution does apply and L
      now prints population and source beside the figure; **A6's conclusion
      stands on counts** (1,709 layers at >= 0.9, 249 below 0.3). The `hash()`
      sweep came back **zero live uses** -- 12 `hashlib.sha256` wrappers and 11
      T97 comments, value-position none -- **and L swept the `solid-fill` tree
      too**, because `l18_known_cells.py` imports from it through `sys.path`, so
      L's numbers would inherit a defect living there.
    - **M's primary population is 777 / 779 and the foundation's 856/860 is
      withdrawn**: that arithmetic ran on the set before duplicate resolution,
      and a population is named after the resolution, not before. The standard
      can be built on **507 / 509**.
    - **P13's FAIL is characterised and it renames the standard.** The 306
      chain-less pages against the 566 with a chain differ **2 against 22
      colour-format layers, over tenfold** (style 1.4x, year 0.12-0.84), so the
      standard built on the 566 is **the standard of pages that have a clip
      chain, and not the standard of colour manuscripts**. **42 works are
      chain-less throughout** (115 of the 306 pages), so the list of works that
      have no standard goes beside it: a selection that removes whole works is
      felt later. **The foundation's gloss "pages with more layers and more
      colour layers" is withdrawn the same night.** It holds on the medians and
      dies on the exceptions: `nuru` carries **135 colour layers with no clip
      chain at all** (`NY` 81, `mars` 35), so **thickness is not the
      mechanism**. Track M went to look at the exceptions after the foundation
      had asked only for the medians, which is what a band check is for.
      **The name stays the field the selection actually used.** M proposed *the
      standard of pages painted by clipping to the silhouette* -- a better
      reading, and not a usable name: it interprets `LayerClip`, whose mapping
      the structure DB itself says is unverified, and a way of painting is the
      author's to state (規律6). **M's reading travels as a registered
      hypothesis in the same paragraph**: the author holds at least two ways of
      painting and this standard touches one of them. **And if the first branch
      of M's question is the answer -- that in those works the painted layer
      itself is the region -- then excluding S2 (追補6 §F5) is what puts 42
      works beyond the standard's reach**: material for the author to revisit
      their own call, never grounds to overturn it. **Track M then measured the computable axis before spending the author's time** (the user's standing policy) and registered P26 before scoring it: flat colour layers as a share of colour layers is **0.115 in the 42 chain-less works against a bar of 0.25 -- FAIL -- with the chain-having population at 0.647** as the control and the known cell passing (42/114 = 0.368 on the 8 registered pages). **But the band check narrows what that FAIL says**: the S2 cut adds `top_colour_share >= 0.99`, so what is absent in those works is a **flat** painted layer, not a painted layer -- and the flatness constant came from the chain-having population, which is 規律9-2 showing up inside this project's own instrument. The 460 non-flat layers sit at `colours_q16` median 241 and `top_colour_share` median 0.148 with `has_gradation_fill` true on only 2.4%, so they are **many-coloured paint, not a gradient feature** -- exactly the candidates for a region that is not flat. **So branch (1) is not thinned; its S2 form is**, and the material handed to the author must say so, or it reads as *your file contains no regions*, which was not measured. **The blind spot also splits in two and so does the question**: `ako7` (2018) places flat layers (0.579) without using the chain, so the standard's door may be open there, while `nuru`/`NY`/`mars`/`poolsideIdle` have no flat layers at all (0.000-0.021). **M deliberately wrote no mechanism-prediction for P26** on the grounds that the next table here is the author's answer and not a measurement (規律6) -- correct, and the departure from lesson 15's form is recorded as intended.
