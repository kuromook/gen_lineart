# The fill line's record, 2026-10-23/24

Moved out of `doc/CURRENT.md` on 2026-10-24, verbatim. This is Next Actions
item 11 as it stood when Track L was still a proposal: 1,045 lines of one day's
correspondence with the author and the extraction side -- the manuscript layer
inventory, the trial-work pick and its retraction, the five production classes,
the production fill deliveries, the colour and tone surveys, Track J's door, and
the retractions each of those produced.

**It is a record, not a state.** Several of its claims were superseded within
hours of being written, and some of them by notices that arrived the night this
file was split out:

- The gradient **mask** is called untestable here ("the inventory CSV has no mask
  columns at all"). It was measured on 2026-10-24 --
  `inbox/note_gradient_mask_survey_20261024.md`, 1,754 layers over 85 works, and
  that notice also **invalidates its own 2026-10-23 first run**.
- The composited-page route is open here. It is closed: `CanvasPreview` is
  exactly a quarter of native and no native composite exists in the file
  (`inbox/reply_composited_page_20261024.md`).
- Figures inside it that were retracted are kept **beside their retraction**,
  which is the reason this file exists rather than being deleted: the enclosure
  reframing, `fringe3` as a trial work, the 519 "fill" layers, and L7.

What currently holds is in `doc/CURRENT.md` -- its opening paragraphs, its Track
Ledger entry for Track L, and Next Actions item 11, which is now the short form
of this file.

11. **A line / fill / texture decomposition on ground truth, at region unit --
    a new track, proposed by the user 2026-10-23.** This file already specified
    it on 2026-10-10 and it was never executed: *"the ground-truth side needs
    no rough at all"*, and *"a line/fill/texture decomposition is not only
    Track J's business: part of the words line's unit problem is a fill
    problem."* **Proposed as Track L on 2026-10-23 by user decision**, in a clean
    tree rather than as Track J's next stage -- the proposal is
    `doc/track_l_fill_region_proposal_20261023.md` and is **not yet approved**.
    The user's reason, recorded because it is not derivable: Track J went toward
    muddle, recognised it and stopped, and **only a few specific ideas have been
    refuted**; J's tree carries its prohibitions and a nearly closed registration
    alongside its instruments, so L **borrows the instruments read-only and not
    the framing**.
    - **That it repairs line measurement is already demonstrated twice, not
      hoped for.** (i) `line_width_p50 7.59` read as thick strokes was solid
      fill, and the fill-excluded width is the instrument that does separate
      the pools (housei 1.91 against ako5 2.74) and is invariant to the window.
      (ii) `cluster_set` mixes parts overwhelmingly through hair and shading
      hatch: **three quarters of labelled brows share a cluster with the bangs
      above them, and 14.6% of all strokes enter no cluster at all.**
    - **Why it is not "Track J again", and this is the part that reframes it.**
      Track J's measurement `tools/evaluation/rough_fill_signal.py` (ako5ver2
      167 panels, housei 73, koma-aligned pairs) found that **96% of solid-fill
      area has no dark mark under it in the rough**, 99.7% on the strictest
      reading, so **any "the rough already has it" arm has a recall ceiling of
      4.2%** -- with the author's own account of why: fills are decided before
      the rough is drawn, so the artist does not mark them, and a fill-like
      mark means indecision. But in all four pairs of
      `results/region_closure_20261010/pair_check.png` the hair and clothing
      that are solid black in the line art are drawn in the rough as **an
      outline with white inside**. **The reading built on that, "the information
      is absent as ink and present as enclosure, so the question is which
      enclosed region is filled", is RETRACTED the same day it was written.**
      Track J measured exactly that on 2026-10-10, two ways, and both fail:
      downscaling gives **enclosed 0.0% at every setting** (scales 1 to 1/16,
      two thresholds, barrier from the rough and from the de-filled GT -- a
      2.7px stroke breaks into dashes as the image shrinks, so "the eye bridges
      the gap; this operation widens it"), and morphological closing over 59
      panels and 1,602 regions >=500px reaches **9.3% at r=16 and 15.4% at
      r=32** with tightness collapsing to 0.085 and 0.033 -- the enclosing
      region is about **30x the fill**, bought by swallowing the neighbourhood.
      The same day's D1 adds that a fill's outline is **2.24x** a drawn stroke's
      pieces per 100px against a 2.00x bar, so it is not a drawn curve either.
      **The failure here was the foundation's**: it took the montage sentence
      from that entry's section 2 and built the reframing without reading
      section 3, two screens down, which tests it -- the qualifier that this is
      a *human* reading came loose exactly as "for SDXL bare" came loose from
      `cs2.5`. **What survives**: J refuted *cheap morphology*, not the idea of
      a region proposer, and wrote so ("the requirement the user states is
      right; cheap morphology does not meet it", and "no region proposer is
      built" is a scoping decision). What died is the foundation's claim that
      the reframing made the task **tractable and free of GPU and author time**.
    - **Unit constraint, already measured**: 0.5% of `ako5` tiles sit entirely
      inside a fill and 31.3% are near-blank, so **a region formulation does
      not fit the 480px tile**; Track J has already moved its own input unit to
      the koma panel, and this starts there too.
    - **The lead that now heads the proposal came from the author, 2026-10-23,
      and it is production knowledge this file could not derive**: manga
      manuscript data carries a **tone (トーン) layer**, which is a surface
      representation outright, handled separately from the **ベタ layer, which
      also exists as its own layer**. So the fill ground truth does not have to
      be inferred or asked for -- **it can be dug out of the manuscripts from
      the period when the work was done in separated layers**. And the
      extraction tool **already identifies the ベタ layers in order to discard
      them**: its 2026-08-21 reply records that layers with
      `black_among_drawn == 1.00` are "2値のベタ・フキダシ" and that admitting
      them would add **519 layers across 74% of normal pages** (of 2,048 pages
      surveyed). **That reading of 519 is superseded by the 2026-10-23 inventory
      and was wrong twice over** -- it was a by-product of a line-extraction
      threshold test rather than a fill inventory, it holds **no tone at all**,
      and it mixes ベタ with balloons *and* with page furniture (paper, MASK,
      トンボ, bleed and panel frames, 集中線 -- all 100% pure black among drawn
      pixels and therefore indistinguishable from fill by pixel statistics,
      5,480 layers of them on the same 2,048 pages). **A class defined by a pixel
      statistic collected everything that shares the statistic.** The tool also already emits a
      third raster on the **same folder, naming and canvas** as line/sketch
      (`{prefix}_{page}_koma.jpg`, 1,645 pages), so **existing alignment
      carries over** -- the heaviest technical question is answered before it is
      asked. Two reservations: the tool's test lumps ベタ with **フキダシ**
      (speech balloons, which are lettering furniture and not drawn fill), and
      the author's "separated period" is to be **measured from the inventory,
      not assumed**.
    - **(A) the inventory came back the same day, 2026-10-23, and the premise it
      corrects is the foundation's.** `config/fill_tone_inventory_layers.csv` on
      the extraction side: **45,533 layer rows**, one per layer, with year, work,
      page, layer id, the layer name itself, parent folder, class and pixel
      statistics, over 289 named works -- **270 once the year-folder duplicates
      are folded** -- and 2,048 pages. Layer counts: **tone 8,072,
      gradient 1,999, ベタ 1,246, pure-black-but-unnamed 682, line-named-but-ベタ
      192, balloon 656, 描き文字 843, ノンブル 1,334, furniture 5,480, unknown
      14,069.**
      **Every count on that line is a count of names, not of content, and two of
      them have since moved a long way.** By 2026-10-23 evening the extraction
      side had checked the ベタ column against the pixels: **ベタ layers 1,246 ->
      299 and works with ベタ 84 -> 55**, because **across this corpus as a whole**
      `black` is predominantly a *line*-layer name -- only **70 of 1,012**
      `black` layers met the fill signature, i.e. 942 did not. **Corrected
      2026-10-23 night by the extraction side: that is a corpus-wide majority and
      not a per-work fact**, and its inventory never decided by name -- a layer
      called exactly `black` is classed `beta` or `line` **by its pixel
      signature**. So the author's generation-dependence is followed
      automatically, and `ako6`/`ako7` having all 74 of their `black` layers come
      out as fill **does not contradict the majority**. **The `区分` column is
      that decision folded into one value; read per work, go to the per-layer
      signature columns.** Same shape as colour not being a property of a work. The tone column is separately known to be **2-3x inflated** by
      unused templates (next bullet but two). So the line stands as an index of
      what names exist and **must not be quoted as an inventory of area** --
      see the rule at the end of the `NC4` trap.
      - **トーン was not being discarded. It was invisible.** In CLIP STUDIO a
        tone layer carries no halftone pixels: `LayerEffectInfo` holds the
        enable flag, density %, LPI and angle; `LayerLayerMaskMipmap` holds
        **the shape of the area**; and `LayerRenderMipmap` is **empty**, because
        the dots are composited at render time. Both the extractor and the
        surveyor read only `LayerRenderMipmap`, so every tone layer was logged
        as empty. **The foundation's instruction -- "you already identify them,
        so export instead of discarding" -- was right for ベタ and wrong for
        トーン, where the fix was to read a different field.**
      - **And this is the good outcome**: the area's shape and its density both
        survive intact, so **`mask coverage / 255 x density% / 100` is the
        density map outright, with no inference** -- against a project whose
        fill ground truth has so far been a radius-4 distance transform.
      - **"Has a layer" and "is used" are different facts.** Of 1,114
        tone-configured layers, only **444** have the effect enabled and area in
        the mask; the rest are empty templates duplicated on every page, so
        **the inventory's tone counts are inflated 2-3x**. Real use averages
        **3.4 layers per page** (max 12). Names are accurate about role (443 of
        444 carry a `60L 20%`-style name, one miss) but **7 layers' L notation
        disagrees with their actual LPI**, so density and LPI are read from the
        settings and never from the name. Measured: density 10/11/20/22/30/50%,
        LPI 30/50/60/61, angle 45 degrees and some 29.
      - **The period, measured rather than assumed, is 2015-2017** (2016: 13 of
        14 works; 2017: 14 of 17), falling from 2018 on. **The user's wording was
        right and the foundation's gloss of it was wrong** -- it read "the period
        that is not very old" as the recent side. **But the later years are not
        empty**: every locally inspected work -- 4th (2022b), hamlabi (2023b,
        2024b), housei (2024) -- has tone structurally. The table measures only
        whether a tone-*looking name* exists, so a later-year decline cannot be
        told from a change of naming.
        **And neither year table is deduplicated (flagged 2026-10-23 by the
        extraction side).** The 289 works contain **13 duplicate groups spanning
        year folders, 19 works' worth, so the real count is 270** -- `bakugi`
        alone appears in six groups and five works sit in three year folders
        each. **A work in three year folders is counted three times in every
        per-year row**, so both this tone table and the part-name table below
        are inflated by an amount that differs per year, and the direction
        depends on whether the duplicated work carries the feature. The
        duplicate list runs 2020b-2024c, so **the late years are the inflated
        ones**. **Deduplicated the same evening, and the mountain does not
        move** (`scripts/inspect/dedupe_works.py`,
        `config/work_duplicates.json`): **2015-2021 lose nothing at all** bar one
        work in 2019b, every duplicate sits in 2022 or later, and **the
        2016-2018 rows are unchanged to the work** -- tone 13/14, 14/17, 8/19 and
        part names 14/14, 15/15, 15/19. So *"2016-2018 is nearly every work"*
        **stands verbatim and Track H needs no correction**. Totals fall tone
        **77 -> 75**, ベタ **55 -> 53**, part-named **126 -> 117**, the last being
        the largest because the `bakugi` family carries part names. **What was
        inflated was the absolute count and not the shape**: the late-year
        shares barely move, since folding removes numerator and denominator
        together (part names 2024 11/22 -> 8/16, both 50%; 2022b 9/26 -> 7/22,
        35% -> 32%). **The quoting ban is lifted for the deduplicated rows
        only.** One limit to carry with them: **the representative was taken as
        the oldest year folder by presumption, not by rule** -- if a later folder
        is a revision then the older one is the wrong representative, and this is
        `ako6`'s `done` problem again, which neither side can settle from its own
        data. `ako8` is confirmed *not* a duplicate
        (`2019_ako8` and `2019b_ako8` have different fingerprints and different
        layer counts), so the pattern is per-work and not a rule about `b`
        suffixes. **The matching itself is bounded**: page name plus byte size
        over each work's eight-page sample, which is strong evidence and not a
        full byte comparison, and it does not separate "the same work filed
        twice" from "two versions of one work" -- the second being the same shape
        as `ako6`'s `done` folders.
      - **(B) tone output already exists and is tested on two works**,
        `dataset_tone_4th` (26 of 33 pages, 46 layers) and `dataset_tone_gakuen`
        (16 of 16, 87): per-layer **density maps**, 8-bit grey, white = no tone,
        one PNG per layer plus a composite, with a manifest carrying name,
        density, LPI, angle, coverage and the source `.clip` path. **Same canvas
        confirmed against real data**, not by precedent: 4961x7016 on 26 of 26
        pages of `4th`, matching the existing `4th_pageNNNN_line.jpg`. **No
        alignment work.**
      - **ベタ output is not started** and costs about **1.2 min/page** against
        0.67 s/page for the tone scan, because it needs every layer in the page
        rendered.
      - **Blocked on hardware**: the `F:` drive holding the manuscripts and the
        existing `clip_pairs` output is **offline**, so all of the above comes
        from a 6-work, 232-page mirror on `D:`. When it returns, a 7,663-page
        structural scan (about 25 minutes at 4-way parallelism) replaces the
        name-based table with measurement.
    - **Decided by the user 2026-10-23, and sent back the same day.** Scale: a
      few works as a trial, then widen -- but **split the cheap from the dear**,
      so the **full 7,663-page structural scan runs now** (0.67 s/page, ~25 min;
      **`F:` has been reconnected**) because it replaces the name-based table
      with measurement and retires the "renamed or absent" ambiguity, while the
      **1.2 min/page fill rendering starts on a few works**. Order: **ベタ first,
      トーン after**. Format: **density map plus a binary mask of the shape, and
      no halftone dots** -- in the user's words, the dots would multiply the data
      for nothing when what is wanted is an analysis of areas; the settings are
      recorded, so dots can be generated later if the look ever matters. The
      foundation's pick for the trial works was **`ako5` and `housei`**, on the
      stated reason that they are the pools every fill figure on record sits on,
      and **both the pick and the reason are RETRACTED the same day**. See the
      next bullet.
    - **The trial-works pick was wrong, and its reason was wrong in this week's
      recurring shape (2026-10-23).** The reason given was that every fill
      figure on record sits on `ako5` and `housei`. That is true of the **480px
      tile** pools and not of the corpus Track J actually used: the koma panel
      corpus `dataset/regions_clip_pairs_v3_koma_panels_20260826` holds **5,018
      panels over 57 works, 2015 to 2024, at native manuscript resolution**
      (panels run from about 900px to 4,600px on a side). **The foundation
      inferred the instrument's corpus from the corpus of the figures it
      happened to be quoting** -- the same act as reading a name as a
      measurement, on a fourth target this week. **And the manuscripts refused
      the pick independently**, which is the part worth keeping: the extraction
      side ran both works to the end and reports `ako5` **5 of 60 pages with
      ベタ, 5 layers** and `housei` **9 of 19, 9 layers**, because **`ako5`'s
      fill is not a separate layer at all -- it is dissolved into the line
      layer**. Per the author, both works are late-generation with complex
      manuscript handling. So **the work this project's tile figures rest on
      most heavily is a work whose manuscripts cannot supply an external
      standard**, and the author's "separated period" is a property of the
      **work**, not only of the year.
    - **The pick that replaces it: `ako6` (2018), on a convergence from both
      sides.** The extraction side's own two-stage screen -- a free pass over
      the survey JSON's per-layer pixel statistics (137 works pass), then
      rendering **only the candidate layers** and measuring darkness on the page
      through `CanvasPreview` -- put `ako6` top with **8 black fill layers on a
      single page**, the highest of a one-work-per-year scout. It is also the
      **third largest work in the koma corpus at 227 panels**. Next: `akogoods`
      179, `fringe3` 119, `ako10` 87, `copy` 74, `pc` 49, `hamlabi` 35,
      `gakusai3` 16. Not in the koma corpus at all: hitoduma, yakuza_c, fujiko,
      marisa, tama. **Two conditions sent with it**: use **train-split** panels,
      since the 969-panel held set spans only ten works and holds just `housei`
      45 and `gakusai3` 16 of the candidates -- an instrument check scores no
      model, so the holdout should stay unopened; and **state the resolution on
      both sides**, because the radius-4 mask is applied at native koma scale
      and **a panel's side varies five-fold**, so the radius does not mean the
      same thing across the corpus. That has to be settled before any agreement
      or disagreement is read as a fact about the mask (`sens_fill_radius.py`
      already carries radius 3 and 6).
    - **`ako6` confirmed and running (108 pages), and the visual check says the
      production fill is hair.** The extraction side composited the two ベタ
      layers of `ako6/page0086`: **two characters' hair, as outlined masses of
      paint**, matching the author's account that fill concentrates in hair,
      tops and tights. The layer names are `black` x2 and `線画 cloth`; in 2016
      (`phantom`, `poolsideIdle`) the name is **`線画 HEAD hair`** -- **this
      period's manuscripts separate the fill layer by body part**, which is what
      "a simple area layer" turns out to mean here. **Refined the same day by the
      author: it is not a period but a generation of the workflow, and
      separation and fusion go back and forth** (see Track L's entry) -- ako6
      holds separated and fused fill on the same pages, 77 of 102.
      **This changes what the surviving motive is a motive for.** The form that
      survived Track J's refutation is *"a person can look at the outline and
      tell which area is filled; cheap closing cannot extract it; what can is
      unmeasured"*. The production names say the thing the person is reading is
      **not shape but meaning** -- it is hair, so it is filled. If so, deciding
      an area is a **part-identification** problem, which is the same shape as
      Track H's problem (assign a stroke to a facial part), and **the layer name
      is itself the label**. Asked of the extraction side: what share of
      candidate works name their fill layers by body part. **Not yet a claim**:
      the author's knowledge is the author's to state, and this is one work's
      two layers plus a naming pattern in two more.
    - **The fill-area floor was lowered 1% -> 0.2% and this is not a bar being
      moved.** Recorded with the reason, because this project treats widening a
      population as the same act as moving a bar. The 1% belonged to a
      *different question* -- detecting whether fill had contaminated a line
      composite -- and was inherited rather than loosened after seeing a result;
      the measurement given is `ako6 page0086`'s hair fill at **0.57%** and its
      clothing fill at **0.63%**, both of which 1% was discarding, and a
      one-page trial that returned only 2 layers. **The decisive part: the
      manifest carries `drawn_ratio`**, so the continuous value survives and any
      floor can be re-cut downstream. The general form, adopted: **a collection
      threshold that keeps the continuous value is reversible and is not a
      pre-registration matter; one that discards is neither.**
    - **Delivery path renamed by content, not by the first thing that arrived.**
      `data/incoming_tone/` was created to receive tone and would have received
      ベタ first; **a directory named for one class, holding another, is this
      week's own trap in furniture form.** Replaced with
      `data/incoming_manuscript/<work>/fill/` and `.../tone/`. Masks arrive at
      **native canvas with no downscaling** (`ako6` is **7016x6543**, a spread,
      against 4961x7016 elsewhere) and dimension-matched to the existing
      `{prefix}_{page}_line.jpg` / `_koma.jpg`. **The radius scaling and the crop
      are this side's work** -- accepted deliberately, because a correction made
      here can be checked here, while one made on their side would make the
      mismatch invisible.
    - **The 2016-2018 manuscripts separate the *line* layer by body part too,
      and that is a Track H input, not an item 11 input.** Measured on the same
      2,048-page sample: **117 of 270 works (43%) have part-named layers**
      after folding the year-folder duplicates (126 of 289 before; the fold is
      recorded with the tone table above and moves no year before 2022), with
      the same mountain as the fill separation -- 2016 **14/14**, 2017 15/17,
      2018 15/19, falling to 0/11 by 2023. Counts: `線画 HEAD face` **2,332**,
      `線画 HEAD hair` **830**, `肌グラデ` 430, `線画 skin` 393, `線画 hair` 310,
      `線画 HEAD skin` 224, `線画 cloth` 141, then `SKIN` 52, `HAIR` 30,
      `TEETH` 26, `EYE` 19, `MOUTH` 18. **The extraction side's caveat is the
      load-bearing part and was adopted**: *a layer having a part name and a
      layer being an area are different facts -- most of `線画 HEAD face` is the
      face's **lines** and does not meet the fill signature.* Fifth instance of
      the same shape in one week, so it is taken as given rather than argued.
      **So it does nothing for the fill decomposition.** What it may do is Track
      H's work: H is paying author time to mark which stroke belongs to which
      facial part (518 rows over weeks, group recall 0.351 against a 0.80/0.60
      bar), and **830 `線画 HEAD hair` layers mean "this stroke is hair" was
      already decided in production**, which is exactly the hair/brow and
      hair/ear discrimination H19 is being built for. **The granularity is
      coarser than H's unit** -- eye, brow, nose and mouth are not separated
      (`EYE` 19, `MOUTH` 18), so it is face / hair / skin / cloth, not H's five
      parts, and it cannot replace H18's ground truth. **Routed to Track H to
      judge, not adopted here**: discipline 6 puts what a picture depicts with
      the author, and whether the names match the content is for H to check on
      the pixels. Three of the top part-named works are in the koma corpus
      (`2017_pink` 107 panels and in the held set, `2018_ako6` 227,
      `2017_copy` 74). Asked of the extraction side, for after the tone scan:
      **how many pages carry `線画 HEAD face` and `線画 HEAD hair` together**,
      since a page with only one of them teaches no distinction.
    - **Part-name co-occurrence, and the part-named *fill* population measured
      here from the delivered CSV.** The extraction side answered the
      co-occurrence question: of 2,025 pages, **429 (21%) carry part-named
      layers**, median 9 such layers on those pages (mean 12.4, max 56), and
      **`face` and `hair` appear on the same page 223 times -- 52% of
      part-named pages**, so the "a page with only one teaches no distinction"
      worry holds for about half of them and not the rest. Four parts together
      on 68 pages, six (adding `eye`/`mouth`) on **14**, which confirms by count
      that the granularity is coarser than Track H's five parts. Its part test
      is a disclosed regex on the layer name
      (`face|顔` / `hair|髪` / `skin|肌` / `cloth|服|pants|tights|タイツ` /
      `eye|瞳|眼` / `mouth|teeth|口`) and the names are in the CSV, so **the
      cut is re-definable on this side** -- the statistic was supplied, not the
      verdict, again.
      **Measured here on `inbox/fill_tone_inventory_layers.csv`, because the
      file is now local**: the part-named *fill* is far smaller than the
      part-named *line*. Of 299 `beta` layers only **27 (9%)** carry a part word
      (26 of them `hair black`), while of 192 `beta_in_line_name` layers --
      fills hiding in line-named layers -- **63 (33%)** do (`線画 HEAD hair` 37,
      `線画 HEAD face` 6, `線画 hair` 6, `線画 cloth` 5). **So about 90
      part-named fill layers exist, against 830 part-named hair *line* layers**:
      the part-name signal is on the lines, which is why it routes to H and not
      to the fill decomposition.
      **And the graded class has a non-tone member, which the four-class plan did
      not anticipate.** `肌グラデ` is **430 layers, every one classified
      `gradient`**, `black_among_drawn` median **0.000** and `drawn_gray_mean`
      median **201.8** -- a light, soft-edged area, over 70 works, named by body
      part. It is **not** a tone layer: no halftone effect, a direct gradient
      fill. So *graded area* is a production class reachable two ways -- graded
      tone and gradient fill -- and item 11's fourth class has a named
      population of 430 before any tone is split. The `gradient` class as a
      whole (1,999) stays unusable as a separator for the reason already
      recorded; what is usable is this named subset.
    - **The graded class is two constructions, not one, so the plan has five
      production classes and not four (2026-10-23 evening).** Measured on ako5's
      17 gradient layers: all are `LayerType=2` (gradient fill), carry no tone
      effect, and -- like tone -- keep **the shape in the layer mask with an
      empty body raster**. So the production record supports **line / ベタ /
      hard tone / blurred tone / gradient fill**, where the last two are both
      "an area represented only in part" by the author's description but are
      built differently and need **different instruments**: a blurred tone shows
      up in `mask_mid_per_boundary` (its mask is soft, max 209), a gradient fill
      does not (hard mask in 82.5% of cases, soft paint) and shows up instead in
      the **density range** its gradient uses.
      **The density figures, explicitly provisional.** From `GradationFillInfo`:
      `肌グラデ` **176-255, 3 steps, peak 31%**; `背景グラデ` 192-255, 2 steps,
      25%; `乳首グラデ 円` 120-255, 5 steps, 53%. Two things make them worth
      recording and one makes them quotable only with this sentence attached.
      **It reproduces**: all seven `肌グラデ` measured gave 176-255. **It carries
      its own known cell** -- the same parser on a flat `30L 10%` tone layer
      returns `[229, 255]`, and 255-229 = **10.2%** against the name's 10%, which
      is discipline 2 applied without being asked for. **And the parse is
      guesswork**: runs of four identical bytes in `GradationData` read as
      greyscale, not a spec that was decoded, so these are **an indication of
      density and not a density map**, and they are not a render.
      **Scope: ako5's 17 layers, out of 430 `肌グラデ` over 70 works.** A full
      pass exists (`scripts/inspect/survey_soft_fill_layers.py`, taking a layer-
      name regex) and waits until the tone scan's last group finishes, so the two
      do not slow each other.
      **Rendering the gradients is deliberately deferred.** It would answer
      "are these one class by appearance", but it needs the gradient's direction
      and shape decoded on top of a parse that is already guesswork, and **no
      decision waits on it**: item 11 is not a tree yet, and what it needs first
      is whether ベタ plus hard tone give an external standard at all. The option
      is kept, and `乳首グラデ 円` shows the shape is sometimes in the name.
      **ANSWERED by the author, 2026-10-23 night: two, not one -- and the
      answer brings a second thing with it that is bigger than the class
      question.** The user's words: merging them looks hard; **a gradient is
      basically "a tone that carries a gradation" as a frame**, and its uses are
      various -- skin and background are the commonest, but also whole-character,
      clothes and hair. **Only the skin-gradient use matches the purpose of skin
      tone shading (11% or 10%)**, which the measured vocabulary agrees with:
      11% is 26.3% of all used tone layers and 10% is 18.9%, the top pair.
      **And the re-reading that matters: the gradient's MASK is an area.**
      *"マスク全体は殆どの場合で面を表している可能性が高く、あるいは「そのコマの
      キャラクター以外」を示している場合は逆算してキャラクター自体を示す面を取れる"* --
      so where the mask marks everything **except** the character in that panel,
      **inverting it yields the character's own area**, and the author notes this
      happens with ordinary tone too but **more often with gradients**. **It is a
      source of information about large sets of areas.**
      **This corrects what this file wrote.** The user's original wording -- blur
      or gradation *"representing an area only in part"* -- was carried here as a
      property of the whole graded class. It is a property of the **ink**, not of
      the **mask**: a gradient fill has a **hard mask that is a complete area**
      and graded paint inside it, while a blurred tone has a **soft mask**. So of
      the two mechanisms, **the gradient is the better area source, not the
      weaker one** -- the opposite of how it was filed.
      **The author's verdict on scope: this is a track's worth of work on its
      own, and analysis comes first.** Recorded as a candidate track beside the
      colour manuscripts, not as part of item 11's class scheme.
      **What can be measured now, and what cannot.** The `gradient` class is
      **1,999 layers over 70 works**, and its name distribution is **not** what
      this file implied: **`乳首グラデ` 1,440 including its variants (72%)**,
      `肌グラデ` 430 (22%), `背景グラデ` 90. So the class is dominated by a small
      *local* shading use, and the area and inversion uses live in a minority of
      it. **The mask cannot be tested from anything on hand**: the inventory CSV
      has **no mask columns at all**, and its `drawn_ratio` is a body measure --
      **zero for 100% of tone rows, zero for 66% of `beta` rows and 26% of
      `gradient` rows** -- so a table of gradient `drawn_ratio` would be the
      "instrument not attached to the property" trap a second time, and was not
      published. **Requested, and already running before the request arrived**: the
      tone survey's mask columns for the `gradient` class
      (`scripts/inspect/survey_soft_fill_layers.py`, **85 works x 6 pages**),
      with the density range beside them.
      **And it carries the countermeasure this file should adopt as a rule: the
      instrument records what it was attached to.** The tool measures the body
      raster first and falls back to the mask, and **it emits a `source` column
      saying which one it read**. So "is this number about the mask or about the
      body" stops being a thing to remember and becomes **data on the row** --
      and the 26% of `gradient` rows whose `drawn_ratio` was non-zero will be
      explained rather than guessed at. The extraction side's own note is that
      **without that column it would have fallen into the same hole** this file
      stopped at. **Generalised: where a measurement can attach to more than one
      thing, the attachment point is a column, not a convention.** It is the
      stronger form of "ask what the instrument is physically attached to",
      because it does not depend on anyone asking. Expected to read `mask`
      throughout for gradient layers -- ako5's 17 all did -- **but not verified
      across the 85 works**, so a `raster` row is a differently-built layer and
      is to be read separately.
      **The first run is VOID, and the `source` column is what voided it
      (2026-10-23 night).** Of 1,754 layers it read **`raster` 1,196 and `mask`
      558**, so **68% of the rows have none of the mask columns that were asked
      for**. And checking 60 of the `raster`-classified layers, **all 60 do have
      masks** -- the tool simply never looked, because the body raster was not
      empty. Being rerun to emit **both** sets (`raster_*` and `mask_*`).
      **The lesson cuts at this file.** Two messages earlier the foundation read
      the design -- *"measures the body raster first and falls back to the
      mask"* -- in plain prose, **quoted it approvingly, and wrote the `source`
      column up as the countermeasure**. The extraction side had described the
      defect and neither side heard it. **So recording the attachment point is
      necessary and not sufficient**: a column saying what was measured helps
      only when someone reads **the distribution of that column** before reading
      any value. The provision that follows: **when an instrument can attach to
      either of two things, check how often it chose each, before quoting
      anything it produced.** The extraction side's own framing is that this is
      its turn at the trap the foundation stopped at an hour before, with the
      aggravation that **it had written the design out itself**.
      **What survives run 1 is the author's answer, confirmed at scale.** Split
      by `source` the two populations **separate completely**:
      `mid_per_boundary` p50 **0.721** on the 558 mask rows (96 above 1.0)
      against **131.1** on the 1,196 raster rows (878 of 888 above 1.0). **So a
      gradient layer's paint is almost always soft -- airbrush -- and its mask is
      hard**, which is exactly the author's *"hard mask, a complete area, with
      graded paint inside"*, and ako5's 17-layer reading now holds at 558.
      **And the pooled number was a trap of the first order**: mixed, it reads
      p50 77.3 with 67% blurred, so **publishing it would have concluded "67% of
      gradients are blurred areas" -- the exact opposite of the author's
      answer.** Two different surfaces averaged into one figure.
      **Run 2 landed 2026-10-24** (`inbox/soft_fill_layers.csv` / `.json`, 1,754
      layers over 85 works) **and the author's "the uses are various" is now a
      measurement.**
      **First, the masks are there**: `both` **1,185** / `mask` 558 /
      `raster` **11**, so **1,743 of 1,754 (99.4%) carry a mask with content** --
      run 1's 1,196 "raster" rows were `both` all along, unread because the tool
      stopped at the first non-empty surface.
      **Second, within one layer the two surfaces are opposite**: mask p50
      **0.744** (p90 1.266, 17.5% above 1.0) against raster p50 **131.1** (98.9%
      above 1.0). **The author's "hard mask, a complete area, with graded paint
      inside" holds at 1,727 layers**, where ako5's 17 had only suggested it.
      **Third -- the thing that was asked for -- `mask_cover` separates the uses,
      and the name predicts them:**

      | name | n | `mask_cover` p50 | edge p50 | soft edges |
      |---|---:|---:|---:|---|
      | `乳首グラデ` | 1,347 | **0.0012** | 0.792 | 281/1,331 (**21%**) |
      | `肌グラデ` | 321 | **0.0086** | 0.581 | 19/316 (6%) |
      | `背景グラデ` | 69 | **0.0252** | 0.701 | **0/67 (0%)** |

      **Coverage medians run 1 : 7 : 21.** `背景グラデ` is **entirely hard-edged
      and the smallest family (3.9%)** -- the most area-like use is the rarest,
      as the name distribution had suggested.
      **And "a gradient fill's mask is hard" is too strong**: 21% of
      `乳首グラデ` masks are soft, because a local shade is sometimes feathered
      in the mask itself. **The majority (82.5%) are hard**, which is what the
      author's account supports and all this file should say.
      **The "mask as a whole is an area / marks everything but the character"
      pattern lives in the tail**: of 1,754 layers, **59 (3.4%) cover >= 0.10 of
      the page**, 28 >= 0.20, 10 >= 0.30, 4 >= 0.50 (overall p50 0.0022, p99
      0.2452, max 0.9725). **These are a lower bound, being page-normalised** --
      a mask filling one panel of five reads as about 15% of a page -- and the
      intersection with the panel boxes remains this side's work.
      **Anchored on arrival**: recomputed here from the delivered CSV, `source`
      comes back **both 1,185 / mask 558 / raster 11**, `mask_cover` p50
      **0.0022** p90 0.0425 max 0.9725, `mask_mid_per_boundary` p50 **0.7440**
      p90 1.2660, `raster_mid_per_boundary` p50 **131.0670** -- every headline
      figure identical.
      Name distribution confirmed at this scale: `乳首グラデ` 1,301 + 28 + 18,
      `肌グラデ` **321**, `背景グラデ` 69, `マンガ用グラデーション` 13.
      **On the tone side the mask columns do exist, and the pattern is real but
      cannot be sized at page scale.** `mask_cover` over 17,695 used tone layers
      runs median **0.0088**, p90 **0.0704**, max 0.852, with only **3 layers
      above 0.5 and none above 0.9**. But coverage here is **of the page**, and
      the author's pattern is **per panel** -- a mask filling one panel of five
      reads as about 15% of a page -- so page normalisation **under-measures it
      by roughly the panel count** and these figures bound nothing. The tail that
      could hold panel-scale masks: **618 layers (3.5%) above 0.10, 162 (0.9%)
      above 0.15 over 29 works and 90 pages, 57 above 0.30**. Those are an
      **upper bound on candidates, not a count of them**, and the test needs the
      mask intersected with the panel boxes this corpus already has in
      `manifest.csv`. The densities in that tail are the ordinary ones (10%, 20%,
      30%, 11%) and the works at the top are `ako6` and `ako7` again.
      **Three independent rankings now put `ako6` and `ako7` on top -- fill,
      tone, and this high-coverage tail -- and the extraction side reads it as
      "these two separate their layers well". The general version of that is
      already refuted**: `ako5` is second for tone and supplies no fill standard
      at all, so **there is no single "separation quality" that ranks works once
      for every target**. What the agreement supports is narrower and still
      useful -- **`ako6` and `ako7` are jointly good**, which is why taking tone
      up later needs no new selection. Unverified either way, and flagged as
      such by the side that raised it.
      **The extraction side declined to say whether the two are one class, and
      that was correct.** Its reasoning: it holds **records of how the thing was
      made, not of how it looks**, and it had just been burned by
      `has_gradation_fill` -- a tool setting read as an appearance. **So the
      question of whether "blurred area" is one category or two belongs to the
      author** (discipline 6), and it only needs asking if a decision ever turns
      on it. The user's original wording -- *blur or gradation, mostly shading or
      background placement, representing an area only in part* -- describes an
      **appearance**, and the production record has now shown two mechanisms
      under it.
    - **Trial works: `ako6` first. `fringe3` is RETRACTED on author knowledge the
      same day** -- **the fringe manuscripts are drawn for colour, not
      monochrome.** The user's statement: a colour manuscript **puts colour on
      every area**, which makes it an excellent source for *knowing how areas
      divide* but **a different type of manuscript**, and taking it on deserves
      a separate track. All three volumes had been scouted (2019, 24/26/24
      pages: `fringe` 9 black / 1 white, `fringe2` 13/4, `fringe3` 9/4) and
      `fringe3` had been picked on match thickness (119 koma against 71).
      **The control argument that justified it was confounded.** `fringe` was
      chosen as a part-name-free contrast to `ako6`, to test whether part naming
      is load-bearing for area detection -- but it **differs on two axes at
      once**, part naming *and* colour, so either result would have been
      unreadable. **A control must differ on the one axis under test**; this one
      was picked for the axis it was wanted for and not checked for others.
      The replacement second work is **`2019_ako7`**, changed again the same
      evening after it was measured: `2024c ako` had been picked to serve the
      control framing, and when the framing went, **the work choice was left
      standing with nothing holding it up**. On the sampled eight pages
      `2024c ako` has **`beta` 0 and `beta_in_line_name` 2** against
      `2019_ako7`'s **19** and `ako6`'s **18** -- an order of magnitude, and the
      point of the second work is ベタ. `2024c ako` stays on the candidate list
      as the late-year example of a clean manuscript, with the correction that
      **its cleanness was zero white-fill contamination and not quantity of
      ベタ**. **The extraction side located the fault in its own list rather than
      in the reading of it**: `config/simple_fill_candidates.json`'s "black
      areas" column is a count over the **three or four pages it opened while
      screening**, so it says *this work has some* and not *this work has many*.
      **A screening number ranks nothing.** It is the same error as quoting a
      number without its construction, with the construction being the sample
      size, and it is now the fourth thing this week that was true of one
      population and read of another.
    - **The author's "ako6 through ako10 generation" is right for the first half
      and wrong for the second, and what changes is *which* separation
      survives.** The user's reading, 2026-10-23, was that this era's
      manuscripts are probably the good ones, naming ako6 to ako10. Measured on
      the inventory's eight-page samples:

      | work | tone% | ベタ | part-named | koma here |
      |---|---:|---:|---:|---:|
      | 2018_ako6 | 42.1% | 18 | **187** | 227 |
      | 2018_ako7 | 40.7% | 10 | **182** | 33 |
      | 2019_ako7 | 32.8% | 19 | **147** | 157 |
      | 2019b_ako8 | 27.3% | 11 | **8** | 33 |
      | 2020_ako9 | 26.5% | **2** | **10** | 93 |
      | 2020b_ako10 | 31.1% | **1** | **3** | 87 |
      | 2022b_ako10 | 31.5% | 4 | **1** | 6 |

      (`2019b_ako7` is omitted: it is the same manuscript as `2019_ako7`, filed
      under a second year folder.)

      **Tone separation holds all the way to ako10 at 26-31%, while part naming
      collapses at ako8 (182 -> 8) and the ベタ layers fall to 1-4 from ako9 on.**
      So the good window for ベタ and part names is **ako6-ako7, 2018-2019**, and
      the generation stays good only in the tone axis. **Not yet measured**: the
      sample is eight pages per work, so a ベタ count is per-eight-pages and
      ako9/ako10's 1-2 could be a sample that happened to miss them -- the gap
      from ako6's 2-per-page is large enough that this file reads it as real, and
      reads it as **unmeasured** until the page count is raised. **A 20-30 page
      pass over ako8, ako9 and ako10 is approved and queued** (about 45 minutes).
      **And the extraction side named the confound that pass still would not
      settle: a work with few ベタ *layers* is not a work with little ベタ.**
      `ako5` is the proof already on record -- its fill is dissolved into the
      line layer. So ako9/ako10's 1-4 layers could mean *the author stopped
      filling* or *the author stopped separating the fill*, and **a layer count
      cannot tell those apart**; the paper side can (black area from
      `CanvasPreview` or the line raster), and it comes out in the same pass.
      **Settled 2026-10-23 night, 25 pages per work, and the answer is "stopped
      separating", not "stopped drawing".**

      | work | pages with fill | fill layers | layers/page | thick black on paper (r4) |
      |---|---:|---:|---:|---:|
      | 2018 ako6 | 102/107 | 781 | **7.3** | not measured |
      | 2019 ako7 | 23/25 | 149 | **6.0** | 4.0% |
      | 2019b ako8 | 23/25 | 58 | 2.3 | 5.9% |
      | 2020 ako9 | 20/25 | 30 | 1.2 | 2.6% |
      | 2020b ako10 | 16/25 | 22 | **0.9** | **7.9%** |

      **The two measures are decoupled, and that is the finding.** Layers per
      page falls monotonically by **6.7x** across the five works, while the ink
      on the paper **has no trend at all** (4.0, 5.9, 2.6, 7.9) -- `ako10` has
      the fewest fill layers and the most thick black of any of them. **So the
      fill goes on being drawn and stops being separable**, exactly as `ako5`
      does. **And this file's "maybe the eight-page sample missed them" is
      refuted**: at 25 pages the ordering is unchanged and ako9/ako10 stay low.
      **The caveat the extraction side attached, which bounds the claim**: the
      paper measure is **not a fill area** -- `CanvasPreview` cropped to the
      white bounding box (some manuscripts are black outside the paper;
      `paper_fraction` is recorded), black at luminance < 128, eroded at radius
      1/2/4 with **the radius deliberately not chosen** (native radius 4 is about
      r1 in a preview at quarter size). It keeps line art, thick frames and
      balloon edges. So a work with more black *backgrounds* would read the same
      way, and what is established is **the decoupling**, which refutes "the
      author stopped filling" unless every extra mark is frame or background.
      **The general form, which this file needs because its whole "separated
      period" argument rests on layer counts: a layer count measures production
      practice, not content.** For finding training data that is the right
      measure, because what is wanted *is* the separated practice. For any claim
      about what the author drew it is the wrong one.
      **The extraction side then applied it to its own deliverables**, which is
      where it bites hardest: `fill_tone_inventory_works.csv`'s tone-layer and
      ベタ-layer columns are **layer counts**, and the tone column **includes the
      empty template layers**. **The whole-corpus scan puts the inflation at
      about five-fold: of 86,308 tone-configured layers, 17,695 are actually
      used -- 20.5%.** And the local 232-page figure this file had been quoting,
      **444 of 1,114 (40%)**, was **twice too high**, because those six works
      were tone-heavy: **a sampling bias of 2x inside a statement about
      sampling bias**. So every figure in the inventory is a figure about
      practice, and the whole file inherits this sentence rather than just the
      ako9/ako10 reading.
    - **The manuscript tree duplicates works across year folders too.**
      `2019_ako7` and `2019b_ako7` are identical down to the layer count (512),
      the ベタ count (14), the tone count (168) and the page file
      (`page0069.clip`), differing only in the year directory -- while
      `2019_ako8` (41 layers) and `2019b_ako8` (238) genuinely differ. **So the
      289-work figure may count fewer real works than it says.** Same shape as
      this corpus holding one page at several stages and as the extraction
      side's colliding output names: **duplication is present at every layer of
      this pipeline**, and the safe assumption is that a count of anything here
      is an upper bound until the duplicates are tested for.
      **But the control framing is dropped entirely, because the extraction side
      turned the rule back on the replacement and it holds.** `ako6` and
      `2024c ako` differ in **year (six apart), page count (108 vs 139) and part
      naming**, so they are not a one-axis contrast either -- the first version
      of this bullet had merely swapped one confound for others. **They are two
      trial works chosen for coverage, not a controlled comparison.** Whether
      part naming is load-bearing for area detection needs a design that can
      actually test it -- matched pages inside one work, or part names stripped
      from the input -- and that is a question for item 11 once it has a tree,
      not something a work choice can answer. **The rule caught its author's
      second attempt as well as the first**, which is the strongest evidence it
      is worth keeping.
    - **Colour is a property of the page, not of the work -- the work-level
      answer was the wrong *shape* of answer, and the extraction side retracted
      its own (2026-10-23 night).** Over 289 works: **214 all-monochrome, 28
      all-colour, 47 mixed.** Two reciprocal counter-examples kill both simple
      rules: **`fringe2`, which the author calls colour, reads monochrome on 2
      of 5 sampled pages** (its siblings read 4/5 and 5/5), so a majority vote
      fails; and **`2019_ako8`, a monochrome manga, has one page at 92% coloured
      pixels** -- a cover -- so "any page with colour" fails too. Monochrome
      books carry colour covers and frontispieces, and colour books have pages
      with little colour in them. **So `work_color_mode_pages.csv` (1,348 rows,
      one per page) is the file to use and exclusion is per page**; the
      work-level column is kept for reference and is wrong for 47 works.
      **And the earlier "`2024c ako` is monochrome" was right only by luck** --
      it happened to be 0 of 5 with no mixture, so the form of the answer was
      never tested. **This is the day's recurring failure in its fifth unit**:
      name against signature, layer against content, screening against ranking,
      population against measurement, and now **work against page**. Each time
      the number was fine and **the thing it was a number *of* was wrong**.
      Five candidate works are all-colour and drop off the list: `2019_plecia`,
      `2019b_hitoduma`, `2021_yakuza_c`, `2022b_fujiko`, `2024c_tama`.
    - **The colour question is measured now, not inferred, and the tone-share
      hypothesis below is deleted by it (2026-10-23 evening).** The extraction side read
      **saturation off `CanvasPreview`** -- the page image CSP itself rendered,
      1240-1754px, a few hundred KB per page -- so it asks *is there colour on
      the paper* instead of reasoning from layer names, tone share or colour
      settings (`scripts/inspect/check_color_pages.py`). **`2024c ako` is
      monochrome: 0.000% coloured pixels, mean saturation 0.00 on five sampled
      pages**, against `fringe`'s **8-19%**, so it is confirmed as the second
      trial work. The check carries its own known cell -- `fringe`, already known
      to be colour, comes back 4 of 5 pages colour -- and **the fifth page falls
      on the monochrome side at 1.80%**, which is the useful part: a colour
      manuscript has pages with little colour in them, so **the judgement is
      per-work and not per-page**. Its threshold (saturation >= 24 on >= 2% of
      pixels) is **declared uncalibrated**, justified only by JPEG bleed
      reaching about 10. **It also supersedes the author-label falsifier**: the
      same check runs over all 289 named works, so the tone-share hypothesis can be
      tested against a measurement instead of against hand labels, and the 46
      second-stage candidates take minutes. **Requested.**
    - **`ako6/color/` is one colour cover, and the reconciliation is exact.**
      70.92% coloured pixels, mean saturation 47.85, a single `.clip` beside
      `H1H4.psd` / `layoutH1H4.psd` (H1H4 = front and back cover), while
      `ako6/`'s own pages read 0.000%. **Excluded from delivery for the same
      reason as fringe.** And `ako6` has `.clip` files in **10 subfolders, 108
      pages; minus the cover, 107 -- matching this corpus's 9 slugs exactly**, so
      nothing is missing on either side (the koma corpus draws panels from 63 of
      those pages).
    - **A candidate signature for colour manuscripts, now a hypothesis with a
      better instrument pointed at it.** In the sampled layers,
      the four works known or named as colour carry **zero tone layers** --
      `2022b_hyoui_color` (adjustment **14.7%**, unknown 47.1%), `2019_fringe`
      (unknown 37.1%), `fringe2` (40.0%), `fringe3` (42.6%) -- while every
      monochrome work in the koma corpus runs **15.8% to 42.5% tone**
      (`ako6` 42.1%, `ako5` 42.5%, `pink` 41.6%, `copy` 32.7%, `pc` 15.8%).
      **But zero tone cannot be the rule**: 212 of 289 named works have zero tone in
      their sampled pages, because tone is sparse per page (group 1 found tone on
      212 of 1,724) and an eight-page sample misses it. Inside the koma corpus the
      set is small and enriched -- **6 of 57 works are zero-tone, 427 panels**,
      three of them the fringe family and `hyoui_color` -- which is suggestive and
      is not a measurement. **The falsifier to run before anyone relies on it**:
      have the author label a dozen works colour or monochrome and see whether
      tone share separates them; a high `adjustment` share is the second
      candidate axis and only nine works carry one.
    - **Colour manuscripts as their own track, the user's call (2026-10-23).**
      The reasoning is the author's and this file could not have produced it:
      **colour places a colour in every area, so the area division is complete
      rather than sparse** -- against ベタ, which covers hair, tops and tights and
      nothing else. That makes it the richest source of area ground truth on the
      shelf **and** a different problem, since a coloured area is not a filled
      area and the decision being learned is not the same decision. It is
      recorded here as a candidate track and not as part of item 11; the corpus
      already holds **427 panels over 6 candidate works**, `2022b_hyoui_color`
      the largest at 190.
    - **The extraction side's output naming would have overwritten pages
      silently, and this file's version report is what caught it.** Its name was
      `{prefix}_{page}_beta.png` with no subfolder, so `ako6/page0028` and
      `done2/page0028` collided. Now
      `018_2018_ako6_<subdir>_<page>_beta.png`, which maps to this corpus's
      `pid` by replacing `__` with `_`. **Fourth time this week one side's check
      found the other's defect** (519, `black`, the foundation's population, this
      naming) -- and the only reason any of them surfaced is that both sides were
      reporting measurements rather than conclusions.
      Also recorded, from the same reply: **the composited `_beta.png` has the
      floor already applied, so a re-cut must be built from the per-layer PNGs.**
      Reversibility lives in the per-layer output and not in the composite,
      which is the concrete form of the threshold rule above.
    - **`ako6` delivered 2026-10-23, and it is the first production fill this
      project has ever held.** `data/incoming_manuscript/ako6/fill/`, **883 PNGs
      + `beta_manifest.json`, 78MB**: **107 pages, 102 with fill, 781 fill
      layers, 0 errors**, one canvas throughout (**7016x6543**), composite
      coverage mean **8.8%** / median 5.9% / max 43.5%. Checked on arrival --
      every file on disk is named in the manifest and every name resolves to a
      file, 883 both ways. **The decision rule is recorded in the manifest
      itself** (`black_among_drawn >= 0.95`, `drawn_gray_mean <= 12.0`,
      `drawn_ratio` in [0.002, 0.9], `preview_dark >= 0.5`, name families
      `nombre`/`balloon`/`text`/`furniture` excluded), which is what makes a
      re-cut possible.
      **The name-vs-signature lesson paid for itself here**: by name family the
      781 layers are **`line` 549, `beta` 193, `unknown` 39** -- so **70% of the
      production fill lives in layers called `線画 ...`** (`線画 hair` 171,
      `線画 skin` 154, `線画` 144, `線画 cloth` 52) and a name-based design would
      have dropped them. `black` is 177, the name that caused the 84 -> 55
      correction.
      **Two notes from the arrival check, in the order their size deserves.**
      (i) **Three layers of 781 (0.4%) come from a `sketch set` folder**, one of
      them named `下絵`. Seeing that one record first, this file was about to
      report a defect; measured, it is 0.4% and not a pattern -- **the same
      error it spent the day recording, caught on itself.**
      (ii) **The signature has no shape term, and this is the useful half.**
      Colour purity, coverage and page darkness cannot separate *a pure-black
      fill* from *a pure-black dense line layer*, and 469 of the 781 are
      line-named. **This project holds exactly the missing instrument**:
      `fill_mask.py`'s radius-4 opening is a shape test -- the region that can
      hold a disk of that radius -- which a stroke fails and an area passes.
      **So the instrument check is bidirectional and the circularity worry
      resolves differently than it was framed**: they test colour, we test shape,
      so **agreement is informative and disagreement says which side to look
      at**. **But "neither was built from the other" -- the clause this file
      wrote and praised twice -- is WRONG, and Track L found it by walking into
      it (2026-10-24).** The two axes are not independent: **a layer that fails
      the colour test never reaches the shape test at all.** The colour test is
      the entrance, and it narrows what the shape test can ever see. **So the
      pair is not symmetric and agreement cannot be read as two witnesses.** It remains a track's job to
      run, not the foundation's (discipline 5) -- and **it is now Track L's
      first job**.
      **The extraction side confirmed this with numbers the same night, and the
      consequence lands on its own delivery.** Of the 124 part-named layers it
      exported for Track H, **63 pass `black_among_drawn >= 0.95`** -- half of
      them meet the fill signature's colour test -- but their `drawn_ratio` runs
      **median 0.0004, min 0.00004**, and **only 3 of the 63 reach the
      inventory's 0.01**. That is not an area: **this corpus draws line art
      without anti-aliasing, so its strokes are 100% pure black.** Its own
      conclusion: the point applies to the 781 layers it delivered, because
      **size is not shape** -- 0.5% as a bundle of thin strokes and 0.5% as a
      blob give the same number.
      **Measured here on the manifests, which is inspection and not an
      experiment**: the delivered 781 run `drawn_ratio` p10 **0.00226**, median
      **0.00431**, p90 **0.02298**, max 0.3964, and **75.7% sit below 0.01**.
      So **the delivery's own floor of 0.002 already removes the bulk of the
      1-bit line layers** (whose median, 0.00045, is well under it), and it
      cannot remove a *dense* line layer above that floor, with three quarters
      of the delivery inside the band where the two populations can overlap.
      **So the 781 are candidate fills until the shape test runs.** The header's
      claim that this is the first production fill this project holds stands;
      the claim that all 781 are fill does not, and was never made.
    - **Track H's five pages arrived the same night**:
      `data/incoming_manuscript/part_layers/`, **129 layer records, 24MB** --
      each part-named layer as its own PNG at full canvas in the same position,
      a per-page JSON (layer id, name, parent folder, matched part word, pixel
      statistics, whether it rendered), and **the line composite copied from the
      August delivery rather than rebuilt**, so it is the very file this
      corpus's panels were cut from. Five layers did not render because they are
      **empty**, recorded as such. **The extraction side judged nothing**, which
      is what H asked for. One row it flagged without interpreting:
      `gakusai1-3DL p15`'s `線画 HEAD face` has **`drawn_ratio` 0.9984** -- a
      part-named layer covering 99.8% of the page in pure black, probably
      inverted or an underlay, **unverified**.
      It also found the cause of the three `sketch set` layers in the ako6
      delivery: its exclusions are by **name family** only and never look at
      whether the **parent folder** is a sketch folder. `ancestors` is in the
      manifest, so this side can drop them. **It declined to change the
      delivery rule for 0.4%**, on the reasoning that doing so would be the same
      kind of judgement as ranking by a screening number -- so it waits to be
      asked.
    - **`ako7` is being delivered too, by the author's instruction (2026-10-23
      night).** 72 pages (`ako7/` 53, `done/` 13, `done2/` 5), identifiers
      `026_2019_ako7_<subdir>_<page>` matching this corpus's `work_id`, landing
      in `data/incoming_manuscript/ako7/fill/`. **`color_rejected/ako7.clip` is
      excluded at 25.4% coloured pixels**, the same treatment as `ako6/color/`.
      `2019b_ako7` is untouched, being the year-folder duplicate (fingerprints
      agree; this corpus has 157 panels under one and 0 under the other). At
      6.0 layers per page the full pass should land near 430 fill layers, which
      would roughly halve again the distance to a second work's worth of
      production fill.
    - **`2019_ako8` is a different kind of thing and must not be counted as
      ako8.** Not the duplicate question -- that was settled, the two ako8
      entries genuinely differ -- but the **character**: `2019_ako8` has fill on
      **2 of 25 pages and 0.1% black on the paper**, where every other work in
      the sweep runs 9-18%. Probably roughs or an unfinished bundle,
      **unverified**. **The ako8 row above is `2019b_ako8`.** So the year-folder
      pair needed two separate checks -- are they the same file, and are they
      the same kind of material -- and only the first had been run.
    - **`ako7` delivered 2026-10-23, so the production fill now covers two
      works.** `data/incoming_manuscript/ako7/fill/`, **465 PNGs +
      `beta_manifest.json`, 44MB**: **71 pages, 66 with fill, 399 fill layers,
      0 errors**, one canvas throughout (7016x6543), composite coverage mean
      **6.9%** / median 6.8% / max 23.0%, thick black on the paper median 4.0%.
      Checked on arrival here, and **the delivery is ignored by git** -- the
      `data/` rule added after the incident holds, so nothing was staged.
      **The name split repeats ako6's almost exactly**: `line` **299**, `beta`
      74, `unknown` 26 -- **75% line-named against ako6's 70%**, which is the
      second instance of production fill living in layers called `線画 ...`.
      **Together: 178 pages, 1,180 fill layers**, against an estimate of ~430
      for ako7 and ~1,200 in total.
      **This is the second case for the question that matters most** -- whether
      this project's line rasters structurally lack the separated fill. Track L
      has pre-registered the readout: **`black`'s median `keep4` and the
      `beta`-named ink-presence rate, with ako6's 0.994 and 0.1421 in the same
      run as the known cell.**
    - **Relayed for Track H, 2026-10-23**: five pages exported layer by layer at
      full canvas, with a JSON of layer id, name and parent folder, so H and the
      author can check **whether the part names match their own pixels** against
      the author's per-stroke marks. Author's approval on record ("let's look at
      a few pages"). H picked pages that carry marked training faces and matched
      the versions, and deliberately included two pages holding
      `beta_in_line_name` layers as cases where the name is not the content.
      `inbox/request_track_h_part_layer_export_20261023.md`; **H states the
      judgement is the author's and the extraction side is asked only to
      export** -- the division discipline 6 asks for, arrived at by the track
      without being told.
    - **Delivered 2026-10-23 to `inbox/`**: `fill_tone_inventory_layers.csv`
      (45,533 rows), `..._works.csv` (289), `..._inventory.json`,
      `tone_layers_local.csv` (1,114 structurally-confirmed tone layers with
      edge-hardness statistics), `reply_fill_tone_layers_20261023.md` (whose
      figures predate the `black` correction; the CSVs do not) and a one-page
      notice. The structural scan stands at **3 of 4 groups complete**.
    - **A second category inside tone, from the author, and it changes
      candidate 5.** The user's wording: *"some tone is not a cleanly cut area
      but has blur or gradation (mostly shading, or placed in the background);
      these represent an area only partially, so extract them, but keep them
      separable."* So **tone holds two different things** -- **hard-edged tone,
      which is an area**, and **blurred or graded tone, which is shading and
      represents an area only in part**. The decomposition this project has
      framed as 線 / ベタ / ツヤベタ therefore lands on **four** production
      classes: line, ベタ, hard tone, graded tone. **The ask to the extraction
      side was for the statistic and not the verdict** -- per layer, the share of
      mask pixels that are neither 0 nor 255, plus whether the layer is of the
      `gradient` class it already separates (1,999 layers) -- so that the
      threshold stays on this side. **This is the furniture lesson inverted**:
      there, one pixel statistic collected unrelated classes; here, one name
      would collect two kinds of area. **And it makes candidate 5 answerable** --
      "do highlight fills separate from plain fills by shape alone", which the
      radius-4 mask cannot do, becomes a question with production labels.
    - **A circularity the foundation flagged before sending its measurements.**
      The extraction side asked for our figures to validate its layer
      thresholds. But **this project's fill ground truth is the radius-4 inferred
      mask, and it has never been checked against anything external**. If their
      thresholds are tuned to agree with that mask and we then check that mask
      against their layers, the check is circular -- the instrument measuring
      itself, which is this file's most repeated failure. **The protocol sent**:
      their classification is fixed from layer structure and names, which are
      independent of us; our numbers are for scale only and are not a target to
      fit; the mask comparison happens **once, after their thresholds are
      frozen**, and is not iterated.
    - **Request sent 2026-10-23** to the manuscript-layer session on `um350`
      (now 192.168.1.5; host key verified identical to the old .9 entry, key
      auth confirmed), asking for the inventory first and output second. Text:
      `outbox/request_fill_tone_layers_20261023.md`. **It went by cross-session
      message over Remote Control, a route that reports nothing back** -- so
      delivery is known and reading is not, and silence is not agreement.
      **The user expects the extraction to take days.** The reply will land in
      `inbox/` or as a message; `ls -t inbox | head` is step 2 of this project's
      start-up procedure, which is what will catch it.
    - **Group 1 of the scan came back the same day, and it moved three of the
      numbers in this file (2026-10-23).** **1,724 pages over 9 works: 212 pages
      carry tone, 595 tone layers in use.** The scan went from **3.1 to
      0.28 s/page** once the layer read was narrowed, which is what makes a full
      pass over the newly-connected `F:` drive discussable at all rather than a
      quoted month. Two corrections arrived with it and are recorded as traps
      above -- **ベタ works 84 -> 55, ベタ layers 1,246 -> 299**, and the white-fill
      blind spot in `plane0` that `preview_dark` now covers. **The direction of
      both is the same: the first figures were too high**, so nothing that was
      planned on the strength of them is safe without re-reading.
    - **The hard/graded split is a real split and NOT a clean one -- the
      "bimodal, so the threshold needs no judgement" reading is RETRACTED, and
      the extraction side retracted its own word first (2026-10-23 night).**
      `mask_mid_per_boundary` is the share of mask pixels that are neither 0 nor
      255, normalised by the shape's boundary length, so a hard edge scores near
      its own perimeter and a graded one far above it.
      **On the first group (595 layers)** it ran **p90 0.929 -> p95 7.874** -- an
      order of magnitude between adjacent deciles -- and **54 layers (9.1%)**
      above 1.0, which this file wrote up as *two populations, not a continuum*,
      with the threshold settable by measurement.
      **On all 17,695 used layers the valley fills in**: p10 0.000 / p25 0.000 /
      p50 0.329 / p75 0.812 / **p90 1.058 / p95 1.797 / p99 6.136** -- continuous
      where the sample jumped. The share survives (**1,916 of 17,695 = 10.8%**
      above 1.0, against the sample's 9.1%) and so does the shape in the weak
      sense -- **there are two hills, but the ground between them is not
      empty**. The order-of-magnitude gap was a property of the first group
      (2024, 2023b, 2020). **So the cut at 1.0 sits around p90 with layers in
      its neighbourhood, and the count moves with where it is placed**: any
      "blurred tone = N layers" figure must carry its threshold and a
      sensitivity, and **the threshold is a choice, not a reading.** This
      corrects what was said to the user as well as what is written here.
      **The `gradient` class cannot carry the split either**: all 595 layers in
      the sample had `has_gradation_fill = True`, so that flag records a tool
      setting and not a look -- the second name that week about production
      rather than appearance, after `black`.
    - **The whole-corpus tone survey, 2026-10-23: this is item 11's population.**
      7,667 pages: **17,695 used tone layers over 2,047 pages (26.7%) and 79 of
      243 works**. Files in `inbox/`: `tone_layers_full_layers.csv` (one row per
      used layer with the edge statistics, 17,695), `..._works.csv` (243),
      `..._full.json`, and the notice.
      **The halftone vocabulary is tiny, which bounds what "hard tone" has to
      be**: five densities -- **11% / 20% / 10% / 30% / 22% cover 96%** -- and
      **60L alone is 91.5%** of line counts, 99.8% with 55L and 30L. Effectively
      **5 densities x 3 rulings**, so a hard-tone class can expect a small
      discrete set rather than a continuum.
      **By layers the year peak is sharper than by works**: 2016 2,202 / **2017
      8,413 (48% of all used layers)** / 2018 2,967 / 2019 843, then 162-565 a
      year. The work-level mountain said "2016-2018 is nearly every work"; the
      layer-level one says **2017 is nearly half the material**.
      **And the two works chosen for fill are top for tone as well** -- `ako6`
      1,129 used layers over 100 of 108 pages, `ako7` 484 over 62 of 72 -- so
      taking tone up later needs no new work selection. **`ako5` is second for
      tone at 1,311 layers** while being the work that could supply no fill
      standard at all: **a work's quality swaps with what is being extracted**,
      which is the same lesson as a population being quotable only with its
      measurement. (`2019_ako7` and `2019b_ako7` agree to the layer -- 484 /
      311 / 173 -- confirming the duplicate a third way.)
      **Anchored on arrival, and the one divergence was the foundation's own
      join.** Recomputed here from the delivered CSV: the edge percentiles come
      back **identical to three decimals** (p50 0.329 / p75 0.812 / p90 1.058 /
      p95 1.797 / p99 6.136), the density vocabulary **96.6% in five values**
      (11/20/10/30/22), **60L 91.5%**, 2017 **8,413 layers = 47.5%**, `ako5`
      1,311, `ako6` 1,129, `ako4` 1,343 -- all matching. **The two figures that
      did not match were ones this file computed by work *name*:** `ako7` came
      out 1,140 against their 484, and `ako4` 1,390 against 1,343, because the
      name pools **2018_ako7 172 + 2019_ako7 484 + 2019b_ako7 484** and
      `2024b_ako4` 47. **Keyed on (year, work) every figure agrees.** The fifth
      unit of the day's failure, caught inside the check that was anchoring
      someone else's numbers -- which is the argument for discipline 2 as such:
      the known cell did not just confirm their arithmetic, it located the one
      place this side was still joining wrongly. (Their count has a second
      denominator worth keeping: **1,916 above 1.0 out of the 17,550 layers that
      have a value**, not of 17,695 -- 145 carry none.)
    - **How Track J's door closes changes how L reads candidate 2, and J sent
      this ahead of its verdict (2026-10-23).** If C3 fires, *"no learned model
      is needed"* will most likely hold in the form **"both arms are near
      zero"** rather than *"the non-learned arm won"* -- the best non-learned arm
      beats *paint nothing* by only **+0.0026**. These are different claims and
      J will print the distinction. **"Both near zero" closes the formulation and
      leaves the semantic route open; "the non-learned arm won" would say the
      frequency table is already enough.** L counts the eighth door accordingly.
      **Settled 2026-10-23: the door closed in form B.** Every arm sits in the
      `blank`-to-ceiling band, so **the eighth door is "this formulation does not
      predict fill", not "the non-learned route suffices"** -- which leaves the
      semantic route open, as J predicted. **And Track J asked that item 11 be an
      independent track rather than its own next stage**, matching the user's
      decision of the same day; its three reasons are worth keeping: the verdict
      refuted *a per-pixel U-Net against this mask at this scale* and not a
      ground-truth decomposition; **J's own next seat should stay open for a
      region proposer**, which is neither the cheap morphology nor the U-Net
      already refuted; and item 11 **meets the external-standard problem before
      J does**, since a decomposition whose ground truth is the radius-4 mask is
      only a decomposition of an estimate. **A clean negative should close as a
      negative** rather than be softened by a successor stage inside the same
      track.
      **The motive for item 11, in the form that survives** -- Track J's wording,
      replacing the retracted one: **a person can look at the outline and tell
      which area is filled; cheap closing cannot extract it; what can is
      unmeasured.**
    - Open before a registration: approval for the L tree, and whether to create
      it now or after the inventory returns. Track J's verdict (item 5) bounds
      candidate 2 and is still unread.
    - The retracted enclosure framing had a **fourth** store nobody here knew
      about: Track J had copied it into the briefing it rewrote the same morning,
      which is the first file a fresh session in that tree reads. J found and
      fixed it unprompted, with the 2026-10-10 measurements beside it. A
      retraction pass is only as wide as the list of stores, and this one was
      wider than the foundation's list. **And Track J supplied the mechanism,
      which is better than the observation**: the foundation's list was not
      careless -- at the moment it was drawn, J's briefing existed only in its
      thirteen-day-old version and **the sentence being retracted had not yet
      been written there**. So **when a retraction and a rewrite run on the same
      day, the store list is structurally one short**, because a new store is
      being created while the pass is walking. J's countermeasure, adopted: on a
      day it rewrites its briefing, it posts one line to `../lineart/inbox/`
      saying what it copied into the current-state section, which is enough for
      this file's list to catch up.
    - **Amendment J3 (disclosure only), approved 2026-10-23**: Track J's 956-panel
      headline will name its mask -- *"against the radius-4 estimated mask"* --
      beside the registered `MIN_REGION_PX = 500` / `MATCH_IOU = 0.5` region-F1.
      No bar, number or population moves; the population was always that. Same
      shape as J's own amendment J2, which made the headline name its metric.
      **The condition the foundation attached**: the qualifier must not become a
      reservation. The mask is common to every arm including `blank` and
      `all_black`, so **C1-C3 are comparisons on one shared mask and the caveat
      does not weaken them** -- what it bounds is any absolute claim about how
      much fill is really recoverable. A clean negative must not be softened by
      a caveat that does not touch it.
