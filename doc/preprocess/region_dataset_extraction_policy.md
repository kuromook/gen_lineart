# Region Dataset Extraction Policy

## Alignment Gate vs Style Gate

Refined 2026-07-26 after comparing model output across ako5ver2 native,
fitness, housei, and fighting trained together: the tile filter must keep two
kinds of rejection structurally separate, because they answer different
questions and have different scopes.

**Alignment gate** — does the rough edge sit on top of the line edge. A purely
geometric question (chamfer distance, edge correspondence at a fixed pixel
tolerance). The criteria are fixed constants shared by every source
(`ALIGNMENT_*` in `tools/pair_extraction/tile_region_manifest_480.py`), never a
per-dataset CLI argument. Alignment quality is not a property of a source's
drawing style; a rough stroke either lines up with its inked counterpart
within N pixels or it does not, regardless of which manuscript it came from.
If the fixed constants ever need to change, that is one deliberate decision
made with evidence for all sources at once (see "Determining The Alignment
Threshold" below), not a per-source tuning knob.

**Style gate** — stroke width, gray/soft-ink fringe, ink density, black-fill,
panel-border rejection. These encode a source's drawing style (pencil vs
marker, thin vs thick default line weight) and are expected to differ per
dataset. Keep these as CLI arguments and tune them per source, using
`tools/pair_extraction/diagnose_gate_funnel.py` to find which one is actually
the yield bottleneck before loosening it (see `doc/dataset_status.md`'s housei
section for an example: `soft_ink_ratio` was the housei-specific bottleneck,
not the tile-score cutoff).

Do not conflate the two. Evidence for why this split matters: comparing
chamfer distance across ako5ver2/fitness/housei/fighting's already-accepted
tiles showed a clean correlation with trained-model output crispness (lower
chamfer, crisper output) that no style-gate metric showed on its own — see
`doc/raw_dataset_extraction_knowledge.md`. Style-gate tuning alone (e.g.
loosening `soft_ink_ratio` for housei) does not substitute for alignment
quality, and alignment-gate tuning should never be conflated with a source's
drawing-style tuning.

### Determining The Alignment Threshold

The alignment gate's specific numeric thresholds (how tight is "aligned
enough") are still being determined empirically as of 2026-07-26; do not treat
the current constants as final. A first attempt at post-hoc chamfer filtering
(`chamfer <= 12`, informally derived from fighting's own distribution) did not
produce a visible model-output quality improvement in a direct same-sample
comparison against the unfiltered pool, despite the cross-source correlation
holding. Open questions for the next investigation:

- whether the threshold needs to be substantially tighter than 12 (fighting's
  own median was 10.45, so 12 barely excludes fighting's own weaker half)
- whether chamfer alone is a sufficient alignment metric, or whether the
  strict-tolerance edge recall/precision pair needs its own independent
  cutoff rather than riding along with chamfer
- whether 10-epoch short warmstart training is sensitive enough to reveal an
  alignment-driven improvement at all, independent of the data change

## Cross-Dataset Strategy: What Is Universal, What Is Per-Source

Confirmed across ako5ver2, fitness (formerly kurip), housei, and fighting
(formerly lineart) in the 2026-07-25/26 sessions: the model-training pipeline
is comparatively stable, but mining usable tiles from raw manuscripts is not
one fixed procedure. Each raw source needs its own extraction strategy inside
a shared framework, not a single copy-pasted recipe.

### Universal (do this for every source, no exceptions)

1. Never assume same-coordinate correspondence. Run
   `tools/pair_extraction/diagnose_pair_alignment.py` first and let its
   recommendation pick the route (see `doc/EXTRACTION_RULES.md`).
2. The finished line image is always the anchor; search the rough page for
   matching content, never the reverse.
3. Materialize and tile at native scale (`src_per_out` near 1.0), not a fixed
   long-side normalization that mixes scales across regions of different
   source size. See "Source Scale Band" below.
4. Generate a valid mask before scoring/training, to protect against partial
   content mismatch (layer differences, added/removed props, panel
   boundaries).
5. Review at full native tile resolution — both the top and the tail of any
   ranked candidate set — before fixing a score cutoff. Thumbnail QC hides the
   exact failure modes that matter: ako5ver2 had a top-gate-passing tile with
   unrelated rough/line content that only a full-resolution look caught;
   housei's yield bottleneck (a specific gate, not the score cutoff) only
   showed up by tabulating a gate funnel, not by eyeballing thumbnails.
6. Dry-run, review, save, integrity-audit, in that order. Never skip a step
   because a source "looks easy."
7. Record what gate values were used and why per source in
   `doc/raw_dataset_extraction_knowledge.md`, including which gate was the
   yield bottleneck (`tools/pair_extraction/diagnose_gate_funnel.py`).

### Per-source (expect these to differ, do not carry values over blindly)

1. Which route applies: `same_coordinate_ok` / `needs_global_or_local_alignment`
   / `needs_region_matching` / `hold_for_manual_review`. Decided by the
   diagnose tool per source, not assumed from a similar-looking prior source.
2. Matching mechanism weight follows directly from the route, and the two
   observed so far are genuinely different procedures, not variations of one:
   - **Heavyweight (`needs_region_matching`)** — used for ako5ver2. Line-side
     panel/character-scale region proposals (`match_hamlabi_regions.py`),
     rough-side translation+scale search, recursive child search inside
     oversized parents (`refine_hamlabi_large_regions.py`), then post-align.
   - **Lightweight (`needs_global_or_local_alignment`)** — used for fitness and
     housei. Line side scanned on a fixed 480px grid, each window's rough
     correspondence found by local offset search only
     (`match_kurip_regions.py`, name predates use beyond its original source).
     No panel identification, no recursive parent/child split.
   Do not default to the heavyweight route out of habit; it is more expensive
   and was not needed once the diagnose tool showed the lighter one sufficed.
3. Style-gate thresholds (ink range, gray/soft-ink cutoff, stroke width
   cutoff, black-fill cutoff, thick-ink cutoff) encode the source's drawing
   style and must be treated as a per-source fit, not a fixed default. See
   "Alignment Gate vs Style Gate" above for why these are kept structurally
   separate from alignment thresholds, which are fixed across every source.
   `min-tile-score` is a per-source yield/ranking control, not a hard gate; it
   sorts already-gate-passed tiles by a score that mixes both alignment and
   style signals, it does not decide alignment pass/fail on its own. Example:
   housei's yield bottleneck is `soft_ink_ratio` (pencil-heavy rough style),
   not the score cutoff; a different source could bottleneck on black-fill or
   width instead. Diagnose with
   `diagnose_gate_funnel.py` before assuming a gate needs loosening.

### fighting Is A Quality-Bar Reference, Not An Extraction Template

`fighting` (see `doc/dataset_status.md`) arrived as already-cropped, already
line-anchored 480x480 rough/line pairs — someone else had already solved
panel identification and alignment before upload. Its near-ideal result (no
gray fringe, no black fill, tight correspondence) is therefore evidence of
what a fully-solved pair looks like, useful as the target quality profile to
judge other sources against. It is not evidence that any particular extraction
mechanism is easy or universal, because fighting never went through region
proposal, matching, or alignment at all. Do not try to reuse "how fighting was
extracted" as a procedure for a raw multi-page manuscript; there is no
extraction procedure to reuse.

## Rule

Do not create new raw-manuscript training pairs by XY-coordinate equality alone.

This applies even when:

- rough and line pages have the same pixel dimensions
- rough and line files are paired in a manifest
- a same-XY crop has acceptable local edge metrics

Same-XY crop coordinates may only be used after a content / region matching step
has selected the rough and line regions as corresponding visual content.

Do not use fixed-size regular grid crops as the primary extraction method.
Simple equal-size rectangular tiling is only a baseline / diagnostic path,
because successful matches from identical XY coordinates and identical crop
sizes are expected to be rare in manuscript datasets.

## Preferred Unit

Prefer these units over fixed page-grid tiles:

- panel / koma region
- character region
- large body / face / hand region
- other semantically coherent line regions

The line image should normally be the anchor for region proposals. The rough
image should be searched for the corresponding region with translation and scale
candidates.

## Normalization

Avoid treating 480x480 as the source extraction constraint.

This applies to hamlabi and to future pair expansion for other datasets. Raw
pair creation should happen at variable aspect ratio first; 480px should be a
training/materialization policy, not the rule that defines the source pair.

For region datasets:

- preserve aspect ratio
- normalize by long side, such as 768 or 1024 px
- save a manifest with size and source bbox metadata
- only derive 480x480 tensors later, inside training or materialization code

## Source Scale Band

Long-side normalization is not scale-neutral. A region whose source long side
is 300 px and one whose source long side is 7,000 px both become 768 px, so the
same 480 px tile can represent a 6x upscaled fragment or a 9x downscaled page.
Stroke width, gray fringe, and content scale then vary by more than an order of
magnitude inside one dataset.

Track the ratio explicitly:

```text
src_per_out = source_region_long_side / normalized_long_side
```

Observed effect on ako5ver2 keep281 (see
`doc/raw_dataset_extraction_knowledge.md`):

- `src_per_out` far below 1: blurred upscaled rough, near-empty line target
- `src_per_out` near 1.2 to 3.5: character / body-part scale, crisp strokes
- `src_per_out` far above 3.5: multi-panel page composition, gray mush strokes,
  heavy black fill

Rules:

- record `src_per_out` in region manifests and tile CSVs
- filter tiles by a scale band, not by a single upper cap
- do not rank tiles by edge correspondence alone across mixed scales, because
  downscaling inflates edge overlap on both sides
- prefer re-materializing an oversized region near its native scale over
  downscaling it into one normalized image

## Tile Score Is Not A Content-Match Guarantee

Even within a correct source-scale band, a tile can score well on geometric
edge correspondence while showing unrelated rough and line content, because
local edge density alone cannot distinguish real stroke correspondence from
coincidental overlap between busy hatching and an unrelated line fragment.

Observed on ako5ver2 native strict tiles: a tile scored inside the accepted
range on every individual gate (edge F1, recall, precision, ink, width) while
the rough was unrelated scratch marks and the line was an unrelated stitch/scar
symbol. It ranked in the bottom decile of `tile_score`, so a rank or score-based
cutoff removes this class of case in practice, but the underlying gates alone
did not catch it.

Rules:

- inspect the tail of any ranked candidate set at full tile resolution, not
  only shrunken thumbnails, before deciding a score cutoff
- treat a smooth score distribution with no natural cliff as normal; pick the
  cutoff from where visual quality degrades, not from a statistical elbow
- keep the review sheet at native tile size for this check, since thumbnails
  hide exactly the gray fringe and content mismatch that distinguish a real
  match from a coincidental one

## Review Gate

New raw datasets should initially produce:

- candidate CSV
- candidate JSON manifest
- rough / line / overlay QC montage

They should not directly write a production train list until review decisions
are recorded.

Black-fill-heavy regions are not automatically rejected. If the rough/line
content matches, keep them with a `black_fill` label so MoE/router work can
later specialize an expert for dense black areas.

Large panel or multi-character regions are also not automatically rejected.
They should be kept as parent candidates and may be recursively searched for
smaller child regions.

## Matching Process

The matching process remains an active research target.

Current preferred direction:

- use the finished line image as the anchor for region proposals
- generate semantically coherent line regions first
- search the rough page for matching content using translation, scale, and
  aspect-preserving normalization candidates
- score candidates by multiple signals, not XY overlap alone
- keep parent regions when a precise child match is not yet separable
- recursively split oversized parent regions into child candidates when review
  shows only part of the parent matches

The matcher should aim to find the rough content corresponding to the selected
line region, not to assume that rough and line pages share layout coordinates.

## Post Alignment

Visual content correspondence is not sufficient if the normalized rough and
line crops are visibly shifted.

Before training, region manifests should run a post-alignment pass that searches
for small residual translation after variable-aspect normalization.

Guidelines:

- correct small residual shifts before materialization or training
- keep alignment metadata such as `align_dx`, `align_dy`, and score gain
- prefer conservative shift limits first, such as 12 px at 768 px
- treat large shifts that hit the search boundary as review signals, not as
  automatically trusted corrections
- if the best correction differs across subregions, keep the region as a parent
  and search for child regions instead of forcing one global translation

Current implementation:

- `tools/pair_extraction/post_align_region_manifest.py`

## VLM Usage

Use VLMs as a second-pass review / rerank / reject gate, not as the primary
extractor.

The deterministic matcher must record:

- line bbox
- rough bbox
- translation / scale candidates
- match scores
- source page metadata

Then a VLM may judge:

- whether rough and line show the same character / pose / panel region
- whether the pair is usable for supervised learning
- coarse region type, such as face, hand, torso, full body, or panel fragment
- reject reasons, such as wrong content, margin, text/noise, too partial, or
  black-fill dominance

VLM review should run with deterministic settings such as temperature 0 and
write its decisions to a CSV. Human review remains the authority before a new
raw dataset is promoted to a training list.

## hamlabi Status

The first `hamlabi_*.jpg` same-coordinate extraction is not a production
dataset. It is retained only as a failed strict baseline / QC reference unless
explicitly removed later.

The current branch work uses `tools/pair_extraction/match_hamlabi_regions.py`
to generate review candidates without relying on same-XY extraction.

After deterministic candidates are generated,
`tools/pair_extraction/vlm_review_hamlabi_regions.py` can create VLM review
panels or ask a local Ollama vision model for second-pass decisions.
