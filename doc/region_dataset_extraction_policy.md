# Region Dataset Extraction Policy

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
