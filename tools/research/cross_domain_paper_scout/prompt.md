You are running unattended (headless, no human present) as part of a
slow, ongoing background research task for a rough-sketch-to-line-art
image translation project.

## Context (self-contained -- you have no memory of previous runs)

The project trains models to convert pencil rough sketches into clean
line art. A recurring failure mode is "stroke continuity": models
produce fragmented, broken strokes instead of long continuous lines, and
generic pixel-overlap loss functions/evaluation metrics don't directly
target this.

One fix that worked well: **clDice** (Shit et al., "clDice -- a Novel
Topology-Preserving Loss Function for Tubular Structure Segmentation",
CVPR 2021), a loss/metric originally designed for *medical vessel
segmentation* (retinal vessels, blood vessels in CT/MRI) -- nothing to do
with art or illustration. Its core idea (compare predicted and
ground-truth *skeletons* against each other's full masks, not just raw
pixel overlap) transferred directly and produced a measurable, genuine
improvement in this project's stroke-continuity metrics.

This established a hypothesis: there may be more mathematically
transferable ideas -- loss functions, evaluation metrics, or algorithms
-- sitting in fields that study thin, continuous, topologically important
structures for reasons that have nothing to do with art (medical imaging,
remote sensing, materials inspection, geophysics, connectomics, etc.),
which could transfer the same way clDice did.

## Your task this run

Focus field for this run: **__FOCUS_AREA__**

1. Web search for papers in this field that:
   - work with thin, curvilinear, branching, or networked structures
     (vessels, roads, cracks, wires, neurons, fault lines, root systems,
     etc.)
   - propose a loss function, evaluation metric, or algorithm
     specifically about *topology*, *connectivity*, or *continuity* of
     those structures (not just generic segmentation accuracy)
   - are NOT from an art/illustration/anime/manga context (that's the
     point -- looking outside the genre)
2. Before writing anything, read the existing `candidates.md` in this
   same directory to see what's already been logged. Do not add
   duplicates (same paper, even if found via a different search) -- check
   by title and by arXiv ID / DOI if present.
3. For each genuinely new, plausible candidate (aim for 2-5 per run;
   fewer is fine, zero is fine if nothing this round clears the bar),
   append an entry to `candidates.md` in this exact format:

```
### <Paper Title> (<year>)
- **Field**: <one phrase, e.g. "retinal vessel segmentation">
- **Venue/link**: <venue + arXiv ID or URL>
- **Core idea**: <1-2 sentences, the actual mechanism, not just "it's
  about lines">
- **Transfer hypothesis**: <1-2 sentences: specifically what in this
  line-art stroke-continuity problem this could apply to, and why you
  think the mechanism would carry over -- not just "could be useful for
  lines">
- **Confidence**: <low/medium/high -- your honest read of how likely
  this is to actually transfer, not how interesting the paper sounds>
```

Add new entries under a `## <today's date, YYYY-MM-DD> -- <focus field>`
heading (create the heading if it doesn't exist yet for today).

## Standards

Be skeptical, not exhaustive. A paper about "lines" or "curves" in an
unrelated field is not automatically relevant -- the bar is a *specific,
articulable mechanism* that plausibly transfers, the way clDice's
skeleton-vs-mask comparison did. If you don't find anything that clears
this bar, it is completely fine to make no edits this run -- do not pad
the log with weak matches to have something to show. Do not fabricate
papers or links; only log things you actually found via search with a
real, checkable link.

Do not touch any file other than `candidates.md`. Do not run git commands
-- the wrapper script handles committing.
