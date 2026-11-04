You are running unattended (headless, no human present) as part of a
slow, ongoing background research task for a line-art research project.

This is the project's **topology track**. (A separate track of the same
tool searches for work on what line drawings convey and how they are
judged; that one is not your concern this run -- do not log those papers
here, and do not touch its files.)

## Context (self-contained -- you have no memory of previous runs)

The project's aim, as of 2026-10-30, is to **work out what line drawing as
a form of expression actually conveys, and make that usable**. Generating
line art is a means to that end, not the goal, and the direction of the
work is set by human judgement, not by another number from the same
pipeline. (The project started as rough-sketch-to-line-art translation;
that framing is out of date.)

The question this track serves right now comes from an analysis of one
author's manuscripts. The ground truth is the author's own *regions*,
taken from layer masks (the visible part of a layer's mask minus the
masks of the layers above it). The question is: **can the group of lines
that bounds such a region be found from the line art?** The difficulty is
that the lines have gaps. In the author's words, the regions are "not
closed, there are gaps". So filling the areas that the lines happen to
enclose recovers almost none of the author's regions. Whatever finds the
bounding group has to tolerate gaps, and has to decide which lines belong
to which region.

Background on why this track looks outside art: in 2026-08, **clDice**
(Shit et al., "clDice -- a Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation", CVPR 2021), a loss/metric from *medical
vessel segmentation*, was carried over to this project's stroke-continuity
problem and scored better on the stroke-continuity metrics the project
used at the time. Treat that as a 2026-08 result: it predates the
project's current bars (no trained model has yet beaten a fixed
preprocessor on its own, and a diffusion model with its conditioning
switched off already scores bsds_f1 0.1941, which is the floor any F1
must be read against), and it has not been re-checked against them. What
carries over is the method of looking, not a proven win: fields that care
about thin, broken, networked structures, or about closing regions from
incomplete boundaries, for reasons unrelated to art, sometimes have a
mechanism that transfers.

## Your task this run

Focus field for this run: **__FOCUS_AREA__**

1. Web search for papers in this field that:
   - deal with recovering regions, closed contours, or grouped structure
     from boundaries or lines that are incomplete, broken, or noisy, or
     with the topology (nodes, branches, connections, loops) of line
     networks
   - propose a specific mechanism -- an algorithm, a model of perceptual
     grouping, a loss, or an evaluation metric -- for closing gaps,
     grouping fragments, assigning a boundary to the region it belongs
     to, or measuring any of these
   - are NOT from an art/illustration/anime/manga context (that's the
     point -- looking outside the genre). Vision science and general
     computer vision are in scope.
2. Before writing anything, read the existing `candidates.md` in this
   same directory to see what's already been logged. Do not add
   duplicates (same paper, even if found via a different search) -- check
   by title and by arXiv ID / DOI if present.
3. For each genuinely new, plausible candidate (aim for 2-5 per run;
   fewer is fine, zero is fine if nothing this round clears the bar),
   append an entry to `candidates.md` in this exact format:

```
### <Paper Title> (<year>)
- **Field**: <one phrase, e.g. "floor-plan room segmentation">
- **Venue/link**: <venue + arXiv ID or URL>
- **Core idea**: <1-2 sentences, the actual mechanism, not just "it's
  about lines">
- **Transfer hypothesis**: <1-2 sentences: specifically what in this
  project's question -- finding the gappy group of lines that bounds an
  author's region -- this could apply to, and why you think the mechanism
  would carry over; say whether it would be a method, a measurement, or
  a null/baseline>
- **Confidence**: <low/medium/high -- your honest read of how likely
  this is to actually transfer, not how interesting the paper sounds>
```

Add new entries under a `## <today's date, YYYY-MM-DD> -- <focus field>`
heading (create the heading if it doesn't exist yet for today).

## Standards

Be skeptical, not exhaustive. A paper about "lines", "curves" or
"segmentation" in an unrelated field is not automatically relevant -- the
bar is a *specific, articulable mechanism* that plausibly transfers to
gap-tolerant region grouping. Prefer papers that say how their result was
checked against people or against a baseline, and note it when a method
has no such check. If you don't find anything that clears this bar, it is
completely fine to make no edits this run -- do not pad the log with weak
matches to have something to show. Do not fabricate papers or links; only
log things you actually found via search with a real, checkable link.

Do not touch any file other than `candidates.md`. Do not run git commands
-- the wrapper script handles committing.
