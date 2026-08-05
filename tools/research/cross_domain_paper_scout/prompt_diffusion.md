You are running unattended (headless, no human present) as part of a
slow, ongoing background research task for a rough-sketch-to-line-art
image translation project.

This is the project's **diffusion track**. (A separate track of the same
tool looks for cross-domain topology/continuity ideas from non-art
fields; that one is not your concern this run -- do not duplicate its
subject matter or touch its files.)

## Context (self-contained -- you have no memory of previous runs)

The project converts pencil rough sketches into clean line art. The
long-running approach is deterministic CNN/U-Net translation, which is
being continued quietly. The active exploration is **diffusion**, and it
is currently stuck in a specific, well-characterized way:

- Setup: frozen SD1.5-family UNet (an anime-style checkpoint) plus a
  trained ControlNet adapter (~361M params) conditioned on the rough
  tile; standard latent-diffusion noise-prediction training,
  `L = MSE(eps_hat, eps)`. Roughly 1,489 paired training tiles -- a
  small dataset by diffusion standards.
- What worked: this was the first time the project broke through the
  "soft/marbled" ceiling that every CNN+GAN variant hit. In only 10
  epochs it produced crisp, confident, fully binary anime-style ink.
- What is broken: **the output only loosely corresponds to the input
  rough**. It confidently generates plausible *different* content
  (different expressions/poses on some tiles). So it looks good but
  scores worse than the CNN+GAN baseline: F1@2px 0.20 vs 0.42, and
  ink_ratio ~6x (massively over-inked).
- What has already been ruled out: sweeping conditioning-scale and
  guidance-scale at inference did not fix it. The "ControlNet sudden
  convergence / undertraining" hypothesis was tested with a 10x longer
  run (1,860 -> 18,600 steps) and **rejected** -- F1 moved only
  0.202 -> 0.216 and ink_ratio 6.36 -> 5.93. More training steps is a
  dead end.
- Remaining suspected variables: dataset size/quality, the
  *conditioning mechanism* itself, and per-tile captions (currently a
  single fixed caption for all tiles; WD14 auto-tags exist but are not
  yet used in training).
- A parallel line of work has stepped back from the translation task to
  study unconditional/domain-only generation quality of each domain
  (rough, line) on its own, including domain-only LoRA training, partly
  because unpaired rough material is plentiful while paired data is not.

For contrast: the deterministic single-stage U-Net has the opposite
failure mode -- it clearly tries to follow the rough's actual strokes but
comes out wobbly and fragmented. So the project's two paradigms fail on
opposite axes: **fidelity to the input condition** vs **stroke
confidence/continuity**.

## Your task this run

Focus topic for this run: **__FOCUS_AREA__**

1. Web search for papers on this topic that propose a *mechanism* --
   architecture, conditioning scheme, training objective, or evaluation
   metric -- and not merely a new application or a bigger model.
2. Unlike the topology track, art/anime/illustration papers **are** in
   scope here; the relevant filter is not the domain but whether the
   mechanism addresses one of the failure modes above.
3. Before writing anything, read the existing `candidates_diffusion.md`
   in this same directory to see what has already been logged. Do not add
   duplicates (same paper, even if found via a different search) -- check
   by title and by arXiv ID / DOI if present.
4. For each genuinely new, plausible candidate (aim for 2-5 per run;
   fewer is fine, zero is fine if nothing this round clears the bar),
   append an entry to `candidates_diffusion.md` in this exact format:

```
### <Paper Title> (<year>)
- **Topic**: <one phrase, e.g. "control-signal conditioning">
- **Venue/link**: <venue + arXiv ID or URL>
- **Core idea**: <1-2 sentences, the actual mechanism, not just what it
  achieves>
- **Relevance here**: <1-2 sentences: which of the failure modes above
  it speaks to (condition adherence / over-ink / small paired dataset /
  caption conditioning / two-stage structure), and what concretely could
  be tried in this project as a result>
- **Cost to try**: <rough read on whether this means a new training run,
  an inference-time change, a data change, or an architecture change>
- **Confidence**: <low/medium/high -- your honest read of how likely
  this is to actually help *this* setup, not how strong the paper is>
```

Add new entries under a `## <today's date, YYYY-MM-DD> -- <focus topic>`
heading (create the heading if it doesn't exist yet for today).

## Standards

Be skeptical, not exhaustive. This project has already burned a 10x
training run on a plausible-sounding hypothesis that turned out to be
wrong, so the bar is a mechanism that plausibly addresses a *specific*
failure mode described above -- not a paper that is merely
state-of-the-art, popular, or adjacent. Prefer papers that diagnose
*why* something fails over papers that only report better numbers. If a
paper's gains come from far more data or compute than this project has
(~1.5k paired tiles, single GPU), say so in the confidence field. If you
find nothing that clears the bar, it is completely fine to make no edits
this run -- do not pad the log. Do not fabricate papers or links; only
log things you actually found via search with a real, checkable link.

Do not touch any file other than `candidates_diffusion.md`. Do not run
git commands -- the wrapper script handles committing.
