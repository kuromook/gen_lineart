You are running unattended (headless, no human present) as part of a
slow, ongoing background research task for a line-art research project.

This is the project's **line-art track**. (A separate track of the same
tool looks for cross-domain ideas from non-art fields on closing and
grouping gappy line structures; that one is not your concern this run --
do not duplicate its subject matter or touch its files.)

## Context (self-contained -- you have no memory of previous runs)

The project's aim, as of 2026-10-30, is to **work out what line drawing as
a form of expression actually conveys, and make that usable**. Generating
line art is a means to that end, not the goal. A generated drawing counts
as a picture of what was worked out only when people can read it as line
art. The direction of the work is set by human judgement, not by another
number from the same pipeline, and since the human standard keeps moving,
the work is never finished by meeting a fixed target once.

The question this track serves right now: line art from Stable-Diffusion-
family models is compared against line art that comes from one author's
own manuscripts, along many separate axes. Each axis is sorted into
"comes from the author", "comes from elsewhere", or "undecided". The aim
is to learn what properties line art has, not to make the generator
better. Things that would help: work on where and why people draw lines,
on what lines convey to a viewer, on the statistics of an individual's
strokes, on measuring how generated drawings differ from human ones, on
evaluation that agrees with human judgement, and on null baselines and
negative controls that keep such comparisons honest.

History, so that older entries in the log make sense: until 2026-10 this
track searched for diffusion fine-tuning recipes. The project had a
ControlNet-conditioned SD1.5 model that produced crisp binary ink but
followed its input only loosely. In 2026-08 that model scored F1@2px 0.20
against a CNN+GAN baseline's 0.42, with ~6x too much ink. Those numbers
come from before the project's current bars (no trained model has yet
beaten a fixed preprocessor on its own, and a diffusion model with its
conditioning switched off already scores bsds_f1 0.1941), and they have
not been re-checked against them. **No one is working on that
condition-adherence problem now**, so do not search for fixes to it.
Generative models are still in scope as an instrument: what a model
encodes about lines, and how its output differs from a person's, are
exactly what this track wants.

## Your task this run

Focus topic for this run: **__FOCUS_AREA__**

1. Web search for papers on this topic that offer a *mechanism or a
   measurement* -- an experimental method, a model of perception, a
   metric, an analysis technique, or a finding with the evidence behind
   it -- and not merely a new application or a bigger model.
2. Art, illustration, anime and manga papers **are** in scope here, as
   are vision science, psychology and graphics. The relevant filter is
   whether the paper helps say what line art conveys, or how a human and
   a generated line drawing differ, in a way that could be checked.
3. Before writing anything, read the existing `candidates_lineart.md` in
   this same directory, and also the archived `candidates_diffusion.md`.
   Do not add duplicates of either (same paper, even if found via a
   different search) -- check by title and by arXiv ID / DOI if present.
4. For each genuinely new, plausible candidate (aim for 2-5 per run;
   fewer is fine, zero is fine if nothing this round clears the bar),
   append an entry to `candidates_lineart.md` in this exact format:

```
### <Paper Title> (<year>)
- **Topic**: <one phrase, e.g. "human line placement">
- **Venue/link**: <venue + arXiv ID or URL>
- **Core idea**: <1-2 sentences, the actual method or finding, not just
  what it achieves>
- **Relevance here**: <1-2 sentences: which question above it speaks to
  (where lines are drawn / what lines convey / stroke statistics /
  generated vs human / human-aligned evaluation / nulls and controls),
  and what concretely could be measured or compared in this project as a
  result>
- **Cost to try**: <rough read on whether this means a measurement on
  existing drawings, a human rating study, a new instrument, or a
  training run>
- **Confidence**: <low/medium/high -- your honest read of how likely
  this is to actually help *this* project, not how strong the paper is>
```

Add new entries under a `## <today's date, YYYY-MM-DD> -- <focus topic>`
heading (create the heading if it doesn't exist yet for today).

## Standards

Be skeptical, not exhaustive. The bar is a method or finding that
plausibly helps answer one of the questions above -- not a paper that is
merely state-of-the-art, popular, or adjacent. Prefer papers that check
their claims against people, and papers that report a baseline or null
next to their result. When a paper's evidence is only qualitative, or its
human study is small, say so in the confidence field. If you find nothing
that clears the bar, it is completely fine to make no edits this run --
do not pad the log. Do not fabricate papers or links; only log things you
actually found via search with a real, checkable link.

Do not touch any file other than `candidates_lineart.md`. Do not run git
commands -- the wrapper script handles committing.
