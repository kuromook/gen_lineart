# Line-Art Track -- Candidates Log

Accumulating log of papers that might help work out what line art conveys
and how human and generated line drawings differ: where people draw lines,
what lines convey to a viewer, stroke statistics and style, generated vs
human comparisons, evaluation that agrees with human judgement, and null
baselines and negative controls. See `README.md` for how this file gets
populated and how to review it, and `prompt_lineart.md` for the full
context handed to each run.

This track replaced the diffusion track on 2026-11-04. That track's
entries (2026-08-10 to 2026-11-02) stay in `candidates_diffusion.md`,
which is no longer appended to. Topology candidates from non-art fields
live in the separate `candidates.md`.

Entries are grouped by the date/focus-topic of the run that found them.

## 2026-11-16 -- abstraction in sketching: what people keep and what they drop when they simplify a drawing

### Style and Abstraction in Portrait Sketching (2013)
- **Topic**: per-artist stroke statistics across abstraction levels
- **Venue/link**: ACM TOG 32(4) / SIGGRAPH 2013 (Berger, Shamir, Mahler, Carter, Hodgins) -- https://www.ri.cmu.edu/publications/121987 ; PDF (Berger's MSc thesis on the same project): https://runi.ac.il/media/bh1ihat4/style-and-abstraction-2013.pdf
- **Core idea**: Seven artists each drew the same face photos under time limits of 270/90/30/15 s (672 sketches, ~8k strokes per artist). Strokes were labelled shading / complex contour / simple contour, and the authors measured, per artist and per level, stroke-length and curvature distributions, how many strokes of each type were used, and how far the shape deviated from the photo's facial features. The finding is that what gets dropped first, and how lines deviate from the photo, depend on the artist. A perceptual study then checks whether people can match sketches by artist and tell real sketches from synthesized ones.
- **Relevance here**: This is the closest published template for "stroke statistics of one individual", and it adds abstraction as a controlled variable. The same descriptors (stroke-length and curvature histograms, contour vs shading ratio, geometric deviation from the reference) could be computed on the author's manuscripts and on SD output, giving one axis each. The time-limit design also suggests a check: does SD output resemble the author's at some abstraction level, or at none?
- **Cost to try**: The descriptors are a measurement on existing drawings, but they need vectorised strokes (from manuscripts this means stroke extraction, which is non-trivial). Re-running the human style-matching test would be a small rating study.
- **Confidence**: medium-high. It is directly on topic, but there are only 7 artists and one subject (faces), and the stroke statistics are descriptive with no null model.

### SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction (2023)
- **Topic**: human vs model sketch abstraction; recognizability under sparsity
- **Venue/link**: NeurIPS 2023 Datasets & Benchmarks (Mukherjee, Huey, Lu, ... Fan) -- arXiv:2312.03035, https://arxiv.org/abs/2312.03035
- **Core idea**: About 90k sketches of 128 concepts by 5,563 people, drawn under 4/8/16/32 s limits, so sparsity is manipulated experimentally. Human recognition responses, including the full distribution of guesses and not just accuracy, are collected per sketch. Vision models are scored on whether they reproduce human accuracy *and* human uncertainty. CLIPasso sketches are put through the same pipeline and come out human-like only in limited settings.
- **Relevance here**: For human-aligned evaluation and generated-vs-human comparison, it gives a protocol: compare drawings by the *distribution of human interpretations* they elicit, matched at a given sparsity. For nulls, the time-limit ladder is a ready-made way to match ink amount before comparing anything else, which matters here given SD's history of producing ~6x too much ink. Its human sketch-recognition data could also be used to calibrate any model-based "readability" score before trusting it on this project's drawings.
- **Cost to try**: Reusing the dataset/protocol to validate a readability metric is cheap (it is a measurement). Applying it to the author's drawings needs a human recognition study, which is small with a forced-choice design.
- **Confidence**: medium. The method transfers well, but the drawings are object-category doodles by non-artists, far from manuscript line art, so its specific findings may not carry over.

### Pragmatic inference and visual abstraction enable contextual flexibility during visual communication (2019/2020)
- **Topic**: why detail is kept or dropped -- communicative context
- **Venue/link**: Computational Brain & Behavior (Fan, Hawkins, Wu, Goodman) -- arXiv:1903.04448, https://arxiv.org/abs/1903.04448
- **Core idea**: In a sketching reference game (N=192), drawers sketched a target so that a partner could pick it from 4 objects. When the distractors came from other categories, drawers used fewer strokes, less ink and less time and were still recognized accurately; same-category distractors produced more detailed sketches. A model combining a visual-abstraction module (sketch-object correspondence in a CNN feature space) with a pragmatic-inference module fits the data, and lesioned variants without either module fit worse.
- **Relevance here**: This bears on what lines convey and where lines are drawn. On this account, detail is kept to tell the subject apart from plausible alternatives, not to copy the photo. That gives a testable axis: is the author's line detail concentrated where it disambiguates (identity cues, expression, overlapping parts), while SD spreads detail evenly? The lesioned-model comparison is also a good pattern for nulls.
- **Cost to try**: A light version is a measurement (where ink density concentrates relative to discriminative regions, compared author vs SD). A faithful version would need a human reference-game study.
- **Confidence**: low-medium. The mechanism is well evidenced, but the setting (simple objects, a contrastive game) is far from authored illustration, so the "discriminative detail" hypothesis would need to be operationalised from scratch here.

### Learning Deep Sketch Abstraction (2018)
- **Topic**: stroke-level saliency via recognizability-preserving removal
- **Venue/link**: CVPR 2018 (Muhammad, Yang, Song, Xiang, Hospedales) -- arXiv:1804.04804, https://arxiv.org/abs/1804.04804
- **Core idea**: Abstraction is cast as a trade-off between how many strokes are kept and how recognizable the sketch stays. An RL policy removes strokes one at a time, rewarded by the target class's rank under a stroke-order-aware RNN classifier. The order of removal gives a per-stroke saliency estimate.
- **Relevance here**: It could serve as an instrument for "which lines carry the identification". Run greedy or learned stroke ablation on author drawings and on SD drawings, then compare how fast recognizability decays (an ablation curve), with random-order removal as the null. A difference in curve shape would say whether the two put information-bearing strokes in different places.
- **Cost to try**: It needs vector strokes and a recognizer suited to this domain (the paper's is trained on QuickDraw/Sketchy), so in practice a new instrument. The random-removal baseline is cheap once that exists.
- **Confidence**: low. Its notion of saliency is defined by a classifier, not by people, and the paper does not check stroke saliency against human judgements of which strokes matter. Treat it as a method template, not evidence.
