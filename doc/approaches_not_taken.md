# Approaches never taken up

Moved out of `doc/CURRENT.md` on 2026-10-24, verbatim: the section that records
what has *not* been tried and why, so that "we must have tried that" is never
assumed and nothing here is re-asked from scratch. Two records: the paper
scout's nine reports triaged against Track K's spec (2026-10-19), and the
ControlNet + IP-Adapter survey (2026-09-30).

`doc/CURRENT.md § Approaches Never Taken Up` keeps the claims; this file keeps
the reasoning, the citations and the numbers behind them.

## Approaches Never Taken Up

### The paper scout's nine reports, triaged against Track K's spec (2026-10-19)

The scout on `main` (renamed `paper-scout` later the same day; see `doc/worktree_policy.md`) has accumulated **9 runs since 2026-08-10 -- 11 topology
papers and 15 diffusion papers -- and nothing on `foundation` had ever cited
it.** Read in full and judged against the one open requirement this project
has: **K's registered spec, a feature that lifts the precision of the author's
`broken` mark among collapsed-bridge-passing tokens from 3.8% past 50%.**

Criteria, fixed before the verdicts: a candidate answers the spec only if it
supplies (1) a **per-token quantity**, since the decision is per token, not an
image-level loss or score; (2) computable from **material this project has**
(ink raster, skeleton, spans; GT line art for training), not from labels the
corpus lacks; (3) discriminating **inside** the exposure, separating the 134
marked tokens from the 4,048 unmarked -- merely detecting that a bridge exists
is the exposure itself, at precision 3.2%.

**Verdict: one candidate of the right type, none that can be said to reach the
bar. The spec is not answered.**

- **The one**: *Individual fibre separation in 3D fibrous materials imaged by
  X-ray tomography* (Depriester et al. 2022, J. Microscopy, DOI
  10.1111/jmi.13096). It splits a binary mask of touching and crossing fibres
  using **only the local orientation field**: compute the misorientation angle
  between neighbouring local orientations and cut where it exceeds one
  threshold. That is a per-location split/no-split decision keyed on a quantity
  computable from the ink alone, taken **inside** the merged-object set -- K's
  exact position. It also lands on the axis this project already measured as
  the one that works at crossings: **tangent agreement and lateral offset, not
  curvature**. So it is not a new hope; it is a published, single-parameter
  instantiation of an axis the stroke-curve work already established.
  Why it still cannot be said to clear the bar: nothing in it bears on the
  **rate** in this material, and its own named failure mode is that **a sharp
  corner looks like a contact** -- in hand-drawn line art a deliberate sharp
  bend is common, and that is precisely the false-positive population that
  holds precision down. The scout rated it **low**, mainly for being a
  heuristic post-process rather than a trainable loss and for being 3D; **for K
  that downgrade does not apply**, because K needs a feature, not a loss.
- **A partial contributor**: Batra et al. 2019's auxiliary **orientation head**
  (per-pixel tangent angle, supervision derivable free from GT line art) is the
  learned way to produce the field the fibre method thresholds, and would be
  more robust than tangents computed on a 1px skeleton near a junction -- which
  is where K has measured the skeleton to be least trustworthy. It decides
  nothing by itself.
- **Three families that answer a different question.** (a) *Losses that stop a
  model breaking a line*: region-separation (Oner 2020), CAPE, MALIS/TOPO-Loss,
  TopoM-CrackNet, Betti matching, Topograph, Beyond-pixel-wise. These address
  **under-connection** -- K's other direction, the one with no author mark --
  and none is a per-token feature. (b) *Metrics*: APLS, Cracks-Per-Patch, and
  SAMSEM's **ESD** (per-instance open/short rate), which is the closest thing
  to the measurement this project wants and **shares K's own obstacle**: it
  presumes a per-instance decomposition, and crossings are exactly where that
  is ambiguous. (c) *The entire diffusion file*, 15 papers on condition
  adherence, over-inking, attention control and the VAE bottleneck: **none of
  them is about splitting a merged token.** Zero bear on the spec.

**The tenth report arrived the same day** (2026-10-19, diffusion track:
"control-signal representation -- how sparse, noisy, or weak conditions are
encoded"), hours after the triage above, and is the strongest of the ten for
*this project* while still **not touching K's spec at all** -- it is entirely
input-side diffusion. Two things in it are worth a decision rather than a read:

- **One free measurement that has never been done here.** From the inverted
  diagnosis of *Block and Detail* (arXiv:2402.18116): measure, over the
  training pairs, the **per-pixel correspondence between dilated rough ink and
  line ink** -- precision and recall of rough-ink as a predictor of line-ink.
  High recall with low precision would say the condition is *intrinsically
  ambiguous* (the inker discards much of the rough), which would implicate the
  dataset rather than the architecture. Evaluation-only, no GPU.
  **Checked 2026-10-19, and half of it already exists** -- read, not
  re-measured, from Track D's
  `../lineart-pair-signal/results/pair_alignment_strata_20260915/per_pair.csv`
  (8,467 pairs). Its `rough_le3` / `rough_3to8` / `rough_gt8` columns are the
  **recall side**: the share of GT stroke length whose nearest rough ink lies
  within 3px, 3-8px, and beyond. Medians **by pool** (never pooled):

  | pool | n | <=3px | 3-8px | >8px | `aligned` |
  |---|---:|---:|---:|---:|---:|
  | clippairskoma | 6,978 | **0.3537** | 0.3795 | 0.1976 | 0.3014 |
  | ako5ver2koma | 536 | **0.2005** | 0.4366 | 0.3152 | 0.1742 |
  | fitnesskoma | 455 | **0.2007** | 0.4301 | 0.2940 | 0.1873 |
  | gakuenkoma | 202 | **0.2417** | 0.4373 | 0.2392 | 0.2198 |
  | hamlabikoma | 164 | **0.2026** | 0.4734 | 0.2631 | 0.1766 |
  | houseikoma | 132 | **0.2450** | 0.4020 | 0.2283 | 0.2176 |

  Chance (the conditioning map rotated 180 degrees) is 0.000-0.012, so the
  signal is real, and **the shape is not the one the paper predicted**: the
  expected case was high recall with low precision, a messy superset the inker
  thins. Instead **recall at 3px is 0.20-0.35, and 20-32% of GT stroke length
  has no rough ink within 8px at all.** The rough has ink *near* most GT
  strokes (the `<=8px` share totals 0.63-0.73) but does not place it. This is
  the **rough**, not the conditioning map, so it is not the 0.604
  oracle-recall cap already on record -- it is a harder number about the task
  itself, and it bounds what any rough-conditioned model can be asked to do,
  the words line included.
  **What is missing is the precision side, and the unit is not the pixel** --
  corrected here before the measurement was commissioned, because the two
  directions have to be the same construction to be comparable. D's script
  cuts the **GT skeleton into segments** and takes each segment's **median**
  distance to the source skeleton, classifying the whole segment into
  <=3 / 3-8 / >8 and weighting by segment length; both sides are skeletons, the
  rough's at `dark < 200`. So the precision side is that mirrored: the rough's
  skeleton cut into segments, each segment's median distance to the GT
  skeleton, weighted by the rough's segment length, with the chance arm the GT
  skeleton rotated 180 degrees. The script in D's tree already
  computes distance transforms, so this is one direction added to code that
  exists rather than a new measurement -- which is also why **the owner is
  Track D**, closed since 2026-09-17 and holding exactly this machinery.
  **Reopened for this one measurement by user approval 2026-10-19**;
  registration in `../lineart-pair-signal/doc/track_d_reopen_precision_20261019.md`
  (commit `6193385` in that tree, unpushed -- D pushes its own branch).
  **The band is registered before the number exists, and it is deliberately
  not a pass mark**: nothing is being accepted or rejected, so what is frozen
  is a reference. The reference is derived rather than picked -- if the <=3px
  correspondence is roughly symmetric then recall ~ M/L_gt and precision ~
  M/L_rough for the same matched length M, so

  > **predicted precision = recall x (L_gt / L_rough)**

  with `L_gt` the median `gt_len` and `L_rough` the median `rough_density` x
  230,400. Frozen today: **0.139 / 0.084 / 0.113 / 0.172 / 0.088 / 0.143** for
  clippairskoma / ako5ver2koma / fitnesskoma / gakuenkoma / hamlabikoma /
  houseikoma, from `L_gt/L_rough` of 0.394 / 0.421 / 0.563 / 0.712 / 0.433 /
  0.586. **The rough's skeleton is 1.4-2.5x longer than the GT's**, so a low
  precision is explainable by quantity alone and the reading turns on measured
  against predicted: at the prediction, the rough's extra length explains
  everything and it carries no placement information beyond what its length
  buys; far above, the rough's ink concentrates where the line goes; far below,
  it is worse than its own length allows. **No abstention condition**, and the
  reason is on the record: this is a census of all 8,467 pairs, not a sample,
  so the reachability computation that every bar otherwise owes does not
  apply.
  **Done the same day**: Track D was reopened for it, measured it, and closed
  again -- see its ledger entry above. The shape is that **the pair is loose in
  both directions**, with about half the rough's ink having no GT line within
  8px, bounded by Track D's own caveat that a `gray < 200` threshold
  skeletonises pencil grain.
- **One entry that would reopen a closed line.** CtrLoRA (ICLR 2025,
  arXiv:2410.09400) is the only paper in the whole log that treats the
  **condition encoder** as the suspect -- ControlNet's randomly-initialised
  4-layer conv stack "cannot extract any useful information from the condition
  image at the beginning of training" -- and it re-frames the hypothesis this
  project already spent a 10x-steps run rejecting: not undertrained, but
  badly *started*, which more steps cannot fix. Its budget fits (~37M LoRA
  params against 361M, ~1,000 pairs, released SD1.5 Base ControlNet). But the
  ControlNet lineage is **closed** by Tracks A and B, so this is a request to
  reopen it, and the ledger's bar for reopening a held or closed track is a new
  yardstick, not a new paper. Logged, not promoted.

**The reusable reading, and it is about the instrument**: a 13.3x precision
lift is **information**, and a loss function is not information. The literature
K needs answers *"what distinguishes a junction from a continuation"*, and the
scout's topology focus areas are pointed at *"how to train for connectivity"*.
That is why its ratings invert against K's need -- the only entry of the right
type is rated lowest, and the five highest-rated entries are losses K cannot
use. **The mis-rating is systematic, not random**, so the fix is a focus-area
line, not a re-read. Two diffusion entries looked worth their own decision for reasons unrelated to
K, and **one of them was wrong on the facts -- the foundation's error, found
2026-10-19 while writing back Track D's result.** The scout calls the frozen
VAE's round-trip on GT line tiles an unmeasured ceiling, and this file repeated
it; **Track D measured it and refuted it** long before, in its own ledger entry
above: GT round-trips at f1 **0.9605** (SD1.5) and **0.9806** (SDXL) on
lineart_family, against a best trained model of 0.25. So the VAE is not the
ceiling, and the CtrLoRA caveat built on "run the round-trip test first" is
void -- the test is on record. **This is the logged trap about citing the file
when stating a fact about a track's material, walked into a second time**, and
the file it should have cited was four hundred lines up in this document.
The other entry stands as logged: the flat `MSE` spends ~90% of its gradient on
blank paper. Neither is promoted here.

Recorded so they are not re-asked from scratch, and so that "we must have tried
that" is not assumed.

**ControlNet + IP-Adapter (image-prompt / image-embedding conditioning) has
never been tried, and was never even proposed.** Surveyed 2026-09-30 across
every worktree's `doc/`, `inbox/`, `outbox/`, `experiments/`, `scripts/`,
`tools/`, all branches' commit messages, and the Directions 1-9 survey in
`doc/model_directions.md`: zero occurrences, of the name or of the concept
(`image_encoder`, `CLIPVision`, T2I-Adapter, reference-only, unCLIP-style image
conditioning, exemplar conditioning). Direction 3's "learned line-art encoder if
available later" is a perceptual-loss feature extractor, not conditioning, and
Track E's "CLIP features + linear" feeds a judge, not a generator.

**It fell off by expiry, not by judgement.** The one sentence in the corpus that
would have covered it is `doc/work_log.md` 2026-08-08, Next Action 2: after
finding that stacking the unconditional line-domain LoRA on the ControlNet was
actively harmful, it says to *"revisit whether any form of style adapter is safe
to add only after the paired fine-tune itself is working."* The paired fine-tune
never started working -- Tracks A and B both closed below the preprocessor
baseline -- so the revisit never triggered.

**Tooling was never the constraint**: diffusers 0.39.0 with `IPAdapterMixin`,
and both `StableDiffusionControlNetPipeline.load_ip_adapter` and
`set_ip_adapter_scale` are available in the venv today, no new dependencies.

**Assessment (2026-09-30), against what is now measured.** It does not address
the failure in lesson 8, because that failure is *spatial*: where the
conditioning map lacks a GT stroke the model draws it 9.3% of the time, and
output skeletons sit near the map (36%) rather than near GT alone (9%). An
IP-Adapter embedding is pooled and spatially unlocalised -- it carries
appearance, not where a stroke goes -- so against this project's own
decomposition it addresses neither half: not which of the preprocessor's
1.4x-GT strokes to delete, and not the ~16% of GT stroke length the map never
finds that caps oracle recall at 0.604. The closest analogue this project has
actually measured points the wrong way: the 2026-08-08 result above, where a
strong global style prior **overrode** the ControlNet's conditioning.
Architecturally IP-Adapter sits nearer that failure than a fix -- it injects at
cross-attention (`attn2`), which is precisely the layer
`scripts/pnp_line_from_rough.py` deliberately left alone, its docstring naming
`attn2` as carrying the signal it wanted to keep steering with.

**Where it would fit is a different axis.** The one thing training demonstrably
moves is tone (near_white 0.13 -> 0.83; fixed-`t` loss correlates with paper
white at r -0.98 and not with f1), and a reference-image channel is the right
instrument for tone. That makes it relevant to Track E's question -- aim at the
base model's own look rather than at GT -- and most natural of all in a
line-art-to-line-art framing, where spatial content comes from the source and a
clean exemplar supplies style. **Note that no line-to-line theme exists in this
project**: `line2line` and equivalents return nothing across `doc/`, and
`lineart-cleanup-refiner` is the nearest thing and has no reference
conditioning either.

**If it is tested, the honest first move is inference-only**: public ControlNet
at its best config (`lineart_coarse`, cs2.5) plus IP-Adapter with a line-art
tile as image prompt, scored on the usual axes *plus* the conditioning map's own
baseline (lesson 6) and Track D's `gt_only` / `vs_condition` columns. Using GT
as the reference leaks, so like the delete oracle it bounds the mechanism rather
than demonstrating an achievable score -- which is this project's established
way of starting.
