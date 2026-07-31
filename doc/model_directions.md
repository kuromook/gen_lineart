# Model Direction Survey

Updated: 2026-07-19

This note collects candidate directions after the clean BCE baseline,
ResNet-GAN atari generator, and 2ch U-Net refiner experiments.

Current clean-eval rule still applies: do not use leak-era `shape1` metrics as
targets. Use montage inspection as the adoption gate, with F1/chamfer/ink as
reference metrics only.

## Current State

Useful current checkpoints:

- clean BCE baseline:
  `checkpoints/shape1_clean_split_bce/best.pth`
- mild clean BCE variant:
  `checkpoints/shape1_clean_split_bce_milddup800_ft10_lr1e5/best.pth`
- ResNet-GAN atari generator:
  `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`
- conservative 2ch refiner:
  `checkpoints/line_refiner_tight_e2_refiner_unet_bin12_ink14/best.pth`
- aggressive 2ch refiner candidate:
  `checkpoints/line_refiner_tight_e2_refiner_unet_bin20_ink16/best.pth`

Current interpretation:

- ResNet-GAN is useful as an atari/structure generator, not as the final line
  model.
- Plain 2ch `refiner_unet` is more useful than `unet_skip50`.
- Static binary/ink sweeps and staged fine-tune did not escape soft atari-copy.
- `bin12_ink14` is the conservative refiner; `bin20_ink16` is an aggressive
  recall/black-growth expert candidate.
- Router/MoE is relevant later, especially for rough/line correspondence
  groups, but daytime work should now explore line-specific model/loss options.

## Deferred Target/Data Split Direction

Rough/line correspondence is a data property and should become a MoE/router
axis later:

- high rough-line agreement: deterministic/refiner/BCE-like experts
- low rough-line agreement: atari/GAN/hallucination-oriented experts
- agreement score can become both a training split and router feature

Do not prioritize this now. The current high-confidence tile count is likely
small, and raw manuscript extraction should be expanded first when this becomes
the active direction.

## Direction 1: Multi-Scale PatchGAN

Replace or augment the single PatchGAN with multiple discriminators at different
image scales.

Expected benefit:

- local discriminator can enforce line texture and black/white contrast
- coarse discriminator can enforce larger face/hair/body structure
- closer to pix2pixHD-style image translation

Risk:

- more unstable than current short GAN runs
- may increase dark texture if feature matching is not used

Minimum experiment:

- ResNet-GAN or 2ch refiner with two PatchDiscriminators: full scale and
  half scale
- keep epochs short
- compare against `resnet_atari`, `tight_bin12`, and `bin20_ink16`

## Direction 2: Feature Matching Loss

Use discriminator intermediate activations as a stabilizing reconstruction
target, matching real and fake feature maps.

Expected benefit:

- adversarial pressure becomes less binary and less noisy
- may encourage line-art-like structure without only adding dark ink
- natural pair with multi-scale PatchGAN

Risk:

- if discriminator features are poor, matching them is not useful
- implementation needs discriminator feature outputs

Minimum experiment:

- single or multi-scale PatchGAN with feature matching weight
- compare against the same GAN without feature matching

## Direction 3: Line Perceptual / Structure Feature Loss

Use edge/structure features instead of generic VGG perceptual loss.

Possible features:

- Canny/DoG pyramids
- Sobel orientation maps
- learned line-art encoder if available later

Expected benefit:

- pushes line-like structure without requiring exact pixel agreement
- may help preserve long contours

Risk:

- handcrafted edge features can over-reward noisy rough texture
- VGG/LPIPS-style natural image perceptual losses may be poorly aligned with
  manga line art

Minimum experiment:

- add Sobel/DoG pyramid L1 between prediction and GT line
- keep normal BCE/ink losses active

## Direction 4: Diffusion / ControlNet-Like Refinement

Train or adapt a conditional generative model using rough and/or atari as
condition.

Expected benefit:

- strongest hallucination/refinement capacity
- likely better for low rough-line correspondence tiles

Risk:

- high implementation and compute cost
- data volume is likely insufficient right now
- harder to evaluate quickly and consistently

Minimum experiment:

- defer until more raw manuscript pairs exist
- consider only after simpler i2i routes are exhausted

## Direction 5: Shallow Cleanup / Residual Refiner

Use a smaller network that is structurally biased toward cleaning the atari
output instead of regenerating everything.

Possible forms:

- `rough + atari -> correction logits`
- `output = atari + correction`
- `atari -> cleaned line` with rough as weak context

Expected benefit:

- reduces U-Net's tendency to softly repaint the whole tile
- may better behave as a cleanup stage
- low implementation cost

Risk:

- may be unable to add missing structure
- can inherit atari errors too strongly

Minimum experiment:

- small residual CNN with 2ch input
- compare against `refiner_unet`, `tight_bin12`, and `bin20_ink16`

**Result (2026-07-31, koma dataset):** tried both bidirectional
(`cleanup`, adopted as `combined_koma_lucy_mild_msgan_20260729`) and
darken-only (`cleanupdark`, `combined_koma_cleanupdark_20260730`). Both
converge to the same soft/marbled F1@2px ~0.41 ceiling; darken-only is not
a net improvement (worse chamfer/recall). See `doc/work_log.md`
("2026-07-31: Direction 6/8/9 Survey Concluded").

## Direction 6: Confidence / Thickness Multi-Head

Separate line presence from line strength/thickness.

Possible outputs:

- line confidence
- thickness/ink strength
- optional final composed line output

Expected benefit:

- targets the mid-gray ambiguity directly
- gives MoE/router more interpretable outputs later

Risk:

- requires defining pseudo-targets for confidence/thickness
- can add complexity before the base refiner is good

Minimum experiment:

- two-head U-Net where one head predicts skeleton/center confidence and the
  other predicts normal ink

**Result (2026-07-31, koma dataset):** first attempt (no residual anchor to
the aux input) was chronically under-inked (F1@2px 0.198, 3 epochs) --
root-caused to the new architecture reconstructing ink from scratch instead
of correcting the aux baseline like every adopted model. Fixed
(`DualHeadRefinerGenerator` now predicts `aux_logits + bounded_correction`)
and retrained: F1@2px 0.410 / best-of-family chamfer 4.530, but
precision/ink_ratio show clear over-inking (2.61x) and the same
soft/marbled texture family, not a qualitative jump. Not adopted. See
`doc/work_log.md` ("2026-07-31: Direction 6/8/9 Survey Concluded").

## Direction 7: Soft Morphology / Line-Width Loss

Add differentiable constraints on output morphology.

Possible losses:

- soft erosion/dilation consistency
- line-width penalty
- isolated blob penalty
- soft skeleton consistency

Expected benefit:

- directly attacks thick blobs, soft fields, and mid-gray spread
- can be added to current refiner

Risk:

- easy to over-thin useful line width
- implementation needs careful numerical behavior

Minimum experiment:

- add a cheap local width/blob penalty based on max-pooling and mean ink
- compare to the skeleton-target sweep

## Direction 8: HED / DexiNed-Style Edge Head

Treat the task more like edge detection plus line rendering.

Expected benefit:

- architecture aligns with line extraction
- multi-scale side outputs may help long and short lines

Risk:

- natural-image edge detector assumptions may not match manga roughs
- more implementation than residual cleanup

Minimum experiment:

- U-Net with side-output edge predictions at decoder scales
- supervise side outputs with downsampled GT line/skeleton

**Result (2026-07-31, koma dataset):** same under-anchored first attempt
(F1@2px 0.190, 3 epochs), same residual-anchor fix applied. Even after the
fix, needed the project's standard 10-epoch from-scratch-unet budget
(3 epochs only reached 0.235) to reach F1@2px 0.328 -- still below the
adopted ~0.41 ceiling and judged to be converging toward it rather than a
different one; not worth further epochs. Not adopted. See
`doc/work_log.md` ("2026-07-31: Direction 6/8/9 Survey Concluded").

## Direction 9: Attention / Swin-Like Refiner

Add attention so the refiner can reason over longer line structures.

Expected benefit:

- may improve long contours and face/hair consistency
- helps when local texture is ambiguous

Risk:

- implementation and memory cost
- attention can still copy atari softness if supervision is weak

Minimum experiment:

- add a lightweight bottleneck self-attention block to 2ch U-Net
- compare to plain `refiner_unet`

**Result (2026-07-31, koma dataset):** implemented with the residual-anchor
lesson (from Directions 6/8's first attempts) applied from the start,
trained 10 epochs directly. F1@2px 0.348, best-balanced ink_ratio of the
family (1.016) but still below the adopted ~0.41 ceiling, no visible
long-range-coherence benefit from the attention block specifically. Not
adopted. **This closes out the short architecture survey (Directions
5/6/8/9 all land at or below the same ~0.40-0.42 F1@2px soft/marbled
ceiling)** -- per the decision rule below, next work moves to data/target
design (unpaired-rough integration, tried and not yet successful -- see
`doc/work_log.md`) or Direction 4. See `doc/work_log.md` ("2026-07-31:
Direction 6/8/9 Survey Concluded").

## Short Survey Priority

Run these before committing to a longer route:

1. Shallow residual cleanup refiner
2. Multi-scale PatchGAN with feature matching
3. Soft morphology / line-width loss

Why this order:

- residual cleanup is the fastest way to test whether architecture bias can
  reduce soft repainting
- multi-scale GAN + feature matching is the most plausible GAN-i2i upgrade
- morphology loss is the most direct loss-side attack on mid-gray/line-width
  behavior

Each survey should use the same clean eval montage and compare at minimum:

- `milddup800`
- `resnet_atari`
- `tight_bin12`
- `bin20_ink16`
- new candidate(s)
- GT line

## Cleanup Follow-Up Plan

The first shallow residual cleanup survey was numerically promising but showed
white/emboss-like halo artifacts in montage inspection.

If continuing the cleanup direction, test these changes instead of simply
adding more epochs:

- reduce residual `max_delta`
- add a correction magnitude regularizer
- restrict correction to darkening/ink-addition only, or heavily penalize
  lightening
- use the atari output as a cleanup mask/context rather than directly adding
  correction around atari logits

Current status:

- keep `cleanup_skel06` as a promising reference
- do not adopt current cleanup output globally
- finish the remaining cross-direction survey first before deepening cleanup

## Remaining Short Surveys

Run next in this order:

1. multi-scale PatchGAN + feature matching
2. soft morphology / line-width loss
3. line/structure perceptual loss

Decision rule:

- if a direction gives a clear montage improvement, schedule a longer controlled
  run
- if it only improves metrics by adding ink or texture, keep it as an expert
  candidate at most
- if no remaining direction gives a quality jump, return to data/target design
  and MoE/router planning

## Halo Mitigation Plan

Observation:

- many downstream failures look like halo/fringe in the atari output being
  amplified by the next stage
- `cleanup_msgan_fm` is more balanced than earlier cleanup variants, but still
  preserves some pencil/halo texture
- before deepening a single cleanup family, test whether preventing halo from
  entering or being directly reused changes the outcome

Run a combined halo survey around the comparatively promising
multi-scale-GAN + feature-matching cleanup setup.

Candidate axes:

1. Halo-suppressed atari source
   - generate a cleaner atari/cache using the current `cleanup_msgan_fm`
     checkpoint
   - then train another msgan+FM cleanup/refiner using that cleaner atari as
     the auxiliary channel
   - tests whether reducing halo at the atari source helps downstream cleanup

2. Atari postprocess / hint conversion
   - keep the ResNet atari generator fixed
   - preprocess atari into a less halo-prone hint before training the next
     stage
   - first variants: low-density cutoff and DoG/local-background subtraction
   - tests whether "do not pass the halo" helps without retraining atari

3. Atari application order change
   - avoid directly adding corrections around atari logits
   - use atari as context/mask rather than as the base output
   - tests whether direct residual-around-atari is causing halo amplification

Expected comparison:

- `milddup800`
- raw `resnet_atari`
- `cleanup_msgan_fm`
- candidate 1
- candidate 2
- candidate 3
- GT line

Decision rule:

- prefer candidates that reduce halo/emboss texture in montage even if metric
  gains are modest
- if halo is reduced by postprocess, later retrain the atari generator to
  produce that kind of hint directly
- if application order wins, cleanup architecture should move away from direct
  residual addition around atari logits
