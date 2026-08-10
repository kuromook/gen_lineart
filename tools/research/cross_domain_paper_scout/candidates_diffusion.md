# Diffusion Track -- Candidates Log

Accumulating log of diffusion papers whose *mechanism* might address the
specific failure modes of this project's diffusion direction: condition
adherence (ControlNet output only loosely following the input rough),
over-inking, fine-tuning on a small paired dataset (~1.5k tiles), caption
conditioning, and two-stage structure-then-refine formulations. See
`README.md` for how this file gets populated and how to review it, and
`prompt_diffusion.md` for the full context handed to each run.

Topology/continuity candidates from non-art fields live in the separate
`candidates.md`.

Entries are grouped by the date/focus-topic of the run that found them.

## 2026-08-10 -- structure- and layout-preserving image-to-image diffusion (keeping input geometry, not just input semantics)

### A Tilted Seesaw: Revisiting Autoencoder Trade-off for Controllable Diffusion (2026)
- **Topic**: latent-space condition drift / VAE as a ceiling on controllability
- **Venue/link**: arXiv:2601.21633 (https://arxiv.org/abs/2601.21633)
- **Core idea**: Argues that autoencoders selected for generative quality (best gFID) rather than reconstruction quality induce "condition drift" in latent space, and that gFID is only weakly predictive of condition preservation while reconstruction-oriented (especially instance-level) metrics are strongly predictive. Proposes a multi-dimensional condition-drift evaluation protocol, including training lightweight predictors that try to recover the condition back out of the latent.
- **Relevance here**: Speaks directly to condition adherence, and possibly to over-inking: if the SD1.5 VAE cannot round-trip a binary thin-line tile, no amount of ControlNet training can recover geometry the latent never carried, and decode artifacts around thin strokes would inflate ink_ratio. Concretely: encode/decode the *ground-truth line tiles* through the frozen VAE with no diffusion at all and score F1@2px and ink_ratio of the reconstruction — that number is the hard ceiling for the whole diffusion track, and it is currently unmeasured.
- **Cost to try**: No training. A one-off evaluation script (VAE encode → decode → existing metrics). Hours, not GPU-days.
- **Confidence**: medium-high *as a diagnostic* (it is cheap and it either exonerates or indicts the latent space, which would redirect the whole track); low-medium as a *fix*, since the remedy would be swapping/finetuning the VAE, which is a much larger change.

### ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems (2023)
- **Topic**: control-adapter architecture / feedback bandwidth
- **Venue/link**: arXiv:2312.06573 (https://arxiv.org/abs/2312.06573)
- **Core idea**: Frames adapter-based control as a feedback-control system and diagnoses that in vanilla ControlNet the feedback from the generation process to the control branch is "timely sparse and has a small number of bits", so corrective signals arrive long after the features they should correct. The fix is architectural: rewire the control/generation communication to be high-frequency and large-bandwidth, which also lets the control branch be much smaller ("noticeably fewer parameters", ~2x faster train and inference).
- **Relevance here**: This is a *why-it-fails* account of exactly the observed symptom — the UNet commits to plausible content early and the control signal never pulls it back. It also happens to shrink the adapter, which matters a lot when a 361M-param adapter is being fit to 1,489 tiles; the retrained-10x result suggests the problem is not optimization but capacity/architecture mismatch. Concretely: retrain the adapter in the XS configuration instead of the standard ControlNet copy, at a fraction of the parameters.
- **Cost to try**: Architecture change plus a full retrain of the adapter (but a cheaper retrain than the current one).
- **Confidence**: medium. The diagnosis fits the symptom well and the smaller parameter count fits the data budget, but its results are demonstrated on depth/canny/segmentation over large datasets, not on a 1.5k-tile domain-specific pairing.

### ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback (2024)
- **Topic**: training objective / explicit pixel-level condition consistency
- **Venue/link**: ECCV 2024, arXiv:2404.07987 (https://arxiv.org/abs/2404.07987)
- **Core idea**: Diagnoses that ControlNet optimizes controllability only *implicitly*, through the latent denoising loss, with no term that ever measures whether the output actually matches the condition (they show e.g. 27.46 vs 50.7 mIoU for segmentation control). Adds an explicit cycle-consistency reward: run a pretrained discriminative extractor on the generated image, compare the recovered condition to the input condition. To make this affordable they avoid backprop through the sampling chain by adding noise at a small timestep and doing a single-step denoise to get an image to reward (reported as taking memory from ~340GB to feasible).
- **Relevance here**: The project's loss is exactly the diagnosed one — `MSE(eps_hat, eps)` never sees the rough. The single-step-denoise reward trick is what makes this runnable on one GPU. Concretely: add an auxiliary reward on the single-step-decoded x0 against the ground-truth line tile using the metrics that are actually failing — a differentiable soft-F1/chamfer alignment term and an ink-ratio penalty — so over-inking and geometry drift are penalized directly rather than hoped for.
- **Cost to try**: Training-objective change plus a finetune of the existing adapter (does not require restarting from scratch).
- **Confidence**: medium. The mechanism attacks both failure modes head-on and this project has paired ground truth, so the "reward model" can be direct supervision rather than a noisy extractor. Risk: reward finetuning on 1.5k tiles could collapse diversity or over-thin strokes, and the paper's gains are on large-scale conditions.

### Heeding the Inner Voice: Aligning ControlNet Training via Intermediate Features Feedback (InnerControl) (2025)
- **Topic**: per-timestep alignment of intermediate UNet features to the control signal
- **Venue/link**: arXiv:2507.02321 (https://arxiv.org/abs/2507.02321)
- **Core idea**: Argues the previous fix (ControlNet++ above) only aligns at late denoising steps, while the image's structure is actually decided early. Trains lightweight convolutional probes that reconstruct the input control signal from intermediate UNet features at *every* denoising step, then uses those reconstructions as a per-step alignment loss across the whole trajectory.
- **Relevance here**: This is the sharpest available statement of the suspected mechanism — the layout is chosen in the early steps, which is precisely when a weakly-adhering adapter loses the rough's geometry, and it explains why inference-time conditioning-scale and guidance sweeps did nothing (they cannot undo a layout already committed). Concretely: train small probes to predict the rough tile from mid-block features at each step and add that as an auxiliary loss during adapter finetuning; even *before* changing the loss, the probes are an instrument — plot per-timestep recoverability of the rough to see exactly when the current model stops tracking it.
- **Cost to try**: Training change (extra probe networks + auxiliary loss), on top of a ControlNet++-style setup. The diagnostic-only version is cheap.
- **Confidence**: medium. Strong fit to the failure mode and the diagnostic use is low-risk, but it stacks on ControlNet++ so it inherits that complexity, and the paper is evaluated on standard large-scale control benchmarks.

### Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation (2022/2023)
- **Topic**: training-free structure injection via UNet spatial features and self-attention
- **Venue/link**: CVPR 2023, arXiv:2211.12572 (https://arxiv.org/abs/2211.12572)
- **Core idea**: Analyses (via PCA over UNet activations) *where* structure lives in a pretrained SD UNet: decoder layer-4 spatial features carry localized semantic layout shared across domains, and self-attention maps encode the spatial grouping. Translation is then done with no training at all — invert the guidance image, and during generation replace layer-4 features (up to timestep ~40/50) and decoder self-attention matrices (up to ~25/50) with those cached from the guidance image; the thresholds are lowered to 25/25 for primitive/textureless guidance images.
- **Relevance here**: Pure geometry preservation with zero data and zero training, i.e. the exact axis the diffusion track fails on, at the exact scale this project can afford. The "primitive/textureless guidance" caveat matters — a pencil rough *is* a textureless guidance image, and the paper already prescribes a different schedule for that case. Concretely: DDIM-invert the rough tile and inject its features into the anime-checkpoint sampling, and use the injection thresholds as a structure/style knob — a knob that is qualitatively different from the conditioning-scale and guidance-scale knobs already ruled out.
- **Cost to try**: Inference-time only. No retraining; can be tested against the existing eval harness in a day.
- **Confidence**: low-medium for a standalone fix — inverting an out-of-distribution pencil rough into an anime SD1.5 prior is a real domain gap, and PnP preserves layout but has no notion of producing binary ink. Still logged because the cost is near zero and because the layer/timestep analysis is reusable: it tells you which layers to touch in *any* of the trained approaches above.
