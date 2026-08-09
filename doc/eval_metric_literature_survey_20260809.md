# Literature Survey: Illustration/Line-Art Comparison Metrics (2026-08-09)

Background-agent literature check, triggered by the same-day chamfer-to-GT
discrimination finding (`doc/eval_metric_inventory.md`,
`doc/work_log.md` "Workstream C" entries). Question: is our point-distance
(chamfer) family's density-saturation weakness a known, already-solved
problem in some adjacent field, and is there a better-established
alternative for our two open questions --
(a) "did the output structurally follow its rough/conditioning input" and
(b) "does the output look like genuine finished line art" (no paired
reference)?

## Headline finding

**Our finding is not novel.** The boundary-detection field hit the exact
same wall decades ago: Pratt's Figure of Merit (1978, an inverse-distance
metric structurally similar to Chamfer) was recognized as inadequate and
replaced by the BSDS precision-recall protocol (Martin/Fowlkes/Malik, early
2000s; BSDS500, Arbelaez et al., TPAMI 2011), which every edge/contour
detector since (Canny successors, HED, RCF, BDCN...) is scored against. The
same class of problem is still active research territory in 2026 (MatchED,
CVPR 2026, explicitly names "edge thickening from many-to-one matches" as
an unsolved issue even for learned edge detectors).

**Mechanism**: BSDS matches predicted-vs-GT boundary pixels via **one-to-one
bipartite graph matching** (Hungarian / `linear_sum_assignment`) within a
small tolerance radius, not nearest-neighbor distance thresholding. Because
each GT pixel can be claimed by at most one predicted pixel, extra ink near
an already-matched region becomes an unmatched false positive -- exactly the
penalty missing from both our chamfer and our `f1_2px`/`precision_2px`/
`recall_2px` (which are computed from the *same* distance-transform field as
chamfer, just thresholded -- a many-to-one match, not a rigorous one).

## Metric families found

### 1. BSDS-style bipartite-matched precision/recall/F
Field-standard boundary-detection eval since the early 2000s. Refs:
[Martin/Fowlkes/Malik](https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/martin_et_al.pdf),
[BSDS500](https://github.com/BIDS/BSDS500),
[survey](https://mdpi.com/2313-433X/4/6/74/htm).

### 2. MatchED (CVPR 2026)
Confirms the same failure mode is still being actively fixed for learned
edge detectors in 2026, via confidence-weighted bipartite matching baked
into the training loss (not just eval).
[arxiv.org/html/2602.20689](https://arxiv.org/html/2602.20689)

### 3. Chamfer's density-blindness in the point-cloud literature
Different domain (3D shape), same root mechanism. Density-aware Chamfer
Distance (NeurIPS 2021) reweights for local density mismatch; a separate
2026 analysis frames the "many-to-one collapse" as gradient-structural when
Chamfer is used as an *optimization objective* (less directly relevant to
us since we only use it for measurement, not as a loss).
[DCD](https://dl.acm.org/doi/abs/10.5555/3540261.3542489),
[structural-failure analysis](https://arxiv.org/html/2603.09925)

### 4. ControlNet / ControlNet++ "controllability" metrics
The original ControlNet paper already does a roundtrip-style eval for
segmentation conditioning (re-segment the output, report IoU against the
original mask). **ControlNet++ (ECCV 2024) formalizes this as a
"controllability" consistency reward with a per-condition-type comparison
function: mIoU for segmentation, RMSE for depth, and *SSIM for line-art/edge
conditions specifically*.**
[ControlNet](https://arxiv.org/abs/2302.05543),
[ControlNet++](https://arxiv.org/abs/2404.07987) /
[code](https://github.com/liming-ai/ControlNet_Plus_Plus)

Our `condition_roundtrip_fidelity.py`, built independently the same day,
has the same *framework* (re-extract conditioning from the output, compare
to the original) -- convergent, correct design. Its weak point is narrower
than "the whole idea might be wrong": the comparison function inside it is
still Chamfer (imported from `tile_region_manifest_480.py`), the same
primitive that failed for `evaluate_fixed_outputs.py`. It scored well in
today's 5/6 test likely because self-consistent conditioning maps are
closer in density to each other than model-output-vs-hand-drawn-GT is, not
because the underlying mechanism became more robust -- **not yet
stress-tested at higher density.**

### 5. Sketch-based image retrieval (SBIR) structure descriptors
Classical baselines (Shape Context, Hausdorff matching) are documented in
this field as fragile/noise-sensitive -- consistent with our finding.
**GF-HOG** (Gradient Field HOG; Hu, Barnard, Collomosse, CVPR 2010)
computes local gradient-orientation histograms over sketch strokes instead
of point coincidence -- reported to outperform SIFT/HOG/shape-context/
structure-tensor for sketch retrieval. Plausibly useful specifically for
detecting content-independent repeating-texture hallucination (narrow
orientation histogram) vs. genuine varied linework (broad, content-driven
histogram) -- independent of ink density.
[GF-HOG](https://eprints.kingston.ac.uk/id/eprint/43488/),
[eval](https://www.sciencedirect.com/science/article/abs/pii/S1077314213000349)

Learned sketch embeddings (TU-Berlin/QuickDraw/Sketchy triplet nets) exist
but target retrieval/classification ("same object category"), not
fine-grained per-stroke structural correspondence -- lower priority for us.

### 6. Manga/line-art extraction papers' own eval protocols
"Deep extraction of manga structural lines" (SIGGRAPH Asia 2017) and
related line-extraction work report pixel-level precision/recall/
F-measure -- i.e. this subfield already inherited the BSDS convention
rather than using raw Chamfer. Colorization-adjacent papers ("Deep Line
Art Video Colorization...") report SSIM/PSNR/MS-SSIM/FID/MOS for output
quality (not directly a line-structure-fidelity metric, but confirms SSIM
is the default "did the structure survive" tool in this exact content
domain).
[manga line extraction](https://dl.acm.org/doi/10.1145/3072959.3073675) /
[code](https://github.com/ljsabc/MangaLineExtraction_PyTorch),
[line art colorization](https://arxiv.org/abs/2003.10685)

### 7. Perceptual/learned metrics (LPIPS, DISTS, CLIP, DINO)
No paper found validating these specifically on binary/sparse line-art
content -- both are ImageNet-pretrained (natural-photo) backbones, and
LPIPS is already documented to degrade off-distribution elsewhere. DISTS
is more texture-sensitive than LPIPS, which cuts both ways: it might catch
the crosshatch-hallucination artifact better, but could also over-penalize
legitimate dense/textured line-art regions as "different" even when
structurally correct -- an inverse failure mode to today's finding.
**Treat as exploratory only; re-validate against the hand ranking before
trusting.**

### 8. No-reference distributional metrics (FID/KID analogs)
No line-art-specific FID variant found. FID is documented as sensitive to
domain shift away from ImageNet's natural-photo distribution -- a direct
argument against a naive "FID-for-line-art" using an Inception/CLIP/DINO
backbone. **This validates `measure_lineart_profile.py`'s hand-designed
structural-statistics approach as the more defensible choice, not a
corner-cut** -- worth stating explicitly as the *reason* for that design in
its own docs, not just an accident of how it was built.

## Prioritized candidates to prototype next

1. **BSDS-style one-to-one bipartite-matched precision/recall/F**
   (`scipy.optimize.linear_sum_assignment` on a sparse candidate-pair cost
   matrix within tolerance radius) as an addition to `evaluate_fixed_
   outputs.py` and `condition_roundtrip_fidelity.py`. Most targeted fix for
   today's exact failure mode; decades of prior art and reference
   implementations (BSDS `correspondPixels`) to check against. For (a).
2. **Add/swap SSIM as the comparison function inside
   `condition_roundtrip_fidelity.py`**, matching ControlNet++'s established
   per-condition-type metric for edge/line-art conditions. Cheap
   (`skimage.metrics.structural_similarity`), directly informed by prior
   art doing the same roundtrip idea at scale. For (a).
3. **A GF-HOG-inspired local gradient-orientation-histogram descriptor**,
   compared for orientation-distribution similarity rather than point
   coincidence -- targets the content-independent-repeating-texture
   hallucination mode specifically. Secondary check, for (a).
4. **Re-validate `condition_roundtrip_fidelity.py` at higher ink density**
   than today's 10-tile set -- the 5/6 agreement may not reflect a fixed
   mechanism, just a less-dense test case. Cheap sanity check before
   leaning on it further.
5. **No action on `measure_lineart_profile.py`** -- confirmed as the
   mechanically correct choice for (b), better grounded than a naive
   FID/LPIPS-for-line-art alternative given documented domain-mismatch
   issues.
6. **Lower priority / exploratory**: DISTS or CLIP-similarity as an extra
   signal on a handful of tiles, explicitly flagged unvalidated on this
   content type, cross-checked against the hand ranking before being
   trusted for any decision.

## Explicit flags on our own approach

- The density-saturation failure of Chamfer/tolerance-F1 on dense edge maps
  is the exact reason boundary detection moved off Pratt's FOM decades ago
  -- correct finding, not a new one; the fix already exists and is
  well-tested.
- `f1_2px`/`precision_2px`/`recall_2px` in `evaluate_fixed_outputs.py` are
  named like BSDS-family F-scores (which readers may import intuition
  from) but are computed with a materially weaker many-to-one matching
  rule. Worth a naming/documentation caveat so nobody reads these numbers
  expecting BSDS-equivalent rigor.
- The conditioning-roundtrip *framework* is independently-converged,
  correct design (matches ControlNet's own eval and ControlNet++'s
  formalized controllability reward) -- not a reinvented wheel. Its weak
  point is narrower: the comparison function inside it (Chamfer) is the
  same outdated primitive, and established literature already specifies
  the fix (SSIM) for this exact condition type.
- `measure_lineart_profile.py`'s no-reference structural-profile approach
  is arguably *ahead* of naive practice, not behind it -- it sidesteps a
  real, documented FID/LPIPS domain-mismatch problem. Don't second-guess it
  based on today's findings.

## Sources

- [Martin, Fowlkes, Malik -- boundary detection / BSDS precision-recall protocol](https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/martin_et_al.pdf)
- [BSDS500 (Arbelaez, Maire, Fowlkes, Malik, TPAMI 2011)](https://github.com/BIDS/BSDS500)
- [MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision (CVPR 2026)](https://arxiv.org/html/2602.20689)
- [Density-aware Chamfer Distance (NeurIPS 2021)](https://dl.acm.org/doi/abs/10.5555/3540261.3542489)
- [On the Structural Failure of Chamfer Distance in 3D Shape Optimization](https://arxiv.org/html/2603.09925)
- [ControlNet (Zhang, Rombach, Agrawala)](https://arxiv.org/abs/2302.05543)
- [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](https://arxiv.org/abs/2404.07987) / [code](https://github.com/liming-ai/ControlNet_Plus_Plus)
- [Gradient Field HOG for sketch-based retrieval (Hu, Barnard, Collomosse)](https://eprints.kingston.ac.uk/id/eprint/43488/) / [performance eval](https://www.sciencedirect.com/science/article/abs/pii/S1077314213000349)
- [Deep extraction of manga structural lines (SIGGRAPH Asia 2017)](https://dl.acm.org/doi/10.1145/3072959.3073675) / [code](https://github.com/ljsabc/MangaLineExtraction_PyTorch)
- [Deep Line Art Video Colorization with a Few References](https://arxiv.org/abs/2003.10685)
- [A Review of Supervised Edge Detection Evaluation Methods (MDPI 2018)](https://mdpi.com/2313-433X/4/6/74/htm)
- [Evaluating Edge Detection through Boundary Detection (EURASIP 2006)](https://asp-eurasipjournals.springeropen.com/articles/10.1155/ASP/2006/76278)
- [Fréchet Inception Distance -- mechanism and domain-shift limitations](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
