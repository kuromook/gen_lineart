# Cross-Domain Paper Scout -- Candidates Log

Accumulating log of papers from outside image/illustration/art that might
have mathematically transferable ideas for this project's line-art
stroke-continuity/topology problem, the way clDice (originally a
vessel-segmentation metric) did. See `README.md` for how this file gets
populated and how to review it. Diffusion-track candidates live in the
separate `candidates_diffusion.md`.

Entries are grouped by the date/focus-area of the run that found them.

## 2026-08-17 -- remote sensing: road network extraction from satellite/aerial imagery

### Promoting Connectivity of Network-Like Structures by Enforcing Region Separation (2020)
- **Field**: aerial imagery, road and irrigation-canal delineation
- **Venue/link**: arXiv:2009.07011 (Oner, Koziński, Citraro, Dadap, Konings, Fua) -- https://arxiv.org/abs/2009.07011
- **Core idea**: Rather than scoring the predicted foreground, the loss scores the *background*: a ground-truth road separates the background into regions on either side of it, so a gap in the prediction lets two background regions that should be disconnected touch. The loss is computed on small crops and penalizes exactly those spurious background merges (and, symmetrically, spurious separations for false positives), making connectivity a differentiable property of the complement.
- **Transfer hypothesis**: This is a dual of clDice's trick and fits line art unusually well, because a broken stroke's practical damage in illustration *is* a background leak -- it merges two regions that should be closed cells (the same failure that breaks bucket-fill/flat colorization downstream). A one-pixel gap that a Dice/BCE loss barely notices produces a large region-merge penalty here, which is precisely the error asymmetry the stroke-continuity problem needs.
- **Confidence**: high

### Beyond the Pixel-Wise Loss for Topology-Aware Delineation (2018)
- **Field**: curvilinear structure delineation (aerial road imagery, neuronal microscopy)
- **Venue/link**: CVPR 2018, arXiv:1712.02190 -- https://arxiv.org/abs/1712.02190
- **Core idea**: Adds a loss term comparing prediction and ground truth in the feature space of a pretrained VGG19 (whose mid-level filters respond to connectivity/junction/elongation patterns) rather than per pixel, plus an iterative refinement pipeline that re-feeds the network its own previous delineation so later passes can close gaps left by earlier ones.
- **Transfer hypothesis**: The iterative-refinement half is the more credible transfer: a second pass conditioned on (rough sketch + first-pass line art) gives the model an explicit opportunity to bridge fragments, which a single feed-forward pass never gets. The perceptual-topology term is worth trying but is only an indirect proxy for topology -- unlike clDice it has no explicit skeleton/connectivity construction, so its gain may not survive on stroke data.
- **Confidence**: medium

### Improved Road Connectivity by Joint Learning of Orientation and Segmentation (2019)
- **Field**: road network extraction from satellite imagery
- **Venue/link**: CVPR 2019 (Batra, Singh, Pang, Basu, Jawahar, Paluri) -- https://openaccess.thecvf.com/content_CVPR_2019/papers/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.pdf ; code https://github.com/anilbatra2185/road_connectivity
- **Core idea**: Adds an auxiliary per-pixel *orientation* head (the local tangent angle of the road, quantized into bins) trained jointly with the segmentation head in a stacked multi-branch module, so the shared features must encode "which way this structure is going." A second-stage refinement network is pretrained to repair *artificially corrupted* ground-truth masks (segments erased) and then fine-tuned to fix real predictions.
- **Transfer hypothesis**: Strokes have a much cleaner tangent field than roads do, and orientation supervision can be derived for free from the ground-truth line art -- forcing the features to be direction-aware should discourage the model from dropping a pixel mid-stroke where the tangent is unambiguous. The corrupt-then-repair pretraining is directly reusable as a cheap gap-closing stage: randomly erase segments from clean line art and train a network to restore them.
- **Confidence**: medium

### APLS (Average Path Length Similarity) graph-theoretic metric for road networks (2019)
- **Field**: road network extraction evaluation (SpaceNet challenge)
- **Venue/link**: described in Van Etten, "City-Scale Road Extraction from Satellite Imagery v2", WACV 2020, arXiv:1908.09715 §5.1 -- https://arxiv.org/abs/1908.09715 ; reference implementation https://github.com/CosmiQ/apls
- **Core idea**: Convert both ground truth and prediction to graphs, snap proposal nodes to ground-truth nodes within a buffer distance, then sum the relative differences in shortest-path length between node pairs across the two graphs, scoring any path that is missing in the proposal as 0. Edge weights are arbitrary, so the same machinery scores different notions of "cost along the structure."
- **Transfer hypothesis**: This is a *metric*, which is the gap the project explicitly has -- pixel-overlap scores are nearly blind to one-pixel breaks, whereas APLS is maximally sensitive to them (a single break sends every path through it to 0). Skeletonize both line arts, build graphs, and APLS becomes a direct "can you still travel along this stroke" continuity score. Caveats to test: node correspondence is fiddly on free-form strokes, and dense hatching produces large graphs, so sampling control nodes (as the paper does for city-scale) would be needed.
- **Confidence**: medium

### CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation (2025)
- **Field**: curvilinear structure delineation (neuron tracing in EM, retinal vessels, light-microscopy brain volumes)
- **Venue/link**: arXiv:2504.00753 (Esmaeilzadeh, Garaaghaji, Hallaji Azad, Oner) -- https://arxiv.org/abs/2504.00753
- **Core idea**: Makes the shortest-path idea *differentiable and trainable* rather than just an eval metric: sample pairs of vertices known to be connected in the ground-truth graph, run Dijkstra on the predicted distance map (search restricted to a dilated corridor around the ground-truth path), and penalize the cost accumulated along that predicted path -- so any break forces the path to detour through high-cost pixels and produces gradient exactly at the break.
- **Transfer hypothesis**: It attacks the same failure this project has -- a loss that is flat with respect to a tiny gap -- by making the penalty a function of traversal cost rather than pixel count, and it localizes gradient to the breaking pixels instead of spreading it over the whole mask. Sampling endpoint pairs along ground-truth strokes is straightforward. Main risk is cost: Dijkstra per sampled pair per training step, and the paper is a 2025 preprint evaluated only on biomedical data, so the result is not yet independently confirmed.
- **Confidence**: medium

## 2026-09-14 -- materials science / NDT: crack detection in concrete, pavement, or composites

### TOPO-Loss for continuity-preserving crack detection using deep learning (2022)
- **Field**: masonry-building crack detection from earthquake-damage photographs (Construction and Building Materials)
- **Venue/link**: Construction and Building Materials, DOI:10.1016/j.conbuildmat.2022.128264 -- https://www.sciencedirect.com/science/article/pii/S0950061822019250 ; code https://github.com/eesd-epfl/topo_crack_detection
- **Core idea**: Adds a MALIS-style connectivity-oriented loss term (the repo exposes it as `malis_neg`/`malis_pos` weights) on top of MSE+Dice -- MALIS (originally from connectomics/EM neuron-affinity segmentation) finds the maximin edge along the path connecting two pixels in the predicted affinity graph and pushes gradient onto exactly that bottleneck edge when it disagrees with ground-truth connectivity, rather than spreading loss uniformly. The paper also introduces a new "Cracks Per Patch" (CPP) evaluation metric that counts fragment breaks per image patch, explicitly built to catch what Dice/IoU miss, and the method is designed to stay robust under imprecise/noisy human crack annotations.
- **Transfer hypothesis**: This is itself a domain-transfer precedent for the exact hypothesis this project is testing -- a connectomics topology loss (MALIS) carried over to crack images, with essentially the same "flat pixel loss is blind to a one-pixel gap" motivation as clDice. CPP is a cheap, interpretable metric (fragment count per patch) that's simpler to implement than skeleton-graph-based metrics and worth adding alongside clDice for stroke-continuity evaluation; the malis-style maximin-edge gradient targeting is an alternative bottleneck-localized loss mechanism to try if clDice's soft-skeleton gradient turns out too diffuse.
- **Confidence**: medium

### Topology-informed deep learning for pavement crack detection: Preserving consistent crack structure and connectivity (TopoM-CrackNet) (2025)
- **Field**: pavement crack segmentation for automated road maintenance
- **Venue/link**: Automation in Construction, vol. 174, article 106120 (Jing, Ding, Xu, Xu, Jinchao, Hong, Hainian) -- https://www.sciencedirect.com/science/article/abs/pii/S0926580525001608 ; https://trid.trb.org/View/2526982
- **Core idea**: Combines a persistent-homology topological loss (in the lineage of Hu et al.'s TopoLoss, NeurIPS 2019/CVPR) with a U-Net augmented by the Vmamba state-space backbone. Rather than comparing skeletons pixel-by-pixel like clDice, persistent homology computes a persistence diagram of topological features (connected components, loops/Betti numbers) for the predicted likelihood map, matches it against the ground-truth diagram, and back-propagates gradient through the specific critical pixels responsible for each topological discrepancy (a spurious break or a spurious loop). Reported as beating a directly-cited "Topoloss" baseline and achieving a lower Betti-number error alongside standard mIoU.
- **Transfer hypothesis**: This is a mechanistically distinct alternative to clDice -- it targets topological invariants (Betti numbers / component count) directly via persistent homology rather than via mask-skeleton overlap, so it could catch failure modes clDice's soft-skeleton comparison misses (e.g., a spurious small loop or an isolated stray fragment that doesn't much affect the skeleton-recall term). Main risk for this project is implementation cost and speed: persistent-homology losses require computing persistence diagrams and critical-point matching per training step, which is more complex and slower than clDice's soft-skeletonization, and gains here are reported only against crack/road-style baselines, not independently reproduced.
- **Confidence**: medium
