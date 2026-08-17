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
