# Architecture Decision Log

Updated: 2026-08-01 JST

This file is the single place to check **which model architectures have
been tried, why, what the exact math/code is, and whether the result was
actually eye-balled against ground truth** (not just judged by F1/chamfer
numbers). `doc/work_log.md` has the narrative history of *how* each run
happened session-by-session; this file is the flat reference table of
*what exists and what it proved*.

**Why this file exists:** several nights of work produced cases where a
numeric metric improved but a montage review showed the change did not
serve the architecture's actual stated purpose at all (see the
"目視評価" column below — e.g. the no-adversarial-loss ablation looked
*worse* numerically but the montage review revealed *why*: GAN vs no-GAN
turned out to be a confidence/darkness recalibration on an unchanged
spatial pattern, not the structural difference the number implied). Do
not trust a row's numbers without also reading its 目視評価 cell.

**Branch note:** the CNN+GAN "cleanup"/atari family below lives on the
`cleanup-refiner` branch (`lineart/model_zoo.py`, `scripts/train_i2i_survey.py`).
Direction 4 (diffusion/ControlNet) lives on a separate `diffusion-controlnet`
branch (`scripts/train_controlnet.py`) — architecturally unrelated, kept
separate on purpose. This file is intended to be kept in sync on both
branches since it's a cross-cutting summary; if you find it's diverged,
merge by hand rather than trusting only one copy.

**Data note:** unless stated otherwise, all "koma" rows below train and
evaluate on `combined_koma_20260729` (1489 tiles, 5 sources, current
clean/aligned data) against the 8-tile `eval_clean_lineart004_8.txt`
clean eval set. Rows predating this dataset (BCE baseline, ResNet-GAN
atari, Direction 7 width-loss sweeps) used older data generations — noted
per row.

---

## Foundational models (inputs to the whole `cleanup`/atari family)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Reference baseline: how far does plain BCE alone get you | Plain U-Net, rough(1ch)→line(1ch) direct regression, no aux/hint | `L = BCEWithLogits(pred, target)` | `lineart/unetgenerator.py::UNetGenerator`, `--model unet` in `scripts/train.py`/`train_i2i_survey.py` | Yes — established as the reference "soft, no commitment" texture that every later direction tried to escape | `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5*.png` (see `doc/model_results_summary.md`) | F1@2px 0.301, chamfer 7.08. Not a target, a floor. |
| Generate a "structure hint" (atari) from rough alone, to condition every downstream cleanup/refiner model | ResNet encoder-residual-decoder (`ResnetGenerator`) + PatchGAN adversarial loss | `L_G = L_recon + λ_adv·L_adv(D(rough,pred))`; `ResnetGenerator`: 7×7 head → 2× stride-2 down → 6× `ResnetBlock` (conv-IN-ReLU-conv-IN + skip) → 2× transpose-conv up → 7×7 tail | `lineart/model_zoo.py::ResnetGenerator`, `PatchDiscriminator`; checkpoint `checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth` | Yes, repeatedly, across many later comparisons | (used as the `--aux-dir` input to every cleanup/refiner model below, not evaluated standalone in later koma montages) | **Root cause of the whole family's soft/marbled ceiling (found 2026-08-01)**: this generator's own raw output is already halftone/dithered-soft (verified by direct pixel inspection — see 2026-08-01 entries in `doc/work_log.md`). Every downstream model below inherits this softness because it only computes a small bounded correction around this output. |

## Direction 5: Shallow Residual Cleanup (`--model cleanup` / `cleanupdark`)

Bundles Directions 1 (multi-scale PatchGAN), 2 (feature matching), and 3
(structure/edge loss) as loss terms rather than testing them in isolation
— see "Directions 1/2/3 status" below.

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| **[ADOPTED / production candidate]** Bias the network toward *correcting* the atari hint instead of repainting from scratch, to escape the BCE baseline's soft texture | `ResidualCleanupGenerator`: 2ch in (rough+aux) → 1 conv + 5× (conv-IN-ReLU) blocks → 1 conv → `tanh` bounded correction, added to the aux input's own logit | `out = logit(1 − aux) + tanh(net(rough,aux))·4.0`; loss (`combined_koma_lucy_mild_msgan_20260729`): `L = 0.75·BCE(pos_w=5) + 0.03·L1 + 0.08·tolerant_f1 + 0.14·ink_loss + 0.10·binary_conf + 0.04·structure_pyramid + 0.03·adv(multiscale) + 0.08·feature_match` | `lineart/model_zoo.py::ResidualCleanupGenerator`; `scripts/train_i2i_survey.py`; `experiments/run_combined_koma_lucy_mild_msgan_20260729.sh` | Yes — montage-reviewed repeatedly across many nights, this is the current adopted "least bad" candidate | `results/compare_combined_koma_lucy_mild_msgan_20260729.png` | F1@2px 0.4175, chamfer 4.680, ink_ratio 1.557. **Still soft/marbled in every review** — "adopted" means "best of this family," not "solved." |
| Test whether one-directional (ink-only-adding) correction avoids the soft bidirectional-tanh blending that lets the net paint partial-erasure gray everywhere | `DarkenOnlyCleanupGenerator`: same trunk, but `sigmoid`-bounded, zero/neg-bias-initialized so it starts near the aux baseline and can only *add* ink, never remove | `out = logit(1 − aux) + sigmoid(net(rough,aux))·1.5` | `lineart/model_zoo.py::DarkenOnlyCleanupGenerator`, `--model cleanupdark`; `experiments/run_combined_koma_cleanupdark_20260730.sh` (same loss recipe as msgan above) | Yes | `results/compare_combined_koma_cleanupdark_20260730.png` | F1@2px 0.409 vs msgan's 0.4175 (worse), chamfer worse too. **Not adopted** — one-directional constraint did not net improve. |
| Test whether removing the adversarial/feature-match/structure loss terms (holding architecture fixed) recovers crisper, less-marbled output — i.e. is the marbling a *loss-design* artifact | Same `ResidualCleanupGenerator` architecture as the adopted msgan row, **no GAN at all** | `out` formula identical to msgan row; loss: `L = 0.8·BCE(pos_w=3) + 0.2·L1 + 0.5·edge_loss(Canny)`, no adversarial/feature-match/shape/ink/binary/structure terms | `experiments/run_combined_koma_lucy_mild_noadv_20260801.sh`; `--edge-weight` flag added to `scripts/train_i2i_survey.py` this session (wires up the pre-existing but previously-unused `lineart/losses.py::edge_loss`) | **Yes — this is the exact case that motivated this document.** Numerically worse (F1 0.20 vs 0.42, severe under-inking) *and* still visually soft/marbled — just fainter, not the crisp result the loss-design hypothesis predicted. A follow-up pixel-correlation check (0.94-0.95 correlation between this output and the msgan output) showed GAN vs no-GAN barely changes *where* ink goes — it mostly recalibrates overall darkness/confidence on an unchanged fuzzy spatial pattern | `results/compare_combined_koma_lucy_mild_noadv_20260801.png` | **Hypothesis disproved.** Root cause instead traced to the atari generator's own soft output (see Foundational Models row above) propagating through this architecture's small `tanh`-bounded correction (`max_delta=4.0`), which structurally cannot diverge far from a soft anchor regardless of loss. See `doc/work_log.md` "2026-08-01: No-Adversarial-Loss Ablation." |

## Direction 6: Confidence/Thickness Dual-Head (`--model dualhead`)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Separate "is there a line here" (confidence/skeleton) from "how thick/dark" (ink), targeting the mid-gray ambiguity directly, in case one shared output tensor was the bottleneck | `DualHeadRefinerGenerator`: shared trunk (same conv stack as `cleanup`) → two heads: `ink_head` (bounded correction, same as `cleanup`) + `skeleton_head` (confidence), composed as `final = aux_logits + ink_delta + gain·sigmoid(skeleton_logits)` | v1 (buggy): `final = ink_head(trunk(x))` **with no aux anchor at all**. v2 (fixed): `final = logit(1−aux) + tanh(ink_head(feat))·4.0 + 1.5·sigmoid(skeleton_head(feat))`; loss adds `+ skeleton_weight·BCE(skeleton_logits, GT_skeleton)` | `lineart/model_zoo.py::DualHeadRefinerGenerator`; `experiments/run_combined_koma_dualhead_20260731.sh` (v1). v2 was run as an ad-hoc re-invocation with no dedicated experiment script, no `.done` marker, and (until tonight) no saved montage — only its per-tile output PNGs under `results/combined_koma_dualhead_v2_20260731/` survived | Yes for both, but **v2's montage/metrics did not actually exist on disk until this document was written (2026-08-01)** — the numbers in `doc/work_log.md`/`doc/model_directions.md` were correct, but nothing rendered them until now. Regenerated tonight from the surviving per-tile PNGs: `tools/compare/make_multi_model_eval_compare.py` + `tools/evaluation/evaluate_fixed_outputs.py`, confirmed F1@2px 0.4096 / ink_ratio 2.609 matches the historical prose exactly | v1: `results/compare_combined_koma_dualhead_20260731.png`. v2 (regenerated 2026-08-01): `results/compare_combined_koma_dualhead_v2_20260731.png`, metrics `results/fixed_output_metrics_combined_koma_dualhead_v2_20260731_compare.csv` | v1: no-anchor bug → chronically under-inked, F1@2px 0.198. v2 (anchored): F1@2px 0.4096, best-of-family chamfer 4.530, but ink_ratio 2.609 (over-inking, visibly darker than msgan/v1 in the montage) and a visible dark-blob artifact on one sample. **Not adopted** — same soft/marbled family, no qualitative jump. **This row is itself an example of the exact documentation gap this file exists to prevent**: a real numeric result with no accessible visual artifact for months. |

## Direction 8: HED-Style Multi-Scale Side Outputs (`--model hed`)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Borrow the HED (edge-detection) trick of supervising intermediate decoder scales directly, in case multi-scale supervision helps both long and short lines | `HedUNetGenerator`: full `UNetGenerator` 2ch encoder/decoder + 2 extra 1×1-conv "side output" heads reading the 1/4- and 1/2-resolution decoder features, each supervised against downsampled GT; final output is a bounded correction around aux, same pattern as `cleanup` | `final = logit(1−aux) + tanh(out_conv(d1))·4.0`; side loss: `L += side_weight · mean(BCE(side3, downsample(GT)), BCE(side2, downsample(GT)))`; side heads have zero effect on the final output itself, only shape decoder gradients | `lineart/model_zoo.py::HedUNetGenerator`; `experiments/run_combined_koma_hed_20260731.sh` (v1, buggy), retried as `hed_v2` (still 3 epochs) then `hed_v3` (10 epochs) | Yes, all three attempts | v3 (final, most informative): `results/compare_combined_koma_hed_v3_20260731.png` | v1: same no-anchor bug as dualhead v1 → F1@2px 0.190. v2 (anchored, 3 epochs): F1@2px 0.235, still far under. v3 (anchored, 10 epochs — this project's standard from-scratch-unet budget): F1@2px 0.328, still below the ~0.41 ceiling, judged to be *converging toward* the same ceiling rather than a different one. **Not adopted**, not worth further epochs. |

## Direction 9: Bottleneck Self-Attention (`--model attn`)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Give the refiner an O(1)-hop path between distant pixels (e.g. reconcile hair strands across the face) instead of only the U-Net's local conv receptive field | `AttentionUNetGenerator`: full `UNetGenerator` 2ch encoder/decoder + one `SelfAttention2d` block at the 512-channel bottleneck (60×60 for a 480px input); zero-initialized `gamma` so attention starts as a no-op; output is a bounded correction around aux, same pattern as `cleanup` | `attn(x) = x + γ·softmax(QKᵀ/√d)·V` (γ starts at 0); `final = logit(1−aux) + tanh(out_conv(d1))·4.0`, same loss recipe as the msgan row above (with GAN) | `lineart/model_zoo.py::SelfAttention2d`, `AttentionUNetGenerator`; `experiments/run_combined_koma_attn_20260731.sh` | Yes | `results/compare_combined_koma_attn_20260731.png` | Built with the residual-anchor lesson applied from the start (no buggy v1 needed). F1@2px 0.348, best-balanced ink_ratio of the whole family (1.016), but still below the ~0.41 ceiling and **no visible long-range-coherence benefit from attention specifically** in the montage. Not adopted. **This closed out the Direction 5/6/8/9 survey** — all land at or below the same ~0.40-0.42 F1@2px soft/marbled ceiling. |

## Direction 4: Diffusion / ControlNet (separate `diffusion-controlnet` branch)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| After the CNN+GAN family (Directions 5/6/8/9) all converged to the same ceiling, try a fundamentally different generative paradigm with real prior knowledge of "what a line drawing looks like" | SD1.5-family UNet (`AOM3A1B_orangemixs.safetensors`, frozen) + trained ControlNet adapter (~361M params, conditioned on the rough tile); standard latent-diffusion noise-prediction training | `ε̂ = UNet(z_t, t, τ, down/mid_residuals)`, `down/mid_residuals = ControlNet(z_t, t, τ, rough)`; `L = MSE(ε̂, ε)`; single fixed caption `τ` for all tiles (later replaced with per-tile WD14 auto-tags, not yet used in a real run) | `scripts/train_controlnet.py`, `scripts/infer_controlnet.py`, `scripts/tag_wd14.py` (all on `diffusion-controlnet` branch only) | Yes | `results/compare_controlnet_koma_direction4_20260731.png`, sweep variant `results/compare_controlnet_koma_direction4_20260731_sweep.png` | **First result in the whole project to visually break the soft/marbled ceiling** — crisp, confident, fully binary anime-style ink after only 10 epochs (1860 steps). But it hallucinates content only loosely related to the specific input rough (wrong pose/expression on some tiles), so F1@2px is *worse* than the CNN+GAN baseline (0.20 vs 0.42) despite looking qualitatively better. A conditioning-scale/guidance-scale sweep did not fix the hallucination (see sweep montage) — diagnosed as undertraining (ControlNet's documented "sudden convergence phenomenon"), not a fixable inference setting. **10x longer run (18,600 steps) scheduled via crontab for 2026-08-03 00:00 JST**, result not in yet as of this writing. |

## Single-stage direct regression (replicates `notebooks/gen_lineart.ipynb`'s original architecture on current clean data)

| 目的 | アーキテクチャ | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Test the one architecture variable untouched by every row above: all of Directions 5/6/8/9 (and the no-adversarial-loss ablation) held the atari+bounded-correction two-stage design fixed. The pre-leak-fix-era notebook model that started this whole investigation was a **single-stage** direct rough→line regression with no atari intermediate at all, and reportedly looked crisper | `UNetGenerator`, no aux/atari input at all (`in_channels=1`) — architecturally identical to the notebook's model (64→128→256→512 channels, `ResBlock` + dilated convs, concat skip connections) | `pred_logits = UNet(rough)` (no residual anchor); loss identical to the noadv row: `L = 0.8·BCE(pos_w=3) + 0.2·L1 + 0.5·edge_loss`, no GAN | `lineart/unetgenerator.py::UNetGenerator`, `--model unet` with no `--aux-dir`; `experiments/run_combined_koma_direct_unet_20260801.sh` (3 epochs), `run_combined_koma_direct_unet_100ep_20260801.sh` (100 epochs, launched overnight 2026-08-01, result pending as of this writing) | Yes (3-epoch version) | `results/compare_combined_koma_direct_unet_20260801.png`; 100-epoch montage not generated yet | **3-epoch result is not informative and should not be cited as evidence against this architecture**: near-blank output (F1@2px 0.012), but training loss was still dropping steadily with zero sign of convergence (0.377→0.292→0.273), unlike the atari-anchored family which starts warm and converges in 2-3 epochs. A cold-start full U-Net learning the whole mapping from scratch needs far more steps — the notebook itself trained 50 epochs. 100-epoch rerun launched to give this a fair test; check `results/compare_combined_koma_direct_unet_100ep_20260801.png` before drawing any conclusion here. |

## Data-split ablations (not architecture, but directly relevant to "why is confidence low")

Included here because they were run specifically to rule in/out a
*non-architecture* explanation for the same symptom (low-confidence
output) that every row above is trying to fix.

| 目的 | 「アーキテクチャ」(実際はデータ分割) | 数式 | 実装のコード | 目視評価がされたか・そのレビュー | montageの場所 | その他メモ |
|---|---|---|---|---|---|---|
| Test whether training on a *mix* of high- and low- rough/line-correspondence tiles forces the model to hedge (vs. training separately on each) | Same `cleanup` architecture/loss as the adopted msgan row; only the training file-list differs (top/bottom 240 or 450 tiles by a correspondence score) | Same as msgan row; correspondence score varies by test (see notes) | `tools/evaluation/score_pair_agreement.py` (generic Canny+distance-transform score), `tools/evaluation/split_koma_by_tile_edge_f1.py` (pipeline-native post-alignment `edge_f1`, more precise); `experiments/run_agreement_halo_survey.sh` (2026-07-20, old data), `run_combined_koma_agreement_halo_20260801.sh`, `run_combined_koma_tile_edge_f1_halo_20260801.sh` | Yes, all three tests | `results/compare_agreement_halo_e2.png` (2026-07-20), `results/compare_combined_koma_agreement_halo_20260801.png`, `results/compare_combined_koma_tile_edge_f1_halo_20260801.png` | **Monotonically-shrinking effect across three tests, converging to ~zero**: 2026-07-20 (data still had unresolved coordinate misalignment) F1 gap 0.155 (high 0.436 vs low 0.281); 2026-08-01 generic score on aligned data, gap 0.009; 2026-08-01 pipeline-native post-alignment `edge_f1`, gap **reversed** (-0.008, low slightly higher). **Conclusion: most of the original effect was a byproduct of the since-fixed coordinate-misalignment problem, not intrinsic content-correspondence ambiguity.** Closed out — don't revisit without new evidence. |

## Directions 1/2/3/7: not tested in isolation

| Direction | Status |
|---|---|
| 1: Multi-Scale PatchGAN | Never tested standalone. Folded directly into the adopted msgan recipe as `--multiscale-gan` (`lineart/model_zoo.py::MultiScalePatchDiscriminator`, scales `(1.0, 0.5)`) — every "msgan" row above already includes this. |
| 2: Feature Matching Loss | Same — folded into the msgan recipe as `--feature-match-weight 0.08` (`feature_matching_loss` on discriminator intermediate activations). Never isolated from Direction 1. |
| 3: Line Perceptual / Structure Feature Loss | Two separate instantiations exist, both folded into loss recipes rather than isolated: (a) `structure_pyramid_loss` (Sobel/DoG-like 2-scale comparison), used at `--structure-weight 0.04` in the msgan recipe; (b) `edge_loss` (Canny-based L1, `lineart/losses.py`), used at `--edge-weight 0.5` in the 2026-08-01 noadv/direct-unet ablations. Never compared against each other or against "no structure loss at all" in a controlled way. |
| 7: Soft Morphology / Line-Width Loss | `soft_width_loss` exists and was used (`--width-weight`) in **pre-koma** surveys (`run_line_refiner_survey.sh`, `run_badrough_inkwidth_survey.sh`, `run_badrough_lucy_thin_*_survey.sh`) — never rerun on current `combined_koma_20260729` data, no result recorded in `doc/model_directions.md` under a "Direction 7" heading. Status on current data: **untested**. |

## How to keep this file honest

- Never add a row from metrics alone. If you have not looked at the
  montage (or don't have one yet), write "not yet reviewed" in the 目視
  column and do not mark the row as adopted/rejected.
- If a numeric result and a visual review disagree, the visual review
  wins, and the disagreement itself is worth a sentence in その他メモ
  (see the noadv row for the canonical example).
- When a checkpoint gets superseded (v1 → v2 style), keep both rows
  rather than overwriting — the failure mode of the earlier version is
  often exactly the lesson that produced the fix.
- Cross-reference `doc/work_log.md` by date for the full narrative if a
  row's summary isn't enough context.
