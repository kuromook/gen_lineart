"""Inference-only intervention sweep (2026-09-05). Hypotheses #1-5 all
rejected, and a key structural fact surfaced while reading
../lineart/scripts/train_controlnet.py: the base UNet is frozen in EVERY
run this track has ever done (`unet.requires_grad_(False)`; LoRA is added
to the ControlNet only). So the UNet's own text response -- and with it
whatever hatch-heavy prior SD1.5 pretraining installed -- has never been
trainable here. That makes the inference-side knobs worth exhausting
before spending another ~10h on a retrain, and it also means these
prompt-side tests are NOT confounded the way the "manga panel" caption
test was: there is no training-time counterpart to negative prompting.

Four axes, each holding the other three at the baseline value:
- 1a negative_prompt: never used at all in this track so far. Mechanically
  different from removing a positive word (CFG steers the score function
  AWAY from the negative concept rather than merely not mentioning it).
- 1b guidance_scale: fixed at infer_controlnet.py's default 3.0 in every
  run to date; never swept.
- 1c controlnet_conditioning_scale: swept once (results/cond_scale_sweep/,
  2026-08-26) but under lineart_anime preprocessing, an older checkpoint,
  and before orientation_entropy/line_width_p50 existed as axes -- redone
  here on current terms, including values above 1.0 to push ControlNet
  harder against the frozen UNet's own image prior.
- 1d alternate positive style vocabulary: the 63-tag sweep
  (results/word_ablation_20260905/) only added/removed single Danbooru
  tags. These instead invoke wholly different style concepts that SD1.5
  pretraining associates with outline-only, shading-free art.
"""

BASE_CAPTION = "monochrome line art, manga panel, black and white"
BASE_NEGATIVE = ""
BASE_GUIDANCE = 3.0
BASE_COND_SCALE = 1.0

NEG_HATCH = "hatching, cross-hatching"
NEG_SCREENTONE = "screentone, halftone, dot pattern"
NEG_SHADING = "shading, tone, gradient"
NEG_COMBINED = (
    "hatching, cross-hatching, screentone, halftone, dot pattern, shading, tone, "
    "gradient, dense linework, scribble, sketchy lines, texture"
)


def variant(label, caption=BASE_CAPTION, negative=BASE_NEGATIVE, guidance=BASE_GUIDANCE, cond_scale=BASE_COND_SCALE):
    return {
        "label": label,
        "caption": caption,
        "negative_prompt": negative,
        "guidance_scale": guidance,
        "controlnet_conditioning_scale": cond_scale,
    }


VARIANTS = [
    variant("baseline"),
    # 1a negative prompt
    variant("neg_hatching", negative=NEG_HATCH),
    variant("neg_screentone", negative=NEG_SCREENTONE),
    variant("neg_shading", negative=NEG_SHADING),
    variant("neg_combined", negative=NEG_COMBINED),
    # 1b guidance_scale (3.0 is the baseline, not repeated)
    variant("cfg1.5", guidance=1.5),
    variant("cfg5.0", guidance=5.0),
    variant("cfg7.5", guidance=7.5),
    variant("cfg10.0", guidance=10.0),
    # 1c controlnet_conditioning_scale (1.0 is the baseline, not repeated)
    variant("cs0.8", cond_scale=0.8),
    variant("cs1.2", cond_scale=1.2),
    variant("cs1.5", cond_scale=1.5),
    variant("cs2.0", cond_scale=2.0),
    # 1d alternate positive style vocabulary
    variant(
        "style_coloringbook",
        caption="clean black and white line art, coloring book page, vector outline, no shading, flat white background",
    ),
    variant(
        "style_technicalpen",
        caption="technical pen ink outline, monochrome line art, no cross-hatching, no screentone",
    ),
    variant(
        "style_flat2d",
        caption="flat 2d illustration outline only, black lines on white background, no shading, no gradients",
    ),
]

# Round 2 (1e). Round 1 found controlnet_conditioning_scale to be the one
# axis that moves ink_ratio and components_per_1k_ink_px decisively toward
# GT (cs2.0: ink 0.1258 vs baseline 0.2992, components 22.27 vs GT 21.01),
# visually confirmed -- the cross-hatch that fills the background at cs1.0
# is largely gone by cs2.0. The trend was still monotonic at the edge of
# the ladder, so extend past 2.0, and combine the winning scale with the
# raised CFG values that helped on their own.
VARIANTS_ROUND2 = [
    variant("cs2.5", cond_scale=2.5),
    variant("cs3.0", cond_scale=3.0),
    variant("cs2.0_cfg5.0", cond_scale=2.0, guidance=5.0),
    variant("cs2.0_cfg7.5", cond_scale=2.0, guidance=7.5),
    variant("cs2.5_cfg5.0", cond_scale=2.5, guidance=5.0),
    variant("cs2.0_neg_combined", cond_scale=2.0, negative=NEG_COMBINED),
]

# Round 3. cs3.0 was still improving at the edge of round 2 (ink_ratio
# 0.0891, gt_bsds_f1 0.2274 -- the best F1 ever recorded in this track,
# vs 0.0945-0.1558 across all ten trained models). Push until it breaks:
# components_per_1k_ink_px already overshoots GT at cs3.0 (47.06 vs 21.01),
# which is the signature of strokes starting to fragment, so the optimum
# is likely in this range.
VARIANTS_ROUND3 = [
    variant("cs3.5", cond_scale=3.5),
    variant("cs4.0", cond_scale=4.0),
    variant("cs5.0", cond_scale=5.0),
]
