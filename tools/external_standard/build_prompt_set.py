"""Build the registered prompt set for Track N (external standard).

Registered in doc/prereg_external_standard_20261030.md section 9-2. The
vocabulary and the seed below are part of the registration: once the
registration commit lands, adding a term here moves the bar, in either
direction, because the external pool is part of the exposure.

457 prompts, drawn deterministically with random.Random(20261030), matching
the 457 distinct source panels on the author's side.
"""

import argparse
import json
import random

SEED = 20261030
N_PROMPTS = 457

COMMON = "masterpiece, best quality, monochrome, lineart, white background, simple background, thin lines"
NEGATIVE = (
    "color, colored, photo, 3d, watercolor, painting, blurry, text, signature, watermark"
)

SUBJECT = [
    "1girl, solo",
    "1boy, solo",
    "2girls",
    "1girl, 1boy",
    "2boys",
    "no humans",
]

VIEW = [
    "full body",
    "upper body",
    "portrait, close-up",
    "cowboy shot",
    "from side",
    "from behind",
    "from above",
    "from below",
]

POSE = [
    "standing",
    "sitting",
    "walking",
    "running",
    "lying down",
    "leaning forward",
    "arms crossed",
    "hands on hips",
    "looking back",
    "reaching out",
]

SCENE = [
    "indoors",
    "outdoors",
    "classroom",
    "city street",
    "forest",
    "bedroom",
    "kitchen",
    "rooftop",
    "train station",
    "beach",
    "office",
    "library",
]

DETAIL = [
    "school uniform",
    "casual clothes",
    "jacket",
    "dress",
    "coat",
    "sportswear",
    "long hair",
    "short hair",
    "glasses",
    "backpack",
    "holding a book",
    "holding a bag",
]


def build(seed=SEED, n=N_PROMPTS):
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        subject = rng.choice(SUBJECT)
        if subject == "no humans":
            parts = [COMMON, subject, rng.choice(SCENE), rng.choice(VIEW)]
        else:
            parts = [
                COMMON,
                subject,
                rng.choice(VIEW),
                rng.choice(POSE),
                rng.choice(DETAIL),
                rng.choice(SCENE),
            ]
        prompts.append(", ".join(parts))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/external_standard/prompt_set_20261030.json")
    args = parser.parse_args()

    prompts = build()
    payload = {
        "seed": SEED,
        "n": len(prompts),
        "negative": NEGATIVE,
        "common": COMMON,
        "prompts": prompts,
    }
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as file:
        json.dump(payload, file, ensure_ascii=False, indent=1)

    distinct = len(set(prompts))
    print(f"prompts: {len(prompts)}  distinct: {distinct}  duplicated: {len(prompts) - distinct}")
    print(f"seed: {SEED}")
    for p in prompts[:3]:
        print(f"  {p}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
