"""Prompt sets for the close-up angle (2026-11-06).

Same vocabulary and builder as build_prompt_set.py (section 9-2 of the first
registration), with two changes only: the VIEW slot is fixed to one close-up
wording, and "no humans" is not drawn (a close-up of nothing has no face).
POSE / DETAIL / SCENE are drawn exactly as before.

--survey writes small sets for the wording survey (faces only, no axis is
computed on them); the registered set is written by the registered call.
"""

import argparse
import json
import random
from pathlib import Path

import build_prompt_set as B

WORDINGS = {
    "pc": "portrait, close-up",
    "pcf": "portrait, close-up, face focus",
    "xcu": "extreme close-up, face focus",
}


def build(view, seed, n):
    rng = random.Random(seed)
    subjects = [s for s in B.SUBJECT if s != "no humans"]
    out = []
    for _ in range(n):
        out.append(", ".join([B.COMMON, rng.choice(subjects), view, rng.choice(B.POSE),
                              rng.choice(B.DETAIL), rng.choice(B.SCENE)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wording", required=True, choices=list(WORDINGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = build(WORDINGS[a.wording], a.seed, a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": a.seed, "n": a.n, "wording": WORDINGS[a.wording], "negative": B.NEGATIVE,
               "common": B.COMMON, "prompts": prompts}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"{a.wording}: {len(prompts)} prompts, distinct {len(set(prompts))} -> {a.out}")


if __name__ == "__main__":
    main()
