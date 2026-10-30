"""Generate the external-standard image pool, exactly as registered.

Settings come from doc/prereg_external_standard_20261030.md section 9-2/9-3
and are hard-coded here on purpose: changing one after the run moves the bar,
in either direction, because the external pool is part of the exposure.

Resumable by design -- one file per image, already-present files are skipped
and counted, so a crash loses at most the image in flight. The log is opened
in append mode so a restart does not erase the earlier run.
"""

import argparse
import base64
import collections
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:7860/sdapi/v1/txt2img"

# Registered generation settings (section 9-3).
SAMPLER = "DPM++ 2M"
SCHEDULER = "Karras"
STEPS = 28
CFG = 6.0
SIZE = 1024
SEED_BASE = 20261030000

# Second registration (2026-10-30, user's ruling to match the image pipelines).
# The author's tiles come from panels that were downscaled before the 480 window
# was cut; the first run's external side was never downscaled, which showed up
# as half the line width and twenty times the pure black. So the external side
# is now made at 2048 and downscaled before the window is cut.
#
# The user judged the three candidates by eye and picked the ESRGAN pass: the
# direct-2048 render was read as having frayed strokes and the Latent pass as
# having coloured fringes. That choice is recorded with its cost -- an ESRGAN
# upscale carries 1024px worth of stroke detail into a 2048px canvas, so
# downscaling it returns roughly to where it started. The history is matched
# only halfway, and the registration says so.
HIRES = {
    "enable_hr": True,
    "hr_scale": 2.0,
    "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
    "hr_second_pass_steps": 14,
    "denoising_strength": 0.40,
    "hr_additional_modules": [],  # forge requires this key or the call 500s
}

# Registered checkpoints (section 9-1).
CHECKPOINTS = {
    "animagine_xl_31": "animagine-xl-3.1.safetensors",
    "illustrious_xl_v01": "illustriousXL_v01.safetensors",
}


def generate(checkpoint, prompt, negative, seed, timeout):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": STEPS,
        "cfg_scale": CFG,
        "sampler_name": SAMPLER,
        "scheduler": SCHEDULER,
        "width": SIZE,
        "height": SIZE,
        "seed": seed,
        "override_settings": {"sd_model_checkpoint": checkpoint},
        "override_settings_restore_afterwards": False,
        **HIRES,
    }
    request = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return base64.b64decode(body["images"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-set", default="data/external_standard/prompt_set_20261030.json")
    parser.add_argument("--out-root", default="data/external_standard")
    parser.add_argument("--log", default="data/external_standard/generate_log.txt")
    parser.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    parser.add_argument("--tag", default="", help="suffix for the output dirs (smoke test)")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    spec = json.load(open(args.prompt_set))
    prompts = spec["prompts"]
    negative = spec["negative"]
    if args.limit:
        prompts = prompts[: args.limit]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")

    def say(line):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{stamp} {line}\n")
        log.flush()
        print(f"{stamp} {line}", flush=True)

    say(f"=== start: {len(prompts)} prompts x {len(CHECKPOINTS)} checkpoints ===")
    counts = collections.Counter()
    started = time.time()

    for name, checkpoint in CHECKPOINTS.items():
        out_dir = Path(args.out_root) / f"gen_{name}{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        say(f"[{name}] -> {out_dir}")
        for i, prompt in enumerate(prompts):
            seed = SEED_BASE + i
            path = out_dir / f"external_{name}_{i:04d}_seed{seed}.png"
            if path.exists():
                counts[f"{name}/skipped_present"] += 1
                continue
            for attempt in range(args.retries):
                try:
                    image = generate(checkpoint, prompt, negative, seed, args.timeout)
                    path.write_bytes(image)
                    counts[f"{name}/written"] += 1
                    break
                except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
                    counts[f"{name}/retry"] += 1
                    say(f"[{name}] {i:04d} attempt {attempt + 1} failed: {type(error).__name__}: {error}")
                    time.sleep(10)
            else:
                counts[f"{name}/failed"] += 1
                say(f"[{name}] {i:04d} GAVE UP after {args.retries} attempts")
            if (i + 1) % 25 == 0:
                done = sum(v for k, v in counts.items() if k.endswith("/written"))
                rate = (time.time() - started) / max(done, 1)
                say(f"[{name}] {i + 1}/{len(prompts)}  written={done}  {rate:.1f}s/image")

    say("=== done ===")
    # Selection is a measurement: print what was dropped, by reason.
    for key in sorted(counts):
        say(f"  {key}: {counts[key]}")
    say(f"  elapsed: {(time.time() - started) / 3600:.2f} h")
    log.close()


if __name__ == "__main__":
    main()
