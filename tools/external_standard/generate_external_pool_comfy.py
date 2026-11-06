"""ComfyUI twin of generate_external_pool.py (2026-11-07).

The user is moving image generation from Forge to ComfyUI. This instrument
sends the SAME registered settings (checkpoint file, DPM++ 2M Karras, 28 steps,
CFG 6.0, 1024x1024, seed, prompt, negative) to ComfyUI as a plain API graph
(CheckpointLoaderSimple -> CLIPTextEncode x2 -> EmptyLatentImage -> KSampler ->
VAEDecode -> SaveImage). No custom nodes, no A1111-emulation.

It is NOT the registered generator: its images are a different generation
setting and never join a Forge pool. Outputs go under a dir with a `_comfy` tag.
"""

import argparse
import collections
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HOST = "http://192.168.1.34:8188"  # comfyui.service binds the LAN address only
SAMPLER, SCHEDULER, STEPS, CFG, SIZE = "dpmpp_2m", "karras", 28, 6.0, 1024
CHECKPOINTS = {
    "animagine_xl_31": "animagine-xl-3.1.safetensors",
    "illustrious_xl_v01": "illustriousXL_v01.safetensors",
}


def graph(ckpt, prompt, negative, seed, cfg):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": seed, "steps": STEPS, "cfg": cfg, "sampler_name": SAMPLER,
            "scheduler": SCHEDULER, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "lineart_external_comfy"}},
    }


def call(path, data=None, timeout=60):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def generate(ckpt, prompt, negative, seed, timeout, cfg=CFG):
    pid = json.loads(call("/prompt", {"prompt": graph(ckpt, prompt, negative, seed, cfg),
                                      "client_id": str(uuid.uuid4())}))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(call(f"/history/{pid}"))
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise ValueError(f"comfy error: {entry['status']}")
            img = entry["outputs"]["7"]["images"][0]
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            return call(f"/view?{q}")
        time.sleep(1)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set", default="data/external_standard/prompt_set_20261030.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="_comfy")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    spec = json.load(open(args.prompt_set))
    prompts = spec["prompts"][: args.limit] if args.limit else spec["prompts"]
    log = open(Path(args.log), "a")

    def say(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        log.write(line + "\n"); log.flush(); print(line, flush=True)

    say(f"=== start comfy: {len(prompts)} prompts | seed_base={args.seed_base} {SAMPLER}/{SCHEDULER} "
        f"steps={STEPS} cfg={args.cfg} size={SIZE} ===")
    counts = collections.Counter()
    for name, ckpt in CHECKPOINTS.items():
        out = Path(args.out_root) / f"gen_{name}{args.tag}"
        out.mkdir(parents=True, exist_ok=True)
        for i, prompt in enumerate(prompts):
            seed = args.seed_base + i
            path = out / f"external_{name}_{i:04d}_seed{seed}.png"
            if path.exists():
                counts[f"{name}/skipped_present"] += 1
                continue
            try:
                path.write_bytes(generate(ckpt, prompt, spec["negative"], seed, args.timeout, args.cfg))
                counts[f"{name}/written"] += 1
            except Exception as e:  # counted, never silent
                counts[f"{name}/failed"] += 1
                say(f"[{name}] {i:04d} failed: {type(e).__name__}: {e}")
    say("=== done ===")
    for k in sorted(counts):
        say(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
