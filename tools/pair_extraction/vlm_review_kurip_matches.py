"""Review matched kurip rough/line pairs with a local Ollama vision model.

The deterministic matcher remains the first pass. This script is a second-pass
visual reviewer: it builds a rough/line/overlay panel for each saved tile and
asks a VLM whether the pair appears to show the same local drawing region.
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MATCHES = "results/kurip_matched_tiles_strict_all_noac.csv"
ROUGH_DIR = "dataset/pairs_480/train/rough"
LINE_DIR = "dataset/pairs_480/train/line_kurip_matched_clean_t192_cc8"
OUT = "results/kurip_vlm_review_qwen3vl.csv"
QC_DIR = "results/kurip_vlm_review_panels"
MODEL = "qwen3-vl:4b-instruct-q8_0"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def make_panel(rough_path, line_path, label):
    rough = np.asarray(Image.open(rough_path).convert("L"))
    line = np.asarray(Image.open(line_path).convert("L"))
    overlay = np.full((*rough.shape, 3), 255, np.uint8)
    overlay[edge_map(rough)] = (255, 60, 60)
    overlay[edge_map(line)] = (40, 80, 255)

    thumb, label_h = 360, 34
    canvas = Image.new("RGB", (thumb * 3, thumb + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    images = [
        Image.fromarray(rough).convert("RGB"),
        Image.fromarray(line).convert("RGB"),
        Image.fromarray(overlay),
    ]
    for index, image in enumerate(images):
        canvas.paste(image.resize((thumb, thumb)), (index * thumb, 0))
    draw.text((5, thumb + 8), f"{label} | left=rough center=line right=overlay(red rough, blue line)", fill="black", font=font)
    return canvas


def image_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ask_ollama(model, panel, timeout):
    prompt = (
        "You are reviewing a training pair for sketch-to-lineart learning. "
        "The image has three columns: rough sketch, clean line target, and an overlay where red is rough edges and blue is line edges. "
        "Judge whether the rough and line show the same local drawing region and are usable as a supervised training pair. "
        "Return strict JSON only with keys: same_region (true/false), usable (true/false), score (0-5), reason (short English phrase). "
        "Reject if the rough and line are different body parts/objects, shifted too far, rotated differently, or only share generic straight lines."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64(panel)]}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    return parse_json(content)


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default=MATCHES)
    parser.add_argument("--rough-dir", default=ROUGH_DIR)
    parser.add_argument("--line-dir", default=LINE_DIR)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--qc-dir", default=QC_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()

    with open(args.matches, newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.save_panels:
        os.makedirs(args.qc_dir, exist_ok=True)

    fields = list(rows[0].keys()) + ["vlm_same_region", "vlm_usable", "vlm_score", "vlm_reason", "vlm_decision"]
    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            rough_path = os.path.join(args.rough_dir, row["name"])
            line_path = os.path.join(args.line_dir, row["name"])
            panel = make_panel(rough_path, line_path, f'{index}/{len(rows)} {row["name"]}')
            if args.save_panels:
                panel.save(os.path.join(args.qc_dir, row["name"].replace(".jpg", ".png")))
            if args.panel_only:
                score = 0.0
                same_region = False
                usable = False
                reason = "panel_only"
                decision = "panel_only"
            else:
                try:
                    result = ask_ollama(args.model, panel, args.timeout)
                    score = float(result.get("score", 0))
                    same_region = truthy(result.get("same_region", False))
                    usable = truthy(result.get("usable", False))
                    reason = str(result.get("reason", ""))
                    decision = "accept" if same_region and usable and score >= args.min_score else "reject"
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as error:
                    score = 0.0
                    same_region = False
                    usable = False
                    reason = f"error: {error}"
                    decision = "error"
            writer.writerow({
                **row,
                "vlm_same_region": same_region,
                "vlm_usable": usable,
                "vlm_score": score,
                "vlm_reason": reason,
                "vlm_decision": decision,
            })
            file.flush()
            print(f"{index}/{len(rows)} {row['name']} {decision} score={score:.1f} {reason}", flush=True)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
