"""Second-pass VLM review for hamlabi region candidates.

The deterministic region matcher writes candidate boxes. This script rebuilds
rough/line/overlay panels from those boxes and optionally asks a local Ollama
vision model whether the pair depicts the same character or drawing region.
"""

import argparse
import ast
import base64
import csv
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ZIP = "dataset_hamlabi.zip"
DEFAULT_ZIP_ROOT = "dataset_hamlabi"
DEFAULT_CANDIDATES = "results/hamlabi_region_candidates.csv"
DEFAULT_OUT = "results/hamlabi_region_vlm_review_qwen3vl.csv"
DEFAULT_PANEL_DIR = "results/hamlabi_region_vlm_review_panels"
MODEL = "qwen3-vl:4b-instruct-q8_0"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def read_zip_member(zf, root, name):
    for candidate in (f"{root}/{name}", f"{root}\\{name}", name):
        try:
            return zf.read(candidate)
        except KeyError:
            pass
    raise KeyError(f"missing zip member for {name!r}")


def load_manifest(zip_path, zip_root):
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(read_zip_member(zf, zip_root, "manifest.json"))
    return {Path(entry.get("file", entry["line"])).stem.replace("page", ""): entry for entry in manifest}


def load_pair(zf, zip_root, entry):
    rough = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["sketch"]))).convert("L")
    line = Image.open(io.BytesIO(read_zip_member(zf, zip_root, entry["line"]))).convert("L")
    rough = ImageOps.autocontrast(rough, cutoff=0)
    return np.asarray(rough), np.asarray(line)


def parse_box(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in ast.literal_eval(value))


def crop_resize(image, box, long_side):
    x0, y0, x1, y1 = parse_box(box)
    crop = image[y0:y1, x0:x1]
    h, w = crop.shape
    scale = long_side / max(w, h)
    out_w = max(32, int(round(w * scale)))
    out_h = max(32, int(round(h * scale)))
    return np.asarray(Image.fromarray(crop).resize((out_w, out_h), Image.Resampling.LANCZOS))


def edge_map(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(blur, 45, 135) > 0


def fit_square(image, size):
    if image.ndim == 2:
        pil = Image.fromarray(image).convert("RGB")
    else:
        pil = Image.fromarray(image).convert("RGB")
    pil.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(pil, ((size - pil.width) // 2, (size - pil.height) // 2))
    return canvas


def make_panel(rough, line, row, thumb):
    rough_edge = edge_map(rough)
    line_edge = edge_map(line)
    h = max(rough.shape[0], line.shape[0])
    w = max(rough.shape[1], line.shape[1])
    rough_pad = np.full((h, w), 255, np.uint8)
    line_pad = np.full((h, w), 255, np.uint8)
    rough_pad[: rough.shape[0], : rough.shape[1]] = rough
    line_pad[: line.shape[0], : line.shape[1]] = line
    rough_edge_pad = np.zeros((h, w), bool)
    line_edge_pad = np.zeros((h, w), bool)
    rough_edge_pad[: rough_edge.shape[0], : rough_edge.shape[1]] = rough_edge
    line_edge_pad[: line_edge.shape[0], : line_edge.shape[1]] = line_edge
    overlay = np.full((h, w, 3), 255, np.uint8)
    overlay[rough_edge_pad] = (255, 60, 60)
    overlay[line_edge_pad] = (40, 80, 255)

    label_h = 42
    canvas = Image.new("RGB", (thumb * 3, thumb + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for col, image in enumerate((rough_pad, line_pad, overlay)):
        canvas.paste(fit_square(image, thumb), (col * thumb, 0))
    region_id = row.get("region_index", row.get("child_index", ""))
    label = (
        f'{row["rank"]} page={row["page"]} region={region_id} '
        f'score={float(row["match_score"]):.2f} F1={float(row["edge_f1"]):.2f} '
        "left=rough center=line right=overlay"
    )
    draw.text((5, thumb + 6), label, fill="black", font=font)
    return canvas


def image_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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


def ask_ollama(model, panel, timeout):
    prompt = (
        "You are reviewing a rough-sketch to clean-line training pair. "
        "The image has three columns: rough sketch, clean line, and overlay where red is rough edges and blue is line edges. "
        "Judge whether rough and line depict the same character, pose, body part, or manga panel region. "
        "Reject if they are different content, too partial to learn from, mostly margins/text/noise, or only share generic straight lines. "
        "Return strict JSON only with keys: same_content (true/false), usable (true/false), score (0-5), region_type (short label), reason (short English phrase)."
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
    return parse_json(data.get("message", {}).get("content", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--thumb", type=int, default=360)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()

    with open(args.candidates, newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.save_panels:
        Path(args.panel_dir).mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.zip_path, args.zip_root)
    fields = list(rows[0].keys()) + [
        "vlm_same_content", "vlm_usable", "vlm_score",
        "vlm_region_type", "vlm_reason", "vlm_decision", "panel_path",
    ]
    with zipfile.ZipFile(args.zip_path) as zf, open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        cache = {}
        for index, row in enumerate(rows, 1):
            page = row["page"]
            if page not in cache:
                cache[page] = load_pair(zf, args.zip_root, manifest[page])
            rough_page, line_page = cache[page]
            rough = crop_resize(rough_page, row["rough_box"], args.long_side)
            line = crop_resize(line_page, row["line_box"], args.long_side)
            panel = make_panel(rough, line, row, args.thumb)
            panel_path = ""
            if args.save_panels:
                panel_path = str(Path(args.panel_dir) / f'hamlabi_region_{int(row["rank"]):04d}.png')
                panel.save(panel_path)
            if args.panel_only:
                same_content = False
                usable = False
                score = 0.0
                region_type = "panel_only"
                reason = "panel_only"
                decision = "panel_only"
            else:
                try:
                    result = ask_ollama(args.model, panel, args.timeout)
                    same_content = truthy(result.get("same_content", False))
                    usable = truthy(result.get("usable", False))
                    score = float(result.get("score", 0.0))
                    region_type = str(result.get("region_type", ""))
                    reason = str(result.get("reason", ""))
                    decision = "accept" if same_content and usable and score >= args.min_score else "reject"
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as error:
                    same_content = False
                    usable = False
                    score = 0.0
                    region_type = "error"
                    reason = f"error: {error}"
                    decision = "error"
            writer.writerow({
                **row,
                "vlm_same_content": same_content,
                "vlm_usable": usable,
                "vlm_score": score,
                "vlm_region_type": region_type,
                "vlm_reason": reason,
                "vlm_decision": decision,
                "panel_path": panel_path,
            })
            file.flush()
            print(f"{index}/{len(rows)} {decision} score={score:.1f} {reason}", flush=True)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
