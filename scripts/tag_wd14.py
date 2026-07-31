"""Auto-tag tiles with a WD14-style anime tagger (SmilingWolf/wd-v1-4-moat-tagger-v2),
producing per-tile danbooru-style tags for use as ControlNet training captions
(replacing the current single fixed caption -- see doc/work_log.md 2026-07-31
Direction 4 entries for why: the fixed caption gives the model no signal to
distinguish tiles, which is a likely contributor to the observed hallucination).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_DIR = os.path.expanduser("~/disk/checkpoint/wd14_tagger")


def load_tags(tags_csv):
    names, categories = [], []
    with open(tags_csv) as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            categories.append(int(row["category"]))
    return names, categories


def preprocess(path, size):
    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, "WHITE")
    canvas.paste(image, mask=image)
    image = canvas.convert("RGB")
    array = np.asarray(image)[:, :, ::-1]  # RGB -> BGR

    h, w = array.shape[:2]
    side = max(h, w)
    padded = np.full((side, side, 3), 255, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top : top + h, left : left + w] = array

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)[None, ...]


def tag_image(session, input_name, size, names, categories, path, general_threshold, char_threshold):
    batch = preprocess(path, size)
    (probs,) = session.run(None, {input_name: batch})
    probs = probs[0]

    general_tags, char_tags = [], []
    for name, category, prob in zip(names, categories, probs):
        if category == 9:
            continue  # rating tag (general/sensitive/questionable/explicit)
        if category == 4 and prob >= char_threshold:  # character
            char_tags.append((name, prob))
        elif category == 0 and prob >= general_threshold:  # general
            general_tags.append((name, prob))
    general_tags.sort(key=lambda x: -x[1])
    char_tags.sort(key=lambda x: -x[1])
    return char_tags, general_tags


def build_caption(char_tags, general_tags, suffix):
    tags = [n for n, _ in char_tags] + [n for n, _ in general_tags]
    return ", ".join(tags + [suffix]) if tags else suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", help="ad-hoc mode: tag these files and print to stdout")
    parser.add_argument("--file-list", help="batch mode: newline-separated tile names (as in dataset/pairs_480 manifests)")
    parser.add_argument("--image-dir", help="batch mode: directory the --file-list names live in")
    parser.add_argument("--output-csv", help="batch mode: write name,tags,caption here")
    parser.add_argument(
        "--caption-suffix",
        default="monochrome line art, clean linework, manga panel",
        help="appended to every per-tile tag list so the framing from the old fixed caption is preserved",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--general-threshold", type=float, default=0.35)
    parser.add_argument("--char-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        os.path.join(args.model_dir, "model.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    size = input_meta.shape[1]
    names, categories = load_tags(os.path.join(args.model_dir, "selected_tags.csv"))

    if args.file_list:
        with open(args.file_list) as f:
            file_names = [line.strip() for line in f if line.strip()]

        start = time.time()
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "tags", "caption"])
            writer.writeheader()
            for i, name in enumerate(file_names):
                path = os.path.join(args.image_dir, name)
                char_tags, general_tags = tag_image(
                    session, input_name, size, names, categories, path,
                    args.general_threshold, args.char_threshold,
                )
                tag_list = [n for n, _ in char_tags] + [n for n, _ in general_tags]
                caption = build_caption(char_tags, general_tags, args.caption_suffix)
                writer.writerow({"name": name, "tags": ", ".join(tag_list), "caption": caption})
                if (i + 1) % 20 == 0 or (i + 1) == len(file_names):
                    f.flush()
                    elapsed = time.time() - start
                    rate = elapsed / (i + 1)
                    print(
                        f"[tag_wd14] {i + 1}/{len(file_names)} elapsed={elapsed:.0f}s "
                        f"({rate:.2f}s/img, eta={rate * (len(file_names) - i - 1):.0f}s)",
                        flush=True,
                    )
        print(f"[tag_wd14] wrote {len(file_names)} rows to {args.output_csv}")
        return

    for path in args.images:
        char_tags, general_tags = tag_image(
            session, input_name, size, names, categories, path,
            args.general_threshold, args.char_threshold,
        )
        tags = [n for n, _ in char_tags] + [n for n, _ in general_tags]
        print(f"{path}")
        print(f"  tags: {', '.join(tags) if tags else '(none above threshold)'}")
        top10 = sorted(char_tags + general_tags, key=lambda x: -x[1])[:10]
        print(f"  top10 w/ scores: {[(n, round(p, 3)) for n, p in top10]}")


if __name__ == "__main__":
    main()
