"""Run a region-manifest checkpoint and build a rough/model/GT montage."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps

from lineart.model_zoo import load_generator_checkpoint
from lineart.region_dataset import fit_square, read_manifest, resolve_path


def row_name(row, index):
    for key in ("v2_name", "final_name", "name", "id"):
        if row.get(key):
            return Path(row[key]).stem
    return f"region_{index:04d}"


def row_path(row, manifest_path, keys):
    for key in keys:
        if row.get(key):
            return resolve_path(row[key], manifest_path)
    raise KeyError(f"missing path key; tried {keys}")


def load_image(path, size, fit_mode, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return fit_square(image, size, fit_mode)


def tensor_from_image(image):
    return TF.to_tensor(image).unsqueeze(0)


def path_keys(explicit_key, candidates):
    keys = [explicit_key] if explicit_key else []
    keys.extend(candidates)
    return keys


def draw_montage(items, output, image_size):
    columns = ["rough", "model", "line (GT)"]
    header_h = 28
    label_h = 22
    canvas = Image.new(
        "RGB",
        (image_size * len(columns), header_h + (image_size + label_h) * len(items)),
        (210, 210, 210),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small_font = ImageFont.load_default()

    for col, title in enumerate(columns):
        width = draw.textlength(title, font=font)
        draw.text((col * image_size + (image_size - width) / 2, 6), title, fill=0, font=font)

    for row, item in enumerate(items):
        y = header_h + row * (image_size + label_h)
        for col, image in enumerate((item["rough"], item["model"], item["line"])):
            canvas.paste(image.convert("RGB"), (col * image_size, y))
        draw.text((6, y + image_size + 3), item["name"], fill=0, font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--montage", required=True)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--fit-mode", choices=["square_pad", "resize_stretch"], default="square_pad")
    parser.add_argument("--rough-key", default=None)
    parser.add_argument("--line-key", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("--require-cuda was set, but torch.cuda.is_available() is False")

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_name = load_generator_checkpoint(args.checkpoint, device)
    model.eval()

    items = []
    with torch.no_grad():
        for index, row in enumerate(rows[: args.limit], start=1):
            name = row_name(row, index)
            rough_path = row_path(
                row,
                manifest_path,
                path_keys(
                    args.rough_key,
                    ("aligned_rough_path", "v2_rough_path", "final_rough_path", "rough_path"),
                ),
            )
            line_path = row_path(
                row,
                manifest_path,
                path_keys(
                    args.line_key,
                    ("aligned_line_path", "v2_line_path", "final_line_path", "line_path"),
                ),
            )
            rough = load_image(rough_path, args.image_size, args.fit_mode, args.autocontrast)
            line = load_image(line_path, args.image_size, args.fit_mode)
            pred = torch.sigmoid(model(tensor_from_image(rough).to(device)))
            output = TF.to_pil_image((1.0 - pred[0].cpu()).clamp(0, 1))
            output.save(output_dir / f"{name}_out.png")
            items.append({"name": name, "rough": rough, "model": output, "line": line})

    draw_montage(items, Path(args.montage), args.image_size)
    print(f"saved {len(items)} outputs to {output_dir} model={model_name} device={device}")


if __name__ == "__main__":
    main()
