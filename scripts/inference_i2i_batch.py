import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from lineart.model_zoo import load_generator_checkpoint


IMAGE_SIZE = 480


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def load_tensor(path, autocontrast=False):
    image = Image.open(path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    return TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aux-dir", default=None)
    parser.add_argument("--autocontrast", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_name = load_generator_checkpoint(args.checkpoint, device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with torch.no_grad():
        for sample in read_list(args.file_list):
            base = Path(sample).stem
            tensors = [
                load_tensor(Path(args.rough_dir) / f"{base}.jpg", autocontrast=args.autocontrast)
            ]
            if args.aux_dir:
                tensors.append(load_tensor(Path(args.aux_dir) / f"{base}_out.png"))
            x = torch.cat(tensors, dim=0).unsqueeze(0).to(device)
            pred = torch.sigmoid(model(x))
            output = (1.0 - pred).clamp(0, 1)
            TF.to_pil_image(output[0].cpu()).save(output_dir / f"{base}_out.png")
            count += 1

    print(f"saved {count} files to {output_dir} model={model_name}")


if __name__ == "__main__":
    main()
