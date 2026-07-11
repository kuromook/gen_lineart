import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from lineart.unetgenerator import UNetGenerator

IMAGE_SIZE = 480


def run_inference(checkpoint, input_path, output_path, autocontrast=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img = Image.open(input_path).convert("L")
    if autocontrast:
        img = ImageOps.autocontrast(img, cutoff=0)
    img_tensor = TF.to_tensor(TF.resize(img, (IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(device)

    with torch.no_grad():
        out = torch.sigmoid(model(img_tensor))

    out = (1.0 - out).clamp(0, 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(out[0].cpu()).save(output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/base/best.pth")
    parser.add_argument("--input",        default="test/rough/sample.jpg")
    parser.add_argument("--output",       default="results/output.png")
    parser.add_argument("--autocontrast", action="store_true",
                        help="roughのコントラストを自動正規化してから推論")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input, args.output, args.autocontrast)
