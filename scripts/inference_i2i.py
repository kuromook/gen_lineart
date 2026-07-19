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


def run_inference(checkpoint, input_path, output_path, autocontrast=False, aux_input=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_name = load_generator_checkpoint(checkpoint, device)
    model.eval()
    image = Image.open(input_path).convert("L")
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=0)
    tensors = [TF.to_tensor(TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE)))]
    if aux_input:
        aux = Image.open(aux_input).convert("L")
        tensors.append(TF.to_tensor(TF.resize(aux, (IMAGE_SIZE, IMAGE_SIZE))))
    tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
    output = (1.0 - pred).clamp(0, 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(output[0].cpu()).save(output_path)
    print(f"saved: {output_path} model={model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--aux-input", default=None)
    args = parser.parse_args()
    run_inference(args.checkpoint, args.input, args.output, args.autocontrast, args.aux_input)


if __name__ == "__main__":
    main()
