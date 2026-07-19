import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from scripts.train_line_field_refiner import IMAGE_SIZE, LineFieldUNet


def run_inference(checkpoint, input_path, aux_input, output_path, autocontrast=False, debug_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device)
    model = LineFieldUNet(in_channels=payload.get("in_channels", 2)).to(device)
    model.load_state_dict(payload["G_state"])
    model.eval()

    rough = Image.open(input_path).convert("L")
    aux = Image.open(aux_input).convert("L")
    if autocontrast:
        rough = ImageOps.autocontrast(rough, cutoff=0)
    x = torch.cat(
        [
            TF.to_tensor(TF.resize(rough, (IMAGE_SIZE, IMAGE_SIZE))),
            TF.to_tensor(TF.resize(aux, (IMAGE_SIZE, IMAGE_SIZE))),
        ],
        dim=0,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        ink_logits, center_logits, offset = model(x)
        ink = torch.sigmoid(ink_logits)
        center = torch.sigmoid(center_logits)
    output = (1.0 - ink).clamp(0, 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(output[0].cpu()).save(output_path)

    if debug_dir:
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        stem = Path(output_path).stem.replace("_out", "")
        TF.to_pil_image((1.0 - center).clamp(0, 1)[0].cpu()).save(debug / f"{stem}_center.png")
        dx = (offset[:, :1] + 1.0) * 0.5
        dy = (offset[:, 1:2] + 1.0) * 0.5
        TF.to_pil_image(dx[0].cpu()).save(debug / f"{stem}_dx.png")
        TF.to_pil_image(dy[0].cpu()).save(debug / f"{stem}_dy.png")
    print(f"saved: {output_path} model=linefield_unet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--aux-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()
    run_inference(
        args.checkpoint,
        args.input,
        args.aux_input,
        args.output,
        autocontrast=args.autocontrast,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
